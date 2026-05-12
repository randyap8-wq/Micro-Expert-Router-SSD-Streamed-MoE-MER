# Micro-Expert-Router, SSD-Streamed MoE Execution Engine

A Rust execution engine for **Mixture-of-Experts** models that keeps the
router resident in RAM and **hot-swaps individual experts on demand** from a
PCIe-attached NVMe drive into a pool of pre-allocated, page-aligned RAM
buffers using **`O_DIRECT`** positional reads (`pread(2)` via
`tokio::task::block_in_place`, kernel-page-cache bypass). Each routed
expert then **executes a real Mixtral / Llama-style
SwiGLU FFN forward pass** directly over the bytes that just arrived from the
drive.

The premise is straightforward: a **modern PCIe-4 / 5 NVMe SSD sustains
6-14 GB/s** of sequential read; a Mixtral-class expert is ~88 MB; pulling
the top-K active experts per token therefore costs a few milliseconds of
I/O even when the *full* parameter set is 10-100× DRAM. So you can run
much larger models on much more modest hardware by treating the SSD as
the main weight store and DRAM as a small cache of *active* experts.
This engine is the substrate that makes that tradeoff observable and
measurable.

The engine lives under [`rust-engine/`](./rust-engine).

---

## What it actually does

A standard Mixtral-style transformer activates only `K` of `N` experts per
token (e.g. `K=2`, `N=64`). For inference on hardware whose DRAM cannot hold
all `N` experts you have two options:

1. **mmap the weights**, relies on the OS page cache. Works, but the kernel's
   prefetcher knows nothing about the routing pattern, you double-copy through
   the page cache, and you can't bypass the readahead heuristics.
2. **Manage the cache yourself**, what this engine does. Open each expert as
   its own file, read it with `O_DIRECT` `pread(2)` (dispatched off the
   Tokio runtime via `block_in_place`) so the bytes go
   directly from the NVMe DMA engine into a page-aligned RAM buffer, and run
   a custom LRU + a **three-signal predictive controller** that fuses a
   2nd-order Markov chain, a sliding-window locality monitor, and a
   small online-trained neural speculator into one speculative-I/O
   fetch set.

### End-to-end pipeline

```
        +------------+    +-------------+    +-----------+    +------------------+
token → |   Router   | → | Expert IDs   | → | LRU Cache | → | SwiGLU FFN        |
        | LinearGate |   |  e.g. [3,7]  |   +-----+-----+    | per expert,       |
        |  or Markov |   +------+-------+         | miss     | gate-weighted sum |
        +-----+------+          |                 ↓          +------------------+
              │                 │        +------------------+
              │ hidden state    │        | BufferPool slot  | ←─────┐
              ↓                 │        |  (aligned, pre-  |       │
   +------------------------+   │        |   allocated)     |       │
   | Predictive controller  |   │        +--------+---------+       │
   |   S = 2nd-order Markov |   │                 ↓                 │
   |   L = LocalityMonitor  | → │        +------------------+       │ on Arc drop
   |   M = NeuralSpeculator |   │        |  pread(2) read   |       │ (LRU evict
   |   E = S ∪ L ∪ M        |   │        |  O_DIRECT + (opt)|       │  or buffer
   +-----------+------------+   │        |  io_uring fixed  |       │  release)
               │                │        +--------+---------+       │
               ↓                ↓                 ↓                 │
       non-evicting prefetches             NVMe SSD → DMA → RAM ────┘
                                                   ↓
                                          bytes reinterpreted as
                                          f32 / f16 / int8 / Q4_K_M /
                                          Q4_0 weights → matmul
```

After every token the engine updates **three predictors in parallel**:

* **S, a sparse 2nd-order Markov model** over `(prev_prev, prev) → next`
  (with a 1st-order fallback). Smoothed with a Laplace prior so cold
  rows still return a valid distribution.
* **L, a `LocalityMonitor`** keeping a sliding-window heat map of the
  most recently activated experts. Hot experts are **pinned** in the
  LRU cache for as long as their count stays above the threshold,
  the LRU cannot evict them even if cold experts arrive in a flurry.
* **M, a `NeuralSpeculator`**: a 2-layer MLP
  (`d_model → hidden → num_experts`, ReLU + softmax) trained online by
  SGD against the gate's actual top-K. Training is dispatched to an
  **off-path background worker** through a bounded `mpsc` queue, so
  the inference critical path never blocks on backprop.

All three feed `PredictiveLoader::predict_unified`, which sums a
weighted score per candidate id (speculator × 0.42, Markov × 0.33,
locality × 0.25) and returns the top-fanout union `E = S ∪ L ∪ M` for
speculative I/O. The weights encode the intent that the **speculator
is the strongest signal** (it conditions on the actual hidden state),
the **Markov chain is next** (statistical smoothing of transitions),
and **locality is the weakest tiebreaker** (a coarse "recently hot"
prior); see `PredictiveLoader::predict_unified` in `router.rs` for
the canonical constants. Prefetches use `try_acquire` only and
**never evict a resident slot**, speculation can't starve real work.

The **router** itself is either the legacy `TopKRouter`
(deterministic Markov chain over expert ids, clustered locality by
default, or load a precomputed `N×N` transition matrix via
`--router-matrix`) or, when `[real_transformer].enabled = true`, the
production `LinearGate` (`softmax(W_gate · x) → top-K`, the exact
routing equation Mixtral / Llama-MoE uses). Both produce the same
`RoutingDecision { experts, weights }` shape; the experts are still
streamed from the SSD by the same cache regardless.

### What "running" actually does

For each token, the engine:

1. asks the **router** for K distinct expert ids, either
   `softmax(W_gate · x) → top-K` (real-transformer path) or
   `P(next | last_expert)` from the Markov chain (benchmark path);
2. for each id, hits the LRU cache or streams the expert file off the
   NVMe drive into a page-aligned pool buffer via `O_DIRECT` (one
   `pread(2)` per miss, or one fused `io_uring_enter` for the whole
   batch with `--io-uring`);
3. **reinterprets the buffer as weight matrices** in the configured
   dtype (`F32`, `F16`, `Int8`, `Q4_K_M`, or `Q4_0`, see
   [Quantization](#1-on-disk-quantization---dtype)). For the
   floating-point dtypes this is a zero-copy reinterpretation; for
   the integer / block-quantised dtypes a small per-fetch
   dequantisation runs over the bytes that just arrived. Layout is
   always `gate_proj || up_proj || down_proj`, row-major, the
   standard Mixtral / Llama / DeepSeek FFN layout;
4. runs a real **SwiGLU FFN forward pass**:
   `y = down_proj · ( silu(gate_proj · x) ⊙ (up_proj · x) )`,
  or, with `--io-only`, XOR-checksums every read byte instead, to
   isolate pure SSD-streaming cost from FFN compute;
5. combines the K expert outputs. Under `[real_transformer]` this is
   the gate's softmax-weighted sum (the actual Mixtral combine);
   under the legacy benchmark path it is a uniform average;
6. observes the routing decision into all three predictors and kicks
   off the speculative `E = S ∪ L ∪ M` prefetch union, deduplicated
   against ids already resident or in flight.

The forward pass is **scalar `f32` Rust by default**, no BLAS, no
SIMD, no GPU, because the project's thesis is about **storage
bandwidth**, not compute. Two opt-in cargo features escalate the
dense-matmul kernel without touching call sites:

* `--features simd` routes the dense projections inside
  `TransformerLayer` / `LMHead` through a `std::thread::scope`-based
  **row-parallel** matmul (no extra crate dep).
* `--features blas` swaps in the **`matrixmultiply` SGEMV
  microkernel** (the same hand-tuned BLAS-shaped path `ndarray` uses
  for its `dot` op). Mutually exclusive with `simd`; a static
  `compile_error!` enforces this in `transformer.rs`.

The SwiGLU expert kernel itself stays plain scalar, it just has to
be real enough to exercise every byte that came off the drive
(compiler can't fold it away) and surface a believable
compute-vs-I/O latency picture in the per-token logs.

---

## Architecture

The Rust crate (`rust-engine/`) is organised into single-responsibility modules:

| Module | Responsibility |
|---|---|
| `aligned_buffer` | Heap-allocated, page-aligned buffer (`std::alloc::alloc` with a `Layout`). The defining requirement of `O_DIRECT`: kernel rejects unaligned buffers with `EINVAL`. |
| `buffer_pool` | Fixed-capacity slab of `AlignedBuffer`s, optionally split into **primary** + **shadow** halves sharing one `Notify`. Hands out `PooledBuffer` RAII guards; `try_acquire`/`try_acquire_shadow` route to the corresponding free list; dropping a guard returns the buffer to its originating list and wakes waiters. `promote_shadow` does the zero-copy slot-tag swap when a speculative prefetch is confirmed. The literal "pre-allocated RAM buffer" the spec asks for. |
| `expert_cache` | LRU map `expert_id → Arc<ExpertResident>`, with a separate **pin set** so frequency-pinned and locality-hot experts skip eviction. Eviction returns the `Arc`; once all references drop, the buffer goes back to the pool automatically. |
| `multi_layer_cache` | Per-layer `ExpertCache` wrapper keyed on `(layer, expert)`. Lets multi-layer Mixtral / DeepSeek configurations give each layer its own LRU budget instead of sharing one global cache. |
| `block_pool` | Server-wide physical block pool for the **paged KV cache**. A pre-allocated slab plus a heap-backed overflow slab that grows on demand, with O(1) free-list alloc/release. The `BlockManager` is a per-request handle that auto-returns all of its blocks on `Drop`. |
| `io_provider` | NVMe storage layer. Opens each expert as its own file (`O_DIRECT` on Linux), keeps fds resident, and reads via `tokio::task::block_in_place` + `pread(2)` (`FileExt::read_at`). Supports **multi-drive striping** (`NvmeStorage::striped`), experts are sharded across `N` mountpoints by `id % N`. Includes synthetic test generators (for every dtype) and a portable Unix fallback for development on macOS. |
| `io_uring_storage` | Linux-only `io_uring` backend with **registered fixed buffers** (`IORING_REGISTER_BUFFERS`) and a batched `submit_and_wait(K)` entry point. Built behind the `io_uring` cargo feature. |
| `router` | The three-signal predictive controller in one module: `TopKRouter` (deterministic 1st-order Markov router, clustered locality by default, or a precomputed `N×N` matrix), `PredictiveLoader` (online **1st- and 2nd-order** sparse Markov predictor with a Laplace prior, plus the unified `predict_unified(S ∪ L ∪ M)` scoring API), `LocalityMonitor` (sliding-window heat map, the **L** arm), and `NeuralSpeculator` (2-layer MLP trained online by SGD on an off-path worker thread, the **M** arm). |
| `gating` | Production routing path: `LinearGate` computes `softmax(W_gate · x) → top-K` exactly the way Mixtral does. `Router` is an enum the engine holds polymorphically, `Router::Linear` in the real-transformer path, `Router::Markov` for the benchmark / `--io-only` path. |
| `inference` | SwiGLU expert FFN (`y = down · (silu(gate·x) ⊙ (up·x))`), implemented per dtype: `run_inference` (F32, zero-copy reinterpret), `run_inference_f16` / `_int8` / `_q4k` / `_q4_0` (dequantise then scalar `f32` matmul), and `run_inference_partial` (load only the top-M input columns by magnitude). All variants run directly over the bytes streamed off NVMe. |
| `transformer` | Scalar `f32` dense pieces of the Mixtral / Llama decoder layer: `RmsNorm`, `apply_rope_inplace`, `MultiHeadSelfAttention` (with **GQA** when `num_kv_heads < num_heads` and optional **sliding-window** attention), `TransformerLayer`, `KvCache` (16-token blocks, can be backed by the `block_pool` slab), `LMHead`, and the `matmul_row_major` dispatch (scalar / `simd` / `blas`). |
| `model` | `RealModel`, full multi-layer decoder built on top of `transformer`. Owns the dense (resident) weights, drives the per-token forward (`embedding → stacked layers → final RMSNorm → LM head`), and addresses experts as `global_id = layer * num_experts + local_id` so the existing single-namespace cache + storage layers work unchanged. Loads dense weights from per-tensor `.bin` files (`from_dir`) **or** HuggingFace `.safetensors` shards (`from_safetensors`); `from_dir_auto` picks the right one. Missing tensors fall back to a deterministic seeded init. |
| `sampling` | OpenAI-compatible next-token sampler, temperature, top-K, top-P (nucleus), `(seed, position)`-driven RNG. `temperature == 0.0` short-circuits to greedy `argmax`. |
| `tokenizer` | HuggingFace `tokenizers` crate when the `tokenizer` cargo feature is enabled and a `tokenizer.json` is configured; deterministic byte-level fallback otherwise. |
| `session` | In-memory KV-cache session store (`DashMap`-backed) for multi-turn chat. Per-session position cursor + idle-TTL evictor. |
| `batch_scheduler` | **Continuous batching.** An `mpsc`-fed background task drains per-token `StepRequest`s, fuses up to `max_batch_size` requests (or whatever has arrived within `batch_timeout_ms`) into a single batch, and runs their `RealModel::step` calls concurrently against the shared `Engine`. Owns a central `RequestRegistry` so the channel carries only `{ id, token, pos, params }` per token, never the full `Vec<KvCache>`, and optionally owns the shared `BlockPool` for paged KV. |
| `engine` | Top-level orchestrator. Owns the router, predictor (`S` + `L` + `M`), cache, pool, storage, alias map, frequency-pin counters, HDR histograms, and the alias/locality/speculator atomic telemetry. Drives the per-token cycle (`Engine::generate` and `Engine::moe_step`), schedules `union_prefetch`es, and reconciles the locality hot set with the cache's pin set. |
| `gguf` | Minimal **GGUF reader** (versions 1, 2, 3): magic / metadata / tensor table, recognises `F32`, `F16`, `Q4_0`, `Q4_K`, `Q6_K`. Two readers ship side-by-side: `GgufFile::open` (eager — slurps the file into RAM, useful for tests) and `GgufStreamReader::open` (streaming — keeps only the header resident and seeks tensor bodies on demand, the default for `gguf-convert`). Both implement the `GgufSource` trait so the loader is reader-agnostic. |
| `gguf_loader` | Glue from a `GgufSource` → per-expert `.bin` files + `metadata.json` + dense weight files. Each expert file is page-aligned and (by default) prefixed with a 64-byte **Unified Tensor Header**; `--no-uth` opts out. Driven by the `gguf-convert` subcommand. |
| `tensor_header` | 64-byte **U.T.H.** (`UTH1` magic, dtype, shape, quant-scale offset, AMX tile hint, flags). Self-describing prefix written by `gguf-convert` and transparently stripped by `ExpertResident::data()` so downstream kernels never see it. |
| `kernels` | Runtime CPU-feature dispatcher (`mod.rs::detect()` + `current()`), with `scalar.rs` (always on), `avx512.rs` (`--features avx512`, `#[target_feature]` fused int8 dequant + dot), and `amx.rs` (`--features amx`, skeleton until tile intrinsics stabilise on stable Rust). The selected backend is logged once at startup. |
| `numa` | `MER_PIN_CORES=N` env honoured at startup → `sched_setaffinity(2)` first `N` CPUs of NUMA node 0 (Linux only, best-effort; no-op + warn elsewhere). |
| `metrics` | Prometheus `Registry` + handles for every counter / histogram exported on `/metrics`. |
| `config` | TOML schema for `serve --config`: `[server]`, `[sampling]`, `[model]`, `[storage]`, `[tokenizer]`, `[real_transformer]`, `[predictive]`. Validated at startup. |
| `server` | OpenAI-compatible HTTP server (`axum`): `/health`, `/metrics`, `/v1/completions`, `/v1/chat/completions` (both streaming SSE and one-shot), `DELETE /v1/sessions/{id}`. |
| `main` | `clap`-based CLI with `gen-data`, `run`, `gguf-convert`, `validate-predictor`, and `serve` subcommands; structured `tracing` logs; `--first-token 3,7` to reproduce the spec example; `--io-only` for pure-I/O benchmarking; `--force-ssd` to refuse page-cache shortcuts; `--data-dir DIR1,DIR2,...` for multi-drive striping; and auto-loading of `metadata.json` (written by `scripts/extract_mixtral_experts.py` or `gguf-convert`) so a real Mixtral checkpoint runs with no further flags. |

### Key design decisions

- **Race-free fetch-then-evict.** When the cache is full and the pool is
  exhausted, naive "acquire then evict" deadlocks: every buffer is held by a
  cache `Arc`, and the cache will only release one *after* the new buffer is
  filled. The fetch path therefore evicts the LRU first, then `try_acquire`s,
  and re-evicts in a loop if a concurrent prefetch swiped the freed slot.
- **Non-evicting speculative prefetch.** Speculation must never starve real
  work. Prefetches use `try_acquire` only and skip if the pool is busy. The
  pool is sized as `cache_slots + predict_fanout` so there is always
  headroom for in-flight prefetches without growing the resident set.
- **Online sparse 1st + 2nd-order Markov predictor with prior.** Per-row
  sparse maps of observed `(from, to)` counts and `(prev_prev, prev) → to`
  counts plus a uniform Laplace prior (every cell starts at an implicit
  count of 1). `predict_next2` blends the 2nd-order row 50/50 with its
  1st-order fallback, and `predict_unified` further fuses Markov,
  locality, and speculator signals into a single weighted ranking
  (speculator × 0.42 + Markov × 0.33 + locality × 0.25 — speculator
  is strongest because it conditions on the actual hidden state,
  Markov is next as a statistical smoother of transitions, locality
  is a coarse tiebreaker). Sparse-by-row
  means memory scales with the number of *visited* pairs, not `O(N²)`
  or `O(N³)` up front, important once `N` reaches Mixtral 8x22B /
  DeepSeek-V3 expert counts.
- **Pluggable router.** The legacy `TopKRouter` is a deterministic
  Markov chain over expert ids, useful for benchmarks where you want
  a fixed routing distribution independent of the model weights. The
  production `LinearGate` (in `gating.rs`) computes
  `softmax(W_gate · x) → top-K` from the actual hidden state, and is
  the path `[real_transformer].enabled = true` selects. Both produce
  the same `RoutingDecision { experts, weights }`; the experts are
  still streamed by the same SSD-backed `ExpertCache`. Given a
  `--seed`, either path is reproducible.
- **Pluggable I/O backend.** The hot path uses `tokio::task::block_in_place`
  to dispatch a synchronous `pread(2)` (via `std::os::unix::fs::FileExt::read_at`)
  on the current Tokio worker; the runtime donates that worker to blocking
  work and other tasks are picked up by sibling workers. On non-Linux Unix
  (e.g. macOS dev boxes) the same code path runs without `O_DIRECT` so the
  engine still runs end-to-end during development.

### `pread` + `block_in_place` *or* io_uring (`--features io_uring`)

The default backend uses positional `pread(2)` driven on Tokio's
blocking thread pool via `block_in_place`. It's `O_DIRECT`-compatible,
deep-queue-friendly on NVMe, and avoids touching the file offset so
concurrent reads against the same fd are safe, and it works on every
Unix without any extra dependencies.

A real **io_uring backend with registered fixed buffers** ships in
`src/io_uring_storage.rs` and is built when the cargo feature
`io_uring` is enabled (Linux only). On startup it
`io_uring_register(IORING_REGISTER_BUFFERS)`s every `BufferPool` slot
with the kernel exactly once, so subsequent reads are
`IORING_OP_READ_FIXED` SQEs that reference a buffer *index*, the
kernel never has to walk the user mapping or pin pages on the hot
path. A batched submission entry point
(`IoUringStorage::read_experts_batch_fixed`) pushes `K` SQEs and calls
`submit_and_wait(K)` once when a token misses on multiple experts.
Pass `--io-uring` on the CLI (or build with `--features io_uring`) to
opt in. When the kernel doesn't support the registration (older
kernels, restrictive sandboxes) we log a warning and stay on the
portable `pread` backend.

For comparison with other Rust async-I/O options on this workload:

| Crate | Verdict for this workload |
|---|---|
| **`pread` + `block_in_place`** *(default)* | Zero extra deps, works on every Unix, exercises the full `O_DIRECT` + page-aligned-buffer + LRU + prefetch logic. The compute and storage stay observably distinct in the per-token logs. |
| **`io-uring`** *(`--features io_uring`, used here)* | The thinnest binding to the kernel ABI. We use it with **registered (fixed) buffers** + cached fds, which removes per-op address validation in the kernel, the single biggest win for sustained NVMe throughput. |
| **`tokio-uring`** | Best ergonomic fit if you live in Tokio. Single-threaded per ring, requires `#[tokio_uring::start]` instead of `#[tokio::main]`, would force a runtime restructuring. |
| **`glommio`** | Thread-per-core, polled io_uring. Made for NVMe-bound workloads (ScyllaDB heritage). For a pure expert-fetch service pinning workers to cores feeding local rings, glommio is arguably the *fastest* answer on Linux. Trade-off: incompatible with Tokio (it owns the runtime). |

The clean separation between `io_provider` / `io_uring_storage` and the
rest of the engine means swapping in `glommio` or `tokio-uring` later
is a self-contained change.

---

## Building and running

### Quickstart (Docker / docker-compose)

If you just want to see the engine answer an HTTP request on a fresh
laptop, the project ships a `Dockerfile`, a `docker-compose.yml`, and a
helper script that generates a small synthetic dataset before starting
the server:

```bash
# 1. (optional) generate ~128 MiB of synthetic expert files into ./data
./scripts/quickstart.sh    # generates data + runs the binary directly

# 2. or use Docker (builds a slim runtime image, mounts ./data + config.toml)
docker compose up --build

# 3. smoke test
curl -sS http://localhost:8080/health
curl -sS -X POST http://localhost:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","max_tokens":4,"stream":true}'
```

The compose file defaults to building with `--features io_uring`
(set the `FEATURES=""` build arg to opt out). Edit `config.toml` on
the host and `docker compose restart mer` to reload settings, the
file is bind-mounted read-only into `/etc/mer/config.toml`.

### Prerequisites

- **Linux** for the default `pread(2)` + `O_DIRECT` I/O path on real
  NVMe. The optional `--features io_uring` backend needs **kernel ≥
  5.6** (and a sandbox that doesn't filter `io_uring_setup`).
- **Rust 1.74+** (uses `clap 4`, edition 2021).
- A real **block-device-backed filesystem** (ext4, xfs, btrfs on NVMe) for
  the `O_DIRECT` path. tmpfs / overlayfs / many FUSE mounts return `EINVAL`
  on `open(O_DIRECT)`. Use `--no-direct` if you need to run on those for
  development.
- **macOS** is supported for development: `O_DIRECT` is unavailable on
  Darwin, so the engine auto-falls back to buffered reads and prints a
  startup warning that measured I/O latency includes OS page-cache
  effects. Use a Linux host on real NVMe for clean numbers. Pass
  `--force-ssd` if you want the engine to *refuse* to run with any
  page-cache shortcut on Linux (it requires `O_DIRECT`).
- **Optional Python deps** for the Mixtral extraction script
  (`scripts/extract_mixtral_experts.py`):
  `pip install 'transformers>=4.38' torch`. The Rust engine itself has
  no Python or PyTorch dependency.

### Build

```bash
cd rust-engine
cargo build --release                       # default, portable, scalar
cargo build --release --features io_uring   # Linux: enables IoUringStorage
cargo build --release --features simd       # row-parallel dense matmul (std::thread::scope)
cargo build --release --features blas       # matrixmultiply SGEMV microkernel (mutually exclusive with `simd`)
cargo build --release --features tokenizer  # real HuggingFace tokenizer (pulls in `onig`)
```

Features compose freely except `simd` and `blas`, which are mutually
exclusive (enforced at compile time by a `compile_error!` in
`transformer.rs`).

### Generate synthetic expert files

```bash
# 64 experts × 16 MiB each = 1 GiB of test data on disk.
# Default FFN shape: d_model=512, d_ff=2048 → 12 MiB of f32 SwiGLU weights
# per expert + 4 MiB zero-padding (so the file size stays a multiple of
# 4096 bytes for O_DIRECT).
./target/release/micro-expert-router gen-data \
  --data-dir ./data \
  --num-experts 64 \
  --expert-size $((16 * 1024 * 1024)) \
  --d-model 512 \
  --d-ff 2048
```

Each file holds three deterministically-generated `f32` matrices in
`gate_proj || up_proj || down_proj` order (row-major), drawn from
`U(-1/√d_model, +1/√d_model)`. That keeps the SwiGLU forward pass
numerically stable for any chosen `d_model`/`d_ff` and lets reads be
verified end-to-end.

> **Sizing rule of thumb.** The weights occupy `3 · d_model · d_ff · 4`
> bytes; pad up to a multiple of `--block-align` (4096) for `O_DIRECT`.
> `gen-data` enforces this and errors if `--expert-size` is too small.

### Run the simulation

```bash
# 200-token stream, top-2 routing. The cache holds 4 experts at a time
# (the engine's default, the whole point is to stream from SSD, so a
# big in-RAM cache hides the metric you're trying to measure; the
# engine warns above 16). d_model / d_ff MUST match what was passed to
# gen-data.
./target/release/micro-expert-router run \
  --data-dir ./data \
  --num-experts 64 \
  --expert-size $((16 * 1024 * 1024)) \
  --d-model 512 \
  --d-ff 2048 \
  --cache-slots 4 \
  --top-k 2 \
  --tokens 200 \
  --predict-fanout 2 \
  --predict-min-prob 0.05
```

To reproduce the **exact spec example** (router selects expert 3 and 7 first):

```bash
./target/release/micro-expert-router run \
  --data-dir ./data --tokens 50 \
  --first-token 3,7
```

### Run as an OpenAI-compatible HTTP server

The same engine, same SSD-streaming expert cache, same `O_DIRECT`
reads, same SwiGLU FFN over the bytes that just arrived from disk, can
be run as a long-lived HTTP server with an OpenAI-compatible API.

```bash
# Start the server. Reads everything from a TOML config file (see
# `config.toml` at the repo root for an annotated example).
./target/release/micro-expert-router serve --config ../config.toml
```

Endpoints:

| method   | path                       | purpose                                            |
| -------- | -------------------------- | -------------------------------------------------- |
| `GET`    | `/health`                  | liveness probe (`{"status":"ok",...}`)             |
| `GET`    | `/metrics`                 | Prometheus text format: cache hit rate, request latency histograms, tokens generated, per-token I/O wait, and, when the predictive arms are enabled, `mer_locality_hits_total`, `mer_locality_misses_total`, `mer_speculator_hits_total`, `mer_speculator_misses_total`, `mer_speculator_accuracy_total`, and the `mer_ssd_stall_seconds` histogram |
| `POST`   | `/v1/completions`          | OpenAI text-completion shape (`prompt`, `max_tokens`, ...) |
| `POST`   | `/v1/chat/completions`     | OpenAI chat-completion shape (`messages`, ...)       |
| `DELETE` | `/v1/sessions/{id}`        | explicitly drop a saved KV-cache session (see [Session API](#session-api)) |

Example:

```bash
curl -s http://127.0.0.1:8080/v1/completions \
  -H "content-type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":32}' | jq .
```

By default each request is **fully stateless**: the server drives the
model for `max_tokens` cycles and returns the decoded tokens in one
shot, with per-request KV caches that never alias other in-flight
requests. Cross-request KV reuse for multi-turn chat is opt-in via
the [Session API](#session-api), and concurrent decoder steps are
fused via the [Continuous batching scheduler](#continuous-batching).

**Streaming (`stream: true`) is supported.** The server emits one
`data: { ... }` SSE event per generated token in OpenAI's
`text_completion` / `chat.completion.chunk` shape, terminated with a
`data: [DONE]` line. For chat completions the first event carries the
`{ "role": "assistant" }` delta before any content tokens, matching
the OpenAI wire protocol exactly. Non-streaming requests (the default
when `stream` is absent or `false`) keep returning a single JSON
response.

#### Sampling

Both `/v1/completions` and `/v1/chat/completions` accept the standard
OpenAI-style sampling fields. They are merged with the server-wide
`[sampling]` defaults from `config.toml`; per-request fields
override the server defaults one knob at a time.

| field         | type    | meaning                                                                 |
| ------------- | ------- | ----------------------------------------------------------------------- |
| `temperature` | `f32`   | Softmax temperature. `0.0` (or any non-positive value) ⇒ greedy `argmax`, bit-for-bit reproducible. Mainstream values: `0.7` (creative) ... `1.0` (default). |
| `top_p`       | `f32`   | Nucleus cumulative-mass cutoff. `1.0` disables. `0.9` keeps the smallest set of tokens whose cumulative softmax probability `≥ 0.9`. |
| `top_k`       | `usize` | Top-K truncation. `0` disables. Combined with `top_p`, the more restrictive of the two takes effect. |
| `seed`        | `u64`   | Sampling RNG seed. Combined with the absolute token position via splitmix64, so the same `(prompt, seed, max_tokens)` produces the same completion bit-for-bit even at `temperature > 0`. |

```bash
curl -s http://127.0.0.1:8080/v1/completions \
  -H "content-type: application/json" \
  -d '{
        "prompt": "Once upon a time",
        "max_tokens": 32,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "seed": 1
      }' | jq .
```

`temperature == 0.0` skips the softmax / top-K / top-P pipeline
entirely and returns `argmax(logits)`, matching the legacy
deterministic behaviour every existing test relies on. With
`temperature > 0` the cost is a partial-sort + small softmax over
`vocab_size`, negligible relative to a full transformer step. The
sampling pipeline lives in [`sampling.rs`](./rust-engine/src/sampling.rs)
and applies only to the real-transformer path
(`[real_transformer].enabled = true`); the legacy synthetic generator
ignores these fields apart from `seed`, which still drives its
deterministic id stream.

#### Session API

Multi-turn chat normally re-runs attention over the entire
conversation on every turn, quadratic in the chat length. With
sessions, the server persists each conversation's per-layer KV
caches between requests so the next turn only attends over the
*new* tokens (linear amortised cost).

Sessions are **opt-in** at the server level, set
`server.session_ttl_secs > 0` in `config.toml` to enable the in-
memory store, then attach a stable `session_id` to each request:

```bash
# First turn: server stores the KV cache under "chat-42" after
# generating the response. Idle sessions are evicted after
# `session_ttl_secs` seconds.
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
        "session_id": "chat-42",
        "messages": [{"role":"user","content":"Hello!"}],
        "max_tokens": 64
      }' | jq .

# Second turn: same session_id resumes the saved KV cache; the new
# user message is fed at the position the previous turn left off.
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
        "session_id": "chat-42",
        "messages": [{"role":"user","content":"And what is 2+2?"}],
        "max_tokens": 64
      }' | jq .

# Free the saved KV cache when the conversation is done.
curl -s -X DELETE http://127.0.0.1:8080/v1/sessions/chat-42 | jq .
```

When `session_ttl_secs == 0` (the default) the session store is
disabled entirely, every request is stateless, and `DELETE
/v1/sessions/{id}` returns `404`. A request that names a
`session_id` no other request has registered simply starts fresh,
matching how vLLM, llama.cpp and ollama's session APIs behave.

The store is backed by a lock-free `dashmap` so concurrent HTTP
handlers don't contend on a global lock; `take` is destructive while
the request is active so two concurrent requests for the *same*
`session_id` never interleave attention state. Implementation lives
in [`session.rs`](./rust-engine/src/session.rs).

#### Continuous batching

When the real-transformer pipeline is enabled, all in-flight
requests' decoder steps are routed through a `BatchScheduler` that
fuses up to `max_batch_size` concurrent requests (or whatever
arrives within `batch_timeout_ms`) into a single batch and runs
them concurrently against the shared `Engine`. Per-request KV
caches travel through the channel + a `oneshot` reply, so attention
state stays strictly per-request while expert streaming and decoder
compute overlap. The two knobs live under `[real_transformer]`:

```toml
max_batch_size  = 8   # max concurrent requests fused per step (1 disables batching)
batch_timeout_ms = 5  # how long to wait for more requests to join a partial batch
```

#### Predictive architecture (`[predictive]`)

The HTTP server can opt the engine into the dual-path predictive
architecture (the **L** and **M** arms of `E = S ∪ L ∪ M`, see
[Predictive architecture](#7-predictive-architecture-s--l--m-speculative-io))
without any code changes, via an additional TOML block:

```toml
[predictive]
locality_enabled       = true   # turn on the LocalityMonitor (L arm)
locality_window        = 256    # sliding window length, in observations
locality_threshold_pct = 0.10   # heat ratio for declaring an expert "hot"
speculator_enabled     = true   # turn on the NeuralSpeculator (M arm)
speculator_hidden_dim  = 128    # MLP hidden size; 128 is the spec recommendation
speculator_top_k       = 0      # 0 ⇒ inherit `model.top_k`
```

The defaults (everything off) reproduce the legacy Markov-only
prefetch path bit-for-bit; flipping each flag wires its arm into
the prefetch union and lights up the corresponding Prometheus
counters on `/metrics`. The schema is validated at startup
(`config.rs`) and rejects nonsensical settings, `locality_window
== 0`, `locality_threshold_pct` outside `(0, 1]`, `speculator_top_k >
num_experts`, etc.

#### Real-transformer pipeline

By default the server runs the **legacy benchmark generator**: each
request drives `Engine::generate` for `max_tokens` cycles and synthesises
a deterministic id stream. The SSD-streaming substrate is exercised
identically.

When `[real_transformer].enabled = true` in the TOML config, requests go
through the **full decoder forward pass**:

```
embedding → for each layer: ( RMSNorm → MultiHeadSelfAttention → +
                              RMSNorm → LinearGate.route → moe_step → +)
            → final RMSNorm → LMHead → sample
```

`moe_step` is what reads expert weights from SSD via the LRU cache, so
the same hits / misses / I/O wait counters get populated regardless of
which path drives the loop.

The dense (resident) weights, embedding, attention projections, MoE
gate, RMSNorm gains, LM head, are loaded by `RealModel::from_dir_auto`,
which transparently picks the right format:

* **HuggingFace `safetensors`** (`model.safetensors` or sharded
  `model-00001-of-00002.safetensors` etc.), keyed by the standard
  `model.layers.{L}.self_attn.{q,k,v,o}_proj.weight` /
  `model.layers.{L}.block_sparse_moe.gate.weight` names; `bf16` /
  `f16` shards are dequantised to `f32` at load time.
* **Per-tensor `.bin` files** (one little-endian `f32` per file,
  `embed.bin`, `attn_<L>_q.bin`, `gate_<L>.bin`, ...), written by
  `gguf-convert` or by a custom extractor.

Either way, **expert FFN weights are not loaded here**, they live
on disk in `expert_<id>.bin` (single-layer) or `expert_<L>_<id>.bin`
(multi-layer) and stream through the cache on demand. Tensors that
aren't present fall back to a deterministic seeded initialisation,
so the engine always has an end-to-end runnable path even without
real model files. Multi-layer experts share the existing
single-namespace cache via the global addressing scheme
`global_id = layer * num_experts + local_id`, so the run summary
statistics are populated by the same instrumentation regardless of
layer count.

```toml
[real_transformer]
enabled = true
# Optional. Missing tensors fall back to a deterministic seeded init.
# weights_dir = "./data/dense"
vocab_size = 256          # match the tokenizer (256 for the byte fallback)
num_heads = 8
num_kv_heads = 2          # 0 = MHA (auto-set to num_heads); GQA otherwise
head_dim = 0              # 0 = auto (d_model / num_heads)
rope_base = 10000.0       # Llama-3.1 long-context: 500000.0
rms_eps = 1e-6
window_size = 0           # 0 = full causal; Mixtral uses 4096 (sliding-window attention)
seed = 0xC0FFEE
max_batch_size = 8        # continuous batching (see below)
batch_timeout_ms = 5
```

##### Paged KV cache (block pool)

When the real-transformer pipeline is enabled the per-layer KV
caches can be backed by a **shared physical block pool**
(`block_pool::BlockPool`) instead of allocating a new
`Vec<Box<[f32]>>` per request. The pool uses a pre-allocated slab
plus a heap-backed overflow slab that grows on demand, with O(1)
free-list alloc/release. Each request gets a thin `BlockManager`
that records the block ids it owns and auto-returns every block to
the pool on `Drop`. The scheduler picks block-pool sizing up via
`BatchConfig::block_pool_capacity` and `block_pool_kv_dim` and logs
a warning the first time a request touches the overflow slab so
operators can size the primary capacity for steady-state workloads
while staying safe under bursts.

#### Optional row-parallel / BLAS matmul (`simd` / `blas` features)

The dense projections inside `TransformerLayer` and `LMHead` are routed
through `transformer::matmul_row_major`, which is feature-gated:

* **default** (no features): scalar fused-loop matmul, single-threaded.
* `--features simd`: dispatches to a `std::thread::scope`-based
  row-parallel implementation (no extra crate dep, output rows are
  disjoint, so no synchronisation is needed).
* `--features blas`: routes through `matrixmultiply`'s hand-tuned
  SGEMV microkernel, the same BLAS-shaped path `ndarray::dot` uses.
  Mutually exclusive with `simd` (a static `compile_error!` in
  `transformer.rs` enforces this).

```bash
cargo build --release --features simd     # row-parallel
cargo build --release --features blas     # matrixmultiply SGEMV
```

The call sites are unchanged, so a future PR can swap the body for a
`candle::Tensor` op or a CUDA kernel without touching the layer
definitions.

Tokenization is via the [`tokenizers`] crate when the optional
`tokenizer` cargo feature is enabled and a `tokenizer.json` is configured,
or a deterministic byte-level fallback otherwise (so the server is
useful end-to-end for testing without shipping a 60 MB tokenizer file).

```bash
# Build with the real HuggingFace tokenizer wired in.
cargo build --release --features tokenizer
```

Configuration lives in TOML, see [`config.toml`](./config.toml) for
the full annotated schema (server bind address, model dimensions, cache
slots, `O_DIRECT` block alignment, predictive prefetch fanout, optional
tokenizer path).

To **isolate pure I/O cost** (skip the SwiGLU FFN; XOR every byte read
to force the page in):

```bash
./target/release/micro-expert-router run --io-only --tokens 200 ...
```

To **refuse any page-cache shortcut** (Linux only, fails fast if
`--no-direct` is also set):

```bash
./target/release/micro-expert-router run --force-ssd --tokens 200 ...
```

To run on a filesystem that doesn't support `O_DIRECT` (CI, tmpfs, macOS dev):

```bash
./target/release/micro-expert-router run --no-direct ...
```

Increase log verbosity:

```bash
RUST_LOG=micro_expert_router=debug ./target/release/micro-expert-router run ...
# or
./target/release/micro-expert-router --log debug run ...
```

### Sample output

```
INFO starting engine num_experts=16 top_k=2 cache_slots=4 expert_mib=16 d_model=512 d_ff=2048 weight_mib=12 direct_io=false io_only=false force_ssd=false
INFO router: deterministic Markov chain with structured cluster locality clusters=4 intra_cluster_p=0.9
INFO buffer pool sized with prefetch headroom cache_slots=4 pool_slots=6 prefetch_headroom=2
INFO streaming tokens (latency / throughput logs follow) tokens=30
INFO tick token=0  cycle_us=5569 tps="179.6" hits=0 misses=2 kib=32768 resident=[12, 15, 2, 10]
INFO tick token=17 cycle_us=3620 tps="276.2" hits=2 misses=0 kib=0     resident=[4, 3, 2, 12]
INFO tick token=29 cycle_us=3448 tps="290.0" hits=2 misses=0 kib=0     resident=[0, 4, 3, 7]
INFO stream complete wall_s=0.124 sustained_tps=243 avg_throughput_mibps=2717 hit_rate_pct=71.7
INFO ===================== run summary =====================
INFO experts:       16 (top-2), cache=4 slots, pool=6 slots
INFO ffn shape:     d_model=512  d_ff=2048  bytes/expert=12582912
INFO lookups:       hits=43  misses=17  hit_rate=71.67%
INFO prefetches:    completed=4   predictor_observations=116
INFO i/o:           reads=17  bytes=336.00 MiB
INFO i/o latency:   p50=1057us  p95=1743us  p99=1743us
INFO compute:       p50=3435us  p95=3521us  p99=3617us  (SwiGLU FFN per token)
INFO cycle latency: p50=3451us  p95=5643us  p99=6915us  max=6915us
INFO per-token avg: io_wait=572.4us  compute=3478.0us  (over 30 tokens)
INFO I/O share:     14.13% of token cycle time spent waiting on SSD reads
INFO energy knobs:  dtype=f32  partial_load_fraction=1.00  pinned=0  alias_redirects=0
```

When the predictive `L` / `M` arms are enabled, one extra line is appended:

```
INFO predictive:    locality=on (hit_rate=64.32%)  speculator=on (accuracy=58.10%)  ssd_stall=12.4ms
```

The `compute` row is the actual SwiGLU forward pass (per-token, summed
over the K active experts). The trailing **`per-token avg`** + **`I/O
share`** lines are the headline numbers: they tell you, on this run,
how many microseconds the engine spent waiting on the SSD per token and
what fraction of total token time that represents. Re-run with
`--io-only` to drop the SwiGLU compute and isolate pure I/O:

```
INFO compute:       p50=18us  p95=42us  p99=51us  (io-only XOR digest, FFN skipped)
INFO I/O share:     74.6% of token cycle time spent waiting on SSD reads
```

On a real PCIe-4 NVMe with `O_DIRECT` the `i/o` row drops further; on
bigger `d_model`/`d_ff` the `compute` row grows linearly, exactly the
trade you'd want to surface when reasoning about SSD-as-RAM viability
for a given model size.

### CLI reference

```
micro-expert-router gen-data
  --data-dir <PATH>          Output directory (default ./data)
  --num-experts <N>          Number of expert files (default 64)
  --expert-size <BYTES>      Bytes per file, multiple of 4096 (default 16 MiB)
  --d-model <N>              FFN hidden dim (default 512)
  --d-ff <N>                 FFN intermediate dim (default 2048)
  --block-align <BYTES>      O_DIRECT alignment (default 4096)
  --dtype <DTYPE>            f32 | f16 | int8 | q4k | q4_0 (default f32)

micro-expert-router run
  --data-dir <PATH>          Directory with expert_<id>.bin files
                              (auto-loads metadata.json if present).
                              Accepts a comma-separated list to shard
                              across multiple NVMe mountpoints, see
                              "Multi-drive striping" below.
  --num-experts <N>          Total experts in the model
  --expert-size <BYTES>      Must match gen-data
  --d-model <N>              Must match gen-data
  --d-ff <N>                 Must match gen-data
  --cache-slots <N>          Resident experts (default 4; warns if > 16)
  --top-k <K>                Active experts per token (default 2, distinct)
  --tokens <N>               Stream length
  --dtype <DTYPE>            f32 | f16 | int8 | q4k | q4_0 (default f32).
                              Must match gen-data / the offline extractor.
  --predict-fanout <N>       Prefetch candidates per token (default 2)
  --predict-min-prob <P>     Skip prefetch below this probability (default 0.05)
  --partial-load-fraction <F>  Fraction (0.1..=1.0) of input dimensions
                              loaded per expert. 1.0 (default) loads the
                              full expert.
  --pin-after-observations <N>  After N routing observations, pin the
                              expert permanently in the cache (0 disables).
  --alias-map <PATH>         JSON map {"src_id": canonical_id, ...} from
                              `scripts/compute_expert_aliases.py`: pairs
                              of near-identical experts share one
                              resident copy.
  --no-direct                Disable O_DIRECT (use page cache; CI / tmpfs / macOS)
  --block-align <BYTES>      O_DIRECT alignment, default 4096
  --first-token <IDS>        Comma-separated expert ids to warm into cache
  --no-prefetch              Disable predictive loader (for ablation)
  --io-only                  Skip the SwiGLU FFN; XOR every byte to isolate I/O cost
  --force-ssd                Refuse to run with anything that lets the OS serve
                              experts from RAM (requires O_DIRECT on Linux)
  --router-clusters <N>      Markov router cluster count (default 4)
  --router-intra-p <P>       P(stay in current cluster) (default 0.9)
  --router-matrix <PATH>     Load a precomputed N×N transition matrix from a
                              text file (whitespace-separated f64, row-major).
                              Overrides --router-clusters / --router-intra-p.
  --gate-weights <PATH>      Load a real gating-network weight matrix
                              ([num_experts × d_model] little-endian f32,
                              row-major, no header). When set, the run loop
                              bypasses the Markov router and computes
                              softmax(W_gate · x) → top-K per token (the
                              real Mixtral routing equation), with the
                              selected experts still streamed from the SSD
                              by Engine::moe_step.
  --io-uring                 Probe the IoUringStorage backend (Linux,
                              --features io_uring); also pins the process
                              to NUMA-local cores (override count via
                              MER_PIN_CORES env var).
  --token-pause-us <N>       Sleep between tokens to throttle the stream
  --seed <U64>               PRNG seed for reproducibility
  --trace-out <PATH>         Append a JSONL routing trace (one record per
                              token). Feed into `validate-predictor` or
                              `scripts/compute_transition_matrix.py`.

  # Multi-drive striping:
  --data-dir <DIR1,DIR2,...> Comma-separated list of mountpoints shards
                              experts across N NVMe drives by `id % N`.
                              Use `scripts/gen_striped_data.sh` to lay
                              out an existing dataset across drives.

micro-expert-router gguf-convert
  --gguf-path <PATH>         Source GGUF (Mixtral-style) checkpoint
  --out-dir <PATH>           Output dir for expert_<id>.bin + metadata.json
                              + dense weight files
  --num-layers <N>           Override (defaults to llama.block_count)
  --num-experts <N>          Experts per layer (defaults to llama.expert_count)

micro-expert-router validate-predictor
  --trace <PATH>             JSONL trace from `run --trace-out`
  --cache-slots <N>...       Cache sizes to sweep (default: 2 4 8 16)
```

### Running on real Mixtral weights

There are **three** ways to feed real Mixtral / Llama-MoE weights into
the engine, depending on what format you have them in:

**1. From a Hugging Face checkpoint (per-expert `.bin` files).**
`scripts/extract_mixtral_experts.py` dumps a single transformer
layer's expert FFNs from a HuggingFace Mixtral checkpoint into the
on-disk format the engine expects (`expert_<id>.bin` blobs +
`metadata.json`):

```
pip install 'transformers>=4.38' torch
python scripts/extract_mixtral_experts.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --layer 0 --out ./mixtral-data

cargo run --release --manifest-path rust-engine/Cargo.toml -- \
    run --data-dir ./mixtral-data --tokens 200
```

**2. From a GGUF checkpoint (`gguf-convert`).** No Python required,
the engine's built-in GGUF reader handles llama.cpp / Ollama-style
files directly. Supports `F32`, `F16`, `Q4_0`, `Q4_K_M` natively;
`Q6_K` tensors are recognised but fall back to seeded init (the
engine doesn't dequantise Q6_K). The output directory has the same
shape as the Mixtral extractor's: `expert_<layer>_<id>.bin` blobs +
`metadata.json` + per-tensor dense weight files.

`gguf-convert` defaults to a **streaming reader** that parses only
the GGUF header and seeks each tensor body on demand — a strict win
for ≥ 100 GB checkpoints. Pass `--legacy-eager` to fall back to the
in-memory reader. By default every `expert_<id>.bin` is prefixed with
a 64-byte **Unified Tensor Header** (dtype + shape + AMX tile hint +
quant-scale offset) padded to 4 KiB so the weight payload still
starts at an `O_DIRECT`-friendly boundary; pass `--no-uth` to opt
out for compatibility with consumers that pre-date the header.

```bash
./target/release/micro-expert-router gguf-convert \
    --gguf-path ./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --out-dir   ./mixtral-data \
    # add --no-uth and/or --legacy-eager if your tooling needs them

./target/release/micro-expert-router run --data-dir ./mixtral-data
```

**3. From HuggingFace `safetensors` shards.** When
`real_transformer.weights_dir` points at a directory containing
`model.safetensors` (or sharded `model-00001-of-00002.safetensors`
files), `RealModel::from_safetensors` picks them up automatically.
Tensor names follow the standard
`model.layers.{L}.self_attn.{q,k,v,o}_proj.weight` /
`model.layers.{L}.block_sparse_moe.gate.weight` convention; `bf16`
and `f16` shards are dequantised to `f32` at load time. Expert FFN
weights still come through the SSD-streaming path
(`expert_<id>.bin`), `from_safetensors` only handles the dense
(resident) tensors. `RealModel::from_dir_auto` will pick the right
loader.

The `metadata.json` written by `extract_mixtral_experts.py` or
`gguf-convert` lets `run` auto-fill `--num-experts`, `--d-model`,
`--d-ff`, `--top-k`, and `--expert-size` so the subsequent commands
need no further flags. Each Mixtral 8x7B expert is ~88 MiB at `f16`
(zero-padded to a 4 KiB multiple), ~700 MiB on disk for one layer,
fully streamable from any modern NVMe; at `q4_0` / `q4k` the same
expert is ~25 MiB.

### Routing model, Markov chain, transition matrix, or LinearGate

Three routers are available, all reproducible given `--seed`:

1. **Clustered Markov (default `run` path).** A deterministic 1st-order
   Markov chain over expert ids. Experts are partitioned into
   `--router-clusters` groups (by `id % cluster_count`) and the chain
   stays inside its current cluster with probability
   `--router-intra-p` (default `0.9`). Produces the "topic-sticky"
   behaviour real MoE traces show, the predictor converges quickly
   and prefetch hit rate climbs above 60%.
2. **Loaded transition matrix (`--router-matrix path.txt`).** Supply a
   whitespace-separated `num_experts × num_experts` matrix of `f64`
   transition probabilities, row-major. Rows are normalised to sum to
   1. Use this to feed a real Mixtral routing trace (e.g. produced by
   hooking `block_sparse_moe`'s gate softmax during a HuggingFace
   inference run) directly into the engine.
3. **Real `LinearGate` (`--gate-weights path.bin` or
   `[real_transformer].enabled = true`).** Load a real gating-network
   weight matrix (`[num_experts × d_model]` little-endian `f32`,
   row-major, no header) and route by `softmax(W_gate · x) → top-K`
   from the actual hidden state at each token. This is the same
   routing equation production Mixtral / Llama-MoE inference uses;
   the experts are still streamed from the SSD by the same cache.

### macOS

`O_DIRECT` is Linux-only. On macOS the engine automatically falls back
to buffered reads (`--no-direct`) and prints a startup warning that
measured I/O latency will include OS page-cache effects (and therefore
under-report cold-NVMe latency). Use a Linux host on a real NVMe device
for clean numbers.

---

## What can it actually run today?

**Today, in this repository: a real Mixtral / Llama-style transformer
forward pass with weights streamed from NVMe.** When
`[real_transformer].enabled = true` the server runs the full decoder

```
embedding -> for each layer: ( RMSNorm -> MultiHeadSelfAttention -> +
                               RMSNorm -> LinearGate.route -> moe_step -> + )
            -> final RMSNorm -> LMHead -> sample
```

`moe_step` reads expert weights from SSD via the LRU cache, so the
SSD-streaming substrate is exercised on every routed expert. The
dense (resident) tensors, embedding, attention projections, the
learned MoE gate, RMSNorm gains, LM head, are loaded from
`real_transformer.weights_dir` (or fall back to a deterministic
seeded init when files are missing, so a smoke run is always
possible). Each routed expert performs the exact
`down · (silu(gate·x) ⊙ (up·x))` block that every modern sparse
MoE transformer uses for its experts, at synthetic-or-real
dimensions (`d_model`, `d_ff`).

Tokenisation goes through the [`tokenizers`] crate when the
`tokenizer` cargo feature is enabled and `tokenizer.json` is
configured, with a deterministic byte-level fallback otherwise.
Next-token selection runs through a configurable
softmax-temperature + top-K + top-P sampler (see
[Sampling](#sampling)), and per-request KV caches can be persisted
between HTTP calls via the [Session API](#session-api).

What is **still synthetic by default**:

- **The router** is, by default, a deterministic Markov chain over
  expert ids (clustered locality, or load a real Mixtral routing-trace
  matrix via `--router-matrix`). When `[real_transformer]` is enabled
  routing instead goes through the per-layer learned `LinearGate`
  driven by the actual hidden state, the same `softmax`-over-gate-
  logits a real Mixtral implementation uses. The Markov path stays
  available for benchmarks where you want a fixed, reproducible
  routing distribution independent of the model weights.
- **Combining** uses the gate's softmax probabilities as weights on
  the K expert outputs (real gate-weighted sum). The legacy
  uniform-average path is preserved for the synthetic / benchmark
  pipeline only.

So: the engine demonstrates **the per-token forward pass of a sparse
MoE transformer**, end-to-end, with experts paged off the SSD and
real logit-driven token sampling. The expected drop-in path for
production use is to replace `inference::run_inference` with a call
into a tensor library such as `candle`, `tch`, or `cudarc`; the
byte→`f32` view at `inference::ExpertWeights::from_bytes` already
does zero-copy reinterpretation, so the swap is cleanly localised.

Real Mixtral expert weights can already be fed to the engine end-to-end
via [`scripts/extract_mixtral_experts.py`](./scripts/extract_mixtral_experts.py),
which dumps a single layer's experts into the on-disk format the
engine expects (plus a `metadata.json` that `run` auto-loads). See
[Running on real Mixtral weights](#running-on-real-mixtral-weights).

That said, the architecture (per-expert files, fixed expert size,
top-K activation, LRU + prefetch) is shaped specifically for **sparse
Mixture-of-Experts transformers where the expert FFNs are the dominant
weight**. Concretely, the following published models drop into this layout
with no architectural changes, only a real attention/embedding kernel and
a sharding script that splits their `safetensors` into one
`expert_<id>.bin` per expert (or per-layer-per-expert, see "Sharding
granularity" below):

| Model | Total params | Active / token | Experts | Top-K | Per-expert FFN (bf16) | Notes |
|---|---|---|---|---|---|---|
| **Mixtral 8x7B** | ~47 B | ~12.9 B | 8 × 32 layers | 2 | ~88 MB | Canonical fit. ~22 GB of expert weight, easily streamed from a single PCIe-4 NVMe. |
| **Mixtral 8x22B** | ~141 B | ~39 B | 8 × 56 layers | 2 | ~240 MB | Comfortable on PCIe-5 NVMe. Cache 8-16 experts; prefetcher learns the routing well. |
| **Phi-3.5-MoE-instruct** | ~42 B | ~6.6 B | 16 × 32 layers | 2 | ~80 MB | Smaller experts, more of them, exercises the predictor harder. |
| **Qwen1.5-MoE-A2.7B / Qwen2-MoE** | ~14 B | ~2.7 B | 60 × 24 layers | 4 | ~10 MB | Fine-grained experts; ideal for demonstrating prefetch hit-rate. |
| **DeepSeek-MoE 16B** | ~16.4 B | ~2.8 B | 64 routed + 2 shared × 28 layers | 6 | ~5-8 MB | "Shared experts" should be pinned (use `--first-token` to warm them, set `--cache-slots` ≥ shared count). |
| **DeepSeek-V2-Lite / V2** | 16 B / 236 B | 2.4 B / 21 B | 64-160 × many layers | 6 | small | Same shape, larger scale. V2-full needs PCIe-5 + ≥ 32 cache slots to keep p99 sane. |
| **DeepSeek-V3 / V3-0324** | 671 B | 37 B | 256 routed + 1 shared × 61 layers | 8 | small but many | Stress test of the design, ~15 K expert tensors. Sharding at per-layer-per-expert is mandatory. |
| **OLMoE-1B-7B** | 7 B | 1.3 B | 64 × 16 layers | 8 | ~6 MB | Open-everything; good for benchmarking and reproducibility. |
| **Snowflake Arctic** | 480 B | 17 B | 128 × 35 layers | 2 | medium | Top-2 makes prefetcher very effective. |
| **Grok-1** | 314 B | ~78 B | 8 × 64 layers | 2 | ~600 MB | Per-expert footprint approaches GB; keep `--cache-slots` modest and let the LRU breathe. |

What this means in practice:

- **Any **sparse MoE** transformer whose forward pass is "router → top-K MLPs"
  is compatible.** That covers essentially every modern open-weights MoE.
- **Dense models do not benefit.** A dense Llama-3 has no experts to swap.
- **Vision/multimodal MoEs (e.g. DeepSeek-VL2-MoE) work the same way** as
  long as the visual encoder is held resident in RAM (it's tiny next to the
  expert pool).

### Agents

The engine is *inference infrastructure*, not an agent runtime. There is
nothing here that loops over tool calls, parses ReAct traces, or manages
memory between turns. However, **any agent framework that delegates
generation to one of the LLMs above can use this engine as the underlying
serving layer once a tensor backend is wired in**, LangChain, LangGraph,
Microsoft AutoGen, CrewAI, llama-index, OpenAI-Agents-SDK, and the
`smolagents` family are all framework-agnostic about the model server. The
practical path is: this engine → an OpenAI-compatible HTTP shim →
the agent framework's standard client.

### Sharding granularity

Two ways to lay an MoE on disk; both are supported by the engine
unchanged, only `--num-experts` and `--expert-size` differ:

1. **One file per expert (all layers concatenated).** Smaller `--num-experts`,
   larger `--expert-size`. Best when DRAM is large enough to hold the
   active set of "whole experts". Higher prefetch payoff (one read per
   miss). Mixtral works well like this.
2. **One file per (layer, expert) pair.** Larger `--num-experts =
   layers × experts`, smaller `--expert-size`. Best for very wide models
   (DeepSeek-V3-class) where a single concatenated expert wouldn't fit a
   pool slot. Routing per-layer becomes the natural granularity for the
   predictor too.

The included `gen-data` subcommand creates the **same fixed-size per-expert
file format** the engine expects, so you can prototype a new model's
layout without writing a real sharder first: just pick `--num-experts` and
`--expert-size` to match the geometry above and you'll get realistic
latency / throughput numbers for that model's I/O profile.

### Picking a tensor backend (when you wire one in)

| Backend | Language | MoE support | Notes |
|---|---|---|---|
| **`mistral.rs`** | Rust | First-class (Mixtral, DeepSeek, Phi-MoE, Qwen-MoE) | Closest fit. Replace its weight loader with this engine's `ExpertCache::get` and you're done. |
| **`candle`** | Rust | Mixtral example in-tree | Tensor lib with no engine; you write the routing loop. Cleanest integration target. |
| **`burn`** | Rust | Generic; community Mixtral | Good if you want pluggable compute backends (wgpu, cuda, ndarray). |
| **`llama.cpp` (GGUF MoE)** | C++ | Mixtral, DeepSeek, Qwen-MoE, OLMoE | FFI required. GGUF stores experts contiguously per layer, easy to map to per-expert files. |
| **`vLLM`** | Python | Excellent | FFI required (the storage layer would expose a `/expert/<id>` server). Hardest, highest payoff for scale. |

## Tests

```bash
cd rust-engine
cargo test --release
```

Covers:

- buffer alignment + size invariants (`O_DIRECT` preconditions),
- buffer-pool acquire/release cycle and async waiter wakeup,
- LRU eviction returns buffers to the pool only after all `Arc`s drop,
- the cache never exceeds its configured slot count, even mid-stream
  under heavy eviction churn,
- the **Markov-chain router** produces distinct top-K ids, is fully
  reproducible given a `--seed`, prefers in-cluster transitions for the
  generated locality, and round-trips a transition matrix from disk,
- the **predictor** (sparse 1st- + 2nd-order Markov) learns simple
  transitions, respects `min_prob`, falls back to the Laplace prior
  when nothing has been observed, counts only real observations,
  handles zero fanout, and the unified `predict_unified(S ∪ L ∪ M)`
  ranker fuses Markov / locality / speculator contributions
  deterministically,
- the `f32` weight-view partitions buffers correctly,
- the SwiGLU forward pass produces finite, deterministic outputs of the
  correct shape, and zeroed weights yield a zero output,
- the `metadata.json` mini-parser handles both compact and
  pretty-printed JSON,
- the **transformer block**, RMSNorm normalises to unit variance,
  RoPE preserves vector norm and is the identity at position 0,
  sliding-window attention matches full attention inside its span,
  the MoE pre-routing picks top-K experts and the post-combine
  weights expert outputs correctly, the LM head projects to
  `vocab_size`, and `KvCache` grows by one slot per forward pass,
- the **real-transformer model**, multi-layer expert id namespacing
  (`global_id = layer * num_experts + local_id`) partitions
  correctly, `safetensors` loaders pull dense tensors with the
  expected shapes, `from_dir` auto-dispatches based on whether
  `.safetensors` files are present, two `step()` calls with the same
  inputs produce the same token id, and the dense-config validator
  rejects bad shapes,
- the **token sampler**, `temperature == 0.0` is greedy `argmax`,
  `top_k == 1` collapses to argmax even at high temperature, top-P
  truncation excludes the tail, the same `(seed, position)` always
  produces the same token, and high temperatures can pick lower-
  ranked logits,
- the **session store**, `put` then `take` round-trips the persisted
  state, `delete` reports prior existence, the TTL evictor drops
  stale entries while keeping fresh ones, and `ttl == 0` disables
  eviction entirely,
- the **batch scheduler**, fused decoder steps are functionally
  equivalent to direct `RealModel::step` calls, and concurrent
  batched wall-clock stays within 1.5× the strictly-sequential
  baseline,
- the **predictive architecture**, the `LocalityMonitor` tracks
  its sliding window correctly (heat counts, hot-set membership,
  out-of-range ids, reset semantics), the `NeuralSpeculator`
  produces distinct sorted top-K ids, deterministically reproduces
  the same prediction for a given seed, drives loss down on a
  fixed target across SGD steps, gracefully handles empty / invalid
  inputs, and `predict_topk_with_probs` returns a normalised
  distribution; the `Engine` integration tests verify that the
  hot set actually pins experts in the cache, the speculator
  correctly records hit / miss telemetry against the gate's
  decision, and `predictive_telemetry` reports non-zero SSD-stall
  microseconds when the engine had to wait on cache-miss reads,
- the **HTTP server**, `/health`, `/metrics`, `/v1/completions`
  (streaming and non-streaming), `/v1/chat/completions` (streaming
  and non-streaming) round-trip, the real-model path actually
  samples from logits, and the empty-prompt error path returns
  `400 Bad Request`.

---

## Energy Efficiency Features

The engine spends almost all of its energy in two places: **moving
expert bytes off the SSD** (per-byte cost: PCIe + NVMe controller +
DRAM write) and **executing the SwiGLU FFN** (per-FLOP cost: SIMD
units + L1/L2 cache traffic). Every change in this section attacks one
of those two terms by reducing the *number* of bytes moved or the
*number* of FLOPs executed, i.e. they reduce work, which is the only
durable way to reduce energy. Knobs that merely shift cost around (e.g.
faster CPU at the same workload) are out of scope.

The headline numbers shipped in `EngineReport.print_summary` are
`bytes_read` (Joules ∝ bytes for SSD reads), `pct_time_io` (the share
of token cycle time the CPU sits waiting on SSD, multiplying its idle
energy), `pinned_count`, and `alias_redirects`. Each subsection below
explains which of these the change moves and why.

### 1. On-disk quantization (`--dtype`)

The engine reads weight bytes straight off the SSD; halving, or
quartering, the byte width of each weight halves / quarters every
read. Five on-disk dtypes are first-class:

| `--dtype` | Bytes / weight | Per-blob header | Dequant kernel | Use |
|---|---:|:---:|---|---|
| `f32` | 4 | none | zero-copy reinterpret | reference / highest fidelity |
| `f16` | 2 | none | per-fetch `f16 → f32` | ~2× less SSD energy than `f32` |
| `int8` | 1 | 12 B (`[gate, up, down]: [f32; 3]` per-tensor scales) | symmetric per-tensor dequant | ~4× less SSD energy than `f32` |
| `q4k` | ~0.5625 | none (block-internal) | `Q4_K_M` 256-block (f16 super-scale + 6-bit sub-scales + 4-bit weights) | GGUF-compatible 4-bit |
| `q4_0` | ~0.5625 | none (block-internal) | `Q4_0` 32-block (f16 scale, symmetric 4-bit nibbles) | the most widely-used 4-bit format; chosen by the predictive-controller spec |

Selectable on **`gen-data`** (synthetic data, every dtype has a
matching generator arm), on **`gguf-convert`** (input format detected
from the GGUF tensor dtype, output written in the same dtype), and on
**`run` / `serve`** (must match the on-disk files). The forward pass
dispatches to `inference::run_inference_*` per dtype, all producing
the same scalar `f32` SwiGLU output, so a benchmark run is a
one-flag diff.

**How this saves energy.** Every cache miss reads
`3 · d_model · d_ff` weights off the SSD. Going from `f32` → `f16`
halves NVMe bandwidth and DRAM writes; `int8` quarters them; `q4k` /
`q4_0` get an additional ~30% on top. The dequantisation step is
`d_model · d_ff` cheap scalar ops per expert; on modern SIMD this is
far less energy than the bytes-moved savings recover.

### 2. 2nd-order Markov + gate-lookahead prefetching

The `PredictiveLoader` now keeps two count tables: the legacy
`(prev -> next)` 1st-order rows and a sparse `(prev_prev, prev) -> next`
2nd-order map (`rows2`). `predict_next2(prev_prev, prev)` blends the
two distributions 50/50 and returns the top-fanout next experts. The
engine remembers the previous *and* previous-previous active sets and
calls the 2nd-order path automatically once two tokens of history are
available.

**How this saves energy.** Speculative prefetches that *miss* burn full
SSD-read energy for nothing. A sharper conditional distribution
(`p(next | prev, prev_prev)` is strictly more informative than
`p(next | prev)`) means we issue **fewer wasted prefetches** for the
same prefetch hit rate, or alternatively hit the same hit rate at
**lower fanout**, both reduce `bytes_read` directly. The 2nd-order
table is sparse (`HashMap` keyed by `(prev_prev, prev)`), so memory
overhead stays tiny.

### 3. Partial weight loading (`--partial-load-fraction`)

`OwnedExpertWeights::from_bytes_partial` accepts a packed-column blob
produced by `NvmeStorage::read_expert_columns`, only the M most
relevant input dimensions of `gate_proj` and `up_proj` are loaded
(plus the full `down_proj`). `forward_partial` sums the dot products
only over those M columns. The fraction `M / d_model` is configurable
via `--partial-load-fraction` and `storage.partial_load_fraction`.

**How this saves energy.** Each gate/up matmul today is `d_ff · d_model`
multiply-adds per expert. Reducing to M loaded columns turns those
into `d_ff · M` MAdds, **proportional to M / d_model**. With
M = d_model / 2 you save ~50 % of the gate/up FLOPs, which is most of
the per-expert compute cost. The forward pass remains correct on a
finite, well-shaped output; the trade is a small, bounded accuracy
delta. `1.0` (default) preserves byte-exact legacy behaviour. The SSD
*bandwidth* saving requires a column-major on-disk layout, that's a
follow-up change to the offline extractor; today's runtime saves the
compute term and prepares the API surface for the bandwidth term.

### 4. io_uring with registered fixed buffers (`--features io_uring`, `--io-uring`)

`io_uring_storage.rs` is a real Linux backend, gated behind the
`io_uring` cargo feature. `IoUringStorage::new` calls
`io_uring_register(IORING_REGISTER_BUFFERS)` over every
`BufferPool::raw_iovecs` slot at startup; `read_expert_fixed` then
submits an `IORING_OP_READ_FIXED` SQE that references the buffer
*index* and waits for the completion. A batched submission entry point
(`read_experts_batch_fixed`) pushes K SQEs and `submit_and_wait(K)`
once when a token misses on multiple experts. Per-expert file
descriptors are cached on the first call, mirroring `NvmeStorage`'s
`fd_for` behaviour. `--io-uring` on the CLI now actually probes the
backend at startup (logging `registered_buffers`) and surfaces a clean
error path when the kernel rejects the registration (older kernels,
restrictive sandboxes); we then keep running on the portable `pread`
backend so the run completes either way.

**How this saves energy.** Each `pread(2)` cache miss today is one
syscall plus a per-read iovec setup. With `io_uring` + fixed buffers,
a token that misses on K experts becomes **one syscall**
(`io_uring_enter`) referencing K pre-pinned buffer indices, the
kernel never has to walk the user mapping or pin pages on the hot
path. Published microbenchmarks report 30-50 % less per-read CPU on
NVMe-class SSDs. CPU time during I/O wait is pure overhead, the same
bytes were going to leave the device either way; `io_uring` just makes
the kernel cheaper, which is energy out of the budget. Build with
`cargo build --release --features io_uring` (Linux only) to enable;
the engine selects the `pread(2)` backend by default so the portable
path stays the warning-free default.

### 5. Frequency-based expert pinning (`--pin-after-observations N`)

`ExpertCache` now holds a `pinned: HashSet<u32>`. Once the engine has
observed an expert as a routing destination N times, it calls
`cache.pin(id)` and the LRU eviction path skips that id permanently.
`evict_lru` and `insert` both walk past pinned ids; the cache returns
`None` from `evict_lru` if every entry is pinned (caught by the engine's
existing "wait for a free buffer" loop, so progress is preserved).

**How this saves energy.** MoE workloads have heavy-tailed expert
usage, a small subset of experts handles a large fraction of tokens.
A plain LRU still evicts those popular experts when a flurry of cold
ones arrives, paying their full SSD read energy on the next miss. By
pinning the demonstrated-hot ones, **every subsequent activation of
those experts is a cache hit** (zero SSD bytes, zero I/O wait). With a
realistic Zipfian routing distribution this typically eliminates the
top-N contributors to `bytes_read`. `0` (default) preserves legacy
behaviour.

### 6. Expert deduplication via alias map (`--alias-map`)

`scripts/compute_expert_aliases.py` scans every `expert_<id>.bin` in a
data directory, computes pairwise cosine similarity over the full
weight blob, and emits a JSON map `{ src_id: canonical_id, ... }` for
pairs whose similarity is above a threshold (default 0.995). The
engine loads it via `Engine::with_alias_map` (CLI flag `--alias-map`)
and resolves every routed and predicted id through it before
consulting the cache.

**How this saves energy.** Without aliasing, two near-identical experts
each consume one cache slot and one SSD read on first activation,
even though their weight bytes are nearly the same. With the map,
both expert ids resolve to a *single* canonical id; the cache holds
one resident copy, the SSD reads it once, and **every redirect counted
in `EngineReport.alias_redirects` is a cache lookup that didn't burn
SSD bytes**. The detection runs offline (no runtime cost), and the
runtime overhead is one `HashMap` lookup per routed expert per token.
Empty / absent maps disable the feature entirely.

### 7. Predictive architecture (`S ∪ L ∪ M` speculative I/O)

The original prefetcher is a single Markov-chain predictor (call it
`S`). Real MoE traces have two other exploitable signals it can't
see:

* **temporal locality**, within a topic, the same handful of experts
  fire over and over for hundreds of tokens, *regardless* of the
  precise transition the chain just made;
* **semantic intent**, the hidden state itself encodes which
  experts the gate is *about* to pick, often before the routing
  decision is finalised.

Two opt-in components capture those signals and **union** their
hints with the Markov chain's into a single speculative-I/O fetch
set `E = S ∪ L ∪ M`:

* **L, `LocalityMonitor`** (`router::LocalityMonitor`). A sliding
  window of the most recent `locality_window` routing observations
  with a flat `Vec<u32>` heat map. An expert whose count crosses
  `locality_threshold_pct * window_len` is "hot" and is **pinned in
  the LRU cache** until it falls back below the threshold,
  protected from eviction even when the Markov chain wanders
  elsewhere. Reconciliation runs after every token: ids that just
  joined the hot set get pinned, ids that just left get unpinned.
* **M, `NeuralSpeculator`** (`router::NeuralSpeculator`). A tiny
  two-layer MLP (`d_model -> hidden -> num_experts`, ReLU + softmax,
  default `hidden = 128`) trained **online** by SGD against the
  gate's actual top-K decision at each token. Cheap enough to run
  on the critical path, with He-uniform init, gradient clipping at
  `±1`, and a `clamp_finite` weight guard so a stuck speculator
  never NaNs out the predictor. Training is dispatched to a
  dedicated **off-path worker thread** through a bounded `mpsc`
  queue: `predict_topk` on the hot path takes a read-lock snapshot
  of `(W1, b1, W2, b2)`; the worker drains the queue and writes new
  weights with `try_write_for` so the predictor is never blocked by
  backprop. The queue is bounded so a runaway producer can't pin
  unbounded memory, when full, the newest sample is dropped
  (training is a *prefetch hint*; the real routing still flows
  through the gate downstream).

Both arms are wired into `Engine::union_prefetch`: per token, the
engine builds the union of (a) the predictor's `predict_next2(prev_prev,
prev)` Markov hint `S`, (b) the locality monitor's `hot_set(threshold)`
`L`, and (c) the speculator's `predict_topk(hidden_state)` `M`,
deduplicates against ids already in flight or already resident,
and spawns prefetches for the rest. The unified ranking is computed by
`PredictiveLoader::predict_unified`, which combines all three signals
with weights `0.42 · speculator + 0.33 · markov + 0.25 · locality` and
returns the top-fanout ids; an expert that lights up in every arm is
therefore prioritised over one that lights up in only one. The
weighting encodes "speculator is the strongest signal, Markov is
next, locality is the weakest tiebreaker" — see
`PredictiveLoader::predict_unified` in `router.rs` for the canonical
constants.

Online speculator training is **dispatched to an off-path worker
thread** through a bounded queue, so a `predict_topk` on the hot
path never blocks on backprop, it takes a brief read-lock on the
current `(W1, b1, W2, b2)` snapshot, the worker takes a
`try_write_for` (and drops the sample if the lock isn't immediately
available) so the predictor is never starved.

**How this saves energy.** `S` alone misses two failure modes:
prefetches **wasted** when the chain wanders out of the active
topic (the resident hot set still gets evicted by cold experts,
then re-paged on every loopback), and prefetches **never issued**
when the gate is about to pick an expert the chain has no
short-history evidence for. Adding `L` keeps the recently-hot set
pinned so it is read **at most once per topic** instead of
re-paged every few tokens; adding `M` lets the prefetcher react
to the *hidden state* before the routing decision lands, so cache
misses on real-but-rare transitions can be hidden behind compute.
Both reduce the count of cache-miss reads, the dominant byte
mover in `bytes_read`, at the cost of a tiny CPU budget (one
small MLP forward + SGD step per token, plus one ring-buffer
update) that is far below the energy cost of even a single
NVMe expert read.

**Telemetry.** The engine maintains four atomic counters
(`spec_hits`, `spec_misses`, `locality_hits`, `locality_misses`)
and a cumulative `total_ssd_stall_us`, all readable through
`Engine::predictive_telemetry()` and exported as Prometheus
counters / a histogram on `/metrics`:

| Metric | Meaning |
|---|---|
| `mer_speculator_hits_total` | Per-token speculator predictions that intersected the gate's actual top-K. |
| `mer_speculator_misses_total` | Per-token speculator predictions that did not. The ratio is the speculator's running accuracy. |
| `mer_speculator_accuracy_total` | Tokens for which the speculator's **top-1** prediction matched the gate's actual top-1 routed expert. The primary quality signal called out by the Omniscient Predictive Architecture spec; divide by tokens-generated to read accuracy as a fraction. |
| `mer_locality_hits_total` | Routed experts that were already in the locality monitor's hot set at routing time (would-be cache miss avoided by pinning). |
| `mer_locality_misses_total` | Routed experts that were not. |
| `mer_ssd_stall_seconds` | Histogram of cumulative SSD critical-path stall time per token, the wall-clock window the engine spent blocked waiting for cache-miss reads to land. The headline number the L / M arms aim to drive down. |

The CLI run summary (`print_summary`) appends an extra line when
either arm is enabled, e.g.:

```
predictive:    locality=on (hit_rate=64.32%)  speculator=on (accuracy=58.10%)  ssd_stall=12.4ms
```

When `[predictive]` is left at its defaults (everything off), the
engine takes the legacy Markov-chain prefetch path bit-for-bit and
the new line is omitted from the summary, so existing benchmarks
and golden outputs are unchanged.

### How to combine them

The seven knobs are independent and compose freely. The first six
live on the CLI; the predictive arms (`L` + `M`) are config-driven
and apply equally to the CLI run loop and the HTTP server. A
reasonable "low-energy" preset on a Linux NVMe box looks like:

```bash
micro-expert-router run \
    --data-dir ./data \
    --dtype f16 \                  # Change 1: 2× less SSD energy per miss
    --predict-fanout 4 \           # Change 2: 2nd-order kicks in automatically
    --partial-load-fraction 0.5 \  # Change 3: ~50% less gate/up compute
    --io-uring \                   # Change 4: cheaper kernel I/O path
    --pin-after-observations 8 \   # Change 5: hot experts never re-read
    --alias-map ./data/aliases.json  # Change 6: deduplicated experts share a slot
```

...and, alongside it in `config.toml` (or the equivalent `serve`-time
TOML), Change 7, the predictive `L` + `M` arms:

```toml
[predictive]
locality_enabled       = true
locality_window        = 256
locality_threshold_pct = 0.10
speculator_enabled     = true
speculator_hidden_dim  = 128
speculator_top_k       = 0
```

`print_summary` reports each knob's state and effect (`pinned`,
`alias_redirects`, `dtype`, `partial_load_fraction`, plus a
`predictive:` line when L / M are enabled, locality hit-rate,
speculator accuracy, and cumulative SSD stall) on every run, so
you can verify the energy-saving paths actually engaged.

---

## Limitations / next steps

- **Per-expert kernel dispatcher.** The engine ships a runtime
  CPU-feature dispatcher (`src/kernels/`) that selects between a
  scalar reference path (always on), an AVX-512 fused
  int8-dequant + dot path (`--features avx512`), and an Intel AMX
  tile skeleton (`--features amx`, currently routes back to scalar
  on stable Rust until the tile intrinsics stabilise). The chosen
  backend is logged once at startup. The dense `transformer`
  projections additionally benefit from the existing `simd` /
  `blas` features. Dropping in BLAS / a CUDA kernel via `tch` /
  `candle` / `cudarc` for the SwiGLU FFN remains a clean
  extension — `inference::ExpertWeights::from_bytes` already does
  zero-copy reinterpretation of the buffer.
- **NUMA budget.** `MER_PIN_CORES=N` is honoured at startup to
  `sched_setaffinity(2)` the process to the first `N` CPUs of
  NUMA node 0 (Linux only, best-effort; no-op + warn elsewhere).
  See `src/numa.rs`. Real per-ring per-node pinning would still
  need one io_uring ring per node and per-node buffer pools, a
  deeper refactor.
- **Streaming GGUF reader.** `gguf-convert` defaults to a streaming
  reader (`crate::gguf::GgufStreamReader`) that parses only the
  header + tensor-info table into memory and reads each tensor
  body on demand with `seek + read_exact`. Pass `--legacy-eager`
  to fall back to the in-memory `GgufFile::open` reader for
  small fixtures or compatibility testing. The streaming path is
  a strict win for ≥ 100 GB checkpoints.
- **Unified Tensor Header (U.T.H.).** Every `expert_<id>.bin`
  produced by `gguf-convert` is prefixed by default with a
  64-byte U.T.H. (`UTH1` magic + dtype + shape + quant-scale
  offset + AMX tile hint + flags), page-padded to 4 KiB so the
  weight payload still starts at a `O_DIRECT`-friendly boundary.
  Older consumers that need the legacy bare-payload layout can
  pass `--no-uth`. The runtime
  (`ExpertResident::data()` / `expert_cache.rs`) transparently
  strips the header when present, so the rest of the engine sees
  exactly the same bytes as before.
- **Primary / Shadow buffer pool.** `BufferPool::new_with_shadow`
  carves the pool into a primary half (resident LRU) and an
  optional shadow half reserved for speculative prefetches.
  Speculation calls `try_acquire_shadow` so it can never starve
  primary work; on confirmation, `promote_shadow` does a
  zero-copy slot-tag swap so the same backing memory becomes a
  resident without re-reading the SSD.

