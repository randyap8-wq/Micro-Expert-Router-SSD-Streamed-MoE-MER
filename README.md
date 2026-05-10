# Micro-Expert-Router — SSD-Streamed MoE Execution Engine

A Rust execution engine for **Mixture-of-Experts** models that keeps the
router resident in RAM and **hot-swaps individual experts on demand** from a
PCIe-attached NVMe drive into a pool of pre-allocated, page-aligned RAM
buffers using **`O_DIRECT`** positional reads (`pread(2)` via
`tokio::task::block_in_place`, kernel-page-cache bypass). Each routed
expert then **executes a real Mixtral / Llama-style
SwiGLU FFN forward pass** directly over the bytes that just arrived from the
drive.

The premise is straightforward: a **modern PCIe-4 / 5 NVMe SSD sustains
6–14 GB/s** of sequential read; a Mixtral-class expert is ~88 MB; pulling
the top-K active experts per token therefore costs a few milliseconds of
I/O even when the *full* parameter set is 10–100× DRAM. So you can run
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

1. **mmap the weights** — relies on the OS page cache. Works, but the kernel's
   prefetcher knows nothing about the routing pattern, you double-copy through
   the page cache, and you can't bypass the readahead heuristics.
2. **Manage the cache yourself** — what this engine does. Open each expert as
   its own file, read it with `O_DIRECT` `pread(2)` (dispatched off the
   Tokio runtime via `block_in_place`) so the bytes go
   directly from the NVMe DMA engine into a page-aligned RAM buffer, and run
   a custom LRU + speculative prefetcher driven by the router's own
   activation history.

### End-to-end pipeline

```
        +-----------+     +-------------+     +-----------+     +------------------+
token → |  Router   | →  | Expert IDs   | →  | LRU Cache | →  | SwiGLU FFN       |
        |  (top-K)  |    |  e.g. [3,7]  |    +-----+-----+     | per expert,      |
        +-----------+    +--------------+         | miss       | combine outputs  |
                                                  ↓            +------------------+
                                         +------------------+
                                         | BufferPool slot  | ←───┐
                                         |  (aligned, pre-  |     │
                                         |   allocated)     |     │
                                         +--------+---------+     │
                                                  ↓               │
                                         +------------------+     │ on Arc drop
                                         |  pread(2) read   |     │ (LRU evict
                                         |  O_DIRECT, no    |     │  or buffer
                                         |  page cache      |     │  release)
                                         +--------+---------+     │
                                                  ↓               │
                                         NVMe SSD → DMA → RAM ────┘
                                                  ↓
                                         bytes reinterpreted as
                                         f32 weights → matmul
```

After every token the engine also updates a first-order **Markov model** of
expert transitions and uses it to **speculatively prefetch** the most likely
next experts on the side. The prefetch path is non-blocking and
non-evicting — it never starves a real cache miss.

The router *itself* is also a deterministic first-order Markov chain over
expert ids — not a random uniform top-K sampler. Synthetic runs use
**clustered locality** (4 expert groups, 0.9 in-cluster transition
probability) so the prefetcher has signal to learn from; real Mixtral
routing traces can be loaded directly via `--router-matrix`. See the
[Routing model](#routing-model--markov-chain-over-expert-ids) section.

### What "running" actually does

For each token, the engine:

1. asks the **Markov-chain router** for K distinct expert ids (sampled from
   `P(next | last_expert)` under the configured transition matrix —
   either generated with cluster locality or loaded from a file);
2. for each id, hits the LRU cache or streams the expert file off the
   NVMe drive into a page-aligned pool buffer via `O_DIRECT`;
3. **reinterprets the buffer as `f32` weight matrices** (`gate_proj`,
   `up_proj`, `down_proj`, in that order, row-major — the standard
   Mixtral / Llama / DeepSeek FFN layout);
4. runs a real **SwiGLU FFN forward pass**:
   `y = down_proj · ( silu(gate_proj · x) ⊙ (up_proj · x) )`
   — or, with `--io-only`, XOR-checksums every read byte instead, to
   isolate pure SSD-streaming cost from FFN compute;
5. averages the K expert outputs (mock combine — a real router would do a
   weighted sum using its softmax gates);
6. updates the Markov predictor and kicks off speculative prefetches.

The forward pass is plain scalar `f32` Rust — no BLAS, no SIMD, no GPU.
That's deliberate: the project's thesis is about **storage bandwidth**,
not compute, so the kernel is just real enough to exercise every byte
that came off the drive (compiler can't fold it away) and to surface a
believable compute-vs-I/O latency picture in the per-token logs.

---

## Architecture

The Rust crate (`rust-engine/`) is organised into single-responsibility modules:

| Module | Responsibility |
|---|---|
| `aligned_buffer` | Heap-allocated, page-aligned buffer (`std::alloc::alloc` with a `Layout`). The defining requirement of `O_DIRECT`: kernel rejects unaligned buffers with `EINVAL`. |
| `buffer_pool` | Fixed-capacity slab of `AlignedBuffer`s. Hands out `PooledBuffer` RAII guards; dropping a guard returns the buffer to the free list and notifies waiters. This is the literal "pre-allocated RAM buffer" the spec asks for. |
| `expert_cache` | LRU map `expert_id → Arc<ExpertResident>`. Eviction returns the `Arc`; once all references drop, the buffer goes back to the pool automatically. |
| `io_provider` | NVMe storage layer. Opens each expert as its own file (`O_DIRECT` on Linux), keeps fds resident, and reads via `tokio::task::block_in_place` + `pread(2)` (`FileExt::read_at`). Includes a `gen-data` helper to create synthetic test files and a portable Unix fallback for development on macOS. |
| `router` | `TopKRouter` (deterministic first-order Markov chain over expert ids — clustered locality by default, or load a precomputed `N×N` transition matrix via `--router-matrix`) and `PredictiveLoader` (online sparse first-order Markov predictor over observed transitions, smoothed with a uniform Laplace prior so predictions are immediately usable). |
| `inference` | Real SwiGLU expert FFN (`y = down · (silu(gate·x) ⊙ (up·x))`) computed in scalar `f32` directly over the bytes streamed off NVMe. Reinterprets each pool buffer as three weight matrices (no copy). Replace with `tch`/`candle`/`cudarc` for SIMD / GPU. |
| `engine` | Top-level orchestrator. Owns the router/predictor/cache/pool/storage, drives the per-token cycle, schedules prefetches, records HDR histograms. |
| `main` | `clap`-based CLI with `gen-data` and `run` subcommands, structured `tracing` logs, `--first-token 3,7` to reproduce the spec example, `--io-only` for pure-I/O benchmarking, `--force-ssd` to refuse page-cache shortcuts, and auto-loading of `metadata.json` (written by `scripts/extract_mixtral_experts.py`) so a real Mixtral checkpoint runs with no further flags. |

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
- **Online sparse Markov predictor with prior.** Per-row sparse maps of
  observed `(from, to)` counts plus a uniform Laplace prior (every cell
  starts at an implicit count of 1). On every token transition we
  increment `counts[from][to]`. `predict_next` returns
  `(count + prior) / row_total`, sorted descending and filtered by
  `min_prob`. Sparse-by-row means memory scales with the number of
  *visited* `(from, to)` pairs, not `O(N²)` up front — important once
  `N` reaches Mixtral 8x22B / DeepSeek-V3 expert counts.
- **Deterministic Markov-chain router.** The router itself samples from
  `P(next | last_expert)` under a fixed `N×N` transition matrix that is
  either generated with structured cluster locality
  (`--router-clusters`, `--router-intra-p`) or loaded from a file
  (`--router-matrix`). Given a `--seed`, an entire run is reproducible.
- **Pluggable I/O backend.** The hot path uses `tokio::task::block_in_place`
  to dispatch a synchronous `pread(2)` (via `std::os::unix::fs::FileExt::read_at`)
  on the current Tokio worker; the runtime donates that worker to blocking
  work and other tasks are picked up by sibling workers. On non-Linux Unix
  (e.g. macOS dev boxes) the same code path runs without `O_DIRECT` so the
  engine still runs end-to-end during development.

### "Why `pread` + `block_in_place` and not io_uring?"

Earlier drafts used the `rio` io_uring wrapper, but `rio 0.9.4` carries an
unfixed use-after-free advisory and the crate is unmaintained, so the
dependency was removed. The current backend is intentionally simple —
positional `pread` is `O_DIRECT`-compatible, deep-queue-friendly on NVMe,
and avoids touching the file offset so concurrent reads against the same
fd are safe. Future `io_provider` upgrade paths:

| Crate | Verdict for this workload |
|---|---|
| **`pread` + `block_in_place`** *(used here)* | Zero extra deps, works on every Unix, exercises the full `O_DIRECT` + page-aligned-buffer + LRU + prefetch logic. The compute and storage stay observably distinct in the per-token logs. |
| **`tokio-uring`** | Best ergonomic fit if you live in Tokio. Single-threaded per ring, requires `#[tokio_uring::start]` instead of `#[tokio::main]` — would force a runtime restructuring. |
| **`io-uring`** (raw, by tokio-rs) | The thinnest binding to the kernel ABI. Lets you use **registered (fixed) buffers** + **registered files**, which removes per-op address validation in the kernel — the single biggest win for sustained NVMe throughput. **This is what a production build of this engine would use.** |
| **`glommio`** | Thread-per-core, polled io_uring. Made for NVMe-bound workloads (ScyllaDB heritage). For a pure expert-fetch service pinning workers to cores feeding local rings, glommio is arguably the *fastest* answer on Linux. Trade-off: incompatible with Tokio (it owns the runtime). |

The clean separation between `io_provider` and the rest of the engine means
swapping the backend for `io-uring` or `glommio` is a self-contained change.

---

## Building and running

### Prerequisites

- **Linux kernel ≥ 3.0** is enough — the I/O path uses `pread(2)` with
  `O_DIRECT`, not io_uring. No special kernel features are required.
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
cargo build --release
```

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
# (the engine's default — the whole point is to stream from SSD, so a
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

The same engine — same SSD-streaming expert cache, same `O_DIRECT`
reads, same SwiGLU FFN over the bytes that just arrived from disk — can
be run as a long-lived HTTP server with an OpenAI-compatible API.

```bash
# Start the server. Reads everything from a TOML config file (see
# `config.toml` at the repo root for an annotated example).
./target/release/micro-expert-router serve --config ../config.toml
```

Endpoints:

| method   | path                    | purpose                                            |
| -------- | ----------------------- | -------------------------------------------------- |
| `GET`    | `/health`               | liveness probe (`{"status":"ok",...}`)             |
| `GET`    | `/metrics`              | Prometheus text format: cache hit rate, request latency histograms, tokens generated, per-token I/O wait |
| `POST`   | `/v1/completions`       | OpenAI text-completion shape (`prompt`, `max_tokens`, …) |
| `POST`   | `/v1/chat/completions`  | OpenAI chat-completion shape (`messages`, …)       |

Example:

```bash
curl -s http://127.0.0.1:8080/v1/completions \
  -H "content-type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":32}' | jq .
```

The server is intentionally **stateless per request** in this PR: each
request drives the model for `max_tokens` cycles and returns the decoded
tokens in one shot. Streaming (`stream: true`) is accepted in the
request body for OpenAI compatibility but currently produces a
non-streaming response. Per-request KV caches mean that concurrent
axum requests never alias each other's attention state — that is the
foundational request-scheduler piece that token-level cross-request
batching will layer on top of.

#### Real-transformer pipeline (gist Phase 5/6)

By default the server runs the **legacy benchmark generator**: each
request drives `Engine::generate` for `max_tokens` cycles and synthesises
a deterministic id stream. The SSD-streaming substrate is exercised
identically.

When `[real_transformer].enabled = true` in the TOML config, requests go
through the **full decoder forward pass**:

```
embedding → for each layer: ( RMSNorm → MultiHeadSelfAttention → +
                              RMSNorm → LinearGate.route → moe_step → +)
            → final RMSNorm → LMHead → argmax
```

`moe_step` is what reads expert weights from SSD via the LRU cache, so
the same hits / misses / I/O wait counters get populated regardless of
which path drives the loop.

The dense (resident) weights — embedding, attention projections, MoE
gate, RMSNorm gains, LM head — are loaded from the directory in
`real_transformer.weights_dir` (one `.bin` file per tensor, raw
little-endian `f32`; see `RealModel::from_dir` for the file-name
schema). Tensors that aren't present fall back to a deterministic
seeded initialisation, so the engine always has an end-to-end runnable
path even without real model files. Multi-layer experts share the
existing single-namespace cache via the global addressing scheme
`global_id = layer * num_experts + local_id` — so the run summary
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
rope_base = 10000.0
rms_eps = 1e-6
seed = 0xC0FFEE
```

#### Optional row-parallel matmul (`simd` feature)

The dense projections inside `TransformerLayer` and `LMHead` are routed
through `transformer::matmul_row_major`. With the `simd` cargo feature
enabled, that function dispatches to a `std::thread::scope`-based
row-parallel implementation (no extra crate dep — output rows are
disjoint, so no synchronisation is needed):

```bash
cargo build --release --features simd
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

Configuration lives in TOML — see [`config.toml`](./config.toml) for
the full annotated schema (server bind address, model dimensions, cache
slots, `O_DIRECT` block alignment, predictive prefetch fanout, optional
tokenizer path).

To **isolate pure I/O cost** (skip the SwiGLU FFN; XOR every byte read
to force the page in):

```bash
./target/release/micro-expert-router run --io-only --tokens 200 ...
```

To **refuse any page-cache shortcut** (Linux only — fails fast if
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
bigger `d_model`/`d_ff` the `compute` row grows linearly — exactly the
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

micro-expert-router run
  --data-dir <PATH>          Directory with expert_<id>.bin files
                              (auto-loads metadata.json if present)
  --num-experts <N>          Total experts in the model
  --expert-size <BYTES>      Must match gen-data
  --d-model <N>              Must match gen-data
  --d-ff <N>                 Must match gen-data
  --cache-slots <N>          Resident experts (default 4; warns if > 16)
  --top-k <K>                Active experts per token (default 2, distinct)
  --tokens <N>               Stream length
  --predict-fanout <N>       Prefetch candidates per token (default 2)
  --predict-min-prob <P>     Skip prefetch below this probability (default 0.05)
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
  --token-pause-us <N>       Sleep between tokens to throttle the stream
  --seed <U64>               PRNG seed for reproducibility
```

### Running on real Mixtral weights

`scripts/extract_mixtral_experts.py` dumps a single transformer
layer's expert FFNs from a Hugging Face Mixtral checkpoint into the
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

The `metadata.json` written by the script lets `run` auto-fill
`--num-experts`, `--d-model`, `--d-ff`, `--top-k`, and `--expert-size`
so the second command needs no further flags. Each Mixtral 8x7B expert
is ~88 MiB (zero-padded to a 4 KiB multiple) — ~700 MiB on disk for
one layer, fully streamable from any modern NVMe.

### Routing model — Markov chain over expert ids

The router is a **deterministic first-order Markov chain**, not a
random uniform top-K sampler: this is the property that makes the
prefetcher worth running. Two ways to build the chain:

1. **Generated** (default): experts are partitioned into
   `--router-clusters` groups (by `id % cluster_count`) and the chain
   stays inside its current cluster with probability
   `--router-intra-p` (default `0.9`). This produces the same
   "topic-sticky" behaviour real MoE traces show — the predictor
   converges quickly and prefetch hit rate climbs above 60%.
2. **Loaded** (`--router-matrix path.txt`): supply a whitespace-separated
   `num_experts × num_experts` matrix of `f64` transition probabilities,
   row-major. Rows are normalised to sum to 1. Use this to feed a real
   Mixtral routing trace (e.g. produced by hooking `block_sparse_moe`'s
   gate softmax during a Hugging Face inference run) directly into the
   engine.

Given a fixed `--seed`, the routed sequence is fully reproducible.

### macOS

`O_DIRECT` is Linux-only. On macOS the engine automatically falls back
to buffered reads (`--no-direct`) and prints a startup warning that
measured I/O latency will include OS page-cache effects (and therefore
under-report cold-NVMe latency). Use a Linux host on a real NVMe device
for clean numbers.

---

## What can it actually run today?

**Today, in this repository: a real Mixtral / Llama-style SwiGLU expert
FFN over weights streamed from NVMe.** Each routed expert performs the
exact `down · (silu(gate·x) ⊙ (up·x))` block that every modern sparse
MoE transformer uses for its experts — at synthetic, configurable
dimensions (default `d_model=512, d_ff=2048`).

What is **still mocked**:

- **The router** is a deterministic Markov chain over expert ids
  (clustered locality by default, or load a real Mixtral routing-trace
  matrix via `--router-matrix`) — not a learned `softmax` over a
  gating projection driven by the actual hidden state. The transition
  matrix is fixed for a given run; it doesn't condition on `x`.
- **Combining** averages the K expert outputs; a real gate-weighted sum
  using router probabilities is one line of code away once a real router
  is wired in.
- **Attention, embedding, layer norm, the residual stream, and tokenizer**
  are not implemented. Only the *expert FFN* — the dominant weight-bound
  block in every MoE — is real.

So: the engine demonstrates **the per-expert compute path of a sparse MoE
transformer**, end-to-end, with weights paged off the SSD. Wiring the
remaining transformer machinery (attention, layer norm, embeddings) and a
real tokenizer is the missing step to a turn-key model server. The
expected drop-in path is to replace `inference::run_inference` with a
call into a tensor library such as `candle`, `tch`, or `cudarc`.

Real Mixtral expert weights can already be fed to the engine end-to-end
via [`scripts/extract_mixtral_experts.py`](./scripts/extract_mixtral_experts.py),
which dumps a single layer's experts into the on-disk format the
engine expects (plus a `metadata.json` that `run` auto-loads). See
[Running on real Mixtral weights](#running-on-real-mixtral-weights).

That said, the architecture (per-expert files, fixed expert size,
top-K activation, LRU + prefetch) is shaped specifically for **sparse
Mixture-of-Experts transformers where the expert FFNs are the dominant
weight**. Concretely, the following published models drop into this layout
with no architectural changes — only a real attention/embedding kernel and
a sharding script that splits their `safetensors` into one
`expert_<id>.bin` per expert (or per-layer-per-expert, see "Sharding
granularity" below):

| Model | Total params | Active / token | Experts | Top-K | Per-expert FFN (bf16) | Notes |
|---|---|---|---|---|---|---|
| **Mixtral 8x7B** | ~47 B | ~12.9 B | 8 × 32 layers | 2 | ~88 MB | Canonical fit. ~22 GB of expert weight, easily streamed from a single PCIe-4 NVMe. |
| **Mixtral 8x22B** | ~141 B | ~39 B | 8 × 56 layers | 2 | ~240 MB | Comfortable on PCIe-5 NVMe. Cache 8–16 experts; prefetcher learns the routing well. |
| **Phi-3.5-MoE-instruct** | ~42 B | ~6.6 B | 16 × 32 layers | 2 | ~80 MB | Smaller experts, more of them — exercises the predictor harder. |
| **Qwen1.5-MoE-A2.7B / Qwen2-MoE** | ~14 B | ~2.7 B | 60 × 24 layers | 4 | ~10 MB | Fine-grained experts; ideal for demonstrating prefetch hit-rate. |
| **DeepSeek-MoE 16B** | ~16.4 B | ~2.8 B | 64 routed + 2 shared × 28 layers | 6 | ~5–8 MB | "Shared experts" should be pinned (use `--first-token` to warm them, set `--cache-slots` ≥ shared count). |
| **DeepSeek-V2-Lite / V2** | 16 B / 236 B | 2.4 B / 21 B | 64–160 × many layers | 6 | small | Same shape, larger scale. V2-full needs PCIe-5 + ≥ 32 cache slots to keep p99 sane. |
| **DeepSeek-V3 / V3-0324** | 671 B | 37 B | 256 routed + 1 shared × 61 layers | 8 | small but many | Stress test of the design — ~15 K expert tensors. Sharding at per-layer-per-expert is mandatory. |
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
serving layer once a tensor backend is wired in** — LangChain, LangGraph,
Microsoft AutoGen, CrewAI, llama-index, OpenAI-Agents-SDK, and the
`smolagents` family are all framework-agnostic about the model server. The
practical path is: this engine → an OpenAI-compatible HTTP shim →
the agent framework's standard client.

### Sharding granularity

Two ways to lay an MoE on disk; both are supported by the engine
unchanged — only `--num-experts` and `--expert-size` differ:

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
- the **predictor** (sparse first-order Markov) learns simple
  transitions, respects `min_prob`, falls back to the Laplace prior
  when nothing has been observed, counts only real observations, and
  handles zero fanout,
- the `f32` weight-view partitions buffers correctly,
- the SwiGLU forward pass produces finite, deterministic outputs of the
  correct shape, and zeroed weights yield a zero output,
- the `metadata.json` mini-parser handles both compact and
  pretty-printed JSON.

---

## Energy Efficiency Features

The engine spends almost all of its energy in two places: **moving
expert bytes off the SSD** (per-byte cost: PCIe + NVMe controller +
DRAM write) and **executing the SwiGLU FFN** (per-FLOP cost: SIMD
units + L1/L2 cache traffic). Every change in this section attacks one
of those two terms by reducing the *number* of bytes moved or the
*number* of FLOPs executed — i.e. they reduce work, which is the only
durable way to reduce energy. Knobs that merely shift cost around (e.g.
faster CPU at the same workload) are out of scope.

The headline numbers shipped in `EngineReport.print_summary` are
`bytes_read` (Joules ∝ bytes for SSD reads), `pct_time_io` (the share
of token cycle time the CPU sits waiting on SSD, multiplying its idle
energy), `pinned_count`, and `alias_redirects`. Each subsection below
explains which of these the change moves and why.

### 1. fp16 quantization on disk (`--dtype f16`)

Each weight is stored as a 2-byte little-endian `f16` instead of a
4-byte `f32`. The engine dequantises on the fly via
`OwnedExpertWeights::from_bytes_f16` and runs the same SwiGLU forward
pass on the resulting `Vec<f32>`.

**How this saves energy.** Every cache miss reads
`3 · d_model · d_ff` weights off the SSD. Halving the byte width
halves the bytes the NVMe controller has to deliver, halves the PCIe
traffic, and halves the DRAM writes — roughly a **2× reduction in
SSD-read energy per miss**. That is by far the dominant term in any
benchmark with a non-trivial miss rate. The dequantisation step is
~`d_model · d_ff` cheap `f16 -> f32` conversions per expert; on modern
SIMD this is far less energy than the bytes-moved savings recover.

`gen-data` and the offline extractor both accept `--dtype`, so you can
choose per-run whether to spend the disk space on f32 (highest
fidelity) or f16 (lowest energy).

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
**lower fanout** — both reduce `bytes_read` directly. The 2nd-order
table is sparse (`HashMap` keyed by `(prev_prev, prev)`), so memory
overhead stays tiny.

### 3. Partial weight loading (`--partial-load-fraction`)

`OwnedExpertWeights::from_bytes_partial` accepts a packed-column blob
produced by `NvmeStorage::read_expert_columns` — only the M most
relevant input dimensions of `gate_proj` and `up_proj` are loaded
(plus the full `down_proj`). `forward_partial` sums the dot products
only over those M columns. The fraction `M / d_model` is configurable
via `--partial-load-fraction` and `storage.partial_load_fraction`.

**How this saves energy.** Each gate/up matmul today is `d_ff · d_model`
multiply-adds per expert. Reducing to M loaded columns turns those
into `d_ff · M` MAdds — **proportional to M / d_model**. With
M = d_model / 2 you save ~50 % of the gate/up FLOPs, which is most of
the per-expert compute cost. The forward pass remains correct on a
finite, well-shaped output; the trade is a small, bounded accuracy
delta. `1.0` (default) preserves byte-exact legacy behaviour. The SSD
*bandwidth* saving requires a column-major on-disk layout — that's a
follow-up change to the offline extractor; today's runtime saves the
compute term and prepares the API surface for the bandwidth term.

### 4. io_uring with registered fixed buffers (`--io-uring`)

A new feature-gated `io_uring_storage.rs` module declares the API
surface for the Linux `io_uring` backend with **registered fixed
buffers**. `BufferPool::raw_iovecs` exposes every pool slot as a stable
`(ptr, len)` so the kernel can be told about all of them up front
exactly once. `--io-uring` accepts the flag, logs which path is in use,
and falls back gracefully on non-Linux / non-feature builds.

**How this saves energy.** Each `pread(2)` cache miss today is one
syscall plus a per-read iovec setup. With `io_uring` + fixed buffers,
a token that misses on K experts becomes **one syscall** (`io_uring_enter`)
referencing K pre-pinned buffer indices — the kernel never has to walk
the user mapping or pin pages on the hot path. Published microbenchmarks
report 30–50 % less per-read CPU on NVMe-class SSDs. CPU time during
I/O wait is pure overhead — the same bytes were going to leave the
device either way; `io_uring` just makes the kernel cheaper, which is
energy out of the budget. Build with `cargo build --release --features
io_uring` (Linux only) to enable; the engine selects the `pread(2)`
backend by default so the portable path stays the warning-free default.

### 5. Frequency-based expert pinning (`--pin-after-observations N`)

`ExpertCache` now holds a `pinned: HashSet<u32>`. Once the engine has
observed an expert as a routing destination N times, it calls
`cache.pin(id)` and the LRU eviction path skips that id permanently.
`evict_lru` and `insert` both walk past pinned ids; the cache returns
`None` from `evict_lru` if every entry is pinned (caught by the engine's
existing "wait for a free buffer" loop, so progress is preserved).

**How this saves energy.** MoE workloads have heavy-tailed expert
usage — a small subset of experts handles a large fraction of tokens.
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
each consume one cache slot and one SSD read on first activation —
even though their weight bytes are nearly the same. With the map,
both expert ids resolve to a *single* canonical id; the cache holds
one resident copy, the SSD reads it once, and **every redirect counted
in `EngineReport.alias_redirects` is a cache lookup that didn't burn
SSD bytes**. The detection runs offline (no runtime cost), and the
runtime overhead is one `HashMap` lookup per routed expert per token.
Empty / absent maps disable the feature entirely.

### How to combine them

The six knobs are independent and compose freely. A reasonable
"low-energy" preset on a Linux NVMe box looks like:

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

`print_summary` reports each knob's state and effect (`pinned`,
`alias_redirects`, `dtype`, `partial_load_fraction`) on every run, so
you can verify the energy-saving paths actually engaged.

---

## Limitations / next steps

- **Static top-K router (in the legacy `run` and `serve` paths).** Real
  Mixtral routing is a learned `softmax` over the gating projection
  conditioned on the per-token hidden state. The Markov-chain router
  used by `Engine::generate` captures the *temporal locality* of real
  routing traces (and can be fed one directly via `--router-matrix`),
  but doesn't condition on `x`. The production code path lives in
  `gating::LinearGate` (`x @ W_gate.T → softmax → top-K`) — wiring it
  into the engine's per-token loop is the next step.
- **Scalar `f32` matmul.** The expert FFN runs as a plain triple-nested
  scalar loop. That is intentional — the project's thesis is about
  storage bandwidth — but a real serving deployment would call into BLAS
  / a SIMD kernel / a CUDA kernel via `tch` / `candle` / `cudarc`. The
  byte→`f32` view in `inference::ExpertWeights::from_bytes` already does
  zero-copy reinterpretation, so any of those backends slot in cleanly.
- **Single-layer wiring in the live engine.** `MultiLayerExpertCache`
  (per-layer LRU keyed on `(layer, expert_id)`) and the dense
  transformer pieces (`RmsNorm`, RoPE, scalar causal MHA with KV
  cache, MoE output combiner) now live in-tree under `multi_layer_cache`,
  `transformer`, and `gating`, with unit tests; the `serve` path drives
  one layer through them. Stacking 32 layers and threading the hidden
  state across them is the follow-up.
- **No streaming responses.** The HTTP server accepts `stream: true` for
  OpenAI compatibility but returns a non-streaming response; SSE
  streaming + continuous batching is on the roadmap.
- **Synchronous `pread` on a blocking-donated worker.** The current
  backend uses `tokio::task::block_in_place` + `pread(2)`. A production
  deployment should switch to the raw `io-uring` crate with **registered
  fixed buffers** and **registered files** — the cleanest single
  throughput win available.
- **No NUMA pinning.** On multi-socket boxes you'd want one ring per NUMA
  node and to pin worker threads + buffers locally.
- **No batched / vectored reads.** When two experts on the same token both
  miss, we issue two independent `pread` calls; on a NVMe drive with deep
  queues that's already efficient, but on slower devices you might want
  `readv`-style batching (or io_uring submission batching).

## License

Licensed under either of MIT or Apache-2.0 at your option.

