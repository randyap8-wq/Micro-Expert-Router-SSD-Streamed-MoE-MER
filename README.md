# Micro-Expert-Router — SSD-Streamed MoE Execution Engine

A Rust execution engine for **Mixture-of-Experts** models that keeps the
router resident in RAM and **hot-swaps individual experts on demand** from a
PCIe-attached NVMe drive into a pool of pre-allocated, page-aligned RAM
buffers using **io_uring** with **`O_DIRECT`** (zero-copy, kernel-page-cache
bypass).

It is the I/O substrate you'd pair with a Mixtral-style model whose total
parameter footprint exceeds available DRAM but whose *active* parameter
footprint per token (top-K experts) does not. Instead of mmap-thrashing the
page cache, this engine treats the NVMe like a manual paging device with an
LRU cache and a learned prefetcher.

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
   its own file, read it through io_uring with `O_DIRECT` so the bytes go
   directly from the NVMe DMA engine into a page-aligned RAM buffer, and run
   a custom LRU + speculative prefetcher driven by the router's own
   activation history.

### End-to-end pipeline

```
        +-----------+     +-------------+     +-----------+     +-----------+
token → |  Router   | →  | Expert IDs   | →  | LRU Cache | →  | Inference |
        |  (top-K)  |    |  e.g. [3,7]  |    +-----+-----+     +-----------+
        +-----------+    +--------------+         | miss
                                                  ↓
                                         +------------------+
                                         | BufferPool slot  | ←───┐
                                         |  (aligned, pre-  |     │
                                         |   allocated)     |     │
                                         +--------+---------+     │
                                                  ↓               │
                                         +------------------+     │ on Arc drop
                                         |  io_uring read   |     │ (LRU evict
                                         |  O_DIRECT, no    |     │  or buffer
                                         |  page cache      |     │  release)
                                         +--------+---------+     │
                                                  ↓               │
                                         NVMe SSD → DMA → RAM ────┘
```

After every token the engine also updates a first-order **Markov model** of
expert transitions and uses it to **speculatively prefetch** the most likely
next experts on the side. The prefetch path is non-blocking and
non-evicting — it never starves a real cache miss.

---

## Architecture

The Rust crate (`rust-engine/`) is organised into single-responsibility modules:

| Module | Responsibility |
|---|---|
| `aligned_buffer` | Heap-allocated, page-aligned buffer (`std::alloc::alloc` with a `Layout`). The defining requirement of `O_DIRECT`: kernel rejects unaligned buffers with `EINVAL`. |
| `buffer_pool` | Fixed-capacity slab of `AlignedBuffer`s. Hands out `PooledBuffer` RAII guards; dropping a guard returns the buffer to the free list and notifies waiters. This is the literal "pre-allocated RAM buffer" the spec asks for. |
| `expert_cache` | LRU map `expert_id → Arc<ExpertResident>`. Eviction returns the `Arc`; once all references drop, the buffer goes back to the pool automatically. |
| `io_provider` | NVMe storage layer. Opens each expert as its own file (`O_DIRECT` on Linux), keeps fds resident, submits async reads through `rio`. Includes a `gen-data` helper to create synthetic test files and a portable Unix fallback for development on macOS. |
| `router` | `TopKRouter` (distinct top-K, weighted) and `PredictiveLoader` (first-order Markov over expert ids, learns online, smoothed with a uniform prior so predictions are immediately usable). |
| `inference` | Placeholder "compute" — a strided FNV-1a hash that touches every page so the I/O is actually observable. Replace with `tch`/`candle`/`cudarc` for real weights. |
| `engine` | Top-level orchestrator. Owns the router/predictor/cache/pool/storage, drives the per-token cycle, schedules prefetches, records HDR histograms. |
| `main` | `clap`-based CLI with `gen-data` and `run` subcommands, structured `tracing` logs, `--first-token 3,7` to reproduce the spec example. |

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
- **Online Markov predictor with prior.** A flat `[N][N]` count matrix
  initialised to `1` (uniform prior). On every token transition we increment
  `counts[from][to]`. `predict_next` divides by the row total and filters by
  `min_prob`, so cold-start is graceful and learning is incremental.
- **Pluggable I/O backend.** The hot path uses `rio` (small io_uring
  wrapper). On non-Linux Unix (e.g. macOS dev boxes) it falls back to
  `pread(2)` so the engine still runs end-to-end during development, with
  the same logical pipeline.

### "Why `rio` and not X?"

| Crate | Verdict for this workload |
|---|---|
| **`rio`** *(used here)* | Small, ergonomic, easy to drop into Tokio. **Unmaintained since 2021.** No registered fixed buffers, no SQPOLL toggle, no submission-queue size knob. Fine for a reference implementation. |
| **`tokio-uring`** | Best ergonomic fit if you live in Tokio. Single-threaded per ring, requires `#[tokio_uring::start]` instead of `#[tokio::main]` — would force a runtime restructuring. |
| **`io-uring`** (raw, by tokio-rs) | The thinnest binding to the kernel ABI. Lets you use **registered (fixed) buffers** + **registered files**, which removes per-op address validation in the kernel — the single biggest win for sustained NVMe throughput. **This is what a production build of this engine would use.** |
| **`glommio`** | Thread-per-core, polled io_uring. Made for NVMe-bound workloads (ScyllaDB heritage). For a pure expert-fetch service pinning workers to cores feeding local rings, glommio is arguably the *fastest* answer on Linux. Trade-off: incompatible with Tokio (it owns the runtime). |

The clean separation between `io_provider` and the rest of the engine means
swapping `rio` for `io-uring` or `glommio` is a self-contained change.

---

## Building and running

### Prerequisites

- **Linux kernel ≥ 5.6** for io_uring support (5.10+ recommended).
- **Rust 1.74+** (uses `clap 4`, edition 2021).
- A real **block-device-backed filesystem** (ext4, xfs, btrfs on NVMe) for
  the `O_DIRECT` path. tmpfs / overlayfs / many FUSE mounts return `EINVAL`
  on `open(O_DIRECT)`. Use `--no-direct` if you need to run on those for
  development.

### Build

```bash
cd rust-engine
cargo build --release
```

### Generate synthetic expert files

```bash
# 64 experts × 16 MiB each = 1 GiB of test data on disk.
./target/release/micro-expert-router gen-data \
  --data-dir ./data \
  --num-experts 64 \
  --expert-size $((16 * 1024 * 1024))
```

Each file is filled with a deterministic per-expert byte pattern so reads
can be verified end-to-end.

### Run the simulation

```bash
# 200-token stream, top-2 routing, 16 experts resident, with O_DIRECT.
./target/release/micro-expert-router run \
  --data-dir ./data \
  --num-experts 64 \
  --expert-size $((16 * 1024 * 1024)) \
  --cache-slots 16 \
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
INFO starting engine num_experts=64 top_k=2 cache_slots=16 expert_mib=16 direct_io=true
INFO buffer pool sized with prefetch headroom cache_slots=16 pool_slots=18 prefetch_headroom=2
INFO streaming tokens (latency / throughput logs follow) tokens=200
INFO tick token=0 cycle_us=812 tps="1231.0" hits=0 misses=2 kib=32768 resident=[15, 5]
INFO tick token=1 cycle_us=634 tps="1577.7" hits=1 misses=1 kib=16384 resident=[4, 0, 15, 5]
...
INFO tick token=199 cycle_us=178 tps="5597.8" hits=2 misses=0 kib=0    resident=[13, 4, 1, 9]
INFO stream complete wall_s=0.034 sustained_tps=5732 avg_throughput_mibps=11866 hit_rate_pct=34.5
INFO ===================== run summary =====================
INFO experts:       64 (top-2), cache=16 slots, pool=18 slots
INFO lookups:       hits=138  misses=262  hit_rate=34.50%
INFO prefetches:    completed=152  predictor_observations=796
INFO i/o:           reads=262  bytes=4192.00 MiB
INFO i/o latency:   p50=684us  p95=925us  p99=1142us
INFO cycle latency: p50=1083us p95=1718us p99=1966us  max=1966us
```

### CLI reference

```
micro-expert-router gen-data
  --data-dir <PATH>          Output directory (default ./data)
  --num-experts <N>          Number of expert files (default 64)
  --expert-size <BYTES>      Bytes per file, multiple of 4096 (default 16 MiB)

micro-expert-router run
  --data-dir <PATH>          Directory with expert_<id>.bin files
  --num-experts <N>          Total experts in the model
  --expert-size <BYTES>      Must match gen-data
  --cache-slots <N>          Resident experts (LRU capacity)
  --top-k <K>                Active experts per token (default 2, distinct)
  --tokens <N>               Stream length
  --predict-fanout <N>       Prefetch candidates per token (default 2)
  --predict-min-prob <P>     Skip prefetch below this probability (default 0.05)
  --no-direct                Disable O_DIRECT (use page cache; CI / tmpfs)
  --block-align <BYTES>      O_DIRECT alignment, default 4096
  --first-token <IDS>        Comma-separated expert ids to warm into cache
  --no-prefetch              Disable predictive loader (for ablation)
  --token-pause-us <N>       Sleep between tokens to throttle the stream
  --seed <U64>               PRNG seed for reproducibility
```

---

## What can it actually run today?

**Today, in this repository: nothing end-to-end.** `inference::run_inference`
is a placeholder FNV-1a hash over the buffer — there is no matmul, no
softmax, no tokenizer. The engine demonstrates the **I/O substrate** that a
real MoE runtime would sit on top of, not the runtime itself. Wiring a
tensor library into `run_inference` is the missing step.

That said, the architecture (per-expert files, fixed expert size,
top-K activation, LRU + prefetch) is shaped specifically for **sparse
Mixture-of-Experts transformers where the expert FFNs are the dominant
weight**. Concretely, the following published models drop into this layout
with no architectural changes — only a real inference kernel and a sharding
script that splits their `safetensors` into one `expert_<id>.bin` per
expert (or per-layer-per-expert, see "Sharding granularity" below):

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
- top-K router produces distinct ids,
- predictor learns simple transitions and respects `min_prob`.

---

## Limitations / next steps

- **Static top-K router.** Real Mixtral routing is a learned `softmax` over
  the gating projection. The mocked router is sufficient to drive the I/O
  pipeline; integrating a real one is mostly plumbing.
- **`rio` is unmaintained.** A production deployment should switch to the
  raw `io-uring` crate with **registered fixed buffers** and **registered
  files** — the cleanest single throughput win available.
- **No NUMA pinning.** On multi-socket boxes you'd want one ring per NUMA
  node and to pin worker threads + buffers locally.
- **No batched / vectored reads.** When two experts on the same token both
  miss, we issue two SQEs; on a NVMe drive with deep queues that's already
  efficient, but on slower devices you might want `readv`-style batching.
- **Inference is a placeholder.** Wire `tch::nn::Module::forward` /
  `candle::Tensor::matmul` / a CUDA kernel into `inference::run_inference`
  to make this a real model server.

## License

Licensed under either of MIT or Apache-2.0 at your option.

