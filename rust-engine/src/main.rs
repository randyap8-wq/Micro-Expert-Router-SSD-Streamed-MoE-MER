//! Micro-Expert-Router — MoE execution engine that hot-swaps experts from a
//! PCIe-attached NVMe drive into pre-allocated, page-aligned RAM via
//! `O_DIRECT` `pread(2)` (dispatched off the Tokio runtime with
//! `block_in_place`).
//!
//! See README.md at the repository root for architecture and design notes.

// Gist Task 3 — "Nightly AMX feature gating". See the matching
// comment in `lib.rs`. Off by default; opt in with
// `--features nightly-amx` and a nightly toolchain to unlock the
// real Intel AMX tile intrinsic surface. When this feature is not
// enabled, the AMX dispatch path falls back to AVX-512.
#![cfg_attr(feature = "nightly-amx", feature(stdarch_x86_amx))]

#[cfg(feature = "alloc-count")]
mod alloc_count {
    use super::AllocationSnapshot;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub struct CountingAllocator;

    static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
    static DEALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
    static REALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
    static BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
    static BYTES_DEALLOCATED: AtomicU64 = AtomicU64::new(0);
    static CURRENT_BYTES: AtomicU64 = AtomicU64::new(0);
    static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: this allocator only observes the operation, then forwards
            // the exact layout to the platform allocator.
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                record_allocation(layout.size() as u64);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            DEALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            record_deallocation(layout.size() as u64);
            // SAFETY: `ptr`/`layout` are the exact pair passed to this global
            // allocator by the standard library, so forwarding them is valid.
            unsafe { System.dealloc(ptr, layout) };
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: forwards the original pointer/layout and requested size
            // unchanged to the platform allocator.
            let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                REALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                let old_size = layout.size() as u64;
                let new_size = new_size as u64;
                if new_size >= old_size {
                    record_allocation(new_size - old_size);
                } else {
                    record_deallocation(old_size - new_size);
                }
            }
            new_ptr
        }
    }

    pub fn reset() {
        ALLOCATION_CALLS.store(0, Ordering::Relaxed);
        DEALLOCATION_CALLS.store(0, Ordering::Relaxed);
        REALLOCATION_CALLS.store(0, Ordering::Relaxed);
        BYTES_ALLOCATED.store(0, Ordering::Relaxed);
        BYTES_DEALLOCATED.store(0, Ordering::Relaxed);
        CURRENT_BYTES.store(0, Ordering::Relaxed);
        PEAK_BYTES.store(0, Ordering::Relaxed);
    }

    pub fn snapshot() -> AllocationSnapshot {
        AllocationSnapshot {
            allocation_calls: ALLOCATION_CALLS.load(Ordering::Relaxed),
            deallocation_calls: DEALLOCATION_CALLS.load(Ordering::Relaxed),
            reallocation_calls: REALLOCATION_CALLS.load(Ordering::Relaxed),
            bytes_allocated: BYTES_ALLOCATED.load(Ordering::Relaxed),
            bytes_deallocated: BYTES_DEALLOCATED.load(Ordering::Relaxed),
            current_bytes: CURRENT_BYTES.load(Ordering::Relaxed),
            peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
        }
    }

    fn record_allocation(bytes: u64) {
        BYTES_ALLOCATED.fetch_add(bytes, Ordering::Relaxed);
        let current = CURRENT_BYTES
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes);
        update_peak(current);
    }

    fn record_deallocation(bytes: u64) {
        BYTES_DEALLOCATED.fetch_add(bytes, Ordering::Relaxed);
        let _ = CURRENT_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(bytes))
        });
    }

    fn update_peak(current: u64) {
        let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
        while current > peak {
            match PEAK_BYTES.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }
}

#[cfg(feature = "alloc-count")]
#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_count::CountingAllocator = alloc_count::CountingAllocator;

mod aligned_buffer;
mod architecture;
mod backend;
mod batch_scheduler;
mod block_pool;
mod buffer_pool;
mod config;
mod dequant;
mod dense_tensor;
mod distributed;
mod draft;
mod engine;
mod expert_cache;
mod gating;
mod gguf;
mod gguf_loader;
#[cfg(feature = "grpc")]
mod grpc;
#[cfg(feature = "grpc")]
mod grpc_gen;
mod inference;
mod io_provider;
mod io_reactor;
#[cfg(all(feature = "io_uring", target_os = "linux"))]
mod io_uring_storage;
mod kernels;
mod metrics;
mod middleware;
mod mla;
mod model;
mod multi_layer_cache;
mod numa;
mod packed_storage;
mod parallel;
mod prefetch_governor;
mod pregate;
mod residency;
mod router;
mod rpc;
mod sampling;
mod server;
mod session;
mod stage_timing;
mod tensor_header;
mod tokenizer;
mod transformer;
#[cfg(feature = "tui")]
mod tui;
mod workload;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::backend::Backend;
use crate::buffer_pool::BufferPool;
use crate::engine::{Engine, EngineOptions, ModelShape};
use crate::inference::expert_weight_bytes_for;
use crate::io_provider::{NvmeStorage, StorageConfig};
use crate::multi_layer_cache::MultiLayerExpertCache;
use crate::router::{
    LayeredExpertAffinity, LocalityMonitor, NeuralSpeculator, PredictiveLoader, TopKRouter,
};

const SUPPORTED_SYNTHETIC_DTYPES: &str = "f32, f16, bf16, int8, q4k, q4_0, q8_0, mxfp4";
const SUPPORTED_RUNTIME_DTYPES: &str =
    "f32, f16, bf16, int8, q4k, q4_0, q5k, q6k, q8_0, mxfp4, mixed";

/// MoE execution engine that streams experts from NVMe via O_DIRECT pread(2).
#[derive(Parser, Debug)]
#[command(name = "micro-expert-router", version, about)]
struct Cli {
    /// Logging filter (e.g. `info`, `debug`, `micro_expert_router=debug`).
    #[arg(long, global = true, default_value = "info", env = "RUST_LOG")]
    log: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum BenchRealCacheReset {
    /// Keep the same engine/cache across warmup and measured runs.
    Keep,
    /// Rebuild the runtime before every run, giving each run a cold cache.
    FreshRuntime,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum BenchRealOutputFormat {
    Human,
    Json,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate synthetic expert files for testing on local disk.
    GenData {
        /// Directory to write `expert_<id>.bin` files into.
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Number of experts to create.
        #[arg(long, default_value_t = 64)]
        num_experts: u32,
        /// Bytes per expert. Must be a multiple of 4096 for O_DIRECT and
        /// at least `3 * d_model * d_ff * 4` bytes (the SwiGLU weights);
        /// any extra bytes are zero-padded.
        ///
        /// Default 16 MiB pairs cleanly with `d_model=512 d_ff=2048`
        /// (12 MiB of weights + 4 MiB of padding).
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        expert_size: usize,
        /// Hidden / residual-stream dimension of the FFN (Mixtral: 4096,
        /// DeepSeek-V3: 7168). Default 512 keeps the synthetic compute
        /// cheap so I/O remains observable.
        #[arg(long, default_value_t = 512)]
        d_model: usize,
        /// Intermediate FFN dimension (Mixtral: 14336, Llama-3-MoE: 14336).
        /// Default 2048.
        #[arg(long, default_value_t = 2048)]
        d_ff: usize,
        /// Block alignment for `O_DIRECT` (4096 on most NVMe). The
        /// generated file size (`expert_size`) must be a multiple of
        /// this so the run path can read each expert with `O_DIRECT`
        /// without `EINVAL`. Must match what `run` is invoked with.
        #[arg(long, default_value_t = 4096)]
        block_align: usize,
        /// On-disk weight dtype for synthetic files: f32, f16, bf16, int8,
        /// q4k, q4_0, q8_0, or mxfp4. q5k, q6k, and mixed are GGUF/runtime
        /// formats and are not synthesized by this generator.
        #[arg(long, default_value = "f32")]
        dtype: String,
    },

    /// **Tier 2.** Repack a directory of `expert_<id>.bin` files into a
    /// single packed blob + JSON manifest for the packed storage layout.
    /// Experts are written back-to-back (one block-aligned `expert_size`
    /// slot each) in an order chosen by `--profile` / `--order` so the
    /// engine can coalesce co-fetched experts into single `preadv`s.
    Repack {
        /// Source directory containing `expert_<id>.bin` (or
        /// `expert_<layer>_<local>.bin` with `--num-experts-per-layer`).
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Output blob path (all expert payloads concatenated).
        #[arg(long)]
        out_blob: PathBuf,
        /// Output manifest path. Defaults to `<out_blob>.manifest.json`.
        #[arg(long)]
        out_manifest: Option<PathBuf>,
        /// Number of experts to pack (ids `0..num_experts`, unless
        /// `--order` restricts the set).
        #[arg(long, default_value_t = 64)]
        num_experts: u32,
        /// Bytes per expert. Must equal the source files' `expert_size`.
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        expert_size: usize,
        /// Block alignment (must match `gen-data` / the source files).
        #[arg(long, default_value_t = 4096)]
        block_align: usize,
        /// Disable `O_DIRECT` when reading the source files (needed on
        /// tmpfs / macOS / CI).
        #[arg(long)]
        no_direct: bool,
        /// Experts per layer for layer-qualified source naming.
        #[arg(long)]
        num_experts_per_layer: Option<u32>,
        /// Order experts hottest-first using a routing-frequency profile
        /// JSON (as produced by `run --profile-out`). Unobserved experts
        /// are appended in numeric order. Ignored if `--order` is set.
        #[arg(long)]
        profile: Option<PathBuf>,
        /// Explicit physical layout: a file listing expert ids (one per
        /// line, or a JSON array). Overrides `--profile`. Only the listed
        /// experts are packed.
        #[arg(long)]
        order: Option<PathBuf>,
    },

    /// Run the token-generation simulation against the on-disk experts.
    Run {
        /// Directory with `expert_<id>.bin` files. May also contain a
        /// `metadata.json` written by `scripts/extract_mixtral_experts.py`,
        /// in which case `num_experts`, `d_model`, `d_ff`, `top_k`, and
        /// `expert_size` are auto-loaded (CLI flags still override).
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Total number of experts in the model.
        #[arg(long, default_value_t = 64)]
        num_experts: u32,
        /// Bytes per expert. Must equal what was used in `gen-data`.
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        expert_size: usize,
        /// Hidden / residual-stream dimension. Must match `gen-data`.
        #[arg(long, default_value_t = 512)]
        d_model: usize,
        /// Intermediate FFN dimension. Must match `gen-data`.
        #[arg(long, default_value_t = 2048)]
        d_ff: usize,
        /// LRU cache + buffer pool capacity (resident experts at once).
        ///
        /// The whole point of this engine is that experts stream from
        /// SSD; making the cache big defeats that. The default is 4
        /// slots and the engine warns if more than 16 are requested.
        #[arg(long, default_value_t = 4)]
        cache_slots: usize,
        /// Top-K experts the router activates per token.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        /// Number of tokens to simulate.
        #[arg(long, default_value_t = 200)]
        tokens: u64,
        /// Predictive prefetch fanout (how many candidates to issue per token).
        #[arg(long, default_value_t = 2)]
        predict_fanout: usize,
        /// **Look-ahead pipeline depth.** In serve mode, controls how many MoE
        /// layers ahead the speculator prefetches (the sliding window
        /// `layer + 1 ..= layer + pipeline_depth`), hiding cold SSD reads behind
        /// compute. `1` reproduces the legacy single-layer look-ahead.
        ///
        /// In `run`, this currently only scales speculative prefetch headroom
        /// (shadow buffer budget = `predict_fanout * pipeline_depth`).
        #[arg(long, default_value_t = crate::engine::DEFAULT_PIPELINE_DEPTH)]
        pipeline_depth: u32,
        /// Don't prefetch below this transition probability. The default
        /// (`0.0`) auto-scales the threshold to `2 / num_experts` so
        /// it remains achievable as the expert pool grows; pass an
        /// explicit positive value to override (e.g. `--predict-min-prob 0.05`).
        #[arg(long, default_value_t = 0.0)]
        predict_min_prob: f64,
        /// Disable O_DIRECT (use buffered reads). Required on tmpfs/overlay/CI
        /// and on macOS, where O_DIRECT is not supported. When set, the run
        /// summary's I/O latency includes any page-cache effects — see the
        /// startup warning emitted in this case.
        #[arg(long)]
        no_direct: bool,
        /// Block alignment for O_DIRECT (4096 on most NVMe).
        #[arg(long, default_value_t = 4096)]
        block_align: usize,
        /// PRNG seed for reproducible runs.
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        /// On-disk weight dtype. Accepts f32, f16, bf16, int8, q4k, q4_0,
        /// q5k, q6k, q8_0, mxfp4, or mixed. Must match the generated or
        /// converted expert dataset.
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Fraction (`0.1..=1.0`) of input dimensions loaded per expert
        /// when partial column loading is enabled. `1.0` (default)
        /// disables partial loading. The forward pass still produces
        /// finite, correct-shape outputs for any value in range; lower
        /// fractions trade a small amount of accuracy for proportionally
        /// less compute / dequant energy.
        #[arg(long, default_value_t = 1.0)]
        partial_load_fraction: f64,
        /// After an expert has been observed in routing this many times,
        /// pin it permanently in the LRU cache. `0` (default) disables
        /// frequency-based pinning. Pinned experts are never reloaded
        /// from SSD, eliminating their I/O energy.
        #[arg(long, default_value_t = 0)]
        pin_after_observations: u64,
        /// Optional alias map JSON: `{ "src_id": canonical_id, ... }`.
        /// Pairs of experts the offline analyser flagged as numerically
        /// near-identical share a single resident copy at runtime,
        /// eliminating duplicate SSD reads.
        #[arg(long)]
        alias_map: Option<PathBuf>,
        /// Use the Linux `io_uring` storage backend with registered
        /// fixed buffers (one syscall to enqueue many reads, kernel
        /// reads directly into pre-pinned pool buffers). Requires the
        /// `io_uring` cargo feature; without it this flag logs a
        /// warning and the engine falls back to the default `pread(2)`
        /// path.
        #[arg(long)]
        io_uring: bool,
        /// Sleep this many micros between tokens (0 = as fast as possible).
        #[arg(long, default_value_t = 0)]
        token_pause_us: u64,
        /// Force-route the first token to these expert ids (comma-separated).
        /// Spec example: `--first-token 3,7`.
        #[arg(long, value_delimiter = ',')]
        first_token: Vec<u32>,
        /// Disable predictive prefetching entirely (for ablation).
        #[arg(long)]
        no_prefetch: bool,
        /// **I/O-only benchmarking mode**: skip the SwiGLU FFN forward
        /// pass entirely; still read every expert from SSD and XOR every
        /// byte to force the read to fully materialise. Use this to
        /// isolate the SSD-streaming cost from FFN compute.
        #[arg(long)]
        io_only: bool,
        /// **Force SSD reads.** Refuse to run with optimisations that
        /// would let the OS serve experts from RAM (page cache) instead
        /// of the device. Concretely: requires `O_DIRECT` (i.e. the run
        /// fails if `--no-direct` is also set on Linux). On macOS, where
        /// O_DIRECT is unavailable, this flag prints a warning and runs
        /// in best-effort mode.
        #[arg(long)]
        force_ssd: bool,
        /// Number of cluster groups for the router's first-order Markov
        /// chain (default 4: matches the gist's example). Each cluster
        /// is a group of experts that the router prefers to keep
        /// activating consecutively.
        #[arg(long, default_value_t = 4)]
        router_clusters: usize,
        /// Probability the Markov router stays inside its current
        /// expert cluster on each step. Higher = stronger temporal
        /// locality = more prefetch signal.
        #[arg(long, default_value_t = 0.9)]
        router_intra_p: f64,
        /// Optional path to a precomputed transition matrix. Whitespace-
        /// separated `f64` values, row-major, `num_experts^2` entries.
        /// Overrides `--router-clusters` / `--router-intra-p` when set.
        #[arg(long)]
        router_matrix: Option<PathBuf>,
        /// Optional path to a real **gating-network** weight matrix
        /// (`f32` little-endian, row-major, shape `[num_experts × d_model]`).
        ///
        /// When set, the run loop bypasses the deterministic Markov
        /// `TopKRouter` and instead computes per-token routing the way
        /// production Mixtral does: `softmax(W_gate · x) → top-K`.
        /// Each routed expert is still streamed from the SSD via the
        /// LRU cache (`Engine::moe_step`), so the SSD-bandwidth /
        /// cache-hit metrics reported at the end are directly
        /// comparable to the legacy Markov path.
        ///
        /// File format: bare little-endian `f32`s, no header. Generate
        /// one with `numpy.tofile` from a real Mixtral checkpoint, or
        /// use the seeded synthetic fallback if you only want to
        /// exercise the path (omit this flag to keep the legacy
        /// Markov router).
        ///
        /// May also point at a **directory** of per-layer `gate_<L>.bin`
        /// files (the same naming the model loader uses): they are
        /// auto-discovered, sorted by layer index, and concatenated in
        /// order, so you don't have to `cat` them into one file first.
        #[arg(long)]
        gate_weights: Option<PathBuf>,
        /// Optional path to write a JSONL **routing trace** to. Each
        /// line records one token's `{token, layer, experts,
        /// cache_hit}`, suitable for offline analysis with
        /// `scripts/compute_transition_matrix.py` and the
        /// `validate-predictor` subcommand.
        #[arg(long)]
        trace_out: Option<PathBuf>,
        /// Initialise the GPU compute backend before running the
        /// benchmark so the FFN forward pass uses GPU matmul where
        /// available. The run path also installs a bounded VRAM
        /// `GpuExpertCache` so hot experts can promote and be served
        /// from device memory. Falls back to the default CPU backend
        /// with a warning if GPU init fails.
        #[arg(long)]
        gpu: bool,
        /// VRAM budget, in MiB, for the run-mode GPU expert cache
        /// (only with `--gpu`). Hot experts promote into this cache and
        /// are served from device memory. The 4 GiB default fits ~40
        /// Mixtral-8x7B Q4 experts (~99 MiB each — 512 MiB would hold
        /// barely 5); lower it on cards with less free VRAM.
        #[arg(long, default_value_t = 4096)]
        gpu_cache_mb: usize,
        /// Enable the **neural speculator** (arm `M`): a 2-layer MLP
        /// trained online against the gate's actual top-K. Predicts
        /// from the residual stream — the same feature the gate sees —
        /// so it is the strongest single prefetch signal and the one
        /// that actually drives `speculate_layer_ahead`. Off by default
        /// so the legacy Markov-only path is unchanged; turn it on to
        /// measure whether the predictive arms move the hit rate.
        #[arg(long)]
        speculator: bool,
        /// Hidden width of the speculator MLP (only when `--speculator`).
        #[arg(long, default_value_t = 128)]
        speculator_hidden_dim: usize,
        /// Top-K experts pulled from the speculator each step. `0`
        /// inherits `--top-k`.
        #[arg(long, default_value_t = 0)]
        speculator_top_k: usize,
        /// Enable the **locality monitor** (arm `L`): a sliding window
        /// over recent activations whose hot set is unioned into the
        /// prefetch set *and* pinned in the LRU so genuinely hot experts
        /// stop being evicted by cold ones — a frequency-aware upgrade
        /// over plain LRU eviction.
        #[arg(long)]
        locality: bool,
        /// Locality sliding-window size, in routing observations.
        #[arg(long, default_value_t = 256)]
        locality_window: usize,
        /// Heat threshold: an expert is "hot" once it appears in this
        /// fraction of the locality window. `0.10` ≈ 10% of recent
        /// activations.
        #[arg(long, default_value_t = 0.10)]
        locality_threshold_pct: f32,
        /// Enable the **per-layer expert-affinity** arm: folds co-fired
        /// and disk-adjacent neighbours of high-confidence predictions
        /// into the prefetch union. Only effective on multi-layer runs
        /// (`--num-experts-per-layer` set).
        #[arg(long)]
        affinity: bool,
        /// Number of co-fired neighbours pulled per seed (with `--affinity`).
        #[arg(long, default_value_t = 4)]
        affinity_neighbors_k: usize,
        /// Exponential-decay epoch for the affinity counters, in
        /// cumulative observations (with `--affinity`).
        #[arg(long, default_value_t = 10_000)]
        affinity_decay_epoch: u64,
        /// **Tier 4 — adaptive prefetch governor.** Throttle speculative
        /// prefetches by measured precision (consumed / completed) and
        /// foreground-read contention, so low-value speculation can't
        /// inflate the latency of the foreground misses that actually
        /// block token generation. Off by default (legacy unbounded
        /// admission). This is the highest-leverage knob on a
        /// bandwidth-bound SSD.
        #[arg(long)]
        prefetch_governor: bool,
        /// Precision floor / optimistic EWMA seed for the governor, in
        /// `[0, 1]` (with `--prefetch-governor`).
        #[arg(long, default_value_t = 0.05)]
        prefetch_precision_floor: f64,
        /// Per-outstanding-foreground-read multiplier on the governor's
        /// admission threshold (with `--prefetch-governor`). Higher ⇒
        /// speculation backs off harder while real misses are in flight.
        #[arg(long, default_value_t = 1.0)]
        prefetch_contention_weight: f64,
        /// **Tier 4 — cost-aware eviction.** Evict the coldest resident
        /// by decaying heat score instead of strict LRU, so a hot expert
        /// that briefly fell to the LRU tail isn't dumped ahead of a
        /// one-shot cold expert. Off by default (pure LRU).
        #[arg(long)]
        cost_aware_eviction: bool,
        /// **Tier 3 — per-layer pre-gate predictor.** Train an online
        /// layer-L→L+1 conditional map and drive high-precision
        /// next-layer prefetch from it. Off by default.
        #[arg(long)]
        pregate: bool,
        /// **Tier 1 — static residency.** Fraction of the global expert
        /// namespace to pin permanently in RAM (the hottest experts), in
        /// `(0, 1]`. `0.0` (default) disables it. Lifts the hit-rate
        /// ceiling above the bare cache fraction on a *skewed* workload.
        #[arg(long, default_value_t = 0.0)]
        static_residency_fraction: f64,
        /// Tokens to observe before deriving the online static-residency
        /// hot set (ignored when `--static-residency-profile` is given).
        #[arg(long, default_value_t = 0)]
        static_residency_warmup_tokens: u64,
        /// Path to an offline expert-popularity profile JSON
        /// (`{ "<id>": <count> }`) to seed static residency at startup.
        /// When omitted, the hot set is derived online.
        #[arg(long)]
        static_residency_profile: Option<String>,
        /// Write the run's accumulated route-observation profile to this
        /// JSON path at shutdown (consumable by
        /// `--static-residency-profile` on a later run).
        #[arg(long)]
        profile_out: Option<String>,
        /// **Benchmark workload.** `synthetic` (default) keeps the legacy
        /// uniform-i.i.d. stream (the engine/gate routes its own hidden
        /// state); `skewed` drives `moe_step` from a Zipf-popular,
        /// Markov-correlated expert generator (so static residency and
        /// the predictors are exercisable); `replay` replays a recorded
        /// JSONL routing trace via `--replay-trace`.
        #[arg(long, default_value = "synthetic")]
        workload: String,
        /// Zipf exponent for `--workload skewed` (larger ⇒ more skew;
        /// `1.0` ≈ classic Zipf, `0.0` ≈ uniform).
        #[arg(long, default_value_t = 1.1)]
        zipf_s: f64,
        /// Markov stay-probability for `--workload skewed`, in `[0, 1]`:
        /// the chance a token reuses the previous token's expert set
        /// (temporal correlation the predictors can exploit).
        #[arg(long, default_value_t = 0.0)]
        workload_correlation: f64,
        /// JSONL routing trace to replay with `--workload replay` (the
        /// `--trace-out` format).
        #[arg(long)]
        replay_trace: Option<String>,
        /// Number of transformer layers, used to size the affinity
        /// matrix. `1` (default) is the single-namespace synthetic
        /// benchmark.
        #[arg(long, default_value_t = 1)]
        num_layers: u32,
        /// Experts **per layer** for a layer-qualified id geometry. When
        /// set, `speculate_layer_ahead` restricts the speculator's
        /// global head to the next layer's slice and actually prefetches
        /// `layer + 1 ..= layer + pipeline_depth` ahead — the mechanism
        /// that hides SSD latency behind compute. Leave unset for the
        /// flat single-namespace benchmark (no layer-ahead).
        #[arg(long)]
        num_experts_per_layer: Option<u32>,
        /// **Tier 2 — packed storage.** Read every expert from this single
        /// packed blob (produced by the `repack` subcommand) instead of
        /// one file per expert. Requires `--packed-manifest`. Adjacent
        /// experts are fetched with coalesced `preadv` syscalls.
        #[arg(long)]
        packed_blob: Option<PathBuf>,
        /// **Tier 2.** JSON manifest (`id -> offset,len`) accompanying
        /// `--packed-blob`. Required when `--packed-blob` is set.
        #[arg(long)]
        packed_manifest: Option<PathBuf>,
    },

    /// Convert a GGUF checkpoint (Mixtral-style) into the engine's
    /// per-expert binary format plus a `metadata.json` and the dense
    /// weight files [`RealModel::from_dir`] consumes. Phase 2.
    GgufConvert {
        /// Path to a normal `*.gguf` file or any file in a standard
        /// `*-00001-of-00005.gguf` shard set.
        #[arg(long)]
        gguf_path: PathBuf,
        /// Output directory. Created if it doesn't exist.
        #[arg(long)]
        out_dir: PathBuf,
        /// Override the number of layers (defaults to
        /// `llama.block_count` from the GGUF metadata).
        #[arg(long, default_value_t = 0)]
        num_layers: usize,
        /// Override the experts-per-layer (defaults to
        /// `llama.expert_count` from the GGUF metadata).
        #[arg(long, default_value_t = 0)]
        num_experts: usize,
        /// Skip the Unified Tensor Header (U.T.H.) prefix on every
        /// `expert_<id>.bin`. By default the converter emits a 64-byte
        /// page-padded UTH so the loader knows the dtype + shape +
        /// tile-hint before reading any weight bytes; pass this flag
        /// to produce legacy bare-payload files for compatibility
        /// with consumers that pre-date UTH support.
        #[arg(long, default_value_t = false)]
        no_uth: bool,
        /// Use the legacy eager GGUF reader (slurps the entire file
        /// into RAM before slicing tensors out). The default is the
        /// streaming reader which keeps only the header + tensor
        /// info table resident — a strict win for ≥ 100 GB
        /// checkpoints. The eager path is still useful in tests and
        /// for small fixtures.
        #[arg(long, default_value_t = false)]
        legacy_eager: bool,
        /// **Native quantised pass-through.** When set and the
        /// source GGUF stores its expert tensors as `Q4_0`, `Q4_K`,
        /// `Q5_K`, `Q6_K`, or `Q8_0`, write the raw quantised block stream to disk
        /// instead of dequantising to F32 first. The output
        /// `expert_<id>.bin` stays quantized. Mixed projection triples
        /// are written with UTH2 headers. If quantized output is not
        /// possible, conversion fails before writing expert files.
        #[arg(long, default_value_t = false)]
        native_quant: bool,
        /// Convert only routed expert blobs and metadata. Dense
        /// transformer tensors are skipped deliberately; the output is
        /// valid for expert-streaming benchmarks, not full-model runs.
        #[arg(long, default_value_t = false)]
        experts_only: bool,
    },

    /// Validate a converted expert dataset before running inference.
    ValidateData {
        /// Directory containing `expert_<id>.bin` files and metadata.json.
        #[arg(long)]
        data_dir: PathBuf,
    },

    /// Replay a routing trace through the predictive prefetcher and
    /// print per-K hit-rate statistics. Phase 6.
    ValidatePredictor {
        /// Path to a JSONL routing trace (produced by `run --trace-out`).
        #[arg(long)]
        trace: PathBuf,
        /// LRU cache size to simulate. Repeat the flag to evaluate
        /// multiple sizes in one run (e.g. `--cache-slots 4
        /// --cache-slots 8 --cache-slots 16`). Defaults to a sweep of
        /// 2, 4, 8, 16.
        #[arg(long)]
        cache_slots: Vec<usize>,
    },

    /// Start the OpenAI-compatible HTTP server (Phase 6 / 8 / 9).
    ///
    /// Reads server, model, storage, and tokenizer settings from a TOML
    /// config file. The engine is built exactly as in `run`, but instead
    /// of streaming a fixed token count it stays up serving requests on
    /// `POST /v1/completions`, `POST /v1/chat/completions`, and exports
    /// Prometheus metrics on `GET /metrics`.
    Serve {
        /// Path to the TOML config file. See `config.toml` at the
        /// repository root for an example.
        #[arg(long)]
        config: PathBuf,
    },

    /// Benchmark the real transformer path without starting HTTP.
    ///
    /// Uses the same TOML config surface as `serve`, requires
    /// `[real_transformer] enabled = true`, and reports prompt/decode
    /// timing separately from the legacy synthetic `run` sustained_tps
    /// metric.
    BenchReal {
        /// Path to the TOML config file.
        #[arg(long)]
        config: PathBuf,
        /// Prompt text to encode and benchmark.
        #[arg(long, conflicts_with = "request_json")]
        prompt: Option<String>,
        /// OpenAI-style request JSON containing `prompt`, or chat
        /// `messages`, and optionally `max_tokens`.
        #[arg(long, conflicts_with = "prompt")]
        request_json: Option<PathBuf>,
        /// Number of completion tokens to generate. Overrides
        /// `max_tokens` from `--request-json` when both are supplied.
        #[arg(long)]
        output_tokens: Option<usize>,
        /// Warmup runs to execute before measurements.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,
        /// Measured runs to report.
        #[arg(long, default_value_t = 1)]
        measured_runs: usize,
        /// Cache reset policy between runs.
        #[arg(long, value_enum, default_value_t = BenchRealCacheReset::Keep)]
        cache_reset: BenchRealCacheReset,
        /// Force deterministic greedy decoding for benchmark parity.
        #[arg(long)]
        greedy: bool,
        /// Output format.
        #[arg(long, value_enum, default_value_t = BenchRealOutputFormat::Human)]
        format: BenchRealOutputFormat,
    },

    /// Benchmark dense matvec backends on the target Qwen3-Coder CPU shapes.
    ///
    /// This is intentionally separate from `bench-real`: it isolates Q/K/V/O,
    /// router gate, and LM-head projection kernels so operators can choose a
    /// production `[real_transformer].dense_matvec_backend` without running a
    /// full checkpoint. The LM-head shape allocates about 1.2 GiB of weights;
    /// use `--skip-lm-head` for quick local smoke runs.
    MatvecMicrobench {
        /// Backend(s) to benchmark. Repeat or pass comma-separated values.
        /// Defaults to matrixmultiply, rayon, and rayon-matrixmultiply.
        #[arg(long = "backend", value_delimiter = ',')]
        backend: Vec<crate::parallel::DenseMatvecBackend>,
        /// Warmup iterations per shape/backend.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,
        /// Measured iterations per shape/backend.
        #[arg(long, default_value_t = 3)]
        measured_runs: usize,
        /// Skip the 151936 × 2048 LM-head shape.
        #[arg(long)]
        skip_lm_head: bool,
        /// Emit JSON instead of human-readable rows.
        #[arg(long)]
        json: bool,
    },

    /// Count heap allocations for the old transformer wrappers vs request scratch buffers.
    ///
    /// Requires `--features alloc-count`. The synthetic layer keeps this
    /// benchmark independent of checkpoint files while exercising the same
    /// attention, RMSNorm, router, residual, and MoE-combine methods used by
    /// the real decode loop.
    ScratchAllocMicrobench {
        /// Hidden size for the synthetic transformer layer. Must be a multiple of 4.
        #[arg(long, default_value_t = 32)]
        d_model: usize,
        /// Number of routed experts in the synthetic gate.
        #[arg(long, default_value_t = 8)]
        num_experts: usize,
        /// Experts selected per token.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        /// Tokens to run before resetting allocation counters.
        #[arg(long, default_value_t = 1)]
        warmup_tokens: usize,
        /// Tokens measured after warmup. Defaults below the 16-token KV page size.
        #[arg(long, default_value_t = 8)]
        measured_tokens: usize,
        /// Emit JSON instead of human-readable rows.
        #[arg(long)]
        json: bool,
    },
    /// Native terminal dashboard — Phase 4 of the three-tier memory
    /// hierarchy. Polls a running `serve` instance and renders a live
    /// ratatui view of the SSD → RAM → VRAM hit grid, current cache
    /// state, VRAM utilisation, and I/O reactor activity. Pure
    /// observability; the dashboard does not mutate engine state.
    ///
    /// Requires the binary to be built with the `tui` cargo feature
    /// (on by default). With `--no-default-features` this subcommand
    /// exits with a helpful error message.
    Monitor {
        /// Base URL of the `serve` HTTP endpoint to poll. Defaults to
        /// `http://127.0.0.1:8080` to match the example config.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// How often to refresh the dashboard, in milliseconds.
        #[arg(long, default_value_t = 500)]
        refresh_ms: u64,
    },
}

/// Resolve the effective `predict_min_prob` for a given expert-pool size.
///
/// A configured value of `0.0` (or negative — treated identically) is the
/// "auto" sentinel and scales the threshold to `2 / num_experts`, so
/// the Laplace-smoothed posteriors in [`PredictiveLoader::predict_next`]
/// can actually clear the gate as the pool grows (a fixed `0.05` becomes
/// mathematically unreachable past ~20 experts). Any positive value is
/// passed through unchanged, preserving operator overrides.
fn resolve_predict_min_prob(configured: f64, num_experts: u32) -> f64 {
    if configured > 0.0 {
        configured
    } else {
        let n = num_experts.max(1) as f64;
        2.0 / n
    }
}

fn init_logging(filter: &str) {
    let env_filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_level(true)
        .try_init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    init_logging(&cli.log);

    // Best-effort NUMA pinning: honoured before any tokio runtime or
    // background thread spawns so child threads inherit the affinity
    // mask. See `numa::apply_mer_pin_cores_env` for the contract.
    let pin = crate::numa::apply_mer_pin_cores_env();
    let startup_pinned = matches!(pin, crate::numa::PinResult::Pinned { .. });
    info!("{}", pin.as_log_line());

    // `MER_PIN_CORES` is now consumed centrally at process start via the
    // `numa` module. Clear it so any legacy later parsing in subcommands
    // (for example, `cmd_serve`) does not attempt to re-apply affinity
    // and drift from the startup contract.
    std::env::remove_var("MER_PIN_CORES");

    // Size the shared compute (`rayon`) pool now: after affinity pinning so
    // its workers inherit the startup mask, and before any matmul touches
    // it. By default it spans the host's logical cores *minus a small
    // reservation* (`parallel::default_compute_threads`) so a saturated
    // compute fan-out can't starve the async runtime under continuous
    // batching; an explicit `RAYON_NUM_THREADS` overrides the default.
    crate::parallel::init_global_pool();

    // Log the selected math kernel backend once. The dispatcher itself
    // is lazy, but emitting this at startup gives ops a single line in
    // the journal that tells them "you're running the scalar path"
    // before they go looking for missing AVX-512 perf.
    crate::kernels::log_backend();

    // Install the default plugin-system math backend (gist Task 2).
    // Logged on the same boot line so ops can see both the low-level
    // CPU-feature dispatch and the high-level Backend in one place.
    //
    // For the `serve` subcommand we **defer** the default install
    // until `cmd_serve` has loaded the TOML config — the hybrid
    // compute offload (`[real_transformer].compute_offload`, gist
    // Part 2 fix #5) is selected from there, and it must run
    // *before* `install_default` claims the OnceLock. Other
    // subcommands keep the immediate install so their math path is
    // ready as soon as `main` returns into them.
    //
    // The `run` subcommand grows one extra wrinkle: when invoked with
    // `--gpu` it must initialise the GPU compute backend *before* the
    // default CPU backend claims the `OnceLock` (gist Fix 2), mirroring
    // `cmd_serve`. A failed GPU init falls back to `install_default`.
    let run_gpu_cache = if let Cmd::Run {
        gpu: true,
        gpu_cache_mb,
        ..
    } = &cli.cmd
    {
        install_run_gpu_backend(*gpu_cache_mb)
    } else if !matches!(cli.cmd, Cmd::Serve { .. }) {
        crate::backend::install_default();
        let b = crate::backend::current();
        info!(
            backend = b.device_name(),
            compute_plane = b.compute_plane(),
            "math backend installed"
        );
        None
    } else {
        None
    };

    match cli.cmd {
        Cmd::GenData {
            data_dir,
            num_experts,
            expert_size,
            d_model,
            d_ff,
            block_align,
            dtype,
        } => cmd_gen_data(
            &data_dir,
            num_experts,
            expert_size,
            d_model,
            d_ff,
            block_align,
            &dtype,
        ),
        Cmd::Repack {
            data_dir,
            out_blob,
            out_manifest,
            num_experts,
            expert_size,
            block_align,
            no_direct,
            num_experts_per_layer,
            profile,
            order,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_repack(RepackArgs {
                data_dir,
                out_blob,
                out_manifest,
                num_experts,
                expert_size,
                block_align,
                no_direct,
                num_experts_per_layer,
                profile,
                order,
            }))
        }
        Cmd::Run { .. } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(async move {
                if let Cmd::Run {
                    data_dir,
                    num_experts,
                    expert_size,
                    d_model,
                    d_ff,
                    cache_slots,
                    top_k,
                    tokens,
                    predict_fanout,
                    predict_min_prob,
                    no_direct,
                    block_align,
                    seed,
                    dtype,
                    partial_load_fraction,
                    pin_after_observations,
                    alias_map,
                    io_uring,
                    token_pause_us,
                    first_token,
                    no_prefetch,
                    io_only,
                    force_ssd,
                    router_clusters,
                    router_intra_p,
                    router_matrix,
                    gate_weights,
                    trace_out,
                    gpu,
                    gpu_cache_mb: _,
                    pipeline_depth,
                    speculator,
                    speculator_hidden_dim,
                    speculator_top_k,
                    locality,
                    locality_window,
                    locality_threshold_pct,
                    affinity,
                    affinity_neighbors_k,
                    affinity_decay_epoch,
                    prefetch_governor,
                    prefetch_precision_floor,
                    prefetch_contention_weight,
                    cost_aware_eviction,
                    pregate,
                    static_residency_fraction,
                    static_residency_warmup_tokens,
                    static_residency_profile,
                    profile_out,
                    workload,
                    zipf_s,
                    workload_correlation,
                    replay_trace,
                    num_layers,
                    num_experts_per_layer,
                    packed_blob,
                    packed_manifest,
                } = cli.cmd
                {
                    let dtype =
                        crate::inference::WeightDtype::from_str_opt(&dtype).ok_or_else(|| {
                            format!(
                                "--dtype: unknown value {dtype:?} (supported: {SUPPORTED_RUNTIME_DTYPES})"
                            )
                        })?;
                    cmd_run(
                        RunArgs {
                            data_dir,
                            num_experts,
                            expert_size,
                            d_model,
                            d_ff,
                            cache_slots,
                            top_k,
                            tokens,
                            predict_fanout,
                            predict_min_prob,
                            no_direct,
                            block_align,
                            seed,
                            dtype,
                            partial_load_fraction,
                            pin_after_observations,
                            alias_map_path: alias_map,
                            io_uring,
                            token_pause_us,
                            first_token,
                            no_prefetch,
                            io_only,
                            force_ssd,
                            router_clusters,
                            router_intra_p,
                            router_matrix,
                            gate_weights,
                            trace_out,
                            gpu_expert_cache: if gpu { run_gpu_cache.clone() } else { None },
                            pipeline_depth,
                            speculator,
                            speculator_hidden_dim,
                            speculator_top_k,
                            locality,
                            locality_window,
                            locality_threshold_pct,
                            affinity,
                            affinity_neighbors_k,
                            affinity_decay_epoch,
                            prefetch_governor,
                            prefetch_precision_floor,
                            prefetch_contention_weight,
                            cost_aware_eviction,
                            pregate,
                            static_residency_fraction,
                            static_residency_warmup_tokens,
                            static_residency_profile,
                            profile_out,
                            workload,
                            zipf_s,
                            workload_correlation,
                            replay_trace,
                            num_layers,
                            num_experts_per_layer,
                            packed_blob,
                            packed_manifest,
                        },
                        startup_pinned,
                    )
                    .await
                } else {
                    unreachable!()
                }
            })
        }
        Cmd::Serve { config } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_serve(config))
        }
        Cmd::BenchReal {
            config,
            prompt,
            request_json,
            output_tokens,
            warmup_runs,
            measured_runs,
            cache_reset,
            greedy,
            format,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_bench_real(BenchRealArgs {
                config,
                prompt,
                request_json,
                output_tokens,
                warmup_runs,
                measured_runs,
                cache_reset,
                greedy,
                format,
            }))
        }
        Cmd::MatvecMicrobench {
            backend,
            warmup_runs,
            measured_runs,
            skip_lm_head,
            json,
        } => cmd_matvec_microbench(MatvecMicrobenchArgs {
            backends: backend,
            warmup_runs,
            measured_runs,
            skip_lm_head,
            json,
        }),
        Cmd::ScratchAllocMicrobench {
            d_model,
            num_experts,
            top_k,
            warmup_tokens,
            measured_tokens,
            json,
        } => cmd_scratch_alloc_microbench(ScratchAllocMicrobenchArgs {
            d_model,
            num_experts,
            top_k,
            warmup_tokens,
            measured_tokens,
            json,
        }),
        Cmd::GgufConvert {
            gguf_path,
            out_dir,
            num_layers,
            num_experts,
            no_uth,
            legacy_eager,
            native_quant,
            experts_only,
        } => cmd_gguf_convert(
            &gguf_path,
            &out_dir,
            num_layers,
            num_experts,
            !no_uth,
            legacy_eager,
            native_quant,
            experts_only,
        ),
        Cmd::ValidateData { data_dir } => cmd_validate_data(&data_dir),
        Cmd::ValidatePredictor { trace, cache_slots } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_validate_predictor(&trace, &cache_slots))
        }
        Cmd::Monitor { url, refresh_ms } => {
            #[cfg(feature = "tui")]
            {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()?;
                rt.block_on(crate::tui::run_monitor(&url, refresh_ms))
            }
            #[cfg(not(feature = "tui"))]
            {
                let _ = (url, refresh_ms);
                Err("monitor subcommand requires the `tui` cargo feature; \
                     rebuild without `--no-default-features` to enable it"
                    .into())
            }
        }
    }
}

/// Install the GPU compute backend for the `run` subcommand (gist
/// Fix 2).
///
/// Mirrors the GPU init path in [`cmd_serve`]: it builds a
/// bounded [`GpuExpertCache`] and initialises a
/// [`BackendBox`](crate::backend::BackendBox) before the default CPU
/// backend claims the `OnceLock`. On any failure — no GPU device or a
/// `set_backend` race — it falls back to
/// [`install_default`](crate::backend::install_default) with a
/// warning so the benchmark still runs on CPU.
fn install_run_gpu_backend(
    gpu_cache_mb: usize,
) -> Option<Arc<crate::expert_cache::GpuExpertCache>> {
    // The KV-cache geometry below only sizes the dense-backbone cache,
    // which the synthetic `run` benchmark does not exercise — it routes
    // everything through `expert_matmul`.
    //
    // Give run-mode GPU promotion a bounded (but non-zero) budget so
    // repeated experts can become true GPU hits. Sized by
    // `--gpu-cache-mb` (default 4 GiB — a single Mixtral-8x7B Q4
    // expert is ~99 MiB, so anything much smaller thrashes).
    let gpu_expert_cache = std::sync::Arc::new(crate::expert_cache::GpuExpertCache::new(
        gpu_cache_mb.saturating_mul(1024 * 1024),
        0.5,
        16,
    ));
    let backend_box = crate::backend::BackendBox::init_blocking(
        1, // num_layers
        1, // max_seq_len
        1, // num_heads
        1, // num_kv_heads
        1, // head_dim
        1, // v_head_dim
        gpu_expert_cache.clone(),
    );
    if !backend_box.is_gpu() {
        warn!("GPU REQUEST FAILED — RUNNING ON CPU");
        crate::backend::install_default();
        let b = crate::backend::current();
        info!(
            backend = b.device_name(),
            compute_plane = b.compute_plane(),
            "math backend installed"
        );
        return None;
    }
    let device_name = backend_box.device_name().to_string();
    let compute_plane = backend_box.compute_plane().to_string();
    let gpu = std::sync::Arc::new(backend_box);
    if let Err(e) = crate::backend::set_backend(gpu) {
        warn!(error = e, "GPU REQUEST FAILED — RUNNING ON CPU");
        crate::backend::install_default();
        let b = crate::backend::current();
        info!(
            backend = b.device_name(),
            compute_plane = b.compute_plane(),
            "math backend installed"
        );
        None
    } else {
        info!(
            device = device_name,
            compute_plane,
            vram_capacity_mb = gpu_cache_mb,
            "GpuBackend installed for run benchmark"
        );
        Some(gpu_expert_cache)
    }
}

/// **Tier 2.** Attach a packed expert blob to `storage` when both the blob
/// and its manifest are configured, after validating the manifest's slot
/// size against the engine's `expert_size`. Returns the storage unchanged
/// when no packed layout is configured (the default). Shared by the
/// `serve` and `run` engine-build paths.
fn maybe_attach_packed_blob(
    storage: NvmeStorage,
    packed_blob: Option<&std::path::Path>,
    packed_manifest: Option<&std::path::Path>,
    use_direct_io: bool,
    expert_size: usize,
) -> Result<NvmeStorage, Box<dyn std::error::Error>> {
    match (packed_blob, packed_manifest) {
        (Some(blob_path), Some(manifest_path)) => {
            let blob =
                crate::packed_storage::PackedBlob::open(blob_path, manifest_path, use_direct_io)?;
            blob.validate()
                .map_err(|e| format!("packed blob validation failed: {e}"))?;
            let slot = blob.manifest().expert_size;
            if slot != expert_size as u64 {
                return Err(format!(
                    "packed manifest expert_size ({slot}) != expert_size ({expert_size}); \
                     re-run `repack` with the matching --expert-size"
                )
                .into());
            }
            info!(
                experts = blob.len(),
                blob = %blob_path.display(),
                "Tier 2: packed expert blob attached (single-fd reads + coalesced preadv)"
            );
            Ok(storage.with_packed_blob(Arc::new(blob)))
        }
        (Some(_), None) | (None, Some(_)) => Err(
            "both packed_blob and packed_manifest must be set to enable the packed layout".into(),
        ),
        (None, None) => Ok(storage),
    }
}

struct BenchRealArgs {
    config: PathBuf,
    prompt: Option<String>,
    request_json: Option<PathBuf>,
    output_tokens: Option<usize>,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: BenchRealCacheReset,
    greedy: bool,
    format: BenchRealOutputFormat,
}

struct MatvecMicrobenchArgs {
    backends: Vec<crate::parallel::DenseMatvecBackend>,
    warmup_runs: usize,
    measured_runs: usize,
    skip_lm_head: bool,
    json: bool,
}

#[cfg_attr(not(feature = "alloc-count"), allow(dead_code))]
struct ScratchAllocMicrobenchArgs {
    d_model: usize,
    num_experts: usize,
    top_k: usize,
    warmup_tokens: usize,
    measured_tokens: usize,
    json: bool,
}

#[derive(Serialize)]
struct MatvecMicrobenchReport {
    benchmark: &'static str,
    model: &'static str,
    d_model: usize,
    d_ff: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    warmup_runs: usize,
    measured_runs: usize,
    build: BenchRealBuildInfo,
    results: Vec<MatvecMicrobenchResult>,
}

#[derive(Serialize)]
struct MatvecMicrobenchResult {
    shape: &'static str,
    backend: String,
    rows: usize,
    cols: usize,
    multiply_accumulates: usize,
    best_ms: f64,
    mean_ms: f64,
    checksum: u64,
}

#[cfg(feature = "alloc-count")]
#[derive(Clone, Copy, Debug, Default, Serialize)]
struct AllocationSnapshot {
    allocation_calls: u64,
    deallocation_calls: u64,
    reallocation_calls: u64,
    bytes_allocated: u64,
    bytes_deallocated: u64,
    current_bytes: u64,
    peak_bytes: u64,
}

#[cfg(feature = "alloc-count")]
#[derive(Serialize)]
struct ScratchAllocMicrobenchReport {
    benchmark: &'static str,
    model: &'static str,
    d_model: usize,
    num_experts: usize,
    top_k: usize,
    warmup_tokens: usize,
    measured_tokens: usize,
    build: BenchRealBuildInfo,
    results: Vec<ScratchAllocMicrobenchResult>,
}

#[cfg(feature = "alloc-count")]
#[derive(Serialize)]
struct ScratchAllocMicrobenchResult {
    variant: &'static str,
    elapsed_ms: f64,
    allocations: AllocationSnapshot,
    allocation_calls_per_token: f64,
    bytes_allocated_per_token: f64,
    checksum: u64,
}

#[cfg(feature = "alloc-count")]
#[derive(Clone, Copy)]
enum ScratchAllocVariant {
    CompatibilityWrappers,
    ScratchBuffers,
}

#[cfg(feature = "alloc-count")]
impl ScratchAllocVariant {
    fn label(self) -> &'static str {
        match self {
            Self::CompatibilityWrappers => "compatibility-wrappers",
            Self::ScratchBuffers => "scratch-buffers",
        }
    }
}

#[derive(Clone, Copy)]
struct MatvecShape {
    name: &'static str,
    rows: usize,
    cols: usize,
}

struct BenchRealInput {
    prompt: String,
    output_tokens: usize,
}

struct BenchRealRuntime {
    cfg: crate::config::Config,
    engine: Arc<Engine>,
    model: Arc<crate::model::RealModel>,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
}

#[derive(Serialize)]
struct BenchRealSuiteReport {
    benchmark: &'static str,
    config: String,
    prompt: String,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: BenchRealCacheReset,
    greedy: bool,
    build: BenchRealBuildInfo,
    aggregate: BenchRealAggregate,
    runs: Vec<BenchRealRunReport>,
}

#[derive(Serialize)]
struct BenchRealBuildInfo {
    git_commit: String,
    build_features: Vec<&'static str>,
    threads: usize,
    dense_matvec_backend: String,
}

#[derive(Clone, Serialize)]
struct BenchRealRunReport {
    run_index: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    total_api_tokens: usize,
    model_forward_evaluations: usize,
    lm_head_evaluations: usize,
    prompt_seconds: f64,
    prompt_tps: f64,
    decode_seconds: f64,
    decode_tps: f64,
    time_to_first_token_seconds: f64,
    total_seconds: f64,
    decode_token_latency_p50_ms: f64,
    decode_token_latency_p95_ms: f64,
    decode_token_latency_p99_ms: f64,
    decode_token_latency_max_ms: f64,
    cache_hits: u64,
    cache_misses: u64,
    hit_rate: f64,
    ssd_bytes: u64,
    ssd_stall_seconds: f64,
    rss_bytes: Option<u64>,
    output_token_ids: Vec<u32>,
    output_text: String,
    stage_timings: std::collections::BTreeMap<String, crate::stage_timing::StageTimingSnapshot>,
}

#[derive(Serialize)]
struct BenchRealAggregate {
    prompt_seconds_mean: f64,
    prompt_tps_mean: f64,
    decode_seconds_mean: f64,
    decode_tps_mean: f64,
    time_to_first_token_p50_seconds: f64,
    total_seconds_mean: f64,
    cache_hits_total: u64,
    cache_misses_total: u64,
    hit_rate: f64,
    ssd_bytes_total: u64,
    output_token_parity: bool,
}

async fn cmd_bench_real(args: BenchRealArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input = load_bench_real_input(&args)?;
    if args.measured_runs == 0 {
        return Err("bench-real requires --measured-runs > 0".into());
    }

    if args.cache_reset == BenchRealCacheReset::Keep {
        let runtime = build_bench_real_runtime(&args.config).await?;
        let params = bench_sampling_params(&runtime.cfg, args.greedy);
        for i in 0..args.warmup_runs {
            let _ = run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i)
                .await?;
        }
        let softmax_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut runs = Vec::with_capacity(args.measured_runs);
        for i in 0..args.measured_runs {
            runs.push(
                run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i)
                    .await?,
            );
        }
        assert_no_softmax_fallbacks(softmax_before)?;
        emit_bench_real_report(&args, input, runs)?;
    } else {
        for i in 0..args.warmup_runs {
            let runtime = build_bench_real_runtime(&args.config).await?;
            let params = bench_sampling_params(&runtime.cfg, args.greedy);
            let _ = run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i)
                .await?;
        }
        let softmax_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut runs = Vec::with_capacity(args.measured_runs);
        for i in 0..args.measured_runs {
            let runtime = build_bench_real_runtime(&args.config).await?;
            let params = bench_sampling_params(&runtime.cfg, args.greedy);
            runs.push(
                run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i)
                    .await?,
            );
        }
        assert_no_softmax_fallbacks(softmax_before)?;
        emit_bench_real_report(&args, input, runs)?;
    }
    Ok(())
}

/// Strict real-model benchmarks must not emit attention-softmax non-finite
/// fallbacks: any increase over the measured window signals a numerically
/// invalid run (a `NaN`/`inf` propagated into attention), so the benchmark is
/// rejected rather than reporting a valid-looking throughput (gist Finding 9).
fn assert_no_softmax_fallbacks(before: u64) -> Result<(), Box<dyn std::error::Error>> {
    let delta = crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(before);
    if delta > 0 {
        return Err(format!(
            "bench-real INVALID: {delta} attention-softmax non-finite fallback(s) occurred during \
             the measured window; the run produced NaN/inf attention scores and is not a valid \
             measurement"
        )
        .into());
    }
    Ok(())
}

fn cmd_matvec_microbench(args: MatvecMicrobenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.measured_runs == 0 {
        return Err("matvec-microbench requires --measured-runs > 0".into());
    }
    let backends = if args.backends.is_empty() {
        vec![
            crate::parallel::DenseMatvecBackend::Matrixmultiply,
            crate::parallel::DenseMatvecBackend::Rayon,
            crate::parallel::DenseMatvecBackend::RayonMatrixmultiply,
        ]
    } else {
        args.backends.clone()
    };
    let mut shapes = vec![
        MatvecShape {
            name: "q-projection",
            rows: 32 * 128,
            cols: 2048,
        },
        MatvecShape {
            name: "k-projection",
            rows: 4 * 128,
            cols: 2048,
        },
        MatvecShape {
            name: "v-projection",
            rows: 4 * 128,
            cols: 2048,
        },
        MatvecShape {
            name: "o-projection",
            rows: 2048,
            cols: 32 * 128,
        },
        MatvecShape {
            name: "router-gate",
            rows: 128,
            cols: 2048,
        },
    ];
    if !args.skip_lm_head {
        shapes.push(MatvecShape {
            name: "lm-head",
            rows: 151_936,
            cols: 2048,
        });
    }

    let mut results = Vec::with_capacity(shapes.len() * backends.len());
    for (shape_idx, shape) in shapes.iter().enumerate() {
        let x = deterministic_f32_vec(shape.cols, 0x9e37_79b9 ^ shape_idx as u64);
        let w = deterministic_f32_vec(
            shape.rows * shape.cols,
            0xd1b5_4a32_d192_ed03u64 ^ shape_idx as u64,
        );
        for &backend in &backends {
            for _ in 0..args.warmup_runs {
                let y = crate::transformer::matmul_row_major_with_backend(
                    std::hint::black_box(&w),
                    std::hint::black_box(&x),
                    shape.rows,
                    shape.cols,
                    backend,
                );
                std::hint::black_box(&y);
            }
            let mut total = Duration::ZERO;
            let mut best = Duration::MAX;
            let mut checksum = 0u64;
            for _ in 0..args.measured_runs {
                let started = Instant::now();
                let y = crate::transformer::matmul_row_major_with_backend(
                    std::hint::black_box(&w),
                    std::hint::black_box(&x),
                    shape.rows,
                    shape.cols,
                    backend,
                );
                let elapsed = started.elapsed();
                std::hint::black_box(&y);
                checksum = checksum_f32_bits(&y);
                total += elapsed;
                best = best.min(elapsed);
            }
            results.push(MatvecMicrobenchResult {
                shape: shape.name,
                backend: backend.to_string(),
                rows: shape.rows,
                cols: shape.cols,
                multiply_accumulates: shape.rows.saturating_mul(shape.cols),
                best_ms: best.as_secs_f64() * 1_000.0,
                mean_ms: (total.as_secs_f64() * 1_000.0) / args.measured_runs as f64,
                checksum,
            });
        }
    }

    let report = MatvecMicrobenchReport {
        benchmark: "matvec-microbench",
        model: "Qwen3-Coder-30B-A3B-Instruct Q8_0",
        d_model: 2048,
        d_ff: 768,
        num_heads: 32,
        num_kv_heads: 4,
        head_dim: 128,
        vocab_size: 151_936,
        warmup_runs: args.warmup_runs,
        measured_runs: args.measured_runs,
        build: BenchRealBuildInfo {
            git_commit: git_commit_short(),
            build_features: build_features(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            dense_matvec_backend: crate::parallel::dense_matvec_backend().to_string(),
        },
        results,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_matvec_microbench_human(&report);
    }
    Ok(())
}

fn cmd_scratch_alloc_microbench(
    args: ScratchAllocMicrobenchArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "alloc-count"))]
    {
        let _ = args;
        Err(
            "scratch-alloc-microbench requires `--features alloc-count` so the binary installs the counting allocator"
                .into(),
        )
    }

    #[cfg(feature = "alloc-count")]
    {
        if args.measured_tokens == 0 {
            return Err("scratch-alloc-microbench requires --measured-tokens > 0".into());
        }
        if args.d_model == 0 || !args.d_model.is_multiple_of(4) {
            return Err(
                "scratch-alloc-microbench requires --d-model to be a positive multiple of 4".into(),
            );
        }
        if args.num_experts == 0 {
            return Err("scratch-alloc-microbench requires --num-experts > 0".into());
        }
        if args.top_k == 0 || args.top_k > args.num_experts {
            return Err("scratch-alloc-microbench requires 0 < --top-k <= --num-experts".into());
        }

        let backend = crate::backend::current();
        let results = vec![
            run_scratch_alloc_variant(
                &args,
                ScratchAllocVariant::CompatibilityWrappers,
                &backend,
            ),
            run_scratch_alloc_variant(
                &args,
                ScratchAllocVariant::ScratchBuffers,
                &backend,
            ),
        ];

        let report = ScratchAllocMicrobenchReport {
            benchmark: "scratch-alloc-microbench",
            model: "synthetic-transformer-layer",
            d_model: args.d_model,
            num_experts: args.num_experts,
            top_k: args.top_k,
            warmup_tokens: args.warmup_tokens,
            measured_tokens: args.measured_tokens,
            build: BenchRealBuildInfo {
                git_commit: git_commit_short(),
                build_features: build_features(),
                threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
                dense_matvec_backend: crate::parallel::dense_matvec_backend().to_string(),
            },
            results,
        };
        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_scratch_alloc_microbench_human(&report);
        }
        Ok(())
    }
}

#[cfg(feature = "alloc-count")]
fn run_scratch_alloc_variant(
    args: &ScratchAllocMicrobenchArgs,
    variant: ScratchAllocVariant,
    backend: &crate::backend::BackendBox,
) -> ScratchAllocMicrobenchResult {
    let layer = make_synthetic_transformer_layer(args.d_model, args.num_experts, args.top_k);
    let expert_bank = make_synthetic_expert_bank(args.d_model, args.num_experts);
    let mut selected_outputs = vec![vec![0.0f32; args.d_model]; args.top_k];
    let mut hidden = deterministic_f32_vec(args.d_model, 0x7a11_0ca7_ed15_ea5e);
    let mut kv = crate::transformer::KvCache::new_kv(layer.kv_dim(), layer.v_dim());
    let mut pos = 0usize;

    let mut scratch = crate::transformer::TransformerLayerScratch::new();
    let mut next_hidden = Vec::with_capacity(args.d_model);

    for _ in 0..args.warmup_tokens {
        run_synthetic_decode_step(
            &layer,
            variant,
            &mut hidden,
            pos,
            &mut kv,
            backend,
            &expert_bank,
            &mut selected_outputs,
            &mut scratch,
            &mut next_hidden,
        );
        pos += 1;
    }

    alloc_count::reset();
    let started = Instant::now();
    for _ in 0..args.measured_tokens {
        run_synthetic_decode_step(
            &layer,
            variant,
            &mut hidden,
            pos,
            &mut kv,
            backend,
            &expert_bank,
            &mut selected_outputs,
            &mut scratch,
            &mut next_hidden,
        );
        pos += 1;
        std::hint::black_box(&hidden);
    }
    let elapsed = started.elapsed();
    let allocations = alloc_count::snapshot();
    ScratchAllocMicrobenchResult {
        variant: variant.label(),
        elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
        allocations,
        allocation_calls_per_token: allocations.allocation_calls as f64
            / args.measured_tokens as f64,
        bytes_allocated_per_token: allocations.bytes_allocated as f64 / args.measured_tokens as f64,
        checksum: checksum_f32_bits(&hidden),
    }
}

#[cfg(feature = "alloc-count")]
#[allow(clippy::too_many_arguments)]
fn run_synthetic_decode_step(
    layer: &crate::transformer::TransformerLayer,
    variant: ScratchAllocVariant,
    hidden: &mut Vec<f32>,
    pos: usize,
    kv: &mut crate::transformer::KvCache,
    backend: &crate::backend::BackendBox,
    expert_bank: &[Vec<f32>],
    selected_outputs: &mut [Vec<f32>],
    scratch: &mut crate::transformer::TransformerLayerScratch,
    next_hidden: &mut Vec<f32>,
) {
    match variant {
        ScratchAllocVariant::CompatibilityWrappers => {
            let after_attn = layer.attn_block_with_timing(hidden, pos, 0, kv, backend, None);
            let (_normed, routing) = layer.moe_pre_with_timing(&after_attn, None);
            copy_selected_expert_outputs(&routing, expert_bank, selected_outputs);
            *hidden = layer.moe_combine_with_timing(
                &after_attn,
                &selected_outputs[..routing.experts.len()],
                &routing.weights,
                None,
            );
        }
        ScratchAllocVariant::ScratchBuffers => {
            layer.attn_block_into_with_timing(
                hidden,
                pos,
                0,
                kv,
                backend,
                scratch,
                next_hidden,
                None,
            );
            std::mem::swap(hidden, next_hidden);
            next_hidden.clear();

            let routing = layer.moe_pre_into_with_timing(hidden, scratch, None);
            copy_selected_expert_outputs(&routing, expert_bank, selected_outputs);
            let mut moe_accum = std::mem::take(&mut scratch.moe_accum);
            layer.moe_combine_into_with_timing(
                hidden,
                &selected_outputs[..routing.experts.len()],
                &routing.weights,
                &mut moe_accum,
                next_hidden,
                None,
            );
            scratch.moe_accum = moe_accum;
            std::mem::swap(hidden, next_hidden);
            next_hidden.clear();
            scratch.routing.recycle_decision(routing);
        }
    }
}

#[cfg(feature = "alloc-count")]
fn make_synthetic_transformer_layer(
    d_model: usize,
    num_experts: usize,
    top_k: usize,
) -> crate::transformer::TransformerLayer {
    let head_dim = 4;
    let num_heads = d_model / head_dim;
    let num_kv_heads = num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let mk = |rows: usize, cols: usize, scale: f32| {
        (0..rows * cols)
            .map(|i| ((i % 7) as f32 - 3.0) * scale)
            .collect::<Vec<f32>>()
    };
    crate::transformer::TransformerLayer {
        rms_attn: crate::transformer::RmsNorm::new(vec![1.0; d_model], 1e-6),
        attn: crate::transformer::MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: crate::dense_tensor::DenseWeight::from_f32(
                mk(q_dim, d_model, 0.05),
                q_dim,
                d_model,
            ),
            wk: crate::dense_tensor::DenseWeight::from_f32(
                mk(kv_dim, d_model, 0.05),
                kv_dim,
                d_model,
            ),
            wv: crate::dense_tensor::DenseWeight::from_f32(
                mk(kv_dim, d_model, 0.05),
                kv_dim,
                d_model,
            ),
            wo: crate::dense_tensor::DenseWeight::from_f32(
                mk(d_model, q_dim, 0.05),
                d_model,
                q_dim,
            ),
            window_size: None,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        },
        mla: None,
        rms_moe: crate::transformer::RmsNorm::new(vec![1.0; d_model], 1e-6),
        gate: crate::gating::LinearGate::new(
            mk(num_experts, d_model, 0.1),
            num_experts,
            d_model,
            top_k,
        ),
        shared_expert: None,
        dense_ffn: None,
    }
}

#[cfg(feature = "alloc-count")]
fn make_synthetic_expert_bank(d_model: usize, num_experts: usize) -> Vec<Vec<f32>> {
    (0..num_experts)
        .map(|expert| deterministic_f32_vec(d_model, 0x5eed_0000_u64 | expert as u64))
        .collect()
}

#[cfg(feature = "alloc-count")]
fn copy_selected_expert_outputs(
    routing: &crate::gating::RoutingDecision,
    expert_bank: &[Vec<f32>],
    selected_outputs: &mut [Vec<f32>],
) {
    debug_assert!(routing.experts.len() <= selected_outputs.len());
    for (idx, &expert_id) in routing.experts.iter().enumerate() {
        let source = &expert_bank[expert_id as usize % expert_bank.len()];
        selected_outputs[idx].clear();
        selected_outputs[idx].extend_from_slice(source);
    }
}

#[cfg(feature = "alloc-count")]
fn print_scratch_alloc_microbench_human(report: &ScratchAllocMicrobenchReport) {
    println!("scratch-alloc-microbench");
    println!(
        "  model={} d_model={} num_experts={} top_k={}",
        report.model, report.d_model, report.num_experts, report.top_k
    );
    println!(
        "  warmup_tokens={} measured_tokens={} git={} threads={} features={}",
        report.warmup_tokens,
        report.measured_tokens,
        report.build.git_commit,
        report.build.threads,
        report.build.build_features.join(",")
    );
    for result in &report.results {
        println!(
            "  {:<23} tokens={:<4} elapsed={:>8.3}ms allocs={:<7} reallocs={:<5} bytes={:<10} peak={:<10} allocs/token={:>7.2} bytes/token={:>9.1} checksum={:#016x}",
            result.variant,
            report.measured_tokens,
            result.elapsed_ms,
            result.allocations.allocation_calls,
            result.allocations.reallocation_calls,
            result.allocations.bytes_allocated,
            result.allocations.peak_bytes,
            result.allocation_calls_per_token,
            result.bytes_allocated_per_token,
            result.checksum
        );
    }
    if let [baseline, scratch] = report.results.as_slice() {
        println!(
            "  reduction vs wrappers: alloc_calls={:>6.2}% bytes_allocated={:>6.2}%",
            percent_reduction(
                baseline.allocations.allocation_calls,
                scratch.allocations.allocation_calls
            ),
            percent_reduction(
                baseline.allocations.bytes_allocated,
                scratch.allocations.bytes_allocated
            )
        );
    }
}

#[cfg(feature = "alloc-count")]
fn percent_reduction(baseline: u64, measured: u64) -> f64 {
    if baseline == 0 {
        return 0.0;
    }
    ((baseline as f64 - measured as f64) / baseline as f64) * 100.0
}

fn deterministic_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let v = ((bits >> 40) as u32 & 0xffff) as f32 / 32768.0 - 1.0;
        out.push(v);
    }
    out
}

fn checksum_f32_bits(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn print_matvec_microbench_human(report: &MatvecMicrobenchReport) {
    println!("matvec-microbench");
    println!(
        "  model={} d_model={} d_ff={} heads={} kv_heads={} head_dim={} vocab={}",
        report.model,
        report.d_model,
        report.d_ff,
        report.num_heads,
        report.num_kv_heads,
        report.head_dim,
        report.vocab_size
    );
    println!(
        "  warmup_runs={} measured_runs={} git={} threads={} features={}",
        report.warmup_runs,
        report.measured_runs,
        report.build.git_commit,
        report.build.threads,
        report.build.build_features.join(",")
    );
    for result in &report.results {
        println!(
            "  {:<13} {:<22} rows={:<6} cols={:<5} best={:>9.3}ms mean={:>9.3}ms checksum={:#016x}",
            result.shape,
            result.backend,
            result.rows,
            result.cols,
            result.best_ms,
            result.mean_ms,
            result.checksum
        );
    }
}

fn load_bench_real_input(
    args: &BenchRealArgs,
) -> Result<BenchRealInput, Box<dyn std::error::Error>> {
    let mut json_max_tokens = None;
    let prompt = if let Some(prompt) = args.prompt.as_ref() {
        prompt.clone()
    } else if let Some(path) = args.request_json.as_ref() {
        let body = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&body)?;
        json_max_tokens = value
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
            prompt.to_string()
        } else if let Some(messages) = value.get("messages").and_then(|v| v.as_array()) {
            flatten_bench_messages(messages)
        } else {
            return Err(
                "--request-json must contain a string `prompt` or chat `messages` array".into(),
            );
        }
    } else {
        return Err("bench-real requires either --prompt or --request-json".into());
    };
    if prompt.is_empty() {
        return Err("bench-real prompt must be non-empty".into());
    }
    let output_tokens = args.output_tokens.or(json_max_tokens).unwrap_or(16);
    if output_tokens == 0 {
        return Err("bench-real requires output token count > 0".into());
    }
    Ok(BenchRealInput {
        prompt,
        output_tokens,
    })
}

fn flatten_bench_messages(messages: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        out.push_str(role);
        out.push_str(": ");
        out.push_str(content);
        out.push('\n');
    }
    out
}

fn bench_sampling_params(
    cfg: &crate::config::Config,
    greedy: bool,
) -> crate::sampling::SamplingParams {
    if greedy {
        crate::sampling::SamplingParams::greedy()
    } else {
        cfg.sampling.to_params()
    }
}

async fn build_bench_real_runtime(
    config_path: &Path,
) -> Result<BenchRealRuntime, Box<dyn std::error::Error>> {
    use crate::config::Config;
    use crate::metrics::Metrics;
    use crate::tokenizer::Tokenizer;

    let mut cfg = Config::from_file(config_path)?;
    crate::parallel::set_dense_matvec_backend(cfg.real_transformer.dense_matvec_backend);
    if !cfg.real_transformer.enabled {
        return Err("bench-real requires [real_transformer] enabled = true".into());
    }
    if cfg.real_transformer.compute_offload == crate::backend::ComputeOffload::Gpu
        || cfg.gpu_cache.enabled
    {
        return Err(
            "bench-real is CPU-only for this sprint; disable real_transformer.compute_offload = \"gpu\" and [gpu_cache].enabled"
                .into(),
        );
    }
    if cfg.distributed.enabled {
        return Err(
            "bench-real runs the local direct real-model path; disable distributed.enabled".into(),
        );
    }
    if cfg.real_transformer.weights_dir.is_none() {
        return Err("bench-real requires real_transformer.weights_dir; seeded fallback benchmarks are not production measurements".into());
    }
    if cfg.real_transformer.allow_seeded_fallback {
        return Err(
            "bench-real rejects real_transformer.allow_seeded_fallback = true; seeded fallback \
             benchmarks are not production measurements"
                .into(),
        );
    }
    if !cfg.real_transformer.strict_weights {
        return Err(
            "bench-real requires real_transformer.strict_weights = true; a benchmark that may \
             include seeded fallback tensors is not a production measurement"
                .into(),
        );
    }

    let (resolved_architecture, resolved_first_k_dense_replace, resolved_advanced) =
        reconcile_bench_real_config(&mut cfg)?;
    // Finding 7: validate the fully-resolved configuration before any weights
    // are loaded or backends installed.
    validate_resolved_bench_real_config(
        &cfg,
        resolved_architecture,
        resolved_first_k_dense_replace,
        &resolved_advanced,
    )?;

    crate::backend::install_default();
    {
        let b = crate::backend::current();
        info!(
            backend = b.device_name(),
            compute_plane = b.compute_plane(),
            "bench-real math backend installed"
        );
    }

    if !cfg.model.data_dir.is_dir() {
        return Err(format!(
            "data dir {} does not exist; run `gen-data` or the extractor first",
            cfg.model.data_dir.display()
        )
        .into());
    }

    let storage = NvmeStorage::new(StorageConfig {
        base_path: cfg.model.data_dir.clone(),
        expert_size: cfg.model.expert_size,
        block_align: cfg.storage.block_align,
        use_direct_io: !cfg.storage.no_direct,
        num_experts_per_layer: if cfg.model.num_layers > 1 {
            Some(cfg.model.num_experts)
        } else {
            None
        },
    })?;
    let storage = maybe_attach_packed_blob(
        storage,
        cfg.storage.packed_blob.as_deref(),
        cfg.storage.packed_manifest.as_deref(),
        !cfg.storage.no_direct,
        cfg.model.expert_size,
    )?;
    let storage = Arc::new(storage);
    let total_experts_for_files =
        (cfg.model.num_layers as u32).saturating_mul(cfg.model.num_experts);
    if !storage.is_packed() {
        storage.warmup_fds(0..total_experts_for_files)?;
    }

    let pipeline_depth = cfg.storage.pipeline_depth.max(1) as usize;
    let shadow_slots = cfg.storage.predict_fanout.saturating_mul(pipeline_depth);
    let primary_slots = cfg.storage.cache_slots + 1;
    let pool = if shadow_slots > 0 {
        BufferPool::new_with_shadow(
            primary_slots,
            shadow_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    } else {
        BufferPool::new(
            primary_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    };
    let cache = {
        let num_layers = cfg.model.num_layers.max(1);
        let per_layer = cfg.model.num_experts.max(1);
        let total = cfg.storage.cache_slots.max(1);
        let base = total / num_layers;
        let extra = total % num_layers;
        let caps: Vec<usize> = (0..num_layers)
            .map(|i| base + if i < extra { 1 } else { 0 })
            .collect();
        if num_layers == 1 {
            Arc::new(MultiLayerExpertCache::single_layer(total))
        } else {
            Arc::new(MultiLayerExpertCache::with_capacities(caps, per_layer))
        }
    };
    let total_experts: u32 = (cfg.model.num_layers as u32)
        .saturating_mul(cfg.model.num_experts)
        .max(cfg.model.num_experts);
    let predictor = Arc::new(PredictiveLoader::new(
        total_experts,
        cfg.storage.predict_fanout,
        resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts),
        0xC0FFEE,
    ));

    let rt = &cfg.real_transformer;
    let head_dim = if rt.head_dim == 0 {
        cfg.model.d_model / rt.num_heads
    } else {
        rt.head_dim
    };
    let num_kv_heads = if rt.num_kv_heads == 0 {
        rt.num_heads
    } else {
        rt.num_kv_heads
    };
    let model_cfg = crate::model::RealModelConfig {
        d_model: cfg.model.d_model,
        d_ff: cfg.model.d_ff,
        num_heads: rt.num_heads,
        num_kv_heads,
        head_dim,
        vocab_size: rt.vocab_size,
        num_layers: cfg.model.num_layers,
        num_experts: cfg.model.num_experts as usize,
        top_k: cfg.model.top_k,
        rope_base: rt.rope_base,
        rms_eps: rt.rms_eps,
        window_size: if rt.window_size == 0 {
            None
        } else {
            Some(rt.window_size)
        },
        architecture: resolved_architecture,
        first_k_dense_replace: resolved_first_k_dense_replace,
        advanced: resolved_advanced,
    };
    let model = Arc::new(crate::model::RealModel::from_dir_auto_with_options(
        model_cfg,
        rt.weights_dir.as_ref().expect("weights_dir checked above"),
        rt.seed,
        crate::model::RealModelLoadOptions {
            strict_weights: rt.strict_weights,
        },
    )?);
    if model.load_status.seeded_fallback_remained {
        return Err(format!(
            "bench-real loaded an incomplete checkpoint: {} of {} required tensors loaded, \
             seeded fallback remained; not a production measurement",
            model.load_status.loaded_tensors, model.load_status.required_tensors
        )
        .into());
    }
    info!(
        strict = model.load_status.strict,
        loader = model.load_status.loader,
        loaded_tensors = model.load_status.loaded_tensors,
        required_tensors = model.load_status.required_tensors,
        seeded_fallback_remained = model.load_status.seeded_fallback_remained,
        "bench-real model-loading status"
    );

    info!(
        num_experts = model.layers[0].gate.num_experts,
        d_model = model.layers[0].gate.d_model,
        top_k = model.layers[0].gate.top_k,
        "bench-real routing: LinearGate wired from real model"
    );
    let router = crate::gating::Router::Linear(Arc::new(model.layers[0].gate.clone()));
    let metrics = Metrics::new();
    let mut engine_builder = Engine::with_options(
        cache,
        pool,
        storage,
        router,
        predictor,
        ModelShape {
            d_model: cfg.model.d_model,
            d_ff: cfg.model.d_ff,
            hidden_seed: 0xC0FFEE,
        },
        EngineOptions {
            io_only: false,
            dtype: cfg.model.dtype,
            partial_load_fraction: cfg.storage.partial_load_fraction,
            pin_after_observations: cfg.storage.pin_after_observations,
            use_qmm_for_q4: true,
            expert_execution_policy: cfg.real_transformer.expert_execution_policy,
            max_concurrent_prefetches: cfg.real_transformer.max_concurrent_prefetches,
            max_fetch_yields: cfg.real_transformer.max_fetch_yields,
            prefetch_governor: cfg.predictive.prefetch_governor,
            prefetch_precision_floor: cfg.predictive.prefetch_precision_floor,
            prefetch_contention_weight: cfg.predictive.prefetch_contention_weight,
            cost_aware_eviction: cfg.predictive.cost_aware_eviction,
            pregate_enabled: cfg.predictive.pregate_enabled,
            collect_route_profile: false,
        },
    );
    engine_builder = engine_builder.with_pipeline_depth(cfg.storage.pipeline_depth);
    if cfg.predictive.locality_enabled {
        let window = cfg
            .predictive
            .locality_window
            .saturating_mul(cfg.model.num_layers.max(1));
        let monitor = Arc::new(LocalityMonitor::new(total_experts, window));
        engine_builder =
            engine_builder.with_locality_monitor(monitor, cfg.predictive.locality_threshold_pct);
    }
    if cfg.predictive.speculator_enabled {
        let top_k = if cfg.predictive.speculator_top_k == 0 {
            cfg.model.top_k
        } else {
            cfg.predictive.speculator_top_k
        };
        let spec = Arc::new(NeuralSpeculator::new(
            cfg.model.d_model,
            cfg.predictive.speculator_hidden_dim,
            total_experts,
            0xC0FFEE,
        ));
        engine_builder = engine_builder.with_speculator(spec, top_k);
    }
    if cfg.predictive.affinity_enabled {
        let affinity = Arc::new(LayeredExpertAffinity::new(
            cfg.model.num_layers.max(1),
            cfg.model.num_experts,
        ));
        engine_builder = engine_builder.with_affinity(
            affinity,
            cfg.predictive.affinity_neighbors_k,
            cfg.predictive.affinity_decay_epoch,
        );
    }
    if cfg.predictive.static_residency_fraction > 0.0 {
        let profile = match cfg.predictive.static_residency_profile.as_ref() {
            Some(path) => Some(crate::residency::ResidencyProfile::load_json(
                std::path::Path::new(path),
            )?),
            None => None,
        };
        engine_builder = engine_builder.with_static_residency(
            cfg.predictive.static_residency_fraction,
            cfg.predictive.static_residency_warmup_tokens,
            profile,
        );
    }
    if cfg.predictive.pregate_enabled {
        let pregate = Arc::new(crate::pregate::PerLayerPreGate::new(
            cfg.model.num_layers.max(1),
            cfg.model.top_k,
        ));
        engine_builder = engine_builder.with_pregate(pregate);
    }
    let engine = Arc::new(engine_builder.with_metrics(metrics));

    let tokenizer = match cfg.tokenizer.path.as_ref() {
        Some(p) => match Tokenizer::from_file(p) {
            Ok(t) => Arc::new(t),
            Err(e) => {
                return Err(format!(
                    "bench-real failed to load tokenizer {}: {e}",
                    p.display()
                )
                .into());
            }
        },
        None => {
            return Err(
                "bench-real requires tokenizer.path; byte tokenizer fallback is not a production benchmark"
                    .into(),
            );
        }
    };
    // The tokenizer must be addressable by the reconciled model vocabulary
    // (Finding 4): every emittable token id must be < model.vocab_size.
    tokenizer
        .validate_vocab_compat(model.config.vocab_size)
        .map_err(|e| -> Box<dyn std::error::Error> {
            format!("bench-real tokenizer is incompatible with the model: {e}").into()
        })?;

    Ok(BenchRealRuntime {
        cfg,
        engine,
        model,
        tokenizer,
    })
}

fn reconcile_bench_real_config(
    cfg: &mut crate::config::Config,
) -> Result<
    (
        crate::architecture::Architecture,
        usize,
        crate::model::AdvancedConfig,
    ),
    Box<dyn std::error::Error>,
> {
    let mut resolved_architecture = crate::architecture::Architecture::Mixtral;
    let mut resolved_first_k_dense_replace = 0usize;
    let mut resolved_advanced = crate::model::AdvancedConfig::default();

    // Parse the TOML-declared architecture (if any) up front. It is used to
    // seed the default resolution and to cross-check against a checkpoint
    // `config.json` — it must never suppress checkpoint reconciliation.
    let toml_architecture = match cfg.real_transformer.architecture.clone() {
        Some(arch_str) => Some(
            crate::architecture::Architecture::from_model_type(&arch_str).ok_or_else(|| {
                format!(
                    "[real_transformer] architecture = \"{arch_str}\" is not a recognised model_type"
                )
            })?,
        ),
        None => None,
    };
    if let Some(arch) = toml_architecture {
        resolved_architecture = arch;
    }

    // Always reconcile from the checkpoint's `config.json` when it exists,
    // regardless of whether the TOML pins an architecture. This keeps
    // checkpoint-specific advanced routing semantics (norm_topk_prob,
    // scoring_func, routed_scaling_factor, n_group, topk_group,
    // num_shared_experts, rope_scaling, …) authoritative even when the user
    // points `weights_dir` at original safetensors.
    if let Some(dir) = cfg.real_transformer.weights_dir.clone() {
        match crate::architecture::HfConfig::from_dir(&dir) {
            Ok(Some(hf)) => {
                info!(
                    architecture = ?hf.architecture,
                    "config.json detected; reconciling bench-real hyperparameters from checkpoint"
                );
                // If TOML also pinned an architecture, it must agree with the
                // checkpoint. Silently preferring one over the other risks
                // loading a checkpoint under the wrong routing/attention path.
                if let Some(toml_arch) = toml_architecture {
                    if toml_arch != hf.architecture {
                        return Err(format!(
                            "[real_transformer] architecture = \"{}\" conflicts with checkpoint \
                             config.json architecture \"{}\" in {}; remove the TOML override or \
                             point weights_dir at a matching checkpoint",
                            toml_arch.model_type(),
                            hf.architecture.model_type(),
                            dir.display()
                        )
                        .into());
                    }
                }
                resolved_architecture = hf.architecture;
                resolved_first_k_dense_replace = hf.first_k_dense_replace.unwrap_or(0);
                resolved_advanced = crate::model::RealModelConfig::from_hf_config(&hf).advanced;
                crate::inference::set_swiglu_limit(resolved_advanced.swiglu_limit);
                cfg.model.d_model = hf.hidden_size;
                cfg.model.d_ff = hf.resolved_d_ff();
                cfg.model.num_layers = hf.num_hidden_layers;
                cfg.model.num_experts = hf.num_routed_experts.unwrap_or(1).max(1) as u32;
                cfg.model.top_k = hf
                    .num_experts_per_tok
                    .unwrap_or(1)
                    .clamp(1, cfg.model.num_experts.max(1) as usize);
                cfg.real_transformer.vocab_size = hf.vocab_size;
                cfg.real_transformer.num_heads = hf.num_attention_heads;
                cfg.real_transformer.num_kv_heads = if hf.num_key_value_heads == 0 {
                    hf.num_attention_heads
                } else {
                    hf.num_key_value_heads
                };
                cfg.real_transformer.head_dim = hf.resolved_head_dim();
                cfg.real_transformer.rope_base = hf.rope_theta;
                cfg.real_transformer.rms_eps = hf.rms_norm_eps;
                cfg.real_transformer.window_size = hf.sliding_window.unwrap_or(0);
            }
            // No `config.json` present: preserve TOML-only behaviour, using the
            // TOML architecture already resolved above (or the Mixtral default).
            Ok(None) => {}
            Err(e) => {
                return Err(
                    format!("failed to read config.json from {}: {e}", dir.display()).into(),
                );
            }
        }
    }
    Ok((
        resolved_architecture,
        resolved_first_k_dense_replace,
        resolved_advanced,
    ))
}

/// Finding 7: post-reconciliation, architecture-aware validation of the
/// fully-resolved bench-real configuration.
///
/// This runs *after* [`reconcile_bench_real_config`] has merged the TOML and
/// checkpoint `config.json` into `cfg`. It validates the final resolved
/// architecture, model shape, routing configuration, storage/cache
/// relationships, and the integer conversions the loader will subsequently
/// perform.
///
/// It deliberately does **not** reuse the generic [`crate::config::Config`]
/// validation's universal `num_heads * head_dim == d_model` rule: that
/// identity is invalid for several supported architectures (MLA decomposes
/// Q/K/V through low-rank latents, and MiMo-V2-Flash uses an asymmetric
/// `v_head_dim != head_dim` with a partial-rotary Q/K width), so applying it
/// would reject valid checkpoints.
fn validate_resolved_bench_real_config(
    cfg: &crate::config::Config,
    arch: crate::architecture::Architecture,
    first_k_dense_replace: usize,
    advanced: &crate::model::AdvancedConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut errs: Vec<String> = Vec::new();

    let d_model = cfg.model.d_model;
    let d_ff = cfg.model.d_ff;
    let num_layers = cfg.model.num_layers;
    let num_experts = cfg.model.num_experts as usize;
    let top_k = cfg.model.top_k;
    let vocab_size = cfg.real_transformer.vocab_size;
    let num_heads = cfg.real_transformer.num_heads;
    let num_kv_heads = cfg.real_transformer.num_kv_heads;
    let head_dim = cfg.real_transformer.head_dim;
    let v_head_dim = advanced.v_head_dim.unwrap_or(head_dim);

    // ---- Resolved model shape ----
    if d_model == 0 {
        errs.push("resolved d_model is 0".to_string());
    }
    if d_ff == 0 {
        errs.push("resolved d_ff is 0".to_string());
    }
    if num_layers == 0 {
        errs.push("resolved num_layers is 0".to_string());
    }
    if vocab_size == 0 {
        errs.push("resolved vocab_size is 0".to_string());
    }
    if num_heads == 0 {
        errs.push("resolved num_heads is 0".to_string());
    }
    if head_dim == 0 {
        errs.push("resolved head_dim is 0".to_string());
    }
    if v_head_dim == 0 {
        errs.push("resolved v_head_dim is 0".to_string());
    }

    // ---- Architecture-aware attention head geometry ----
    // MLA carries its own low-rank projection dims and does not obey the
    // dense GQA head arithmetic; validate the GQA invariants only for the
    // standard dense-attention families.
    if advanced.mla.is_none() {
        if num_kv_heads == 0 {
            errs.push("resolved num_kv_heads is 0".to_string());
        } else if num_heads % num_kv_heads != 0 {
            errs.push(format!(
                "grouped-query attention requires num_heads ({num_heads}) to be a multiple of \
                 num_kv_heads ({num_kv_heads})"
            ));
        }
        // A separate SWA KV-head count (MiMo-V2-Flash) must divide num_heads too.
        if let Some(swa_kv) = advanced.swa_num_key_value_heads {
            if swa_kv == 0 || num_heads % swa_kv != 0 {
                errs.push(format!(
                    "swa_num_key_value_heads ({swa_kv}) must be a non-zero divisor of num_heads \
                     ({num_heads})"
                ));
            }
        }
    }

    // ---- Routing configuration ----
    if num_experts == 0 {
        errs.push("resolved num_experts is 0".to_string());
    }
    if top_k == 0 {
        errs.push("resolved top_k is 0".to_string());
    }
    if top_k > num_experts {
        errs.push(format!(
            "top_k ({top_k}) exceeds num_experts ({num_experts})"
        ));
    }
    if first_k_dense_replace > num_layers {
        errs.push(format!(
            "first_k_dense_replace ({first_k_dense_replace}) exceeds num_layers ({num_layers})"
        ));
    }
    // Group-limited routing (DeepSeek-V3 `n_group`/`topk_group`).
    if advanced.n_group > 1 {
        if num_experts % advanced.n_group != 0 {
            errs.push(format!(
                "group-limited routing requires num_experts ({num_experts}) to be a multiple of \
                 n_group ({})",
                advanced.n_group
            ));
        }
        if advanced.topk_group == 0 || advanced.topk_group > advanced.n_group {
            errs.push(format!(
                "topk_group ({}) must be in 1..=n_group ({})",
                advanced.topk_group, advanced.n_group
            ));
        }
    }

    // ---- Storage / cache relationships ----
    let cache_slots = cfg.storage.cache_slots;
    let block_align = cfg.storage.block_align;
    let expert_size = cfg.model.expert_size;
    // The per-layer LRU must be able to hold every expert activated for a
    // single token (the routed top-K plus the always-on shared experts);
    // otherwise a single decode step thrashes and cannot make progress.
    let activated_per_layer = top_k.saturating_add(advanced.num_shared_experts);
    if cache_slots < activated_per_layer {
        errs.push(format!(
            "storage.cache_slots ({cache_slots}) is smaller than the experts activated per layer \
             (top_k {top_k} + shared {} = {activated_per_layer}); the cache cannot hold one \
             layer's working set",
            advanced.num_shared_experts
        ));
    }
    if expert_size == 0 {
        errs.push("model.expert_size is 0".to_string());
    }
    if !cfg.storage.no_direct {
        if block_align == 0 {
            errs.push("storage.block_align is 0 while O_DIRECT is enabled".to_string());
        } else if expert_size % block_align != 0 {
            errs.push(format!(
                "model.expert_size ({expert_size}) must be a multiple of storage.block_align \
                 ({block_align}) for O_DIRECT reads (set storage.no_direct = true to relax)"
            ));
        }
    }

    // ---- Integer conversions the loader will perform ----
    // These products index/allocate resident tensors; a usize overflow here
    // would silently wrap during loading. Validate they are representable.
    let q_dim = num_heads.checked_mul(head_dim);
    if q_dim.is_none() {
        errs.push(format!(
            "num_heads ({num_heads}) * head_dim ({head_dim}) overflows usize"
        ));
    }
    if d_model.checked_mul(vocab_size).is_none() {
        errs.push(format!(
            "d_model ({d_model}) * vocab_size ({vocab_size}) overflows usize (embedding size)"
        ));
    }
    if let Some(q) = q_dim {
        if q.checked_mul(d_model).is_none() {
            errs.push("q_proj element count overflows usize".to_string());
        }
    }
    if d_model.checked_mul(d_ff).is_none() {
        errs.push(format!(
            "d_model ({d_model}) * d_ff ({d_ff}) overflows usize (dense FFN size)"
        ));
    }
    if num_experts > u32::MAX as usize {
        errs.push(format!(
            "num_experts ({num_experts}) does not fit in u32"
        ));
    }

    if errs.is_empty() {
        info!(
            architecture = %arch.model_type(),
            d_model,
            num_layers,
            num_experts,
            top_k,
            "bench-real resolved configuration passed architecture-aware validation"
        );
        Ok(())
    } else {
        Err(format!(
            "resolved bench-real configuration for architecture \"{}\" failed validation:\n  - {}",
            arch.model_type(),
            errs.join("\n  - ")
        )
        .into())
    }
}

async fn run_bench_real_once(
    runtime: &BenchRealRuntime,
    prompt: &str,
    output_tokens: usize,
    params: crate::sampling::SamplingParams,
    run_index: usize,
) -> Result<BenchRealRunReport, Box<dyn std::error::Error>> {
    let prompt_ids = runtime.tokenizer.encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err("bench-real prompt encoded to zero tokens".into());
    }
    let stage_timings = crate::stage_timing::StageTimings::default();
    let mut kv = runtime.model.fresh_kv_caches();
    let pre = runtime.engine.report();
    let total_started = Instant::now();
    let prompt_started = Instant::now();
    let mut pos = 0usize;
    let mut forward_evaluations = 0usize;
    let mut lm_head_evaluations = 0usize;
    let mut completion_ids = Vec::with_capacity(output_tokens);
    let mut decode_latencies_us = Vec::with_capacity(output_tokens.saturating_sub(1));

    for &tid in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
        let _ = runtime
            .model
            .forward_token_hidden_with_timing(
                &runtime.engine,
                tid,
                pos,
                &mut kv,
                Some(&stage_timings),
            )
            .await;
        forward_evaluations += 1;
        pos += 1;
    }

    let final_prompt = *prompt_ids.last().expect("prompt_ids checked non-empty");
    let final_prompt_pos = pos;
    let first_started = Instant::now();
    let final_hidden = runtime
        .model
        .forward_token_hidden_with_timing(
            &runtime.engine,
            final_prompt,
            final_prompt_pos,
            &mut kv,
            Some(&stage_timings),
        )
        .await;
    forward_evaluations += 1;
    pos += 1;
    let prompt_elapsed = prompt_started.elapsed();
    stage_timings.record(crate::stage_timing::TOTAL_PROMPT, prompt_elapsed);
    let prompt_seconds = prompt_elapsed.as_secs_f64();
    let first = runtime.model.sample_hidden_with_timing(
        &final_hidden,
        &params,
        final_prompt_pos,
        Some(&stage_timings),
    );
    lm_head_evaluations += 1;
    let _first_token_latency_us = first_started.elapsed().as_micros() as u64;
    let time_to_first_token_seconds = total_started.elapsed().as_secs_f64();
    completion_ids.push(first);

    let decode_started = Instant::now();
    let mut last = first;
    while completion_ids.len() < output_tokens {
        let step_started = Instant::now();
        let next = runtime
            .model
            .decode_step_with_timing(
                &runtime.engine,
                last,
                pos,
                &mut kv,
                &params,
                Some(&stage_timings),
            )
            .await;
        forward_evaluations += 1;
        lm_head_evaluations += 1;
        decode_latencies_us.push(step_started.elapsed().as_micros() as u64);
        completion_ids.push(next);
        last = next;
        pos += 1;
    }
    let decode_elapsed = decode_started.elapsed();
    stage_timings.record(crate::stage_timing::TOTAL_DECODE, decode_elapsed);
    let decode_seconds = decode_elapsed.as_secs_f64();
    let total_seconds = total_started.elapsed().as_secs_f64();
    debug_assert_eq!(
        forward_evaluations,
        bench_real_expected_forward_evaluations(prompt_ids.len(), output_tokens)
    );
    debug_assert_eq!(lm_head_evaluations, output_tokens);

    let post = runtime.engine.report();
    let cache_hits = post.hits.saturating_sub(pre.hits);
    let cache_misses = post.misses.saturating_sub(pre.misses);
    let total_lookups = cache_hits + cache_misses;
    let hit_rate = if total_lookups == 0 {
        0.0
    } else {
        cache_hits as f64 / total_lookups as f64
    };
    let ssd_bytes = post.bytes_read.saturating_sub(pre.bytes_read);
    let ssd_stall_us = post
        .predictive
        .ssd_stall_us
        .saturating_sub(pre.predictive.ssd_stall_us);
    decode_latencies_us.sort_unstable();
    let output_text = runtime.tokenizer.decode(&completion_ids)?;
    let stage_timings = stage_timings.snapshot();

    Ok(BenchRealRunReport {
        run_index,
        prompt_tokens: prompt_ids.len(),
        completion_tokens: output_tokens,
        total_api_tokens: prompt_ids.len() + output_tokens,
        model_forward_evaluations: forward_evaluations,
        lm_head_evaluations,
        prompt_seconds,
        prompt_tps: rate_per_second(prompt_ids.len(), prompt_seconds),
        decode_seconds,
        decode_tps: rate_per_second(output_tokens.saturating_sub(1), decode_seconds),
        time_to_first_token_seconds,
        total_seconds,
        decode_token_latency_p50_ms: percentile_us_to_ms(&decode_latencies_us, 0.50),
        decode_token_latency_p95_ms: percentile_us_to_ms(&decode_latencies_us, 0.95),
        decode_token_latency_p99_ms: percentile_us_to_ms(&decode_latencies_us, 0.99),
        decode_token_latency_max_ms: decode_latencies_us.last().copied().unwrap_or(0) as f64
            / 1000.0,
        cache_hits,
        cache_misses,
        hit_rate,
        ssd_bytes,
        ssd_stall_seconds: ssd_stall_us as f64 / 1_000_000.0,
        rss_bytes: current_rss_bytes(),
        output_token_ids: completion_ids,
        output_text,
        stage_timings,
    })
}

fn emit_bench_real_report(
    args: &BenchRealArgs,
    input: BenchRealInput,
    runs: Vec<BenchRealRunReport>,
) -> Result<(), Box<dyn std::error::Error>> {
    let suite = BenchRealSuiteReport {
        benchmark: "bench-real",
        config: args.config.display().to_string(),
        prompt: input.prompt,
        warmup_runs: args.warmup_runs,
        measured_runs: args.measured_runs,
        cache_reset: args.cache_reset,
        greedy: args.greedy,
        build: BenchRealBuildInfo {
            git_commit: git_commit_short(),
            build_features: build_features(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            dense_matvec_backend: crate::parallel::dense_matvec_backend().to_string(),
        },
        aggregate: aggregate_bench_real(&runs),
        runs,
    };
    match args.format {
        BenchRealOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&suite)?);
        }
        BenchRealOutputFormat::Human => print_bench_real_human(&suite),
    }
    Ok(())
}

fn print_bench_real_human(suite: &BenchRealSuiteReport) {
    println!("bench-real");
    println!("  config: {}", suite.config);
    println!(
        "  warmup_runs={} measured_runs={} cache_reset={:?} greedy={}",
        suite.warmup_runs, suite.measured_runs, suite.cache_reset, suite.greedy
    );
    println!(
        "  build: git={} threads={} dense_matvec_backend={} features={}",
        suite.build.git_commit,
        suite.build.threads,
        suite.build.dense_matvec_backend,
        suite.build.build_features.join(",")
    );
    for run in &suite.runs {
        println!(
            "  run {}: prompt_tokens={} completion_tokens={} forwards={} lm_heads={}",
            run.run_index,
            run.prompt_tokens,
            run.completion_tokens,
            run.model_forward_evaluations,
            run.lm_head_evaluations
        );
        println!(
            "    prompt={:.3}s ({:.3} tok/s) ttft={:.3}s decode={:.3}s ({:.3} tok/s) total={:.3}s",
            run.prompt_seconds,
            run.prompt_tps,
            run.time_to_first_token_seconds,
            run.decode_seconds,
            run.decode_tps,
            run.total_seconds
        );
        println!(
            "    decode latency: p50={:.3}ms p95={:.3}ms p99={:.3}ms max={:.3}ms",
            run.decode_token_latency_p50_ms,
            run.decode_token_latency_p95_ms,
            run.decode_token_latency_p99_ms,
            run.decode_token_latency_max_ms
        );
        println!(
            "    cache: hits={} misses={} hit_rate={:.2}% ssd_bytes={} ssd_stall={:.3}s rss={}",
            run.cache_hits,
            run.cache_misses,
            run.hit_rate * 100.0,
            run.ssd_bytes,
            run.ssd_stall_seconds,
            run.rss_bytes
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        if run.stage_timings.is_empty() {
            println!("    stage timing: unavailable until stage-level timers are enabled");
        } else {
            println!("    stage timing:");
            for (stage, timing) in &run.stage_timings {
                println!(
                    "      {:<32} total={:.6}s count={} mean={:.6}s max={:.6}s",
                    stage,
                    timing.total_seconds,
                    timing.count,
                    timing.mean_seconds,
                    timing.max_seconds
                );
            }
        }
    }
    println!(
        "  aggregate: prompt_tps_mean={:.3} decode_tps_mean={:.3} ttft_p50={:.3}s total_mean={:.3}s parity={}",
        suite.aggregate.prompt_tps_mean,
        suite.aggregate.decode_tps_mean,
        suite.aggregate.time_to_first_token_p50_seconds,
        suite.aggregate.total_seconds_mean,
        suite.aggregate.output_token_parity
    );
}

fn aggregate_bench_real(runs: &[BenchRealRunReport]) -> BenchRealAggregate {
    let n = runs.len().max(1) as f64;
    let cache_hits_total = runs.iter().map(|r| r.cache_hits).sum();
    let cache_misses_total = runs.iter().map(|r| r.cache_misses).sum();
    let total_lookups = cache_hits_total + cache_misses_total;
    let hit_rate = if total_lookups == 0 {
        0.0
    } else {
        cache_hits_total as f64 / total_lookups as f64
    };
    let mut ttft_us: Vec<u64> = runs
        .iter()
        .map(|r| (r.time_to_first_token_seconds * 1_000_000.0).round() as u64)
        .collect();
    ttft_us.sort_unstable();
    let output_token_parity = runs
        .windows(2)
        .all(|pair| pair[0].output_token_ids == pair[1].output_token_ids);
    BenchRealAggregate {
        prompt_seconds_mean: runs.iter().map(|r| r.prompt_seconds).sum::<f64>() / n,
        prompt_tps_mean: runs.iter().map(|r| r.prompt_tps).sum::<f64>() / n,
        decode_seconds_mean: runs.iter().map(|r| r.decode_seconds).sum::<f64>() / n,
        decode_tps_mean: runs.iter().map(|r| r.decode_tps).sum::<f64>() / n,
        time_to_first_token_p50_seconds: percentile_us(&ttft_us, 0.50) as f64 / 1_000_000.0,
        total_seconds_mean: runs.iter().map(|r| r.total_seconds).sum::<f64>() / n,
        cache_hits_total,
        cache_misses_total,
        hit_rate,
        ssd_bytes_total: runs.iter().map(|r| r.ssd_bytes).sum(),
        output_token_parity,
    }
}

fn bench_real_expected_forward_evaluations(
    prompt_tokens: usize,
    completion_tokens: usize,
) -> usize {
    if prompt_tokens == 0 || completion_tokens == 0 {
        0
    } else {
        prompt_tokens + completion_tokens - 1
    }
}

fn rate_per_second(count: usize, seconds: f64) -> f64 {
    if count == 0 || seconds <= 0.0 {
        0.0
    } else {
        count as f64 / seconds
    }
}

fn percentile_us_to_ms(sorted_us: &[u64], q: f64) -> f64 {
    percentile_us(sorted_us, q) as f64 / 1000.0
}

fn percentile_us(sorted_us: &[u64], q: f64) -> u64 {
    if sorted_us.is_empty() {
        return 0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted_us.len() - 1) as f64 * q).round() as usize;
    sorted_us[idx]
}

fn build_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "tokenizer") {
        features.push("tokenizer");
    }
    if cfg!(feature = "io_uring") {
        features.push("io_uring");
    }
    if cfg!(feature = "blas") {
        features.push("blas");
    }
    if cfg!(feature = "alloc-count") {
        features.push("alloc-count");
    }
    if cfg!(feature = "avx512") {
        features.push("avx512");
    }
    if cfg!(feature = "amx") {
        features.push("amx");
    }
    if cfg!(feature = "nightly-amx") {
        features.push("nightly-amx");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "tui") {
        features.push("tui");
    }
    if cfg!(feature = "grpc") {
        features.push("grpc");
    }
    features
}

fn git_commit_short() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let body = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kib.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let kib: u64 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .ok()?;
        Some(kib.saturating_mul(1024))
    }
}

/// Resolve the serving tokenizer, enforcing the real-vs-synthetic policy
/// (Finding 4).
///
/// * `real_enabled` — `real_transformer.enabled`.
/// * `tokenizer_path` — configured `tokenizer.path`, if any.
/// * `model_vocab_size` — reconciled model vocabulary, when a real model
///   is present, used to validate tokenizer/model compatibility.
///
/// When a real transformer is enabled, a real tokenizer is mandatory: a
/// missing path or a load failure is fatal, the byte fallback is never
/// used, and every emittable token id must be `< model_vocab_size`. When
/// the real transformer is disabled, the byte-level fallback is preserved
/// for the synthetic/legacy path.
fn resolve_serving_tokenizer(
    real_enabled: bool,
    tokenizer_path: Option<&std::path::Path>,
    model_vocab_size: Option<usize>,
) -> Result<Arc<crate::tokenizer::Tokenizer>, crate::config::ConfigError> {
    use crate::config::ConfigError;
    use crate::tokenizer::Tokenizer;

    if real_enabled {
        let path = tokenizer_path.ok_or_else(|| {
            ConfigError::Invalid(
                "real_transformer.enabled requires tokenizer.path; the byte-level fallback \
                 tokenizer is not valid for real-checkpoint inference"
                    .to_string(),
            )
        })?;
        let tok = Tokenizer::from_file(path).map_err(|e| {
            ConfigError::Invalid(format!(
                "failed to load tokenizer {} for real-transformer serving: {e}",
                path.display()
            ))
        })?;
        if let Some(vocab) = model_vocab_size {
            tok.validate_vocab_compat(vocab).map_err(|e| {
                ConfigError::Invalid(format!(
                    "tokenizer {} is incompatible with the model: {e}",
                    path.display()
                ))
            })?;
        }
        Ok(Arc::new(tok))
    } else {
        match tokenizer_path {
            Some(p) => match Tokenizer::from_file(p) {
                Ok(t) => Ok(Arc::new(t)),
                Err(e) => {
                    warn!(path = %p.display(), error = %e, "tokenizer load failed; falling back to byte tokenizer");
                    Ok(Arc::new(Tokenizer::bytes()))
                }
            },
            None => {
                info!("no tokenizer.json configured; using byte-level fallback tokenizer");
                Ok(Arc::new(Tokenizer::bytes()))
            }
        }
    }
}

async fn cmd_serve(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use crate::config::Config;
    use crate::metrics::Metrics;
    use crate::server::{serve, AppState};

    // NUMA-local pinning via `MER_PIN_CORES` is consumed centrally
    // at process start in `main()` (see `numa::apply_mer_pin_cores_env`)
    // and the env var is then cleared. We deliberately do **not**
    // re-read it here — every subcommand goes through that single
    // startup contract, so a per-subcommand re-read would be dead
    // code (gist feedback #2.3).

    let mut cfg = Config::from_file(&config_path)?;
    crate::parallel::set_dense_matvec_backend(cfg.real_transformer.dense_matvec_backend);

    // ---- Architecture resolution & hyperparameter reconciliation ----
    //
    // Resolve the model family and, when a Hugging Face `config.json` is
    // present in `weights_dir`, remap its hyperparameters onto the
    // engine-visible `[model]` / `[real_transformer]` config *before* we
    // size the expert cache, the layer-qualified expert namespace, or the
    // routing tables below. Doing this here (rather than only when the
    // `RealModel` is built) keeps a single source of truth: the engine and
    // the real model always agree on `num_layers` / `num_experts` / dims,
    // so a checkpoint never streams against a mismatched namespace.
    //
    // Precedence: explicit `[real_transformer] architecture = "…"` override
    // (exact HF `model_type`) wins; otherwise auto-detect from
    // `config.json`; otherwise default to Mixtral. An unrecognised
    // architecture is a hard error — we never silently mislabel a model.
    let mut resolved_architecture = crate::architecture::Architecture::Mixtral;
    let mut resolved_first_k_dense_replace = 0usize;
    // Advanced routing surface (DeepSeek-V3 aux-loss-free balancing:
    // sigmoid scoring, group-limited top-K, routed scaling, plus
    // `norm_topk_prob`). Reconciled from `config.json` alongside the
    // dims below so the real model's per-layer `LinearGate` is built via
    // `with_routing` with the checkpoint's actual scoring function — not
    // silently defaulted to Mixtral-style softmax. Mixtral / Qwen3-MoE
    // checkpoints map to the same values `AdvancedConfig::default()`
    // already carries, so this is behaviour-preserving for them.
    let mut resolved_advanced = crate::model::AdvancedConfig::default();
    if cfg.real_transformer.enabled {
        if let Some(arch_str) = cfg.real_transformer.architecture.clone() {
            resolved_architecture = crate::architecture::Architecture::from_model_type(&arch_str)
                .ok_or_else(|| {
                format!(
                    "[real_transformer] architecture = \"{arch_str}\" is not a recognised \
                         model_type (expected one of: mixtral, qwen3, qwen3_moe, deepseek_v3, \
                         mistral3, phi3)"
                )
            })?;
        } else if let Some(dir) = cfg.real_transformer.weights_dir.clone() {
            match crate::architecture::HfConfig::from_dir(&dir) {
                Ok(Some(hf)) => {
                    info!(
                        architecture = ?hf.architecture,
                        "config.json detected; reconciling [model] hyperparameters from checkpoint"
                    );
                    resolved_architecture = hf.architecture;
                    resolved_first_k_dense_replace = hf.first_k_dense_replace.unwrap_or(0);
                    // Map the checkpoint's routing hyperparameters (scoring
                    // function, group-limited top-K, routed scaling factor,
                    // `norm_topk_prob`) so they reach the per-layer gate.
                    resolved_advanced = crate::model::RealModelConfig::from_hf_config(&hf).advanced;
                    // GPT-OSS SwiGLU gate clamp: install the process-global
                    // limit so the expert-FFN hot path applies it (no-op for
                    // every architecture that leaves `swiglu_limit` unset).
                    crate::inference::set_swiglu_limit(resolved_advanced.swiglu_limit);
                    // Engine-visible dims (cache + expert namespace + router).
                    cfg.model.d_model = hf.hidden_size;
                    cfg.model.d_ff = hf.resolved_d_ff();
                    cfg.model.num_layers = hf.num_hidden_layers;
                    cfg.model.num_experts = hf.num_routed_experts.unwrap_or(1).max(1) as u32;
                    cfg.model.top_k = hf
                        .num_experts_per_tok
                        .unwrap_or(1)
                        .clamp(1, cfg.model.num_experts.max(1) as usize);
                    // Attention / norm hyperparameters consumed when the
                    // `RealModelConfig` is built further below.
                    cfg.real_transformer.vocab_size = hf.vocab_size;
                    cfg.real_transformer.num_heads = hf.num_attention_heads;
                    cfg.real_transformer.num_kv_heads = if hf.num_key_value_heads == 0 {
                        hf.num_attention_heads
                    } else {
                        hf.num_key_value_heads
                    };
                    cfg.real_transformer.head_dim = hf.resolved_head_dim();
                    cfg.real_transformer.rope_base = hf.rope_theta;
                    cfg.real_transformer.rms_eps = hf.rms_norm_eps;
                    cfg.real_transformer.window_size = hf.sliding_window.unwrap_or(0);
                }
                Ok(None) => {} // no config.json — keep TOML-derived config
                Err(e) => {
                    return Err(
                        format!("failed to read config.json from {}: {e}", dir.display()).into(),
                    );
                }
            }
        }
    }

    info!(
        bind = %cfg.server.bind,
        data_dir = %cfg.model.data_dir.display(),
        num_experts = cfg.model.num_experts,
        num_layers = cfg.model.num_layers,
        top_k = cfg.model.top_k,
        d_model = cfg.model.d_model,
        d_ff = cfg.model.d_ff,
        "loaded server config"
    );

    // Hybrid compute offload (gist Part 2, fix #6). Selects which
    // `Backend` instance owns the dense transformer body; runs
    // *before* `install_default` so the OnceLock keeps our pointer.
    // The startup log below reports the actual device runtime
    // (`cpu-fallback` / `cuda-0` / `wgpu-vulkan`) as `GpuBackend::name`
    // surfaces it — no more stale hardcoded `"gpu-fallback"` strings.
    //
    // The GPU expert cache is constructed up-front so the same `Arc`
    // can be threaded into both `GpuBackend` (which checks VRAM
    // residency before falling back to NVMe streaming) and
    // `Engine::install_gpu_cache` further below. When
    // `[gpu_cache].enabled = false` we still allocate a zero-capacity
    // cache to satisfy the `BackendBox::init_blocking` signature —
    // the cache simply never promotes anything in that mode.
    let gpu_expert_cache = {
        let capacity_bytes = if cfg.gpu_cache.enabled {
            (cfg.gpu_cache.vram_capacity_mb as usize) * 1024 * 1024
        } else {
            0
        };
        std::sync::Arc::new(crate::expert_cache::GpuExpertCache::new(
            capacity_bytes,
            cfg.gpu_cache.vram_anchor_ratio,
            cfg.gpu_cache.promote_after_hits,
        ))
    };
    let mut backend_fallback_occurred = false;
    if matches!(
        cfg.real_transformer.compute_offload,
        crate::backend::ComputeOffload::Gpu | crate::backend::ComputeOffload::Auto
    ) {
        let num_layers = cfg.model.num_layers;
        let max_seq_len = if cfg.real_transformer.window_size == 0 {
            4096
        } else {
            cfg.real_transformer.window_size
        };
        let num_heads = cfg.real_transformer.num_heads;
        let num_kv_heads = if cfg.real_transformer.num_kv_heads == 0 {
            num_heads
        } else {
            cfg.real_transformer.num_kv_heads
        };
        let head_dim = if cfg.real_transformer.head_dim == 0 {
            if num_heads > 0 {
                cfg.model.d_model / num_heads
            } else {
                64
            }
        } else {
            cfg.real_transformer.head_dim
        };
        let backend_box = crate::backend::BackendBox::init_blocking(
            num_layers,
            max_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            // Finding 12: the run path does not surface an asymmetric
            // `v_head_dim` here; pass 0 (auto → head_dim). Asymmetric-V
            // models force the CPU attention path via the eligibility guard
            // in transformer.rs, so the symmetric GPU sizing is never used
            // for them.
            0,
            gpu_expert_cache.clone(),
        );
        let has_device = backend_box.is_gpu();
        // Reconcile the operator's request with the observed init result
        // (Finding 5). An explicit `gpu` request fails closed; `auto`
        // demotes to CPU and records a fallback. `is_gpu()` is the signal
        // because `init_blocking` internally swallows GPU errors and returns
        // a CPU backend.
        let resolution = crate::backend::resolve_backend_selection(
            cfg.real_transformer.compute_offload,
            || {
                if has_device {
                    Ok(())
                } else {
                    Err("GPU backend initialization did not produce a GPU device".to_string())
                }
            },
        )?;
        backend_fallback_occurred = resolution.fallback_occurred;
        let device_name = backend_box.device_name().to_string();
        let compute_plane = backend_box.compute_plane().to_string();
        if resolution.fallback_occurred {
            warn!(
                requested = ?resolution.requested,
                resolved = ?resolution.resolved,
                "compute_offload = \"auto\": GPU initialization failed; resolved to CPU"
            );
        }
        let gpu = std::sync::Arc::new(backend_box);
        if let Err(e) = crate::backend::set_backend(gpu) {
            warn!(
                error = e,
                "failed to install GpuBackend; falling back to default"
            );
        } else {
            info!(
                device = device_name,
                compute_plane,
                has_device,
                requested = ?resolution.requested,
                resolved = ?resolution.resolved,
                "math backend installed for dense backbone"
            );
        }
    }
    crate::backend::install_default();
    {
        let b = crate::backend::current();
        info!(
            backend = b.device_name(),
            compute_plane = b.compute_plane(),
            compute_offload = ?cfg.real_transformer.compute_offload,
            "math backend installed"
        );
    }

    if !cfg.model.data_dir.is_dir() {
        return Err(format!(
            "data dir {} does not exist; run `gen-data` or the extractor first",
            cfg.model.data_dir.display()
        )
        .into());
    }

    // Wire the multi-layer extractor naming when num_layers > 1, so
    // either `expert_<id>.bin` or `expert_<layer>_<local>.bin` works.
    let storage = NvmeStorage::new(StorageConfig {
        base_path: cfg.model.data_dir.clone(),
        expert_size: cfg.model.expert_size,
        block_align: cfg.storage.block_align,
        use_direct_io: !cfg.storage.no_direct,
        num_experts_per_layer: if cfg.model.num_layers > 1 {
            Some(cfg.model.num_experts)
        } else {
            None
        },
    })?;
    // Tier 2: attach the packed blob if configured (defaults: no-op).
    let storage = maybe_attach_packed_blob(
        storage,
        cfg.storage.packed_blob.as_deref(),
        cfg.storage.packed_manifest.as_deref(),
        !cfg.storage.no_direct,
        cfg.model.expert_size,
    )?;
    let storage = Arc::new(storage);
    // Warm fds across the whole multi-layer namespace (one global id
    // per (layer, local_expert) pair) so the steady-state path never
    // pays the open() cost. Skipped in packed mode: every expert is
    // served from the single already-open blob fd, and the per-expert
    // files may not even exist on disk.
    let total_experts = (cfg.model.num_layers as u32).saturating_mul(cfg.model.num_experts);
    if !storage.is_packed() {
        storage.warmup_fds(0..total_experts)?;
    }

    // Double-buffered pool (Parts 1–2): split the RAM buffers into a
    // **primary** (Buffer A) half that backs the resident LRU plus one
    // reserved slot the foreground miss path is always guaranteed, and a
    // **shadow** (Buffer B) half that backs speculative look-ahead
    // prefetches. The shadow half is sized to the prefetch fanout scaled
    // by the look-ahead `pipeline_depth` (`predict_fanout * pipeline_depth`)
    // so a depth-N windowed look-ahead (`speculate_layer_ahead` priming
    // `layer + 1 ..= layer + pipeline_depth`) has a buffer per in-flight
    // layer and can never steal the buffer a real cache miss needs. The
    // prefetch semaphore is derived from this shadow capacity in
    // `Engine::with_options`, so it scales automatically. A fanout of 0
    // disables Buffer B and the engine falls back to the legacy
    // single-pool prefetch path.
    let pipeline_depth = cfg.storage.pipeline_depth.max(1) as usize;
    let shadow_slots = cfg.storage.predict_fanout.saturating_mul(pipeline_depth);
    let primary_slots = cfg.storage.cache_slots + 1;
    let pool = if shadow_slots > 0 {
        BufferPool::new_with_shadow(
            primary_slots,
            shadow_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    } else {
        BufferPool::new(
            primary_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    };
    let cache = {
        let num_layers = cfg.model.num_layers.max(1);
        let per_layer = cfg.model.num_experts.max(1) as u32;
        // Split the configured residency budget across layers so the
        // *aggregate* capacity matches the operator's `cache_slots`
        // setting. Layers each get a fair share with the remainder
        // distributed to the lower-indexed layers (which tend to be
        // hotter in MoE workloads).
        let total = cfg.storage.cache_slots.max(1);
        let base = total / num_layers;
        let extra = total % num_layers;
        let caps: Vec<usize> = (0..num_layers)
            .map(|i| base + if i < extra { 1 } else { 0 })
            .collect();
        if num_layers == 1 {
            Arc::new(MultiLayerExpertCache::single_layer(total))
        } else {
            Arc::new(MultiLayerExpertCache::with_capacities(caps, per_layer))
        }
    };

    // Multi-layer addressing: the engine's expert cache uses a single
    // global namespace `(layer * num_experts_per_layer) + local`, so
    // the router / predictor / locality monitor / speculator must all
    // be sized against the *total* expert count, not the per-layer
    // count. Otherwise layer-≥1 ids silently fall outside the
    // predictor's row table and the locality monitor's `is_hot` always
    // returns false for them.
    let total_experts: u32 = (cfg.model.num_layers as u32)
        .saturating_mul(cfg.model.num_experts)
        .max(cfg.model.num_experts);

    let predictor = Arc::new(PredictiveLoader::new(
        total_experts,
        cfg.storage.predict_fanout,
        resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts),
        0xC0FFEE,
    ));

    // Build the real transformer (if enabled) *before* the engine so
    // its per-layer `LinearGate` can be wired into the engine as the
    // production routing path. When `[real_transformer].enabled =
    // true`, the engine holds `Router::Linear` from the loaded
    // model's first-layer gate — that is the path
    // `Engine::generate` will exercise on the benchmark / warmup
    // surfaces. The per-token `RealModel::step` loop in serve mode
    // routes each MoE layer through *its own* layer-local gate
    // (`TransformerLayer::moe_pre`) and calls `engine.moe_step` with
    // the already-routed ids, so the engine's router does not
    // override per-layer routing — but it does mean the engine's
    // self-reported `num_experts` / `top_k` now reflect the actual
    // gate shape rather than the legacy Markov stand-in, which is
    // the gist's "wire `LinearGate` into `serve`" ask.
    let real_model: Option<Arc<crate::model::RealModel>> = if cfg.real_transformer.enabled {
        let rt = &cfg.real_transformer;
        let head_dim = if rt.head_dim == 0 {
            cfg.model.d_model / rt.num_heads
        } else {
            rt.head_dim
        };
        let num_kv_heads = if rt.num_kv_heads == 0 {
            rt.num_heads
        } else {
            rt.num_kv_heads
        };
        // Hyperparameters were already reconciled from `config.json` (when
        // present) at the top of `cmd_serve`, so `cfg.model` /
        // `cfg.real_transformer` are the single source of truth here. We
        // just stamp the resolved architecture + dense/MoE boundary onto
        // the `RealModelConfig`. Recognised-but-unrunnable families
        // (DeepSeek-V3: MLA + FP8) fail loud inside `from_safetensors`.
        let model_cfg = crate::model::RealModelConfig {
            d_model: cfg.model.d_model,
            d_ff: cfg.model.d_ff,
            num_heads: rt.num_heads,
            num_kv_heads,
            head_dim,
            vocab_size: rt.vocab_size,
            num_layers: cfg.model.num_layers,
            num_experts: cfg.model.num_experts as usize,
            top_k: cfg.model.top_k,
            rope_base: rt.rope_base,
            rms_eps: rt.rms_eps,
            window_size: if rt.window_size == 0 {
                None
            } else {
                Some(rt.window_size)
            },
            architecture: resolved_architecture,
            first_k_dense_replace: resolved_first_k_dense_replace,
            advanced: resolved_advanced,
        };
        let load_options = crate::model::RealModelLoadOptions {
            strict_weights: rt.strict_weights,
        };
        // Fail-closed real-model weight policy (gist Finding 1). A missing
        // `weights_dir` or `strict_weights = false` is rejected unless the
        // operator explicitly opts into the development seeded fallback.
        let weight_policy = rt.resolve_weight_policy()?;
        let m = match (rt.weights_dir.as_ref(), weight_policy) {
            (Some(dir), _) => {
                let loaded = crate::model::RealModel::from_dir_auto_with_options(
                    model_cfg,
                    dir,
                    rt.seed,
                    load_options,
                )?;
                if loaded.load_status.seeded_fallback_remained {
                    warn!(
                        loader = loaded.load_status.loader,
                        loaded_tensors = loaded.load_status.loaded_tensors,
                        required_tensors = loaded.load_status.required_tensors,
                        "DEVELOPMENT FALLBACK: real_transformer served with seeded weights \
                         remaining — output is NOT real-checkpoint inference"
                    );
                }
                loaded
            }
            (None, crate::config::RealWeightPolicy::SeededDev) => {
                warn!(
                    "DEVELOPMENT FALLBACK: real_transformer.enabled with no weights_dir and \
                     allow_seeded_fallback = true — serving deterministic SEEDED weights, \
                     output is NOT real-checkpoint inference"
                );
                crate::model::RealModel::new_seeded(model_cfg, rt.seed)
            }
            // resolve_weight_policy already rejects (None, StrictReal).
            (None, crate::config::RealWeightPolicy::StrictReal) => unreachable!(
                "resolve_weight_policy rejects a missing weights_dir without seeded fallback"
            ),
        };
        info!(
            strict = m.load_status.strict,
            loader = m.load_status.loader,
            loaded_tensors = m.load_status.loaded_tensors,
            required_tensors = m.load_status.required_tensors,
            seeded_fallback_remained = m.load_status.seeded_fallback_remained,
            "real transformer model-loading status"
        );
        Some(Arc::new(m))
    } else {
        None
    };

    // Build draft engine for speculative decoding when the speculator is
    // enabled and a real model is available. `DraftEngine::from_main`
    // avoids loading any extra weights from disk, but it currently
    // **clones** the main model's embedding into a fresh `Arc<Vec<f32>>`
    // rather than sharing the `RealModel`'s allocation, so enabling this
    // path costs one additional `vocab_size * d_model * 4` bytes of
    // resident memory. See `draft::DraftEngine::from_main` for the exact
    // allocation site.
    let draft_engine: Option<Arc<crate::draft::DraftEngine>> = if cfg.predictive.speculator_enabled
    {
        real_model.as_ref().map(|m| {
            let d = crate::draft::DraftEngine::from_main(m);
            tracing::info!(
                vocab_size = m.config.vocab_size,
                d_model = m.config.d_model,
                "draft engine built for speculative decoding"
            );
            Arc::new(d)
        })
    } else {
        None
    };

    let speculation_k = cfg.real_transformer.speculation_base_depth.max(1);

    let router = if let Some(ref m) = real_model {
        // Production routing path: the engine's `route()` runs the
        // first layer's `softmax(W_gate · x) → top-K` (Mixtral-style)
        // instead of the legacy deterministic Markov chain. Per-layer
        // gates still drive per-layer routing inside `RealModel::step`
        // — this engine-level gate is what `Engine::generate` and
        // anything else that asks the engine for a routing decision
        // sees.
        info!(
            num_experts = m.layers[0].gate.num_experts,
            d_model = m.layers[0].gate.d_model,
            top_k = m.layers[0].gate.top_k,
            "engine routing: LinearGate (production softmax-gated path) wired from real model"
        );
        crate::gating::Router::Linear(Arc::new(m.layers[0].gate.clone()))
    } else {
        info!(
            total_experts,
            clusters = 4,
            "engine routing: clustered Markov chain (no real model loaded)"
        );
        crate::gating::Router::Markov(Arc::new(TopKRouter::clustered(
            total_experts,
            cfg.model.top_k,
            4,
            0.9,
            0xC0FFEE,
        )))
    };

    let metrics = Metrics::new();
    // Surface any auto GPU->CPU demotion decided above now that metrics
    // exist (Finding 5).
    if backend_fallback_occurred {
        metrics.record_gpu_cpu_fallback(1);
    }
    let mut engine_builder = Engine::with_options(
        cache,
        pool,
        storage,
        router,
        predictor,
        ModelShape {
            d_model: cfg.model.d_model,
            d_ff: cfg.model.d_ff,
            hidden_seed: 0xC0FFEE,
        },
        EngineOptions {
            io_only: false,
            dtype: cfg.model.dtype,
            partial_load_fraction: cfg.storage.partial_load_fraction,
            pin_after_observations: cfg.storage.pin_after_observations,
            use_qmm_for_q4: true,
            expert_execution_policy: cfg.real_transformer.expert_execution_policy,
            max_concurrent_prefetches: cfg.real_transformer.max_concurrent_prefetches,
            max_fetch_yields: cfg.real_transformer.max_fetch_yields,
            prefetch_governor: cfg.predictive.prefetch_governor,
            prefetch_precision_floor: cfg.predictive.prefetch_precision_floor,
            prefetch_contention_weight: cfg.predictive.prefetch_contention_weight,
            cost_aware_eviction: cfg.predictive.cost_aware_eviction,
            pregate_enabled: cfg.predictive.pregate_enabled,
            collect_route_profile: false,
        },
    );
    // Apply the configured look-ahead pipeline depth (`[storage]
    // pipeline_depth`). Controls how many layers ahead
    // `speculate_layer_ahead` primes; sized in tandem with the shadow
    // buffer-pool budget above.
    engine_builder = engine_builder.with_pipeline_depth(cfg.storage.pipeline_depth);
    // Attach the speculative-architecture components requested via
    // the `[predictive]` config section. Sized against the global
    // expert namespace (see `total_experts` above) so multi-layer
    // models don't silently drop layer-≥1 ids on the floor.
    if cfg.predictive.locality_enabled {
        // Scale the sliding window by the layer count: with a
        // layer-qualified namespace every token contributes
        // `num_layers × top_k` activations, so a flat 256-deep window
        // only holds ~8 activations *per layer* — far too few for the
        // per-layer heat threshold (`effective_locality_threshold`,
        // which divides by the layer count) to discriminate anything.
        // Multiplying the window by the layer count keeps the
        // *per-layer* history depth equal to what the operator
        // configured for a flat namespace.
        let window = cfg
            .predictive
            .locality_window
            .saturating_mul(cfg.model.num_layers.max(1));
        let monitor = Arc::new(LocalityMonitor::new(total_experts, window));
        engine_builder =
            engine_builder.with_locality_monitor(monitor, cfg.predictive.locality_threshold_pct);
    }
    if cfg.predictive.speculator_enabled {
        let top_k = if cfg.predictive.speculator_top_k == 0 {
            cfg.model.top_k
        } else {
            cfg.predictive.speculator_top_k
        };
        let spec = Arc::new(NeuralSpeculator::new(
            cfg.model.d_model,
            cfg.predictive.speculator_hidden_dim,
            total_experts,
            0xC0FFEE,
        ));
        engine_builder = engine_builder.with_speculator(spec, top_k);
    }
    // Per-layer expert-affinity arm: tracks which experts co-fire inside
    // the same MoE layer and folds their co-fired + disk-adjacent
    // neighbours into the prefetch union. Sized in the *layer-local* id
    // namespace (one `num_experts × num_experts` matrix per layer); the
    // engine maps global ids ↔ layer-local before observing / looking up
    // neighbours. Only effective when the model exposes a
    // layer-qualified geometry (`num_experts_per_layer`).
    if cfg.predictive.affinity_enabled {
        let num_layers = cfg.model.num_layers.max(1);
        let affinity = Arc::new(LayeredExpertAffinity::new(
            num_layers,
            cfg.model.num_experts,
        ));
        engine_builder = engine_builder.with_affinity(
            affinity,
            cfg.predictive.affinity_neighbors_k,
            cfg.predictive.affinity_decay_epoch,
        );
    }
    // Tier 1 — static residency. Pin the hottest `fraction` of experts
    // permanently (from an offline profile when `static_residency_profile`
    // is set, else online after the warmup window).
    if cfg.predictive.static_residency_fraction > 0.0 {
        let profile = match cfg.predictive.static_residency_profile.as_ref() {
            Some(path) => {
                let p = crate::residency::ResidencyProfile::load_json(std::path::Path::new(path))?;
                info!(
                    path = %path,
                    experts = p.len(),
                    "loaded static-residency popularity profile"
                );
                Some(p)
            }
            None => None,
        };
        engine_builder = engine_builder.with_static_residency(
            cfg.predictive.static_residency_fraction,
            cfg.predictive.static_residency_warmup_tokens,
            profile,
        );
    }
    // Tier 3 — per-layer pre-gate. Predict + prefetch the next layer's
    // experts from the current layer's routed set on the multi-layer
    // `moe_step` path.
    if cfg.predictive.pregate_enabled {
        let pregate = Arc::new(crate::pregate::PerLayerPreGate::new(
            cfg.model.num_layers.max(1),
            cfg.model.top_k,
        ));
        engine_builder = engine_builder.with_pregate(pregate);
    }
    // Phase 2: optional VRAM (GPU) expert cache (3-tier hierarchy
    // SSD → RAM → VRAM). When `[gpu_cache].enabled = false` (default)
    // the engine retains its historical 2-tier posture.
    if cfg.gpu_cache.enabled {
        // `gpu_cache.dtype` is currently advisory — it is validated by
        // `AppConfig::validate` (so typos fail fast) and surfaced here
        // for observability, but the promotion path copies on-disk
        // bytes into VRAM without conversion or repacking. Parse it
        // here purely so the startup log records the operator's
        // declared intent.
        let dtype_for_logging = crate::inference::WeightDtype::from_str_opt(&cfg.gpu_cache.dtype)
            .unwrap_or(crate::inference::WeightDtype::F16);
        info!(
            vram_capacity_mb = cfg.gpu_cache.vram_capacity_mb,
            anchor_ratio = cfg.gpu_cache.vram_anchor_ratio,
            promote_after_hits = cfg.gpu_cache.promote_after_hits,
            dtype_advisory = %dtype_for_logging.as_str(),
            "VRAM (GPU) expert cache enabled — 3-tier SSD→RAM→VRAM hierarchy active (dtype is advisory; bytes copied as-is)"
        );
        engine_builder.install_gpu_cache(gpu_expert_cache.clone());
    }
    let engine = Arc::new(engine_builder.with_metrics(metrics.clone()));

    let tokenizer = resolve_serving_tokenizer(
        cfg.real_transformer.enabled,
        cfg.tokenizer.path.as_deref(),
        real_model.as_ref().map(|m| m.config.vocab_size),
    )?;

    // Optional real-transformer pipeline. When enabled, every request
    // runs `embedding -> stacked layers (each with SSD-streamed MoE) ->
    // LM head -> argmax`; when disabled, the legacy benchmark generator
    // is used (the engine still streams expert FFN compute either way).
    // Note: `real_model` was constructed above so its per-layer gate
    // could be wired into the engine; here we just spawn the
    // continuous-batching scheduler against the already-built model.
    let (real_model, batch_scheduler) = if let Some(model_arc) = real_model {
        let rt = &cfg.real_transformer;
        let head_dim = if rt.head_dim == 0 {
            cfg.model.d_model / rt.num_heads
        } else {
            rt.head_dim
        };
        let num_kv_heads = if rt.num_kv_heads == 0 {
            rt.num_heads
        } else {
            rt.num_kv_heads
        };
        let batch_cfg = crate::batch_scheduler::BatchConfig {
            max_batch_size: rt.max_batch_size,
            batch_timeout: std::time::Duration::from_millis(rt.batch_timeout_ms),
            idle_eviction_threshold: std::time::Duration::from_millis(
                rt.idle_eviction_threshold_ms,
            ),
            speculation_base_depth: rt.speculation_base_depth,
            // Pool back-pressure ladder is now config-driven
            // (gist Part 1, fix #4). Validation in `Config::validate`
            // already enforces 0 < high <= critical <= 1.
            pressure_thresholds: crate::block_pool::PressureThresholds::try_new(
                rt.pressure_high_threshold,
                rt.pressure_critical_threshold,
            )
            .expect("pressure thresholds validated by Config::validate")
            .with_max_overflow_capacity(rt.max_overflow_capacity),
            ..Default::default()
        };
        // Expert-placement layer: single-node default (every id
        // local), or the `[distributed]` `id % num_nodes` hash
        // partitioning over the configured mesh when enabled.
        let shard_router: std::sync::Arc<dyn crate::distributed::ShardRouter> =
            if cfg.distributed.enabled {
                let router = crate::distributed::RpcShardRouter::from_modulo_placement(
                    &cfg.distributed.nodes,
                    cfg.distributed.self_index,
                    total_experts,
                    std::time::Duration::from_millis(cfg.distributed.remote_fetch_timeout_ms),
                );
                info!(
                    nodes = cfg.distributed.nodes.len(),
                    self_index = cfg.distributed.self_index,
                    total_experts,
                    remote_fetch_timeout_ms = cfg.distributed.remote_fetch_timeout_ms,
                    "distributed expert partitioning enabled (id % num_nodes)"
                );
                std::sync::Arc::new(router)
            } else {
                std::sync::Arc::new(crate::distributed::LocalShardRouter)
            };
        let scheduler = crate::batch_scheduler::BatchScheduler::spawn_with_shard_router(
            model_arc.clone(),
            engine.clone(),
            batch_cfg,
            shard_router,
        );
        info!(
            num_layers = cfg.model.num_layers,
            num_heads = rt.num_heads,
            num_kv_heads,
            head_dim,
            vocab_size = rt.vocab_size,
            max_batch_size = rt.max_batch_size,
            batch_timeout_ms = rt.batch_timeout_ms,
            idle_eviction_threshold_ms = rt.idle_eviction_threshold_ms,
            speculation_base_depth = rt.speculation_base_depth,
            "real transformer pipeline enabled (with continuous batching)"
        );
        (Some(model_arc), Some(Arc::new(scheduler)))
    } else {
        info!("real_transformer disabled; using legacy benchmark generator");
        (None, None)
    };

    let sessions = if cfg.server.session_ttl_secs > 0 {
        let store = crate::session::SessionStore::new(std::time::Duration::from_secs(
            cfg.server.session_ttl_secs,
        ));
        // Sweep every TTL/2 (or once a minute, whichever is shorter) so
        // peak memory stays bounded but the evictor doesn't dominate
        // the wakeup budget.
        let sweep =
            std::time::Duration::from_secs((cfg.server.session_ttl_secs / 2).max(1).min(60));
        store.spawn_evictor(sweep);
        Some(store)
    } else {
        None
    };

    // Background overflow-slab reclaimer: every 60s, ask the paged-KV
    // pool to return any heap-backed overflow blocks that are no
    // longer in use. Cheap when there's nothing to reclaim (single
    // mutex check + early return), so safe to run unconditionally.
    if let Some(pool) = batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
        let pool = pool.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let reclaimed = pool.shrink_overflow_to_fit();
                if reclaimed > 0 {
                    tracing::info!(
                        reclaimed,
                        "background sweep: reclaimed paged-KV overflow blocks"
                    );
                }
            }
        });
    }

    // Build the production-readiness middleware bundle:
    //  - API-key gate (optional, off by default)
    //  - in-process token-bucket rate limit (optional, off by default)
    //  - admission controller (concurrency cap + paged-KV free-block watermark)
    use crate::middleware::{Admission, ApiKeyGate, MiddlewareState, RateLimiter};
    let api_keys = ApiKeyGate::new(&cfg.security.api_keys);
    let rate_limit = RateLimiter::new(cfg.security.rate_limit_rps, cfg.security.rate_limit_burst);
    let free_probe: Option<std::sync::Arc<dyn Fn() -> usize + Send + Sync>> =
        match batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
            Some(p) => {
                let p = p.clone();
                Some(std::sync::Arc::new(move || p.free_blocks()))
            }
            None => None,
        };
    let admission = Admission::new(
        cfg.server.max_concurrent_requests,
        cfg.server.admission_min_free_blocks,
        free_probe,
    );
    let middleware_state = MiddlewareState {
        api_keys,
        rate_limit,
        admission,
    };

    // Live, atomically-swappable runtime configuration. The hot
    // token-evaluation path reads sampling defaults and the
    // `max_tokens` cap through `runtime.snapshot()` (a single relaxed
    // atomic load — see `LiveConfig` in `crate::config`). SIGHUP
    // refreshes this in place.
    let runtime = crate::config::LiveConfig::from_config(&cfg);

    let state = AppState {
        engine,
        tokenizer,
        metrics,
        real_model,
        batch_scheduler,
        draft_engine,
        speculation_k,
        runtime: runtime.clone(),
        sessions,
        middleware: middleware_state,
    };
    // SIGHUP-triggered config reload.
    //
    // For fields covered by [`crate::config::RuntimeConfig`] (sampling
    // defaults, max-tokens cap, telemetry flags) we apply the reload
    // live via `runtime.try_reload(&new)` — an atomic `ArcSwap` store.
    // In-flight requests holding a `runtime.snapshot()` keep observing
    // their previous `Arc<RuntimeConfig>` until they drop it; concurrent
    // SIGHUPs never block on each other and never block readers.
    //
    // For restart-required fields (storage prefetch settings, batch
    // scheduler timing, etc.) we still emit a structured diff at WARN
    // level so operators know a restart is needed to fully apply the
    // file. If parsing or validation fails the in-memory runtime is
    // left **pristine** and a single `tracing::warn!` line documents
    // the rejection.
    #[cfg(unix)]
    {
        let path = config_path.clone();
        let baseline = cfg.clone();
        let runtime = runtime;
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sig = match signal(SignalKind::hangup()) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "SIGHUP handler install failed; config reload disabled");
                    return;
                }
            };
            let mut prev = baseline;
            while sig.recv().await.is_some() {
                info!("SIGHUP received; reloading config from {}", path.display());
                let new = match Config::from_file(&path) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(
                            error = %e,
                            "config reload rejected; existing runtime configuration left un-mutated",
                        );
                        continue;
                    }
                };
                // Apply the safe-to-reload subset live. `try_reload`
                // re-validates internally; on rejection it logs a
                // structured `tracing::warn!` and leaves the live
                // `ArcSwap<RuntimeConfig>` un-mutated. The atomic store
                // is contention-free with request-path readers
                // (`runtime.snapshot()` is a single relaxed atomic load).
                match runtime.try_reload(&new) {
                    Ok(rc) => info!(
                        sampling_temperature = rc.sampling.temperature,
                        sampling_top_p = rc.sampling.top_p,
                        sampling_top_k = rc.sampling.top_k,
                        max_tokens_cap = rc.max_tokens_cap,
                        "live runtime configuration swapped atomically",
                    ),
                    Err(_) => {
                        // try_reload already emitted a structured warn;
                        // skip applying restart-key diffs against an
                        // invalid file.
                        continue;
                    }
                }
                if prev.real_transformer.dense_matvec_backend
                    != new.real_transformer.dense_matvec_backend
                {
                    crate::parallel::set_dense_matvec_backend(
                        new.real_transformer.dense_matvec_backend,
                    );
                }
                // Restart-required diff: surface changes that the live
                // swap does **not** cover so operators know which fields
                // still demand a process restart.
                let restart_keys: &[(&str, String, String)] = &[
                    (
                        "storage.predict_fanout",
                        prev.storage.predict_fanout.to_string(),
                        new.storage.predict_fanout.to_string(),
                    ),
                    (
                        "real_transformer.batch_timeout_ms",
                        prev.real_transformer.batch_timeout_ms.to_string(),
                        new.real_transformer.batch_timeout_ms.to_string(),
                    ),
                    (
                        "real_transformer.idle_eviction_threshold_ms",
                        prev.real_transformer.idle_eviction_threshold_ms.to_string(),
                        new.real_transformer.idle_eviction_threshold_ms.to_string(),
                    ),
                    (
                        "real_transformer.speculation_base_depth",
                        prev.real_transformer.speculation_base_depth.to_string(),
                        new.real_transformer.speculation_base_depth.to_string(),
                    ),
                    (
                        "real_transformer.expert_execution_policy",
                        format!("{:?}", prev.real_transformer.expert_execution_policy),
                        format!("{:?}", new.real_transformer.expert_execution_policy),
                    ),
                    (
                        "storage.predict_min_prob",
                        prev.storage.predict_min_prob.to_string(),
                        new.storage.predict_min_prob.to_string(),
                    ),
                    (
                        "storage.partial_load_fraction",
                        prev.storage.partial_load_fraction.to_string(),
                        new.storage.partial_load_fraction.to_string(),
                    ),
                ];
                for (k, before, after) in restart_keys {
                    if before != after {
                        warn!(key = k, before = %before, after = %after,
                            "config changed but requires restart to take effect");
                    }
                }
                prev = new;
            }
        });
    }

    serve(state, &cfg.server.bind).await
}

fn cmd_gen_data(
    data_dir: &std::path::Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    block_align: usize,
    dtype_str: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::inference::WeightDtype;
    let dtype = WeightDtype::from_str_opt(dtype_str).ok_or_else(|| {
        format!(
            "--dtype: unknown value {dtype_str:?} (supported for gen-data: {SUPPORTED_SYNTHETIC_DTYPES})"
        )
    })?;
    if matches!(
        dtype,
        WeightDtype::Q5K | WeightDtype::Q6K | WeightDtype::Mixed
    ) {
        return Err(format!(
            "gen-data does not synthesize dtype {}; use gguf-convert --native-quant or an offline extractor for this layout",
            dtype.as_str()
        )
        .into());
    }
    if block_align == 0 || !block_align.is_power_of_two() {
        return Err(format!(
            "--block-align ({block_align}) must be a positive power of two \
             (4096 on most NVMe)."
        )
        .into());
    }
    if expert_size % block_align != 0 {
        return Err(format!(
            "--expert-size ({expert_size}) must be a multiple of --block-align \
             ({block_align}) so the run path can read each expert with O_DIRECT \
             without EINVAL."
        )
        .into());
    }
    let weight_bytes = crate::inference::expert_weight_bytes_for(d_model, d_ff, dtype);
    if weight_bytes > expert_size {
        return Err(format!(
            "expert_size ({expert_size}) is too small for the SwiGLU weights of \
             d_model={d_model}, d_ff={d_ff} dtype={} ({weight_bytes} bytes). Increase \
             --expert-size or shrink --d-model / --d-ff.",
            dtype.as_str()
        )
        .into());
    }
    info!(
        path = %data_dir.display(),
        num_experts,
        expert_size_mib = expert_size as f64 / (1024.0 * 1024.0),
        d_model,
        d_ff,
        block_align,
        dtype = dtype.as_str(),
        weight_mib = weight_bytes as f64 / (1024.0 * 1024.0),
        "generating synthetic SwiGLU expert weights"
    );
    let started = Instant::now();
    crate::io_provider::generate_synthetic_experts_with_dtype(
        data_dir,
        num_experts,
        expert_size,
        d_model,
        d_ff,
        dtype,
    )?;
    let total_bytes = num_experts as u64 * expert_size as u64;
    info!(
        elapsed_s = started.elapsed().as_secs_f64(),
        total_mib = total_bytes as f64 / (1024.0 * 1024.0),
        "expert files written"
    );
    Ok(())
}

struct RepackArgs {
    data_dir: PathBuf,
    out_blob: PathBuf,
    out_manifest: Option<PathBuf>,
    num_experts: u32,
    expert_size: usize,
    block_align: usize,
    no_direct: bool,
    num_experts_per_layer: Option<u32>,
    profile: Option<PathBuf>,
    order: Option<PathBuf>,
}

/// Parse an explicit `--order` file: either a JSON array of ids or a
/// newline / whitespace-separated list (blank lines and `#` comments
/// ignored).
fn parse_order_file(path: &std::path::Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let trimmed = raw.trim_start();
    if trimmed.starts_with('[') {
        let ids: Vec<u32> = serde_json::from_str(trimmed)?;
        return Ok(ids);
    }
    let mut ids = Vec::new();
    for line in raw.lines() {
        let without_comment = line.split_once('#').map_or(line, |(body, _)| body);
        for tok in without_comment.split(|c: char| c.is_whitespace() || c == ',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            ids.push(t.parse::<u32>()?);
        }
    }
    Ok(ids)
}

fn validate_order(ids: &[u32], num_experts: u32) -> Result<(), String> {
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    for &id in ids {
        if id >= num_experts {
            return Err(format!(
                "--order id {id} is out of range for --num-experts {num_experts}"
            ));
        }
        if !seen.insert(id) {
            return Err(format!("--order contains duplicate expert id {id}"));
        }
    }
    Ok(())
}

/// **Tier 2.** Build a packed blob + manifest from a per-expert directory.
async fn cmd_repack(args: RepackArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.block_align == 0 || !args.block_align.is_power_of_two() {
        return Err(format!(
            "--block-align ({}) must be a positive power of two",
            args.block_align
        )
        .into());
    }
    if args.expert_size % args.block_align != 0 {
        return Err(format!(
            "--expert-size ({}) must be a multiple of --block-align ({})",
            args.expert_size, args.block_align
        )
        .into());
    }
    if !args.data_dir.is_dir() {
        return Err(format!("data dir {} does not exist", args.data_dir.display()).into());
    }

    // Resolve the physical layout order.
    let order: Vec<u32> = if let Some(order_path) = &args.order {
        let ids = parse_order_file(order_path)?;
        if ids.is_empty() {
            return Err(format!("--order file {} listed no ids", order_path.display()).into());
        }
        validate_order(&ids, args.num_experts)?;
        let missing = args.num_experts as usize - ids.len();
        if missing > 0 {
            warn!(
                missing,
                "repack: explicit order omits experts; running/serving in packed mode will hard-error with NotFound if an omitted expert is routed"
            );
        }
        info!(
            count = ids.len(),
            path = %order_path.display(),
            "repack: using explicit expert order"
        );
        ids
    } else if let Some(profile_path) = &args.profile {
        let profile = crate::residency::ResidencyProfile::load_json(profile_path)?;
        // Hottest-first over the whole namespace, then append any expert
        // the profile never observed so the blob still covers 0..N.
        let mut ranked = profile.hot_set(1.0, args.num_experts as usize);
        let seen: std::collections::HashSet<u32> = ranked.iter().copied().collect();
        for id in 0..args.num_experts {
            if !seen.contains(&id) {
                ranked.push(id);
            }
        }
        info!(
            observed = seen.len(),
            total = ranked.len(),
            path = %profile_path.display(),
            "repack: ordering experts hottest-first from profile"
        );
        ranked
    } else {
        info!(
            num_experts = args.num_experts,
            "repack: using numeric expert order"
        );
        (0..args.num_experts).collect()
    };

    let manifest_path = args.out_manifest.clone().unwrap_or_else(|| {
        let mut p = args.out_blob.clone().into_os_string();
        p.push(".manifest.json");
        PathBuf::from(p)
    });

    let storage = NvmeStorage::new(StorageConfig {
        base_path: args.data_dir.clone(),
        expert_size: args.expert_size,
        block_align: args.block_align,
        use_direct_io: !args.no_direct,
        num_experts_per_layer: args.num_experts_per_layer,
    })?;
    // One reusable buffer is enough (we read sequentially), but a tiny
    // pool keeps the acquire/release ergonomics and alignment.
    let pool = BufferPool::new(2, args.expert_size, args.block_align);

    info!(
        experts = order.len(),
        out_blob = %args.out_blob.display(),
        out_manifest = %manifest_path.display(),
        "repack: writing packed blob"
    );
    let started = Instant::now();
    let manifest =
        crate::io_provider::pack_experts(&storage, &pool, &order, &args.out_blob, &manifest_path)
            .await?;
    info!(
        elapsed_s = started.elapsed().as_secs_f64(),
        blob_mib = manifest.blob_len() as f64 / (1024.0 * 1024.0),
        experts = manifest.len(),
        "repack complete — point [storage] packed_blob / packed_manifest at these files"
    );
    Ok(())
}

struct RunArgs {
    data_dir: PathBuf,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    cache_slots: usize,
    top_k: usize,
    tokens: u64,
    predict_fanout: usize,
    predict_min_prob: f64,
    no_direct: bool,
    block_align: usize,
    seed: u64,
    dtype: crate::inference::WeightDtype,
    partial_load_fraction: f64,
    pin_after_observations: u64,
    alias_map_path: Option<PathBuf>,
    io_uring: bool,
    token_pause_us: u64,
    first_token: Vec<u32>,
    no_prefetch: bool,
    io_only: bool,
    force_ssd: bool,
    router_clusters: usize,
    router_intra_p: f64,
    router_matrix: Option<PathBuf>,
    gate_weights: Option<PathBuf>,
    trace_out: Option<PathBuf>,
    gpu_expert_cache: Option<Arc<crate::expert_cache::GpuExpertCache>>,
    pipeline_depth: u32,
    speculator: bool,
    speculator_hidden_dim: usize,
    speculator_top_k: usize,
    locality: bool,
    locality_window: usize,
    locality_threshold_pct: f32,
    affinity: bool,
    affinity_neighbors_k: usize,
    affinity_decay_epoch: u64,
    prefetch_governor: bool,
    prefetch_precision_floor: f64,
    prefetch_contention_weight: f64,
    cost_aware_eviction: bool,
    pregate: bool,
    static_residency_fraction: f64,
    static_residency_warmup_tokens: u64,
    static_residency_profile: Option<String>,
    profile_out: Option<String>,
    workload: String,
    zipf_s: f64,
    workload_correlation: f64,
    replay_trace: Option<String>,
    num_layers: u32,
    num_experts_per_layer: Option<u32>,
    packed_blob: Option<PathBuf>,
    packed_manifest: Option<PathBuf>,
}

async fn cmd_run(
    mut args: RunArgs,
    startup_pinned: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // 0) If `metadata.json` exists alongside the expert blobs (e.g. as
    //    written by `scripts/extract_mixtral_experts.py`), use it to fill
    //    in any args the user didn't override on the command line. We
    //    detect "user didn't override" by comparing against clap defaults
    //    — anyone who actually passes a flag overrides the metadata.
    apply_metadata_if_present(&mut args);

    let weight_bytes = expert_weight_bytes_for(args.d_model, args.d_ff, args.dtype);
    let reported_weight_bytes = if args.dtype == crate::inference::WeightDtype::Mixed {
        args.expert_size
    } else {
        weight_bytes
    };
    info!(
        num_experts = args.num_experts,
        top_k = args.top_k,
        cache_slots = args.cache_slots,
        expert_mib = args.expert_size as f64 / (1024.0 * 1024.0),
        d_model = args.d_model,
        d_ff = args.d_ff,
        weight_mib = reported_weight_bytes as f64 / (1024.0 * 1024.0),
        direct_io = !args.no_direct,
        block_align = args.block_align,
        io_only = args.io_only,
        force_ssd = args.force_ssd,
        "starting engine"
    );

    if args.cache_slots > 16 {
        warn!(
            cache_slots = args.cache_slots,
            "--cache-slots is larger than 16. The whole point of this engine is to \
             stream experts from SSD; a large in-RAM cache hides exactly the metric \
             you're trying to measure. Consider 4-8."
        );
    }

    // macOS / non-Linux: O_DIRECT is not available. Force the user (or the
    // run config) into buffered reads and explain what that means for the
    // measurements.
    //
    // Note: the if-else branch selection is decided once at line entry —
    // the `args.no_direct = true` mutation inside the `if` body does NOT
    // retroactively flip the condition. The `else` branch fires when the
    // user *already* passed `--no-direct` on the command line.
    #[cfg(not(target_os = "linux"))]
    {
        if !args.no_direct {
            warn!(
                "O_DIRECT is not supported on this OS (Linux-only). Falling back \
                 to buffered reads (`--no-direct`); measured I/O latency therefore \
                 includes OS page-cache effects and will under-report cold-read \
                 latency on a real NVMe device."
            );
            args.no_direct = true;
        } else {
            warn!(
                "Running with `--no-direct` (buffered reads). Measured I/O latency \
                 includes OS page-cache effects."
            );
        }
        if args.force_ssd {
            warn!(
                "`--force-ssd` was requested but O_DIRECT is unavailable on this OS. \
                 Running in best-effort mode: the OS may still serve some reads from \
                 the page cache. Use a Linux host on a real NVMe device for a clean \
                 measurement."
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        if args.force_ssd && args.no_direct {
            return Err(
                "--force-ssd requires O_DIRECT (do not pass --no-direct alongside it). \
                 With buffered reads the OS page cache can serve repeats from RAM, \
                 which defeats the SSD-bandwidth measurement."
                    .into(),
            );
        }
        if args.no_direct {
            warn!(
                "Running with `--no-direct` (buffered reads). I/O latency in the \
                 summary includes OS page-cache effects."
            );
        }
    }

    if args.expert_size % args.block_align != 0 {
        return Err(format!(
            "expert_size ({}) must be a multiple of block_align ({}) for O_DIRECT",
            args.expert_size, args.block_align
        )
        .into());
    }
    if weight_bytes > 0 && weight_bytes > args.expert_size {
        return Err(format!(
            "expert_size ({}) is too small for the SwiGLU weights of d_model={}, \
             d_ff={} ({} bytes). Increase --expert-size or shrink --d-model / --d-ff \
             so it matches what gen-data wrote.",
            args.expert_size, args.d_model, args.d_ff, weight_bytes
        )
        .into());
    }
    // Multi-drive striping (gist Phase 4). If `--data-dir` contains
    // commas (e.g. `--data-dir /mnt/nvme0,/mnt/nvme1`), we shard
    // experts across the listed directories by `id % n_drives`. The
    // single-dir path is unchanged. Done early because the io_uring
    // NUMA probe below also takes the (canonical) data dir.
    let data_dirs: Vec<PathBuf> = parse_striped_data_dir(&args.data_dir)?;
    let primary_dir = data_dirs
        .first()
        .cloned()
        .unwrap_or_else(|| args.data_dir.clone());
    for d in &data_dirs {
        if !d.is_dir() {
            return Err(format!(
                "data dir {} does not exist; run `gen-data` first",
                d.display()
            )
            .into());
        }
    }
    if data_dirs.len() > 1 {
        info!(
            drives = data_dirs.len(),
            dirs = ?data_dirs,
            "multi-drive striping enabled (experts sharded by id % n_drives)"
        );
    }
    // Treat the first dir as the canonical metadata source for any
    // `metadata.json` / `alias-map` lookups downstream. The other
    // directories only need to contain `expert_<id>.bin`.
    args.data_dir = primary_dir.clone();

    if args.io_uring {
        // Best-effort affinity: keep the engine on the NUMA node that
        // owns CPU 0 to avoid cross-socket DRAM hops on every io_uring
        // completion. Honored only on Linux.
        //
        // Respect startup `MER_PIN_CORES` affinity if it already pinned
        // the process; otherwise fall back to the io_uring default.
        if startup_pinned {
            info!("startup affinity already applied; skipping io_uring repin");
        } else {
            let n = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(8);
            if let Err(e) = pin_to_local_cores(n) {
                warn!(error = %e, "could not set CPU affinity (continuing without pinning)");
            }
        }
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            // Best-effort: detect which NUMA node the data dir's
            // backing block device sits on, and ask the io_uring
            // backend to pin its constructing thread there. The
            // detection function is a no-op on systems where the
            // sysfs entries are missing — it just leaves
            // `numa_node = None` and `IoUringStorage::new` skips
            // pinning entirely.
            let numa_node = detect_data_dir_numa_node(&args.data_dir);
            if let Some(n) = numa_node {
                info!(
                    numa_node = n,
                    "detected NUMA node for data dir; will pin io_uring"
                );
            }
            // Build the backend from the same pool we'll hand the
            // engine; the registration happens inside ::new(). We log
            // the result and then continue with the portable backend
            // for the actual generate() loop — `IoUringStorage` is a
            // drop-in alternative read API (`read_expert_fixed` /
            // `read_experts_batch_fixed`) that callers can wire into
            // their own `Storage` impl. Validating it here gives users
            // a clear error path on misconfigured kernels without
            // reaching the hot path.
            let probe_pool = crate::buffer_pool::BufferPool::new(
                args.cache_slots.max(1),
                args.expert_size,
                args.block_align,
            );
            match crate::io_uring_storage::IoUringStorage::new(
                crate::io_uring_storage::IoUringConfig {
                    base_path: args.data_dir.clone(),
                    expert_size: args.expert_size,
                    block_align: args.block_align,
                    queue_depth: 64,
                    numa_node,
                },
                &probe_pool,
            ) {
                Ok(s) => info!(
                    registered_buffers = s.registered_buffers(),
                    "io_uring backend initialised: registered fixed buffers + ring ready. \
                     The engine still drives reads through the portable pread path; \
                     IoUringStorage::read_experts_batch_fixed is available for \
                     custom integrations."
                ),
                Err(e) => warn!(
                    error = %e,
                    "io_uring backend probe failed (kernel may not support it); \
                     continuing with the portable pread backend."
                ),
            }
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            warn!(
                "--io-uring was passed but this binary was built without the \
                 `io_uring` cargo feature (or is not on Linux). Falling back \
                 to the default `pread(2)` storage backend. Rebuild on Linux \
                 with `--features io_uring` to enable."
            );
        }
    }

    let storage_cfg = StorageConfig {
        base_path: primary_dir.clone(),
        expert_size: args.expert_size,
        block_align: args.block_align,
        use_direct_io: !args.no_direct,
        // The CLI `generate` path is a single-namespace benchmark
        // (`gen-data` produces `expert_<id>.bin`); the multi-layer
        // fallback is only relevant to the `serve` HF-extractor path.
        // `--num-experts-per-layer` opts a `run` into the same
        // layer-qualified geometry so `speculate_layer_ahead` can
        // restrict the speculator head per layer and prefetch ahead.
        num_experts_per_layer: args.num_experts_per_layer,
    };
    let storage = if data_dirs.len() > 1 {
        NvmeStorage::striped(storage_cfg, data_dirs.clone())?
    } else {
        NvmeStorage::new(storage_cfg)?
    };
    // Tier 2: attach the packed blob if configured (defaults: no-op).
    let storage = maybe_attach_packed_blob(
        storage,
        args.packed_blob.as_deref(),
        args.packed_manifest.as_deref(),
        !args.no_direct,
        args.expert_size,
    )?;
    let storage = Arc::new(storage);
    if !storage.is_packed() {
        storage.warmup_fds(0..args.num_experts)?;
    }

    let pipeline_depth = args.pipeline_depth.max(1) as usize;
    let prefetch_headroom = if args.no_prefetch || args.predict_fanout == 0 {
        0
    } else {
        // Scale the speculative headroom by the look-ahead pipeline depth:
        // a depth-N windowed look-ahead (`speculate_layer_ahead` priming
        // `layer + 1 ..= layer + pipeline_depth`) needs a shadow buffer per
        // in-flight layer. The prefetch semaphore is derived from this
        // shadow capacity in `Engine::with_options`, so it scales with it.
        args.predict_fanout.saturating_mul(pipeline_depth)
    };
    // Double-buffered pool: primary (Buffer A) = resident LRU + one
    // reserved foreground slot; shadow (Buffer B) = speculative
    // look-ahead prefetches (sized to `predict_fanout * pipeline_depth`).
    // See `cmd_serve` for the full rationale. `--no-prefetch` (headroom 0)
    // disables Buffer B and keeps the legacy single-pool layout.
    let shadow_slots = prefetch_headroom;
    let primary_slots = args.cache_slots + 1;
    let pool_slots = primary_slots + shadow_slots;

    // Rough RAM heuristic: we don't want to pin more than ~1/4 of total
    // RAM in the buffer pool. This is *advisory* — we warn rather than
    // hard-fail because the user may know their environment better than
    // our /proc/meminfo guess. Skip silently if we can't read RAM.
    if let Some(total_ram) = total_ram_bytes() {
        let pool_bytes = pool_slots as u64 * args.expert_size as u64;
        let budget = total_ram / 4;
        if pool_bytes > budget {
            warn!(
                pool_mib = pool_bytes / (1024 * 1024),
                budget_mib = budget / (1024 * 1024),
                total_ram_mib = total_ram / (1024 * 1024),
                "buffer pool ({} slots × {:.1} MiB/expert) exceeds 1/4 of total RAM. \
                 Lower --cache-slots / --predict-fanout / --pipeline-depth or risk OOM / heavy swapping.",
                pool_slots,
                args.expert_size as f64 / (1024.0 * 1024.0)
            );
        }
    }

    info!(
        cache_slots = args.cache_slots,
        pool_slots = pool_slots,
        prefetch_headroom = prefetch_headroom,
        pipeline_depth = pipeline_depth,
        "buffer pool sized with prefetch headroom (shadow = predict_fanout × pipeline_depth)"
    );
    let pool = if shadow_slots > 0 {
        BufferPool::new_with_shadow(
            primary_slots,
            shadow_slots,
            args.expert_size,
            args.block_align,
        )
    } else {
        BufferPool::new(primary_slots, args.expert_size, args.block_align)
    };
    let cache = Arc::new(MultiLayerExpertCache::single_layer(args.cache_slots));

    // Build the Markov router. If the user supplied a precomputed matrix
    // (e.g. derived from a real Mixtral routing trace), prefer that;
    // otherwise generate a clustered transition matrix.
    let router = if let Some(path) = args.router_matrix.as_ref() {
        info!(matrix = %path.display(), "loading router transition matrix from file");
        Arc::new(TopKRouter::from_matrix_file(
            path,
            args.num_experts,
            args.top_k,
            args.seed,
        )?)
    } else {
        info!(
            clusters = args.router_clusters,
            intra_cluster_p = args.router_intra_p,
            "router: deterministic Markov chain with structured cluster locality"
        );
        Arc::new(TopKRouter::clustered(
            args.num_experts,
            args.top_k,
            args.router_clusters,
            args.router_intra_p,
            args.seed,
        ))
    };
    let predictor = Arc::new(PredictiveLoader::new(
        args.num_experts,
        if args.no_prefetch {
            0
        } else {
            args.predict_fanout
        },
        resolve_predict_min_prob(args.predict_min_prob, args.num_experts),
        args.seed,
    ));

    let engine = Arc::new({
        let mut base = Engine::with_options(
            cache.clone(),
            pool.clone(),
            storage.clone(),
            crate::gating::Router::Markov(router.clone()),
            predictor.clone(),
            ModelShape {
                d_model: args.d_model,
                d_ff: args.d_ff,
                hidden_seed: args.seed,
            },
            EngineOptions {
                io_only: args.io_only,
                dtype: args.dtype,
                partial_load_fraction: args.partial_load_fraction,
                pin_after_observations: args.pin_after_observations,
                use_qmm_for_q4: true,
                expert_execution_policy: crate::engine::ExpertExecutionPolicy::Auto,
                max_concurrent_prefetches: 64,
                max_fetch_yields: crate::engine::DEFAULT_MAX_FETCH_YIELDS,
                prefetch_governor: args.prefetch_governor,
                prefetch_precision_floor: args.prefetch_precision_floor,
                prefetch_contention_weight: args.prefetch_contention_weight,
                cost_aware_eviction: args.cost_aware_eviction,
                pregate_enabled: args.pregate,
                collect_route_profile: args.profile_out.is_some(),
            },
        );
        if let Some(gpu_cache) = args.gpu_expert_cache.clone() {
            base.install_gpu_cache(gpu_cache);
        }
        // Apply the configured look-ahead pipeline depth (sized in tandem
        // with the shadow buffer-pool budget above). No-op for the legacy
        // Markov path (no speculator installed); takes effect when a
        // speculator drives `speculate_layer_ahead`.
        let mut base = base.with_pipeline_depth(args.pipeline_depth);
        // Predictive arms (opt-in, mirroring `cmd_serve`'s `[predictive]`
        // wiring). These are what turn the speculative-I/O union-fetch
        // `E = S ∪ L ∪ M` from "Markov-only" into the full predictor:
        //   * M — neural speculator over the residual stream (also the
        //     only arm that drives `speculate_layer_ahead` look-ahead),
        //   * L — sliding-window locality monitor whose hot set is pinned
        //     (frequency-aware eviction on top of plain LRU),
        //   * affinity — per-layer co-occurrence + disk-adjacency fold.
        // All off by default so the legacy benchmark is bit-for-bit; turn
        // them on to measure whether they move the hit rate / I/O share.
        if args.locality {
            // Mirror `cmd_serve`: when the run uses a layer-qualified
            // namespace, scale the window by the layer count so the
            // per-layer history depth matches the configured value
            // (see `effective_locality_threshold` in engine.rs).
            let num_layers = args
                .num_experts_per_layer
                .filter(|&p| p > 0)
                .map(|p| args.num_experts.div_ceil(p).max(1) as usize)
                .unwrap_or(1);
            let monitor = Arc::new(LocalityMonitor::new(
                args.num_experts,
                args.locality_window.saturating_mul(num_layers),
            ));
            base = base.with_locality_monitor(monitor, args.locality_threshold_pct);
        }
        if args.speculator {
            let top_k = if args.speculator_top_k == 0 {
                args.top_k
            } else {
                args.speculator_top_k
            };
            let spec = Arc::new(NeuralSpeculator::new(
                args.d_model,
                args.speculator_hidden_dim,
                args.num_experts,
                args.seed,
            ));
            base = base.with_speculator(spec, top_k);
        }
        if args.affinity {
            // The affinity arm is only consulted on the layer-qualified
            // `moe_step` path (the `--gate-weights` / multi-layer route);
            // the flat single-namespace `generate` benchmark never folds
            // it in. Warn rather than silently no-op when the user asks
            // for affinity without a layer geometry.
            if args.num_experts_per_layer.is_none() {
                warn!(
                    "--affinity has no effect without --num-experts-per-layer: the \
                     affinity fold only runs on the layer-qualified moe_step path. \
                     Pass --num-experts-per-layer (and typically --gate-weights) to \
                     exercise it."
                );
            }
            let per_layer = args.num_experts_per_layer.unwrap_or(args.num_experts);
            let affinity = Arc::new(LayeredExpertAffinity::new(
                args.num_layers.max(1) as usize,
                per_layer,
            ));
            base = base.with_affinity(
                affinity,
                args.affinity_neighbors_k,
                args.affinity_decay_epoch,
            );
        }
        // Tier 1 — static residency. Pin the hottest `fraction` of the
        // expert namespace permanently. With `--static-residency-profile`
        // the hot set comes from an offline popularity profile (warm at
        // startup); otherwise it is derived online from route counts after
        // `--static-residency-warmup-tokens`.
        if args.static_residency_fraction > 0.0 {
            let profile = match args.static_residency_profile.as_ref() {
                Some(path) => {
                    let p =
                        crate::residency::ResidencyProfile::load_json(std::path::Path::new(path))?;
                    info!(
                        path = %path,
                        experts = p.len(),
                        "loaded static-residency popularity profile"
                    );
                    Some(p)
                }
                None => None,
            };
            base = base.with_static_residency(
                args.static_residency_fraction,
                args.static_residency_warmup_tokens,
                profile,
            );
        }
        // Tier 3 — per-layer pre-gate. Predict (and prefetch) the next
        // layer's experts from the current layer's routed set. Only
        // effective on the multi-layer `moe_step` path; warn when no
        // layer geometry is configured so it can't actually fire.
        if args.pregate {
            if args.num_layers <= 1 {
                warn!(
                    "--pregate has no effect with --num-layers 1: the pre-gate predicts \
                     the *next* layer's experts, so it needs a multi-layer geometry \
                     (set --num-layers > 1, typically with --gate-weights / a real model)."
                );
            }
            let pregate = Arc::new(crate::pregate::PerLayerPreGate::new(
                args.num_layers.max(1) as usize,
                args.top_k,
            ));
            base = base.with_pregate(pregate);
        }
        // Optional alias map (Change 6: expert deduplication).
        match args.alias_map_path.as_ref() {
            Some(path) => {
                let map = load_alias_map(path)?;
                info!(
                    path = %path.display(),
                    entries = map.len(),
                    "loaded expert alias map (deduplicated experts share resident copies)"
                );
                base.with_alias_map(map)
            }
            None => base,
        }
    });

    // Optional warm-up to mirror the spec example ("the router selects
    // Expert ID 3 and 7"): fetch those experts up front so the first real
    // token routes against an already-warm cache.
    if !args.first_token.is_empty() {
        let target = router.fixed(&args.first_token);
        info!(experts = ?target, "warm-up fetch (mirrors spec example)");
        engine.warm_with(&target).await?;
    }

    // Optional JSONL routing trace (gist Phase 6). When set, every
    // call to `engine.generate` appends one record. Wired up *after*
    // the warm-up so warm-fetched experts don't pollute the trace
    // with synthetic tokens (`Engine::warm_with` doesn't go through
    // `generate`).
    let trace_writer = match args.trace_out.as_ref() {
        Some(path) => {
            info!(path = %path.display(), "writing routing trace");
            let w = Arc::new(crate::engine::TraceWriter::open(path)?);
            engine.set_trace_writer(Some(w.clone()));
            Some(w)
        }
        None => None,
    };

    let stream_started = Instant::now();
    info!(
        tokens = args.tokens,
        "streaming tokens (latency / throughput logs follow)"
    );

    // Optional production gating network. When present, every token's
    // expert ids come from `softmax(W_gate · x) → top-K` (real Mixtral
    // routing) instead of the deterministic Markov `TopKRouter`. The
    // SSD-streaming substrate is identical either way — only the *id
    // selection* changes — so the cycle / I/O / hit-rate metrics are
    // directly comparable across the two paths.
    let gate: Option<crate::gating::LinearGate> = match args.gate_weights.as_ref() {
        Some(path) => {
            info!(
                gate_weights = %path.display(),
                num_experts = args.num_experts,
                d_model = args.d_model,
                top_k = args.top_k,
                "loading gating-network weight matrix"
            );
            Some(load_gate_weights(
                path,
                args.num_experts as usize,
                args.d_model,
                args.top_k,
            )?)
        }
        None => None,
    };

    // Benchmark workload selection (Tier 1/3 falsifiability). `synthetic`
    // keeps the legacy uniform-i.i.d. stream; `skewed`/`replay` drive
    // `moe_step` with an explicit, structured expert set so the
    // skew-aware and correlation-aware machinery is exercisable.
    let workload = crate::workload::Workload::from_str_opt(&args.workload).ok_or_else(|| {
        format!(
            "--workload: unknown value {:?} (use 'synthetic', 'skewed', or 'replay')",
            args.workload
        )
    })?;
    let mut skewed_stream = if workload == crate::workload::Workload::Skewed {
        info!(
            zipf_s = args.zipf_s,
            correlation = args.workload_correlation,
            top_k = args.top_k,
            "workload: skewed (Zipf popularity + Markov correlation)"
        );
        Some(crate::workload::SkewedStream::new(
            args.num_experts,
            args.top_k,
            args.zipf_s,
            args.workload_correlation,
            args.seed,
        ))
    } else {
        None
    };
    let mut replay_stream = if workload == crate::workload::Workload::Replay {
        let path = args
            .replay_trace
            .as_ref()
            .ok_or("--workload replay requires --replay-trace <path>")?;
        let stream = crate::workload::ReplayStream::load(std::path::Path::new(path))?;
        if stream.is_empty() {
            return Err(format!("--replay-trace {path}: no usable routing records").into());
        }
        info!(path = %path, records = stream.len(), "workload: replay JSONL routing trace");
        Some(stream)
    } else {
        None
    };

    for t in 0..args.tokens {
        let start = Instant::now();
        let stats = match workload {
            // Structured workloads: drive `moe_step` with the harness's
            // explicit expert set and measure the engine-counter delta.
            crate::workload::Workload::Skewed | crate::workload::Workload::Replay => {
                let (tok_idx, layer_idx, experts): (u64, u32, Vec<u32>) = match workload {
                    crate::workload::Workload::Skewed => (
                        t,
                        0,
                        skewed_stream
                            .as_mut()
                            .expect("skewed stream")
                            .next_experts(),
                    ),
                    _ => {
                        let record = replay_stream
                            .as_mut()
                            .expect("replay stream")
                            .next_record()
                            .expect("replay stream non-empty");
                        let layer = u32::try_from(record.layer).map_err(|_| {
                            format!("replay layer {} does not fit in u32", record.layer)
                        })?;
                        (record.token, layer, record.experts)
                    }
                };
                let hidden = crate::inference::synth_hidden_state(tok_idx, args.d_model, args.seed);
                let pre = engine.report();
                let _ = engine.moe_step(tok_idx, layer_idx, &hidden, &experts).await;
                let post = engine.report();
                crate::engine::CycleStats {
                    hits: post.hits.saturating_sub(pre.hits),
                    misses: post.misses.saturating_sub(pre.misses),
                    prefetch_hits: 0,
                    bytes_read: post.bytes_read.saturating_sub(pre.bytes_read),
                }
            }
            crate::workload::Workload::Synthetic => {
                if let Some(gate) = gate.as_ref() {
                    // Real gating-network path. Hidden state is the same
                    // synthetic activation `Engine::generate` would have
                    // used, so the only difference relative to the legacy
                    // path is *which* experts are selected.
                    let hidden = crate::inference::synth_hidden_state(t, args.d_model, args.seed);
                    let dec = gate.route(&hidden);
                    let pre = engine.report();
                    let _ = engine.moe_step(t, 0, &hidden, &dec.experts).await;
                    let post = engine.report();
                    crate::engine::CycleStats {
                        hits: post.hits.saturating_sub(pre.hits),
                        misses: post.misses.saturating_sub(pre.misses),
                        prefetch_hits: 0,
                        bytes_read: post.bytes_read.saturating_sub(pre.bytes_read),
                    }
                } else {
                    engine.generate(t).await?
                }
            }
        };
        let elapsed = start.elapsed();
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            1.0 / elapsed.as_secs_f64()
        } else {
            f64::INFINITY
        };
        info!(
            token = t,
            cycle_us = elapsed.as_micros() as u64,
            tps = format!("{throughput:.1}"),
            hits = stats.hits,
            misses = stats.misses,
            kib = stats.bytes_read / 1024,
            resident = ?cache.resident_ids(),
            "tick"
        );
        if args.token_pause_us > 0 {
            tokio::time::sleep(Duration::from_micros(args.token_pause_us)).await;
        }
    }

    let wall = stream_started.elapsed();
    let r = engine.report();
    let total_lookups = (r.hits + r.misses).max(1);
    info!(
        wall_s = wall.as_secs_f64(),
        sustained_tps = args.tokens as f64 / wall.as_secs_f64(),
        avg_throughput_mibps = (r.bytes_read as f64 / (1024.0 * 1024.0)) / wall.as_secs_f64(),
        hit_rate_pct = (r.hits as f64 / total_lookups as f64) * 100.0,
        "stream complete"
    );
    engine.print_summary();

    if r.misses > 0 && r.io_p50_us == 0 {
        warn!(
            "I/O latency histogram is empty despite cache misses; check that \
             tracing is enabled and runs are long enough to produce samples."
        );
    }

    // Flush the trace before returning so the JSONL file is complete.
    if let Some(tw) = trace_writer.as_ref() {
        tw.flush();
    }

    // Tier 1 — emit the accumulated expert-popularity profile so a later
    // run can warm-start static residency with `--static-residency-profile`.
    if let Some(path) = args.profile_out.as_ref() {
        engine
            .dump_route_profile(std::path::Path::new(path))
            .map_err(|e| format!("failed to write route profile {}: {e}", path))?;
        info!(path = %path, "wrote route-observation profile");
    }

    Ok(())
}

/// Per-CLI defaults. We compare an `args` value to its default to detect
/// whether the user actually passed the flag, so `metadata.json` can fill
/// in just the values the user *didn't* override.
mod cli_defaults {
    pub const NUM_EXPERTS: u32 = 64;
    pub const EXPERT_SIZE: usize = 16 * 1024 * 1024;
    pub const D_MODEL: usize = 512;
    pub const D_FF: usize = 2048;
    pub const TOP_K: usize = 2;
    pub const BLOCK_ALIGN: usize = 4096;
}

/// Hand-rolled `metadata.json` parser. The only fields we care about are
/// numeric scalars (`num_experts`, `d_model`, `d_ff`, `top_k`,
/// `expert_size`); pulling in `serde_json` for that would add a heavy
/// dependency the rest of the engine doesn't need.
fn apply_metadata_if_present(args: &mut RunArgs) {
    let path = args.data_dir.join("metadata.json");
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(_) => return,
    };
    info!(path = %path.display(), "found metadata.json — auto-filling unspecified args");
    let mut overrode_anything = false;
    let mut set_if_default = |key: &str, current: u64, default: u64, sink: &mut dyn FnMut(u64)| {
        if let Some(v) = parse_json_number(&body, key) {
            // Only fill in values the user didn't override on the CLI.
            if current == default {
                sink(v);
                overrode_anything = true;
            } else if v != current {
                warn!(
                    key,
                    metadata = v,
                    cli = current,
                    "CLI value overrides metadata.json"
                );
            }
        }
    };
    set_if_default(
        "num_experts",
        args.num_experts as u64,
        cli_defaults::NUM_EXPERTS as u64,
        &mut |v| args.num_experts = v as u32,
    );
    set_if_default(
        "d_model",
        args.d_model as u64,
        cli_defaults::D_MODEL as u64,
        &mut |v| args.d_model = v as usize,
    );
    set_if_default(
        "d_ff",
        args.d_ff as u64,
        cli_defaults::D_FF as u64,
        &mut |v| args.d_ff = v as usize,
    );
    set_if_default(
        "top_k",
        args.top_k as u64,
        cli_defaults::TOP_K as u64,
        &mut |v| args.top_k = v as usize,
    );
    set_if_default(
        "expert_size",
        args.expert_size as u64,
        cli_defaults::EXPERT_SIZE as u64,
        &mut |v| args.expert_size = v as usize,
    );
    set_if_default(
        "block_align",
        args.block_align as u64,
        cli_defaults::BLOCK_ALIGN as u64,
        &mut |v| args.block_align = v as usize,
    );
    if args.dtype == crate::inference::WeightDtype::F32 {
        if let Some(dtype_str) = parse_json_string(&body, "dtype") {
            if let Some(dtype) = crate::inference::WeightDtype::from_str_opt(&dtype_str) {
                args.dtype = dtype;
                overrode_anything = true;
            }
        }
    }
    if overrode_anything {
        info!(
            num_experts = args.num_experts,
            d_model = args.d_model,
            d_ff = args.d_ff,
            top_k = args.top_k,
            expert_mib = args.expert_size as f64 / (1024.0 * 1024.0),
            "engine parameters after metadata.json"
        );
    }
}

/// Look up `"key": <number>` in a JSON document and return the integer.
/// Tolerates whitespace and surrounding quotes; returns `None` if the
/// key is missing or the value is non-integer / negative. Good enough
/// for the small handful of scalars in `metadata.json`.
fn parse_json_number(body: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let pos = body.find(&needle)?;
    let after = &body[pos + needle.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    let mut end = 0;
    for (i, c) in after.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    after[..end].parse::<u64>().ok()
}

fn parse_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)? + needle.len();
    let rest = body[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse a tiny JSON object of the form `{ "src_id": canonical_id, ... }`
/// into a `HashMap<u32, u32>`. Hand-rolled to keep `serde_json` out of
/// the engine's dep tree (the rest of the engine uses our smaller
/// `parse_json_number`-style helpers). Returns an error if the file
/// can't be read or contains a malformed entry.
fn load_alias_map(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<u32, u32>, Box<dyn std::error::Error>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read alias map {}: {e}", path.display()))?;
    let body = body.trim();
    let body = body
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| format!("alias map {} must be a JSON object", path.display()))?;
    let mut map = std::collections::HashMap::new();
    for raw in body.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let (k, v) = raw
            .split_once(':')
            .ok_or_else(|| format!("alias map entry {raw:?} missing ':'"))?;
        // Strip optional whitespace + surrounding quotes around the key.
        let k = k.trim().trim_matches('"');
        let v = v.trim();
        let key: u32 = k
            .parse()
            .map_err(|_| format!("alias map key {k:?} must be a non-negative integer"))?;
        let val: u32 = v
            .parse()
            .map_err(|_| format!("alias map value {v:?} must be a non-negative integer"))?;
        map.insert(key, val);
    }
    Ok(map)
}

/// Load a real gating-network weight matrix from disk.
///
/// File format: bare little-endian `f32`s, no header, row-major,
/// `[num_experts × d_model]`. This is the layout `numpy.tofile` writes
/// for `block_sparse_moe.gate.weight` after `astype(np.float32)`. A
/// future PR can teach this to read `safetensors` directly so the user
/// can point it at a HuggingFace shard without a conversion step.
///
/// **Directory input.** When `path` is a directory rather than a file,
/// the per-layer `gate_<L>.bin` files inside it (the same naming the
/// real-model loader writes/reads in `model.rs`) are auto-discovered,
/// sorted ascending by layer index, and concatenated in layer order.
/// This is the in-memory equivalent of
/// `cat gate_0.bin gate_1.bin … gate_N.bin > real_gate.bin`, so users
/// can point `--gate-weights` straight at a model directory instead of
/// hand-concatenating a non-standard monolithic file.
fn load_gate_weights(
    path: &std::path::Path,
    num_experts: usize,
    d_model: usize,
    top_k: usize,
) -> Result<crate::gating::LinearGate, Box<dyn std::error::Error>> {
    let bytes = if path.is_dir() {
        read_gate_dir_concatenated(path)?
    } else {
        std::fs::read(path)
            .map_err(|e| format!("failed to read gate weights {}: {e}", path.display()))?
    };
    let expected = num_experts
        .checked_mul(d_model)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "num_experts * d_model overflowed".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "gate weights {} have {} bytes, expected {} ({} experts × {} d_model × 4 bytes/f32)",
            path.display(),
            bytes.len(),
            expected,
            num_experts,
            d_model
        )
        .into());
    }
    let mut weights = Vec::<f32>::with_capacity(num_experts * d_model);
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        weights.push(f32::from_le_bytes(buf));
    }
    Ok(crate::gating::LinearGate::new(
        weights,
        num_experts,
        d_model,
        top_k,
    ))
}

/// Discover and concatenate the per-layer `gate_<L>.bin` files in `dir`,
/// sorted ascending by layer index. Returns the concatenated raw bytes,
/// which [`load_gate_weights`] then validates against the expected
/// `num_experts × d_model × 4` total — exactly as if the caller had run
/// `cat gate_0.bin gate_1.bin … > real_gate.bin` first.
fn read_gate_dir_concatenated(
    dir: &std::path::Path,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut entries: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| {
        format!(
            "failed to scan gate-weights directory {}: {e}",
            dir.display()
        )
    })? {
        let entry = entry
            .map_err(|e| format!("failed to read a directory entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(idx) = parse_gate_layer_index(name) {
            entries.push((idx, path));
        }
    }
    if entries.is_empty() {
        return Err(format!(
            "no gate_<layer>.bin files found in directory {}; expected per-layer files \
             named like gate_0.bin, gate_1.bin, … (each file is a little-endian f32 shard; concatenation must total [num_experts × d_model])",
            dir.display()
        )
        .into());
    }
    entries.sort_by_key(|(idx, _)| *idx);
    // Reject duplicate layer indices: the concatenation order would be
    // ambiguous and almost certainly indicates a stray file.
    for w in entries.windows(2) {
        if w[0].0 == w[1].0 {
            return Err(format!(
                "duplicate gate layer index {} in directory {} ({} and {})",
                w[0].0,
                dir.display(),
                w[0].1.display(),
                w[1].1.display()
            )
            .into());
        }
    }
    // `entries` is guaranteed non-empty here (early return above), so the
    // first/last layer indices are always present.
    let first_layer = entries.first().map(|(i, _)| *i).expect("entries non-empty");
    let last_layer = entries.last().map(|(i, _)| *i).expect("entries non-empty");
    info!(
        dir = %dir.display(),
        files = entries.len(),
        first_layer,
        last_layer,
        "discovered per-layer gate files; concatenating in ascending layer order"
    );
    let mut bytes = Vec::new();
    for (idx, p) in &entries {
        let mut chunk = std::fs::read(p).map_err(|e| {
            format!(
                "failed to read gate file {} (layer {idx}): {e}",
                p.display()
            )
        })?;
        bytes.append(&mut chunk);
    }
    Ok(bytes)
}

/// Parse the layer index `N` out of a `gate_<N>.bin` filename. Returns
/// `None` for any name that doesn't match that exact pattern (so
/// unrelated files in the directory are simply ignored).
fn parse_gate_layer_index(name: &str) -> Option<u32> {
    let idx = name.strip_prefix("gate_")?.strip_suffix(".bin")?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    idx.parse::<u32>().ok()
}

/// Best-effort total-RAM probe. Returns `None` (heuristic disabled) on
/// platforms or filesystems we don't recognise. We intentionally avoid
/// pulling in a `sysinfo`-style dependency for one number.
fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let body = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kib.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Best-effort NUMA / CPU-affinity hint.
///
/// On Linux, when `num_cores > 0`, pin the current process to the first
/// `num_cores` CPUs of the NUMA node owning CPU 0 (via
/// `sched_setaffinity(2)`). The intent is to keep io_uring completion
/// processing, the engine's matmul threads, and the tokio runtime on
/// the same memory controller — for an SSD-streaming MoE this avoids
/// cross-socket DRAM hops on every cache fill, which dominate latency
/// at high QPS.
///
/// We deliberately don't pull in a NUMA crate: this function picks the
/// CPUs from `/sys/devices/system/node/node0/cpulist` (a kernel-exposed
/// comma+dash list). When that file isn't available we fall back to
/// CPUs `0..num_cores`. On non-Linux targets this is a no-op that
/// returns `Ok(())` so callers can always invoke it unconditionally.
fn pin_to_local_cores(num_cores: usize) -> std::io::Result<()> {
    if num_cores == 0 {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let cpus = read_local_node_cpus().unwrap_or_else(|| (0..num_cores).collect());
        let chosen: Vec<usize> = cpus.into_iter().take(num_cores).collect();
        if chosen.is_empty() {
            return Ok(());
        }
        // SAFETY: `cpu_set_t` is a POD bitset; we zero it with
        // `mem::zeroed`, fill in valid CPU bits with the libc helpers,
        // then hand it to `sched_setaffinity` which only reads.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            for cpu in &chosen {
                libc::CPU_SET(*cpu, &mut set);
            }
            let rc = libc::sched_setaffinity(
                0, // current process
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const _,
            );
            if rc != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        info!(
            cores = ?chosen,
            "pinned process to NUMA-local CPU set (best-effort; sched_setaffinity)"
        );
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        warn!(
            "core affinity hint requested ({} cores) but pinning is Linux-only; \
             continuing without affinity",
            num_cores
        );
        Ok(())
    }
}

/// Parse `/sys/devices/system/node/node0/cpulist` into an ordered
/// `Vec<usize>` of CPU ids (e.g. `"0-3,8,10-11"` -> `[0,1,2,3,8,10,11]`).
/// Returns `None` if the file is missing / unparseable — callers fall
/// back to a contiguous `0..N` range in that case.
#[cfg(target_os = "linux")]
fn read_local_node_cpus() -> Option<Vec<usize>> {
    let body = std::fs::read_to_string("/sys/devices/system/node/node0/cpulist").ok()?;
    let mut out: Vec<usize> = Vec::new();
    for part in body.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.parse().ok()?;
            let hi: usize = hi.parse().ok()?;
            if hi < lo {
                return None;
            }
            out.extend(lo..=hi);
        } else {
            out.push(part.parse().ok()?);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Detect the NUMA node of the block device backing `data_dir`.
///
/// Returns `Some(node)` when both probes succeed:
///   1. `stat(2)` on `data_dir` yields a device id whose major/minor
///      we map to `/sys/dev/block/MAJ:MIN/device/numa_node`.
///   2. The contents of that sysfs entry parse to a non-negative integer.
///
/// Any failure (non-Linux build, sysfs entry missing, NUMA disabled in
/// the kernel which reports `-1`, permission errors) returns `None`
/// and lets the caller continue without NUMA pinning. This is a
/// *hint*; it must never block startup.
pub fn detect_data_dir_numa_node(data_dir: &std::path::Path) -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        let md = std::fs::metadata(data_dir).ok()?;
        // st_dev is encoded as major:minor; major = (dev >> 8) & 0xfff
        // for the legacy layout but Linux uses a more flexible
        // encoding. libc::major()/minor() handle both.
        let dev = md.dev();
        // `libc::major` / `libc::minor` are safe `const fn`s in libc ≥ 0.2.156;
        // no `unsafe` block is required.
        let major = libc::major(dev) as u32;
        let minor = libc::minor(dev) as u32;
        let sys_path = format!("/sys/dev/block/{}:{}/device/numa_node", major, minor);
        let body = std::fs::read_to_string(&sys_path).ok()?;
        let node: i32 = body.trim().parse().ok()?;
        // Kernel reports `-1` when NUMA is disabled or unknown.
        if node < 0 {
            None
        } else {
            Some(node)
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = data_dir;
        None
    }
}

/// Parse `--data-dir` into a list of directories. If the path
/// stringifies to a comma-separated list, split it; otherwise return a
/// single-element vec. Used by gist Phase 4 (multi-drive striping).
fn parse_striped_data_dir(p: &std::path::Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let s = p.to_string_lossy();
    if s.contains(',') {
        let dirs: Vec<PathBuf> = s
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(PathBuf::from)
            .collect();
        if dirs.is_empty() {
            return Err(format!(
                "invalid --data-dir '{}': comma-separated list must contain at least one \
                 non-empty directory path",
                p.display()
            )
            .into());
        }
        Ok(dirs)
    } else {
        Ok(vec![p.to_path_buf()])
    }
}

fn cmd_gguf_convert(
    gguf_path: &PathBuf,
    out_dir: &PathBuf,
    num_layers: usize,
    num_experts: usize,
    emit_uth: bool,
    legacy_eager: bool,
    native_quant: bool,
    experts_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        path = %gguf_path.display(),
        emit_uth,
        legacy_eager,
        native_quant,
        experts_only,
        "opening GGUF file"
    );
    let opts = crate::gguf_loader::ExtractOptions {
        emit_uth,
        native_quant,
        experts_only,
    };
    let source = crate::gguf::open_gguf_source(gguf_path, legacy_eager)?;
    if let Some(arch) = source.architecture() {
        info!(architecture = arch, "GGUF source opened");
    }
    let report = crate::gguf_loader::extract_experts_from_source(
        &*source,
        out_dir,
        num_layers,
        num_experts,
        opts,
    )?;
    let total_gib = report.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let read_time_at_7gbps = report.total_bytes as f64 / (7.0 * 1024.0 * 1024.0 * 1024.0);
    info!(
        experts_written = report.experts_written,
        dense_written = report.dense_written,
        skipped = report.skipped,
        total_bytes = report.total_bytes,
        total_gib,
        expected_read_seconds_at_7gbps = read_time_at_7gbps,
        d_model = report.d_model,
        d_ff = report.d_ff,
        num_layers = report.num_layers,
        num_experts_per_layer = report.num_experts_per_layer,
        "gguf-convert complete"
    );
    println!(
        "gguf-convert: wrote {} expert files + {} dense tensors ({:.2} GiB total). \
         At 7 GB/s aggregate SSD read bandwidth, a full warm-up scan would take ~{:.2}s.",
        report.experts_written, report.dense_written, total_gib, read_time_at_7gbps
    );
    Ok(())
}

fn cmd_validate_data(data_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let report = crate::gguf_loader::validate_data_dir(data_dir)?;
    println!(
        "validate-data: ok (experts={}, expert_size={} bytes, block_align={}, dtype={}, mixed_experts={})",
        report.num_experts,
        report.expert_size,
        report.block_align,
        report.dtype.as_str(),
        report.mixed_experts
    );
    Ok(())
}

async fn cmd_validate_predictor(
    trace_path: &PathBuf,
    cache_slots: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(trace_path)?;
    let text = String::from_utf8_lossy(&bytes);
    // Parse JSONL records {"token": .., "layer": .., "experts": [..], "cache_hit": [..]}.
    // We extract just the per-token expert id sequence; the predictor
    // validation replays them through a fresh LRU and prints per-K
    // hit rates plus per-layer breakdown and top-1 / top-2 accuracy.
    #[derive(Default)]
    struct LayerStats {
        tokens: u64,
        // for top-1 / top-2 accuracy we compare the predicted set of
        // size K against the actual top-1 / top-2 routed experts.
        top1_hits: u64,
        top2_hits: u64,
    }

    // Flat list of (token, layer, experts) records in the order
    // they were observed in the JSONL file. We rely on a stable
    // sort over the global `token` field to reconstruct the engine's
    // per-token, per-layer interleaving — even if a multi-layer
    // trace's records were appended in any order. Pre gist
    // feedback #2.2 we instead grouped by layer first and then
    // flattened, which produced "all of layer 0, then all of
    // layer 1, …" — meaningless on real multi-layer (e.g. Mixtral's
    // 32 layers) traces because the per-layer caches saw an entirely
    // synthetic recent-history.
    let mut records: Vec<(u64, u32, Vec<u32>)> = Vec::new();
    let mut by_layer: std::collections::BTreeMap<u32, LayerStats> = Default::default();
    for (file_idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Fall back to the file-line index when a record lacks
        // an explicit `token` so legacy traces still order
        // monotonically by appearance.
        let token = json_get_u64(line, "token").unwrap_or(file_idx as u64);
        let layer = json_get_u64(line, "layer").unwrap_or(0) as u32;
        let experts = json_get_u32_array(line, "experts");
        if experts.is_empty() {
            continue;
        }
        by_layer.entry(layer).or_default().tokens += 1;
        records.push((token, layer, experts));
    }
    // Stable sort by (token, layer) reconstructs the original
    // interleaved order the engine produced — for token T the
    // entries for layer 0, 1, 2, … appear in order, then the same
    // for token T+1, etc. — which is what the LRU saw in production.
    records.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    // Per-cache-size simulation: maintain a single LRU shared across
    // *all* layers in the trace and count hits. This matches
    // `scripts/compute_transition_matrix.py::simulate_lru`, which
    // replays the trace through one global LRU rather than per-layer
    // caches — having both lets the Rust and Python paths produce
    // identical hit-rate numbers for the same trace.
    let ks: Vec<usize> = if cache_slots.is_empty() {
        vec![2, 4, 8, 16]
    } else {
        cache_slots.to_vec()
    };
    println!("validate-predictor: trace={}", trace_path.display());
    for k in &ks {
        let mut hits = 0u64;
        let mut total = 0u64;
        // Maintain order in a VecDeque and membership in a HashSet
        // so the per-token hit check is O(1) instead of O(N). The
        // VecDeque carries the LRU ordering (front = oldest); the
        // HashSet mirrors the same id set for fast `contains`. This
        // is the same hit-rate as before, just without the O(N·M)
        // walk over `lru.iter().any(...)` that the prior version
        // performed (gist feedback #2.5 — keeps `validate-predictor`
        // workable on long real-engine traces).
        let mut lru: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let mut lru_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (_, _, experts) in &records {
            for &e in experts.iter() {
                if lru_set.contains(&e) {
                    hits += 1;
                    // Move-to-back: O(N) here but only on a hit
                    // (cheap relative to the surrounding miss-path).
                    if let Some(pos) = lru.iter().position(|x| *x == e) {
                        lru.remove(pos);
                    }
                } else if lru.len() == *k {
                    if let Some(evicted) = lru.pop_front() {
                        lru_set.remove(&evicted);
                    }
                }
                lru.push_back(e);
                lru_set.insert(e);
                total += 1;
            }
        }
        let rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        println!("  cache_slots={k:>3}  hit_rate={rate:>6.3}  hits={hits}/{total}");
    }

    // Group sorted records into per-layer buckets *after* the LRU
    // replay so we can consume `records` without cloning each `experts`
    // vector.
    let mut tokens_per_layer: std::collections::BTreeMap<u32, Vec<Vec<u32>>> = Default::default();
    for (_, layer, experts) in records.into_iter() {
        tokens_per_layer.entry(layer).or_default().push(experts);
    }

    // Top-1 / Top-2 predictor accuracy: replay one-step-ahead via a
    // simple last-expert Markov predictor (the cheapest baseline the
    // engine has). For each (prev, curr) pair we predict `prev` and
    // count it as a top-1 hit if it appears in `curr`, top-2 if any
    // of {prev, second-most-recent} appears in `curr`.
    for (layer, seq) in tokens_per_layer.iter() {
        let stats = by_layer.entry(*layer).or_default();
        let mut prev: Option<u32> = None;
        let mut prev2: Option<u32> = None;
        for experts in seq {
            if let Some(p) = prev {
                if experts.iter().any(|&x| x == p) {
                    stats.top1_hits += 1;
                }
                let predict2: std::collections::HashSet<u32> =
                    [Some(p), prev2].iter().filter_map(|x| *x).collect();
                if experts.iter().any(|x| predict2.contains(x)) {
                    stats.top2_hits += 1;
                }
            }
            prev2 = prev;
            prev = experts.first().copied();
        }
    }
    println!("\nper-layer Markov predictor accuracy:");
    for (layer, st) in &by_layer {
        let denom = st.tokens.saturating_sub(1).max(1);
        let top1 = st.top1_hits as f64 / denom as f64;
        let top2 = st.top2_hits as f64 / denom as f64;
        println!(
            "  layer={layer:>3}  tokens={:>6}  top1={top1:>6.3}  top2={top2:>6.3}",
            st.tokens
        );
    }
    Ok(())
}

/// Pull a numeric field and a `[..]` u32 array out of one JSONL line.
/// The trace records have a fixed schema (`{token, layer, experts,
/// cache_hit}`), so we route through `serde_json::Value` for safety
/// without paying the cost of deriving a full type.
fn json_get_u64(line: &str, key: &str) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    v.get(key).and_then(|x| x.as_u64())
}

fn json_get_u32_array(line: &str, key: &str) -> Vec<u32> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return Vec::new();
    };
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_real_validity_detects_softmax_fallback_delta() {
        // Snapshot then force a fallback: the current count only grows, so a
        // check against the pre-snapshot baseline must report the run invalid.
        // This is robust under concurrent tests because the counter is
        // monotonic.
        let snapshot = crate::transformer::nonfinite_softmax_fallbacks();
        crate::transformer::record_nonfinite_softmax_fallback();
        assert!(
            assert_no_softmax_fallbacks(snapshot).is_err(),
            "a nonzero softmax-fallback delta must invalidate the benchmark"
        );
    }

    // ---- Finding 4: real serving requires a real tokenizer ----

    #[test]
    fn real_serving_without_tokenizer_path_fails() {
        match resolve_serving_tokenizer(true, None, Some(1000)) {
            Err(crate::config::ConfigError::Invalid(_)) => {}
            _ => panic!("real serving must reject a missing tokenizer path"),
        }
    }

    #[test]
    fn real_serving_with_unloadable_tokenizer_fails() {
        let path = std::path::PathBuf::from("/nonexistent/does-not-exist/tokenizer.json");
        match resolve_serving_tokenizer(true, Some(&path), Some(1000)) {
            Err(crate::config::ConfigError::Invalid(_)) => {}
            _ => panic!("real serving must reject an unloadable tokenizer"),
        }
    }

    #[test]
    fn synthetic_serving_without_tokenizer_uses_byte_fallback() {
        let tok = match resolve_serving_tokenizer(false, None, None) {
            Ok(t) => t,
            Err(_) => panic!("synthetic mode must keep the byte-tokenizer fallback"),
        };
        assert_eq!(tok.vocab_size(), 256, "byte tokenizer expected in synthetic mode");
    }

    #[test]
    fn parses_basic_json_numbers() {
        let body = r#"{ "num_experts": 8, "d_model": 4096, "d_ff": 14336, "top_k": 2, "expert_size": 92274688 }"#;
        assert_eq!(parse_json_number(body, "num_experts"), Some(8));
        assert_eq!(parse_json_number(body, "d_model"), Some(4096));
        assert_eq!(parse_json_number(body, "d_ff"), Some(14336));
        assert_eq!(parse_json_number(body, "top_k"), Some(2));
        assert_eq!(parse_json_number(body, "expert_size"), Some(92274688));
        assert_eq!(parse_json_number(body, "missing"), None);
    }

    #[test]
    fn parses_pretty_printed_json() {
        let body = "{\n  \"num_experts\" : 16,\n  \"d_model\" : 512\n}";
        assert_eq!(parse_json_number(body, "num_experts"), Some(16));
        assert_eq!(parse_json_number(body, "d_model"), Some(512));
    }

    #[test]
    fn parses_gate_layer_index_only_for_exact_pattern() {
        assert_eq!(parse_gate_layer_index("gate_0.bin"), Some(0));
        assert_eq!(parse_gate_layer_index("gate_31.bin"), Some(31));
        // Anything that isn't exactly `gate_<digits>.bin` is ignored.
        assert_eq!(parse_gate_layer_index("gate_.bin"), None);
        assert_eq!(parse_gate_layer_index("gate_1x.bin"), None);
        assert_eq!(parse_gate_layer_index("gate_1.bin.bak"), None);
        assert_eq!(parse_gate_layer_index("rms_moe_1.bin"), None);
        assert_eq!(parse_gate_layer_index("gate.bin"), None);
    }

    #[test]
    fn parse_order_file_strips_inline_comments() {
        let path = tempdir_unique("order-inline-comments.txt");
        std::fs::write(&path, "# full-line comment\n12  # hot expert\n3,4\n8 9\n").unwrap();
        let ids = parse_order_file(&path).unwrap();
        assert_eq!(ids, vec![12, 3, 4, 8, 9]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn repack_order_validation_rejects_duplicate_ids() {
        let err = validate_order(&[0, 1, 1], 4).unwrap_err();
        assert!(
            err.contains("duplicate expert id 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn repack_order_validation_rejects_out_of_range_ids() {
        let err = validate_order(&[0, 4], 4).unwrap_err();
        assert!(err.contains("out of range"), "unexpected error: {err}");
        assert!(err.contains("4"), "unexpected error: {err}");
    }

    #[test]
    fn repack_order_validation_allows_subsets() {
        assert!(validate_order(&[0, 2], 4).is_ok());
    }

    #[test]
    fn load_gate_weights_concatenates_directory_in_layer_order() {
        // Global num_experts=2 spread over 2 layers (1 expert/layer) with
        // d_model=2, so each per-layer file holds [1 expert × 2 d_model]
        // = 2 f32s, and the concatenation is 2 × 2 = 4 f32s = the expected
        // [num_experts × d_model] matrix. Written out of order on disk to
        // prove discovery sorts by layer index.
        let dir = tempdir_unique("gate-dir");
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, vals: &[f32]| {
            let mut bytes = Vec::new();
            for v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join(name), bytes).unwrap();
        };
        // Intentionally write layer 1 before layer 0 and add a decoy.
        write("gate_1.bin", &[3.0, 4.0]);
        write("gate_0.bin", &[1.0, 2.0]);
        write("notes.txt", &[]);

        let gate = load_gate_weights(
            &dir, /*num_experts=*/ 2, /*d_model=*/ 2, /*top_k=*/ 1,
        )
        .expect("directory gate load should succeed");
        // Concatenation must be layer 0 then layer 1.
        assert_eq!(gate.weights, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gate.num_experts, 2);
        assert_eq!(gate.d_model, 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_gate_weights_errors_on_empty_directory() {
        let dir = tempdir_unique("gate-empty");
        std::fs::create_dir_all(&dir).unwrap();
        let err = load_gate_weights(&dir, 2, 2, 1).unwrap_err();
        assert!(
            err.to_string().contains("no gate_<layer>.bin files"),
            "unexpected error: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bench_real_forward_count_matches_prompt_decode_contract() {
        assert_eq!(bench_real_expected_forward_evaluations(15, 16), 30);
        assert_eq!(bench_real_expected_forward_evaluations(1, 1), 1);
        assert_eq!(bench_real_expected_forward_evaluations(1, 4), 4);
        assert_eq!(bench_real_expected_forward_evaluations(4, 1), 4);
        assert_eq!(bench_real_expected_forward_evaluations(0, 4), 0);
        assert_eq!(bench_real_expected_forward_evaluations(4, 0), 0);
    }

    #[test]
    fn bench_real_percentile_reads_sorted_microseconds() {
        let values = vec![100, 200, 300, 400, 500];
        assert_eq!(percentile_us(&values, 0.0), 100);
        assert_eq!(percentile_us(&values, 0.50), 300);
        assert_eq!(percentile_us(&values, 0.95), 500);
        assert_eq!(percentile_us(&values, 1.0), 500);
        assert_eq!(percentile_us_to_ms(&values, 0.50), 0.3);
    }

    #[test]
    fn bench_real_request_json_supports_chat_messages_and_max_tokens() {
        let path = tempdir_unique("bench-real-request.json");
        std::fs::write(
            &path,
            r#"{
                "messages": [
                    { "role": "system", "content": "Be brief." },
                    { "role": "user", "content": "Explain caches." }
                ],
                "max_tokens": 7
            }"#,
        )
        .unwrap();
        let args = BenchRealArgs {
            config: PathBuf::from("config.toml"),
            prompt: None,
            request_json: Some(path.clone()),
            output_tokens: None,
            warmup_runs: 0,
            measured_runs: 1,
            cache_reset: BenchRealCacheReset::Keep,
            greedy: true,
            format: BenchRealOutputFormat::Json,
        };
        let input = load_bench_real_input(&args).unwrap();
        assert_eq!(input.output_tokens, 7);
        assert!(input.prompt.contains("system: Be brief."));
        assert!(input.prompt.contains("user: Explain caches."));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bench_real_cli_output_tokens_override_request_json() {
        let path = tempdir_unique("bench-real-request-override.json");
        std::fs::write(&path, r#"{ "prompt": "hello", "max_tokens": 7 }"#).unwrap();
        let args = BenchRealArgs {
            config: PathBuf::from("config.toml"),
            prompt: None,
            request_json: Some(path.clone()),
            output_tokens: Some(3),
            warmup_runs: 0,
            measured_runs: 1,
            cache_reset: BenchRealCacheReset::Keep,
            greedy: true,
            format: BenchRealOutputFormat::Json,
        };
        let input = load_bench_real_input(&args).unwrap();
        assert_eq!(input.prompt, "hello");
        assert_eq!(input.output_tokens, 3);
        let _ = std::fs::remove_file(path);
    }

    /// Tiny unique temp-dir helper (avoids pulling a dev-dependency for
    /// these filesystem tests).
    fn tempdir_unique(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }

    /// Minimal `Config` for exercising [`reconcile_bench_real_config`]. The
    /// non-`real_transformer` sections carry placeholder values; the tests
    /// only inspect fields the reconciler overwrites.
    fn minimal_bench_cfg() -> crate::config::Config {
        use crate::config::*;
        Config {
            server: ServerConfig {
                bind: "127.0.0.1:8080".into(),
                max_tokens: 64,
                session_ttl_secs: 0,
                max_concurrent_requests: 0,
                admission_min_free_blocks: 0,
            },
            model: ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 64,
                d_ff: 256,
                expert_size: 4096,
                num_layers: 1,
                dtype: WeightDtype::F32,
            },
            storage: StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: false,
                predict_fanout: 2,
                pipeline_depth: crate::engine::DEFAULT_PIPELINE_DEPTH,
                predict_min_prob: 0.0,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
                packed_blob: None,
                packed_manifest: None,
            },
            tokenizer: TokenizerConfig::default(),
            real_transformer: RealTransformerConfig::default(),
            sampling: SamplingConfig::default(),
            predictive: PredictiveConfig::default(),
            security: SecurityConfig::default(),
            gpu_cache: GpuCacheConfig::default(),
            distributed: DistributedConfig::default(),
        }
    }

    /// A distinctive tiny Qwen3-MoE `config.json` (checkpoint dims differ from
    /// [`minimal_bench_cfg`]) with `norm_topk_prob = true`.
    fn write_qwen3_moe_config_json(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let body = serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 100,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "rope_theta": 5_000_000.0,
            "rms_norm_eps": 1e-5
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn explicit_architecture_still_reconciles_config_json() {
        // D. Explicit TOML architecture must not suppress config.json
        // reconciliation of advanced fields and dimensions.
        let dir = tempdir_unique("bench-real-qwen3moe-cfg");
        write_qwen3_moe_config_json(&dir);
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.weights_dir = Some(dir.clone());
        cfg.real_transformer.architecture = Some("qwen3_moe".to_string());

        let (arch, _first_k, advanced) =
            reconcile_bench_real_config(&mut cfg).expect("reconcile should succeed");

        assert_eq!(arch, crate::architecture::Architecture::Qwen3Moe);
        assert!(
            advanced.norm_topk_prob,
            "checkpoint norm_topk_prob=true must be reconciled even with explicit architecture"
        );
        // Dimensions come from config.json, not the placeholder TOML.
        assert_eq!(cfg.model.d_model, 32);
        assert_eq!(cfg.model.num_layers, 3);
        assert_eq!(cfg.model.num_experts, 4);
        assert_eq!(cfg.real_transformer.num_heads, 4);
        assert_eq!(cfg.real_transformer.num_kv_heads, 2);
        assert_eq!(cfg.real_transformer.head_dim, 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_architecture_mismatch_is_rejected() {
        // E. A TOML architecture that disagrees with config.json is a hard
        // configuration error naming both architectures.
        let dir = tempdir_unique("bench-real-arch-mismatch");
        write_qwen3_moe_config_json(&dir);
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.weights_dir = Some(dir.clone());
        cfg.real_transformer.architecture = Some("mixtral".to_string());

        let err = reconcile_bench_real_config(&mut cfg)
            .expect_err("mismatched architecture must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("mixtral"), "error must name TOML arch: {msg}");
        assert!(
            msg.contains("qwen3_moe"),
            "error must name checkpoint arch: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Finding 7: resolved-config architecture-aware validation ----

    #[test]
    fn resolved_config_validation_accepts_reconciled_checkpoint() {
        let dir = tempdir_unique("bench-real-f7-ok");
        write_qwen3_moe_config_json(&dir);
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.weights_dir = Some(dir.clone());
        let (arch, first_k, advanced) =
            reconcile_bench_real_config(&mut cfg).expect("reconcile should succeed");
        validate_resolved_bench_real_config(&cfg, arch, first_k, &advanced)
            .expect("a well-formed reconciled checkpoint must pass validation");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolved_config_validation_rejects_top_k_over_num_experts() {
        let mut cfg = minimal_bench_cfg();
        cfg.model.num_experts = 4;
        cfg.model.top_k = 8; // impossible: more activated than available
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_bench_real_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("top_k > num_experts must fail");
        assert!(err.to_string().contains("top_k"), "{err}");
    }

    #[test]
    fn resolved_config_validation_rejects_gqa_head_indivisibility() {
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.num_heads = 6;
        cfg.real_transformer.num_kv_heads = 4; // 6 % 4 != 0
        cfg.real_transformer.head_dim = 8;
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_bench_real_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("non-divisible GQA head counts must fail");
        assert!(err.to_string().contains("num_kv_heads"), "{err}");
    }

    #[test]
    fn resolved_config_validation_allows_asymmetric_v_head_dim() {
        // MiMo-V2-Flash style: v_head_dim != head_dim must NOT be rejected,
        // and the invalid universal num_heads*head_dim==d_model rule must not
        // be applied.
        let mut cfg = minimal_bench_cfg();
        cfg.model.d_model = 64;
        cfg.real_transformer.num_heads = 4;
        cfg.real_transformer.num_kv_heads = 2;
        cfg.real_transformer.head_dim = 24; // 4*24 = 96 != d_model 64
        let mut advanced = crate::model::AdvancedConfig::default();
        advanced.v_head_dim = Some(16);
        validate_resolved_bench_real_config(
            &cfg,
            crate::architecture::Architecture::MiMoV2,
            0,
            &advanced,
        )
        .expect("asymmetric V geometry must be accepted");
    }

    #[test]
    fn resolved_config_validation_rejects_undersized_cache() {
        let mut cfg = minimal_bench_cfg();
        cfg.model.top_k = 4;
        cfg.storage.cache_slots = 2; // smaller than activated per layer
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_bench_real_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("cache smaller than the per-layer working set must fail");
        assert!(err.to_string().contains("cache_slots"), "{err}");
    }

    #[test]
    fn resolved_config_validation_rejects_odirect_misaligned_expert_size() {
        let mut cfg = minimal_bench_cfg();
        cfg.storage.no_direct = false;
        cfg.storage.block_align = 4096;
        cfg.model.expert_size = 4097; // not a multiple of block_align
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_bench_real_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("O_DIRECT misaligned expert_size must fail");
        assert!(err.to_string().contains("block_align"), "{err}");
    }
}
