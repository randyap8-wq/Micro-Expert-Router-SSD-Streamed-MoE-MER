//! Micro-Expert-Router — MoE execution engine that hot-swaps experts from a
//! PCIe-attached NVMe drive into pre-allocated, page-aligned RAM via
//! `O_DIRECT` `pread(2)` (dispatched off the Tokio runtime with
//! `block_in_place`).
//!
//! See README.md at the repository root for architecture and design notes.

mod aligned_buffer;
mod batch_scheduler;
mod buffer_pool;
mod config;
mod engine;
mod expert_cache;
mod gating;
mod inference;
mod io_provider;
#[cfg(all(feature = "io_uring", target_os = "linux"))]
mod io_uring_storage;
mod metrics;
mod model;
mod multi_layer_cache;
mod router;
mod sampling;
mod server;
mod session;
mod tokenizer;
mod transformer;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::buffer_pool::BufferPool;
use crate::engine::{Engine, EngineOptions, ModelShape};
use crate::expert_cache::ExpertCache;
use crate::inference::expert_weight_bytes;
use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
use crate::router::{PredictiveLoader, TopKRouter};

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
        /// On-disk weight dtype: `f32` (default) or `f16`. Selects the
        /// byte width of every weight in the generated files.
        #[arg(long, default_value = "f32")]
        dtype: String,
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
        /// Don't prefetch below this transition probability.
        #[arg(long, default_value_t = 0.05)]
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
        /// On-disk weight dtype: `f32` (default, 4 B/weight) or `f16`
        /// (2 B/weight). Halving the byte width halves SSD-read bytes
        /// per cache miss, which is the dominant energy term in this
        /// engine. Must match what `gen-data` was invoked with.
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
        #[arg(long)]
        gate_weights: Option<PathBuf>,
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
    match cli.cmd {
        Cmd::GenData {
            data_dir,
            num_experts,
            expert_size,
            d_model,
            d_ff,
            block_align,
            dtype,
        } => cmd_gen_data(&data_dir, num_experts, expert_size, d_model, d_ff, block_align, &dtype),
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
                } = cli.cmd
                {
                    let dtype = crate::inference::WeightDtype::from_str_opt(&dtype)
                        .ok_or_else(|| format!("--dtype: unknown value {dtype:?} (use 'f32' or 'f16')"))?;
                    cmd_run(RunArgs {
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
                    })
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
    }
}

async fn cmd_serve(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use crate::config::Config;
    use crate::metrics::Metrics;
    use crate::server::{serve, AppState};
    use crate::tokenizer::Tokenizer;

    // Best-effort NUMA-local pinning. Off by default; opt in by
    // setting `MER_PIN_CORES=N` in the environment. Useful for
    // reducing cross-socket DRAM hops on multi-NUMA hosts under
    // sustained load. Has no effect on non-Linux builds.
    if let Some(n) = std::env::var("MER_PIN_CORES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        if let Err(e) = pin_to_local_cores(n) {
            warn!(error = %e, "MER_PIN_CORES set but pinning failed; continuing without affinity");
        }
    }

    let cfg = Config::from_file(&config_path)?;
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

    if !cfg.model.data_dir.is_dir() {
        return Err(format!(
            "data dir {} does not exist; run `gen-data` or the extractor first",
            cfg.model.data_dir.display()
        )
        .into());
    }

    let storage = Arc::new(NvmeStorage::new(StorageConfig {
        base_path: cfg.model.data_dir.clone(),
        expert_size: cfg.model.expert_size,
        block_align: cfg.storage.block_align,
        use_direct_io: !cfg.storage.no_direct,
    })?);
    storage.warmup_fds(0..cfg.model.num_experts)?;

    let prefetch_headroom = cfg.storage.predict_fanout.max(1);
    let pool_slots = cfg.storage.cache_slots + prefetch_headroom;
    let pool = BufferPool::new(pool_slots, cfg.model.expert_size, cfg.storage.block_align);
    let cache = Arc::new(ExpertCache::new(cfg.storage.cache_slots));

    let router = Arc::new(TopKRouter::clustered(
        cfg.model.num_experts,
        cfg.model.top_k,
        4,
        0.9,
        0xC0FFEE,
    ));
    let predictor = Arc::new(PredictiveLoader::new(
        cfg.model.num_experts,
        cfg.storage.predict_fanout,
        cfg.storage.predict_min_prob,
        0xC0FFEE,
    ));

    let engine = Arc::new(Engine::with_options(
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
        },
    ));

    let tokenizer = match cfg.tokenizer.path.as_ref() {
        Some(p) => match Tokenizer::from_file(p) {
            Ok(t) => Arc::new(t),
            Err(e) => {
                warn!(path = %p.display(), error = %e, "tokenizer load failed; falling back to byte tokenizer");
                Arc::new(Tokenizer::bytes())
            }
        },
        None => {
            info!("no tokenizer.json configured; using byte-level fallback tokenizer");
            Arc::new(Tokenizer::bytes())
        }
    };

    // Optional real-transformer pipeline. When enabled, every request
    // runs `embedding -> stacked layers (each with SSD-streamed MoE) ->
    // LM head -> argmax`; when disabled, the legacy benchmark generator
    // is used (the engine still streams expert FFN compute either way).
    let (real_model, batch_scheduler) = if cfg.real_transformer.enabled {
        let rt = &cfg.real_transformer;
        let head_dim = if rt.head_dim == 0 {
            cfg.model.d_model / rt.num_heads
        } else {
            rt.head_dim
        };
        let num_kv_heads = if rt.num_kv_heads == 0 { rt.num_heads } else { rt.num_kv_heads };
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
            window_size: if rt.window_size == 0 { None } else { Some(rt.window_size) },
        };
        let m = match rt.weights_dir.as_ref() {
            Some(dir) => crate::model::RealModel::from_dir(model_cfg, dir, rt.seed)?,
            None => crate::model::RealModel::new_seeded(model_cfg, rt.seed),
        };
        let model_arc = Arc::new(m);
        let batch_cfg = crate::batch_scheduler::BatchConfig {
            max_batch_size: rt.max_batch_size,
            batch_timeout: std::time::Duration::from_millis(rt.batch_timeout_ms),
        };
        let scheduler = crate::batch_scheduler::BatchScheduler::spawn(
            model_arc.clone(),
            engine.clone(),
            batch_cfg,
        );
        info!(
            num_layers = cfg.model.num_layers,
            num_heads = rt.num_heads,
            num_kv_heads,
            head_dim,
            vocab_size = rt.vocab_size,
            max_batch_size = rt.max_batch_size,
            batch_timeout_ms = rt.batch_timeout_ms,
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
        let sweep = std::time::Duration::from_secs(
            (cfg.server.session_ttl_secs / 2).max(1).min(60),
        );
        store.spawn_evictor(sweep);
        Some(store)
    } else {
        None
    };

    let state = AppState {
        engine,
        tokenizer,
        metrics: Metrics::new(),
        max_tokens_cap: cfg.server.max_tokens,
        real_model,
        batch_scheduler,
        default_sampling: cfg.sampling.to_params(),
        sessions,
    };
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
        format!("--dtype: unknown value {dtype_str:?} (use 'f32', 'f16', or 'int8')")
    })?;
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
        data_dir, num_experts, expert_size, d_model, d_ff, dtype,
    )?;
    let total_bytes = num_experts as u64 * expert_size as u64;
    info!(
        elapsed_s = started.elapsed().as_secs_f64(),
        total_mib = total_bytes as f64 / (1024.0 * 1024.0),
        "expert files written"
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
}

async fn cmd_run(mut args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    // 0) If `metadata.json` exists alongside the expert blobs (e.g. as
    //    written by `scripts/extract_mixtral_experts.py`), use it to fill
    //    in any args the user didn't override on the command line. We
    //    detect "user didn't override" by comparing against clap defaults
    //    — anyone who actually passes a flag overrides the metadata.
    apply_metadata_if_present(&mut args);

    let weight_bytes = expert_weight_bytes(args.d_model, args.d_ff);
    info!(
        num_experts = args.num_experts,
        top_k = args.top_k,
        cache_slots = args.cache_slots,
        expert_mib = args.expert_size as f64 / (1024.0 * 1024.0),
        d_model = args.d_model,
        d_ff = args.d_ff,
        weight_mib = weight_bytes as f64 / (1024.0 * 1024.0),
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
    if weight_bytes > args.expert_size {
        return Err(format!(
            "expert_size ({}) is too small for the SwiGLU weights of d_model={}, \
             d_ff={} ({} bytes). Increase --expert-size or shrink --d-model / --d-ff \
             so it matches what gen-data wrote.",
            args.expert_size, args.d_model, args.d_ff, weight_bytes
        )
        .into());
    }
    if !args.data_dir.is_dir() {
        return Err(format!(
            "data dir {} does not exist; run `gen-data` first",
            args.data_dir.display()
        )
        .into());
    }

    if args.io_uring {
        // Best-effort affinity: keep the engine on the NUMA node that
        // owns CPU 0 to avoid cross-socket DRAM hops on every io_uring
        // completion. Honored only on Linux. Configurable via the
        // `MER_PIN_CORES` env var (number of cores to pin to); defaults
        // to `min(8, available_parallelism)` when unset, which is a
        // reasonable starting point for a single-socket workstation.
        let n = std::env::var("MER_PIN_CORES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8)
            });
        if let Err(e) = pin_to_local_cores(n) {
            warn!(error = %e, "could not set CPU affinity (continuing without pinning)");
        }
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
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

    let storage = Arc::new(NvmeStorage::new(StorageConfig {
        base_path: args.data_dir.clone(),
        expert_size: args.expert_size,
        block_align: args.block_align,
        use_direct_io: !args.no_direct,
    })?);
    storage.warmup_fds(0..args.num_experts)?;

    let prefetch_headroom = if args.no_prefetch { 0 } else { args.predict_fanout.max(1) };
    let pool_slots = args.cache_slots + prefetch_headroom;

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
                 Lower --cache-slots / --predict-fanout or risk OOM / heavy swapping.",
                pool_slots,
                args.expert_size as f64 / (1024.0 * 1024.0)
            );
        }
    }

    info!(
        cache_slots = args.cache_slots,
        pool_slots = pool_slots,
        prefetch_headroom = prefetch_headroom,
        "buffer pool sized with prefetch headroom"
    );
    let pool = BufferPool::new(pool_slots, args.expert_size, args.block_align);
    let cache = Arc::new(ExpertCache::new(args.cache_slots));

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
        if args.no_prefetch { 0 } else { args.predict_fanout },
        args.predict_min_prob,
        args.seed,
    ));

    let engine = Arc::new({
        let base = Engine::with_options(
            cache.clone(),
            pool.clone(),
            storage.clone(),
            router.clone(),
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
            },
        );
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

    for t in 0..args.tokens {
        let start = Instant::now();
        let stats = if let Some(gate) = gate.as_ref() {
            // Real gating-network path. Hidden state is the same
            // synthetic activation `Engine::generate` would have used,
            // so the only difference relative to the legacy path is
            // *which* experts are selected.
            let hidden = crate::inference::synth_hidden_state(t, args.d_model, args.seed);
            let dec = gate.route(&hidden);
            let pre_hits = engine.report().hits;
            let pre_misses = engine.report().misses;
            let pre_bytes = engine.report().bytes_read;
            let _ = engine.moe_step(t, &hidden, &dec.experts).await;
            let post = engine.report();
            crate::engine::CycleStats {
                hits: post.hits.saturating_sub(pre_hits),
                misses: post.misses.saturating_sub(pre_misses),
                prefetch_hits: 0,
                bytes_read: post.bytes_read.saturating_sub(pre_bytes),
            }
        } else {
            engine.generate(t).await
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
fn load_gate_weights(
    path: &std::path::Path,
    num_experts: usize,
    d_model: usize,
    top_k: usize,
) -> Result<crate::gating::LinearGate, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read gate weights {}: {e}", path.display()))?;
    let expected = num_experts
        .checked_mul(d_model)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "num_experts * d_model overflowed".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "gate weights file {} has {} bytes, expected {} ({} experts × {} d_model × 4 bytes/f32)",
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
    Ok(crate::gating::LinearGate::new(weights, num_experts, d_model, top_k))
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
    if out.is_empty() { None } else { Some(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
