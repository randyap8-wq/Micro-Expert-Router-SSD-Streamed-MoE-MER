//! Micro-Expert-Router — MoE execution engine that hot-swaps experts from a
//! PCIe-attached NVMe drive into pre-allocated, page-aligned RAM via
//! `O_DIRECT` `pread(2)` (dispatched off the Tokio runtime with
//! `block_in_place`).
//!
//! See README.md at the repository root for architecture and design notes.

mod aligned_buffer;
mod buffer_pool;
mod engine;
mod expert_cache;
mod inference;
mod io_provider;
mod router;

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
        } => cmd_gen_data(&data_dir, num_experts, expert_size, d_model, d_ff, block_align),
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
                    token_pause_us,
                    first_token,
                    no_prefetch,
                    io_only,
                    force_ssd,
                    router_clusters,
                    router_intra_p,
                    router_matrix,
                } = cli.cmd
                {
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
                        token_pause_us,
                        first_token,
                        no_prefetch,
                        io_only,
                        force_ssd,
                        router_clusters,
                        router_intra_p,
                        router_matrix,
                    })
                    .await
                } else {
                    unreachable!()
                }
            })
        }
    }
}

fn cmd_gen_data(
    data_dir: &std::path::Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    block_align: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let weight_bytes = expert_weight_bytes(d_model, d_ff);
    if weight_bytes > expert_size {
        return Err(format!(
            "expert_size ({expert_size}) is too small for the SwiGLU weights of \
             d_model={d_model}, d_ff={d_ff} ({weight_bytes} bytes). Increase \
             --expert-size or shrink --d-model / --d-ff."
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
        weight_mib = weight_bytes as f64 / (1024.0 * 1024.0),
        "generating synthetic SwiGLU expert weights"
    );
    let started = Instant::now();
    generate_synthetic_experts(data_dir, num_experts, expert_size, d_model, d_ff)?;
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
    token_pause_us: u64,
    first_token: Vec<u32>,
    no_prefetch: bool,
    io_only: bool,
    force_ssd: bool,
    router_clusters: usize,
    router_intra_p: f64,
    router_matrix: Option<PathBuf>,
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

    let engine = Arc::new(Engine::with_options(
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
        EngineOptions { io_only: args.io_only },
    ));

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
    for t in 0..args.tokens {
        let start = Instant::now();
        let stats = engine.generate(t).await;
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
