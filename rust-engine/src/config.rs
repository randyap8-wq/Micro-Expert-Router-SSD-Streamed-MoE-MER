//! TOML configuration for the production server (gist Phase 8).
//!
//! Replaces the long-tail of CLI flags with a single config file. The
//! existing CLI subcommands (`gen-data`, `run`) keep working unchanged
//! — `serve --config <path>` is the new entry point that reads this
//! struct.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub use crate::inference::WeightDtype;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP bind address, e.g. `127.0.0.1:8080`.
    #[serde(default = "default_bind")]
    pub bind: String,

    /// Maximum tokens any one request is allowed to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Idle TTL for KV-cache sessions, in seconds. `0` (default)
    /// disables the persistent session store entirely (every request
    /// is stateless, matching the legacy behaviour). When > 0,
    /// requests carrying `"session_id": "<id>"` will resume from the
    /// session's saved KV cache, and a background task evicts
    /// sessions idle for longer than this many seconds.
    #[serde(default)]
    pub session_ttl_secs: u64,

    /// Maximum number of HTTP requests allowed to be executing
    /// concurrently. Excess requests are rejected with `503 Service
    /// Unavailable` so the engine never silently degrades into
    /// unbounded queueing. `0` (default) disables the limit.
    #[serde(default)]
    pub max_concurrent_requests: usize,

    /// Minimum number of free blocks the paged-KV pool must hold for
    /// new requests to be admitted. When configured (and a block pool
    /// is configured under `[real_transformer]`), incoming requests
    /// that would push the pool below this watermark are rejected
    /// with `503 Service Unavailable`. `0` (default) disables.
    #[serde(default)]
    pub admission_min_free_blocks: usize,
}

fn default_bind() -> String { "127.0.0.1:8080".to_string() }
fn default_max_tokens() -> usize { 256 }

/// Optional API-key gate + simple in-process rate limiting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Bearer / `X-API-Key` values that may access the API. When
    /// empty, the API-key middleware is disabled and every request
    /// is allowed (legacy behaviour). When non-empty, requests
    /// missing or carrying a value not in this set are rejected with
    /// `401 Unauthorized`. Plumb in distinct keys per tenant so the
    /// access log identifies who issued each request.
    #[serde(default)]
    pub api_keys: Vec<String>,

    /// Optional in-process token-bucket rate limit, expressed in
    /// requests per second per API key. `0` (default) disables.
    /// Bursts of up to `rate_limit_burst` are allowed before the
    /// bucket empties.
    #[serde(default)]
    pub rate_limit_rps: u32,

    /// Burst size for the token bucket. `0` defaults to `rate_limit_rps`.
    #[serde(default)]
    pub rate_limit_burst: u32,

    /// Path to a PEM-encoded TLS certificate. **Production setups
    /// should usually terminate TLS at a reverse proxy** (nginx,
    /// Envoy, AWS ALB) — these knobs exist to document the intended
    /// HTTPS deployment shape for closed-network setups where the
    /// engine binary needs to serve HTTPS directly. See
    /// `docs/production.md`.
    ///
    /// This release does *not* link rustls into the binary. When both
    /// `tls_cert` and `tls_key` are set, [`Config::validate`] logs a
    /// `WARN` and the server continues to bind plain HTTP. Setting
    /// only one of the two is a hard validation error. Wiring rustls
    /// is a one-line `axum_server::bind_rustls` once the deployment
    /// is ready.
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,
    #[serde(default)]
    pub tls_key: Option<PathBuf>,
}

/// Server-wide sampling defaults. Each request can override these via
/// the `temperature` / `top_p` / `top_k` / `seed` JSON fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub seed: u64,
}

fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: 0,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    /// Convert to runtime [`crate::sampling::SamplingParams`].
    pub fn to_params(&self) -> crate::sampling::SamplingParams {
        crate::sampling::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Directory containing `expert_*.bin` files and (optionally)
    /// `metadata.json` and `tokenizer.json`.
    pub data_dir: PathBuf,

    /// Number of experts per layer.
    pub num_experts: u32,

    /// Top-K experts activated per token.
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Hidden / residual-stream dimension.
    pub d_model: usize,

    /// FFN intermediate dimension.
    pub d_ff: usize,

    /// Bytes per expert file (must be a multiple of `block_align`).
    pub expert_size: usize,

    /// Number of transformer layers (1 for the legacy single-layer mode,
    /// 32 for full Mixtral).
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    /// On-disk weight dtype. `f32` (default) reinterprets bytes as
    /// `&[f32]` directly; `f16` halves bytes-per-parameter (and hence
    /// SSD read energy) at the cost of a small dequantisation step on
    /// every fetch.
    #[serde(default = "default_dtype")]
    pub dtype: WeightDtype,
}

fn default_dtype() -> WeightDtype { WeightDtype::F32 }

fn default_top_k() -> usize { 2 }
fn default_num_layers() -> usize { 1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfigToml {
    /// LRU cache slots **per layer**.
    #[serde(default = "default_cache_slots")]
    pub cache_slots: usize,

    /// O_DIRECT block alignment.
    #[serde(default = "default_block_align")]
    pub block_align: usize,

    /// Disable O_DIRECT (required on tmpfs / macOS / CI).
    #[serde(default)]
    pub no_direct: bool,

    /// Predictive prefetcher fanout (0 disables prefetching entirely).
    #[serde(default = "default_predict_fanout")]
    pub predict_fanout: usize,

    /// Don't prefetch below this transition probability.
    #[serde(default = "default_predict_min_prob")]
    pub predict_min_prob: f64,

    /// Fraction of input dimensions to load per expert when partial
    /// column loading is enabled. `1.0` (default) loads every column
    /// (legacy behaviour); values in `[0.1, 1.0]` load only the top-M
    /// columns of `x` by absolute magnitude, reducing bytes read per
    /// miss in proportion to the chosen fraction.
    #[serde(default = "default_partial_load_fraction")]
    pub partial_load_fraction: f64,

    /// After an expert has been observed as a routing destination this
    /// many times, pin it permanently in the LRU cache so it is never
    /// evicted by cold experts. `0` (default) disables pinning.
    #[serde(default = "default_pin_after_observations")]
    pub pin_after_observations: u64,
}

fn default_partial_load_fraction() -> f64 { 1.0 }
fn default_pin_after_observations() -> u64 { 0 }

fn default_cache_slots() -> usize { 4 }
fn default_block_align() -> usize { 4096 }
fn default_predict_fanout() -> usize { 2 }
fn default_predict_min_prob() -> f64 { 0.05 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Optional path to a HuggingFace `tokenizer.json`. If omitted, the
    /// engine falls back to a deterministic byte tokenizer.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self { path: None }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RealTransformerConfig {
    /// When `true`, the server runs requests through the full real
    /// transformer (`embedding → stacked layers → LM head`), with each
    /// layer's MoE block streaming its routed experts from SSD via the
    /// engine. When `false` (default), the legacy benchmark generator is
    /// used: the engine still streams experts, but next-token ids are
    /// synthesised from cycle stats — so the SSD-streaming substrate is
    /// exercised either way.
    #[serde(default)]
    pub enabled: bool,
    /// Optional directory containing pre-extracted dense weight files
    /// (`embed.bin`, `attn_<L>_q.bin`, `gate_<L>.bin`, …). Tensors not
    /// present fall back to a deterministic seeded initialisation, so
    /// the engine always has an end-to-end runnable path. See
    /// `crate::model::RealModel::from_dir` for the file-name schema.
    #[serde(default)]
    pub weights_dir: Option<PathBuf>,
    /// Vocab size. Must match the tokenizer when one is configured (for
    /// the byte fallback this should be 256).
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Number of attention heads.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    /// Grouped-Query-Attention KV head count. `0` = auto (set equal to
    /// `num_heads`, recovering vanilla MHA).
    #[serde(default)]
    pub num_kv_heads: usize,
    /// Per-head dimension. `0` = auto (`d_model / num_heads`).
    #[serde(default)]
    pub head_dim: usize,
    /// RoPE base θ (Mixtral / Llama-3 default = 10000.0; Llama-3.1 uses
    /// 500000.0 for long-context).
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    /// RMSNorm ε.
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f32,
    /// PRNG seed for the deterministic init fallback.
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// Continuous-batching: maximum number of in-flight requests fused
    /// into a single decoder step. The background scheduler waits up to
    /// [`Self::batch_timeout_ms`] for additional requests to arrive
    /// before flushing the current batch. `1` disables batching (each
    /// request runs its decoder step independently). Only takes effect
    /// when `enabled = true`.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Continuous-batching: how long the scheduler waits for more
    /// requests to join a not-yet-full batch, in milliseconds. A short
    /// timeout (e.g. 5 ms) trades a small amount of per-token latency
    /// for substantially better throughput when concurrent requests
    /// arrive. Only takes effect when `enabled = true`.
    #[serde(default = "default_batch_timeout_ms")]
    pub batch_timeout_ms: u64,

    /// Multi-tenant fair-share: cutoff (in milliseconds) after which
    /// a session's KV blocks become candidates for
    /// `BatchScheduler::evict_idle_blocks` when the block pool is
    /// above its 90 % soft-cap. Default: 5 000 ms (5 s), matching
    /// the gist's multi-tenant hardening requirement. Set to a
    /// larger value if your workload has natural mid-stream pauses
    /// (e.g. tool calls) that should not trigger reclamation.
    #[serde(default = "default_idle_eviction_threshold_ms")]
    pub idle_eviction_threshold_ms: u64,

    /// Baseline speculation depth (tokens-ahead) the scheduler's
    /// `SpeculationController` starts from. Under rising
    /// `ssd_stall_us` telemetry the controller grows the active
    /// depth by up to `MAX_LATENCY_BUMP` (2) tokens; under
    /// `BlockPool::PressureLevel::Critical` it clamps depth to 0.
    /// Default: 1.
    #[serde(default = "default_speculation_base_depth")]
    pub speculation_base_depth: usize,

    /// Sliding-window attention span. `0` or omitted = full causal
    /// attention (backward compatible). Mixtral uses `4096`.
    #[serde(default)]
    pub window_size: usize,

    /// **Pool back-pressure: "high" threshold** (gist Part 1, fix #4).
    /// Fraction of [`block_pool::BlockPool`] primary capacity at or
    /// above which the scheduler classifies the pool as
    /// [`block_pool::PressureLevel::High`] and runs preemptive
    /// `evict_idle_blocks`. Defaults to
    /// [`block_pool::SOFT_CAP_RATIO`] (0.90) when omitted.
    #[serde(default = "default_pressure_high_threshold")]
    pub pressure_high_threshold: f32,

    /// **Pool back-pressure: "critical" threshold** (gist Part 1, fix #4).
    /// Fraction of [`block_pool::BlockPool`] primary capacity at or
    /// above which the scheduler classifies the pool as
    /// [`block_pool::PressureLevel::Critical`] and clamps the
    /// speculation depth to 0. Must be >= `pressure_high_threshold`.
    /// Defaults to [`block_pool::CRITICAL_PRESSURE_RATIO`] (0.98)
    /// when omitted.
    #[serde(default = "default_pressure_critical_threshold")]
    pub pressure_critical_threshold: f32,

    /// **Hybrid compute offload** (gist Part 2, fix #5). Picks which
    /// [`crate::backend::Backend`] implementation handles the dense
    /// transformer body's matmul / attention / LM-head. `"cpu"`
    /// (default) routes through [`crate::backend::CandleBackend`] /
    /// the auto-escalating SIMD dispatcher in [`crate::kernels`].
    /// `"gpu"` selects the [`crate::backend::GpuBackend`] integration
    /// seam. At present this is a compatibility/configuration switch
    /// for the GPU backend path rather than a guaranteed operational
    /// GPU offload mode, so execution still falls back to the CPU
    /// dense path when GPU acceleration is not active. The
    /// SSD-streamed expert pipeline stays CPU-side either way,
    /// matching the gist's "budget GPU augments CPU" posture.
    #[serde(default)]
    pub compute_offload: crate::backend::ComputeOffload,

    /// **Bounded speculative prefetches** (gist Part 1, fix #3).
    /// Maximum number of in-flight `Engine::spawn_prefetch` I/Os
    /// allowed at any one time. Each spawn acquires an owned permit
    /// from an internal semaphore; when the ceiling is saturated the
    /// prefetch is dropped and the
    /// `prefetch_dropped_concurrency` counter is incremented.
    /// Defaults to `64` (≈ a typical io_uring queue depth).
    #[serde(default = "default_max_concurrent_prefetches")]
    pub max_concurrent_prefetches: usize,

    /// **Bounded `fetch_once` yield budget** (gist feedback #1.3).
    /// Maximum number of `tokio::task::yield_now()` iterations
    /// [`Engine::fetch_once`] spins through while waiting for a free
    /// `PooledBuffer` when the expert cache is full of pinned
    /// residents. Once the limit is reached the call returns
    /// `FetchOnceError::PoolStarved` instead of yielding indefinitely.
    /// Defaults to `128` — low enough to surface pool-sizing
    /// misconfigurations as a fast error, high enough to absorb a
    /// normal burst of concurrent prefetches under steady-state load.
    #[serde(default = "default_max_fetch_yields")]
    pub max_fetch_yields: usize,

    /// **Overflow-slab cap** (gist Part 1, fix #5). Maximum number of
    /// "overflow" KV blocks the [`block_pool::BlockPool`] may
    /// allocate beyond its primary slab before
    /// [`block_pool::BlockPool::allocate`] starts returning `None`
    /// (admission back-pressure). `None` (omitted) preserves the
    /// historical unbounded growth behaviour; `Some(0)` is normalized
    /// to `None`, so it is treated as unbounded too.
    #[serde(default)]
    pub max_overflow_capacity: Option<usize>,
}

fn default_max_concurrent_prefetches() -> usize {
    crate::engine::DEFAULT_MAX_CONCURRENT_PREFETCHES
}

fn default_max_fetch_yields() -> usize {
    crate::engine::DEFAULT_MAX_FETCH_YIELDS
}

fn default_pressure_high_threshold() -> f32 { crate::block_pool::SOFT_CAP_RATIO }
fn default_pressure_critical_threshold() -> f32 { crate::block_pool::CRITICAL_PRESSURE_RATIO }

fn default_vocab_size() -> usize { 256 }
fn default_num_heads() -> usize { 8 }
fn default_rope_base() -> f32 { 10_000.0 }
fn default_rms_eps() -> f32 { 1e-6 }
fn default_seed() -> u64 { 0xC0FFEE }
fn default_max_batch_size() -> usize { 8 }
fn default_batch_timeout_ms() -> u64 { 5 }
fn default_idle_eviction_threshold_ms() -> u64 { 5_000 }
fn default_speculation_base_depth() -> usize { 1 }

/// Configuration for the predictive architecture (`[predictive]` block).
///
/// All three components are **opt-in** and default to disabled. When
/// disabled the engine runs the legacy Markov-chain-only prefetch path
/// bit-for-bit, so adding the section to an existing config never
/// silently changes behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveConfig {
    /// Enable the sliding-window [`crate::router::LocalityMonitor`].
    /// Sets the **L** arm of the speculative I/O union.
    #[serde(default)]
    pub locality_enabled: bool,
    /// Number of recent activations the locality window remembers.
    /// 256 ≈ a few dozen tokens at top-K = 8.
    #[serde(default = "default_locality_window")]
    pub locality_window: usize,
    /// Heat threshold (fraction of the window) above which an expert
    /// is considered hot. `0.10` matches
    /// [`crate::router::LocalityMonitor::DEFAULT_THRESHOLD_PCT`].
    #[serde(default = "default_locality_threshold")]
    pub locality_threshold_pct: f32,

    /// Enable the [`crate::router::NeuralSpeculator`] (M arm).
    #[serde(default)]
    pub speculator_enabled: bool,
    /// Hidden dimension of the speculator's 2-layer MLP. The spec
    /// recommends 128.
    #[serde(default = "default_speculator_hidden")]
    pub speculator_hidden_dim: usize,
    /// Top-K size pulled from the speculator at every routing
    /// decision. Defaults to the router's `top_k` when zero.
    #[serde(default)]
    pub speculator_top_k: usize,
}

fn default_locality_window() -> usize { 256 }
fn default_locality_threshold() -> f32 { 0.10 }
fn default_speculator_hidden() -> usize { 128 }

impl Default for PredictiveConfig {
    fn default() -> Self {
        Self {
            locality_enabled: false,
            locality_window: default_locality_window(),
            locality_threshold_pct: default_locality_threshold(),
            speculator_enabled: false,
            speculator_hidden_dim: default_speculator_hidden(),
            speculator_top_k: 0,
        }
    }
}

/// `[gpu_cache]` — Phase 1/2 of the 3-Tier Heterogeneous Memory
/// Orchestrator (SSD → RAM → VRAM).
///
/// Off by default. When `enabled = true`, the server is configured to
/// layer a [`GpuExpertCache`](crate::expert_cache::GpuExpertCache) on
/// top of the existing RAM `ExpertCache`. The VRAM cache is split into
/// an **Anchor Core** (high-frequency experts permanently pinned once
/// they cross `promote_after_hits`) and an **LRU Edge** (O(1) LRU
/// queue handling temporal topic shifts). The `vram_anchor_ratio`
/// controls the split between the two regions.
///
/// Note: this field is only a configuration switch. Support for GPU
/// caching depends on how the binary was built and what runtime
/// environment it is started in; this config definition does not
/// itself perform CUDA feature detection or automatic fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCacheConfig {
    /// Master switch. `false` (default) leaves the existing 2-tier
    /// (SSD → RAM) substrate untouched.
    #[serde(default)]
    pub enabled: bool,

    /// VRAM budget, in mebibytes (1 MiB = 1024 × 1024 bytes), available
    /// to the expert cache. Defaults to 0 — i.e. the cache is created
    /// with zero capacity and every lookup misses straight through to
    /// the RAM tier. Operators sizing the cache should leave headroom
    /// for the dense transformer body and the KV cache; a typical
    /// 16 GiB consumer card might allocate 4096–8192 here.
    #[serde(default)]
    pub vram_capacity_mb: usize,

    /// Hit count an expert must accumulate (in [`ExpertResident::hits`])
    /// before it is permanently pinned into the Anchor Core. Defaults
    /// to 16. `0` disables promotion (every expert routes through the
    /// LRU Edge only).
    #[serde(default = "default_promote_after_hits")]
    pub promote_after_hits: u64,

    /// Fraction of `vram_capacity_mb` reserved for the Anchor Core
    /// (the rest is the LRU Edge). Range `0.0..=1.0`. Defaults to
    /// `0.5` — half the VRAM is pinned to the hottest experts, half
    /// floats with topical shifts.
    #[serde(default = "default_vram_anchor_ratio")]
    pub vram_anchor_ratio: f32,

    /// Advisory on-device dtype label for the resident expert bytes.
    /// Accepts the same spellings as [`WeightDtype::as_str`]: `"f32"`,
    /// `"f16"`, `"int8"`, `"q4k"`, `"q4_0"`, `"q8_0"`; defaults to
    /// `"f16"`.
    ///
    /// **Currently advisory only.** The promotion path copies the
    /// on-disk expert bytes into VRAM as-is — it does not yet convert
    /// or repack between dtypes, and VRAM accounting is driven by the
    /// raw byte length of each [`ExpertResident`] rather than by this
    /// field. The value is validated at startup (so typos fail fast)
    /// and logged for observability, and is reserved for a future
    /// promotion-time conversion/sizing path. Operators should size
    /// `vram_capacity_mb` against the on-disk footprint, not against
    /// the dtype label here.
    #[serde(default = "default_gpu_dtype")]
    pub dtype: String,
}

fn default_promote_after_hits() -> u64 { 16 }
fn default_vram_anchor_ratio() -> f32 { 0.5 }
fn default_gpu_dtype() -> String { "f16".to_string() }

impl Default for GpuCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            vram_capacity_mb: 0,
            promote_after_hits: default_promote_after_hits(),
            vram_anchor_ratio: default_vram_anchor_ratio(),
            dtype: default_gpu_dtype(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub storage: StorageConfigToml,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
    #[serde(default)]
    pub real_transformer: RealTransformerConfig,
    #[serde(default)]
    pub sampling: SamplingConfig,
    #[serde(default)]
    pub predictive: PredictiveConfig,
    /// Optional API-key gate, rate limit, and TLS configuration.
    /// Defaults are fully permissive (no auth, no rate limit, plain
    /// HTTP) to preserve the legacy behaviour bit-for-bit.
    #[serde(default)]
    pub security: SecurityConfig,
    /// Optional `[gpu_cache]` section — Phase 1/2 of the 3-tier
    /// heterogeneous memory orchestrator (SSD → RAM → VRAM). Off by
    /// default; the binary behaves identically to the 2-tier engine
    /// when this section is absent.
    #[serde(default)]
    pub gpu_cache: GpuCacheConfig,
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let body = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let cfg: Config = toml::from_str(&body).map_err(|e| ConfigError::Parse(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.num_experts == 0 {
            return Err(ConfigError::Invalid("model.num_experts must be > 0".into()));
        }
        if self.model.top_k == 0 || self.model.top_k as u32 > self.model.num_experts {
            return Err(ConfigError::Invalid(
                "model.top_k must be in 1..=num_experts".into(),
            ));
        }
        if self.model.d_model == 0 || self.model.d_ff == 0 {
            return Err(ConfigError::Invalid(
                "model.d_model and model.d_ff must be > 0".into(),
            ));
        }
        if self.model.num_layers == 0 {
            return Err(ConfigError::Invalid("model.num_layers must be > 0".into()));
        }
        if !self.storage.block_align.is_power_of_two() || self.storage.block_align == 0 {
            return Err(ConfigError::Invalid(
                "storage.block_align must be a positive power of two".into(),
            ));
        }
        if self.model.expert_size % self.storage.block_align != 0 {
            return Err(ConfigError::Invalid(format!(
                "model.expert_size ({}) must be a multiple of storage.block_align ({})",
                self.model.expert_size, self.storage.block_align
            )));
        }
        if self.model.num_layers > 1 && self.storage.cache_slots < self.model.num_layers {
            return Err(ConfigError::Invalid(format!(
                "storage.cache_slots ({}) must be >= model.num_layers ({}) for multi-layer caching",
                self.storage.cache_slots, self.model.num_layers
            )));
        }
        if self.server.max_tokens == 0 {
            return Err(ConfigError::Invalid("server.max_tokens must be > 0".into()));
        }
        if !(0.1..=1.0).contains(&self.storage.partial_load_fraction) {
            return Err(ConfigError::Invalid(format!(
                "storage.partial_load_fraction ({}) must be in [0.1, 1.0]",
                self.storage.partial_load_fraction
            )));
        }
        // [gpu_cache] validation — only meaningful when enabled.
        if self.gpu_cache.enabled {
            if !(0.0..=1.0).contains(&self.gpu_cache.vram_anchor_ratio) {
                return Err(ConfigError::Invalid(format!(
                    "gpu_cache.vram_anchor_ratio ({}) must be in [0.0, 1.0]",
                    self.gpu_cache.vram_anchor_ratio
                )));
            }
            // Parse the dtype string against the WeightDtype contract so
            // we fail fast on a typo rather than at first VRAM
            // promotion.
            if WeightDtype::from_str_opt(&self.gpu_cache.dtype).is_none() {
                return Err(ConfigError::Invalid(format!(
                    "gpu_cache.dtype: unknown weight dtype {:?} (expected one of \
                     f32, f16, int8, q4k, q4_0, q8_0)",
                    self.gpu_cache.dtype
                )));
            }
        }
        if self.real_transformer.enabled {
            let rt = &self.real_transformer;
            if rt.num_heads == 0 {
                return Err(ConfigError::Invalid(
                    "real_transformer.num_heads must be > 0 when enabled".into(),
                ));
            }
            // 0 = auto: head_dim defaults to d_model / num_heads.
            let head_dim = if rt.head_dim == 0 {
                if self.model.d_model % rt.num_heads != 0 {
                    return Err(ConfigError::Invalid(format!(
                        "real_transformer.head_dim is auto but d_model ({}) is not \
                         divisible by num_heads ({})",
                        self.model.d_model, rt.num_heads
                    )));
                }
                self.model.d_model / rt.num_heads
            } else {
                rt.head_dim
            };
            if head_dim * rt.num_heads != self.model.d_model {
                return Err(ConfigError::Invalid(format!(
                    "real_transformer: head_dim*num_heads ({}*{}={}) must equal \
                     model.d_model ({})",
                    head_dim,
                    rt.num_heads,
                    head_dim * rt.num_heads,
                    self.model.d_model
                )));
            }
            let kv_heads = if rt.num_kv_heads == 0 { rt.num_heads } else { rt.num_kv_heads };
            if kv_heads == 0 || rt.num_heads % kv_heads != 0 {
                return Err(ConfigError::Invalid(format!(
                    "real_transformer.num_kv_heads ({kv_heads}) must divide num_heads ({})",
                    rt.num_heads
                )));
            }
            if rt.vocab_size == 0 {
                return Err(ConfigError::Invalid(
                    "real_transformer.vocab_size must be > 0".into(),
                ));
            }
            if rt.max_batch_size == 0 {
                return Err(ConfigError::Invalid(
                    "real_transformer.max_batch_size must be > 0".into(),
                ));
            }
            // Validate the configurable pool back-pressure ladder
            // (gist Part 1, fix #4). Defaults to the legacy
            // 90%/98% constants when the operator omits them.
            crate::block_pool::PressureThresholds::try_new(
                rt.pressure_high_threshold,
                rt.pressure_critical_threshold,
            )
            .map_err(|e| ConfigError::Invalid(format!("real_transformer.{e}")))?;
            if rt.max_concurrent_prefetches == 0 {
                return Err(ConfigError::Invalid(
                    "real_transformer.max_concurrent_prefetches must be > 0".into(),
                ));
            }
            if rt.max_fetch_yields == 0 {
                return Err(ConfigError::Invalid(
                    "real_transformer.max_fetch_yields must be > 0".into(),
                ));
            }
        }
        // [predictive] section.
        let p = &self.predictive;
        if p.locality_enabled {
            if p.locality_window == 0 {
                return Err(ConfigError::Invalid(
                    "predictive.locality_window must be > 0 when locality_enabled".into(),
                ));
            }
            if !(p.locality_threshold_pct > 0.0 && p.locality_threshold_pct <= 1.0) {
                return Err(ConfigError::Invalid(
                    "predictive.locality_threshold_pct must be in (0.0, 1.0]".into(),
                ));
            }
        }
        if p.speculator_enabled {
            if p.speculator_hidden_dim == 0 {
                return Err(ConfigError::Invalid(
                    "predictive.speculator_hidden_dim must be > 0 when speculator_enabled".into(),
                ));
            }
            if p.speculator_top_k > self.model.num_experts as usize {
                return Err(ConfigError::Invalid(format!(
                    "predictive.speculator_top_k ({}) must not exceed model.num_experts ({})",
                    p.speculator_top_k, self.model.num_experts,
                )));
            }
        }
        // [security] section.
        let s = &self.security;
        match (&s.tls_cert, &s.tls_key) {
            (Some(_), None) | (None, Some(_)) => {
                return Err(ConfigError::Invalid(
                    "security.tls_cert and security.tls_key must both be set or both omitted"
                        .into(),
                ));
            }
            (Some(c), Some(k)) => {
                // We don't link rustls in by default; surface the
                // intent clearly so operators understand why HTTPS
                // doesn't actually come up. See `docs/production.md`
                // for the recommended reverse-proxy delegation
                // pattern.
                tracing::warn!(
                    cert = %c.display(),
                    key = %k.display(),
                    "security.tls_cert/key are configured but native TLS is not compiled in; \
                     terminate TLS at a reverse proxy (nginx/Envoy/ALB). See docs/production.md.",
                );
            }
            (None, None) => {}
        }
        if s.rate_limit_burst > 0 && s.rate_limit_rps == 0 {
            return Err(ConfigError::Invalid(
                "security.rate_limit_burst requires rate_limit_rps > 0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io { path: PathBuf, source: std::io::Error },
    Parse(String),
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io { path, source } => write!(f, "config io ({}): {source}", path.display()),
            ConfigError::Parse(m) => write!(f, "config parse: {m}"),
            ConfigError::Invalid(m) => write!(f, "config invalid: {m}"),
        }
    }
}

impl std::error::Error for ConfigError {}

// ---------------------------------------------------------------------------
// Live-reloadable runtime configuration.
//
// `RuntimeConfig` carries the subset of `Config` whose values may legitimately
// change at runtime without rebuilding the engine, scheduler, storage, or
// model. It is the surface read by Tokio worker threads on the **hot
// token-evaluation path** — the sampling defaults applied to every request
// and the per-request `max_tokens` cap.
//
// Hot-path access goes through `arc_swap::ArcSwap<RuntimeConfig>`:
//
//   * `LiveConfig::snapshot()` is a single relaxed atomic load returning an
//     `Arc<RuntimeConfig>`. There is **no mutex** anywhere on the inference
//     path; multiple worker threads reading concurrently never contend.
//   * SIGHUP-triggered reloads `parse → validate → store`. If any step
//     fails, the in-memory runtime stays bit-identical (`tracing::warn!` is
//     emitted instead). Successful reloads are an atomic pointer swap —
//     in-flight readers keep observing the previous `Arc<RuntimeConfig>`
//     until they drop their snapshot, so no visibility tear can occur.
//
// `RuntimeConfig` is therefore intentionally narrow: only fields that can
// be applied without rebuilding stateful subsystems (model weights, block
// pool capacity, ring depth, …) live here. Restart-required knobs stay on
// `Config` and trigger a `WARN`-level diff log on SIGHUP.
// ---------------------------------------------------------------------------

/// Subset of [`Config`] safe to swap atomically while the engine is serving
/// traffic. See module-level comment above.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Server-wide default sampling parameters. Per-request overrides
    /// from the JSON body still win; this is the baseline.
    pub sampling: crate::sampling::SamplingParams,
    /// Per-request cap on generated tokens. The HTTP layer clamps
    /// each request's `max_tokens` to this value before driving the
    /// engine.
    pub max_tokens_cap: usize,
    /// Telemetry flag: emit per-request structured logs (`info`-level
    /// access log line including model name, prompt + completion
    /// token counts, latency, and request id). When `false` only
    /// `warn!` / `error!` lines are emitted from the request path.
    ///
    /// Reserved for future use by the request handlers; carried in
    /// `RuntimeConfig` today so SIGHUP can flip it live without a
    /// `Config` rewrite once the access-log path is wired.
    #[allow(dead_code)]
    pub access_log_enabled: bool,
}

impl RuntimeConfig {
    /// Build a `RuntimeConfig` from the full TOML [`Config`].
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            sampling: cfg.sampling.to_params(),
            max_tokens_cap: cfg.server.max_tokens,
            // Default `true` matches the pre-refactor behaviour where
            // the request handlers unconditionally emitted an info log.
            access_log_enabled: true,
        }
    }
}

/// Thread-safe handle to the live, atomically swappable [`RuntimeConfig`].
///
/// Cloning a [`LiveConfig`] is `O(1)` — both clones share the same
/// `ArcSwap` and therefore observe the same atomic swaps. The hot path
/// only ever calls [`Self::snapshot`], which is a single relaxed atomic
/// load.
#[derive(Clone)]
pub struct LiveConfig {
    inner: std::sync::Arc<arc_swap::ArcSwap<RuntimeConfig>>,
}

impl LiveConfig {
    /// Build a new `LiveConfig` seeded from the given full [`Config`].
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            inner: std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(
                RuntimeConfig::from_config(cfg),
            )),
        }
    }

    /// Zero-overhead hot-path read. Returns a cheap RCU-style guard
    /// that dereferences to the current [`RuntimeConfig`]. Holding the
    /// guard for the duration of one token-evaluation step is fine —
    /// concurrent SIGHUP reloads do not block on it.
    #[inline]
    pub fn snapshot(&self) -> arc_swap::Guard<std::sync::Arc<RuntimeConfig>> {
        self.inner.load()
    }

    /// Snapshot helper that clones the inner `Arc` so callers can hold
    /// the value across an `await` point without keeping the underlying
    /// RCU guard live. Still O(1) — just an atomic refcount bump.
    #[inline]
    #[allow(dead_code)]
    pub fn load_full(&self) -> std::sync::Arc<RuntimeConfig> {
        self.inner.load_full()
    }

    /// Validate `new` and, on success, atomically replace the live
    /// runtime configuration. On any validation failure (schema, range,
    /// invariant) the in-memory runtime is left **pristine** and a
    /// structured `tracing::warn!` is emitted so operators can correlate
    /// the rejected SIGHUP with their config-management pipeline.
    ///
    /// Returns the new runtime config on success so the caller may
    /// inspect / diff against the previous snapshot.
    pub fn try_reload(&self, new: &Config) -> Result<std::sync::Arc<RuntimeConfig>, ConfigError> {
        // Re-validate; never trust a freshly-parsed Config to satisfy
        // the cross-section invariants without an explicit check.
        if let Err(e) = new.validate() {
            tracing::warn!(
                error = %e,
                "live config reload rejected: validation failed; existing runtime \
                 configuration left un-mutated",
            );
            return Err(e);
        }
        let next = std::sync::Arc::new(RuntimeConfig::from_config(new));
        self.inner.store(next.clone());
        Ok(next)
    }
}

impl std::fmt::Debug for LiveConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveConfig")
            .field("snapshot", &*self.snapshot())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_cfg() -> Config {
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
                predict_min_prob: 0.05,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
            },
            tokenizer: TokenizerConfig::default(),
            real_transformer: RealTransformerConfig::default(),
            sampling: SamplingConfig::default(),
            predictive: PredictiveConfig::default(),
            security: SecurityConfig::default(),
            gpu_cache: GpuCacheConfig::default(),
        }
    }

    #[test]
    fn valid_config_passes_validation() {
        minimal_cfg().validate().expect("valid");
    }

    #[test]
    fn rejects_top_k_greater_than_num_experts() {
        let mut c = minimal_cfg();
        c.model.top_k = 99;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_misaligned_expert_size() {
        let mut c = minimal_cfg();
        c.model.expert_size = 5000; // not a multiple of 4096
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_non_power_of_two_block_align() {
        let mut c = minimal_cfg();
        c.storage.block_align = 4097;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_cache_slots_below_layer_count_for_multi_layer() {
        let mut c = minimal_cfg();
        c.model.num_layers = 3;
        c.storage.cache_slots = 2;
        assert!(c.validate().is_err());
    }

    #[test]
    fn allows_zero_overflow_cap_as_unbounded() {
        let mut c = minimal_cfg();
        c.real_transformer = RealTransformerConfig {
            enabled: true,
            vocab_size: 256,
            num_heads: 8,
            max_batch_size: 8,
            pressure_high_threshold: crate::block_pool::SOFT_CAP_RATIO,
            pressure_critical_threshold: crate::block_pool::CRITICAL_PRESSURE_RATIO,
            max_concurrent_prefetches: crate::engine::DEFAULT_MAX_CONCURRENT_PREFETCHES,
            max_fetch_yields: crate::engine::DEFAULT_MAX_FETCH_YIELDS,
            max_overflow_capacity: Some(0),
            ..RealTransformerConfig::default()
        };
        c.validate().expect("0 overflow cap should map to unbounded");
    }

    #[test]
    fn round_trips_through_toml() {
        let c = minimal_cfg();
        let s = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        back.validate().unwrap();
        assert_eq!(back.model.num_experts, c.model.num_experts);
        assert_eq!(back.server.bind, c.server.bind);
    }

    /// `config.toml` documents weight dtypes using lowercase strings
    /// (`"f32"`, `"f16"`, `"int8"`, `"q4k"`, `"q4_0"`, `"q8_0"`), and
    /// the serde layer also accepts additional aliases (for example
    /// `"fp32"`, `"fp16"`, `"half"`, `"i8"`, `"q8"`, `"q4_k_m"`,
    /// `"q4km"`, `"q40"`, `"q4"`, `"q80"`). Deserializing a
    /// `ModelConfig` must accept each supported spelling, plus the
    /// legacy variant names, otherwise users following the in-tree
    /// documentation will hit confusing parse errors.
    #[test]
    fn model_dtype_accepts_documented_spellings() {
        let cases: &[(&str, WeightDtype)] = &[
            ("f32", WeightDtype::F32),
            ("fp32", WeightDtype::F32),
            ("F32", WeightDtype::F32),
            ("f16", WeightDtype::F16),
            ("fp16", WeightDtype::F16),
            ("half", WeightDtype::F16),
            ("F16", WeightDtype::F16),
            ("int8", WeightDtype::Int8),
            ("i8", WeightDtype::Int8),
            ("Int8", WeightDtype::Int8),
            ("q8", WeightDtype::Int8),
            ("Q8", WeightDtype::Int8),
            ("q4k", WeightDtype::Q4K),
            ("Q4K", WeightDtype::Q4K),
            ("q4_k_m", WeightDtype::Q4K),
            ("q4km", WeightDtype::Q4K),
            ("Q4_K_M", WeightDtype::Q4K),
            ("q4_0", WeightDtype::Q4_0),
            ("q40", WeightDtype::Q4_0),
            ("q4", WeightDtype::Q4_0),
            ("Q4_0", WeightDtype::Q4_0),
            ("q8_0", WeightDtype::Q8_0),
            ("q80", WeightDtype::Q8_0),
            ("Q8_0", WeightDtype::Q8_0),
        ];
        for (spelling, expected) in cases {
            let toml_src = format!(
                "data_dir = \"./data\"\n\
                 num_experts = 8\n\
                 d_model = 64\n\
                 d_ff = 256\n\
                 expert_size = 4096\n\
                 dtype = \"{spelling}\"\n"
            );
            let m: ModelConfig = toml::from_str(&toml_src)
                .unwrap_or_else(|e| panic!("dtype={spelling:?} should parse: {e}"));
            assert_eq!(m.dtype, *expected, "dtype={spelling:?}");
        }
    }

    #[test]
    fn predictive_section_defaults_to_disabled() {
        let c = minimal_cfg();
        assert!(!c.predictive.locality_enabled);
        assert!(!c.predictive.speculator_enabled);
        c.validate().expect("disabled predictive section is valid");
    }

    #[test]
    fn predictive_section_validates_when_enabled() {
        let mut c = minimal_cfg();
        c.predictive.locality_enabled = true;
        c.predictive.locality_window = 64;
        c.predictive.locality_threshold_pct = 0.1;
        c.predictive.speculator_enabled = true;
        c.predictive.speculator_hidden_dim = 32;
        c.predictive.speculator_top_k = 2;
        c.validate().expect("valid predictive section");
    }

    #[test]
    fn predictive_rejects_zero_window() {
        let mut c = minimal_cfg();
        c.predictive.locality_enabled = true;
        c.predictive.locality_window = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn predictive_rejects_out_of_range_threshold() {
        let mut c = minimal_cfg();
        c.predictive.locality_enabled = true;
        c.predictive.locality_threshold_pct = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn predictive_rejects_too_large_speculator_topk() {
        let mut c = minimal_cfg();
        c.predictive.speculator_enabled = true;
        c.predictive.speculator_top_k = 9999;
        assert!(c.validate().is_err());
    }
}
