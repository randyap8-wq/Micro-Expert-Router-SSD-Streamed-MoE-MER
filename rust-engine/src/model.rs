//! Real Mixtral / Llama-style decoder-only transformer wired on top of
//! the SSD-streaming engine (gist Phase 5–6: "real transformer + MoE,
//! real generation loop, multi-layer support").
//!
//! What lives here:
//!
//! 1. [`RealModelConfig`] — the hyperparameters of the model
//!    (`d_model`, `d_ff`, `num_heads`, `vocab_size`, `num_layers`,
//!    `num_experts`, `top_k`, …).
//! 2. [`RealModel`] — owns the dense (resident) weights: token embedding,
//!    a stack of [`crate::transformer::TransformerLayer`]s, the final
//!    RMSNorm and the [`crate::transformer::LMHead`]. Expert FFN weights
//!    are **not** held here — they live on disk and are streamed by the
//!    engine on demand.
//! 3. [`RealModel::from_dir`] — loads weights from a directory of `.bin`
//!    files (one per tensor) and falls back to a deterministic seeded
//!    initialisation when files are missing, so the engine always has an
//!    end-to-end runnable path even without real model files.
//! 4. [`RealModel::step`] — the per-token forward driver. Produces the
//!    next-token id by running embedding → stacked layers (each calling
//!    `attn_block`, `moe_pre`, awaiting the engine's SSD-streamed
//!    `moe_step`, then `moe_combine`) → final RMSNorm → LM head → argmax.
//!
//! Multi-layer expert addressing: when `num_layers > 1`, expert ids on
//! disk are encoded as `global_id = layer * num_experts_per_layer +
//! local_id`, so the existing single-namespace [`crate::expert_cache::ExpertCache`]
//! / [`crate::io_provider::NvmeStorage`] (which already use `u32`
//! ids and `expert_<id>.bin` paths) work unchanged. This keeps the run
//! summary statistics (hits, misses, I/O share) populated by the same
//! counters regardless of layer count. An alternative
//! [`crate::multi_layer_cache::MultiLayerExpertCache`] is also available
//! for users who want per-layer LRU isolation; the global-id flat scheme
//! is the default because it lets the existing engine instrumentation
//! and prefetcher keep working without per-layer sharding.

use crate::engine::Engine;
use crate::gating::LinearGate;
use crate::transformer::{
    KvCache, LMHead, MultiHeadSelfAttention, RmsNorm, TransformerLayer,
};
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

/// Hyperparameters of the real transformer.
#[derive(Debug, Clone)]
pub struct RealModelConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    /// Grouped-Query-Attention KV head count. For Mixtral this equals
    /// `num_heads / 4`. Setting it equal to `num_heads` recovers
    /// vanilla multi-head attention.
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    /// Experts per layer (Mixtral-8x7B: 8).
    pub num_experts: usize,
    pub top_k: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    /// Sliding-window attention span. `None` (default) = full causal
    /// attention. Mixtral uses `Some(4096)`.
    pub window_size: Option<usize>,
}

impl RealModelConfig {
    /// Tiny default useful for tests / smoke runs (d_model=32, 1 layer).
    pub fn tiny() -> Self {
        Self {
            d_model: 32,
            d_ff: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            vocab_size: 256,
            num_layers: 1,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim * self.num_heads != self.d_model {
            return Err(format!(
                "head_dim * num_heads ({} * {} = {}) must equal d_model ({})",
                self.head_dim,
                self.num_heads,
                self.head_dim * self.num_heads,
                self.d_model
            ));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "num_heads ({}) must be a positive multiple of num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "top_k ({}) must be in 1..=num_experts ({})",
                self.top_k, self.num_experts
            ));
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        Ok(())
    }
}

/// Decoder-only transformer with MoE FFN blocks. Expert FFN weights are
/// streamed from SSD per token; everything in this struct is dense and
/// stays resident in RAM.
pub struct RealModel {
    pub config: RealModelConfig,
    pub embedding: Vec<f32>, // [vocab_size, d_model]
    pub layers: Vec<TransformerLayer>,
    pub final_rms: RmsNorm,
    pub lm_head: LMHead,
}

impl RealModel {
    /// Build a model with deterministic, well-conditioned random weights
    /// from a seed. Used as the fallback when on-disk weights aren't
    /// supplied — the engine still streams expert FFN weights from SSD,
    /// so the I/O behaviour the rest of the engine measures is unchanged.
    pub fn new_seeded(config: RealModelConfig, seed: u64) -> Self {
        config.validate().expect("invalid RealModelConfig");
        let mut rng = SplitMix64::new(seed);
        let embedding = sample_uniform_vec(&mut rng, config.vocab_size * config.d_model, 0.04);

        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        // Slightly smaller scale for the projections so the residual
        // stream doesn't blow up across many layers.
        let proj_scale = (1.0 / (config.d_model as f32).sqrt()).min(0.05);

        let layers: Vec<TransformerLayer> = (0..config.num_layers)
            .map(|_| TransformerLayer {
                rms_attn: RmsNorm::new(vec![1.0; config.d_model], config.rms_eps),
                attn: MultiHeadSelfAttention {
                    d_model: config.d_model,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    rope_base: config.rope_base,
                    wq: sample_uniform_vec(&mut rng, q_dim * config.d_model, proj_scale),
                    wk: sample_uniform_vec(&mut rng, kv_dim * config.d_model, proj_scale),
                    wv: sample_uniform_vec(&mut rng, kv_dim * config.d_model, proj_scale),
                    wo: sample_uniform_vec(&mut rng, config.d_model * q_dim, proj_scale),
                    window_size: config.window_size,
                },
                rms_moe: RmsNorm::new(vec![1.0; config.d_model], config.rms_eps),
                gate: LinearGate::new(
                    sample_uniform_vec(&mut rng, config.num_experts * config.d_model, proj_scale),
                    config.num_experts,
                    config.d_model,
                    config.top_k,
                ),
            })
            .collect();
        let final_rms = RmsNorm::new(vec![1.0; config.d_model], config.rms_eps);
        let lm_head = LMHead::new(
            sample_uniform_vec(&mut rng, config.vocab_size * config.d_model, proj_scale),
            config.vocab_size,
            config.d_model,
        );
        Self { config, embedding, layers, final_rms, lm_head }
    }

    /// Try to load weights from `dir`, populating any tensor present in
    /// `<dir>/<name>.bin` (raw little-endian `f32`) and falling back to a
    /// seeded random initialisation for anything missing. Logs a one-line
    /// summary of what was loaded vs synthesised.
    ///
    /// Expected file names (all optional — missing ones use the seed):
    /// - `embed.bin`               : `[vocab_size * d_model]` f32
    /// - `final_rms.bin`           : `[d_model]` f32 (gain)
    /// - `lm_head.bin`             : `[vocab_size * d_model]` f32
    /// - `rms_attn_<L>.bin`        : `[d_model]` f32 (gain)
    /// - `rms_moe_<L>.bin`         : `[d_model]` f32 (gain)
    /// - `attn_<L>_q.bin` / `_k` / `_v` / `_o` : projection weights
    /// - `gate_<L>.bin`            : `[num_experts * d_model]` f32
    pub fn from_dir(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
    ) -> Result<Self, std::io::Error> {
        config
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let mut model = Self::new_seeded(config.clone(), seed);
        let mut loaded = 0usize;
        let mut tried = 0usize;

        let try_load = |name: &str, expected: usize| -> Option<Vec<f32>> {
            let path = dir.join(name);
            if !path.is_file() {
                return None;
            }
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let n = bytes.len() / 4;
                    if n < expected {
                        warn!(
                            file = %path.display(),
                            have = n,
                            need = expected,
                            "weight file shorter than expected; falling back to seeded init"
                        );
                        return None;
                    }
                    let mut floats = Vec::with_capacity(expected);
                    for chunk in bytes[..expected * 4].chunks_exact(4) {
                        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    Some(floats)
                }
                Err(e) => {
                    warn!(file = %path.display(), error = %e, "weight file read failed");
                    None
                }
            }
        };

        macro_rules! maybe {
            ($name:expr, $expected:expr, $assign:expr) => {{
                tried += 1;
                if let Some(v) = try_load($name, $expected) {
                    $assign(v);
                    loaded += 1;
                }
            }};
        }

        let d_model = config.d_model;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        maybe!("embed.bin", config.vocab_size * d_model, |v| model.embedding = v);
        maybe!("final_rms.bin", d_model, |v| {
            model.final_rms = RmsNorm::new(v, config.rms_eps);
        });
        maybe!("lm_head.bin", config.vocab_size * d_model, |v| {
            model.lm_head = LMHead::new(v, config.vocab_size, d_model);
        });
        for l in 0..config.num_layers {
            maybe!(&format!("rms_attn_{l}.bin"), d_model, |v| {
                model.layers[l].rms_attn = RmsNorm::new(v, config.rms_eps);
            });
            maybe!(&format!("rms_moe_{l}.bin"), d_model, |v| {
                model.layers[l].rms_moe = RmsNorm::new(v, config.rms_eps);
            });
            maybe!(&format!("attn_{l}_q.bin"), q_dim * d_model, |v| {
                model.layers[l].attn.wq = v;
            });
            maybe!(&format!("attn_{l}_k.bin"), kv_dim * d_model, |v| {
                model.layers[l].attn.wk = v;
            });
            maybe!(&format!("attn_{l}_v.bin"), kv_dim * d_model, |v| {
                model.layers[l].attn.wv = v;
            });
            maybe!(&format!("attn_{l}_o.bin"), d_model * q_dim, |v| {
                model.layers[l].attn.wo = v;
            });
            maybe!(&format!("gate_{l}.bin"), config.num_experts * d_model, |v| {
                model.layers[l].gate = LinearGate::new(
                    v,
                    config.num_experts,
                    d_model,
                    config.top_k,
                );
            });
        }
        info!(
            dir = %dir.display(),
            loaded,
            tried,
            "real transformer weights loaded (missing tensors fell back to seeded init)"
        );
        Ok(model)
    }

    /// Initial KV caches — one per layer, all empty.
    pub fn fresh_kv_caches(&self) -> Vec<KvCache> {
        self.layers
            .iter()
            .map(|l| KvCache::new(l.attn.kv_dim()))
            .collect()
    }

    /// Look up the embedding row for a token id.
    pub fn embed(&self, token_id: u32) -> Vec<f32> {
        let id = (token_id as usize) % self.config.vocab_size;
        let d = self.config.d_model;
        self.embedding[id * d..(id + 1) * d].to_vec()
    }

    /// Translate a `(layer, local_expert_id)` pair into the global flat
    /// expert id used by the engine's cache + storage. See module
    /// docstring; this is the addressing scheme that makes the existing
    /// single-namespace cache work for multi-layer models without any
    /// API changes.
    #[inline]
    pub fn global_expert_id(&self, layer: usize, local: u32) -> u32 {
        (layer as u32) * (self.config.num_experts as u32) + local
    }

    /// Run one decoder step. Returns the sampled next-token id.
    ///
    /// This is the realisation of the gist's pseudocode:
    ///
    /// ```text
    ///   x = embedding[token_id]
    ///   for layer in layers:
    ///       x = layer.attn_block(x, pos, kv[layer])
    ///       (normed, routing) = layer.moe_pre(x)
    ///       experts_y = engine.moe_step(normed, routing.experts)  // SSD-streamed
    ///       x = layer.moe_combine(x, experts_y, routing.weights)
    ///   x = final_rms(x)
    ///   logits = lm_head(x)
    ///   return sample(logits, params, pos)
    /// ```
    ///
    /// `engine.moe_step` is what reads expert weights from SSD via the
    /// LRU cache — that's the whole point of the substrate.
    /// Sampling is delegated to [`crate::sampling::sample`], so
    /// `temperature == 0.0` reproduces the original deterministic
    /// `argmax` behaviour bit-for-bit.
    pub async fn step(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
        params: &crate::sampling::SamplingParams,
    ) -> u32 {
        assert_eq!(
            kv.len(),
            self.config.num_layers,
            "kv cache slice must have one entry per layer"
        );
        let mut x = self.embed(token_id);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Attention sub-block.
            x = layer.attn_block(&x, pos, &mut kv[layer_idx]);
            // MoE sub-block: route, await SSD-streamed expert FFNs, combine.
            let (normed, routing) = layer.moe_pre(&x);
            let global_ids: Vec<u32> = routing
                .experts
                .iter()
                .map(|&local| self.global_expert_id(layer_idx, local))
                .collect();
            // `token_idx` here is just a digest seed; positional info is
            // already baked into RoPE inside `attn_block`.
            let token_idx = (pos as u64).wrapping_mul(self.config.num_layers as u64)
                + layer_idx as u64;
            let expert_outs = engine.moe_step(token_idx, &normed, &global_ids).await;
            debug_assert_eq!(expert_outs.len(), routing.weights.len());
            x = layer.moe_combine(&x, &expert_outs, &routing.weights);
        }
        let normed = self.final_rms.forward(&x);
        self.lm_head.sample(&normed, params, pos as u64)
    }
}

/// Small `splitmix64` PRNG so we can produce deterministic, dependency-free
/// weight initialisations.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Draw an `f32` in `[-scale, scale]`.
    fn next_uniform(&mut self, scale: f32) -> f32 {
        let bits = self.next_u64();
        let u = ((bits >> 40) as u32 & ((1 << 23) - 1)) as f32 / ((1u32 << 23) as f32);
        (u * 2.0 - 1.0) * scale
    }
}

fn sample_uniform_vec(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rng.next_uniform(scale));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::expert_cache::ExpertCache;
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Self-cleaning unique temp directory for test fixtures.
    struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "mer-model-{label}-{}-{n}-{ts}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn config_validation_catches_bad_shapes() {
        let mut c = RealModelConfig::tiny();
        c.head_dim = 7; // doesn't divide d_model
        assert!(c.validate().is_err());
        let mut c = RealModelConfig::tiny();
        c.top_k = 99;
        assert!(c.validate().is_err());
    }

    #[test]
    fn seeded_model_has_correct_shapes() {
        let cfg = RealModelConfig::tiny();
        let m = RealModel::new_seeded(cfg.clone(), 1);
        assert_eq!(m.embedding.len(), cfg.vocab_size * cfg.d_model);
        assert_eq!(m.layers.len(), cfg.num_layers);
        assert_eq!(m.lm_head.weights.len(), cfg.vocab_size * cfg.d_model);
        assert_eq!(m.final_rms.weight.len(), cfg.d_model);
        assert!(m.embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn global_expert_id_partitions_namespace() {
        let cfg = RealModelConfig {
            num_layers: 3,
            num_experts: 8,
            ..RealModelConfig::tiny()
        };
        let m = RealModel::new_seeded(cfg, 7);
        assert_eq!(m.global_expert_id(0, 0), 0);
        assert_eq!(m.global_expert_id(0, 7), 7);
        assert_eq!(m.global_expert_id(1, 0), 8);
        assert_eq!(m.global_expert_id(2, 5), 21);
    }

    fn build_engine_for_model(
        dir: &Path,
        cfg: &RealModelConfig,
    ) -> Arc<Engine> {
        // Total experts across all layers, addressed flat as
        // layer * num_experts + local.
        let total = cfg.num_layers as u32 * cfg.num_experts as u32;
        let weight_bytes = crate::inference::expert_weight_bytes(cfg.d_model, cfg.d_ff);
        let block = 4096usize;
        let expert_size = weight_bytes.div_ceil(block) * block;
        generate_synthetic_experts(dir, total, expert_size, cfg.d_model, cfg.d_ff)
            .expect("gen synthetic experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.to_path_buf(),
                expert_size,
                block_align: block,
                use_direct_io: false,
            })
            .expect("storage"),
        );
        storage.warmup_fds(0..total).expect("warmup");
        let pool = BufferPool::new(total as usize + 2, expert_size, block);
        let cache = Arc::new(ExpertCache::new((total as usize).max(2)));
        // The engine's TopKRouter is unused by `moe_step` (the gate
        // produces ids directly), but the engine constructor still
        // requires one.
        let router = Arc::new(TopKRouter::new(total, cfg.top_k, 1));
        let predictor = Arc::new(PredictiveLoader::new(total, 0, 0.05, 1));
        Arc::new(Engine::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape { d_model: cfg.d_model, d_ff: cfg.d_ff, hidden_seed: 1 },
            EngineOptions::default(),
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_step_produces_valid_token_id() {
        let dir = TempDir::new("step");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 0xDEAD);
        let mut kv = model.fresh_kv_caches();
        let next = model.step(&engine, 42, 0, &mut kv, &crate::sampling::SamplingParams::greedy()).await;
        assert!((next as usize) < cfg.vocab_size);
        // KV caches grew by exactly one position.
        for c in &kv {
            assert_eq!(c.seq_len, 1);
        }
        // The engine's hit/miss counters were touched (cold start =>
        // misses).
        let r = engine.report();
        assert!(r.misses > 0, "first step should miss the cache");
        assert!(r.bytes_read > 0, "engine should have read expert bytes from disk");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_step_is_deterministic_across_two_runs() {
        let dir = TempDir::new("det");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 1);

        let mut kv1 = model.fresh_kv_caches();
        let t1 = model.step(&engine, 7, 0, &mut kv1, &crate::sampling::SamplingParams::greedy()).await;
        let t2 = model.step(&engine, t1, 1, &mut kv1, &crate::sampling::SamplingParams::greedy()).await;

        let mut kv2 = model.fresh_kv_caches();
        let u1 = model.step(&engine, 7, 0, &mut kv2, &crate::sampling::SamplingParams::greedy()).await;
        let u2 = model.step(&engine, u1, 1, &mut kv2, &crate::sampling::SamplingParams::greedy()).await;

        assert_eq!((t1, t2), (u1, u2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_multi_layer_partitions_expert_namespace() {
        let dir = TempDir::new("multi");
        let cfg = RealModelConfig {
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            ..RealModelConfig::tiny()
        };
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 9);
        let mut kv = model.fresh_kv_caches();
        let _ = model.step(&engine, 5, 0, &mut kv, &crate::sampling::SamplingParams::greedy()).await;
        // Both layers contributed to KV cache growth.
        assert_eq!(kv.len(), 2);
        for c in &kv {
            assert_eq!(c.seq_len, 1);
        }
    }
}
