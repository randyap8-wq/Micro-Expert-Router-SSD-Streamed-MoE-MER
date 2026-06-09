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
use crate::architecture::{
    Architecture, ComputeSupport, FfnKind, MlaDims, RopeScaling, TensorNaming,
};
use crate::gating::{LinearGate, ScoringFunc};
use crate::mla::MultiHeadLatentAttention;
use crate::transformer::{
    KvCache, LMHead, MultiHeadSelfAttention, RmsNorm, SharedExpert, TransformerLayer,
};
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

/// Advanced routing / attention parameters that only some architectures
/// use. Defaults reproduce Mixtral / Qwen3 behaviour exactly (softmax
/// routing, top-K renormalisation, no grouping, unit scaling, no MLA, no
/// RoPE scaling), so configs that don't set them are unaffected.
#[derive(Debug, Clone)]
pub struct AdvancedConfig {
    /// Router score function (DeepSeek-V3 uses `Sigmoid`).
    pub scoring_func: ScoringFunc,
    /// Re-normalise the selected top-K mixing weights (`norm_topk_prob`).
    pub norm_topk_prob: bool,
    /// Group-limited routing group count (`n_group`); `1` disables it.
    pub n_group: usize,
    /// Surviving groups in group-limited routing (`topk_group`).
    pub topk_group: usize,
    /// Final routed-weight multiplier (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    /// Always-on shared experts per MoE layer (`n_shared_experts`).
    pub num_shared_experts: usize,
    /// Top-K selection method string (`topk_method`), e.g. `"noaux_tc"`.
    pub topk_method: Option<String>,
    /// MLA projection dims (DeepSeek-V3); `None` for every other family.
    pub mla: Option<MlaDims>,
    /// RoPE scaling (YaRN) parameters; `None` when absent.
    pub rope_scaling: Option<RopeScaling>,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            scoring_func: ScoringFunc::Softmax,
            // Mixtral renormalises differently and has no `norm_topk_prob`
            // key; defaulting to `false` keeps TOML-only (no config.json)
            // Mixtral runs from silently applying Qwen3-style top-K
            // renormalisation. `true` is only set when a checkpoint's
            // config.json explicitly requests it (see `from_hf_config`).
            norm_topk_prob: false,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
            num_shared_experts: 0,
            topk_method: None,
            mla: None,
            rope_scaling: None,
        }
    }
}

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
    /// Model family this config describes. Drives tensor-name mapping in
    /// the `.safetensors` loader (gate name, `language_model.` prefix,
    /// fused Phi-4 projections) and whether the forward-compute path is
    /// supported. Defaults to [`Architecture::Mixtral`], preserving the
    /// historical behaviour for every existing call site.
    pub architecture: Architecture,
    /// Number of leading dense layers for DeepSeek-style MoE
    /// (`first_k_dense_replace`). `0` (default) means every MoE layer is
    /// sparse, matching Mixtral / Qwen3-MoE.
    pub first_k_dense_replace: usize,
    /// Advanced routing / MLA / RoPE-scaling parameters. Defaults to the
    /// Mixtral/Qwen3 behaviour; populated from `config.json` for DeepSeek.
    pub advanced: AdvancedConfig,
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
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: AdvancedConfig::default(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0 || self.num_heads == 0 {
            return Err("head_dim and num_heads must both be > 0".into());
        }
        // Mixtral ties the attention projection width to the residual
        // stream (`num_heads * head_dim == d_model`). Newer families
        // (e.g. Qwen3) specify an explicit `head_dim` that does NOT equal
        // `d_model / num_heads`; `MultiHeadSelfAttention` already supports
        // a query width independent of `d_model` (`wo` maps the
        // `num_heads * head_dim` query space back to `d_model`), so we only
        // enforce the tie for Mixtral.
        if self.architecture == Architecture::Mixtral
            && self.head_dim * self.num_heads != self.d_model
        {
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
        if self.first_k_dense_replace > self.num_layers {
            return Err(format!(
                "first_k_dense_replace ({}) must not exceed num_layers ({})",
                self.first_k_dense_replace, self.num_layers
            ));
        }
        let adv = &self.advanced;
        if adv.n_group == 0 || adv.topk_group == 0 {
            return Err("n_group and topk_group must both be > 0".into());
        }
        if adv.topk_group > adv.n_group {
            return Err(format!(
                "topk_group ({}) must not exceed n_group ({})",
                adv.topk_group, adv.n_group
            ));
        }
        if adv.n_group > 1 && self.num_experts % adv.n_group != 0 {
            return Err(format!(
                "num_experts ({}) must be divisible by n_group ({})",
                self.num_experts, adv.n_group
            ));
        }
        if !adv.routed_scaling_factor.is_finite() || adv.routed_scaling_factor <= 0.0 {
            return Err(format!(
                "routed_scaling_factor ({}) must be a positive, finite number",
                adv.routed_scaling_factor
            ));
        }
        Ok(())
    }

    /// Per-architecture tensor-name mapping for this config, carrying the
    /// `first_k_dense_replace` dense/MoE boundary.
    pub fn tensor_naming(&self) -> TensorNaming {
        TensorNaming::new(self.architecture, self.first_k_dense_replace)
    }

    /// Build a [`RealModelConfig`] from a parsed Hugging Face `config.json`
    /// ([`HfConfig`]). Maps the architecture and hyperparameters so a real
    /// checkpoint can be loaded without hand-editing the TOML `[model]` /
    /// `[real_transformer]` sections. Dense families (no routed experts)
    /// collapse to a single expert with `top_k = 1`.
    pub fn from_hf_config(hf: &crate::architecture::HfConfig) -> Self {
        let num_experts = hf.num_routed_experts.unwrap_or(1).max(1);
        let top_k = hf.num_experts_per_tok.unwrap_or(1).clamp(1, num_experts);
        let scoring_func = match hf.scoring_func.as_deref() {
            Some("sigmoid") => ScoringFunc::Sigmoid,
            _ => ScoringFunc::Softmax,
        };
        // Assemble MLA dims only when the config carries the full set.
        let mla = match (
            hf.q_lora_rank,
            hf.kv_lora_rank,
            hf.qk_rope_head_dim,
            hf.qk_nope_head_dim,
            hf.v_head_dim,
        ) {
            (Some(q), Some(kv), Some(rope), Some(nope), Some(v)) => Some(MlaDims {
                q_lora_rank: q,
                kv_lora_rank: kv,
                qk_rope_head_dim: rope,
                qk_nope_head_dim: nope,
                v_head_dim: v,
            }),
            _ => None,
        };
        let advanced = AdvancedConfig {
            scoring_func,
            norm_topk_prob: hf.norm_topk_prob.unwrap_or(false),
            n_group: hf.n_group.unwrap_or(1).max(1),
            topk_group: hf.topk_group.unwrap_or(1).max(1),
            routed_scaling_factor: hf.routed_scaling_factor.unwrap_or(1.0),
            num_shared_experts: hf.num_shared_experts.unwrap_or(0),
            topk_method: hf.topk_method.clone(),
            mla,
            rope_scaling: hf.rope_scaling.clone(),
        };
        Self {
            d_model: hf.hidden_size,
            d_ff: hf.resolved_d_ff(),
            num_heads: hf.num_attention_heads,
            num_kv_heads: if hf.num_key_value_heads == 0 {
                hf.num_attention_heads
            } else {
                hf.num_key_value_heads
            },
            head_dim: hf.resolved_head_dim(),
            vocab_size: hf.vocab_size,
            num_layers: hf.num_hidden_layers,
            num_experts,
            top_k,
            rope_base: hf.rope_theta,
            rms_eps: hf.rms_norm_eps,
            window_size: hf.sliding_window,
            architecture: hf.architecture,
            first_k_dense_replace: hf.first_k_dense_replace.unwrap_or(0),
            advanced,
        }
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

        // QK-Norm (Qwen3 / Qwen3-MoE): seed unit-weight per-head RMSNorms so
        // the QK-Norm path is active and ready to receive the loaded
        // `self_attn.{q,k}_norm.weight` tensors. Architectures without
        // QK-Norm (Mixtral, Mistral, Phi-4, DeepSeek) leave these `None`.
        let seed_qk_norm = || -> (Option<RmsNorm>, Option<RmsNorm>) {
            if config.architecture.uses_qk_norm() {
                (
                    Some(RmsNorm::new(vec![1.0; config.head_dim], config.rms_eps)),
                    Some(RmsNorm::new(vec![1.0; config.head_dim], config.rms_eps)),
                )
            } else {
                (None, None)
            }
        };

        let layers: Vec<TransformerLayer> = (0..config.num_layers)
            .map(|_| {
                let (q_norm, k_norm) = seed_qk_norm();
                let mla = seed_mla(&config, &mut rng, proj_scale);
                TransformerLayer {
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
                    q_norm,
                    k_norm,
                },
                mla,
                rms_moe: RmsNorm::new(vec![1.0; config.d_model], config.rms_eps),
                gate: LinearGate::new(
                    sample_uniform_vec(&mut rng, config.num_experts * config.d_model, proj_scale),
                    config.num_experts,
                    config.d_model,
                    config.top_k,
                ),
                // Shared experts are an optional, architecture-specific
                // tensor (Qwen2-MoE / DeepSeek-MoE). The seeded fallback
                // leaves them absent so non-shared-expert models and the
                // synthetic smoke path behave identically to before; real
                // shared experts are populated by the on-disk loaders.
                shared_expert: None,
                // Dense FFN (Mistral Small 3 / Phi-4 / DeepSeek dense
                // prefix). Populated by the on-disk loaders for dense
                // layers; `None` means this layer routes to streamed
                // experts (Mixtral / Qwen3-MoE / DeepSeek sparse layers).
                dense_ffn: None,
            }
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
            // Optional Qwen2-MoE / DeepSeek-MoE shared expert. The shared
            // expert's intermediate size is independent of the routed
            // `d_ff`, so we infer it from the on-disk tensor length
            // (`gate floats / d_model`) rather than the model config.
            // Files are emitted by the GGUF extractor under the
            // `layer_{l}_shexp_*` names; absence (Mixtral) is a no-op.
            tried += 1;
            if let Some(se) = Self::load_shared_expert_bin(dir, l, d_model) {
                model.layers[l].shared_expert = Some(se);
                loaded += 1;
            }
        }
        info!(
            dir = %dir.display(),
            loaded,
            tried,
            "real transformer weights loaded (missing tensors fell back to seeded init)"
        );
        Ok(model)
    }

    /// Read a whole little-endian `f32` `.bin` file into a `Vec<f32>`.
    /// Returns `None` if the file is absent, unreadable, or not a whole
    /// number of `f32`s. Unlike the size-capped `try_load` helper this
    /// returns the *entire* tensor, which the shared-expert loader needs
    /// in order to infer the shared intermediate size from the length.
    fn read_full_f32(path: &Path) -> Option<Vec<f32>> {
        use std::io::Read;
        if !path.is_file() {
            return None;
        }
        let file = std::fs::File::open(path).ok()?;
        let len = file.metadata().ok()?.len();
        if len == 0 || len % 4 != 0 {
            return None;
        }
        // Stream-decode 4 bytes at a time straight into the `Vec<f32>`
        // rather than slurping the whole file into a `Vec<u8>` first; the
        // `BufReader` batches the underlying reads, so peak memory is just
        // the output buffer (no second full-size byte buffer).
        let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
        let want = (len / 4) as usize;
        let mut out = Vec::with_capacity(want);
        let mut chunk = [0u8; 4];
        loop {
            match reader.read_exact(&mut chunk) {
                Ok(()) => out.push(f32::from_le_bytes(chunk)),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(_) => return None,
            }
        }
        if out.len() != want {
            return None;
        }
        Some(out)
    }

    /// Load a layer's optional shared expert from the GGUF-extractor
    /// `.bin` files (`layer_{l}_shexp_{gate,up,down,gate_inp}.bin`). The
    /// shared intermediate size is inferred from the gate tensor length;
    /// inconsistent or partial tensor sets degrade to `None` so a missing
    /// or malformed shared expert never aborts the load.
    fn load_shared_expert_bin(dir: &Path, l: usize, d_model: usize) -> Option<SharedExpert> {
        let gate = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_gate.bin")))?;
        let up = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_up.bin")))?;
        let down = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_down.bin")))?;
        if d_model == 0 || gate.len() % d_model != 0 {
            return None;
        }
        let shared_d_ff = gate.len() / d_model;
        if shared_d_ff == 0 {
            return None;
        }
        // The sigmoid gate (Qwen2-MoE) is optional; DeepSeek-MoE omits it.
        let gate_inp = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_gate_inp.bin")))
            .filter(|g| g.len() == d_model);
        let se = SharedExpert::from_projections(d_model, shared_d_ff, &gate, &up, &down, gate_inp);
        if se.is_none() {
            warn!(
                layer = l,
                shared_d_ff,
                d_model,
                "shared expert tensors present but shapes are inconsistent; ignoring"
            );
        }
        se
    }

    /// Like [`Self::from_dir`] but loads dense weights from
    /// HuggingFace-style `.safetensors` shards instead of per-tensor
    /// `.bin` files. The layout mirrors what
    /// `transformers.AutoModelForCausalLM.save_pretrained` writes:
    ///
    /// * `<dir>/model.safetensors` (single-shard) **or**
    /// * `<dir>/model-00001-of-00002.safetensors` etc. (multi-shard,
    ///   concatenated keys). Both are picked up automatically.
    ///
    /// Tensor names follow the standard Mixtral / Llama convention:
    ///
    /// ```text
    ///   model.embed_tokens.weight                                              -> embed
    ///   model.layers.{L}.input_layernorm.weight                                -> rms_attn[L]
    ///   model.layers.{L}.post_attention_layernorm.weight                       -> rms_moe[L]
    ///   model.layers.{L}.self_attn.{q,k,v,o}_proj.weight                       -> attn weights
    ///   model.layers.{L}.block_sparse_moe.gate.weight                          -> gate[L]
    ///   model.norm.weight                                                      -> final_rms
    ///   lm_head.weight                                                         -> lm_head
    /// ```
    ///
    /// Tensors are loaded as `f32` regardless of on-disk dtype: `bf16`
    /// and `f16` are dequantised at load time. Per-expert FFN weights
    /// (`block_sparse_moe.experts.*`) are **not** loaded here — they
    /// come through the SSD-streaming engine via `expert_<id>.bin`.
    /// Anything missing falls back to seeded init, exactly like
    /// [`Self::from_dir`].
    pub fn from_safetensors(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
    ) -> Result<Self, std::io::Error> {
        config
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        // Defensive guard: refuse — *before* touching the (potentially
        // multi-hundred-GB) shards — to build any architecture whose
        // forward-compute path is not implemented. Every recognised
        // architecture is currently executable (DeepSeek-V3 included, via
        // MLA latent-KV attention + FP8 dequant), so this never triggers
        // today; it is retained so a future, mapping-only architecture
        // fails loud here rather than routing on garbage activations.
        if let ComputeSupport::Unsupported { reason } = config.architecture.compute_support() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "cannot build a runnable `{}` model from {}: {reason}",
                    config.architecture.model_type(),
                    dir.display()
                ),
            ));
        }

        let mut model = Self::new_seeded(config.clone(), seed);
        let naming = config.tensor_naming();

        // Discover .safetensors shards in `dir`. We hold each shard's
        // bytes for the duration of the load (SafeTensors borrows from
        // them) and the sharded checkpoints in HF rarely exceed a few
        // GiB on the dense (non-expert) tensors we care about.
        let mut shards: Vec<(std::path::PathBuf, Vec<u8>)> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let bytes = std::fs::read(&p)?;
                shards.push((p, bytes));
            }
        }
        if shards.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no .safetensors files found in {}", dir.display()),
            ));
        }
        // Stable order: sort by path so multi-shard checkpoints load
        // deterministically across runs.
        shards.sort_by(|a, b| a.0.cmp(&b.0));
        let parsed: Vec<safetensors::SafeTensors> = shards
            .iter()
            .map(|(p, bytes)| {
                safetensors::SafeTensors::deserialize(bytes).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("failed to parse {}: {}", p.display(), e),
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        // Closure: search every shard for `name` and decode as f32.
        // Returns `None` when the tensor isn't found in any shard or
        // when the element count doesn't match `expected`.
        let find_f32 = |name: &str, expected: usize| -> Option<Vec<f32>> {
            for st in &parsed {
                if let Ok(view) = st.tensor(name) {
                    let n_elem: usize = view.shape().iter().product();
                    if n_elem != expected {
                        warn!(
                            tensor = name,
                            have = n_elem,
                            need = expected,
                            "safetensors shape mismatch; falling back to seeded init"
                        );
                        return None;
                    }
                    return Some(decode_safetensor_to_f32(&view));
                }
            }
            None
        };

        // Closure: search every shard for the first matching `name` and
        // return its decoded f32 data regardless of length. Used by the
        // shared-expert loader, which infers the shared intermediate size
        // from the tensor length rather than asserting a configured one.
        // DeepSeek-V3 FP8 (`e4m3`) 2D weights are transparently
        // block-dequantised via their companion `<name>_scale_inv`.
        let find_f32_any = |names: &[String]| -> Option<Vec<f32>> {
            use safetensors::tensor::Dtype;
            for name in names {
                for st in &parsed {
                    if let Ok(view) = st.tensor(name) {
                        if view.dtype() != Dtype::F8_E4M3 {
                            return Some(decode_safetensor_to_f32(&view));
                        }
                        let shape = view.shape();
                        if shape.len() != 2 {
                            return None;
                        }
                        let (rows, cols) = (shape[0], shape[1]);
                        let scale_name = format!("{name}_scale_inv");
                        let mut scale_inv = None;
                        for s in &parsed {
                            if let Ok(sv) = s.tensor(&scale_name) {
                                scale_inv = Some(decode_safetensor_to_f32(&sv));
                                break;
                            }
                        }
                        let scale_inv = scale_inv?;
                        let out = crate::mla::dequant_fp8_e4m3_blockwise(
                            view.data(),
                            &scale_inv,
                            rows,
                            cols,
                            FP8_BLOCK,
                        );
                        if out.is_empty() {
                            return None;
                        }
                        return Some(out);
                    }
                }
            }
            None
        };

        // Closure: search every shard for `name`, decoding it as f32 and
        // transparently dequantising DeepSeek-V3 FP8 (`e4m3`) weights via
        // their companion `<name>_scale_inv` block-scale tensor. Non-FP8
        // dtypes route through the standard decoder. Returns `None` when
        // the tensor is missing or its element count != `expected`.
        let find_f32_dequant = |name: &str, expected: usize| -> Option<Vec<f32>> {
            use safetensors::tensor::Dtype;
            for st in &parsed {
                if let Ok(view) = st.tensor(name) {
                    let shape = view.shape();
                    let n_elem: usize = shape.iter().product();
                    if n_elem != expected {
                        warn!(
                            tensor = name,
                            have = n_elem,
                            need = expected,
                            "safetensors shape mismatch; falling back to seeded init"
                        );
                        return None;
                    }
                    if view.dtype() != Dtype::F8_E4M3 {
                        return Some(decode_safetensor_to_f32(&view));
                    }
                    // FP8 block-wise quantised 2D weight. DeepSeek stores a
                    // companion `<name>_scale_inv` of reciprocal block
                    // scales ([ceil(rows/128), ceil(cols/128)] f32).
                    if shape.len() != 2 {
                        warn!(
                            tensor = name,
                            "FP8 weight is not 2D; cannot block-dequantise"
                        );
                        return None;
                    }
                    let (rows, cols) = (shape[0], shape[1]);
                    let scale_name = format!("{name}_scale_inv");
                    let scale_inv = (|| {
                        for s in &parsed {
                            if let Ok(sv) = s.tensor(&scale_name) {
                                return Some(decode_safetensor_to_f32(&sv));
                            }
                        }
                        None
                    })();
                    let Some(scale_inv) = scale_inv else {
                        warn!(
                            tensor = name,
                            scale = %scale_name,
                            "FP8 weight missing companion scale_inv; falling back to seeded init"
                        );
                        return None;
                    };
                    let out = crate::mla::dequant_fp8_e4m3_blockwise(
                        view.data(),
                        &scale_inv,
                        rows,
                        cols,
                        FP8_BLOCK,
                    );
                    if out.is_empty() {
                        warn!(
                            tensor = name,
                            "FP8 block-dequant failed (shape mismatch); falling back to seeded init"
                        );
                        return None;
                    }
                    return Some(out);
                }
            }
            None
        };

        let mut tried = 0usize;
        let mut loaded = 0usize;
        macro_rules! maybe {
            ($name:expr, $expected:expr, $assign:expr) => {{
                tried += 1;
                if let Some(v) = find_f32($name, $expected) {
                    $assign(v);
                    loaded += 1;
                }
            }};
        }
        // FP8 dequant happens via `find_f32_dequant` / `find_f32_any`
        // directly (e.g. in `load_mla_layer`); no extra macro needed.

        let d_model = config.d_model;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        maybe!(&naming.embed(), config.vocab_size * d_model, |v| {
            model.embedding = v;
        });
        maybe!(&naming.final_norm(), d_model, |v| {
            model.final_rms = RmsNorm::new(v, config.rms_eps);
        });
        maybe!(&naming.lm_head(), config.vocab_size * d_model, |v| {
            model.lm_head = LMHead::new(v, config.vocab_size, d_model);
        });
        for l in 0..config.num_layers {
            maybe!(&naming.input_layernorm(l), d_model, |v| {
                model.layers[l].rms_attn = RmsNorm::new(v, config.rms_eps);
            });
            maybe!(&naming.post_attention_layernorm(l), d_model, |v| {
                model.layers[l].rms_moe = RmsNorm::new(v, config.rms_eps);
            });

            // Attention projections. DeepSeek-V3 uses MLA (latent-KV)
            // attention with its own low-rank projection stack; every
            // other family uses standard dense Q/K/V/O. The presence of an
            // `mla` block on the seeded layer (built from `config.advanced.mla`)
            // selects the path.
            if model.layers[l].mla.is_some() {
                load_mla_layer(
                    &mut model.layers[l],
                    l,
                    &naming,
                    &config,
                    &find_f32_dequant,
                    &mut tried,
                    &mut loaded,
                );
            } else if naming.attn_qkv_fused() {
                // Phi-4 ships a single fused `qkv_proj`
                // ([(num_heads + 2*num_kv_heads) * head_dim, d_model],
                // row-major) that we split into separate Q/K/V weights.
                tried += 1;
                if let Some(v) = find_f32(&naming.attn_qkv(l), (q_dim + 2 * kv_dim) * d_model) {
                    let (q_part, rest) = v.split_at(q_dim * d_model);
                    let (k_part, v_part) = rest.split_at(kv_dim * d_model);
                    model.layers[l].attn.wq = q_part.to_vec();
                    model.layers[l].attn.wk = k_part.to_vec();
                    model.layers[l].attn.wv = v_part.to_vec();
                    loaded += 1;
                }
                maybe!(&naming.attn_o(l), d_model * q_dim, |v| {
                    model.layers[l].attn.wo = v;
                });
            } else {
                maybe!(&naming.attn_q(l), q_dim * d_model, |v| {
                    model.layers[l].attn.wq = v;
                });
                maybe!(&naming.attn_k(l), kv_dim * d_model, |v| {
                    model.layers[l].attn.wk = v;
                });
                maybe!(&naming.attn_v(l), kv_dim * d_model, |v| {
                    model.layers[l].attn.wv = v;
                });
                maybe!(&naming.attn_o(l), d_model * q_dim, |v| {
                    model.layers[l].attn.wo = v;
                });
            }

            // QK-Norm (Qwen3 / Qwen3-MoE): per-head RMSNorm weights of
            // length `head_dim`, applied to Q and K before RoPE. Seeded as
            // unit-weight in `new_seeded` for these architectures; overwrite
            // with the loaded weights when present.
            if config.architecture.uses_qk_norm() {
                maybe!(&naming.attn_q_norm(l), config.head_dim, |v| {
                    model.layers[l].attn.q_norm = Some(RmsNorm::new(v, config.rms_eps));
                });
                maybe!(&naming.attn_k_norm(l), config.head_dim, |v| {
                    model.layers[l].attn.k_norm = Some(RmsNorm::new(v, config.rms_eps));
                });
            }

            // Dense FFN layers (Mistral Small 3, Phi-4, and DeepSeek's
            // `first_k_dense_replace` prefix) carry a resident SwiGLU FFN
            // instead of routing to streamed experts. Load it here so
            // `RealModel::step` can run it directly. Phi-4 fuses gate+up
            // into a single `mlp.gate_up_proj` that we split.
            if naming.ffn_kind(l) == FfnKind::Dense {
                let dense = if naming.mlp_gate_up_fused() {
                    // Phi-4: `mlp.gate_up_proj` is `[2*d_ff, d_model]`,
                    // row-major, gate rows first then up rows. `down_proj`
                    // is `[d_model, d_ff]`.
                    let gate_up = find_f32_any(&[naming.mlp_gate_up(l)]);
                    let down = find_f32_any(&[naming.mlp_down(l)]);
                    match (gate_up, down) {
                        (Some(gu), Some(down))
                            if d_model != 0
                                && gu.len() % (2 * d_model) == 0
                                && !gu.is_empty() =>
                        {
                            let ffn_d = gu.len() / (2 * d_model);
                            let (gate, up) = gu.split_at(ffn_d * d_model);
                            SharedExpert::from_projections(
                                d_model, ffn_d, gate, up, &down, None,
                            )
                        }
                        _ => None,
                    }
                } else {
                    let gate = find_f32_any(&[naming.mlp_gate(l)]);
                    let up = find_f32_any(&[naming.mlp_up(l)]);
                    let down = find_f32_any(&[naming.mlp_down(l)]);
                    match (gate, up, down) {
                        (Some(gate), Some(up), Some(down))
                            if d_model != 0
                                && gate.len() % d_model == 0
                                && !gate.is_empty() =>
                        {
                            let ffn_d = gate.len() / d_model;
                            SharedExpert::from_projections(
                                d_model, ffn_d, &gate, &up, &down, None,
                            )
                        }
                        _ => None,
                    }
                };
                tried += 1;
                if let Some(se) = dense {
                    model.layers[l].dense_ffn = Some(se);
                    loaded += 1;
                }
            }

            // Routed MoE gate. Only present on sparse layers — DeepSeek's
            // first `first_k_dense_replace` layers and the dense families
            // (Mistral Small 3, Phi-4, dense Qwen3) have no router. The
            // gate name differs per family (`block_sparse_moe.gate` for
            // Mixtral vs `mlp.gate` for Qwen3-MoE / DeepSeek).
            if config.architecture.is_moe() && naming.ffn_kind(l) == FfnKind::Moe {
                let adv = &config.advanced;
                let expected = config.num_experts * d_model;
                // DeepSeek-V3 aux-loss-free balancing: a per-expert bias
                // added to selection scores only. Absent on Mixtral/Qwen3.
                let correction_bias = find_f32_any(&[naming.moe_gate_correction_bias(l)])
                    .filter(|b| b.len() == config.num_experts);
                // Prefer an extracted `gate_<L>.bin` sitting alongside the
                // shards over re-deriving the gate inline from the
                // checkpoint, so the two loader paths (`from_dir` /
                // `from_safetensors`, dispatched by `from_dir_auto`) agree
                // when both sources are present. The on-disk bin is only
                // honoured when its length matches the configured shape;
                // otherwise we fall back to the inline `moe_gate` tensor.
                tried += 1;
let gate_vec = Self::read_full_f32(&dir.join(format!("gate_{l}.bin")))
    .and_then(|mut v| {
        if v.len() < expected {
            None
        } else {
            v.truncate(expected);
            Some(v)
        }
    })
    .or_else(|| find_f32(&naming.moe_gate(l), expected));
                if let Some(v) = gate_vec {
                    model.layers[l].gate = LinearGate::with_routing(
                        v,
                        config.num_experts,
                        d_model,
                        config.top_k,
                        adv.scoring_func,
                        adv.norm_topk_prob,
                        correction_bias.clone(),
                        adv.n_group,
                        adv.topk_group,
                        adv.routed_scaling_factor,
                    );
                    loaded += 1;
                }
            }
            // Optional shared expert (Qwen2-MoE / DeepSeek-MoE). Only
            // attempt the load when the checkpoint actually declares
            // always-on shared experts (`n_shared_experts > 0`): Qwen3-MoE
            // and Mixtral have none, so probing there is pure wasted I/O.
            // Checkpoints name the shared expert either `shared_expert`
            // (Qwen2-MoE) or `shared_experts` (DeepSeek-MoE), so we probe
            // both. The shared intermediate size is inferred from the gate
            // tensor length. Prefix-aware so the `language_model.`
            // checkpoints resolve correctly.
            if config.advanced.num_shared_experts > 0 {
                let p = naming.prefix();
                tried += 1;
                let shexp_gate = find_f32_any(&[
                    format!("{p}model.layers.{l}.mlp.shared_expert.gate_proj.weight"),
                    format!("{p}model.layers.{l}.mlp.shared_experts.gate_proj.weight"),
                ]);
                let shexp_up = find_f32_any(&[
                    format!("{p}model.layers.{l}.mlp.shared_expert.up_proj.weight"),
                    format!("{p}model.layers.{l}.mlp.shared_experts.up_proj.weight"),
                ]);
                let shexp_down = find_f32_any(&[
                    format!("{p}model.layers.{l}.mlp.shared_expert.down_proj.weight"),
                    format!("{p}model.layers.{l}.mlp.shared_experts.down_proj.weight"),
                ]);
                if let (Some(gate), Some(up), Some(down)) = (shexp_gate, shexp_up, shexp_down) {
                    if d_model != 0 && gate.len() % d_model == 0 && gate.len() / d_model != 0 {
                        let shared_d_ff = gate.len() / d_model;
                        // Sigmoid gate is Qwen2-MoE-only (`shared_expert_gate`).
                        let gate_inp = find_f32_any(&[format!(
                            "{p}model.layers.{l}.mlp.shared_expert_gate.weight"
                        )])
                        .filter(|g| g.len() == d_model);
                        match SharedExpert::from_projections(
                            d_model, shared_d_ff, &gate, &up, &down, gate_inp,
                        ) {
                            Some(se) => {
                                model.layers[l].shared_expert = Some(se);
                                loaded += 1;
                            }
                            None => warn!(
                                layer = l,
                                shared_d_ff,
                                d_model,
                                "shared expert tensors present but shapes are inconsistent; ignoring"
                            ),
                        }
                    }
                }
            }
        }
        info!(
            dir = %dir.display(),
            shards = shards.len(),
            loaded,
            tried,
            "loaded dense weights from .safetensors (missing tensors fell back to seeded init)"
        );
        Ok(model)
    }

    /// Auto-dispatching entry point used by the HTTP server: if `dir`
    /// contains any `.safetensors` files we use the safetensors path;
    /// otherwise we fall back to the legacy raw-`.bin` loader. Either
    /// way, missing tensors degrade to seeded init.
    pub fn from_dir_auto(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
    ) -> Result<Self, std::io::Error> {
        let has_safetensors = std::fs::read_dir(dir)
            .map(|it| {
                it.flatten().any(|e| {
                    e.path().extension().and_then(|s| s.to_str()) == Some("safetensors")
                })
            })
            .unwrap_or(false);
        if has_safetensors {
            Self::from_safetensors(config, dir, seed)
        } else {
            Self::from_dir(config, dir, seed)
        }
    }

    /// Initial KV caches — one per layer, all empty.
    pub fn fresh_kv_caches(&self) -> Vec<KvCache> {
        self.layers
            .iter()
            .map(|l| KvCache::new(l.kv_dim()))
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

    /// **SSD-read pre-pass peek (gist Phase 1).** Return a best-effort
    /// estimate of the global expert ids this step is likely to
    /// require. The pre-pass is cheap and side-effect-free:
    ///
    /// 1. **Layer 0 is exact:** we run the embedding +
    ///    attention-normalised gate against a *clone* of the layer-0
    ///    KV cache, so the prediction matches the actual routing
    ///    decision the upcoming [`Self::step`] will make.
    /// 2. **Deeper layers are approximated** by re-using the
    ///    embedding as a stand-in for each layer's residual stream
    ///    and running its gate. This is not exact — those decisions
    ///    really depend on every preceding MoE FFN output — but it
    ///    captures the strong layerwise correlation seen on
    ///    Mixtral-class checkpoints and provides plenty of useful
    ///    hints for the warm pass. The remaining miss-on-miss
    ///    fetches still funnel through `Engine`'s in-flight
    ///    singleflight, so deduplication holds even when the peek
    ///    is wrong.
    ///
    /// The returned vector is **not** deduplicated; the caller is
    /// expected to fold it into a [`HashSet`] before calling
    /// [`Engine::warm_with`].
    pub fn peek_experts(
        &self,
        token_id: u32,
        pos: usize,
        kv: &[KvCache],
    ) -> Vec<u32> {
        assert_eq!(
            kv.len(),
            self.config.num_layers,
            "kv cache slice must have one entry per layer"
        );
        // Each layer contributes exactly `routing.experts.len()` ids,
        // which by the gating contract equals `config.top_k`. We
        // assert this loosely below via `debug_assert`; the
        // pre-allocation is a tight upper bound that avoids any
        // `Vec` growth even when `top_k` is dynamically reduced
        // (e.g. by alias-deduplication).
        let mut out = Vec::with_capacity(self.config.num_layers * self.config.top_k);
        let embed = self.embed(token_id);
        // Layer 0: run real attention against a *clone* of the KV
        // slot so the cache is not mutated by the peek.
        {
            let layer = &self.layers[0];
            let mut kv0 = kv[0].clone();
            let backend = crate::backend::current();
            let attn_out = layer.attn_block(&embed, pos, 0, &mut kv0, &*backend);
            let (_normed, routing) = layer.moe_pre(&attn_out);
            for &local in &routing.experts {
                out.push(self.global_expert_id(0, local));
            }
        }
        // Layers ≥ 1: use the embedding as a fast residual-stream
        // approximation. Cheap (no attention, no cloning), and good
        // enough to seed the warm pass — anything the peek misses is
        // still caught by the in-flight singleflight on the critical
        // path.
        for layer_idx in 1..self.config.num_layers {
            let layer = &self.layers[layer_idx];
            let (_normed, routing) = layer.moe_pre(&embed);
            for &local in &routing.experts {
                out.push(self.global_expert_id(layer_idx, local));
            }
        }
        out
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
        let backend = crate::backend::current();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Attention sub-block.
            x = layer.attn_block(&x, pos, layer_idx, &mut kv[layer_idx], &*backend);
            // Dense FFN layers (Mistral Small 3, Phi-4, DeepSeek dense
            // prefix) bypass the SSD-streamed expert path entirely: run the
            // resident SwiGLU FFN and skip routing.
            if let Some(dense_out) = layer.dense_forward(&x) {
                x = dense_out;
                continue;
            }
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
            let expert_outs = engine
                .moe_step(token_idx, layer_idx as u32, &normed, &global_ids)
                .await;
            debug_assert_eq!(expert_outs.len(), routing.weights.len());
            x = layer.moe_combine(&x, &expert_outs, &routing.weights);
            // Qwen2-MoE / DeepSeek-MoE shared expert: a dense always-on
            // FFN over the same MoE-normalised hidden, added to the
            // residual alongside the routed experts. `None` for Mixtral
            // (no-op), keeping the engine MoE-architecture-agnostic.
            if let Some(shared) = layer.shared_expert_forward(&normed) {
                x = crate::transformer::add_residual(&x, &shared);
            }
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

/// Build a deterministically seeded [`MultiHeadLatentAttention`] for a
/// single DeepSeek-V3 layer, or `None` when the config carries no MLA
/// dims (every non-DeepSeek architecture). The seeded weights let the
/// engine run DeepSeek-V3 end-to-end through the latent-attention path
/// even without an on-disk checkpoint, exactly mirroring how the
/// standard attention path is seeded in [`RealModel::new_seeded`]; the
/// on-disk loaders overwrite these with real tensors when present.
fn seed_mla(
    config: &RealModelConfig,
    rng: &mut SplitMix64,
    proj_scale: f32,
) -> Option<MultiHeadLatentAttention> {
    let dims = config.advanced.mla?;
    let n_h = config.num_heads;
    let qk_head = dims.qk_nope_head_dim + dims.qk_rope_head_dim;
    let q_total = n_h * qk_head;
    let kv_proj_dim = dims.kv_lora_rank + dims.qk_rope_head_dim;
    let kv_b_out = n_h * (dims.qk_nope_head_dim + dims.v_head_dim);

    let (q_a_proj, q_a_layernorm, q_b_proj) = if dims.q_lora_rank > 0 {
        (
            sample_uniform_vec(rng, dims.q_lora_rank * config.d_model, proj_scale),
            Some(RmsNorm::new(vec![1.0; dims.q_lora_rank], config.rms_eps)),
            sample_uniform_vec(rng, q_total * dims.q_lora_rank, proj_scale),
        )
    } else {
        (
            Vec::new(),
            None,
            sample_uniform_vec(rng, q_total * config.d_model, proj_scale),
        )
    };

    Some(MultiHeadLatentAttention {
        d_model: config.d_model,
        num_heads: n_h,
        q_lora_rank: dims.q_lora_rank,
        kv_lora_rank: dims.kv_lora_rank,
        qk_nope_head_dim: dims.qk_nope_head_dim,
        qk_rope_head_dim: dims.qk_rope_head_dim,
        v_head_dim: dims.v_head_dim,
        rope_base: config.rope_base,
        softmax_scale: MultiHeadLatentAttention::default_softmax_scale(
            dims.qk_nope_head_dim,
            dims.qk_rope_head_dim,
        ),
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj_with_mqa: sample_uniform_vec(rng, kv_proj_dim * config.d_model, proj_scale),
        kv_a_layernorm: RmsNorm::new(vec![1.0; dims.kv_lora_rank], config.rms_eps),
        kv_b_proj: sample_uniform_vec(rng, kv_b_out * dims.kv_lora_rank, proj_scale),
        o_proj: sample_uniform_vec(rng, config.d_model * n_h * dims.v_head_dim, proj_scale),
    })
}

/// DeepSeek-V3 FP8 block-quantisation edge (`weight_scale_inv` is laid
/// out over a 128x128 block grid).
const FP8_BLOCK: usize = 128;

/// Load the on-disk MLA projection tensors for a single DeepSeek-V3
/// layer, overwriting the seeded weights in `layer.mla`. Missing tensors
/// keep their seeded values (matching the rest of the loader's
/// best-effort behaviour). `find` transparently dequantises FP8 weights.
/// Increments `tried`/`loaded` so the loader's summary stays accurate.
fn load_mla_layer<F>(
    layer: &mut TransformerLayer,
    l: usize,
    naming: &TensorNaming,
    config: &RealModelConfig,
    find: &F,
    tried: &mut usize,
    loaded: &mut usize,
) where
    F: Fn(&str, usize) -> Option<Vec<f32>>,
{
    let Some(mla) = layer.mla.as_mut() else { return };
    let d_model = config.d_model;
    let n_h = mla.num_heads;
    let qk_head = mla.qk_nope_head_dim + mla.qk_rope_head_dim;
    let q_total = n_h * qk_head;
    let kv_proj_dim = mla.kv_lora_rank + mla.qk_rope_head_dim;
    let kv_b_out = n_h * (mla.qk_nope_head_dim + mla.v_head_dim);

    // (name, expected_len) -> Option<Vec<f32>>, counting every attempt.
    let mut try_load = |name: String, expected: usize| -> Option<Vec<f32>> {
        *tried += 1;
        let v = find(&name, expected);
        if v.is_some() {
            *loaded += 1;
        }
        v
    };

    if mla.q_lora_rank > 0 {
        if let Some(v) = try_load(naming.mla_q_a_proj(l), mla.q_lora_rank * d_model) {
            mla.q_a_proj = v;
        }
        if let Some(v) = try_load(naming.mla_q_a_layernorm(l), mla.q_lora_rank) {
            mla.q_a_layernorm = Some(RmsNorm::new(v, config.rms_eps));
        }
        if let Some(v) = try_load(naming.mla_q_b_proj(l), q_total * mla.q_lora_rank) {
            mla.q_b_proj = v;
        }
    } else {
        // q_lora_rank == 0: a single dense `q_proj` straight from d_model.
        if let Some(v) = try_load(naming.attn_q(l), q_total * d_model) {
            mla.q_b_proj = v;
        }
    }

    if let Some(v) = try_load(naming.mla_kv_a_proj(l), kv_proj_dim * d_model) {
        mla.kv_a_proj_with_mqa = v;
    }
    if let Some(v) = try_load(naming.mla_kv_a_layernorm(l), mla.kv_lora_rank) {
        mla.kv_a_layernorm = RmsNorm::new(v, config.rms_eps);
    }
    if let Some(v) = try_load(naming.mla_kv_b_proj(l), kv_b_out * mla.kv_lora_rank) {
        mla.kv_b_proj = v;
    }
    if let Some(v) = try_load(naming.attn_o(l), d_model * n_h * mla.v_head_dim) {
        mla.o_proj = v;
    }
}

/// Decode a `safetensors::TensorView` into an owned `Vec<f32>`. We
/// support `f32`, `f16`, and `bf16` source dtypes — the three formats
/// HuggingFace LLM checkpoints actually ship in. `int8` / `int4`
/// quantised checkpoints (AWQ, GPTQ) intentionally aren't supported
/// here: they need full per-tensor zero-points/scales which the
/// upstream extraction pipeline (`extract_mixtral_experts.py`) is the
/// right place to apply. Unknown dtypes fall back to seeded init via
/// the empty `Vec` — the caller's `expected` length check will reject
/// it cleanly.
fn decode_safetensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    use safetensors::tensor::Dtype;
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            warn!(
                dtype = ?other,
                "unsupported safetensors dtype; falling back to seeded init"
            );
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::expert_cache::ExpertCache;
    use crate::multi_layer_cache::MultiLayerExpertCache;
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
                num_experts_per_layer: None,
            })
            .expect("storage"),
        );
        storage.warmup_fds(0..total).expect("warmup");
        let pool = BufferPool::new(total as usize + 2, expert_size, block);
        let cache = Arc::new(MultiLayerExpertCache::single_layer((total as usize).max(2)));
        // The engine's TopKRouter is unused by `moe_step` (the gate
        // produces ids directly), but the engine constructor still
        // requires one.
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::new(total, cfg.top_k, 1)));
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

    /// `from_safetensors` must load the dense projections, embedding,
    /// final RMSNorm and LM head from a HF-style `.safetensors` shard
    /// and leave anything missing as the seeded baseline. We build a
    /// minimal shard by hand and compare specific weights.
    #[test]
    fn from_safetensors_loads_dense_tensors() {
        use safetensors::tensor::{Dtype, TensorView};
        use safetensors::serialize_to_file;
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let q_dim = cfg.num_heads * cfg.head_dim;
        let dir = TempDir::new("safetensors_load");

        // Distinct sentinel values per tensor so we can identify which
        // got loaded vs. which fell back to seeded.
        let embed = vec![0.25f32; cfg.vocab_size * cfg.d_model];
        let q = vec![0.5f32; q_dim * cfg.d_model];
        let final_rms = vec![1.5f32; cfg.d_model];

        // Pack each tensor as little-endian f32 bytes and feed to
        // `safetensors::serialize_to_file` — same on-disk format the
        // HF Python pipeline emits.
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
            out
        };
        let embed_bytes = to_bytes(&embed);
        let q_bytes = to_bytes(&q);
        let rms_bytes = to_bytes(&final_rms);
        let tensors: Vec<(String, TensorView)> = vec![
            (
                "model.embed_tokens.weight".to_string(),
                TensorView::new(Dtype::F32, vec![cfg.vocab_size, cfg.d_model], &embed_bytes).unwrap(),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                TensorView::new(Dtype::F32, vec![q_dim, cfg.d_model], &q_bytes).unwrap(),
            ),
            (
                "model.norm.weight".to_string(),
                TensorView::new(Dtype::F32, vec![cfg.d_model], &rms_bytes).unwrap(),
            ),
        ];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        // Loaded tensors carry their sentinel values verbatim.
        assert!(model.embedding.iter().all(|&x| x == 0.25));
        assert!(model.layers[0].attn.wq.iter().all(|&x| x == 0.5));
        // Anything else is still the seeded init (gate, k/v/o, MoE
        // gate, lm_head, rms_attn / rms_moe, etc.). Sanity: lm_head
        // wasn't provided, so its weights stayed at whatever the seed
        // produced — they must not be all-equal (which would only
        // happen if our find-tensor logic spuriously matched).
        let lm = &model.lm_head.weights;
        let first = lm[0];
        assert!(lm.iter().any(|&x| x != first), "lm_head should remain seeded, not constant");
    }

    /// Phi-4 (`phi3`) ships a single fused `qkv_proj` tensor. The loader
    /// must split it into the engine's separate `wq` / `wk` / `wv` slabs at
    /// the `[q_dim | kv_dim | kv_dim]` row boundaries, in that order.
    #[test]
    fn from_safetensors_splits_phi4_fused_qkv() {
        use safetensors::tensor::{Dtype, TensorView};
        use safetensors::serialize_to_file;
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            num_layers: 1,
            num_experts: 1,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Phi4,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let q_dim = cfg.num_heads * cfg.head_dim; // 4
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 2
        let dir = TempDir::new("phi4_qkv");

        // Fused qkv: [(q_dim + 2*kv_dim) rows, d_model cols], row-major.
        // Use a distinct sentinel per region so the split is unambiguous.
        let mut fused = Vec::new();
        fused.extend(std::iter::repeat(0.1f32).take(q_dim * cfg.d_model));
        fused.extend(std::iter::repeat(0.2f32).take(kv_dim * cfg.d_model));
        fused.extend(std::iter::repeat(0.3f32).take(kv_dim * cfg.d_model));
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
            out
        };
        let fused_bytes = to_bytes(&fused);
        let tensors: Vec<(String, TensorView)> = vec![(
            "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![q_dim + 2 * kv_dim, cfg.d_model],
                &fused_bytes,
            )
            .unwrap(),
        )];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        let attn = &model.layers[0].attn;
        assert_eq!(attn.wq.len(), q_dim * cfg.d_model);
        assert_eq!(attn.wk.len(), kv_dim * cfg.d_model);
        assert_eq!(attn.wv.len(), kv_dim * cfg.d_model);
        assert!(attn.wq.iter().all(|&x| x == 0.1), "wq region");
        assert!(attn.wk.iter().all(|&x| x == 0.2), "wk region");
        assert!(attn.wv.iter().all(|&x| x == 0.3), "wv region");
    }

    /// Mistral Small 3 (`mistral3`) is multimodal; its language-model
    /// tensors carry a `language_model.` prefix. The loader must prepend
    /// that prefix before looking tensors up.
    #[test]
    fn from_safetensors_handles_mistral_language_model_prefix() {
        use safetensors::tensor::{Dtype, TensorView};
        use safetensors::serialize_to_file;
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 1,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::MistralSmall3,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let dir = TempDir::new("mistral3_prefix");
        let embed = vec![0.7f32; cfg.vocab_size * cfg.d_model];
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
            out
        };
        let embed_bytes = to_bytes(&embed);
        // Prefixed name (what a real Mistral3 checkpoint emits) plus a
        // vision-tower tensor the loader must ignore.
        let vision = vec![9.0f32; 4];
        let vision_bytes = to_bytes(&vision);
        let tensors: Vec<(String, TensorView)> = vec![
            (
                "language_model.model.embed_tokens.weight".to_string(),
                TensorView::new(Dtype::F32, vec![cfg.vocab_size, cfg.d_model], &embed_bytes)
                    .unwrap(),
            ),
            (
                "vision_tower.patch_conv.weight".to_string(),
                TensorView::new(Dtype::F32, vec![4], &vision_bytes).unwrap(),
            ),
        ];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        assert!(
            model.embedding.iter().all(|&x| x == 0.7),
            "prefixed embed_tokens must load via language_model. prefix"
        );
    }

    /// DeepSeek-V3 (`deepseek_v3`) is now fully executable: `new_seeded`
    /// builds runnable MLA blocks for every layer instead of the loader
    /// failing loud. This guards against regressing the MLA + FP8
    /// integration back to the old "unsupported" behavior.
    #[test]
    fn deepseek_seeds_runnable_mla() {
        let mut advanced = AdvancedConfig::default();
        advanced.mla = Some(MlaDims {
            q_lora_rank: 0,
            kv_lora_rank: 16,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 8,
        });
        let cfg = RealModelConfig {
            architecture: Architecture::DeepSeekV3,
            first_k_dense_replace: 3,
            advanced,
            num_layers: 4,
            ..RealModelConfig::tiny()
        };
        let model = RealModel::new_seeded(cfg, 1);
        assert!(model.layers.iter().all(|l| l.mla.is_some()));
        // The latent KV-cache width must match the MLA latent dim so the
        // per-layer cache is sized correctly.
        let kv_dim = model.layers[0].kv_dim();
        assert_eq!(kv_dim, 16 + 4);
    }

    /// `from_hf_config` maps a parsed Qwen3-MoE `config.json` (explicit
    /// `head_dim` that differs from `hidden_size / num_heads`) into a
    /// loadable `RealModelConfig` that passes `validate()`.
    #[test]
    fn from_hf_config_maps_qwen3_moe() {
        let json = r#"{
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "num_experts": 128,
            "num_experts_per_tok": 8
        }"#;
        let hf = crate::architecture::HfConfig::from_json_str(json).unwrap();
        let cfg = RealModelConfig::from_hf_config(&hf);
        assert_eq!(cfg.architecture, Architecture::Qwen3Moe);
        assert_eq!(cfg.d_model, 2048);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.top_k, 8);
        // d_ff resolves to the MoE expert width, not the dense one.
        assert_eq!(cfg.d_ff, 768);
        // head_dim * num_heads (4096) != d_model (2048) — must still pass
        // because the Mixtral-only tie is relaxed for other architectures.
        cfg.validate().expect("Qwen3-MoE config must validate");
    }

    /// `from_dir_auto`-adjacent dispatch test.
    #[test]
    fn from_dir_auto_dispatches_on_safetensors_presence() {
        use safetensors::tensor::{Dtype, TensorView};
        use safetensors::serialize_to_file;
        let cfg = RealModelConfig {
            vocab_size: 4,
            d_model: 4,
            d_ff: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };

        // 1) Empty dir without safetensors falls back to from_dir
        // (which itself silently keeps the seeded init since no .bin
        // files exist) — so no panic / error.
        let empty = TempDir::new("auto_empty");
        let _ = RealModel::from_dir_auto(cfg.clone(), &empty.path, 7).unwrap();

        // 2) Dir with a .safetensors shard goes through the safetensors path.
        let st_dir = TempDir::new("auto_st");
        let embed = vec![0.75f32; cfg.vocab_size * cfg.d_model];
        let mut bytes = Vec::with_capacity(embed.len() * 4);
        for &x in &embed { bytes.extend_from_slice(&x.to_le_bytes()); }
        let view = TensorView::new(
            Dtype::F32,
            vec![cfg.vocab_size, cfg.d_model],
            &bytes,
        )
        .unwrap();
        serialize_to_file(
            [("model.embed_tokens.weight".to_string(), view)],
            &None,
            &st_dir.path.join("model.safetensors"),
        )
        .unwrap();
        let model = RealModel::from_dir_auto(cfg, &st_dir.path, 7).unwrap();
        assert!(model.embedding.iter().all(|&x| x == 0.75));
    }

    /// Helper: write a slice of f32 as a little-endian `.bin` file.
    fn write_bin(path: &std::path::Path, v: &[f32]) {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for &x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// `from_dir` must pick up the GGUF-extractor shared-expert `.bin`
    /// files, infer the shared intermediate size from the gate tensor
    /// length, and populate `TransformerLayer::shared_expert`. Models
    /// without those files (Mixtral) keep `shared_expert == None`.
    #[test]
    fn from_dir_loads_shared_expert_and_infers_d_ff() {
        let cfg = RealModelConfig::tiny();
        let d_model = cfg.d_model;
        let shared_d_ff = 5; // deliberately != routed d_ff (64)
        let dir = TempDir::new("shexp_bin");

        write_bin(
            &dir.path.join("layer_0_shexp_gate.bin"),
            &vec![0.1f32; shared_d_ff * d_model],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_up.bin"),
            &vec![0.2f32; shared_d_ff * d_model],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_down.bin"),
            &vec![0.3f32; d_model * shared_d_ff],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_gate_inp.bin"),
            &vec![0.0f32; d_model],
        );

        let model = RealModel::from_dir(cfg.clone(), &dir.path, 1).unwrap();
        let se = model.layers[0]
            .shared_expert
            .as_ref()
            .expect("shared expert should be loaded");
        assert_eq!(se.d_ff, shared_d_ff, "d_ff inferred from gate tensor length");
        assert_eq!(se.d_model, d_model);
        assert!(se.gate_inp.is_some(), "sigmoid gate present");
        // forward produces a finite, d_model-length vector.
        let x: Vec<f32> = (0..d_model).map(|i| 0.01 * i as f32).collect();
        let y = se.forward(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// A layer with no shared-expert files stays `None`, so Mixtral-style
    /// models are unaffected.
    #[test]
    fn from_dir_without_shared_expert_is_none() {
        let cfg = RealModelConfig::tiny();
        let dir = TempDir::new("shexp_absent");
        let model = RealModel::from_dir(cfg, &dir.path, 1).unwrap();
        assert!(model.layers[0].shared_expert.is_none());
    }

    /// `norm_topk_prob` must default to `false` so a TOML-only (no
    /// `config.json`) Mixtral run does not silently apply Qwen3-style
    /// top-K renormalisation. Explicit `true` is only honoured when a
    /// checkpoint's `config.json` sets it (see `from_hf_config`).
    #[test]
    fn advanced_config_norm_topk_prob_defaults_false() {
        assert!(!AdvancedConfig::default().norm_topk_prob);
    }

    /// Inconsistent shared-expert tensors (down proj wrong length) must
    /// degrade to `None` rather than abort the load.
    #[test]
    fn from_dir_inconsistent_shared_expert_is_ignored() {
        let cfg = RealModelConfig::tiny();
        let d_model = cfg.d_model;
        let shared_d_ff = 5;
        let dir = TempDir::new("shexp_bad");
        write_bin(
            &dir.path.join("layer_0_shexp_gate.bin"),
            &vec![0.1f32; shared_d_ff * d_model],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_up.bin"),
            &vec![0.2f32; shared_d_ff * d_model],
        );
        // Wrong length for down -> from_projections returns None.
        write_bin(
            &dir.path.join("layer_0_shexp_down.bin"),
            &vec![0.3f32; d_model * shared_d_ff + 1],
        );
        let model = RealModel::from_dir(cfg, &dir.path, 1).unwrap();
        assert!(model.layers[0].shared_expert.is_none());
    }

    /// The Qwen2-MoE sigmoid gate scales the shared expert output by
    /// `sigmoid(W_gate · x)`. A zero gate vector yields a 0.5 scale; an
    /// absent gate yields an unscaled (1.0) output (DeepSeek-MoE).
    #[test]
    fn shared_expert_sigmoid_gate_scales_output() {
        let d_model = 4;
        let d_ff = 3;
        let gate = vec![0.05f32; d_ff * d_model];
        let up = vec![0.07f32; d_ff * d_model];
        let down = vec![0.09f32; d_model * d_ff];
        let x: Vec<f32> = vec![0.5, -0.25, 0.1, 0.2];

        let ungated =
            SharedExpert::from_projections(d_model, d_ff, &gate, &up, &down, None).unwrap();
        let y_ungated = ungated.forward(&x);

        // Zero gate -> sigmoid(0) = 0.5 -> output halved.
        let gated = SharedExpert::from_projections(
            d_model,
            d_ff,
            &gate,
            &up,
            &down,
            Some(vec![0.0f32; d_model]),
        )
        .unwrap();
        let y_gated = gated.forward(&x);
        for (a, b) in y_ungated.iter().zip(y_gated.iter()) {
            assert!((a * 0.5 - b).abs() < 1e-6, "gate=0 must halve output: {a} {b}");
        }
    }
}
