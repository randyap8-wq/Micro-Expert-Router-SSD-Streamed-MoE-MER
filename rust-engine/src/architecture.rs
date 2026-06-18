//! Model architecture descriptor, HuggingFace `config.json` reader and
//! per-architecture tensor-name mapping (Stage 1 of multi-architecture
//! support).
//!
//! The streaming substrate (expert cache, NVMe/io_uring storage,
//! prefetcher, global expert ids) is already model-neutral. What was
//! hard-wired to Mixtral lived in the *dense-weight* loader
//! ([`crate::model::RealModel::from_safetensors`]): tensor names, the
//! gate name, the attention projection layout and the
//! always-MoE-every-layer assumption.
//!
//! This module centralises the architecture-specific knowledge so the
//! loader can ask "what is the embedding tensor called?", "is this a
//! fused QKV projection?", "is layer `L` dense or MoE?" without spreading
//! `match arch { … }` arms across `model.rs`.
//!
//! Scope (Stage 1): recognise the architecture from `config.json`, map
//! the exact tensor names (handling Mistral Small 3's `language_model.`
//! prefix and Phi-4's fused `qkv_proj` / `gate_up_proj`), classify each
//! layer as dense vs MoE from `first_k_dense_replace`, and surface
//! DeepSeek's companion `weight_scale_inv` tensors as a *side table*
//! rather than mistaking them for weights. MLA attention and FP8
//! dequantisation are **not** implemented here; callers that need the
//! forward-compute path must consult [`Architecture::compute_support`]
//! and fail loud for unsupported variants.
#![allow(dead_code)]

use std::path::Path;

/// The set of model families the loader knows how to map tensors for.
///
/// The variants correspond 1:1 to the HuggingFace `model_type` strings
/// in each family's `config.json` (see [`Architecture::from_model_type`]).
/// `Qwen3` (dense) and `Qwen3Moe` are distinct because the former has a
/// plain SwiGLU FFN per layer while the latter is sparse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// Mixtral-8x7B / 8x22B (`mixtral`). Sparse MoE on every layer,
    /// `block_sparse_moe.*` tensor names, softmax routing.
    Mixtral,
    /// Qwen3 dense (`qwen3`). Plain SwiGLU FFN, QK-Norm attention.
    Qwen3,
    /// Qwen3-MoE (`qwen3_moe`). Sparse MoE, `mlp.experts.*` names,
    /// softmax routing, QK-Norm attention, no shared expert.
    Qwen3Moe,
    /// DeepSeek-V3 / V3.1 (`deepseek_v3`). Fine-grained MoE with a shared
    /// expert, sigmoid routing with an aux-loss-free correction bias and
    /// node-limited group selection, `first_k_dense_replace` leading
    /// dense layers, MLA latent-KV attention and FP8 weights.
    DeepSeekV3,
    /// Mistral Small 3 (`mistral3`). Dense decoder wrapped in a
    /// multimodal `…ForConditionalGeneration` checkpoint, so its text
    /// weights carry a `language_model.` prefix.
    MistralSmall3,
    /// Phi-4 (`phi3`). Dense decoder with **fused** `qkv_proj` and
    /// `gate_up_proj` projections.
    Phi4,
    /// MiMo-V2 / MiMo-V2.5 (Xiaomi, `mimo_v2`). Sparse MoE (309B total /
    /// 15B active) with a **hybrid per-layer attention** pattern: a 5:1
    /// ratio of Sliding-Window-Attention (SWA, 128-token window) layers to
    /// Global-attention layers — every 6th layer (0-indexed 5, 11, 17, …)
    /// is global. Carries Multi-Token-Prediction (`num_nextn_predict_layers`)
    /// heads that MER ignores (their weight tensors are skipped at load
    /// time; see [`crate::model::RealModel::from_safetensors`]). Tensor
    /// naming follows the Qwen3-MoE / DeepSeek convention
    /// (`model.layers.{l}.mlp.experts.{j}.{gate,up,down}_proj`,
    /// `model.layers.{l}.mlp.gate`).
    ///
    /// NOTE: the exact `model_type` / `architectures` strings and the MTP
    /// tensor prefix could not be confirmed against the live HuggingFace
    /// repo (`XiaomiMiMo/MiMo-V2-Flash`) because the sandbox has no network
    /// access to huggingface.co. The values below are the best-known
    /// canonical spellings and should be re-verified against the published
    /// `config.json` / weight index.
    MiMoV2,
    /// GPT-OSS 20B / 120B (OpenAI, `gpt_oss`). Sparse MoE (32 experts for
    /// 20B, 128 for 120B), top-4 softmax routing, **no shared expert**, and
    /// an alternating **1:1 per-layer attention** pattern: even layers use
    /// a 128-token banded sliding window, odd layers use full causal
    /// attention. The router lives at `model.layers.{l}.mlp.router` (not
    /// `mlp.gate`); attention carries a learnable per-layer sink bias
    /// (`self_attn.sinks`) that MER does not yet apply.
    GptOss,
}

impl Default for Architecture {
    fn default() -> Self {
        Architecture::Mixtral
    }
}

/// Why a given architecture cannot (yet) run through the forward-compute
/// path, even though Stage 1 can map and load its tensors. Returned by
/// [`Architecture::compute_support`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeSupport {
    /// The dense + MoE forward path can run this architecture.
    Supported,
    /// Loading is understood but execution needs a capability that is
    /// not implemented yet. `reason` is a human-readable explanation
    /// suitable for surfacing in an error.
    Unsupported { reason: &'static str },
}

impl Architecture {
    /// Map an exact HuggingFace `model_type` string to an architecture.
    ///
    /// These strings are taken verbatim from each family's `config.json`
    /// (`qwen3_moe`, `deepseek_v3`, `phi3`, `mistral3`, …) — they are
    /// **not** guessed display names.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "mixtral" => Some(Self::Mixtral),
            "qwen3" => Some(Self::Qwen3),
            "qwen3_moe" => Some(Self::Qwen3Moe),
            "deepseek_v3" => Some(Self::DeepSeekV3),
            "mistral3" => Some(Self::MistralSmall3),
            "phi3" => Some(Self::Phi4),
            // Best-known canonical `model_type` strings for the two new
            // families (unverifiable in the sandbox — see the variant docs).
            "mimo_v2" => Some(Self::MiMoV2),
            "gpt_oss" => Some(Self::GptOss),
            _ => None,
        }
    }

    /// Map an entry of the `architectures` list in `config.json` (e.g.
    /// `"Qwen3MoeForCausalLM"`) to an architecture.
    pub fn from_hf_architecture(name: &str) -> Option<Self> {
        match name {
            "MixtralForCausalLM" => Some(Self::Mixtral),
            "Qwen3ForCausalLM" => Some(Self::Qwen3),
            "Qwen3MoeForCausalLM" => Some(Self::Qwen3Moe),
            "DeepseekV3ForCausalLM" => Some(Self::DeepSeekV3),
            "Mistral3ForConditionalGeneration" => Some(Self::MistralSmall3),
            "Phi3ForCausalLM" => Some(Self::Phi4),
            "MiMoV2ForCausalLM" => Some(Self::MiMoV2),
            "GptOssForCausalLM" => Some(Self::GptOss),
            _ => None,
        }
    }

    /// The canonical `model_type` string for this architecture.
    pub fn model_type(&self) -> &'static str {
        match self {
            Self::Mixtral => "mixtral",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3_moe",
            Self::DeepSeekV3 => "deepseek_v3",
            Self::MistralSmall3 => "mistral3",
            Self::Phi4 => "phi3",
            Self::MiMoV2 => "mimo_v2",
            Self::GptOss => "gpt_oss",
        }
    }

    /// `true` if the architecture has sparse MoE FFN layers (some layers
    /// may still be dense, e.g. DeepSeek's `first_k_dense_replace`).
    pub fn is_moe(&self) -> bool {
        matches!(
            self,
            Self::Mixtral | Self::Qwen3Moe | Self::DeepSeekV3 | Self::MiMoV2 | Self::GptOss
        )
    }

    /// `true` if the architecture applies per-head QK-Norm (a `head_dim`
    /// RMSNorm on Q and K before RoPE). Qwen3 and Qwen3-MoE do; Mixtral,
    /// Mistral Small 3, Phi-4 and DeepSeek-V3 do not.
    pub fn uses_qk_norm(&self) -> bool {
        matches!(self, Self::Qwen3 | Self::Qwen3Moe)
    }

    /// Whether the forward-compute path can execute this architecture.
    ///
    /// Every variant — including DeepSeek-V3 — now has a runnable compute
    /// path: MLA (latent-KV) attention lives in [`crate::mla`] and FP8
    /// block-wise weight dequantisation is applied at load time, so the
    /// model can be both mapped and executed.
    pub fn compute_support(&self) -> ComputeSupport {
        ComputeSupport::Supported
    }

    // ===================================================================
    // Per-layer hybrid attention (MiMo-V2 5:1 SWA:global, GPT-OSS 1:1).
    //
    // Older families apply a *uniform* attention mode to every layer:
    // full causal (most) or a single sliding window (Mixtral's 4096).
    // MiMo-V2 and GPT-OSS interleave Sliding-Window-Attention (SWA) and
    // Global layers. The interleave is expressed as a single integer
    // "SWA:global ratio" `R` — `R` consecutive SWA layers followed by one
    // global layer — so layer `l` is **global** iff `(l + 1) % (R + 1) == 0`
    // and **SWA** otherwise. MiMo-V2 uses `R = 5` (global at 5, 11, 17, …),
    // GPT-OSS uses `R = 1` (global on every odd layer, SWA on every even
    // layer). `None` means "no hybrid pattern": the uniform window applies.
    // ===================================================================

    /// The architecture's intrinsic SWA:global interleave ratio, or `None`
    /// for families with uniform attention. `Some(5)` for MiMo-V2 (5 SWA
    /// layers per global layer), `Some(1)` for GPT-OSS (alternating).
    ///
    /// This is the *fallback* used when `config.json` does not carry an
    /// explicit ratio field; an explicit config value always wins (see
    /// [`Self::attention_mode`]).
    pub fn swa_global_ratio(&self) -> Option<usize> {
        match self {
            Self::MiMoV2 => Some(5),
            Self::GptOss => Some(1),
            _ => None,
        }
    }

    /// The architecture's intrinsic sliding-window span for its SWA layers
    /// when `config.json` omits `sliding_window`. Both MiMo-V2 and GPT-OSS
    /// use a 128-token banded window. `None` for families whose window (if
    /// any) is always specified in the config (e.g. Mixtral's 4096).
    pub fn default_swa_window(&self) -> Option<usize> {
        match self {
            Self::MiMoV2 | Self::GptOss => Some(128),
            _ => None,
        }
    }

    /// Resolve the [`AttentionMode`] for decoder layer `layer_idx`.
    ///
    /// * `window` is the sliding-window span from `config.json`
    ///   (`sliding_window`); `None`/`Some(0)` disables SWA.
    /// * `swa_global_ratio` is the explicit interleave ratio from the
    ///   config, if any; when `None` the architecture's intrinsic ratio
    ///   ([`Self::swa_global_ratio`]) is used.
    ///
    /// With a hybrid ratio `R` and a positive `window`, layer `l` is
    /// `Global` iff `(l + 1) % (R + 1) == 0`, otherwise
    /// `SlidingWindow { window }`. Without a hybrid ratio the result is
    /// uniform: `SlidingWindow { window }` when a window is set (Mixtral),
    /// else `Global` (Qwen3-MoE, DeepSeek-V3, …) — exactly the legacy
    /// behaviour.
    pub fn attention_mode(
        &self,
        layer_idx: usize,
        window: Option<usize>,
        swa_global_ratio: Option<usize>,
    ) -> AttentionMode {
        let window = window.filter(|&w| w > 0);
        let ratio = swa_global_ratio.or_else(|| self.swa_global_ratio());
        match (window, ratio) {
            // Hybrid interleave: `ratio` SWA layers, then one global. A
            // ratio of 0 (e.g. an explicit `sliding_window_pattern = 1`)
            // degenerates to "every layer global" via the formula below,
            // honouring an explicit request to disable the sliding window.
            (Some(w), Some(r)) => {
                if (layer_idx + 1) % (r + 1) == 0 {
                    AttentionMode::Global
                } else {
                    AttentionMode::SlidingWindow { window: w }
                }
            }
            // Uniform sliding window (Mixtral) — every layer is SWA.
            (Some(w), None) => AttentionMode::SlidingWindow { window: w },
            // No window configured — full causal attention everywhere.
            (None, _) => AttentionMode::Global,
        }
    }
}

/// Per-layer attention mode. Replaces the older "uniform window or
/// nothing" assumption so hybrid models (MiMo-V2, GPT-OSS) can mix
/// Sliding-Window-Attention (SWA) and Global layers within one model.
///
/// Resolved per layer by [`Architecture::attention_mode`] and stored on
/// each [`crate::transformer::MultiHeadSelfAttention`] as its
/// `window_size` (`None` ⇔ [`AttentionMode::Global`], `Some(w)` ⇔
/// [`AttentionMode::SlidingWindow`]). The forward pass branches on this
/// to bound the attention sum (and the decode loop uses it to evict
/// out-of-window KV entries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    /// Full causal attention over all past positions.
    Global,
    /// Sliding window: attend only to the last `window` positions.
    SlidingWindow { window: usize },
}

impl AttentionMode {
    /// The window span for an SWA layer, or `None` for a global layer.
    /// This is the `Option<usize>` the attention block stores as
    /// `window_size`, so a model can be built directly from a resolved
    /// mode without a second match.
    pub fn window(&self) -> Option<usize> {
        match self {
            AttentionMode::Global => None,
            AttentionMode::SlidingWindow { window } => Some(*window),
        }
    }

    /// Build an [`AttentionMode`] from a stored `window_size` field:
    /// `None` ⇒ [`AttentionMode::Global`], `Some(w)` ⇒
    /// [`AttentionMode::SlidingWindow`]. `Some(0)` is treated as Global
    /// (a zero-width window is meaningless).
    pub fn from_window(window: Option<usize>) -> Self {
        match window {
            Some(w) if w > 0 => AttentionMode::SlidingWindow { window: w },
            _ => AttentionMode::Global,
        }
    }
}

/// FFN flavour of a single decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    /// Plain dense SwiGLU FFN (`mlp.{gate,up,down}_proj`, or a fused
    /// `mlp.gate_up_proj` for Phi-4).
    Dense,
    /// Sparse routed MoE block.
    Moe,
}

/// Per-architecture tensor-name mapping.
///
/// All names returned here are the *exact* strings present in the
/// corresponding `.safetensors` checkpoints (verified against real
/// `config.json` / weight-index dumps). The struct also carries the
/// handful of layout flags the loader branches on (fused projections,
/// the `language_model.` prefix) and the `first_k_dense_replace`
/// boundary that classifies dense vs MoE layers.
#[derive(Debug, Clone)]
pub struct TensorNaming {
    arch: Architecture,
    /// `"language_model."` for Mistral Small 3, `""` otherwise. Applied
    /// to every text-model tensor so the rest of the mapping is written
    /// against the un-prefixed names.
    prefix: &'static str,
    /// Number of leading dense layers (DeepSeek `first_k_dense_replace`).
    /// `0` for architectures where every MoE layer is sparse.
    first_k_dense_replace: usize,
}

impl TensorNaming {
    /// Build a naming map for `arch`. `first_k_dense_replace` is only
    /// meaningful for DeepSeek-style MoE (leading dense layers); pass `0`
    /// otherwise. The value is taken straight from `config.json`, never
    /// hard-coded.
    pub fn new(arch: Architecture, first_k_dense_replace: usize) -> Self {
        let prefix = match arch {
            Architecture::MistralSmall3 => "language_model.",
            _ => "",
        };
        Self { arch, prefix, first_k_dense_replace }
    }

    pub fn architecture(&self) -> Architecture {
        self.arch
    }

    /// The `language_model.`-style prefix applied to text-model tensors
    /// (empty for everything except Mistral Small 3).
    pub fn prefix(&self) -> &'static str {
        self.prefix
    }

    /// `true` if attention uses a single fused `qkv_proj` tensor (Phi-4)
    /// rather than separate `q_proj` / `k_proj` / `v_proj`.
    pub fn attn_qkv_fused(&self) -> bool {
        matches!(self.arch, Architecture::Phi4)
    }

    /// `true` if the dense FFN uses a single fused `gate_up_proj` tensor
    /// (Phi-4) rather than separate `gate_proj` / `up_proj`.
    pub fn mlp_gate_up_fused(&self) -> bool {
        matches!(self.arch, Architecture::Phi4)
    }

    /// Classify layer `l` as dense or MoE. Dense-only architectures
    /// (Qwen3 dense, Mistral Small 3, Phi-4) are dense on every layer;
    /// MoE architectures are MoE except for the first
    /// `first_k_dense_replace` layers (DeepSeek-V3).
    pub fn ffn_kind(&self, l: usize) -> FfnKind {
        if !self.arch.is_moe() {
            return FfnKind::Dense;
        }
        if l < self.first_k_dense_replace {
            FfnKind::Dense
        } else {
            FfnKind::Moe
        }
    }

    // -- Global (non-layer) tensors --------------------------------------

    pub fn embed(&self) -> String {
        format!("{}model.embed_tokens.weight", self.prefix)
    }

    pub fn final_norm(&self) -> String {
        format!("{}model.norm.weight", self.prefix)
    }

    pub fn lm_head(&self) -> String {
        format!("{}lm_head.weight", self.prefix)
    }

    // -- Per-layer norms -------------------------------------------------

    pub fn input_layernorm(&self, l: usize) -> String {
        format!("{}model.layers.{l}.input_layernorm.weight", self.prefix)
    }

    pub fn post_attention_layernorm(&self, l: usize) -> String {
        format!("{}model.layers.{l}.post_attention_layernorm.weight", self.prefix)
    }

    // -- Attention projections -------------------------------------------

    /// Fused `qkv_proj` name (only valid when [`Self::attn_qkv_fused`]).
    pub fn attn_qkv(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.qkv_proj.weight", self.prefix)
    }

    pub fn attn_q(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.q_proj.weight", self.prefix)
    }

    pub fn attn_k(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.k_proj.weight", self.prefix)
    }

    pub fn attn_v(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.v_proj.weight", self.prefix)
    }

    pub fn attn_o(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.o_proj.weight", self.prefix)
    }

    /// Per-head Q RMSNorm weight (Qwen3 / Qwen3-MoE "QK-Norm"). Length
    /// `head_dim`. Only present for [`Architecture::uses_qk_norm`] families.
    pub fn attn_q_norm(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.q_norm.weight", self.prefix)
    }

    /// Per-head K RMSNorm weight (Qwen3 / Qwen3-MoE). Length `head_dim`.
    pub fn attn_k_norm(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.k_norm.weight", self.prefix)
    }

    // -- MLA projections (DeepSeek-V3 latent attention) ------------------
    //
    // DeepSeek-V3 replaces the dense q/k/v/o projections with a
    // low-rank "multi-head latent attention" stack. The query path is
    // optionally compressed through `q_a_proj` -> `q_a_layernorm` ->
    // `q_b_proj` (when `q_lora_rank > 0`), and the key/value path is
    // always compressed through `kv_a_proj_with_mqa` -> `kv_a_layernorm`
    // -> `kv_b_proj`. The output projection reuses the standard
    // [`Self::attn_o`] name.

    /// MLA query down-projection (`d_model -> q_lora_rank`). Only present
    /// when `q_lora_rank > 0`.
    pub fn mla_q_a_proj(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.q_a_proj.weight", self.prefix)
    }

    /// RMSNorm applied to the compressed query latent. Length
    /// `q_lora_rank`. Only present when `q_lora_rank > 0`.
    pub fn mla_q_a_layernorm(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.q_a_layernorm.weight", self.prefix)
    }

    /// MLA query up-projection from the compressed latent
    /// (`q_lora_rank -> num_heads * (qk_nope + qk_rope)`). Only present
    /// when `q_lora_rank > 0`; otherwise the loader uses [`Self::attn_q`]
    /// (the single dense `q_proj` straight from `d_model`).
    pub fn mla_q_b_proj(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.q_b_proj.weight", self.prefix)
    }

    /// MLA key/value down-projection with the decoupled RoPE key
    /// (`d_model -> kv_lora_rank + qk_rope_head_dim`).
    pub fn mla_kv_a_proj(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight", self.prefix)
    }

    /// RMSNorm applied to the compressed key/value latent. Length
    /// `kv_lora_rank`.
    pub fn mla_kv_a_layernorm(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.kv_a_layernorm.weight", self.prefix)
    }

    /// MLA key/value up-projection
    /// (`kv_lora_rank -> num_heads * (qk_nope + v_head_dim)`).
    pub fn mla_kv_b_proj(&self, l: usize) -> String {
        format!("{}model.layers.{l}.self_attn.kv_b_proj.weight", self.prefix)
    }

    // -- MoE gate --------------------------------------------------------

    /// Router/gate weight for a MoE layer. Mixtral nests it under
    /// `block_sparse_moe`; Qwen3-MoE and DeepSeek place it at `mlp.gate`.
    pub fn moe_gate(&self, l: usize) -> String {
        match self.arch {
            Architecture::Mixtral => {
                format!("{}model.layers.{l}.block_sparse_moe.gate.weight", self.prefix)
            }
            // GPT-OSS names its router `mlp.router` rather than `mlp.gate`.
            Architecture::GptOss => {
                format!("{}model.layers.{l}.mlp.router.weight", self.prefix)
            }
            // Qwen3-MoE, DeepSeek-V3 and MiMo-V2 all place the router at
            // `mlp.gate`.
            _ => format!("{}model.layers.{l}.mlp.gate.weight", self.prefix),
        }
    }

    /// DeepSeek-V3 per-expert score-correction bias (`e_score_correction_bias`),
    /// stored alongside the gate. Used for selection only (aux-loss-free
    /// load balancing). Absent on Mixtral / Qwen3-MoE.
    pub fn moe_gate_correction_bias(&self, l: usize) -> String {
        format!("{}model.layers.{l}.mlp.gate.e_score_correction_bias", self.prefix)
    }

    // -- Dense FFN (non-MoE layers) --------------------------------------

    /// Fused `gate_up_proj` name (only valid when
    /// [`Self::mlp_gate_up_fused`]).
    pub fn mlp_gate_up(&self, l: usize) -> String {
        format!("{}model.layers.{l}.mlp.gate_up_proj.weight", self.prefix)
    }

    pub fn mlp_gate(&self, l: usize) -> String {
        format!("{}model.layers.{l}.mlp.gate_proj.weight", self.prefix)
    }

    pub fn mlp_up(&self, l: usize) -> String {
        format!("{}model.layers.{l}.mlp.up_proj.weight", self.prefix)
    }

    pub fn mlp_down(&self, l: usize) -> String {
        format!("{}model.layers.{l}.mlp.down_proj.weight", self.prefix)
    }

    // -- Routed experts --------------------------------------------------
    //
    // Expert FFN weights are streamed from SSD per token rather than
    // loaded into the resident `RealModel`, so these helpers exist for
    // the extraction pipeline and for tests that assert the exact names.
    // Mixtral uses `w1`/`w3`/`w2` (gate/up/down); Qwen3-MoE and DeepSeek
    // share `mlp.experts.{j}.{gate,up,down}_proj`.

    /// Expert gate (`w1` / `gate_proj`) for routed expert `j` on layer `l`.
    pub fn expert_gate(&self, l: usize, j: usize) -> String {
        match self.arch {
            Architecture::Mixtral => {
                format!("{}model.layers.{l}.block_sparse_moe.experts.{j}.w1.weight", self.prefix)
            }
            _ => format!("{}model.layers.{l}.mlp.experts.{j}.gate_proj.weight", self.prefix),
        }
    }

    /// Expert up (`w3` / `up_proj`) for routed expert `j` on layer `l`.
    pub fn expert_up(&self, l: usize, j: usize) -> String {
        match self.arch {
            Architecture::Mixtral => {
                format!("{}model.layers.{l}.block_sparse_moe.experts.{j}.w3.weight", self.prefix)
            }
            _ => format!("{}model.layers.{l}.mlp.experts.{j}.up_proj.weight", self.prefix),
        }
    }

    /// Expert down (`w2` / `down_proj`) for routed expert `j` on layer `l`.
    pub fn expert_down(&self, l: usize, j: usize) -> String {
        match self.arch {
            Architecture::Mixtral => {
                format!("{}model.layers.{l}.block_sparse_moe.experts.{j}.w2.weight", self.prefix)
            }
            _ => format!("{}model.layers.{l}.mlp.experts.{j}.down_proj.weight", self.prefix),
        }
    }

    /// Companion FP8 scale tensor for a DeepSeek expert weight. DeepSeek
    /// ships `…_proj.weight` (FP8) alongside `…_proj.weight_scale_inv`
    /// (the per-block dequantisation scale). These scales must be parked
    /// in a *side table* (see [`WeightScaleTable`]) and never used as
    /// weights. `weight_name` is one of [`Self::expert_gate`] /
    /// [`Self::expert_up`] / [`Self::expert_down`].
    pub fn weight_scale_inv_name(weight_name: &str) -> Option<String> {
        let base = weight_name.strip_suffix(".weight")?;
        Some(format!("{base}.weight_scale_inv"))
    }
}

/// A side table holding DeepSeek's FP8 `weight_scale_inv` tensors,
/// indexed by the **weight** tensor name they accompany (i.e. with the
/// `.weight_scale_inv` suffix already mapped back to `.weight`). Keeping
/// scales here makes it impossible for the loader to accidentally treat a
/// scale tensor as a weight while still preserving them for a future FP8
/// dequantisation path.
#[derive(Debug, Default, Clone)]
pub struct WeightScaleTable {
    scales: std::collections::HashMap<String, Vec<f32>>,
}

impl WeightScaleTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the scale tensor that accompanies `weight_name`.
    pub fn insert(&mut self, weight_name: &str, scale: Vec<f32>) {
        self.scales.insert(weight_name.to_string(), scale);
    }

    pub fn get(&self, weight_name: &str) -> Option<&[f32]> {
        self.scales.get(weight_name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.scales.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }
}

/// Error raised while reading or interpreting a HuggingFace `config.json`.
#[derive(Debug)]
pub enum ArchitectureError {
    /// `config.json` was absent or unreadable.
    Io(std::io::Error),
    /// `config.json` was not valid JSON.
    Json(serde_json::Error),
    /// The `architectures` / `model_type` fields did not match any
    /// architecture this build understands.
    UnknownArchitecture { model_type: String, architectures: Vec<String> },
    /// A required hyperparameter was missing or the wrong type.
    MissingField(&'static str),
}

impl std::fmt::Display for ArchitectureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "failed to read config.json: {e}"),
            Self::Json(e) => write!(f, "failed to parse config.json: {e}"),
            Self::UnknownArchitecture { model_type, architectures } => write!(
                f,
                "unsupported model architecture (model_type={model_type:?}, \
                 architectures={architectures:?}); supported model_types are \
                 mixtral, qwen3, qwen3_moe, deepseek_v3, mistral3, phi3, \
                 mimo_v2, gpt_oss"
            ),
            Self::MissingField(name) => {
                write!(f, "config.json is missing required field `{name}`")
            }
        }
    }
}

impl std::error::Error for ArchitectureError {}

impl From<std::io::Error> for ArchitectureError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ArchitectureError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Multi-head latent attention (MLA) projection dims (DeepSeek-V3/V3.1).
///
/// Carried through the loader/config so the future MLA compute workstream
/// has them. They are **not** consumed by the current attention block —
/// DeepSeek-V3 fails loud at load time ([`Architecture::compute_support`])
/// precisely because MLA is not yet implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaDims {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
}

/// RoPE scaling parameters (YaRN), needed by DeepSeek / long-context
/// configs. Carried through from `config.json`'s `rope_scaling` block
/// and applied at attention-compute time via
/// [`crate::transformer::YarnRope`] (blended inverse frequencies +
/// attention-magnitude correction) on both the standard and MLA paths.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    /// `rope_scaling.type` / `rope_type` — e.g. `"yarn"`.
    pub rope_type: String,
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub mscale: f32,
    pub mscale_all_dim: f32,
}

/// Parsed view of a HuggingFace `config.json`, normalised across the
/// per-family key spellings. Every value is sourced from the file — there
/// are no hard-coded layer or expert counts.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub architecture: Architecture,
    pub model_type: String,
    pub hidden_size: usize,
    /// Dense FFN intermediate size (`intermediate_size`).
    pub intermediate_size: usize,
    /// MoE expert intermediate size (`moe_intermediate_size`), if present.
    pub moe_intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit per-head dim (`head_dim`), if the config specifies one
    /// (Qwen3 does; it need not equal `hidden_size / num_heads`).
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    /// Hybrid-attention interleave ratio: number of consecutive
    /// Sliding-Window-Attention layers between each Global layer (e.g. `5`
    /// for MiMo-V2's 5:1 SWA:global pattern, `1` for GPT-OSS's alternating
    /// pattern). `None` ⇒ uniform attention (all Global, or all SWA when
    /// `sliding_window` is set, as for Mixtral). Parsed from the config
    /// when present; otherwise the architecture's intrinsic ratio
    /// ([`Architecture::swa_global_ratio`]) is used at resolve time.
    pub swa_global_ratio: Option<usize>,
    // -- MoE --
    /// Number of routed experts (`num_local_experts` / `num_experts` /
    /// `n_routed_experts` depending on family).
    pub num_routed_experts: Option<usize>,
    /// Experts activated per token (`num_experts_per_tok`).
    pub num_experts_per_tok: Option<usize>,
    pub num_shared_experts: Option<usize>,
    pub first_k_dense_replace: Option<usize>,
    pub scoring_func: Option<String>,
    pub topk_method: Option<String>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub routed_scaling_factor: Option<f32>,
    pub norm_topk_prob: Option<bool>,
    // -- MLA (DeepSeek-V3) --
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub qk_nope_head_dim: Option<usize>,
    pub v_head_dim: Option<usize>,
    // -- RoPE scaling (YaRN) --
    pub rope_scaling: Option<RopeScaling>,
}

fn as_usize(v: &serde_json::Value) -> Option<usize> {
    v.as_u64().map(|n| n as usize)
}

fn as_f32(v: &serde_json::Value) -> Option<f32> {
    v.as_f64().map(|n| n as f32)
}

/// Parse a `config.json` `rope_scaling` object into [`RopeScaling`].
/// Accepts both the `rope_type` and legacy `type` spellings. Returns
/// `None` for a missing/non-object value or one without a `factor`.
fn parse_rope_scaling(v: &serde_json::Value) -> Option<RopeScaling> {
    let obj = v.as_object()?;
    let rope_type = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let factor = obj.get("factor").and_then(as_f32)?;
    let getf = |k: &str, d: f32| obj.get(k).and_then(as_f32).unwrap_or(d);
    Some(RopeScaling {
        rope_type,
        factor,
        original_max_position_embeddings: obj
            .get("original_max_position_embeddings")
            .and_then(as_usize)
            .unwrap_or(0),
        beta_fast: getf("beta_fast", 32.0),
        beta_slow: getf("beta_slow", 1.0),
        mscale: getf("mscale", 1.0),
        mscale_all_dim: getf("mscale_all_dim", 0.0),
    })
}

impl HfConfig {
    /// Read and parse `<dir>/config.json`. Returns `Ok(None)` when the
    /// file does not exist (so callers can fall back to TOML-derived
    /// config), `Ok(Some(_))` on success, and `Err` for a malformed file
    /// or an unrecognised architecture (fail-loud).
    pub fn from_dir(dir: &Path) -> Result<Option<Self>, ArchitectureError> {
        let path = dir.join("config.json");
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::from_json_str(&text).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(ArchitectureError::from(e)),
        }
    }

    /// Parse a `config.json` document from a string.
    pub fn from_json_str(text: &str) -> Result<Self, ArchitectureError> {
        let root: serde_json::Value = serde_json::from_str(text)?;

        // Resolve the architecture from the top-level `architectures`
        // list first (most specific), then fall back to `model_type`.
        let architectures: Vec<String> = root
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let top_model_type =
            root.get("model_type").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let architecture = architectures
            .iter()
            .find_map(|a| Architecture::from_hf_architecture(a))
            .or_else(|| Architecture::from_model_type(&top_model_type))
            .ok_or_else(|| ArchitectureError::UnknownArchitecture {
                model_type: top_model_type.clone(),
                architectures: architectures.clone(),
            })?;

        // Multimodal `…ForConditionalGeneration` checkpoints (Mistral
        // Small 3) nest the language-model hyperparameters under
        // `text_config`; everything else keeps them at the top level.
        let hp = root.get("text_config").unwrap_or(&root);
        let get = |key: &str| hp.get(key).or_else(|| root.get(key));

        let req_usize = |key: &'static str| -> Result<usize, ArchitectureError> {
            get(key).and_then(as_usize).ok_or(ArchitectureError::MissingField(key))
        };

        let hidden_size = req_usize("hidden_size")?;
        let num_hidden_layers = req_usize("num_hidden_layers")?;
        let num_attention_heads = req_usize("num_attention_heads")?;
        let num_key_value_heads =
            get("num_key_value_heads").and_then(as_usize).unwrap_or(num_attention_heads);
        let vocab_size = req_usize("vocab_size")?;
        let intermediate_size = get("intermediate_size").and_then(as_usize).unwrap_or(0);

        let rms_norm_eps = get("rms_norm_eps").and_then(as_f32).unwrap_or(1e-6);
        let rope_theta = get("rope_theta").and_then(as_f32).unwrap_or(10_000.0);
        let head_dim = get("head_dim").and_then(as_usize);
        let sliding_window = get("sliding_window").and_then(as_usize);

        // Hybrid-attention interleave ratio. Spellings vary by family, so
        // we accept a few:
        //   * `swa_global_ratio` — direct value (number of SWA layers per
        //     global layer).
        //   * `sliding_window_pattern` — HF convention where every P-th
        //     layer is global, i.e. `P - 1` SWA layers per global one. A
        //     pattern of 1 maps to ratio 0 (every layer global) and is
        //     preserved; `checked_sub` drops only the nonsensical `P = 0`.
        // When absent, the architecture's intrinsic ratio is applied at
        // resolve time (see `Architecture::attention_mode`), so MiMo-V2 /
        // GPT-OSS still get the correct pattern from a config that only
        // carries `sliding_window`.
        let swa_global_ratio = get("swa_global_ratio")
            .and_then(as_usize)
            .or_else(|| {
                get("sliding_window_pattern")
                    .and_then(as_usize)
                    .and_then(|p| p.checked_sub(1))
            });

        // Routed-expert count spelling differs per family.
        let num_routed_experts = get("num_local_experts")
            .and_then(as_usize)
            .or_else(|| get("num_experts").and_then(as_usize))
            .or_else(|| get("n_routed_experts").and_then(as_usize));

        Ok(HfConfig {
            architecture,
            model_type: if top_model_type.is_empty() {
                architecture.model_type().to_string()
            } else {
                top_model_type
            },
            hidden_size,
            intermediate_size,
            moe_intermediate_size: get("moe_intermediate_size").and_then(as_usize),
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            sliding_window,
            swa_global_ratio,
            num_routed_experts,
            num_experts_per_tok: get("num_experts_per_tok").and_then(as_usize),
            num_shared_experts: get("n_shared_experts").and_then(as_usize),
            first_k_dense_replace: get("first_k_dense_replace").and_then(as_usize),
            scoring_func: get("scoring_func").and_then(|v| v.as_str().map(String::from)),
            topk_method: get("topk_method").and_then(|v| v.as_str().map(String::from)),
            n_group: get("n_group").and_then(as_usize),
            topk_group: get("topk_group").and_then(as_usize),
            routed_scaling_factor: get("routed_scaling_factor").and_then(as_f32),
            norm_topk_prob: get("norm_topk_prob").and_then(|v| v.as_bool()),
            q_lora_rank: get("q_lora_rank").and_then(as_usize),
            kv_lora_rank: get("kv_lora_rank").and_then(as_usize),
            qk_rope_head_dim: get("qk_rope_head_dim").and_then(as_usize),
            qk_nope_head_dim: get("qk_nope_head_dim").and_then(as_usize),
            v_head_dim: get("v_head_dim").and_then(as_usize),
            rope_scaling: get("rope_scaling").and_then(parse_rope_scaling),
        })
    }

    /// A [`TensorNaming`] map for this config (carrying the
    /// `first_k_dense_replace` boundary from the file).
    pub fn tensor_naming(&self) -> TensorNaming {
        TensorNaming::new(self.architecture, self.first_k_dense_replace.unwrap_or(0))
    }

    /// The per-head attention dimension, preferring the explicit
    /// `head_dim` from the config and otherwise deriving it from
    /// `hidden_size / num_attention_heads`.
    pub fn resolved_head_dim(&self) -> usize {
        self.head_dim.unwrap_or_else(|| {
            if self.num_attention_heads == 0 {
                0
            } else {
                self.hidden_size / self.num_attention_heads
            }
        })
    }

    /// The FFN width the resident model should size its routed gate /
    /// dense FFN against: the MoE expert width for MoE families,
    /// otherwise the dense `intermediate_size`.
    pub fn resolved_d_ff(&self) -> usize {
        if self.architecture.is_moe() {
            self.moe_intermediate_size.or(Some(self.intermediate_size)).unwrap_or(0)
        } else {
            self.intermediate_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_type_strings_match_exact_config_values() {
        // Exact `model_type` strings from each family's config.json.
        assert_eq!(Architecture::from_model_type("mixtral"), Some(Architecture::Mixtral));
        assert_eq!(Architecture::from_model_type("qwen3_moe"), Some(Architecture::Qwen3Moe));
        assert_eq!(Architecture::from_model_type("qwen3"), Some(Architecture::Qwen3));
        assert_eq!(Architecture::from_model_type("deepseek_v3"), Some(Architecture::DeepSeekV3));
        assert_eq!(Architecture::from_model_type("mistral3"), Some(Architecture::MistralSmall3));
        assert_eq!(Architecture::from_model_type("phi3"), Some(Architecture::Phi4));
        // Guard against fuzzy/guessed names slipping in.
        assert_eq!(Architecture::from_model_type("qwen3moe"), None);
        assert_eq!(Architecture::from_model_type("deepseek-v3"), None);
        assert_eq!(Architecture::from_model_type("phi4"), None);
        assert_eq!(Architecture::from_model_type("mistral"), None);
    }

    #[test]
    fn hf_architecture_names_map() {
        assert_eq!(
            Architecture::from_hf_architecture("Qwen3MoeForCausalLM"),
            Some(Architecture::Qwen3Moe)
        );
        assert_eq!(
            Architecture::from_hf_architecture("DeepseekV3ForCausalLM"),
            Some(Architecture::DeepSeekV3)
        );
        assert_eq!(
            Architecture::from_hf_architecture("Mistral3ForConditionalGeneration"),
            Some(Architecture::MistralSmall3)
        );
        assert_eq!(
            Architecture::from_hf_architecture("Phi3ForCausalLM"),
            Some(Architecture::Phi4)
        );
        assert_eq!(Architecture::from_hf_architecture("LlamaForCausalLM"), None);
    }

    #[test]
    fn mistral_small_3_carries_language_model_prefix() {
        let n = TensorNaming::new(Architecture::MistralSmall3, 0);
        assert_eq!(n.embed(), "language_model.model.embed_tokens.weight");
        assert_eq!(n.lm_head(), "language_model.lm_head.weight");
        assert_eq!(n.final_norm(), "language_model.model.norm.weight");
        assert_eq!(
            n.attn_q(0),
            "language_model.model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            n.mlp_down(7),
            "language_model.model.layers.7.mlp.down_proj.weight"
        );
        assert!(!n.attn_qkv_fused());
        assert!(!n.mlp_gate_up_fused());
    }

    #[test]
    fn phi4_uses_fused_projections() {
        let n = TensorNaming::new(Architecture::Phi4, 0);
        assert!(n.attn_qkv_fused());
        assert!(n.mlp_gate_up_fused());
        assert_eq!(n.attn_qkv(3), "model.layers.3.self_attn.qkv_proj.weight");
        assert_eq!(n.mlp_gate_up(3), "model.layers.3.mlp.gate_up_proj.weight");
        assert_eq!(n.embed(), "model.embed_tokens.weight");
        assert_eq!(n.lm_head(), "lm_head.weight");
    }

    #[test]
    fn mixtral_vs_qwen_gate_names() {
        let mixtral = TensorNaming::new(Architecture::Mixtral, 0);
        assert_eq!(
            mixtral.moe_gate(2),
            "model.layers.2.block_sparse_moe.gate.weight"
        );
        let qwen = TensorNaming::new(Architecture::Qwen3Moe, 0);
        assert_eq!(qwen.moe_gate(2), "model.layers.2.mlp.gate.weight");
    }

    #[test]
    fn qwen3_and_deepseek_share_expert_pattern() {
        let qwen = TensorNaming::new(Architecture::Qwen3Moe, 0);
        let deepseek = TensorNaming::new(Architecture::DeepSeekV3, 3);
        assert_eq!(qwen.expert_gate(5, 9), "model.layers.5.mlp.experts.9.gate_proj.weight");
        assert_eq!(deepseek.expert_gate(5, 9), "model.layers.5.mlp.experts.9.gate_proj.weight");
        assert_eq!(deepseek.expert_up(5, 9), "model.layers.5.mlp.experts.9.up_proj.weight");
        assert_eq!(deepseek.expert_down(5, 9), "model.layers.5.mlp.experts.9.down_proj.weight");
    }

    #[test]
    fn deepseek_dense_vs_moe_boundary_from_first_k_dense_replace() {
        // first_k_dense_replace = 3 -> layers 0,1,2 dense; 3+ MoE.
        let n = TensorNaming::new(Architecture::DeepSeekV3, 3);
        assert_eq!(n.ffn_kind(0), FfnKind::Dense);
        assert_eq!(n.ffn_kind(2), FfnKind::Dense);
        assert_eq!(n.ffn_kind(3), FfnKind::Moe);
        assert_eq!(n.ffn_kind(60), FfnKind::Moe);
        // Mixtral has no dense prefix: every layer is MoE.
        let mix = TensorNaming::new(Architecture::Mixtral, 0);
        assert_eq!(mix.ffn_kind(0), FfnKind::Moe);
        // Dense families are dense everywhere.
        let phi = TensorNaming::new(Architecture::Phi4, 0);
        assert_eq!(phi.ffn_kind(0), FfnKind::Dense);
        assert_eq!(phi.ffn_kind(40), FfnKind::Dense);
    }

    #[test]
    fn weight_scale_inv_name_mapping() {
        let n = TensorNaming::new(Architecture::DeepSeekV3, 3);
        let w = n.expert_gate(10, 4);
        assert_eq!(w, "model.layers.10.mlp.experts.4.gate_proj.weight");
        assert_eq!(
            TensorNaming::weight_scale_inv_name(&w).unwrap(),
            "model.layers.10.mlp.experts.4.gate_proj.weight_scale_inv"
        );
        assert!(TensorNaming::weight_scale_inv_name("no_suffix").is_none());
    }

    #[test]
    fn weight_scale_table_keeps_scales_separate() {
        let mut t = WeightScaleTable::new();
        assert!(t.is_empty());
        t.insert("model.layers.3.mlp.experts.0.gate_proj.weight", vec![1.0, 2.0]);
        assert_eq!(t.len(), 1);
        assert_eq!(t.get("model.layers.3.mlp.experts.0.gate_proj.weight"), Some(&[1.0, 2.0][..]));
        assert!(t.get("model.layers.3.mlp.experts.0.up_proj.weight").is_none());
    }

    #[test]
    fn deepseek_compute_is_supported() {
        // DeepSeek-V3 is now executable: MLA attention + FP8 dequant are
        // implemented, so every recognised architecture reports Supported.
        assert_eq!(Architecture::DeepSeekV3.compute_support(), ComputeSupport::Supported);
        assert_eq!(Architecture::Qwen3Moe.compute_support(), ComputeSupport::Supported);
        assert_eq!(Architecture::Mixtral.compute_support(), ComputeSupport::Supported);
    }

    #[test]
    fn read_qwen3_moe_config() {
        let json = r#"{
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "norm_topk_prob": true
        }"#;
        let cfg = HfConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen3Moe);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, Some(128));
        assert_eq!(cfg.num_routed_experts, Some(128));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.resolved_d_ff(), 768);
        assert_eq!(cfg.resolved_head_dim(), 128);
    }

    #[test]
    fn read_deepseek_v3_config() {
        let json = r#"{
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "vocab_size": 129280,
            "num_experts_per_tok": 8,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "first_k_dense_replace": 3,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "n_group": 8,
            "topk_group": 4,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true
        }"#;
        let cfg = HfConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.architecture, Architecture::DeepSeekV3);
        assert_eq!(cfg.num_routed_experts, Some(256));
        assert_eq!(cfg.num_shared_experts, Some(1));
        assert_eq!(cfg.first_k_dense_replace, Some(3));
        assert_eq!(cfg.scoring_func.as_deref(), Some("sigmoid"));
        assert_eq!(cfg.topk_method.as_deref(), Some("noaux_tc"));
        assert_eq!(cfg.n_group, Some(8));
        assert_eq!(cfg.topk_group, Some(4));
        assert_eq!(cfg.routed_scaling_factor, Some(2.5));
        // Dense/MoE boundary is driven by the config value.
        let n = cfg.tensor_naming();
        assert_eq!(n.ffn_kind(2), FfnKind::Dense);
        assert_eq!(n.ffn_kind(3), FfnKind::Moe);
    }

    #[test]
    fn read_mistral_small_3_text_config_nesting() {
        // Mistral3ForConditionalGeneration nests the LM hyperparameters
        // under `text_config`.
        let json = r#"{
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3",
            "text_config": {
                "hidden_size": 5120,
                "intermediate_size": 32768,
                "num_hidden_layers": 40,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 131072,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000000.0
            }
        }"#;
        let cfg = HfConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.architecture, Architecture::MistralSmall3);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.resolved_head_dim(), 128);
        // Dense model: resolved FFN width is the dense intermediate size.
        assert_eq!(cfg.resolved_d_ff(), 32768);
    }

    #[test]
    fn read_phi4_config() {
        let json = r#"{
            "architectures": ["Phi3ForCausalLM"],
            "model_type": "phi3",
            "hidden_size": 5120,
            "intermediate_size": 17920,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 10,
            "vocab_size": 100352,
            "rms_norm_eps": 1e-5,
            "rope_theta": 250000.0
        }"#;
        let cfg = HfConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.architecture, Architecture::Phi4);
        assert_eq!(cfg.resolved_head_dim(), 128); // 5120 / 40
        let n = cfg.tensor_naming();
        assert!(n.attn_qkv_fused());
        assert!(n.mlp_gate_up_fused());
    }

    #[test]
    fn unknown_architecture_fails_loud() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        }"#;
        let err = HfConfig::from_json_str(json).unwrap_err();
        assert!(matches!(err, ArchitectureError::UnknownArchitecture { .. }));
    }

    // -- MiMo-V2 / GPT-OSS registration + hybrid attention ----------------

    #[test]
    fn mimo_v2_model_type_maps() {
        assert_eq!(Architecture::from_model_type("mimo_v2"), Some(Architecture::MiMoV2));
        assert_eq!(
            Architecture::from_hf_architecture("MiMoV2ForCausalLM"),
            Some(Architecture::MiMoV2)
        );
        // Round-trip through the canonical string.
        assert_eq!(Architecture::MiMoV2.model_type(), "mimo_v2");
        assert!(Architecture::MiMoV2.is_moe());
        // MiMo-V2 does not use QK-Norm.
        assert!(!Architecture::MiMoV2.uses_qk_norm());
        assert_eq!(Architecture::MiMoV2.compute_support(), ComputeSupport::Supported);
        // Tensor naming follows the Qwen3-MoE / DeepSeek convention.
        let n = TensorNaming::new(Architecture::MiMoV2, 0);
        assert_eq!(n.moe_gate(2), "model.layers.2.mlp.gate.weight");
        assert_eq!(n.expert_gate(5, 9), "model.layers.5.mlp.experts.9.gate_proj.weight");
    }

    #[test]
    fn gpt_oss_model_type_maps() {
        assert_eq!(Architecture::from_model_type("gpt_oss"), Some(Architecture::GptOss));
        assert_eq!(
            Architecture::from_hf_architecture("GptOssForCausalLM"),
            Some(Architecture::GptOss)
        );
        assert_eq!(Architecture::GptOss.model_type(), "gpt_oss");
        assert!(Architecture::GptOss.is_moe());
        assert_eq!(Architecture::GptOss.compute_support(), ComputeSupport::Supported);
        // GPT-OSS routes through `mlp.router`, not `mlp.gate`.
        let n = TensorNaming::new(Architecture::GptOss, 0);
        assert_eq!(n.moe_gate(3), "model.layers.3.mlp.router.weight");
    }

    #[test]
    fn mimo_v2_attention_mode_pattern() {
        // 36-layer MiMo-V2: 5:1 SWA:global. Global at 5, 11, 17, 23, 29, 35;
        // every other layer is SWA with a 128-token window.
        let arch = Architecture::MiMoV2;
        let window = Some(128);
        for l in 0..36 {
            let mode = arch.attention_mode(l, window, None);
            if (l + 1) % 6 == 0 {
                assert_eq!(mode, AttentionMode::Global, "layer {l} should be global");
            } else {
                assert_eq!(
                    mode,
                    AttentionMode::SlidingWindow { window: 128 },
                    "layer {l} should be SWA"
                );
            }
        }
        // Spot-check the boundaries the gist calls out.
        assert_eq!(arch.attention_mode(0, window, None), AttentionMode::SlidingWindow { window: 128 });
        assert_eq!(arch.attention_mode(4, window, None), AttentionMode::SlidingWindow { window: 128 });
        assert_eq!(arch.attention_mode(5, window, None), AttentionMode::Global);
        assert_eq!(arch.attention_mode(6, window, None), AttentionMode::SlidingWindow { window: 128 });
        assert_eq!(arch.attention_mode(11, window, None), AttentionMode::Global);
    }

    #[test]
    fn gpt_oss_attention_mode_alternating() {
        // GPT-OSS alternates 1:1 — even layers SWA (banded 128), odd layers
        // global.
        let arch = Architecture::GptOss;
        let window = Some(128);
        for l in 0..24 {
            let mode = arch.attention_mode(l, window, None);
            if l % 2 == 0 {
                assert_eq!(
                    mode,
                    AttentionMode::SlidingWindow { window: 128 },
                    "even layer {l} should be SWA"
                );
            } else {
                assert_eq!(mode, AttentionMode::Global, "odd layer {l} should be global");
            }
        }
    }

    #[test]
    fn explicit_config_ratio_overrides_intrinsic() {
        // An explicit `swa_global_ratio` from config.json wins over the
        // architecture's intrinsic ratio.
        let arch = Architecture::GptOss; // intrinsic ratio 1
        let window = Some(128);
        // Force a 2:1 pattern: global only at layers 2, 5, 8, …
        assert_eq!(arch.attention_mode(0, window, Some(2)), AttentionMode::SlidingWindow { window: 128 });
        assert_eq!(arch.attention_mode(1, window, Some(2)), AttentionMode::SlidingWindow { window: 128 });
        assert_eq!(arch.attention_mode(2, window, Some(2)), AttentionMode::Global);
    }

    #[test]
    fn uniform_attention_modes_for_legacy_families() {
        // No window ⇒ every layer is Global (Qwen3-MoE, DeepSeek-V3, …).
        for l in 0..8 {
            assert_eq!(Architecture::Qwen3Moe.attention_mode(l, None, None), AttentionMode::Global);
            assert_eq!(Architecture::DeepSeekV3.attention_mode(l, None, None), AttentionMode::Global);
        }
        // Mixtral's uniform sliding window applies to every layer.
        for l in 0..8 {
            assert_eq!(
                Architecture::Mixtral.attention_mode(l, Some(4096), None),
                AttentionMode::SlidingWindow { window: 4096 }
            );
        }
    }

    #[test]
    fn attention_mode_window_roundtrip() {
        assert_eq!(AttentionMode::Global.window(), None);
        assert_eq!(AttentionMode::SlidingWindow { window: 128 }.window(), Some(128));
        assert_eq!(AttentionMode::from_window(None), AttentionMode::Global);
        assert_eq!(AttentionMode::from_window(Some(0)), AttentionMode::Global);
        assert_eq!(
            AttentionMode::from_window(Some(64)),
            AttentionMode::SlidingWindow { window: 64 }
        );
    }

    #[test]
    fn read_gpt_oss_config_parses_window_and_ratio() {
        // Minimal GPT-OSS-style config: `sliding_window_pattern: 2` means
        // every 2nd layer is global ⇒ ratio 1 (alternating).
        let json = r#"{
            "architectures": ["GptOssForCausalLM"],
            "model_type": "gpt_oss",
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "moe_intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "rope_theta": 150000.0,
            "sliding_window": 128,
            "sliding_window_pattern": 2,
            "num_local_experts": 32,
            "num_experts_per_tok": 4
        }"#;
        let cfg = HfConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.architecture, Architecture::GptOss);
        assert_eq!(cfg.sliding_window, Some(128));
        assert_eq!(cfg.swa_global_ratio, Some(1));
        assert_eq!(cfg.num_routed_experts, Some(32));
        assert_eq!(cfg.num_experts_per_tok, Some(4));
    }

    #[test]
    fn read_mimo_v2_config_uses_intrinsic_ratio() {
        // A MiMo-V2 config that only carries `sliding_window` still yields
        // the 5:1 pattern via the architecture's intrinsic ratio.
        let json = r#"{
            "architectures": ["MiMoV2ForCausalLM"],
            "model_type": "mimo_v2",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1536,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "sliding_window": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_nextn_predict_layers": 1
        }"#;
        let cfg = HfConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.architecture, Architecture::MiMoV2);
        assert_eq!(cfg.sliding_window, Some(128));
        // No explicit ratio in the config ⇒ falls back to intrinsic 5.
        assert_eq!(cfg.swa_global_ratio, None);
        assert_eq!(
            cfg.architecture.attention_mode(5, cfg.sliding_window, cfg.swa_global_ratio),
            AttentionMode::Global
        );
        assert_eq!(
            cfg.architecture.attention_mode(0, cfg.sliding_window, cfg.swa_global_ratio),
            AttentionMode::SlidingWindow { window: 128 }
        );
    }
}
