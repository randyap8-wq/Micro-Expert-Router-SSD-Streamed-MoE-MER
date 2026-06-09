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
        }
    }

    /// `true` if the architecture has sparse MoE FFN layers (some layers
    /// may still be dense, e.g. DeepSeek's `first_k_dense_replace`).
    pub fn is_moe(&self) -> bool {
        matches!(self, Self::Mixtral | Self::Qwen3Moe | Self::DeepSeekV3)
    }

    /// Whether the forward-compute path can execute this architecture.
    ///
    /// Stage 1 maps and loads tensors for every variant, but DeepSeek-V3
    /// additionally requires MLA (latent-KV) attention and FP8 weight
    /// dequantisation, neither of which is implemented. Callers building
    /// a runnable model must check this and fail loud rather than run on
    /// garbage activations.
    pub fn compute_support(&self) -> ComputeSupport {
        match self {
            Self::DeepSeekV3 => ComputeSupport::Unsupported {
                reason: "DeepSeek-V3 requires MLA (latent-KV) attention and FP8 weight \
                         dequantisation, which are not implemented yet (tensor mapping \
                         is recognised, but the model cannot be executed)",
            },
            // Dense families (Mistral Small 3, Phi-4) and dense Qwen3 run
            // through the standard attention + SwiGLU path; the
            // dense-FFN-as-single-expert wiring is a later stage but the
            // architecture itself is executable.
            _ => ComputeSupport::Supported,
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

    // -- MoE gate --------------------------------------------------------

    /// Router/gate weight for a MoE layer. Mixtral nests it under
    /// `block_sparse_moe`; Qwen3-MoE and DeepSeek place it at `mlp.gate`.
    pub fn moe_gate(&self, l: usize) -> String {
        match self.arch {
            Architecture::Mixtral => {
                format!("{}model.layers.{l}.block_sparse_moe.gate.weight", self.prefix)
            }
            _ => format!("{}model.layers.{l}.mlp.gate.weight", self.prefix),
        }
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
                 mixtral, qwen3, qwen3_moe, deepseek_v3, mistral3, phi3"
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
}

fn as_usize(v: &serde_json::Value) -> Option<usize> {
    v.as_u64().map(|n| n as usize)
}

fn as_f32(v: &serde_json::Value) -> Option<f32> {
    v.as_f64().map(|n| n as f32)
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
    fn deepseek_compute_is_unsupported_but_recognised() {
        assert!(matches!(
            Architecture::DeepSeekV3.compute_support(),
            ComputeSupport::Unsupported { .. }
        ));
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
}
