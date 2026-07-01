//! GGUF → engine per-expert binary extractor (gist Phase 2).
//!
//! Reads a Mixtral-style GGUF file via [`crate::gguf::GgufFile`] and
//! writes one `expert_<global_id>.bin` per (layer, local_expert) pair,
//! plus a `metadata.json` and the dense weight files
//! [`crate::model::RealModel::from_dir`] consumes.
//!
//! Two on-disk expert layouts are recognised in the wild:
//!
//! * **Interleaved** (mainline `llama.cpp`): one tensor per matrix per
//!   layer, with the expert axis as the **outermost** dim, e.g.
//!   `blk.{layer}.ffn_gate_exps.weight` of shape `[d_model, d_ff,
//!   num_experts]` in GGML order (innermost-first).
//! * **Per-expert**: one tensor per matrix per (layer, expert) pair,
//!   e.g. `blk.{layer}.ffn_gate.{e}.weight`.
//!
//! Both are handled. The output expert file has the layout the engine
//! already consumes: `gate_proj || up_proj || down_proj` row-major, with
//! gate / up shape `[d_ff, d_model]` and down shape `[d_model, d_ff]`.
//! For F32, F16 and BF16 source dtypes the bytes are repacked into that
//! layout directly (BF16 and F16 are decoded to F32); for Q4_0 / Q4_K we
//! currently dequantise to F32 because the
//! GGUF stores each (gate, up, down) tensor as a single block stream
//! that doesn't slice cleanly along the expert axis at the byte level.
//! This preserves the **engine's** on-disk format invariants
//! (`expert_size` is the same for every expert, the file is page-padded,
//! and `metadata.json::dtype` correctly describes the contents).

use crate::dense_tensor::{
    dense_checksum, DenseDType, DenseTensorManifest, DenseTensorManifestEntry,
};
use crate::gguf::{ggml_dtype, GgufFile, GgufSource, GgufTensorInfo, GgufValue};
use crate::inference::{
    dequantize_bf16_to_f32, dequantize_f16_to_f32, expert_weight_bytes_for,
    projection_weight_bytes_for, WeightDtype,
    Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q5K_BLOCK_BYTES,
    Q5K_BLOCK_ELEMS, Q6K_BLOCK_BYTES, Q6K_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use crate::tensor_header::{MixedExpertHeader, ProjectionRange, TensorHeader, UthDtypeId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, warn};

/// Summary of what `extract_experts_from_gguf` wrote.
#[derive(Debug, Clone)]
pub struct ExtractionReport {
    /// Number of expert files written.
    pub experts_written: usize,
    /// Number of dense tensor payload files written. Manifest aliases
    /// such as `embed.bin` / `attn_<L>_q.bin` do not create duplicate
    /// files and are not counted separately.
    pub dense_written: usize,
    /// Number of tensors the loader skipped (missing in the GGUF, or
    /// an unsupported dtype like Q6_K). The engine falls back to seeded
    /// init for these, so the engine still runs.
    pub skipped: usize,
    /// Total bytes written across all output files.
    pub total_bytes: u64,
    /// On-disk dtype every expert file uses (i.e. what gets recorded
    /// in `metadata.json::dtype`). F32 unless every expert tensor in
    /// the source was the same single-block-quant dtype the engine
    /// can ingest unchanged.
    pub expert_dtype: WeightDtype,
    /// `d_model` extracted from the GGUF.
    pub d_model: usize,
    /// `d_ff` extracted from the GGUF.
    pub d_ff: usize,
    /// `num_experts` per layer.
    pub num_experts_per_layer: usize,
    /// Total layers processed.
    pub num_layers: usize,
}

#[derive(Serialize)]
struct ExtractedMetadata<'a> {
    format_version: u32,
    conversion_mode: &'a str,
    model: &'a str,
    architecture: &'a str,
    /// `-1` here means "experts from all layers were concatenated";
    /// we mirror the convention of [`scripts/extract_mixtral_experts.py`]
    /// which writes a single integer for one-layer dumps and `-1` for
    /// multi-layer dumps.
    layer: i32,
    num_experts: usize,
    top_k: usize,
    d_model: usize,
    d_ff: usize,
    expert_size: usize,
    maximum_payload_bytes: usize,
    block_align: usize,
    dtype: &'a str,
    weight_layout: &'a str,
    num_layers: usize,
    num_experts_per_layer: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    expert_layout_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    projection_dtype_histogram: Option<BTreeMap<String, usize>>,
    experts_written: usize,
    dense_tensors_written: usize,
}

/// Block alignment used for every expert file written by this loader.
/// Mirrors the engine's default in `main.rs`.
pub const DEFAULT_BLOCK_ALIGN: usize = 4096;

#[derive(Debug, Clone)]
struct ConversionPreflightPlan {
    architecture: String,
    num_layers: usize,
    num_experts: usize,
    total_experts: usize,
    d_model: usize,
    d_ff: usize,
    top_k: usize,
    expert_size: usize,
    maximum_payload_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct ConversionGeometry {
    num_layers: usize,
    num_experts: usize,
    total_experts: usize,
    d_model: usize,
    d_ff: usize,
    top_k: usize,
}

fn metadata_lookup<'a>(
    meta: &'a std::collections::HashMap<String, GgufValue>,
    arch: Option<&str>,
    suffix: &str,
) -> Option<&'a GgufValue> {
    let dotted = format!(".{suffix}");
    meta.get(&format!("llama.{suffix}"))
        .or_else(|| arch.and_then(|a| meta.get(&format!("{a}.{suffix}"))))
        .or_else(|| {
            meta.iter()
                .filter(|(k, _)| k.ends_with(&dotted))
                .min_by(|(a, _), (b, _)| a.cmp(b))
                .map(|(_, v)| v)
        })
}

fn resolve_geometry(
    gguf: &dyn GgufSource,
    num_layers_hint: usize,
    num_experts_hint: usize,
) -> io::Result<ConversionGeometry> {
    let meta = gguf.metadata();
    let arch = meta.get("general.architecture").and_then(|v| v.as_str());
    let lookup = |suffix: &str| metadata_lookup(meta, arch, suffix);

    let num_layers = if num_layers_hint > 0 {
        num_layers_hint
    } else {
        lookup("block_count")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF missing `block_count`; pass --num-layers explicitly",
                )
            })? as usize
    };
    let num_experts = if num_experts_hint > 0 {
        num_experts_hint
    } else {
        lookup("expert_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize
    };
    if num_experts == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF has no `expert_count` and no --num-experts override",
        ));
    }
    let d_model = lookup("embedding_length")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "GGUF missing embedding_length")
        })? as usize;
    let d_ff = lookup("expert_feed_forward_length")
        .or_else(|| lookup("feed_forward_length"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF missing feed_forward_length",
            )
        })? as usize;
    let top_k = lookup("expert_used_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let total_experts = num_experts.checked_mul(num_layers).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "num_experts * num_layers overflows usize: num_experts={num_experts}, num_layers={num_layers}"
            ),
        )
    })?;

    Ok(ConversionGeometry {
        num_layers,
        num_experts,
        total_experts,
        d_model,
        d_ff,
        top_k,
    })
}

fn build_preflight_plan(
    gguf: &dyn GgufSource,
    num_layers_hint: usize,
    num_experts_hint: usize,
    opts: &ExtractOptions,
) -> io::Result<ConversionPreflightPlan> {
    let geometry = resolve_geometry(gguf, num_layers_hint, num_experts_hint)?;
    let (expert_size, maximum_payload_bytes) = if opts.native_quant {
        let scan = scan_expert_quant_layouts(
            gguf,
            geometry.num_layers,
            geometry.num_experts,
            geometry.d_model,
            geometry.d_ff,
        )?;
        if !scan.rejection_reasons.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "native quantization cannot continue: {}",
                    scan.rejection_reasons.join("; ")
                ),
            ));
        }
        match scan.into_plan(DEFAULT_BLOCK_ALIGN, opts.emit_uth)? {
            NativeQuantPlan::Homogeneous { dtype } => {
                let payload = align_up(
                    expert_weight_bytes_for(geometry.d_model, geometry.d_ff, dtype),
                    DEFAULT_BLOCK_ALIGN,
                );
                (
                    payload
                        + if opts.emit_uth {
                            DEFAULT_BLOCK_ALIGN
                        } else {
                            0
                        },
                    payload,
                )
            }
            NativeQuantPlan::Mixed {
                expert_size,
                payload_slot_size,
                ..
            } => (expert_size, payload_slot_size),
        }
    } else {
        preflight_f32_expert_tensors(gguf, geometry)?;
        let payload = align_up(
            expert_weight_bytes_for(geometry.d_model, geometry.d_ff, WeightDtype::F32),
            DEFAULT_BLOCK_ALIGN,
        );
        (
            payload
                + if opts.emit_uth {
                    DEFAULT_BLOCK_ALIGN
                } else {
                    0
                },
            payload,
        )
    };

    Ok(ConversionPreflightPlan {
        architecture: gguf.architecture().unwrap_or("unknown").to_string(),
        num_layers: geometry.num_layers,
        num_experts: geometry.num_experts,
        total_experts: geometry.total_experts,
        d_model: geometry.d_model,
        d_ff: geometry.d_ff,
        top_k: geometry.top_k,
        expert_size,
        maximum_payload_bytes,
    })
}

fn preflight_f32_expert_tensors(
    gguf: &dyn GgufSource,
    geometry: ConversionGeometry,
) -> io::Result<()> {
    let weights = geometry.d_model.checked_mul(geometry.d_ff).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "d_model * d_ff overflows usize")
    })?;
    for layer in 0..geometry.num_layers {
        let interleaved_gate = format!("blk.{layer}.ffn_gate_exps.weight");
        if gguf.has_tensor(&interleaved_gate) {
            for name in [
                interleaved_gate,
                format!("blk.{layer}.ffn_up_exps.weight"),
                format!("blk.{layer}.ffn_down_exps.weight"),
            ] {
                let info = gguf.tensor_info(&name).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotFound, format!("missing tensor {name}"))
                })?;
                if crate::gguf::ggml_to_weight_dtype(info.ggml_dtype).is_none() {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!(
                            "tensor {name} uses unsupported GGML dtype {}",
                            info.ggml_dtype
                        ),
                    ));
                }
                let expected = weights.checked_mul(geometry.num_experts).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "expert tensor shape overflow")
                })?;
                if info.elem_count() != expected as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "tensor {name} has {} elements, expected {expected}",
                            info.elem_count()
                        ),
                    ));
                }
            }
            continue;
        }

        if !gguf.has_tensor(&format!("blk.{layer}.ffn_gate.0.weight")) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("layer {layer} has no expert tensors"),
            ));
        }
        for expert in 0..geometry.num_experts {
            for name in [
                format!("blk.{layer}.ffn_gate.{expert}.weight"),
                format!("blk.{layer}.ffn_up.{expert}.weight"),
                format!("blk.{layer}.ffn_down.{expert}.weight"),
            ] {
                let info = gguf.tensor_info(&name).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotFound, format!("missing tensor {name}"))
                })?;
                if crate::gguf::ggml_to_weight_dtype(info.ggml_dtype).is_none() {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!(
                            "tensor {name} uses unsupported GGML dtype {}",
                            info.ggml_dtype
                        ),
                    ));
                }
                if info.elem_count() != weights as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "tensor {name} has {} elements, expected {weights}",
                            info.elem_count()
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn ensure_final_output_available(out_dir: &Path) -> io::Result<()> {
    match fs::metadata(out_dir) {
        Ok(meta) if meta.is_dir() => {
            if fs::read_dir(out_dir)?.next().is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "output directory {} already exists and is not empty; remove it before conversion",
                        out_dir.display()
                    ),
                ));
            }
            fs::remove_dir(out_dir)?;
        }
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "output path {} exists and is not a directory",
                    out_dir.display()
                ),
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

fn staging_dir_for(out_dir: &Path) -> io::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = out_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = out_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("gguf-output");
    for _ in 0..1024 {
        let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "could not allocate a staging directory for {} after 1024 attempts",
            out_dir.display()
        ),
    ))
}

/// Extract Mixtral expert + dense weights from `gguf` into `out_dir`.
///
/// If `num_layers` / `num_experts_per_layer` are 0, they're auto-detected
/// from the GGUF metadata (`llama.block_count`, `llama.expert_count`).
///
/// This is the legacy eager entry point — see
/// [`extract_experts_from_source`] for the streaming-friendly variant
/// that also lets the caller opt-in to the Unified Tensor Header.
pub fn extract_experts_from_gguf(
    gguf: &GgufFile,
    out_dir: &Path,
    num_layers_hint: usize,
    num_experts_hint: usize,
) -> io::Result<ExtractionReport> {
    extract_experts_from_source(
        gguf,
        out_dir,
        num_layers_hint,
        num_experts_hint,
        ExtractOptions::default(),
    )
}

/// Per-call knobs for [`extract_experts_from_source`].
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    /// If true, prepend the 64-byte Unified Tensor Header to every
    /// expert file. The header is page-padded so the payload still
    /// starts at the configured `block_align` (4 KiB), preserving the
    /// engine's `O_DIRECT` invariants and per-expert `expert_size`
    /// contract. Default `true`.
    pub emit_uth: bool,

    /// **Native 4-bit pass-through (Industrial Upgrade Task 0).**
    ///
    /// When `true` *and* every expert tensor in a layer is `Q4_0` or
    /// `Q4_K` *and* the per-expert weight count divides cleanly on
    /// the corresponding block boundary (32 weights for `Q4_0`, 256
    /// for `Q4_K`), the converter writes the **raw quantised block
    /// stream** to disk instead of dequantising to F32. The output
    /// `expert_<id>.bin` then contains
    /// `gate_blocks || up_blocks || down_blocks` exactly as
    /// [`crate::inference::OwnedExpertWeights::from_bytes_q4_0`] /
    /// `from_bytes_q4k` expect, page-padded to `block_align` so the
    /// `O_DIRECT` reader stays compatible.
    ///
    /// This shrinks per-expert on-disk footprint by ~7× vs the F32
    /// dequant path (a 4096 × 14336 SwiGLU triple goes from
    /// ~672 MiB f32 down to ~96 MiB Q4_0 — half a GiB less per
    /// expert read off the SSD).
    ///
    /// Layers / models that don't satisfy the native quant preconditions
    /// fail during preflight before any final output is published.
    pub native_quant: bool,

    /// Convert only routed experts. Dense transformer tensors are
    /// skipped deliberately, metadata records `experts_only`, and the
    /// resulting dataset remains valid for expert-streaming runs.
    pub experts_only: bool,

    /// **Explicit safe-architecture override (Finding 6).**
    ///
    /// Dense conversion fails closed on an unknown or missing
    /// `general.architecture`. When the operator knows a checkpoint is a
    /// plain separate-QKV llama-like export whose GGUF metadata omits (or
    /// mislabels) the architecture, they may supply the canonical
    /// architecture string here to resolve the conversion profile instead
    /// of the metadata value. It never downgrades a recognised fused-QKV or
    /// MLA architecture to separate-QKV.
    pub arch_override: Option<String>,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            emit_uth: true,
            native_quant: false,
            experts_only: false,
            arch_override: None,
        }
    }
}

/// Streaming-friendly variant of [`extract_experts_from_gguf`].
///
/// Accepts anything that implements [`GgufSource`] so callers can pass
/// either the eager [`GgufFile`] (which slurps the file into memory at
/// open) or [`crate::gguf::GgufStreamReader`] (which keeps only the
/// header resident and seeks tensor bodies on demand). For checkpoints
/// ≥ ~10 GB the streaming reader is the right default.
pub fn extract_experts_from_source(
    gguf: &dyn GgufSource,
    out_dir: &Path,
    num_layers_hint: usize,
    num_experts_hint: usize,
    opts: ExtractOptions,
) -> io::Result<ExtractionReport> {
    let plan = build_preflight_plan(gguf, num_layers_hint, num_experts_hint, &opts)?;
    info!(
        architecture = %plan.architecture,
        num_layers = plan.num_layers,
        experts_per_layer = plan.num_experts,
        total_experts = plan.total_experts,
        d_model = plan.d_model,
        d_ff = plan.d_ff,
        top_k = plan.top_k,
        expert_size = plan.expert_size,
        maximum_payload_bytes = plan.maximum_payload_bytes,
        native_quant = opts.native_quant,
        experts_only = opts.experts_only,
        "gguf-convert preflight complete"
    );

    ensure_final_output_available(out_dir)?;
    let staging_dir = staging_dir_for(out_dir)?;
    let result = (|| {
        let report = extract_experts_from_source_inner(
            gguf,
            &staging_dir,
            num_layers_hint,
            num_experts_hint,
            opts,
        )?;
        validate_data_dir(&staging_dir)?;
        fs::rename(&staging_dir, out_dir)?;
        Ok(report)
    })();

    if result.is_err() {
        let _ = fs::remove_dir_all(&staging_dir);
    }
    result
}

fn extract_experts_from_source_inner(
    gguf: &dyn GgufSource,
    out_dir: &Path,
    num_layers_hint: usize,
    num_experts_hint: usize,
    opts: ExtractOptions,
) -> io::Result<ExtractionReport> {
    fs::create_dir_all(out_dir)?;

    // Architecture-agnostic metadata resolution. GGUF namespaces hyper-
    // parameter keys under the model's architecture (e.g. `llama.block_count`,
    // `qwen2moe.block_count`, `deepseek2.block_count`). Resolve the active
    // architecture from `general.architecture` and probe, in order:
    //   1. `llama.<suffix>`            (mainline llama.cpp convention)
    //   2. `<general.architecture>.<suffix>` (the file's declared arch)
    //   3. any metadata key ending in `.<suffix>` (last-resort agnostic scan)
    // so conversions succeed regardless of which architecture produced the file.
    let meta = gguf.metadata();
    let arch = meta.get("general.architecture").and_then(|v| v.as_str());
    let lookup = |suffix: &str| -> Option<&GgufValue> {
        let dotted = format!(".{suffix}");
        meta.get(&format!("llama.{suffix}"))
            .or_else(|| arch.and_then(|a| meta.get(&format!("{a}.{suffix}"))))
            .or_else(|| {
                // `meta` is a `HashMap`, so iteration order is
                // non-deterministic. When several metadata keys end in
                // `.<suffix>` (e.g. multiple architecture namespaces),
                // pick the lexicographically smallest key so the chosen
                // hyperparameter value is stable across runs.
                meta.iter()
                    .filter(|(k, _)| k.ends_with(&dotted))
                    .min_by(|(a, _), (b, _)| a.cmp(b))
                    .map(|(_, v)| v)
            })
    };

    let num_layers = if num_layers_hint > 0 {
        num_layers_hint
    } else {
        lookup("block_count")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF missing `block_count`; pass --num-layers explicitly",
                )
            })? as usize
    };
    let num_experts = if num_experts_hint > 0 {
        num_experts_hint
    } else {
        lookup("expert_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize
    };
    if num_experts == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF has no `expert_count` and no --num-experts override",
        ));
    }

    // Required shape parameters.
    let d_model = lookup("embedding_length")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "GGUF missing embedding_length")
        })? as usize;
    // For Mixture-of-Experts files the *routed* experts use a dedicated
    // hidden dimension that is typically much smaller than the dense-layer
    // `feed_forward_length`. Architectures like Qwen2-MoE expose it under
    // `expert_feed_forward_length` (namespaced, e.g. `qwen2moe.<suffix>`).
    // Prefer that key when present so the `num_experts * d_ff * d_model`
    // element math matches the actual expert tensor byte count; fall back
    // to the standard `feed_forward_length` so non-MoE-specific files
    // (e.g. Mixtral, which sizes its experts with `feed_forward_length`)
    // keep converting unchanged.
    let d_ff = lookup("expert_feed_forward_length")
        .or_else(|| lookup("feed_forward_length"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF missing feed_forward_length",
            )
        })? as usize;
    let top_k = lookup("expert_used_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;

    info!(
        num_layers,
        num_experts,
        d_model,
        d_ff,
        top_k,
        emit_uth = opts.emit_uth,
        native_quant = opts.native_quant,
        "gguf-convert: extracting experts"
    );

    let native_plan = if opts.native_quant {
        let scan = scan_expert_quant_layouts(gguf, num_layers, num_experts, d_model, d_ff)?;
        info!("\n{}", scan.concise_report());
        if !scan.rejection_reasons.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "--native-quant requested but quantized expert output is not possible: {}",
                    scan.rejection_reasons.join("; ")
                ),
            ));
        }
        let plan = scan.into_plan(DEFAULT_BLOCK_ALIGN, opts.emit_uth)?;
        match &plan {
            NativeQuantPlan::Homogeneous { dtype, .. } => {
                info!(
                    dtype = dtype.as_str(),
                    "native quant pass-through is eligible"
                );
            }
            NativeQuantPlan::Mixed { expert_size, .. } => {
                info!(
                    expert_size,
                    "native mixed-projection quant pass-through is eligible"
                );
            }
        }
        Some(plan)
    } else {
        None
    };
    let expert_dtype = native_plan
        .as_ref()
        .map(NativeQuantPlan::metadata_dtype)
        .unwrap_or(WeightDtype::F32);

    let mut report = ExtractionReport {
        experts_written: 0,
        dense_written: 0,
        skipped: 0,
        total_bytes: 0,
        expert_dtype,
        d_model,
        d_ff,
        num_experts_per_layer: num_experts,
        num_layers,
    };
    let mut dense_manifest = DenseTensorManifest {
        format_version: 1,
        tensors: Vec::new(),
    };

    // Walk layers and emit per-expert files. Output dtype is `F32`
    // for the dequant path (legacy default), or whichever 4-bit dtype
    // was detected as eligible above.
    //
    // When `opts.emit_uth` is set, each expert file is prefixed with a
    // 64-byte U.T.H., page-padded to DEFAULT_BLOCK_ALIGN so the weight
    // payload still starts at a 4 KiB boundary. The total file size
    // therefore grows by exactly one block (4 KiB) per expert.
    let payload_size = match native_plan.as_ref() {
        Some(NativeQuantPlan::Mixed {
            payload_slot_size, ..
        }) => *payload_slot_size,
        _ => align_up(
            expert_weight_bytes_for(d_model, d_ff, expert_dtype),
            DEFAULT_BLOCK_ALIGN,
        ),
    };
    let header_overhead = if opts.emit_uth {
        DEFAULT_BLOCK_ALIGN
    } else {
        0
    };
    // Defensive: an adversarially-large `llama.embedding_length` /
    // `llama.feed_forward_length` pair could push `payload_size` close
    // to `usize::MAX` and wrap to a small `expert_size` here, which
    // would later cause out-of-bounds writes into the per-expert byte
    // buffer. Surface the overflow as a clean `InvalidData` instead.
    let expert_size = payload_size.checked_add(header_overhead).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expert size overflows usize: payload_size={payload_size}, \
                 header_overhead={header_overhead}"
            ),
        )
    })?;
    // Same hardening for the global id arithmetic below: a corrupt
    // metadata block claiming `num_experts * num_layers > usize::MAX`
    // would wrap into a small id and start overwriting expert files
    // from earlier layers. Refuse to start in that case.
    let total_experts = num_experts.checked_mul(num_layers).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "num_experts * num_layers overflows usize: \
                 num_experts={num_experts}, num_layers={num_layers}"
            ),
        )
    })?;

    for layer in 0..num_layers {
        info!(layer, "extracting layer experts");
        match native_plan.as_ref() {
            Some(NativeQuantPlan::Homogeneous { dtype: d, .. }) => {
                // Native pass-through: pull the raw quantised bytes
                // for this layer's gate/up/down tensors and slice each
                // expert's stride out without dequantising.
                let per_expert =
                    load_layer_expert_native_quant(gguf, layer, num_experts, d_model, d_ff, *d)?;
                for (e, (gate, up, down)) in per_expert.into_iter().enumerate() {
                    // Safe: `layer < num_layers` and `e < num_experts`,
                    // so `layer * num_experts + e < total_experts`,
                    // which we proved above does not overflow.
                    let global_id = layer * num_experts + e;
                    let path = out_dir.join(format!("expert_{global_id}.bin"));
                    let mut bytes = Vec::with_capacity(expert_size);
                    if opts.emit_uth {
                        TensorHeader::for_swiglu_expert(*d, d_model, d_ff)
                            .write_padded(DEFAULT_BLOCK_ALIGN, &mut bytes);
                        debug_assert_eq!(bytes.len(), DEFAULT_BLOCK_ALIGN);
                    }
                    bytes.extend_from_slice(&gate);
                    bytes.extend_from_slice(&up);
                    bytes.extend_from_slice(&down);
                    if bytes.len() < expert_size {
                        bytes.resize(expert_size, 0);
                    }
                    write_file(&path, &bytes)?;
                    report.experts_written += 1;
                    report.total_bytes += bytes.len() as u64;
                }
            }
            Some(NativeQuantPlan::Mixed { layouts, .. }) => {
                let per_expert = load_layer_expert_native_mixed(
                    gguf,
                    layer,
                    num_experts,
                    d_model,
                    d_ff,
                    layouts,
                )?;
                for (e, (layout, gate, up, down)) in per_expert.into_iter().enumerate() {
                    let global_id = layer * num_experts + e;
                    let path = out_dir.join(format!("expert_{global_id}.bin"));
                    let mut bytes = Vec::with_capacity(expert_size);
                    if opts.emit_uth {
                        let gate_range = ProjectionRange {
                            dtype: UthDtypeId::from_weight(layout.gate.dtype),
                            offset: 0,
                            len: gate.len() as u64,
                            weights: layout.gate.weights as u32,
                        };
                        let up_range = ProjectionRange {
                            dtype: UthDtypeId::from_weight(layout.up.dtype),
                            offset: gate.len() as u64,
                            len: up.len() as u64,
                            weights: layout.up.weights as u32,
                        };
                        let down_offset = gate.len().checked_add(up.len()).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "mixed expert payload offset overflow",
                            )
                        })?;
                        let down_range = ProjectionRange {
                            dtype: UthDtypeId::from_weight(layout.down.dtype),
                            offset: down_offset as u64,
                            len: down.len() as u64,
                            weights: layout.down.weights as u32,
                        };
                        MixedExpertHeader::new(d_model, d_ff, gate_range, up_range, down_range)
                            .write_padded(DEFAULT_BLOCK_ALIGN, &mut bytes);
                        debug_assert_eq!(bytes.len(), DEFAULT_BLOCK_ALIGN);
                    }
                    bytes.extend_from_slice(&gate);
                    bytes.extend_from_slice(&up);
                    bytes.extend_from_slice(&down);
                    if bytes.len() < expert_size {
                        bytes.resize(expert_size, 0);
                    }
                    write_file(&path, &bytes)?;
                    report.experts_written += 1;
                    report.total_bytes += bytes.len() as u64;
                }
            }
            None => {
                let per_expert =
                    load_layer_expert_matrices(gguf, layer, num_experts, d_model, d_ff)?;
                for (e, (gate, up, down)) in per_expert.into_iter().enumerate() {
                    // Safe by the same `total_experts` check above.
                    let global_id = layer * num_experts + e;
                    let path = out_dir.join(format!("expert_{global_id}.bin"));
                    let mut bytes = Vec::with_capacity(expert_size);
                    if opts.emit_uth {
                        TensorHeader::for_swiglu_expert(WeightDtype::F32, d_model, d_ff)
                            .write_padded(DEFAULT_BLOCK_ALIGN, &mut bytes);
                        debug_assert_eq!(bytes.len(), DEFAULT_BLOCK_ALIGN);
                    }
                    append_expert_f32(&mut bytes, &gate, &up, &down, expert_size);
                    write_file(&path, &bytes)?;
                    report.experts_written += 1;
                    report.total_bytes += bytes.len() as u64;
                }
            }
        }
    }

    if opts.experts_only {
        info!(
            num_layers,
            "experts-only conversion selected; skipping dense tensor extraction"
        );
    } else {
        // Dense weights. Core resident tensors are written once under a
        // canonical filename and exposed to legacy engine names through
        // `dense_manifest.json` aliases. Q8_0 tensors keep their native
        // bytes; other supported GGUF dense dtypes are materialised as f32.
        //
        // Finding 2/6: the required-tensor inventory is architecture-aware and
        // the architecture is resolved from an explicit allowlist. An unknown
        // or missing `general.architecture` fails closed unless the operator
        // supplies a safe separate-QKV override. Recognised fused-QKV and MLA
        // families use an attention tensor set the expert extractor cannot
        // satisfy, so conversion fails explicitly with an architecture-specific
        // error rather than emitting a partial checkpoint.
        let effective_arch = opts.arch_override.as_deref().or(arch);
        let profile = resolve_conversion_profile(effective_arch);
        if !profile.recognised {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "general.architecture {:?} is not a recognised convertible architecture; \
                     supply an explicit safe architecture override to convert a separate-QKV \
                     llama-like checkpoint",
                    arch.unwrap_or("<missing>")
                ),
            ));
        }
        match profile.attention {
            ConvAttention::SeparateQkv => {}
            ConvAttention::FusedQkv => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "architecture {:?} uses fused QKV attention, which the expert extractor \
                         does not support converting; supply a separate-QKV export",
                        effective_arch.unwrap_or("<unknown>")
                    ),
                ));
            }
            ConvAttention::Mla => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "architecture {:?} uses MLA (latent-KV) attention, which the expert \
                         extractor does not support converting",
                        effective_arch.unwrap_or("<unknown>")
                    ),
                ));
            }
        }

        // Per-head QK-Norm length for architectures that require it: prefer the
        // explicit key length, falling back to `d_model / head_count`.
        let head_dim = if profile.uses_qk_norm {
            let hd = lookup("attention.key_length")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .or_else(|| {
                    lookup("attention.head_count")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .filter(|&h| h != 0)
                        .map(|h| d_model / h)
                })
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "QK-Norm architecture is missing attention.key_length / \
                         attention.head_count needed to validate head_dim",
                    )
                })?;
            Some(hd)
        } else {
            None
        };

        // Token embedding and final norm are always architecture-required.
        emit_dense_manifest_tensor_req(
            gguf,
            out_dir,
            &mut report,
            &mut dense_manifest,
            "token_embd.weight",
            vec!["embedding.bin".to_string(), "embed.bin".to_string()],
            true,
        )?;
        emit_dense_manifest_tensor_req(
            gguf,
            out_dir,
            &mut report,
            &mut dense_manifest,
            "output_norm.weight",
            vec!["final_norm.bin".to_string(), "final_rms.bin".to_string()],
            true,
        )?;

        // Output head. `output.weight` may be omitted when embedding tying is
        // established either (a) from an explicit `tie_word_embeddings`
        // metadata flag, or (b) by the architecture's upstream GGUF contract
        // for verified families (e.g. Qwen3-MoE), which llama.cpp exports with
        // no `output.weight` and a duplicated token embedding (Finding 5). The
        // loader then reconstructs the LM head from `token_embd.weight` via
        // `tied_to`. When `output.weight` is present it is always loaded
        // independently. Absent *and* untied (unknown/untied architecture with
        // no tie flag) is fatal — tying is never inferred from absence alone.
        let tie_word_embeddings = matches!(
            lookup("tie_word_embeddings").or_else(|| meta.get("general.tie_word_embeddings")),
            Some(GgufValue::Bool(true))
        );
        if gguf.tensor_info("output.weight").is_some() {
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                "output.weight",
                vec!["lm_head.bin".to_string()],
                true,
            )?;
        } else if tie_word_embeddings || profile.tied_output_by_contract {
            let embed_dims = gguf
                .tensor_info("token_embd.weight")
                .and_then(dense_manifest_dims)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "tied output head requires token_embd.weight dims to reconstruct the LM head",
                    )
                })?;
            push_tied_output_entry(&mut dense_manifest, embed_dims, "token_embd.weight");
            let basis = if tie_word_embeddings {
                "tie_word_embeddings metadata"
            } else {
                "architecture GGUF contract"
            };
            info!(
                basis,
                "output.weight tied to token_embd.weight (embedding tying established)"
            );
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "output.weight is absent and embedding tying is not established by metadata or \
                 architecture contract; refusing partial conversion",
            ));
        }

        // Per-layer dense tensors.
        for layer in 0..num_layers {
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.attn_q.weight"),
                vec![
                    format!("layer_{layer}_q.bin"),
                    format!("attn_{layer}_q.bin"),
                ],
                true,
            )?;
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.attn_k.weight"),
                vec![
                    format!("layer_{layer}_k.bin"),
                    format!("attn_{layer}_k.bin"),
                ],
                true,
            )?;
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.attn_v.weight"),
                vec![
                    format!("layer_{layer}_v.bin"),
                    format!("attn_{layer}_v.bin"),
                ],
                true,
            )?;
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.attn_output.weight"),
                vec![
                    format!("layer_{layer}_o.bin"),
                    format!("attn_{layer}_o.bin"),
                ],
                true,
            )?;
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.attn_norm.weight"),
                vec![
                    format!("layer_{layer}_attn_norm.bin"),
                    format!("rms_attn_{layer}.bin"),
                ],
                true,
            )?;
            emit_dense_manifest_tensor_req(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.ffn_norm.weight"),
                vec![
                    format!("layer_{layer}_ffn_norm.bin"),
                    format!("rms_moe_{layer}.bin"),
                ],
                true,
            )?;
            // Qwen3 / Qwen3-MoE per-head QK-Norm: learned `head_dim` RMSNorm
            // weights applied to Q and K before RoPE. Emitted under the
            // engine-friendly `q_norm_{layer}.bin` / `k_norm_{layer}.bin`
            // aliases consumed by the converted-directory loader. For
            // architectures where `uses_qk_norm()` holds these are
            // architecture-required: both norms must be present and contain
            // exactly `head_dim` elements, and a missing one fails conversion.
            // Architectures without QK-Norm (e.g. Mixtral) keep them optional
            // and unprobed.
            let q_norm_name = format!("blk.{layer}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{layer}.attn_k_norm.weight");
            if let Some(hd) = head_dim {
                emit_required_dense_tensor_checked(
                    gguf,
                    out_dir,
                    &mut report,
                    &mut dense_manifest,
                    &q_norm_name,
                    vec![format!("q_norm_{layer}.bin")],
                    None,
                    Some(hd),
                )?;
                emit_required_dense_tensor_checked(
                    gguf,
                    out_dir,
                    &mut report,
                    &mut dense_manifest,
                    &k_norm_name,
                    vec![format!("k_norm_{layer}.bin")],
                    None,
                    Some(hd),
                )?;
            }
            // Routed MoE gate. Every layer in this expert extractor carries
            // `num_experts` routed experts, so the router gate is
            // architecture-required and must be logically `[num_experts, d_model]`
            // (Finding 2/3). A transposed or wrong-shaped gate fails conversion.
            emit_required_dense_tensor_checked(
                gguf,
                out_dir,
                &mut report,
                &mut dense_manifest,
                &format!("blk.{layer}.ffn_gate_inp.weight"),
                vec![format!("gate_{layer}.bin")],
                Some([num_experts, d_model]),
                None,
            )?;
            // Qwen2-MoE-style "shared experts" — dense FFN tensors applied to
            // *every* token in addition to the routed experts. They are stored
            // under the `_shexp` suffix and were previously dropped, leaving the
            // converted engine missing weights. Emit them as dense `.bin` files
            // (both the gguf-style name and an engine-friendly alias). Files are
            // only written when the tensor exists, so non-MoE / no-shared-expert
            // architectures (e.g. Mixtral) are unaffected.
            emit_legacy_f32_dense_tensor(
                gguf,
                out_dir,
                &mut report,
                &format!("blk.{layer}.ffn_gate_shexp.weight"),
                &[format!("layer_{layer}_shexp_gate.bin")],
            )?;
            emit_legacy_f32_dense_tensor(
                gguf,
                out_dir,
                &mut report,
                &format!("blk.{layer}.ffn_up_shexp.weight"),
                &[format!("layer_{layer}_shexp_up.bin")],
            )?;
            emit_legacy_f32_dense_tensor(
                gguf,
                out_dir,
                &mut report,
                &format!("blk.{layer}.ffn_down_shexp.weight"),
                &[format!("layer_{layer}_shexp_down.bin")],
            )?;
            emit_legacy_f32_dense_tensor(
                gguf,
                out_dir,
                &mut report,
                &format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
                &[format!("layer_{layer}_shexp_gate_inp.bin")],
            )?;
        }
        let dense_manifest_path = out_dir.join("dense_manifest.json");
        let mut f = fs::File::create(&dense_manifest_path)?;
        serde_json::to_writer_pretty(&mut f, &dense_manifest)?;
        f.write_all(b"\n")?;
        info!(
            path = %dense_manifest_path.display(),
            tensors = dense_manifest.tensors.len(),
            "wrote dense manifest"
        );
    }

    // metadata.json
    let model_name = gguf
        .metadata()
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("gguf-extracted");
    let meta = ExtractedMetadata {
        format_version: 2,
        conversion_mode: if opts.experts_only {
            "experts_only"
        } else {
            "full"
        },
        model: model_name,
        architecture: arch.unwrap_or("unknown"),
        layer: if num_layers == 1 { 0 } else { -1 },
        num_experts: total_experts,
        top_k,
        d_model,
        d_ff,
        expert_size,
        maximum_payload_bytes: payload_size,
        block_align: DEFAULT_BLOCK_ALIGN,
        dtype: expert_dtype.as_str(),
        weight_layout: "gate_proj || up_proj || down_proj (row-major)",
        num_layers,
        num_experts_per_layer: num_experts,
        expert_layout_version: native_plan
            .as_ref()
            .and_then(NativeQuantPlan::metadata_layout_version),
        projection_dtype_histogram: native_plan
            .as_ref()
            .and_then(NativeQuantPlan::metadata_histogram),
        experts_written: report.experts_written,
        dense_tensors_written: report.dense_written,
    };
    let meta_path = out_dir.join("metadata.json");
    let mut f = fs::File::create(&meta_path)?;
    serde_json::to_writer_pretty(&mut f, &meta)?;
    f.write_all(b"\n")?;
    info!(path = %meta_path.display(), "wrote metadata.json");
    Ok(report)
}

fn align_up(n: usize, align: usize) -> usize {
    let rem = n % align;
    if rem == 0 {
        n
    } else {
        n + (align - rem)
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)
}

fn dense_manifest_dims(info: &GgufTensorInfo) -> Option<Vec<usize>> {
    match info.shape.as_slice() {
        [n] => Some(vec![*n as usize, 1]),
        [cols, rows] => Some(vec![*rows as usize, *cols as usize]),
        _ => None,
    }
}

fn dense_manifest_file_name(canonical: &str, dtype: DenseDType) -> String {
    let mut stem = String::with_capacity(canonical.len());
    for ch in canonical.chars() {
        if ch.is_ascii_alphanumeric() {
            stem.push(ch);
        } else {
            stem.push('_');
        }
    }
    format!("dense_{stem}.{}.bin", dtype.as_str())
}

/// The attention-tensor layout a source architecture exports, used to build
/// an architecture-correct required-tensor inventory (Finding 2). llama.cpp
/// GGUF exports split attention into separate `attn_q/k/v` for most families,
/// but fused-QKV (Phi-style) and MLA (DeepSeek-V2/V3 latent-KV) families use a
/// fundamentally different tensor set that the expert extractor cannot satisfy
/// by demanding nonexistent separate Q/K/V tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvAttention {
    /// Separate `attn_q`, `attn_k`, `attn_v`, `attn_output` projections.
    SeparateQkv,
    /// Fused `attn_qkv` projection plus `attn_output`.
    FusedQkv,
    /// Multi-head Latent Attention (DeepSeek): compressed KV latent set.
    Mla,
}

/// Conversion-time architecture profile resolved from `general.architecture`.
#[derive(Debug, Clone, Copy)]
struct ConvArchProfile {
    attention: ConvAttention,
    /// Per-head QK-Norm (`attn_q_norm`/`attn_k_norm`) is architecture-required.
    uses_qk_norm: bool,
    /// Whether the architecture is on the explicit conversion allowlist. An
    /// unknown or missing architecture is *not* recognised and must fail
    /// closed rather than being silently treated as generic separate-QKV
    /// (Finding 6).
    recognised: bool,
    /// The architecture's upstream GGUF contract permits `output.weight` to be
    /// absent and duplicate `token_embd.weight` (embedding tying) even when no
    /// explicit `tie_word_embeddings` metadata flag is present (Finding 5).
    tied_output_by_contract: bool,
}

/// Resolve the conversion profile for a GGUF `general.architecture` string.
///
/// Finding 6: this is an explicit allowlist. Recognised separate-QKV families
/// convert normally; recognised fused-QKV / MLA families are surfaced with an
/// architecture-specific unsupported error; and an unknown or missing
/// architecture is reported as unrecognised so conversion fails closed instead
/// of emitting a partial checkpoint from an unverified tensor layout.
fn resolve_conversion_profile(arch: Option<&str>) -> ConvArchProfile {
    match arch.unwrap_or("") {
        // DeepSeek-V2/V3 latent-KV attention (recognised, unsupported).
        "deepseek2" | "deepseek_v3" | "deepseek3" => ConvArchProfile {
            attention: ConvAttention::Mla,
            uses_qk_norm: false,
            recognised: true,
            tied_output_by_contract: false,
        },
        // Phi-3/Phi-4 fused `qkv_proj` (recognised, unsupported).
        "phi2" | "phi3" | "phi4" => ConvArchProfile {
            attention: ConvAttention::FusedQkv,
            uses_qk_norm: false,
            recognised: true,
            tied_output_by_contract: false,
        },
        // Qwen3 / Qwen3-MoE: separate QKV *with* per-head QK-Norm. Upstream
        // llama.cpp omits `output.weight` for these and duplicates the token
        // embedding, so tying is defined by the architecture contract.
        "qwen3" | "qwen3moe" | "qwen3_moe" => ConvArchProfile {
            attention: ConvAttention::SeparateQkv,
            uses_qk_norm: true,
            recognised: true,
            tied_output_by_contract: true,
        },
        // Recognised separate-QKV llama-like families without QK-Norm. These
        // always ship an explicit `output.weight` (no embedding tying).
        "llama" | "mixtral" | "qwen2" | "qwen2moe" | "qwen2_moe" | "gptoss" | "gpt_oss"
        | "gpt-oss" => ConvArchProfile {
            attention: ConvAttention::SeparateQkv,
            uses_qk_norm: false,
            recognised: true,
            tied_output_by_contract: false,
        },
        // Unknown or missing architecture: not on the allowlist (Finding 6).
        _ => ConvArchProfile {
            attention: ConvAttention::SeparateQkv,
            uses_qk_norm: false,
            recognised: false,
            tied_output_by_contract: false,
        },
    }
}

/// Emit an architecture-required dense tensor while additionally asserting its
/// *logical* shape. Element-count matching alone is insufficient for tensors
/// whose orientation or exact length carries meaning: a routed gate must be
/// `[num_experts, d_model]` (Finding 3) and a QK-Norm vector must contain
/// exactly `head_dim` elements. A mismatch is a hard conversion error.
#[allow(clippy::too_many_arguments)]
fn emit_required_dense_tensor_checked(
    gguf: &dyn GgufSource,
    out_dir: &Path,
    report: &mut ExtractionReport,
    manifest: &mut DenseTensorManifest,
    canonical: &str,
    aliases: Vec<String>,
    expect_dims: Option<[usize; 2]>,
    expect_elems: Option<usize>,
) -> io::Result<()> {
    let info = gguf.tensor_info(canonical).cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("architecture-required dense tensor {canonical} is absent from the GGUF source"),
        )
    })?;
    let dims = dense_manifest_dims(&info).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "architecture-required dense tensor {canonical} has unsupported rank {:?}",
                info.shape
            ),
        )
    })?;
    if let Some(expect) = expect_dims {
        if dims.as_slice() != expect {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "architecture-required dense tensor {canonical} has logical dims {dims:?}, \
                     expected {expect:?}"
                ),
            ));
        }
    }
    if let Some(n) = expect_elems {
        let elems: usize = dims.iter().product();
        if elems != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "architecture-required dense tensor {canonical} has {elems} elements, \
                     expected {n}"
                ),
            ));
        }
    }
    emit_dense_manifest_tensor_req(gguf, out_dir, report, manifest, canonical, aliases, true)
}

/// Record a tied LM-head manifest entry that constructs `output.weight` from
/// the token embedding at load time (Finding 2). No weight file is written;
/// the loader resolves `tied_to` to the embedding tensor. `dims` are the
/// embedding's logical `[vocab, d_model]` so the strict loader can validate the
/// reconstructed head against the configured shape.
fn push_tied_output_entry(
    manifest: &mut DenseTensorManifest,
    dims: Vec<usize>,
    tied_to: &str,
) {
    manifest.tensors.push(DenseTensorManifestEntry {
        canonical_name: "output.weight".to_string(),
        file: String::new(),
        aliases: vec!["lm_head.bin".to_string()],
        dtype: DenseDType::F32,
        dims,
        byte_len: 0,
        checksum: None,
        tied_to: Some(tied_to.to_string()),
    });
}

fn emit_dense_manifest_tensor(
    gguf: &dyn GgufSource,
    out_dir: &Path,
    report: &mut ExtractionReport,
    manifest: &mut DenseTensorManifest,
    canonical: &str,
    aliases: Vec<String>,
) -> io::Result<()> {
    emit_dense_manifest_tensor_req(gguf, out_dir, report, manifest, canonical, aliases, false)
}

/// Emit a dense tensor into the converted directory and record its manifest
/// entry.
///
/// Finding 2: when `required` is set, a successful conversion must guarantee
/// the tensor was actually emitted. A missing tensor, an unsupported rank, or
/// an unsupported/failed quantization decode is therefore promoted to a hard
/// error instead of being silently counted as `skipped`. Optional tensors
/// retain the lenient skip-on-absence behaviour.
fn emit_dense_manifest_tensor_req(
    gguf: &dyn GgufSource,
    out_dir: &Path,
    report: &mut ExtractionReport,
    manifest: &mut DenseTensorManifest,
    canonical: &str,
    aliases: Vec<String>,
    required: bool,
) -> io::Result<()> {
    let Some(info) = gguf.tensor_info(canonical).cloned() else {
        if required {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "architecture-required dense tensor {canonical} is absent from the GGUF source"
                ),
            ));
        }
        report.skipped += 1;
        return Ok(());
    };
    let Some(dims) = dense_manifest_dims(&info) else {
        if required {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "architecture-required dense tensor {canonical} has unsupported rank {:?}",
                    info.shape
                ),
            ));
        }
        warn!(
            name = canonical,
            shape = ?info.shape,
            "skipping dense tensor with unsupported rank"
        );
        report.skipped += 1;
        return Ok(());
    };
    let (dtype, bytes) = if info.ggml_dtype == ggml_dtype::Q8_0 {
        let bytes = gguf.read_tensor_owned(&info.name)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {} has no data slice", info.name),
            )
        })?;
        (DenseDType::Q8_0, bytes)
    } else {
        match dense_tensor_to_f32(gguf, &info) {
            Ok(f32s) => (DenseDType::F32, f32_vec_to_le_bytes(&f32s)),
            Err(err) => {
                if required {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!(
                            "architecture-required dense tensor {canonical} could not be decoded: {err}"
                        ),
                    ));
                }
                warn!(name = canonical, error = %err, "skipping dense tensor");
                report.skipped += 1;
                return Ok(());
            }
        }
    };
    let file = dense_manifest_file_name(canonical, dtype);
    let path = out_dir.join(&file);
    write_file(&path, &bytes)?;
    report.dense_written += 1;
    report.total_bytes += bytes.len() as u64;
    manifest.tensors.push(DenseTensorManifestEntry {
        canonical_name: canonical.to_string(),
        file,
        aliases,
        dtype,
        dims,
        byte_len: bytes.len(),
        checksum: Some(dense_checksum(&bytes)),
        tied_to: None,
    });
    Ok(())
}

fn emit_legacy_f32_dense_tensor(
    gguf: &dyn GgufSource,
    out_dir: &Path,
    report: &mut ExtractionReport,
    canonical: &str,
    aliases: &[String],
) -> io::Result<()> {
    if let Some(info) = gguf.tensor_info(canonical).cloned() {
        match dense_tensor_to_f32(gguf, &info) {
            Ok(f32s) => {
                let bytes = f32_vec_to_le_bytes(&f32s);
                for alias in aliases {
                    let path = out_dir.join(alias);
                    write_file(&path, &bytes)?;
                    report.dense_written += 1;
                    report.total_bytes += bytes.len() as u64;
                }
            }
            Err(err) => {
                warn!(name = canonical, error = %err, "skipping legacy dense tensor");
                report.skipped += 1;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct DataValidationReport {
    pub num_experts: usize,
    pub expert_size: usize,
    pub block_align: usize,
    pub dtype: WeightDtype,
    pub mixed_experts: usize,
}

#[derive(Debug, Deserialize)]
struct ValidationMetadata {
    num_experts: usize,
    d_model: usize,
    d_ff: usize,
    expert_size: usize,
    #[serde(default = "default_validation_block_align")]
    block_align: usize,
    dtype: String,
    #[serde(default)]
    expert_layout_version: Option<u32>,
    #[serde(default)]
    projection_dtype_histogram: Option<BTreeMap<String, usize>>,
    #[serde(default)]
    experts_written: Option<usize>,
}

fn default_validation_block_align() -> usize {
    DEFAULT_BLOCK_ALIGN
}

fn tensor_header_prefix_len(header_bytes: usize, flags: u32, block_align: usize) -> usize {
    if (flags & crate::tensor_header::UTH_FLAG_PAGE_ALIGNED_PAYLOAD) != 0 && block_align > 0 {
        align_up(header_bytes, block_align)
    } else {
        header_bytes
    }
}

fn validate_payload_len(
    path: &Path,
    dtype: WeightDtype,
    payload_len: usize,
    expected_payload: usize,
) -> io::Result<()> {
    if payload_len < expected_payload {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expert file {} payload has {} bytes, expected at least {} for dtype {}",
                path.display(),
                payload_len,
                expected_payload,
                dtype.as_str()
            ),
        ));
    }
    Ok(())
}

pub fn validate_data_dir(data_dir: &Path) -> io::Result<DataValidationReport> {
    let meta_path = data_dir.join("metadata.json");
    let body = fs::read_to_string(&meta_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("failed to read metadata {}: {e}", meta_path.display()),
        )
    })?;
    let meta: ValidationMetadata = serde_json::from_str(&body).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("metadata {} is invalid JSON: {e}", meta_path.display()),
        )
    })?;
    let dtype = WeightDtype::from_str_opt(&meta.dtype).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("metadata dtype {:?} is not supported", meta.dtype),
        )
    })?;
    if meta.block_align == 0 || !meta.block_align.is_power_of_two() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("metadata block_align {} is invalid", meta.block_align),
        ));
    }
    if meta.expert_size == 0 || meta.expert_size % meta.block_align != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "metadata expert_size {} is not aligned to block_align {}",
                meta.expert_size, meta.block_align
            ),
        ));
    }
    if let Some(written) = meta.experts_written {
        if written != meta.num_experts {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "metadata experts_written {written} does not match num_experts {}",
                    meta.num_experts
                ),
            ));
        }
    }
    if dtype == WeightDtype::Mixed && meta.expert_layout_version != Some(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metadata dtype=mixed requires expert_layout_version=2",
        ));
    }
    let expected_payload = if dtype == WeightDtype::Mixed {
        None
    } else {
        Some(align_up(
            expert_weight_bytes_for(meta.d_model, meta.d_ff, dtype),
            meta.block_align,
        ))
    };

    let mut observed_mixed = 0usize;
    let mut observed_hist = BTreeMap::<String, usize>::new();
    for id in 0..meta.num_experts {
        let path = data_dir.join(format!("expert_{id}.bin"));
        let stat = fs::metadata(&path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "expert file {} is missing or unreadable: {e}",
                    path.display()
                ),
            )
        })?;
        if stat.len() != meta.expert_size as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expert file {} has {} bytes, expected {}",
                    path.display(),
                    stat.len(),
                    meta.expert_size
                ),
            ));
        }
        let head_len = meta
            .expert_size
            .min(crate::tensor_header::UTH2_BYTES.max(crate::tensor_header::UTH_BYTES));
        let mut head = vec![0u8; head_len];
        let mut file = fs::File::open(&path)?;
        use std::io::Read;
        file.read_exact(&mut head)?;

        if dtype == WeightDtype::Mixed {
            let header = MixedExpertHeader::probe(&head).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "expert file {} is mixed but has no valid UTH2 header",
                        path.display()
                    ),
                )
            })?;
            if header.d_model as usize != meta.d_model || header.d_ff as usize != meta.d_ff {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "expert file {} UTH2 shape d_model={} d_ff={} disagrees with metadata d_model={} d_ff={}",
                        path.display(),
                        header.d_model,
                        header.d_ff,
                        meta.d_model,
                        meta.d_ff
                    ),
                ));
            }
            let prefix_len = tensor_header_prefix_len(
                crate::tensor_header::UTH2_BYTES,
                header.flags,
                meta.block_align,
            );
            let payload_len = meta.expert_size.checked_sub(prefix_len).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expert payload length underflow",
                )
            })?;
            header
                .validate(payload_len as u64)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let key = format!(
                "{}/{}/{}",
                dtype_label(header.gate.dtype.to_weight()).to_ascii_lowercase(),
                dtype_label(header.up.dtype.to_weight()).to_ascii_lowercase(),
                dtype_label(header.down.dtype.to_weight()).to_ascii_lowercase()
            );
            *observed_hist.entry(key).or_default() += 1;
            observed_mixed += 1;
        } else {
            let payload_len = if let Some(header) = TensorHeader::probe(&head) {
                let header_dtype = header.dtype.to_weight();
                if header_dtype != dtype {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "expert file {} UTH dtype {} disagrees with metadata dtype {}",
                            path.display(),
                            header_dtype.as_str(),
                            dtype.as_str()
                        ),
                    ));
                }
                let prefix_len = tensor_header_prefix_len(
                    crate::tensor_header::UTH_BYTES,
                    header.flags,
                    meta.block_align,
                );
                meta.expert_size.checked_sub(prefix_len).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "expert payload length underflow",
                    )
                })?
            } else {
                meta.expert_size
            };
            validate_payload_len(&path, dtype, payload_len, expected_payload.unwrap())?;
        }
    }

    if let Some(expected_hist) = meta.projection_dtype_histogram {
        if dtype == WeightDtype::Mixed && expected_hist != observed_hist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "mixed projection histogram mismatch: metadata {:?}, observed {:?}",
                    expected_hist, observed_hist
                ),
            ));
        }
    }

    Ok(DataValidationReport {
        num_experts: meta.num_experts,
        expert_size: meta.expert_size,
        block_align: meta.block_align,
        dtype,
        mixed_experts: observed_mixed,
    })
}

fn f32_vec_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Pack one expert's matrices into the engine's on-disk layout:
/// `gate || up || down` row-major f32, padded to `target_size`.
///
/// Used by the legacy non-UTH path. The streaming UTH path uses
/// [`append_expert_f32`] instead which writes into an existing buffer.
#[cfg(test)]
fn pack_expert_f32(gate: &[f32], up: &[f32], down: &[f32], target_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target_size);
    append_expert_f32(&mut out, gate, up, down, target_size);
    out
}

/// Append `gate || up || down` row-major f32 bytes to `out`, then pad
/// the *total* length of `out` to `target_size`. This is the
/// UTH-aware counterpart of `pack_expert_f32`: the caller has already
/// pushed the header + page padding into `out`, so we just continue
/// from wherever `out.len()` currently is.
fn append_expert_f32(
    out: &mut Vec<u8>,
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    target_size: usize,
) {
    for v in gate.iter().chain(up.iter()).chain(down.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    if out.len() < target_size {
        out.resize(target_size, 0);
    }
}

/// Convert any supported dense tensor to a flat row-major `Vec<f32>`.
fn dense_tensor_to_f32(gguf: &dyn GgufSource, info: &GgufTensorInfo) -> io::Result<Vec<f32>> {
    let data = gguf.read_tensor_owned(&info.name)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor {} has no data slice", info.name),
        )
    })?;
    bytes_to_f32(
        &data,
        info.ggml_dtype,
        info.elem_count() as usize,
        &info.name,
    )
}

fn bytes_to_f32(data: &[u8], dtype: u32, elems: usize, name: &str) -> io::Result<Vec<f32>> {
    match dtype {
        ggml_dtype::F32 => {
            let mut out = Vec::with_capacity(elems);
            if data.len() < elems * 4 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("tensor {name}: short F32 buffer"),
                ));
            }
            for c in data[..elems * 4].chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        ggml_dtype::F16 => {
            if data.len() < elems * 2 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("tensor {name}: short F16 buffer"),
                ));
            }
            let mut out = Vec::with_capacity(elems);
            dequantize_f16_to_f32(&data[..elems * 2], &mut out);
            Ok(out)
        }
        ggml_dtype::BF16 => {
            if data.len() < elems * 2 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("tensor {name}: short BF16 buffer"),
                ));
            }
            let mut out = Vec::with_capacity(elems);
            dequantize_bf16_to_f32(&data[..elems * 2], &mut out);
            Ok(out)
        }
        ggml_dtype::Q4_0 => {
            let mut out = Vec::with_capacity(elems);
            crate::inference::dequantize_q4_0_to_f32(data, elems, &mut out);
            Ok(out)
        }
        ggml_dtype::Q4_K => {
            let mut out = Vec::with_capacity(elems);
            crate::inference::dequantize_q4k_to_f32(data, elems, &mut out);
            Ok(out)
        }
        ggml_dtype::Q5_K => {
            let mut out = Vec::with_capacity(elems);
            crate::inference::dequantize_q5k_to_f32(data, elems, &mut out);
            Ok(out)
        }
        ggml_dtype::Q6_K => {
            let mut out = Vec::with_capacity(elems);
            crate::inference::dequantize_q6k_to_f32(data, elems, &mut out);
            Ok(out)
        }
        ggml_dtype::Q8_0 => {
            let mut out = Vec::with_capacity(elems);
            crate::inference::dequantize_q8_0_to_f32(data, elems, &mut out);
            Ok(out)
        }
        other => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("tensor {name}: unsupported dtype code {other}"),
        )),
    }
}

/// Load the (gate, up, down) matrices for **every** expert in one
/// layer. Returns a `Vec` of length `num_experts`, where each element is
/// `(gate, up, down)` shaped as `Engine` expects: gate/up are
/// `[d_ff, d_model]`, down is `[d_model, d_ff]`, all row-major.
///
/// This is the per-layer counterpart of `load_expert_matrices`: it
/// dequantises each interleaved tensor at most **once** per layer
/// (`O(1)` full-tensor decodes instead of `O(num_experts)`) and slices
/// each expert's stride from the cached f32 buffer.
///
/// Required expert tensors are fail-closed: missing, malformed, or
/// unsupported expert weights abort conversion instead of writing
/// placeholder zero blobs.
fn load_layer_expert_matrices(
    gguf: &dyn GgufSource,
    layer: usize,
    num_experts: usize,
    d_model: usize,
    d_ff: usize,
) -> io::Result<Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    let interleaved_gate_name = format!("blk.{layer}.ffn_gate_exps.weight");
    let interleaved_up_name = format!("blk.{layer}.ffn_up_exps.weight");
    let interleaved_down_name = format!("blk.{layer}.ffn_down_exps.weight");
    let per_expert_gate0 = format!("blk.{layer}.ffn_gate.0.weight");

    if gguf.has_tensor(&interleaved_gate_name) {
        // Decode each interleaved tensor exactly once and reuse it for
        // all experts in the layer.
        let gate_all = match dense_layer_tensor_f32(
            gguf,
            &interleaved_gate_name,
            num_experts * d_ff * d_model,
        ) {
            Ok(v) => v,
            Err(err) => return Err(err),
        };
        let up_all = match dense_layer_tensor_f32(
            gguf,
            &interleaved_up_name,
            num_experts * d_ff * d_model,
        ) {
            Ok(v) => v,
            Err(err) => return Err(err),
        };
        let down_all = match dense_layer_tensor_f32(
            gguf,
            &interleaved_down_name,
            num_experts * d_model * d_ff,
        ) {
            Ok(v) => v,
            Err(err) => return Err(err),
        };

        let per_gate_up = d_ff * d_model;
        let per_down = d_model * d_ff;
        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let gs = e * per_gate_up;
            let us = e * per_gate_up;
            let ds = e * per_down;
            out.push((
                gate_all[gs..gs + per_gate_up].to_vec(),
                up_all[us..us + per_gate_up].to_vec(),
                down_all[ds..ds + per_down].to_vec(),
            ));
        }
        Ok(out)
    } else if gguf.has_tensor(&per_expert_gate0) {
        // Per-expert tensors: each expert's tensors are already
        // separate, so a layer-level cache buys nothing.
        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            out.push(load_expert_matrices(
                gguf,
                layer,
                e,
                num_experts,
                d_model,
                d_ff,
            )?);
        }
        Ok(out)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("layer {layer} has no expert weight tensors"),
        ))
    }
}

/// Decode a dense tensor by name and validate it has exactly the
/// expected number of f32 elements. Errors with `InvalidData` on size
/// mismatch and propagates `Unsupported` from `bytes_to_f32`.
fn dense_layer_tensor_f32(
    gguf: &dyn GgufSource,
    name: &str,
    expected: usize,
) -> io::Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .cloned()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing")))?;
    let v = dense_tensor_to_f32(gguf, &info)?;
    if v.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor {name} has {} elements, expected {expected}",
                v.len()
            ),
        ));
    }
    Ok(v)
}

/// Load the (gate, up, down) matrices for one expert. Returns each as a
/// flat row-major `Vec<f32>` in the engine's expected layout: gate / up
/// are `[d_ff, d_model]`, down is `[d_model, d_ff]`.
///
/// Interleaved tensor shapes (GGML order, innermost-first):
///   `ffn_gate_exps.weight`: `[d_model, d_ff, num_experts]` for gate / up
///   `ffn_down_exps.weight`: `[d_ff,    d_model, num_experts]` for down
///
/// Per-expert tensor shapes:
///   `ffn_gate.{e}.weight`: `[d_model, d_ff]` (gate, up)
///   `ffn_down.{e}.weight`: `[d_ff,    d_model]` (down)
///
/// Prefer `load_layer_expert_matrices` for whole-layer extraction: it
/// caches each decoded interleaved tensor so the per-expert slicing is
/// not `O(num_experts)` redundant decodes.
fn load_expert_matrices(
    gguf: &dyn GgufSource,
    layer: usize,
    expert: usize,
    num_experts: usize,
    d_model: usize,
    d_ff: usize,
) -> io::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let interleaved_gate = format!("blk.{layer}.ffn_gate_exps.weight");
    let per_expert_gate = format!("blk.{layer}.ffn_gate.{expert}.weight");

    if gguf.has_tensor(&interleaved_gate) {
        let gate =
            slice_expert_matrix(gguf, &interleaved_gate, expert, num_experts, d_ff, d_model)?;
        let up = slice_expert_matrix(
            gguf,
            &format!("blk.{layer}.ffn_up_exps.weight"),
            expert,
            num_experts,
            d_ff,
            d_model,
        )?;
        let down = slice_expert_matrix(
            gguf,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            expert,
            num_experts,
            d_model,
            d_ff,
        )?;
        Ok((gate, up, down))
    } else if gguf.has_tensor(&per_expert_gate) {
        let gate = load_per_expert_matrix(gguf, &per_expert_gate, d_ff, d_model)?;
        let up = load_per_expert_matrix(
            gguf,
            &format!("blk.{layer}.ffn_up.{expert}.weight"),
            d_ff,
            d_model,
        )?;
        let down = load_per_expert_matrix(
            gguf,
            &format!("blk.{layer}.ffn_down.{expert}.weight"),
            d_model,
            d_ff,
        )?;
        Ok((gate, up, down))
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("layer {layer} expert {expert} has no expert weight tensors"),
        ))
    }
}

/// Read a single per-expert dense tensor and return it as a flat f32 vec
/// matching the engine's `[rows, cols]` row-major layout.
fn load_per_expert_matrix(
    gguf: &dyn GgufSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> io::Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .cloned()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing")))?;
    let v = dense_tensor_to_f32(gguf, &info)?;
    let expected = rows * cols;
    if v.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor {name} has {} elements, expected {expected} ({rows}x{cols})",
                v.len()
            ),
        ));
    }
    // GGML stores tensors innermost-first; for a 2-D `[cols, rows]`
    // GGML shape the flat layout is already row-major in the (rows,cols)
    // sense the engine expects. No transpose needed.
    Ok(v)
}

/// Slice one expert's matrix from an interleaved expert tensor.
///
/// The flat `f32` buffer is laid out as `num_experts × rows × cols`
/// contiguous f32s (GGML stores experts as the outermost dim in the
/// expert-interleaved layout). We dequantise the whole tensor once,
/// then slice the expert's rows × cols stride.
fn slice_expert_matrix(
    gguf: &dyn GgufSource,
    name: &str,
    expert: usize,
    num_experts: usize,
    rows: usize,
    cols: usize,
) -> io::Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .cloned()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing")))?;
    let all = dense_tensor_to_f32(gguf, &info)?;
    let per_expert = rows * cols;
    let want = num_experts * per_expert;
    if all.len() != want {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor {name} has {} elements, expected {want} ({num_experts}x{rows}x{cols})",
                all.len()
            ),
        ));
    }
    let start = expert * per_expert;
    Ok(all[start..start + per_expert].to_vec())
}

// (end of module body)

// ---------------------------------------------------------------------
// Native mixed-projection quant pass-through.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProjectionQuantSpec {
    dtype: WeightDtype,
    block_elems: usize,
    block_bytes: usize,
    weights: usize,
    payload_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProjectionQuantLayout {
    gate: ProjectionQuantSpec,
    up: ProjectionQuantSpec,
    down: ProjectionQuantSpec,
}

impl ProjectionQuantLayout {
    fn payload_bytes(self) -> Option<usize> {
        self.gate
            .payload_bytes
            .checked_add(self.up.payload_bytes)?
            .checked_add(self.down.payload_bytes)
    }

    fn uniform_dtype(self) -> Option<WeightDtype> {
        if self.gate.dtype == self.up.dtype && self.gate.dtype == self.down.dtype {
            Some(self.gate.dtype)
        } else {
            None
        }
    }

    fn all_projections_block_aligned(self) -> bool {
        [self.gate, self.up, self.down]
            .into_iter()
            .all(|spec| spec.weights % spec.block_elems == 0)
    }

    fn triple_key(self) -> String {
        format!(
            "{}/{}/{}",
            dtype_label(self.gate.dtype),
            dtype_label(self.up.dtype),
            dtype_label(self.down.dtype)
        )
    }
}

#[derive(Debug, Clone)]
struct ExpertQuantScan {
    layouts: Vec<ProjectionQuantLayout>,
    projection_histogram: BTreeMap<&'static str, BTreeMap<String, usize>>,
    triple_layers: BTreeMap<String, Vec<usize>>,
    missing_tensor_names: Vec<String>,
    block_alignment_failures: Vec<String>,
    rejection_reasons: Vec<String>,
}

impl ExpertQuantScan {
    fn new() -> Self {
        Self {
            layouts: Vec::new(),
            projection_histogram: BTreeMap::new(),
            triple_layers: BTreeMap::new(),
            missing_tensor_names: Vec::new(),
            block_alignment_failures: Vec::new(),
            rejection_reasons: Vec::new(),
        }
    }

    fn record_layer_layout(&mut self, layer: usize, layout: ProjectionQuantLayout) {
        for (projection, spec) in [
            ("gate", layout.gate),
            ("up", layout.up),
            ("down", layout.down),
        ] {
            *self
                .projection_histogram
                .entry(projection)
                .or_default()
                .entry(dtype_label(spec.dtype).to_string())
                .or_default() += 1;
        }
        let key = layout.triple_key();
        let layers = self.triple_layers.entry(key).or_default();
        if layers.last().copied() != Some(layer) {
            layers.push(layer);
        }
    }

    fn concise_report(&self) -> String {
        let mut lines = vec!["expert quant scan:".to_string()];
        for (triple, layers) in &self.triple_layers {
            lines.push(format!("  {triple}: {} layers", layers.len()));
        }
        for projection in ["gate", "up", "down"] {
            if let Some(hist) = self.projection_histogram.get(projection) {
                let parts = hist
                    .iter()
                    .map(|(dtype, count)| format!("{dtype}={count}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!("  {projection}: {parts}"));
            }
        }
        if !self.missing_tensor_names.is_empty() {
            lines.push(format!(
                "  missing tensors: {}",
                self.missing_tensor_names
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !self.block_alignment_failures.is_empty() {
            lines.push(format!(
                "  block alignment failures: {}",
                self.block_alignment_failures
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        if !self.rejection_reasons.is_empty() {
            lines.push(format!(
                "  rejected: {}",
                self.rejection_reasons
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        lines.join("\n")
    }

    fn into_plan(self, block_align: usize, emit_uth: bool) -> io::Result<NativeQuantPlan> {
        if self.layouts.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native quant scan found no expert layouts",
            ));
        }
        let first = self.layouts[0];
        if let Some(dtype) = first.uniform_dtype() {
            if self
                .layouts
                .iter()
                .all(|layout| layout.uniform_dtype() == Some(dtype))
                && self
                    .layouts
                    .iter()
                    .all(|layout| layout.all_projections_block_aligned())
            {
                return Ok(NativeQuantPlan::Homogeneous { dtype });
            }
        }
        if !emit_uth {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mixed native quant output requires UTH2; remove --no-uth",
            ));
        }
        let histogram = self.expert_histogram();
        let max_payload = self
            .layouts
            .iter()
            .map(|layout| {
                layout.payload_bytes().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "mixed payload size overflow")
                })
            })
            .collect::<io::Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let payload_slot_size = align_up(max_payload, block_align);
        let expert_size = payload_slot_size.checked_add(block_align).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "mixed expert size overflow")
        })?;
        Ok(NativeQuantPlan::Mixed {
            layouts: self.layouts,
            payload_slot_size,
            expert_size,
            histogram,
        })
    }

    fn expert_histogram(&self) -> BTreeMap<String, usize> {
        let mut hist = BTreeMap::new();
        for layout in &self.layouts {
            *hist
                .entry(layout.triple_key().to_ascii_lowercase())
                .or_default() += 1;
        }
        hist
    }
}

#[derive(Debug, Clone)]
enum NativeQuantPlan {
    Homogeneous {
        dtype: WeightDtype,
    },
    Mixed {
        layouts: Vec<ProjectionQuantLayout>,
        payload_slot_size: usize,
        expert_size: usize,
        histogram: BTreeMap<String, usize>,
    },
}

impl NativeQuantPlan {
    fn metadata_dtype(&self) -> WeightDtype {
        match self {
            NativeQuantPlan::Homogeneous { dtype, .. } => *dtype,
            NativeQuantPlan::Mixed { .. } => WeightDtype::Mixed,
        }
    }

    fn metadata_layout_version(&self) -> Option<u32> {
        match self {
            NativeQuantPlan::Homogeneous { .. } => None,
            NativeQuantPlan::Mixed { .. } => Some(2),
        }
    }

    fn metadata_histogram(&self) -> Option<BTreeMap<String, usize>> {
        match self {
            NativeQuantPlan::Homogeneous { .. } => None,
            NativeQuantPlan::Mixed { histogram, .. } => Some(histogram.clone()),
        }
    }
}

fn dtype_label(dtype: WeightDtype) -> &'static str {
    match dtype {
        WeightDtype::F32 => "F32",
        WeightDtype::F16 => "F16",
        WeightDtype::Int8 => "INT8",
        WeightDtype::Q4K => "Q4_K",
        WeightDtype::Q4_0 => "Q4_0",
        WeightDtype::Q8_0 => "Q8_0",
        WeightDtype::Q5K => "Q5_K",
        WeightDtype::Q6K => "Q6_K",
        WeightDtype::BF16 => "BF16",
        WeightDtype::MXFP4 => "MXFP4",
        WeightDtype::Mixed => "MIXED",
    }
}

fn native_block_spec(dtype: WeightDtype) -> Option<(usize, usize)> {
    match dtype {
        WeightDtype::Q4_0 => Some((Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES)),
        WeightDtype::Q4K => Some((Q4K_BLOCK_ELEMS, Q4K_BLOCK_BYTES)),
        WeightDtype::Q5K => Some((Q5K_BLOCK_ELEMS, Q5K_BLOCK_BYTES)),
        WeightDtype::Q6K => Some((Q6K_BLOCK_ELEMS, Q6K_BLOCK_BYTES)),
        WeightDtype::Q8_0 => Some((Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES)),
        _ => None,
    }
}

fn native_spec_for_tensor(
    info: &GgufTensorInfo,
    weights: usize,
    require_block_aligned: bool,
    label: &str,
) -> Result<ProjectionQuantSpec, String> {
    let dtype = crate::gguf::ggml_to_weight_dtype(info.ggml_dtype).ok_or_else(|| {
        format!(
            "{label} uses GGML dtype {}, which MER can size but cannot decode natively",
            info.ggml_dtype
        )
    })?;
    let (block_elems, block_bytes) = native_block_spec(dtype).ok_or_else(|| {
        format!(
            "{label} uses {}, which is not a native quantized expert projection dtype",
            dtype.as_str()
        )
    })?;
    if require_block_aligned && weights % block_elems != 0 {
        return Err(format!(
            "{label} has {weights} weights, not divisible by {} block elements for {}",
            block_elems,
            dtype_label(dtype)
        ));
    }
    let payload_bytes = projection_weight_bytes_for(weights, dtype).ok_or_else(|| {
        format!(
            "{label} payload byte size is unsupported for {}",
            dtype.as_str()
        )
    })?;
    Ok(ProjectionQuantSpec {
        dtype,
        block_elems,
        block_bytes,
        weights,
        payload_bytes,
    })
}

fn scan_expert_quant_layouts(
    gguf: &dyn GgufSource,
    num_layers: usize,
    num_experts: usize,
    d_model: usize,
    d_ff: usize,
) -> io::Result<ExpertQuantScan> {
    let mut scan = ExpertQuantScan::new();
    let weights = d_ff
        .checked_mul(d_model)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "d_ff * d_model overflow"))?;

    for layer in 0..num_layers {
        let interleaved_gate = format!("blk.{layer}.ffn_gate_exps.weight");
        if gguf.has_tensor(&interleaved_gate) {
            let mut specs = Vec::new();
            for (projection, name) in [
                ("gate", interleaved_gate.clone()),
                ("up", format!("blk.{layer}.ffn_up_exps.weight")),
                ("down", format!("blk.{layer}.ffn_down_exps.weight")),
            ] {
                match gguf.tensor_info(&name) {
                    Some(info) => {
                        let label = format!("layer {layer} {projection} projection");
                        match native_spec_for_tensor(info, weights, true, &label) {
                            Ok(spec) => {
                                let expected = num_experts
                                    .checked_mul(spec.payload_bytes)
                                    .ok_or_else(|| {
                                        io::Error::new(
                                            io::ErrorKind::InvalidData,
                                            format!("tensor {name} byte count overflow"),
                                        )
                                    })?;
                                if info.byte_len != expected as u64 {
                                    scan.rejection_reasons.push(format!(
                                        "tensor {name} has {} bytes, expected {expected}",
                                        info.byte_len
                                    ));
                                }
                                specs.push((projection, spec));
                            }
                            Err(reason) => {
                                if reason.contains("not divisible") {
                                    scan.block_alignment_failures.push(reason.clone());
                                }
                                scan.rejection_reasons.push(reason);
                            }
                        }
                    }
                    None => {
                        scan.missing_tensor_names.push(name.clone());
                        scan.rejection_reasons
                            .push(format!("missing tensor {name}"));
                    }
                }
            }
            if specs.len() == 3 {
                let layout = ProjectionQuantLayout {
                    gate: specs[0].1,
                    up: specs[1].1,
                    down: specs[2].1,
                };
                scan.record_layer_layout(layer, layout);
                scan.layouts
                    .extend(std::iter::repeat(layout).take(num_experts));
            }
            continue;
        }

        let per_expert_gate0 = format!("blk.{layer}.ffn_gate.0.weight");
        if gguf.has_tensor(&per_expert_gate0) {
            let mut first_layout_for_layer = None;
            for e in 0..num_experts {
                let mut specs = Vec::new();
                for (projection, name) in [
                    ("gate", format!("blk.{layer}.ffn_gate.{e}.weight")),
                    ("up", format!("blk.{layer}.ffn_up.{e}.weight")),
                    ("down", format!("blk.{layer}.ffn_down.{e}.weight")),
                ] {
                    match gguf.tensor_info(&name) {
                        Some(info) => {
                            let label = format!("layer {layer} expert {e} {projection} projection");
                            match native_spec_for_tensor(info, weights, false, &label) {
                                Ok(spec) => {
                                    if info.byte_len != spec.payload_bytes as u64 {
                                        scan.rejection_reasons.push(format!(
                                            "tensor {name} has {} bytes, expected {}",
                                            info.byte_len, spec.payload_bytes
                                        ));
                                    }
                                    specs.push((projection, spec));
                                }
                                Err(reason) => scan.rejection_reasons.push(reason),
                            }
                        }
                        None => {
                            scan.missing_tensor_names.push(name.clone());
                            scan.rejection_reasons
                                .push(format!("missing tensor {name}"));
                        }
                    }
                }
                if specs.len() == 3 {
                    let layout = ProjectionQuantLayout {
                        gate: specs[0].1,
                        up: specs[1].1,
                        down: specs[2].1,
                    };
                    first_layout_for_layer.get_or_insert(layout);
                    scan.layouts.push(layout);
                }
            }
            if let Some(layout) = first_layout_for_layer {
                scan.record_layer_layout(layer, layout);
            }
        } else {
            scan.missing_tensor_names.push(interleaved_gate.clone());
            scan.missing_tensor_names.push(per_expert_gate0.clone());
            scan.rejection_reasons.push(format!(
                "layer {layer} has neither interleaved expert tensors nor per-expert tensors"
            ));
        }
    }
    Ok(scan)
}

#[cfg(test)]
fn detect_native_quant_dtype(
    gguf: &dyn GgufSource,
    num_layers: usize,
    num_experts: usize,
    d_model: usize,
    d_ff: usize,
) -> Option<WeightDtype> {
    let scan = scan_expert_quant_layouts(gguf, num_layers, num_experts, d_model, d_ff).ok()?;
    if !scan.rejection_reasons.is_empty() {
        return None;
    }
    match scan.into_plan(DEFAULT_BLOCK_ALIGN, true).ok()? {
        NativeQuantPlan::Homogeneous { dtype, .. } => Some(dtype),
        NativeQuantPlan::Mixed { .. } => None,
    }
}

/// Load the (gate, up, down) **raw quantised byte streams** for every
/// expert in one layer.
///
/// Each returned `Vec<u8>` is a contiguous run of `Q4_0_BLOCK_BYTES` /
/// `Q4K_BLOCK_BYTES`-sized blocks ready to be written into an
/// `expert_<id>.bin` payload — `gate_blocks || up_blocks ||
/// down_blocks` is exactly the byte layout
/// [`crate::inference::OwnedExpertWeights::from_bytes_q4_0`] /
/// `from_bytes_q4k` consume.
///
/// Supports both expert tensor layouts (mirroring the F32 dequant
/// path):
///   * **interleaved** — one `[d_model, d_ff, num_experts]` tensor per
///     projection, sliced per expert at byte granularity:
///     `per_expert_bytes = (per_expert_weights / block_elems) *
///     block_bytes`, which the precondition checked by
///     [`detect_native_quant_dtype`] guarantees is exact;
///   * **per-expert** — a separate `[d_model, d_ff]` tensor per expert,
///     whose whole quantised body is exactly one expert's payload.
fn load_layer_expert_native_quant(
    gguf: &dyn GgufSource,
    layer: usize,
    num_experts: usize,
    d_model: usize,
    d_ff: usize,
    dtype: WeightDtype,
) -> io::Result<Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>> {
    let (block_elems, block_bytes) = match dtype {
        WeightDtype::Q4_0 => (Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES),
        WeightDtype::Q4K => (Q4K_BLOCK_ELEMS, Q4K_BLOCK_BYTES),
        WeightDtype::Q5K => (Q5K_BLOCK_ELEMS, Q5K_BLOCK_BYTES),
        WeightDtype::Q6K => (Q6K_BLOCK_ELEMS, Q6K_BLOCK_BYTES),
        WeightDtype::Q8_0 => (Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "load_layer_expert_native_quant: dtype must be Q4_0, Q4_K, Q5_K, Q6_K, or Q8_0",
            ));
        }
    };
    let per_expert_weights = d_ff
        .checked_mul(d_model)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "d_ff * d_model overflow"))?;
    if per_expert_weights % block_elems != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "per-expert weight count {per_expert_weights} is not block-aligned for {}",
                dtype.as_str()
            ),
        ));
    }
    let per_expert_bytes = (per_expert_weights / block_elems)
        .checked_mul(block_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "expert byte size overflow"))?;

    let read_tensor = |name: String| -> io::Result<Vec<u8>> {
        let info = gguf.tensor_info(&name).cloned().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing"))
        })?;
        gguf.read_tensor_owned(&info.name)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {} has no data slice", info.name),
            )
        })
    };

    if gguf.has_tensor(&format!("blk.{layer}.ffn_gate_exps.weight")) {
        // Interleaved layout: read each whole tensor once and slice the
        // per-expert stride out without dequantising.
        let read_layer_tensor = |name: String| -> io::Result<Vec<u8>> {
            let buf = read_tensor(name.clone())?;
            let need = num_experts.checked_mul(per_expert_bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "layer tensor byte size overflow",
                )
            })?;
            if buf.len() < need {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "tensor {name}: expected {need} quantised bytes, got {}",
                        buf.len()
                    ),
                ));
            }
            Ok(buf)
        };

        let gate_all = read_layer_tensor(format!("blk.{layer}.ffn_gate_exps.weight"))?;
        let up_all = read_layer_tensor(format!("blk.{layer}.ffn_up_exps.weight"))?;
        let down_all = read_layer_tensor(format!("blk.{layer}.ffn_down_exps.weight"))?;

        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let s = e * per_expert_bytes;
            let end = s + per_expert_bytes;
            out.push((
                gate_all[s..end].to_vec(),
                up_all[s..end].to_vec(),
                down_all[s..end].to_vec(),
            ));
        }
        Ok(out)
    } else {
        // Per-expert layout: each expert owns separate gate/up/down
        // tensors whose whole quantised body is exactly that expert's
        // payload (one full block run, no slicing).
        let read_per_expert = |name: String| -> io::Result<Vec<u8>> {
            let mut buf = read_tensor(name.clone())?;
            if buf.len() < per_expert_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "tensor {name}: expected {per_expert_bytes} quantised bytes, got {}",
                        buf.len()
                    ),
                ));
            }
            // GGUF rounds a quantised tensor up to a whole block, so the
            // body length matches `per_expert_bytes` exactly; truncate
            // defensively in case of any trailing alignment padding.
            buf.truncate(per_expert_bytes);
            Ok(buf)
        };

        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let gate = read_per_expert(format!("blk.{layer}.ffn_gate.{e}.weight"))?;
            let up = read_per_expert(format!("blk.{layer}.ffn_up.{e}.weight"))?;
            let down = read_per_expert(format!("blk.{layer}.ffn_down.{e}.weight"))?;
            out.push((gate, up, down));
        }
        Ok(out)
    }
}

fn load_layer_expert_native_mixed(
    gguf: &dyn GgufSource,
    layer: usize,
    num_experts: usize,
    _d_model: usize,
    _d_ff: usize,
    layouts: &[ProjectionQuantLayout],
) -> io::Result<Vec<(ProjectionQuantLayout, Vec<u8>, Vec<u8>, Vec<u8>)>> {
    let base = layer
        .checked_mul(num_experts)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "layer expert id overflow"))?;
    let read_tensor = |name: String| -> io::Result<Vec<u8>> {
        let info = gguf.tensor_info(&name).cloned().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing"))
        })?;
        gguf.read_tensor_owned(&info.name)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {} has no data slice", info.name),
            )
        })
    };

    if gguf.has_tensor(&format!("blk.{layer}.ffn_gate_exps.weight")) {
        let layout = *layouts.get(base).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing mixed layout for layer")
        })?;
        let read_layer_tensor = |name: String, spec: ProjectionQuantSpec| -> io::Result<Vec<u8>> {
            let buf = read_tensor(name.clone())?;
            let need = num_experts.checked_mul(spec.payload_bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "layer tensor byte size overflow",
                )
            })?;
            if buf.len() < need {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "tensor {name}: expected {need} quantised bytes, got {}",
                        buf.len()
                    ),
                ));
            }
            Ok(buf)
        };
        let gate_all = read_layer_tensor(format!("blk.{layer}.ffn_gate_exps.weight"), layout.gate)?;
        let up_all = read_layer_tensor(format!("blk.{layer}.ffn_up_exps.weight"), layout.up)?;
        let down_all = read_layer_tensor(format!("blk.{layer}.ffn_down_exps.weight"), layout.down)?;

        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let global = base + e;
            let layout = *layouts.get(global).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing mixed layout for expert",
                )
            })?;
            let slice = |buf: &[u8], spec: ProjectionQuantSpec| -> io::Result<Vec<u8>> {
                let start = e.checked_mul(spec.payload_bytes).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "expert slice offset overflow")
                })?;
                let end = start.checked_add(spec.payload_bytes).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "expert slice end overflow")
                })?;
                Ok(buf[start..end].to_vec())
            };
            out.push((
                layout,
                slice(&gate_all, layout.gate)?,
                slice(&up_all, layout.up)?,
                slice(&down_all, layout.down)?,
            ));
        }
        Ok(out)
    } else {
        let read_per_expert = |name: String, spec: ProjectionQuantSpec| -> io::Result<Vec<u8>> {
            let mut buf = read_tensor(name.clone())?;
            if buf.len() < spec.payload_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "tensor {name}: expected {} quantised bytes, got {}",
                        spec.payload_bytes,
                        buf.len()
                    ),
                ));
            }
            buf.truncate(spec.payload_bytes);
            Ok(buf)
        };
        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let global = base + e;
            let layout = *layouts.get(global).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing mixed layout for expert",
                )
            })?;
            let gate = read_per_expert(format!("blk.{layer}.ffn_gate.{e}.weight"), layout.gate)?;
            let up = read_per_expert(format!("blk.{layer}.ffn_up.{e}.weight"), layout.up)?;
            let down = read_per_expert(format!("blk.{layer}.ffn_down.{e}.weight"), layout.down)?;
            out.push((layout, gate, up, down));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn align_up_rounds_to_block_size() {
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    #[test]
    fn pack_expert_f32_pads_to_target() {
        let gate = vec![1.0f32, 2.0];
        let up = vec![3.0f32, 4.0];
        let down = vec![5.0f32, 6.0];
        let out = pack_expert_f32(&gate, &up, &down, 64);
        assert_eq!(out.len(), 64);
        // First 24 bytes = 6 little-endian f32s in order.
        let v: Vec<f32> = out[..24]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Tail is zeros.
        assert!(out[24..].iter().all(|&b| b == 0));
    }

    #[test]
    fn bytes_to_f32_decodes_f32_and_f16() {
        // F32 round-trip
        let v = vec![1.5f32, -2.0, 3.25];
        let mut bytes = Vec::new();
        for f in &v {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let got = bytes_to_f32(&bytes, ggml_dtype::F32, 3, "t").unwrap();
        assert_eq!(got, v);

        // F16 round-trip
        let v = vec![1.0f32, -2.0, 0.5];
        let mut bytes = Vec::new();
        for f in &v {
            bytes.extend_from_slice(&half::f16::from_f32(*f).to_le_bytes());
        }
        let got = bytes_to_f32(&bytes, ggml_dtype::F16, 3, "t").unwrap();
        assert_eq!(got, v);
    }

    #[test]
    fn bytes_to_f32_decodes_bf16() {
        // BF16 values are exactly representable when their f32 mantissa fits
        // in 7 bits, so these round-trip losslessly.
        let v = vec![1.0f32, -2.0, 0.5, 3.5];
        let mut bytes = Vec::new();
        for f in &v {
            bytes.extend_from_slice(&half::bf16::from_f32(*f).to_le_bytes());
        }
        let got = bytes_to_f32(&bytes, ggml_dtype::BF16, v.len(), "t").unwrap();
        assert_eq!(got, v);

        // A BF16 buffer must not be decoded as F16: the same bytes read as
        // F16 produce different values, proving the arms are distinct.
        let as_f16 = bytes_to_f32(&bytes, ggml_dtype::F16, v.len(), "t").unwrap();
        assert_ne!(as_f16, v);
    }

    #[test]
    fn bytes_to_f32_rejects_short_bf16_buffer() {
        let err = bytes_to_f32(&[0u8; 2], ggml_dtype::BF16, 4, "t").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn dense_manifest_emits_q8_canonical_file_without_alias_duplicate() {
        struct FakeSource {
            metadata: std::collections::HashMap<String, GgufValue>,
            tensors: std::collections::HashMap<String, GgufTensorInfo>,
            data: std::collections::HashMap<String, Vec<u8>>,
        }
        impl GgufSource for FakeSource {
            fn metadata(&self) -> &std::collections::HashMap<String, GgufValue> {
                &self.metadata
            }

            fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
                self.tensors.get(name)
            }

            fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
                Ok(self.data.get(name).cloned())
            }
        }

        use crate::inference::{quantize_q8_0_block, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 31.0) / 13.0).collect();
        let mut q8 = vec![0u8; 2 * Q8_0_BLOCK_BYTES];
        for block in 0..2 {
            quantize_q8_0_block(
                &values[block * Q8_0_BLOCK_ELEMS..(block + 1) * Q8_0_BLOCK_ELEMS],
                &mut q8[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES],
            );
        }
        let canonical = "token_embd.weight".to_string();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            canonical.clone(),
            GgufTensorInfo {
                name: canonical.clone(),
                shape: vec![32, 2],
                ggml_dtype: ggml_dtype::Q8_0,
                offset: 0,
                byte_len: q8.len() as u64,
            },
        );
        let mut data = std::collections::HashMap::new();
        data.insert(canonical.clone(), q8.clone());
        let source = FakeSource {
            metadata: std::collections::HashMap::new(),
            tensors,
            data,
        };
        let tmp = tempfile_dir();
        let mut report = ExtractionReport {
            experts_written: 0,
            dense_written: 0,
            skipped: 0,
            total_bytes: 0,
            expert_dtype: WeightDtype::F32,
            d_model: 32,
            d_ff: 0,
            num_experts_per_layer: 0,
            num_layers: 0,
        };
        let mut manifest = DenseTensorManifest {
            format_version: 1,
            tensors: Vec::new(),
        };
        emit_dense_manifest_tensor(
            &source,
            &tmp,
            &mut report,
            &mut manifest,
            &canonical,
            vec!["embed.bin".to_string(), "embedding.bin".to_string()],
        )
        .unwrap();

        assert_eq!(report.dense_written, 1);
        assert_eq!(manifest.tensors.len(), 1);
        let entry = &manifest.tensors[0];
        assert_eq!(entry.dtype, DenseDType::Q8_0);
        assert_eq!(entry.dims, vec![2, 32]);
        assert_eq!(entry.checksum.as_ref().unwrap(), &dense_checksum(&q8));
        assert_eq!(fs::read(tmp.join(&entry.file)).unwrap(), q8);
        assert!(!tmp.join("embed.bin").exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    // ---- Finding 2: architecture-required tensor emission guarantee ----

    struct F2FakeSource {
        tensors: std::collections::HashMap<String, GgufTensorInfo>,
        data: std::collections::HashMap<String, Vec<u8>>,
        metadata: std::collections::HashMap<String, GgufValue>,
    }
    impl GgufSource for F2FakeSource {
        fn metadata(&self) -> &std::collections::HashMap<String, GgufValue> {
            &self.metadata
        }
        fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
            self.tensors.get(name)
        }
        fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
            Ok(self.data.get(name).cloned())
        }
    }

    fn f2_empty_report() -> ExtractionReport {
        ExtractionReport {
            experts_written: 0,
            dense_written: 0,
            skipped: 0,
            total_bytes: 0,
            expert_dtype: WeightDtype::F32,
            d_model: 4,
            d_ff: 0,
            num_experts_per_layer: 0,
            num_layers: 0,
        }
    }

    #[test]
    fn required_dense_tensor_absence_is_fatal() {
        let source = F2FakeSource {
            tensors: std::collections::HashMap::new(),
            data: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
        };
        let tmp = tempfile_dir();
        let mut report = f2_empty_report();
        let mut manifest = DenseTensorManifest {
            format_version: 1,
            tensors: Vec::new(),
        };
        // Optional: absence is tolerated (counted as skipped).
        emit_dense_manifest_tensor_req(
            &source,
            &tmp,
            &mut report,
            &mut manifest,
            "token_embd.weight",
            vec!["embed.bin".to_string()],
            false,
        )
        .unwrap();
        assert_eq!(report.skipped, 1);
        // Required: absence is a hard error.
        let err = emit_dense_manifest_tensor_req(
            &source,
            &tmp,
            &mut report,
            &mut manifest,
            "token_embd.weight",
            vec!["embed.bin".to_string()],
            true,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn required_dense_tensor_unsupported_quant_is_fatal() {
        // A required tensor stored in an unsupported GGML dtype must fail the
        // conversion rather than being silently skipped.
        let name = "token_embd.weight".to_string();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            name.clone(),
            GgufTensorInfo {
                name: name.clone(),
                shape: vec![4, 2],
                // Q2_K is not decoded by the converter.
                ggml_dtype: ggml_dtype::Q2_K,
                offset: 0,
                byte_len: 64,
            },
        );
        let mut data = std::collections::HashMap::new();
        data.insert(name.clone(), vec![0u8; 64]);
        let source = F2FakeSource {
            tensors,
            data,
            metadata: std::collections::HashMap::new(),
        };
        let tmp = tempfile_dir();
        let mut report = f2_empty_report();
        let mut manifest = DenseTensorManifest {
            format_version: 1,
            tensors: Vec::new(),
        };
        let err = emit_dense_manifest_tensor_req(
            &source,
            &tmp,
            &mut report,
            &mut manifest,
            &name,
            vec!["embed.bin".to_string()],
            true,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        let _ = fs::remove_dir_all(&tmp);
    }

    /// The GGUF converter must extract Qwen QK-Norm tensors
    /// (`blk.{L}.attn_q_norm.weight` / `attn_k_norm.weight`) into the dense
    /// manifest under the engine aliases `q_norm_{L}.bin` / `k_norm_{L}.bin`.
    #[test]
    fn dense_manifest_emits_qk_norm_engine_aliases() {
        struct FakeSource {
            tensors: std::collections::HashMap<String, GgufTensorInfo>,
            data: std::collections::HashMap<String, Vec<u8>>,
            metadata: std::collections::HashMap<String, GgufValue>,
        }
        impl GgufSource for FakeSource {
            fn metadata(&self) -> &std::collections::HashMap<String, GgufValue> {
                &self.metadata
            }
            fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
                self.tensors.get(name)
            }
            fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
                Ok(self.data.get(name).cloned())
            }
        }

        let head_dim = 4usize;
        let canonical = "blk.0.attn_q_norm.weight".to_string();
        let values: Vec<f32> = (0..head_dim).map(|i| 0.5 + i as f32 * 0.1).collect();
        let bytes = f32_vec_to_le_bytes(&values);
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            canonical.clone(),
            GgufTensorInfo {
                name: canonical.clone(),
                shape: vec![head_dim as u64],
                ggml_dtype: ggml_dtype::F32,
                offset: 0,
                byte_len: bytes.len() as u64,
            },
        );
        let mut data = std::collections::HashMap::new();
        data.insert(canonical.clone(), bytes.clone());
        let source = FakeSource {
            tensors,
            data,
            metadata: std::collections::HashMap::new(),
        };
        let tmp = tempfile_dir();
        let mut report = ExtractionReport {
            experts_written: 0,
            dense_written: 0,
            skipped: 0,
            total_bytes: 0,
            expert_dtype: WeightDtype::F32,
            d_model: head_dim,
            d_ff: 0,
            num_experts_per_layer: 0,
            num_layers: 0,
        };
        let mut manifest = DenseTensorManifest {
            format_version: 1,
            tensors: Vec::new(),
        };
        emit_dense_manifest_tensor(
            &source,
            &tmp,
            &mut report,
            &mut manifest,
            &canonical,
            vec!["q_norm_0.bin".to_string()],
        )
        .unwrap();
        // A missing K-Norm (absent from the source) is skipped, not fatal.
        emit_dense_manifest_tensor(
            &source,
            &tmp,
            &mut report,
            &mut manifest,
            "blk.0.attn_k_norm.weight",
            vec!["k_norm_0.bin".to_string()],
        )
        .unwrap();

        assert_eq!(manifest.tensors.len(), 1, "only q_norm should be emitted");
        let entry = &manifest.tensors[0];
        assert_eq!(entry.canonical_name, canonical);
        assert_eq!(entry.aliases, vec!["q_norm_0.bin".to_string()]);
        assert_eq!(entry.dtype, DenseDType::F32);
        assert_eq!(entry.dims, vec![head_dim, 1]);
        assert_eq!(entry.checksum.as_ref().unwrap(), &dense_checksum(&bytes));
        assert_eq!(fs::read(tmp.join(&entry.file)).unwrap(), bytes);
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Smoke-test the full extraction path against a synthetic 1-layer,
    /// 2-expert GGUF (F32, per-expert layout). Verifies that we write
    /// `expert_0.bin`, `expert_1.bin`, and `metadata.json` with
    /// internally consistent sizes.
    #[test]
    fn extract_from_synthetic_per_expert_gguf() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        let out = tmp.join("out");
        let report = extract_experts_from_gguf(&gguf, &out, 1, num_experts).expect("extract");
        assert_eq!(report.experts_written, num_experts);
        // Each expert file is the engine's f32 SwiGLU blob, padded.
        for e in 0..num_experts {
            let path = out.join(format!("expert_{e}.bin"));
            let mut f = fs::File::open(&path).unwrap();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            assert_eq!(
                buf.len() % DEFAULT_BLOCK_ALIGN,
                0,
                "{path:?} not page-padded"
            );
            let needed = 3 * d_model * d_ff * 4;
            assert!(buf.len() >= needed, "expert file too small");
        }
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["d_model"], 4);
        assert_eq!(meta["d_ff"], 8);
        assert_eq!(meta["num_experts"], 2);
        assert_eq!(meta["dtype"], "f32");
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Verify the U.T.H. round-trip:
    ///   * `extract_experts_from_source` with default options writes a
    ///     header at byte 0 of every expert file
    ///   * `TensorHeader::strip` returns the bytes the engine cares
    ///     about (so `ExpertResident::data()` stays a transparent
    ///     view onto the payload)
    ///   * `--no-uth` opts out cleanly: the legacy bare-payload layout
    ///     is recovered and the header probe returns `None`
    #[test]
    fn extract_emits_uth_by_default_and_strips_cleanly() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();

        // Default options → UTH present.
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("with-uth");
        let _ = extract_experts_from_source(&gguf, &out, 1, num_experts, ExtractOptions::default())
            .expect("extract");
        for e in 0..num_experts {
            let path = out.join(format!("expert_{e}.bin"));
            let buf = fs::read(&path).unwrap();
            let header =
                crate::tensor_header::TensorHeader::probe(&buf).expect("U.T.H. must be present");
            assert_eq!(header.dtype.to_weight(), WeightDtype::F32);
            assert_eq!(header.shape[0] as usize, d_ff);
            // After stripping, the first byte must be at a 4 KiB offset.
            let (_, payload) = crate::tensor_header::TensorHeader::strip(&buf, DEFAULT_BLOCK_ALIGN);
            assert_eq!(buf.len() - payload.len(), DEFAULT_BLOCK_ALIGN);
            // Payload is still a whole number of pages.
            assert_eq!(payload.len() % DEFAULT_BLOCK_ALIGN, 0);
        }

        // `--no-uth` → bare-payload layout.
        let out2 = tmp.join("no-uth");
        let _ = extract_experts_from_source(
            &gguf,
            &out2,
            1,
            num_experts,
            ExtractOptions {
                emit_uth: false,
                native_quant: false,
                experts_only: false,
                arch_override: None,
            },
        )
        .expect("extract no-uth");
        for e in 0..num_experts {
            let path = out2.join(format!("expert_{e}.bin"));
            let buf = fs::read(&path).unwrap();
            assert!(
                crate::tensor_header::TensorHeader::probe(&buf).is_none(),
                "no-uth file must not carry a header"
            );
            // strip() must be a no-op when no header is present.
            let (h, payload) = crate::tensor_header::TensorHeader::strip(&buf, DEFAULT_BLOCK_ALIGN);
            assert!(h.is_none());
            assert_eq!(payload.len(), buf.len());
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validate_data_dir_rejects_corrupt_expert_file() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        let out = tmp.join("validate-corrupt");
        extract_experts_from_source(&gguf, &out, 1, num_experts, ExtractOptions::default())
            .expect("extract");
        fs::write(out.join("expert_0.bin"), vec![0u8; 32]).unwrap();

        let err = validate_data_dir(&out).expect_err("corrupt expert must fail validation");
        assert!(
            err.to_string().contains("has 32 bytes, expected"),
            "unexpected error: {err}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validate_data_dir_rejects_truncated_non_mixed_payload_even_when_metadata_matches() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        let out = tmp.join("validate-truncated-payload");
        extract_experts_from_source(&gguf, &out, 1, num_experts, ExtractOptions::default())
            .expect("extract");

        let expert = out.join("expert_0.bin");
        let truncated = fs::read(&expert).unwrap()[..DEFAULT_BLOCK_ALIGN].to_vec();
        fs::write(&expert, truncated).unwrap();
        let meta_path = out.join("metadata.json");
        let mut meta: serde_json::Value =
            serde_json::from_slice(&fs::read(&meta_path).unwrap()).unwrap();
        meta["expert_size"] = serde_json::json!(DEFAULT_BLOCK_ALIGN);
        fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let err = validate_data_dir(&out).expect_err("truncated payload must fail validation");
        assert!(
            err.to_string().contains("payload has 0 bytes"),
            "unexpected error: {err}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn experts_only_conversion_skips_dense_outputs() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        let out = tmp.join("experts-only");
        let report = extract_experts_from_source(
            &gguf,
            &out,
            1,
            num_experts,
            ExtractOptions {
                emit_uth: true,
                native_quant: false,
                experts_only: true,
                arch_override: None,
            },
        )
        .expect("extract experts only");

        assert_eq!(report.experts_written, num_experts);
        assert_eq!(report.dense_written, 0);
        assert!(!out.join("rms_attn_0.bin").exists());
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["conversion_mode"], "experts_only");
        assert_eq!(meta["dense_tensors_written"], 0);
        validate_data_dir(&out).expect("experts-only output validates");
        let _ = fs::remove_dir_all(&tmp);
    }

    fn tempfile_dir() -> std::path::PathBuf {
        static NEXT_TMP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let seq = NEXT_TMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        p.push(format!(
            "gguf-loader-test-{}-{nanos}-{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Synthetic vocabulary size for builder fixtures. Small but > 1 so the
    /// `[vocab, d_model]` embedding / LM-head dims are non-degenerate.
    const SYNTH_VOCAB: usize = 8;

    /// Logically-consistent `(q_dim, kv_dim)` for a synthetic GQA attention
    /// block: `head_dim = 2` (or 1 for odd `d_model`), `n_heads = d_model /
    /// head_dim`, `n_kv_heads = max(1, n_heads / 2)`. Shapes therefore reflect
    /// the builder's `d_model`, heads and KV heads rather than arbitrary
    /// squares.
    fn synth_qkv_dims(d_model: usize) -> (usize, usize) {
        let head_dim = if d_model % 2 == 0 { 2 } else { 1 };
        let n_heads = (d_model / head_dim).max(1);
        let n_kv_heads = (n_heads / 2).max(1);
        (n_heads * head_dim, n_kv_heads * head_dim)
    }

    /// Dense-tensor emission knobs for the synthetic GGUF builder. Defaults
    /// reproduce the historical fixture (independent `output.weight`, no tie
    /// metadata); tests toggle these to exercise the tied-output contract and
    /// the fail-closed untied path (Finding 5).
    #[derive(Clone, Copy)]
    struct SynthDenseKnobs {
        emit_output: bool,
        tie_word_embeddings: bool,
    }

    impl Default for SynthDenseKnobs {
        fn default() -> Self {
            Self {
                emit_output: true,
                tie_word_embeddings: false,
            }
        }
    }

    fn build_synth_gguf(d_model: usize, d_ff: usize, num_experts: usize) -> Vec<u8> {
        build_synth_gguf_arch(d_model, d_ff, num_experts, "llama")
    }

    /// Like [`build_synth_gguf`] but namespaces the hyperparameter
    /// metadata under an arbitrary `arch` (and declares it via
    /// `general.architecture`). Used to exercise the architecture-
    /// agnostic metadata resolution path for non-llama producers.
    fn build_synth_gguf_arch(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        arch: &str,
    ) -> Vec<u8> {
        build_synth_gguf_arch_ext(d_model, d_ff, num_experts, arch, None, SynthDenseKnobs::default())
    }

    /// Like [`build_synth_gguf_arch`] but, when `dense_ffl` is `Some`,
    /// declares a *larger* dense `feed_forward_length` while sizing the
    /// expert tensors with `expert_d_ff` and exposing the latter under
    /// `expert_feed_forward_length`. Exercises the MoE-specific `d_ff`
    /// preference (Qwen2-MoE) without disturbing the dense default.
    fn build_synth_gguf_arch_ext(
        d_model: usize,
        expert_d_ff: usize,
        num_experts: usize,
        arch: &str,
        dense_ffl: Option<usize>,
        knobs: SynthDenseKnobs,
    ) -> Vec<u8> {
        let d_ff = expert_d_ff;
        use crate::gguf::GGUF_MAGIC;
        // Architectures on the QK-Norm contract (Qwen3 / Qwen3-MoE) additionally
        // require per-head `attn_q_norm`/`attn_k_norm` tensors and the head-count
        // metadata used to validate their length.
        let uses_qk_norm = resolve_conversion_profile(Some(arch)).uses_qk_norm;
        let qk_head_dim = if d_model % 2 == 0 { 2 } else { 1 };
        let head_count = (d_model / qk_head_dim).max(1);
        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes()); // version
                                                    // tensor_count: per layer we emit the attention projections
                                                    // (q,k,v,o), attn_norm, ffn_norm and the routed gate (7 dense)
                                                    // plus 3*num_experts FFN, and 2-3 model-global tensors. QK-Norm
                                                    // architectures add attn_q_norm/attn_k_norm (2 more) per layer.
        let per_layer_tensors = 7 + 3 * num_experts + if uses_qk_norm { 2 } else { 0 };
        let dense_global = if knobs.emit_output { 3 } else { 2 };
        let total_tensors = dense_global /* token_embd, output_norm, [output] */ + per_layer_tensors;
        out.extend_from_slice(&(total_tensors as u64).to_le_bytes());
        // metadata
        let kvs: Vec<(&str, u32, Vec<u8>)> = vec![
            ("general.alignment", 4, 32u32.to_le_bytes().to_vec()),
            ("general.architecture", 8, lstring(arch.as_bytes())),
            ("general.name", 8, lstring(b"synth")),
            (
                leak_str(format!("{arch}.block_count")),
                4,
                1u32.to_le_bytes().to_vec(),
            ),
            (
                leak_str(format!("{arch}.expert_count")),
                4,
                (num_experts as u32).to_le_bytes().to_vec(),
            ),
            (
                leak_str(format!("{arch}.embedding_length")),
                4,
                (d_model as u32).to_le_bytes().to_vec(),
            ),
            (
                leak_str(format!("{arch}.feed_forward_length")),
                4,
                (dense_ffl.unwrap_or(d_ff) as u32).to_le_bytes().to_vec(),
            ),
            (
                leak_str(format!("{arch}.expert_used_count")),
                4,
                2u32.to_le_bytes().to_vec(),
            ),
        ];
        // MoE files expose the routed-expert hidden dim separately.
        let kvs = {
            let mut kvs = kvs;
            if dense_ffl.is_some() {
                kvs.push((
                    leak_str(format!("{arch}.expert_feed_forward_length")),
                    4,
                    (expert_d_ff as u32).to_le_bytes().to_vec(),
                ));
            }
            if knobs.tie_word_embeddings {
                // GGUF bool metadata (type 7): single byte payload. The
                // converter reads the `general.tie_word_embeddings` flag.
                kvs.push(("general.tie_word_embeddings", 7, vec![1u8]));
            }
            if uses_qk_norm {
                // Head count lets the converter derive head_dim (d_model /
                // head_count) to validate QK-Norm vector length.
                kvs.push((
                    leak_str(format!("{arch}.attention.head_count")),
                    4,
                    (head_count as u32).to_le_bytes().to_vec(),
                ));
            }
            kvs
        };
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, ty, payload) in &kvs {
            let kb = k.as_bytes();
            out.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            out.extend_from_slice(kb);
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(payload);
        }
        // Tensor info table.
        let mut tensor_data_blobs: Vec<Vec<u8>> = Vec::new();
        let mut tensor_offsets: Vec<u64> = Vec::new();
        let mut cur_off: u64 = 0;
        let push_tensor = |out: &mut Vec<u8>,
                           tensor_data_blobs: &mut Vec<Vec<u8>>,
                           tensor_offsets: &mut Vec<u64>,
                           cur_off: &mut u64,
                           name: &str,
                           shape: &[u64],
                           dtype: u32,
                           values: Vec<f32>| {
            let nb = name.as_bytes();
            out.extend_from_slice(&(nb.len() as u64).to_le_bytes());
            out.extend_from_slice(nb);
            out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&dtype.to_le_bytes());
            tensor_offsets.push(*cur_off);
            out.extend_from_slice(&cur_off.to_le_bytes());
            let mut blob = Vec::with_capacity(values.len() * 4);
            for v in &values {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            *cur_off += blob.len() as u64;
            // Pad each tensor to 32 bytes for alignment compat.
            while *cur_off % 32 != 0 {
                blob.push(0);
                *cur_off += 1;
            }
            tensor_data_blobs.push(blob);
        };
        // 1 attn_norm with d_model elements
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.attn_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            vec![1.0; d_model],
        );
        // Model-global resident tensors now required by the converter (F2):
        // token embedding, final norm, and an (untied) LM head loaded
        // independently. GGUF stores 2-D weights as [in, out]; `[d_model, vocab]`
        // therefore decodes to logical `[vocab, d_model]`.
        let vocab = SYNTH_VOCAB;
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "token_embd.weight",
            &[d_model as u64, vocab as u64],
            ggml_dtype::F32,
            vec![0.05; vocab * d_model],
        );
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "output_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            vec![1.0; d_model],
        );
        if knobs.emit_output {
            push_tensor(
                &mut out,
                &mut tensor_data_blobs,
                &mut tensor_offsets,
                &mut cur_off,
                "output.weight",
                &[d_model as u64, vocab as u64],
                ggml_dtype::F32,
                vec![0.06; vocab * d_model],
            );
        }
        // Attention projections with logically-consistent GQA shapes.
        let (q_dim, kv_dim) = synth_qkv_dims(d_model);
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.attn_q.weight",
            &[d_model as u64, q_dim as u64],
            ggml_dtype::F32,
            vec![0.1; d_model * q_dim],
        );
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.attn_k.weight",
            &[d_model as u64, kv_dim as u64],
            ggml_dtype::F32,
            vec![0.11; d_model * kv_dim],
        );
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.attn_v.weight",
            &[d_model as u64, kv_dim as u64],
            ggml_dtype::F32,
            vec![0.12; d_model * kv_dim],
        );
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.attn_output.weight",
            &[q_dim as u64, d_model as u64],
            ggml_dtype::F32,
            vec![0.13; q_dim * d_model],
        );
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.ffn_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            vec![1.0; d_model],
        );
        if uses_qk_norm {
            // Per-head QK-Norm vectors: exactly `head_dim` elements each.
            push_tensor(
                &mut out,
                &mut tensor_data_blobs,
                &mut tensor_offsets,
                &mut cur_off,
                "blk.0.attn_q_norm.weight",
                &[qk_head_dim as u64],
                ggml_dtype::F32,
                vec![1.0; qk_head_dim],
            );
            push_tensor(
                &mut out,
                &mut tensor_data_blobs,
                &mut tensor_offsets,
                &mut cur_off,
                "blk.0.attn_k_norm.weight",
                &[qk_head_dim as u64],
                ggml_dtype::F32,
                vec![1.0; qk_head_dim],
            );
        }
        // Routed gate: logical `[num_experts, d_model]` (GGUF shape [d_model, E]).
        push_tensor(
            &mut out,
            &mut tensor_data_blobs,
            &mut tensor_offsets,
            &mut cur_off,
            "blk.0.ffn_gate_inp.weight",
            &[d_model as u64, num_experts as u64],
            ggml_dtype::F32,
            vec![0.01; num_experts * d_model],
        );
        // FFN per-expert
        for e in 0..num_experts {
            push_tensor(
                &mut out,
                &mut tensor_data_blobs,
                &mut tensor_offsets,
                &mut cur_off,
                &format!("blk.0.ffn_gate.{e}.weight"),
                &[d_model as u64, d_ff as u64],
                ggml_dtype::F32,
                vec![e as f32 + 0.1; d_ff * d_model],
            );
            push_tensor(
                &mut out,
                &mut tensor_data_blobs,
                &mut tensor_offsets,
                &mut cur_off,
                &format!("blk.0.ffn_up.{e}.weight"),
                &[d_model as u64, d_ff as u64],
                ggml_dtype::F32,
                vec![e as f32 + 0.2; d_ff * d_model],
            );
            push_tensor(
                &mut out,
                &mut tensor_data_blobs,
                &mut tensor_offsets,
                &mut cur_off,
                &format!("blk.0.ffn_down.{e}.weight"),
                &[d_ff as u64, d_model as u64],
                ggml_dtype::F32,
                vec![e as f32 + 0.3; d_model * d_ff],
            );
        }
        // Pad header end to 32 bytes for the tensor data section.
        while out.len() % 32 != 0 {
            out.push(0);
        }
        for blob in &tensor_data_blobs {
            out.extend_from_slice(blob);
        }
        out
    }

    /// `native_quant: true` now means quantized output is required.
    /// F32-source experts must fail before writing expert files rather
    /// than silently expanding a quantized checkpoint into F32 output.
    #[test]
    fn extract_native_quant_fails_when_source_is_f32() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        let out = tmp.join("native-fallback");
        let err = extract_experts_from_source(
            &gguf,
            &out,
            1,
            num_experts,
            ExtractOptions {
                emit_uth: true,
                native_quant: true,
                experts_only: false,
                arch_override: None,
            },
        )
        .expect_err("F32 native quant should fail closed");
        assert!(
            err.to_string()
                .contains("not a native quantized expert projection dtype"),
            "{err}"
        );
        assert!(!out.join("expert_0.bin").exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Regression test for the architecture-agnostic metadata
    /// resolution. Builds a synthetic GGUF whose hyperparameters live
    /// under a non-llama namespace (`qwen2moe.*`) and calls
    /// `extract_experts_from_source` with **zero** layer/expert hints,
    /// forcing `lookup(...)` down the `<general.architecture>.<suffix>`
    /// auto-detect branch (the `llama.<suffix>` probe misses entirely).
    #[test]
    fn extract_auto_detects_non_llama_architecture_metadata() {
        let d_model = 4usize;
        let d_ff = 8usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf_arch(d_model, d_ff, num_experts, "qwen2moe");
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        // 0 hints → all shape/layout params must be auto-detected from
        // the `qwen2moe.*` namespace.
        let out = tmp.join("auto-detect");
        let report = extract_experts_from_source(&gguf, &out, 0, 0, ExtractOptions::default())
            .expect("extract");
        assert_eq!(report.experts_written, num_experts);
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["d_model"], d_model);
        assert_eq!(meta["d_ff"], d_ff);
        assert_eq!(meta["num_experts"], num_experts);
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Regression test for the MoE `d_ff` extraction. Qwen2-MoE files
    /// size their routed experts with `expert_feed_forward_length`, a
    /// value distinct from (and usually much smaller than) the dense
    /// `feed_forward_length`. The converter must prefer the former so
    /// the `num_experts * d_ff * d_model` element math matches the
    /// actual expert tensor byte count. Builds a GGUF whose dense
    /// `feed_forward_length` (32) disagrees with the expert hidden dim
    /// (8) and asserts the expert files / metadata use the expert dim.
    #[test]
    fn extract_moe_prefers_expert_feed_forward_length() {
        let d_model = 4usize;
        let expert_d_ff = 8usize;
        let dense_ffl = 32usize; // deliberately larger / wrong for experts
        let num_experts = 2usize;
        let bytes = build_synth_gguf_arch_ext(
            d_model,
            expert_d_ff,
            num_experts,
            "qwen2moe",
            Some(dense_ffl),
            SynthDenseKnobs::default(),
        );
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        // 0 hints → d_ff must be auto-detected from
        // `qwen2moe.expert_feed_forward_length`, not the dense key.
        let out = tmp.join("moe-d-ff");
        let report = extract_experts_from_source(&gguf, &out, 0, 0, ExtractOptions::default())
            .expect("extract");
        assert_eq!(report.experts_written, num_experts);
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(
            meta["d_ff"], expert_d_ff,
            "MoE conversion must use expert_feed_forward_length"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Finding 6: an unknown `general.architecture` is not on the conversion
    /// allowlist and must fail closed instead of being silently treated as a
    /// generic separate-QKV llama-like checkpoint.
    #[test]
    fn convert_unknown_architecture_fails_closed() {
        let bytes = build_synth_gguf_arch_ext(
            4,
            8,
            2,
            "totally_unknown_arch",
            None,
            SynthDenseKnobs::default(),
        );
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("unknown");
        let err = extract_experts_from_source(&gguf, &out, 1, 2, ExtractOptions::default())
            .expect_err("unknown architecture must fail closed");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported, "{err}");
        assert!(
            err.to_string().contains("not a recognised convertible architecture"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Finding 6: an explicit safe separate-QKV `arch_override` rescues an
    /// otherwise-unrecognised checkpoint, letting the operator convert a
    /// llama-like export whose `general.architecture` is not on the allowlist.
    #[test]
    fn convert_unknown_architecture_rescued_by_override() {
        let bytes = build_synth_gguf_arch_ext(
            4,
            8,
            2,
            "totally_unknown_arch",
            None,
            SynthDenseKnobs::default(),
        );
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("override");
        let report = extract_experts_from_source(
            &gguf,
            &out,
            1,
            2,
            ExtractOptions {
                arch_override: Some("llama".to_string()),
                ..ExtractOptions::default()
            },
        )
        .expect("safe separate-QKV override converts");
        assert_eq!(report.experts_written, 2);
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Finding 6: recognised fused-QKV families (Phi) use an attention tensor
    /// set the expert extractor cannot satisfy, so conversion fails with an
    /// architecture-specific error rather than demanding nonexistent separate
    /// Q/K/V tensors.
    #[test]
    fn convert_fused_qkv_architecture_is_unsupported() {
        let bytes = build_synth_gguf_arch_ext(4, 8, 2, "phi3", None, SynthDenseKnobs::default());
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("phi3");
        let err = extract_experts_from_source(&gguf, &out, 1, 2, ExtractOptions::default())
            .expect_err("fused-QKV architecture is unsupported");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported, "{err}");
        assert!(err.to_string().contains("fused QKV"), "{err}");
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Finding 5: Qwen3-MoE ships no `output.weight` and duplicates the token
    /// embedding — tying is established by the architecture GGUF contract even
    /// with no `tie_word_embeddings` metadata flag. Conversion must succeed and
    /// record a tied LM-head manifest entry pointing at `token_embd.weight`.
    #[test]
    fn convert_qwen3_moe_ties_output_by_contract() {
        let bytes = build_synth_gguf_arch_ext(
            4,
            8,
            2,
            "qwen3moe",
            None,
            SynthDenseKnobs {
                emit_output: false,
                tie_word_embeddings: false,
            },
        );
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("qwen3moe");
        extract_experts_from_source(&gguf, &out, 1, 2, ExtractOptions::default())
            .expect("qwen3-moe tied-by-contract conversion succeeds");
        let manifest: DenseTensorManifest = serde_json::from_slice(
            &fs::read(out.join("dense_manifest.json")).expect("manifest written"),
        )
        .expect("manifest parses");
        let head = manifest
            .tensors
            .iter()
            .find(|t| t.canonical_name == "output.weight")
            .expect("output.weight manifest entry present");
        assert_eq!(
            head.tied_to.as_deref(),
            Some("token_embd.weight"),
            "LM head must be tied to token embedding by architecture contract"
        );
        assert_eq!(head.byte_len, 0, "tied head writes no weight bytes");
        assert!(
            !out.join("lm_head.bin").exists(),
            "tied head must not materialise a weight file"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Finding 5: a recognised *untied* architecture (Mixtral) that is missing
    /// `output.weight` with no tie metadata is a fatal partial conversion —
    /// tying is never inferred from absence alone.
    #[test]
    fn convert_untied_architecture_missing_output_fails() {
        let bytes = build_synth_gguf_arch_ext(
            4,
            8,
            2,
            "mixtral",
            None,
            SynthDenseKnobs {
                emit_output: false,
                tie_word_embeddings: false,
            },
        );
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("mixtral-untied");
        let err = extract_experts_from_source(&gguf, &out, 1, 2, ExtractOptions::default())
            .expect_err("untied architecture missing output.weight must fail");
        assert!(
            err.to_string().contains("embedding tying is not established"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Finding 5: an untied architecture missing `output.weight` but carrying an
    /// explicit `tie_word_embeddings = true` metadata flag ties by metadata.
    #[test]
    fn convert_untied_architecture_ties_by_metadata_flag() {
        let bytes = build_synth_gguf_arch_ext(
            4,
            8,
            2,
            "mixtral",
            None,
            SynthDenseKnobs {
                emit_output: false,
                tie_word_embeddings: true,
            },
        );
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");
        let out = tmp.join("mixtral-tied-meta");
        extract_experts_from_source(&gguf, &out, 1, 2, ExtractOptions::default())
            .expect("tie_word_embeddings metadata establishes tying");
        let manifest: DenseTensorManifest = serde_json::from_slice(
            &fs::read(out.join("dense_manifest.json")).expect("manifest written"),
        )
        .expect("manifest parses");
        let head = manifest
            .tensors
            .iter()
            .find(|t| t.canonical_name == "output.weight")
            .expect("output.weight manifest entry present");
        assert_eq!(head.tied_to.as_deref(), Some("token_embd.weight"));
        let _ = fs::remove_dir_all(&tmp);
    }

    /// stride is `(d_ff*d_model / block_elems) * block_bytes`, and
    /// the eligibility check rejects `d_ff*d_model` values that
    /// don't divide cleanly by the block size.
    #[test]
    fn native_quant_slicing_arithmetic() {
        // Q4_0: 32 elements / 18 bytes per block. 4 * 8 = 32 weights
        // → exactly 1 block per expert per tensor → 18 bytes.
        let block_elems = 32usize;
        let block_bytes = 18usize;
        let per_expert_weights = 4 * 8usize;
        assert_eq!(per_expert_weights % block_elems, 0);
        let per_expert_bytes = (per_expert_weights / block_elems) * block_bytes;
        assert_eq!(per_expert_bytes, 18);

        // Q4_K: 256 / 144. 4 * 8 = 32 weights → ineligible
        // (32 % 256 != 0); the converter must fall back to F32.
        let q4k_block_elems = 256usize;
        assert_ne!(per_expert_weights % q4k_block_elems, 0);
    }

    #[test]
    fn uniform_tail_block_layout_uses_uth2_plan() {
        let weights = Q5K_BLOCK_ELEMS + 1;
        let payload_bytes =
            projection_weight_bytes_for(weights, WeightDtype::Q5K).expect("Q5_K sizing");
        let spec = ProjectionQuantSpec {
            dtype: WeightDtype::Q5K,
            block_elems: Q5K_BLOCK_ELEMS,
            block_bytes: Q5K_BLOCK_BYTES,
            weights,
            payload_bytes,
        };
        let layout = ProjectionQuantLayout {
            gate: spec,
            up: spec,
            down: spec,
        };
        assert_eq!(payload_bytes, Q5K_BLOCK_BYTES * 2);
        assert!(!layout.all_projections_block_aligned());

        let mut scan = ExpertQuantScan::new();
        scan.layouts.push(layout);
        match scan
            .into_plan(DEFAULT_BLOCK_ALIGN, true)
            .expect("UTH2 plan")
        {
            NativeQuantPlan::Mixed {
                payload_slot_size,
                expert_size,
                layouts,
                histogram,
            } => {
                let payload = layout.payload_bytes().expect("layout payload");
                assert_eq!(layouts, vec![layout]);
                assert_eq!(payload_slot_size, align_up(payload, DEFAULT_BLOCK_ALIGN));
                assert_eq!(expert_size, payload_slot_size + DEFAULT_BLOCK_ALIGN);
                assert_eq!(histogram.get("q5_k/q5_k/q5_k"), Some(&1));
            }
            other => panic!("tail-block per-expert layout must use UTH2, got {other:?}"),
        }

        let mut no_uth_scan = ExpertQuantScan::new();
        no_uth_scan.layouts.push(layout);
        let err = no_uth_scan
            .into_plan(DEFAULT_BLOCK_ALIGN, false)
            .expect_err("tail-block layout needs UTH2");
        assert!(
            err.to_string().contains("requires UTH2"),
            "unexpected error: {err}"
        );
    }

    fn lstring(s: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + s.len());
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s);
        out
    }

    /// Leak an owned `String` into a `'static str`. Test-only helper for
    /// building metadata key tables whose names are computed at runtime
    /// (e.g. `<arch>.block_count`); the small leak is bounded by the
    /// number of synthetic GGUFs a test run constructs.
    fn leak_str(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// Build a per-expert-layout GGUF whose FFN expert tensors are raw
    /// `Q4_0` blocks (`blk.0.ffn_{gate,up,down}.{e}.weight`). Each block
    /// is `Q4_0_BLOCK_BYTES` (2-byte f16 scale + 16 quant bytes) and
    /// covers `Q4_0_BLOCK_ELEMS` weights. Used to exercise native
    /// pass-through for the per-expert layout.
    fn build_synth_gguf_q4_0_per_expert(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
    ) -> Vec<u8> {
        use crate::gguf::GGUF_MAGIC;
        assert_eq!((d_ff * d_model) % Q4_0_BLOCK_ELEMS, 0);
        let blocks = (d_ff * d_model) / Q4_0_BLOCK_ELEMS;
        // One Q4_0 block: f16 scale 0.1 (0x2e66) + 16 nibble-pair bytes.
        let q4_0_block: Vec<u8> = {
            let mut b = vec![0x66u8, 0x2e];
            b.extend_from_slice(&[0x77u8; 16]);
            b
        };
        let expert_blob: Vec<u8> = q4_0_block.repeat(blocks);

        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        // 7 per-layer dense (q,k,v,o,attn_norm,ffn_norm,gate) + 3*num_experts
        // FFN + 3 model-global tensors.
        let per_layer_tensors = 7 + 3 * num_experts;
        let total_tensors = 3 + per_layer_tensors;
        out.extend_from_slice(&(total_tensors as u64).to_le_bytes());
        let kvs: Vec<(&str, u32, Vec<u8>)> = vec![
            ("general.alignment", 4, 32u32.to_le_bytes().to_vec()),
            ("general.architecture", 8, lstring(b"llama")),
            ("general.name", 8, lstring(b"synth")),
            ("llama.block_count", 4, 1u32.to_le_bytes().to_vec()),
            (
                "llama.expert_count",
                4,
                (num_experts as u32).to_le_bytes().to_vec(),
            ),
            (
                "llama.embedding_length",
                4,
                (d_model as u32).to_le_bytes().to_vec(),
            ),
            (
                "llama.feed_forward_length",
                4,
                (d_ff as u32).to_le_bytes().to_vec(),
            ),
            ("llama.expert_used_count", 4, 2u32.to_le_bytes().to_vec()),
        ];
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, ty, payload) in &kvs {
            let kb = k.as_bytes();
            out.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            out.extend_from_slice(kb);
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(payload);
        }

        let mut tensor_data_blobs: Vec<Vec<u8>> = Vec::new();
        let mut cur_off: u64 = 0;
        let mut push_raw =
            |out: &mut Vec<u8>, name: &str, shape: &[u64], dtype: u32, mut blob: Vec<u8>| {
                let nb = name.as_bytes();
                out.extend_from_slice(&(nb.len() as u64).to_le_bytes());
                out.extend_from_slice(nb);
                out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                for d in shape {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&dtype.to_le_bytes());
                out.extend_from_slice(&cur_off.to_le_bytes());
                cur_off += blob.len() as u64;
                while cur_off % 32 != 0 {
                    blob.push(0);
                    cur_off += 1;
                }
                tensor_data_blobs.push(blob);
            };

        let mut attn = Vec::with_capacity(d_model * 4);
        for _ in 0..d_model {
            attn.extend_from_slice(&1.0f32.to_le_bytes());
        }
        push_raw(
            &mut out,
            "blk.0.attn_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            attn,
        );
        // Required F32 model-global + attention/norm/gate dense tensors (F2).
        let vocab = SYNTH_VOCAB;
        let (q_dim, kv_dim) = synth_qkv_dims(d_model);
        push_raw(
            &mut out,
            "token_embd.weight",
            &[d_model as u64, vocab as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.05; vocab * d_model]),
        );
        push_raw(
            &mut out,
            "output_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![1.0; d_model]),
        );
        push_raw(
            &mut out,
            "output.weight",
            &[d_model as u64, vocab as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.06; vocab * d_model]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_q.weight",
            &[d_model as u64, q_dim as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.1; d_model * q_dim]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_k.weight",
            &[d_model as u64, kv_dim as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.11; d_model * kv_dim]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_v.weight",
            &[d_model as u64, kv_dim as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.12; d_model * kv_dim]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_output.weight",
            &[q_dim as u64, d_model as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.13; q_dim * d_model]),
        );
        push_raw(
            &mut out,
            "blk.0.ffn_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![1.0; d_model]),
        );
        push_raw(
            &mut out,
            "blk.0.ffn_gate_inp.weight",
            &[d_model as u64, num_experts as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.01; num_experts * d_model]),
        );
        for e in 0..num_experts {
            push_raw(
                &mut out,
                &format!("blk.0.ffn_gate.{e}.weight"),
                &[d_model as u64, d_ff as u64],
                ggml_dtype::Q4_0,
                expert_blob.clone(),
            );
            push_raw(
                &mut out,
                &format!("blk.0.ffn_up.{e}.weight"),
                &[d_model as u64, d_ff as u64],
                ggml_dtype::Q4_0,
                expert_blob.clone(),
            );
            push_raw(
                &mut out,
                &format!("blk.0.ffn_down.{e}.weight"),
                &[d_ff as u64, d_model as u64],
                ggml_dtype::Q4_0,
                expert_blob.clone(),
            );
        }
        while out.len() % 32 != 0 {
            out.push(0);
        }
        for blob in &tensor_data_blobs {
            out.extend_from_slice(blob);
        }
        out
    }

    fn build_synth_gguf_mixed_per_expert(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
    ) -> Vec<u8> {
        use crate::gguf::GGUF_MAGIC;
        assert_eq!((d_ff * d_model) % Q4_0_BLOCK_ELEMS, 0);
        assert_eq!((d_ff * d_model) % Q8_0_BLOCK_ELEMS, 0);
        let q4_blocks = (d_ff * d_model) / Q4_0_BLOCK_ELEMS;
        let q8_blocks = (d_ff * d_model) / Q8_0_BLOCK_ELEMS;
        let q4_block: Vec<u8> = {
            let mut b = half::f16::from_f32(0.1).to_bits().to_le_bytes().to_vec();
            b.extend_from_slice(&[0x88u8; 16]);
            b
        };
        let q8_block: Vec<u8> = {
            let mut b = half::f16::from_f32(0.1).to_bits().to_le_bytes().to_vec();
            b.extend_from_slice(&[1u8; 32]);
            b
        };
        let q4_blob = q4_block.repeat(q4_blocks);
        let q8_blob = q8_block.repeat(q8_blocks);

        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        // 7 per-layer dense + 3*num_experts FFN + 3 model-global tensors.
        let per_layer_tensors = 7 + 3 * num_experts;
        let total_tensors = 3 + per_layer_tensors;
        out.extend_from_slice(&(total_tensors as u64).to_le_bytes());
        let kvs: Vec<(&str, u32, Vec<u8>)> = vec![
            ("general.alignment", 4, 32u32.to_le_bytes().to_vec()),
            ("general.architecture", 8, lstring(b"llama")),
            ("general.name", 8, lstring(b"mixed-synth")),
            ("llama.block_count", 4, 1u32.to_le_bytes().to_vec()),
            (
                "llama.expert_count",
                4,
                (num_experts as u32).to_le_bytes().to_vec(),
            ),
            (
                "llama.embedding_length",
                4,
                (d_model as u32).to_le_bytes().to_vec(),
            ),
            (
                "llama.feed_forward_length",
                4,
                (d_ff as u32).to_le_bytes().to_vec(),
            ),
            ("llama.expert_used_count", 4, 2u32.to_le_bytes().to_vec()),
        ];
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, ty, payload) in &kvs {
            let kb = k.as_bytes();
            out.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            out.extend_from_slice(kb);
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(payload);
        }

        let mut tensor_data_blobs: Vec<Vec<u8>> = Vec::new();
        let mut cur_off: u64 = 0;
        let mut push_raw =
            |out: &mut Vec<u8>, name: &str, shape: &[u64], dtype: u32, mut blob: Vec<u8>| {
                let nb = name.as_bytes();
                out.extend_from_slice(&(nb.len() as u64).to_le_bytes());
                out.extend_from_slice(nb);
                out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                for d in shape {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&dtype.to_le_bytes());
                out.extend_from_slice(&cur_off.to_le_bytes());
                cur_off += blob.len() as u64;
                while cur_off % 32 != 0 {
                    blob.push(0);
                    cur_off += 1;
                }
                tensor_data_blobs.push(blob);
            };

        let mut attn = Vec::with_capacity(d_model * 4);
        for _ in 0..d_model {
            attn.extend_from_slice(&1.0f32.to_le_bytes());
        }
        push_raw(
            &mut out,
            "blk.0.attn_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            attn,
        );
        // Required F32 model-global + attention/norm/gate dense tensors (F2).
        let vocab = SYNTH_VOCAB;
        let (q_dim, kv_dim) = synth_qkv_dims(d_model);
        push_raw(
            &mut out,
            "token_embd.weight",
            &[d_model as u64, vocab as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.05; vocab * d_model]),
        );
        push_raw(
            &mut out,
            "output_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![1.0; d_model]),
        );
        push_raw(
            &mut out,
            "output.weight",
            &[d_model as u64, vocab as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.06; vocab * d_model]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_q.weight",
            &[d_model as u64, q_dim as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.1; d_model * q_dim]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_k.weight",
            &[d_model as u64, kv_dim as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.11; d_model * kv_dim]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_v.weight",
            &[d_model as u64, kv_dim as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.12; d_model * kv_dim]),
        );
        push_raw(
            &mut out,
            "blk.0.attn_output.weight",
            &[q_dim as u64, d_model as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.13; q_dim * d_model]),
        );
        push_raw(
            &mut out,
            "blk.0.ffn_norm.weight",
            &[d_model as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![1.0; d_model]),
        );
        push_raw(
            &mut out,
            "blk.0.ffn_gate_inp.weight",
            &[d_model as u64, num_experts as u64],
            ggml_dtype::F32,
            f32_vec_to_le_bytes(&vec![0.01; num_experts * d_model]),
        );
        for e in 0..num_experts {
            push_raw(
                &mut out,
                &format!("blk.0.ffn_gate.{e}.weight"),
                &[d_model as u64, d_ff as u64],
                ggml_dtype::Q4_0,
                q4_blob.clone(),
            );
            push_raw(
                &mut out,
                &format!("blk.0.ffn_up.{e}.weight"),
                &[d_model as u64, d_ff as u64],
                ggml_dtype::Q4_0,
                q4_blob.clone(),
            );
            push_raw(
                &mut out,
                &format!("blk.0.ffn_down.{e}.weight"),
                &[d_ff as u64, d_model as u64],
                ggml_dtype::Q8_0,
                q8_blob.clone(),
            );
        }
        while out.len() % 32 != 0 {
            out.push(0);
        }
        for blob in &tensor_data_blobs {
            out.extend_from_slice(blob);
        }
        out
    }

    /// Native pass-through must work for the **per-expert** layout
    /// (`blk.0.ffn_gate.{e}.weight`), not just the interleaved
    /// `_exps` layout. A `Q4_0` per-expert source must stay `Q4_0`
    /// (no F32 fallback) and write expert files of the raw quantised
    /// size — `3 * (d_ff*d_model/32) * 18` bytes plus the UTH header.
    #[test]
    fn extract_native_quant_per_expert_q4_0_passes_through() {
        let d_model = 64usize;
        let d_ff = 128usize;
        let num_experts = 3usize;
        let bytes = build_synth_gguf_q4_0_per_expert(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth_q4_0.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        // Eligibility check must accept the per-expert layout.
        assert_eq!(
            detect_native_quant_dtype(&gguf, 1, num_experts, d_model, d_ff),
            Some(WeightDtype::Q4_0),
            "per-expert Q4_0 layout must be eligible for native pass-through"
        );

        let out = tmp.join("native-per-expert");
        let report = extract_experts_from_source(
            &gguf,
            &out,
            1,
            num_experts,
            ExtractOptions {
                emit_uth: true,
                native_quant: true,
                experts_only: false,
                arch_override: None,
            },
        )
        .expect("extract");
        assert_eq!(report.experts_written, num_experts);
        assert_eq!(
            report.expert_dtype,
            WeightDtype::Q4_0,
            "per-expert Q4_0 source must pass through as Q4_0"
        );
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["dtype"], "q4_0");

        // Raw quantised payload size, much smaller than the F32 dequant
        // (which would be 3 * d_ff * d_model * 4 bytes).
        let q4_payload = 3 * (d_ff * d_model / Q4_0_BLOCK_ELEMS) * Q4_0_BLOCK_BYTES;
        let f32_payload = 3 * d_ff * d_model * 4;
        for e in 0..num_experts {
            let buf = fs::read(out.join(format!("expert_{e}.bin"))).unwrap();
            let (_, payload) = crate::tensor_header::TensorHeader::strip(&buf, DEFAULT_BLOCK_ALIGN);
            assert!(
                payload.len() >= q4_payload && payload.len() < f32_payload,
                "expert {e} payload {} should be raw Q4_0 ({q4_payload}), not F32 ({f32_payload})",
                payload.len()
            );
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extract_native_quant_mixed_per_expert_writes_uth2() {
        let d_model = 64usize;
        let d_ff = 128usize;
        let num_experts = 2usize;
        let bytes = build_synth_gguf_mixed_per_expert(d_model, d_ff, num_experts);
        let tmp = tempfile_dir();
        let gguf_path = tmp.join("synth_mixed.gguf");
        fs::write(&gguf_path, &bytes).unwrap();
        let gguf = GgufFile::open(&gguf_path).expect("parse");

        let scan = scan_expert_quant_layouts(&gguf, 1, num_experts, d_model, d_ff).unwrap();
        let report = scan.concise_report();
        assert!(report.contains("Q4_0/Q4_0/Q8_0: 1 layers"), "{report}");
        assert!(report.contains("gate: Q4_0=1"), "{report}");

        let out = tmp.join("native-mixed");
        let extract = extract_experts_from_source(
            &gguf,
            &out,
            1,
            num_experts,
            ExtractOptions {
                emit_uth: true,
                native_quant: true,
                experts_only: false,
                arch_override: None,
            },
        )
        .expect("extract mixed");
        assert_eq!(extract.experts_written, num_experts);
        assert_eq!(extract.expert_dtype, WeightDtype::Mixed);

        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["dtype"], "mixed");
        assert_eq!(meta["expert_layout_version"], 2);
        assert_eq!(
            meta["projection_dtype_histogram"]["q4_0/q4_0/q8_0"],
            num_experts
        );

        let q4_payload = (d_ff * d_model / Q4_0_BLOCK_ELEMS) * Q4_0_BLOCK_BYTES;
        let q8_payload = (d_ff * d_model / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
        let first = fs::read(out.join("expert_0.bin")).unwrap();
        let second = fs::read(out.join("expert_1.bin")).unwrap();
        assert_eq!(first.len(), second.len());
        assert_eq!(first.len() % DEFAULT_BLOCK_ALIGN, 0);
        let (header, payload) =
            crate::tensor_header::MixedExpertHeader::strip(&first, DEFAULT_BLOCK_ALIGN)
                .expect("UTH2");
        assert_eq!(header.gate.dtype.to_weight(), WeightDtype::Q4_0);
        assert_eq!(header.up.dtype.to_weight(), WeightDtype::Q4_0);
        assert_eq!(header.down.dtype.to_weight(), WeightDtype::Q8_0);
        assert_eq!(header.gate.len as usize, q4_payload);
        assert_eq!(header.up.offset as usize, q4_payload);
        assert_eq!(header.down.offset as usize, q4_payload * 2);
        assert_eq!(header.down.len as usize, q8_payload);
        assert!(payload.len() > q4_payload * 2 + q8_payload);
        let weights = crate::inference::OwnedExpertWeights::from_bytes_mixed_quant(
            payload, header, d_model, d_ff,
        )
        .expect("runtime mixed decode");
        assert_eq!(weights.gate.len(), d_ff * d_model);
        assert_eq!(weights.down.len(), d_model * d_ff);

        let pool = crate::buffer_pool::BufferPool::new(1, first.len(), DEFAULT_BLOCK_ALIGN);
        let mut buf = pool.try_acquire().expect("pooled buffer");
        buf.as_mut_slice()[..first.len()].copy_from_slice(&first);
        let resident =
            crate::expert_cache::ExpertResident::new_with_block_align(0, buf, DEFAULT_BLOCK_ALIGN);
        let x: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 + 1.0) / d_model as f32)
            .collect();
        let expected = weights.forward(&x);
        let before_mixed = crate::inference::mixed_expert_dispatches();
        let (_out, actual) =
            crate::inference::run_inference_mixed_quant(7, &resident, &x, d_model, d_ff)
                .expect("direct mixed inference");
        assert!(
            crate::inference::mixed_expert_dispatches() > before_mixed,
            "direct mixed inference should increment the mixed dispatch counter"
        );
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5,
                "mixed direct output mismatch at {idx}: actual={a}, expected={b}"
            );
        }
        let _ = fs::remove_dir_all(&tmp);
    }
}
