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
//! For F32 and F16 source dtypes the bytes are repacked into that layout
//! directly; for Q4_0 / Q4_K we currently dequantise to F32 because the
//! GGUF stores each (gate, up, down) tensor as a single block stream
//! that doesn't slice cleanly along the expert axis at the byte level.
//! This preserves the **engine's** on-disk format invariants
//! (`expert_size` is the same for every expert, the file is page-padded,
//! and `metadata.json::dtype` correctly describes the contents).

use crate::gguf::{ggml_dtype, GgufFile, GgufSource, GgufTensorInfo, GgufValue};
use crate::inference::{
    dequantize_f16_to_f32, expert_weight_bytes_for, projection_weight_bytes_for, WeightDtype,
    Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q5K_BLOCK_BYTES,
    Q5K_BLOCK_ELEMS, Q6K_BLOCK_BYTES, Q6K_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use crate::tensor_header::{MixedExpertHeader, ProjectionRange, TensorHeader, UthDtypeId};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use tracing::{info, warn};

/// Summary of what `extract_experts_from_gguf` wrote.
#[derive(Debug, Clone)]
pub struct ExtractionReport {
    /// Number of expert files written.
    pub experts_written: usize,
    /// Number of dense tensors written (`embed.bin`, `final_rms.bin`,
    /// per-layer attention projections, etc.).
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
    model: &'a str,
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
    block_align: usize,
    dtype: &'a str,
    weight_layout: &'a str,
    num_layers: usize,
    num_experts_per_layer: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    expert_layout_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    projection_dtype_histogram: Option<BTreeMap<String, usize>>,
}

/// Block alignment used for every expert file written by this loader.
/// Mirrors the engine's default in `main.rs`.
pub const DEFAULT_BLOCK_ALIGN: usize = 4096;

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
    /// Layers / models that don't satisfy the alignment precondition
    /// **fall back to F32 dequant** transparently and the converter
    /// keeps making forward progress; the report's `expert_dtype`
    /// records what was actually written so downstream tooling /
    /// `metadata.json` always describe the on-disk reality.
    pub native_quant: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            emit_uth: true,
            native_quant: false,
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

    // Dense weights. The engine's `RealModel::from_dir` uses its own
    // file-name convention (`embed.bin`, `attn_<L>_q.bin`, …), so we
    // write *both* the gist-mandated llama.cpp-style names and the
    // engine's native names. The duplicate write is small relative to
    // the expert files and means the converter satisfies both APIs.
    let dense_specs: Vec<(&str, &[&str], Vec<u64>)> = vec![
        // (gguf_name, [engine_aliases...], expected_shape_innermost_first)
        (
            "token_embd.weight",
            &["embedding.bin", "embed.bin"],
            vec![d_model as u64, 0],
        ),
        (
            "output_norm.weight",
            &["final_norm.bin", "final_rms.bin"],
            vec![d_model as u64],
        ),
        ("output.weight", &["lm_head.bin"], vec![d_model as u64, 0]),
    ];
    for (gname, aliases, _) in &dense_specs {
        if let Some(info) = gguf.tensor_info(gname).cloned() {
            match dense_tensor_to_f32(gguf, &info) {
                Ok(f32s) => {
                    for alias in *aliases {
                        let path = out_dir.join(alias);
                        let bytes = f32_vec_to_le_bytes(&f32s);
                        write_file(&path, &bytes)?;
                        report.dense_written += 1;
                        report.total_bytes += bytes.len() as u64;
                    }
                }
                Err(e) => {
                    warn!(name = gname, error = %e, "skipping dense tensor");
                    report.skipped += 1;
                }
            }
        } else {
            report.skipped += 1;
        }
    }

    // Per-layer dense tensors.
    for layer in 0..num_layers {
        let mut emit = |gname: String, aliases: Vec<String>| -> io::Result<()> {
            if let Some(info) = gguf.tensor_info(&gname).cloned() {
                match dense_tensor_to_f32(gguf, &info) {
                    Ok(f32s) => {
                        let bytes = f32_vec_to_le_bytes(&f32s);
                        for alias in &aliases {
                            let path = out_dir.join(alias);
                            write_file(&path, &bytes)?;
                            report.dense_written += 1;
                            report.total_bytes += bytes.len() as u64;
                        }
                    }
                    Err(e) => {
                        warn!(name = gname, error = %e, "skipping per-layer dense tensor");
                        report.skipped += 1;
                    }
                }
            } else {
                report.skipped += 1;
            }
            Ok(())
        };
        emit(
            format!("blk.{layer}.attn_q.weight"),
            vec![
                format!("layer_{layer}_q.bin"),
                format!("attn_{layer}_q.bin"),
            ],
        )?;
        emit(
            format!("blk.{layer}.attn_k.weight"),
            vec![
                format!("layer_{layer}_k.bin"),
                format!("attn_{layer}_k.bin"),
            ],
        )?;
        emit(
            format!("blk.{layer}.attn_v.weight"),
            vec![
                format!("layer_{layer}_v.bin"),
                format!("attn_{layer}_v.bin"),
            ],
        )?;
        emit(
            format!("blk.{layer}.attn_output.weight"),
            vec![
                format!("layer_{layer}_o.bin"),
                format!("attn_{layer}_o.bin"),
            ],
        )?;
        emit(
            format!("blk.{layer}.attn_norm.weight"),
            vec![
                format!("layer_{layer}_attn_norm.bin"),
                format!("rms_attn_{layer}.bin"),
            ],
        )?;
        emit(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![
                format!("layer_{layer}_ffn_norm.bin"),
                format!("rms_moe_{layer}.bin"),
            ],
        )?;
        emit(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![format!("gate_{layer}.bin")],
        )?;
        // Qwen2-MoE-style "shared experts" — dense FFN tensors applied to
        // *every* token in addition to the routed experts. They are stored
        // under the `_shexp` suffix and were previously dropped, leaving the
        // converted engine missing weights. Emit them as dense `.bin` files
        // (both the gguf-style name and an engine-friendly alias). Files are
        // only written when the tensor exists, so non-MoE / no-shared-expert
        // architectures (e.g. Mixtral) are unaffected.
        emit(
            format!("blk.{layer}.ffn_gate_shexp.weight"),
            vec![format!("layer_{layer}_shexp_gate.bin")],
        )?;
        emit(
            format!("blk.{layer}.ffn_up_shexp.weight"),
            vec![format!("layer_{layer}_shexp_up.bin")],
        )?;
        emit(
            format!("blk.{layer}.ffn_down_shexp.weight"),
            vec![format!("layer_{layer}_shexp_down.bin")],
        )?;
        emit(
            format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
            vec![format!("layer_{layer}_shexp_gate_inp.bin")],
        )?;
    }

    // metadata.json
    let model_name = gguf
        .metadata()
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("gguf-extracted");
    let meta = ExtractedMetadata {
        model: model_name,
        layer: if num_layers == 1 { 0 } else { -1 },
        num_experts: total_experts,
        top_k,
        d_model,
        d_ff,
        expert_size,
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
/// Fallbacks mirror `load_expert_matrices`:
/// * If a layer has neither interleaved nor per-expert tensors,
///   `num_experts` deterministic-zero triples are produced (the engine
///   reseeds these at runtime).
/// * If a tensor uses a dtype `bytes_to_f32` doesn't understand yet
///   (e.g. some Q6_K/Q8_0 variants), the whole layer falls back to
///   zero blobs — matching the documented "recognised for sizing only"
///   behaviour. The convert never aborts on a single unsupported
///   layer.
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

    let zero_layer = |reason: &str| {
        warn!(layer, reason, "emitting zero blob for layer experts");
        (0..num_experts)
            .map(|_| {
                (
                    vec![0.0f32; d_ff * d_model],
                    vec![0.0f32; d_ff * d_model],
                    vec![0.0f32; d_model * d_ff],
                )
            })
            .collect::<Vec<_>>()
    };

    if gguf.has_tensor(&interleaved_gate_name) {
        // Decode each interleaved tensor exactly once and reuse it for
        // all experts in the layer.
        let gate_all = match dense_layer_tensor_f32(
            gguf,
            &interleaved_gate_name,
            num_experts * d_ff * d_model,
        ) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                return Ok(zero_layer("unsupported dtype in interleaved gate tensor"));
            }
            Err(err) => return Err(err),
        };
        let up_all = match dense_layer_tensor_f32(
            gguf,
            &interleaved_up_name,
            num_experts * d_ff * d_model,
        ) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                return Ok(zero_layer("unsupported dtype in interleaved up tensor"));
            }
            Err(err) => return Err(err),
        };
        let down_all = match dense_layer_tensor_f32(
            gguf,
            &interleaved_down_name,
            num_experts * d_model * d_ff,
        ) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                return Ok(zero_layer("unsupported dtype in interleaved down tensor"));
            }
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
        // separate, so a layer-level cache buys nothing — just dispatch
        // to `load_expert_matrices`, catching Unsupported per expert.
        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            match load_expert_matrices(gguf, layer, e, num_experts, d_model, d_ff) {
                Ok(triple) => out.push(triple),
                Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                    warn!(
                        layer,
                        expert = e,
                        "unsupported dtype in per-expert tensor; emitting zero blob"
                    );
                    out.push((
                        vec![0.0f32; d_ff * d_model],
                        vec![0.0f32; d_ff * d_model],
                        vec![0.0f32; d_model * d_ff],
                    ));
                }
                Err(err) => return Err(err),
            }
        }
        Ok(out)
    } else {
        warn!(
            layer,
            "no expert weight tensor found in GGUF; emitting zero blobs"
        );
        Ok(zero_layer("no expert weight tensor"))
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
        // No expert tensors present — fall back to deterministic-zero
        // matrices so the rest of the pipeline produces a syntactically
        // valid expert file (the engine will fall back to seeded init
        // at run time anyway when the file is shaped right but zeroed).
        warn!(
            layer,
            expert, "no expert weight tensor found in GGUF; emitting zero blob"
        );
        Ok((
            vec![0.0; d_ff * d_model],
            vec![0.0; d_ff * d_model],
            vec![0.0; d_model * d_ff],
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
        build_synth_gguf_arch_ext(d_model, d_ff, num_experts, arch, None)
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
    ) -> Vec<u8> {
        let d_ff = expert_d_ff;
        use crate::gguf::GGUF_MAGIC;
        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes()); // version
                                                    // tensor_count: per layer we have 7 dense + 3 * num_experts FFN
                                                    // Actually we'll only put the FFN per-expert tensors and 1 attn_norm.
        let per_layer_tensors = 1 /* attn_norm */ + 3 * num_experts;
        out.extend_from_slice(&(per_layer_tensors as u64).to_le_bytes());
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
        let per_layer_tensors = 1 /* attn_norm */ + 3 * num_experts;
        out.extend_from_slice(&(per_layer_tensors as u64).to_le_bytes());
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
        let per_layer_tensors = 1 + 3 * num_experts;
        out.extend_from_slice(&(per_layer_tensors as u64).to_le_bytes());
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
        let _ = fs::remove_dir_all(&tmp);
    }
}
