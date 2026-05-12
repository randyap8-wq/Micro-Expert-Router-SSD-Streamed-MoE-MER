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

use crate::gguf::{ggml_dtype, GgufFile, GgufSource, GgufTensorInfo};
use crate::inference::{
    dequantize_f16_to_f32, expert_weight_bytes_for, WeightDtype,
};
use crate::tensor_header::TensorHeader;
use serde::Serialize;
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
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self { emit_uth: true }
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

    let num_layers = if num_layers_hint > 0 {
        num_layers_hint
    } else {
        gguf.metadata()
            .get("llama.block_count")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF missing `llama.block_count`; pass --num-layers explicitly",
                )
            })? as usize
    };
    let num_experts = if num_experts_hint > 0 {
        num_experts_hint
    } else {
        gguf.metadata()
            .get("llama.expert_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    };
    if num_experts == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF has no `llama.expert_count` and no --num-experts override",
        ));
    }

    // Required shape parameters.
    let d_model = gguf
        .metadata()
        .get("llama.embedding_length")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "GGUF missing llama.embedding_length")
        })? as usize;
    let d_ff = gguf
        .metadata()
        .get("llama.feed_forward_length")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "GGUF missing llama.feed_forward_length")
        })? as usize;
    let top_k = gguf
        .metadata()
        .get("llama.expert_used_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;

    info!(
        num_layers,
        num_experts,
        d_model,
        d_ff,
        top_k,
        emit_uth = opts.emit_uth,
        "gguf-convert: extracting experts"
    );

    let mut report = ExtractionReport {
        experts_written: 0,
        dense_written: 0,
        skipped: 0,
        total_bytes: 0,
        expert_dtype: WeightDtype::F32,
        d_model,
        d_ff,
        num_experts_per_layer: num_experts,
        num_layers,
    };

    // Walk layers and emit per-expert files. Output dtype is always F32
    // here: the GGUF block-quant streams pack all experts together, so
    // slicing them at byte granularity isn't possible without unpacking.
    // Dequantising at extract time gives the engine the simplest input
    // and preserves accuracy.
    //
    // When `opts.emit_uth` is set, each expert file is prefixed with a
    // 64-byte U.T.H., page-padded to DEFAULT_BLOCK_ALIGN so the weight
    // payload still starts at a 4 KiB boundary. The total file size
    // therefore grows by exactly one block (4 KiB) per expert.
    let payload_size = align_up(
        expert_weight_bytes_for(d_model, d_ff, WeightDtype::F32),
        DEFAULT_BLOCK_ALIGN,
    );
    let header_overhead = if opts.emit_uth { DEFAULT_BLOCK_ALIGN } else { 0 };
    let expert_size = payload_size + header_overhead;

    for layer in 0..num_layers {
        info!(layer, "extracting layer experts");
        let per_expert =
            load_layer_expert_matrices(gguf, layer, num_experts, d_model, d_ff)?;
        for (e, (gate, up, down)) in per_expert.into_iter().enumerate() {
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

    // Dense weights. The engine's `RealModel::from_dir` uses its own
    // file-name convention (`embed.bin`, `attn_<L>_q.bin`, …), so we
    // write *both* the gist-mandated llama.cpp-style names and the
    // engine's native names. The duplicate write is small relative to
    // the expert files and means the converter satisfies both APIs.
    let dense_specs: Vec<(&str, &[&str], Vec<u64>)> = vec![
        // (gguf_name, [engine_aliases...], expected_shape_innermost_first)
        ("token_embd.weight", &["embedding.bin", "embed.bin"], vec![d_model as u64, 0]),
        ("output_norm.weight", &["final_norm.bin", "final_rms.bin"], vec![d_model as u64]),
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
            vec![format!("layer_{layer}_q.bin"), format!("attn_{layer}_q.bin")],
        )?;
        emit(
            format!("blk.{layer}.attn_k.weight"),
            vec![format!("layer_{layer}_k.bin"), format!("attn_{layer}_k.bin")],
        )?;
        emit(
            format!("blk.{layer}.attn_v.weight"),
            vec![format!("layer_{layer}_v.bin"), format!("attn_{layer}_v.bin")],
        )?;
        emit(
            format!("blk.{layer}.attn_output.weight"),
            vec![format!("layer_{layer}_o.bin"), format!("attn_{layer}_o.bin")],
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
        num_experts: num_experts * num_layers,
        top_k,
        d_model,
        d_ff,
        expert_size,
        block_align: DEFAULT_BLOCK_ALIGN,
        dtype: "f32",
        weight_layout: "gate_proj || up_proj || down_proj (row-major)",
        num_layers,
        num_experts_per_layer: num_experts,
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
    bytes_to_f32(&data, info.ggml_dtype, info.elem_count() as usize, &info.name)
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
        warn!(layer, "no expert weight tensor found in GGUF; emitting zero blobs");
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
    let info = gguf.tensor_info(name).cloned().ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing"))
    })?;
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
        let gate = slice_expert_matrix(
            gguf,
            &interleaved_gate,
            expert,
            num_experts,
            d_ff,
            d_model,
        )?;
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
        warn!(layer, expert, "no expert weight tensor found in GGUF; emitting zero blob");
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
    let info = gguf.tensor_info(name).cloned().ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing"))
    })?;
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
    let info = gguf.tensor_info(name).cloned().ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("tensor {name} missing"))
    })?;
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
            assert_eq!(buf.len() % DEFAULT_BLOCK_ALIGN, 0, "{path:?} not page-padded");
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
        let _ = extract_experts_from_source(
            &gguf,
            &out,
            1,
            num_experts,
            ExtractOptions::default(),
        )
        .expect("extract");
        for e in 0..num_experts {
            let path = out.join(format!("expert_{e}.bin"));
            let buf = fs::read(&path).unwrap();
            let header = crate::tensor_header::TensorHeader::probe(&buf)
                .expect("U.T.H. must be present");
            assert_eq!(header.dtype.to_weight(), WeightDtype::F32);
            assert_eq!(header.shape[0] as usize, d_ff);
            // After stripping, the first byte must be at a 4 KiB offset.
            let (_, payload) = crate::tensor_header::TensorHeader::strip(
                &buf,
                DEFAULT_BLOCK_ALIGN,
            );
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
            ExtractOptions { emit_uth: false },
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
            let (h, payload) = crate::tensor_header::TensorHeader::strip(
                &buf,
                DEFAULT_BLOCK_ALIGN,
            );
            assert!(h.is_none());
            assert_eq!(payload.len(), buf.len());
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    fn tempfile_dir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        p.push(format!("gguf-loader-test-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn build_synth_gguf(d_model: usize, d_ff: usize, num_experts: usize) -> Vec<u8> {
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
            ("general.architecture", 8, lstring(b"llama")),
            ("general.name", 8, lstring(b"synth")),
            ("llama.block_count", 4, 1u32.to_le_bytes().to_vec()),
            ("llama.expert_count", 4, (num_experts as u32).to_le_bytes().to_vec()),
            ("llama.embedding_length", 4, (d_model as u32).to_le_bytes().to_vec()),
            ("llama.feed_forward_length", 4, (d_ff as u32).to_le_bytes().to_vec()),
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

    fn lstring(s: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + s.len());
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s);
        out
    }
}
