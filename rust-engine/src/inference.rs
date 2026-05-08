//! Real Mixtral / Llama-style **SwiGLU expert FFN** computed directly over
//! the bytes streamed from the SSD.
//!
//! Each expert file on disk is a flat blob of little-endian `f32` weights
//! laid out as three matrices, in this order:
//!
//! ```text
//!   gate_proj  : [d_ff   x d_model]   row-major
//!   up_proj    : [d_ff   x d_model]   row-major
//!   down_proj  : [d_model x d_ff  ]   row-major
//! ```
//!
//! The forward pass implemented here is the standard gated-MLP block used
//! by Mixtral, Llama-2/3, DeepSeek-V2/V3, Qwen-MoE, OLMoE, etc. — i.e. the
//! exact compute every routed expert in those models performs:
//!
//! ```text
//!   y = down_proj · ( silu(gate_proj · x)  ⊙  (up_proj · x) )
//! ```
//!
//! This is *the* per-expert kernel for sparse MoE transformers. Wiring it
//! here turns the engine from a pure I/O substrate into something that
//! actually executes a model-shaped computation over weights paged in from
//! the SSD — which is the entire point of streaming the weights in the
//! first place.
//!
//! ### Why pure Rust / scalar matmul?
//!
//! The point of this repository is to demonstrate that on modest hardware
//! you can keep the *active* parameter footprint streaming from SSD instead
//! of resident in DRAM. The compute kernel just has to be real enough to
//! exercise every byte that arrived from the drive. We therefore use a
//! straightforward scalar matmul + `silu` in `f32`, with no BLAS / SIMD
//! dependency: the resulting per-token cost is on the order of
//! `O(2 · d_model · d_ff)` MACs per expert, which is small enough on a
//! laptop that I/O remains observable but large enough that the compiler
//! cannot fold the read away. Swap this module for a `tch`/`candle`/`cudarc`
//! kernel when wiring real, larger weights — the I/O substrate around it
//! does not change.
//!
//! ### Why not mmap the bytes as `&[f32]` and hand them to a tensor lib?
//!
//! That is exactly the upgrade path. The buffers handed to this function
//! are page-aligned (the `O_DIRECT` invariant), so reinterpreting them as
//! `&[f32]` is sound — `align_of::<f32>() == 4` and we always allocate at
//! `≥ 4096`-byte alignment. See [`ExpertWeights::from_bytes`].

// The on-disk layout is documented as little-endian `f32`; we reinterpret
// the byte buffer as `&[f32]` in place, so the host's native endianness
// must match. Refuse to compile on big-endian targets rather than
// silently produce wrong weights.
#[cfg(not(target_endian = "little"))]
compile_error!(
    "inference module reinterprets on-disk little-endian f32 weights as \
     &[f32] in place; this only works on little-endian targets. Add an \
     explicit byte-swap conversion path before building for big-endian."
);

use crate::expert_cache::ExpertResident;
use std::fmt;

/// Hidden-state vector flowing through the FFN block (`d_model` floats).
pub type HiddenState = Vec<f32>;

/// Result of running one expert's FFN on a hidden state.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub expert_id: u32,
    /// 64-bit digest over the output activation. Cheap to log; lets us
    /// verify end-to-end that the bytes streamed from disk really did
    /// drive a deterministic computation.
    pub digest: u64,
    /// L2 norm of the output activation. A sanity signal that the matmul
    /// produced a non-degenerate result (NaN/Inf or all-zero would stand
    /// out immediately in the per-token logs).
    pub out_norm: f32,
}

/// Three-matrix view over a flat `&[f32]` blob with the layout described
/// at the top of this module.
pub struct ExpertWeights<'a> {
    pub d_model: usize,
    pub d_ff: usize,
    /// `gate_proj`, row-major `[d_ff x d_model]` (rows of `d_model` floats).
    pub gate: &'a [f32],
    /// `up_proj`, row-major `[d_ff x d_model]`.
    pub up: &'a [f32],
    /// `down_proj`, row-major `[d_model x d_ff]`.
    pub down: &'a [f32],
}

/// Number of `f32` weights an expert with these dimensions occupies.
///
/// Uses `saturating_mul` so absurdly large CLI-provided shapes don't
/// silently wrap in release mode — on overflow this returns
/// `usize::MAX`, which makes every downstream size check (the buffer
/// length comparison in [`ExpertWeights::from_bytes`], the
/// `expert_size` validation in `cmd_gen_data` / `generate_synthetic_experts`,
/// and the engine's startup check) reliably fail.
#[inline]
pub const fn expert_weight_count(d_model: usize, d_ff: usize) -> usize {
    // gate (d_ff * d_model) + up (d_ff * d_model) + down (d_model * d_ff)
    let one = d_model.saturating_mul(d_ff);
    one.saturating_mul(3)
}

/// Number of bytes an expert with these dimensions occupies on disk
/// (one `f32` is 4 bytes). Saturates to `usize::MAX` on overflow; see
/// [`expert_weight_count`].
#[inline]
pub const fn expert_weight_bytes(d_model: usize, d_ff: usize) -> usize {
    expert_weight_count(d_model, d_ff).saturating_mul(4)
}

/// Errors produced when reinterpreting a raw byte buffer as expert
/// weights. These are the conditions that previously aborted the run
/// via `assert!`; the run path now logs them and skips the offending
/// expert instead of panicking on corrupt on-disk data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertWeightsError {
    /// The buffer is shorter than the SwiGLU weight blob requires.
    BufferTooSmall {
        have: usize,
        need: usize,
        d_model: usize,
        d_ff: usize,
    },
    /// The buffer's start address is not aligned for `f32` access.
    /// `BufferPool` always allocates page-aligned buffers, so this is
    /// a contract violation rather than something a corrupt file can
    /// trigger — but we surface it as an error rather than panicking.
    Misaligned { addr: usize, required: usize },
}

impl fmt::Display for ExpertWeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpertWeightsError::BufferTooSmall { have, need, d_model, d_ff } => write!(
                f,
                "expert buffer too small: have {have} bytes, need {need} for \
                 d_model={d_model}, d_ff={d_ff}"
            ),
            ExpertWeightsError::Misaligned { addr, required } => write!(
                f,
                "expert buffer is not f32-aligned: addr=0x{addr:x}, required={required}"
            ),
        }
    }
}

impl std::error::Error for ExpertWeightsError {}

impl<'a> ExpertWeights<'a> {
    /// Reinterpret a page-aligned byte buffer as the three weight matrices.
    ///
    /// Returns [`ExpertWeightsError`] if `bytes` is shorter than
    /// `expert_weight_bytes(d_model, d_ff)` or does not start at an
    /// address aligned to `align_of::<f32>()` (4 bytes). `BufferPool`
    /// allocates with `≥ 4096`-byte alignment, so the alignment branch
    /// is defensive — but the size branch can fire on a truncated /
    /// corrupt on-disk file, and surfacing it as a `Result` lets the
    /// engine log and skip the expert instead of aborting the whole run.
    ///
    /// Any trailing bytes (e.g. padding so the file size is a multiple of
    /// `block_align` for `O_DIRECT`) are ignored.
    pub fn from_bytes(
        bytes: &'a [u8],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let need_floats = expert_weight_count(d_model, d_ff);
        let need_bytes = need_floats.saturating_mul(4);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }
        // Buffers from `BufferPool` are page-aligned; `f32` only needs 4.
        let required_align = std::mem::align_of::<f32>();
        let addr = bytes.as_ptr() as usize;
        if addr % required_align != 0 {
            return Err(ExpertWeightsError::Misaligned {
                addr,
                required: required_align,
            });
        }

        // SAFETY:
        // * `bytes.as_ptr()` is non-null and verified above to be aligned
        //   to `align_of::<f32>()`.
        // * `need_floats * 4 <= bytes.len()`, so the resulting slice stays
        //   inside the original allocation.
        // * `f32` has no validity invariants: every 4-byte sequence is a
        //   valid `f32` (NaN / subnormal / Inf are all well-defined values
        //   for the type system; whether the *model* tolerates them is
        //   `gen-data`'s responsibility, and it writes finite weights).
        // * The on-disk layout is little-endian `f32`; the
        //   `compile_error!` at the top of this module ensures the host
        //   is also little-endian, so a byte-for-byte reinterpretation
        //   is correct.
        // * The lifetime of the resulting `&[f32]` is tied to `bytes`, so
        //   the borrow checker prevents mutation while the view exists.
        let floats: &'a [f32] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), need_floats)
        };

        let gate_len = d_ff * d_model;
        let up_len = d_ff * d_model;
        let down_len = d_model * d_ff;
        let (gate, rest) = floats.split_at(gate_len);
        let (up, rest) = rest.split_at(up_len);
        let (down, _trailing) = rest.split_at(down_len);

        Ok(Self { d_model, d_ff, gate, up, down })
    }

    /// `down_proj · ( silu(gate_proj · x)  ⊙  (up_proj · x) )`
    ///
    /// Allocates one `Vec<f32>` for the gated intermediate (`d_ff`) and
    /// returns a `Vec<f32>` of length `d_model` for the FFN output.
    pub fn forward(&self, x: &[f32]) -> HiddenState {
        debug_assert_eq!(
            x.len(),
            self.d_model,
            "hidden state length must equal d_model"
        );

        // 1) Two parallel projections into d_ff: gate = W_g x, up = W_u x.
        //    Fuse them in the same loop so each row of (gate, up) reads
        //    `x` once — better cache behaviour than two separate matmuls.
        let mut gated = vec![0.0f32; self.d_ff];
        for i in 0..self.d_ff {
            let row = i * self.d_model;
            let g_row = &self.gate[row..row + self.d_model];
            let u_row = &self.up[row..row + self.d_model];
            // Manual sums; the compiler vectorises these well in release.
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for j in 0..self.d_model {
                g += g_row[j] * x[j];
                u += u_row[j] * x[j];
            }
            // SwiGLU intermediate: silu(g) * u.
            gated[i] = silu(g) * u;
        }

        // 2) Down projection: y = W_d · gated  -> length d_model.
        let mut y = vec![0.0f32; self.d_model];
        for i in 0..self.d_model {
            let row = i * self.d_ff;
            let d_row = &self.down[row..row + self.d_ff];
            let mut acc = 0.0f32;
            for j in 0..self.d_ff {
                acc += d_row[j] * gated[j];
            }
            y[i] = acc;
        }
        y
    }
}

/// SiLU / swish activation: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Generate the per-token hidden-state vector that flows into the FFN.
///
/// In a real model this would be the residual-stream activation produced
/// by the previous layer's attention block. Here we synthesise it
/// deterministically from `(token_idx, seed)` so a run is replayable and
/// every routed expert sees a non-trivial input. Values are bounded in
/// `[-1, 1]` so the matmul stays numerically tame.
pub fn synth_hidden_state(token_idx: u64, d_model: usize, seed: u64) -> HiddenState {
    let mut s = token_idx
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(seed.wrapping_mul(0xBF58476D1CE4E5B9));
    let mut out = Vec::with_capacity(d_model);
    for _ in 0..d_model {
        // splitmix64 step
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map the top 24 bits to a float in [-1, 1).
        let u = (z >> 40) as u32; // 24 bits
        let f = (u as f32) / ((1u32 << 23) as f32) - 1.0;
        out.push(f);
    }
    out
}

/// Run one expert's FFN on the hidden state. The buffer behind `resident`
/// is the bytes that came directly off the SSD via `O_DIRECT`. Returns
/// both the activation vector (for combining with other experts) and an
/// [`InferenceOutput`] summary suitable for logging, or an
/// [`ExpertWeightsError`] if the resident buffer can't be reinterpreted
/// as a valid SwiGLU weight blob (e.g. a truncated / corrupt file).
pub fn run_inference(
    token_idx: u64,
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = ExpertWeights::from_bytes(resident.data(), d_model, d_ff)?;
    let y = weights.forward(x);
    let mut sum_sq = 0.0f64;
    for &v in &y {
        sum_sq += (v as f64) * (v as f64);
    }
    let out_norm = (sum_sq.sqrt()) as f32;

    // Cheap, deterministic digest over (token_idx, expert_id, output bits).
    // Folded over `f32::to_bits` so the digest is exactly reproducible
    // bit-for-bit between runs.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut digest = FNV_OFFSET ^ token_idx ^ (resident.id as u64);
    for &v in &y {
        let bits = v.to_bits() as u64;
        digest ^= bits;
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    Ok((
        InferenceOutput { expert_id: resident.id, digest, out_norm },
        y,
    ))
}

/// Fold the top-K expert outputs together. Mixtral / Llama-MoE actually
/// take a **gated weighted sum** of expert outputs (the gate weights come
/// from the router's softmax). Since this engine mocks the router we just
/// average — the byte-for-byte path through every routed expert is what
/// matters for the I/O story.
pub fn combine_outputs(per_expert: &[HiddenState]) -> HiddenState {
    if per_expert.is_empty() {
        return Vec::new();
    }
    let d = per_expert[0].len();
    let mut out = vec![0.0f32; d];
    for vec in per_expert {
        debug_assert_eq!(vec.len(), d);
        for (o, v) in out.iter_mut().zip(vec.iter()) {
            *o += *v;
        }
    }
    let inv = 1.0 / per_expert.len() as f32;
    for o in out.iter_mut() {
        *o *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned_buffer::AlignedBuffer;

    /// Build a buffer holding deterministic, finite f32 weights so we can
    /// exercise `ExpertWeights::from_bytes` and `forward` without going
    /// through the full storage layer.
    fn make_weights_buffer(d_model: usize, d_ff: usize, fill: f32) -> AlignedBuffer {
        let bytes_needed = expert_weight_bytes(d_model, d_ff);
        // round up to a multiple of 4096 to satisfy AlignedBuffer invariants
        let padded = bytes_needed.div_ceil(4096) * 4096;
        let mut buf = AlignedBuffer::new(padded, 4096);
        for chunk in buf.as_mut_slice().chunks_exact_mut(4) {
            chunk.copy_from_slice(&fill.to_le_bytes());
        }
        buf
    }

    #[test]
    fn weights_view_partitions_buffer_correctly() {
        let d_model = 4;
        let d_ff = 8;
        let buf = make_weights_buffer(d_model, d_ff, 0.25);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        assert_eq!(w.gate.len(), d_ff * d_model);
        assert_eq!(w.up.len(), d_ff * d_model);
        assert_eq!(w.down.len(), d_model * d_ff);
        assert!(w.gate.iter().all(|&v| v == 0.25));
        assert!(w.down.iter().all(|&v| v == 0.25));
    }

    #[test]
    fn forward_produces_finite_output_of_correct_shape() {
        let d_model = 16;
        let d_ff = 32;
        // Use small weights so silu*x*x stays in a well-behaved range.
        let buf = make_weights_buffer(d_model, d_ff, 0.05);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(7, d_model, 1234);
        let y = w.forward(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()), "got non-finite output: {y:?}");
    }

    #[test]
    fn forward_is_deterministic() {
        let d_model = 8;
        let d_ff = 16;
        let buf = make_weights_buffer(d_model, d_ff, 0.1);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(42, d_model, 99);
        let a = w.forward(&x);
        let b = w.forward(&x);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_weights_yield_zero_output() {
        let d_model = 4;
        let d_ff = 4;
        let buf = make_weights_buffer(d_model, d_ff, 0.0);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(1, d_model, 1);
        let y = w.forward(&x);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn from_bytes_rejects_truncated_buffer() {
        let d_model = 4;
        let d_ff = 8;
        let buf = make_weights_buffer(d_model, d_ff, 0.0);
        let need = expert_weight_bytes(d_model, d_ff);
        // Hand it a slice that is one f32 short of the requirement.
        let truncated = &buf.as_slice()[..need - 4];
        let err = ExpertWeights::from_bytes(truncated, d_model, d_ff)
            .err()
            .expect("expected an error from a truncated buffer");
        match err {
            ExpertWeightsError::BufferTooSmall { have, need: n, .. } => {
                assert_eq!(have, need - 4);
                assert_eq!(n, need);
            }
            other => panic!("expected BufferTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn expert_weight_count_saturates_on_overflow() {
        // d_model * d_ff would overflow usize on every supported target.
        // saturating_mul must clamp to usize::MAX so downstream size
        // checks (which compare against finite buffer lengths) reliably
        // fail rather than silently wrap.
        let huge = usize::MAX;
        assert_eq!(expert_weight_count(huge, 2), usize::MAX);
        assert_eq!(expert_weight_bytes(huge, 2), usize::MAX);
    }

    #[test]
    fn synth_hidden_state_is_bounded_and_deterministic() {
        let a = synth_hidden_state(123, 32, 7);
        let b = synth_hidden_state(123, 32, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
        assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn silu_basic_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(x) -> x for large positive x
        assert!((silu(20.0) - 20.0).abs() < 1e-3);
        // silu(x) -> 0 for large negative x
        assert!(silu(-20.0).abs() < 1e-3);
    }

    #[test]
    fn combine_outputs_averages_correctly() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![3.0, 2.0, 1.0];
        let c = combine_outputs(&[a, b]);
        assert_eq!(c, vec![2.0, 2.0, 2.0]);
    }
}
