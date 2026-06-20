//! Scalar reference kernels — the always-available fallback that every
//! SIMD / AMX path is validated against.
//!
//! These are kept obviously-correct and `#[inline]`-friendly; the
//! optimizer is responsible for autovectorising them on toolchains
//! where the SIMD cargo features are not enabled.

/// `sum_i a[i] * b[i]`. Length checked in debug builds.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Fused symmetric-int8 dequant + dot: `sum_i scale * q[i] * x[i]`.
///
/// This is the *kernel* used inside the streaming int8 expert path:
/// the on-disk weights stay `i8` until they reach this fused loop,
/// so the per-byte SSD cost translates directly into MACs without a
/// stop in an owned `Vec<f32>` (compare the existing
/// `OwnedExpertWeights::from_bytes_int8` path which materialises one).
#[inline]
pub fn dequant_int8_dot(scale: f32, q: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), x.len());
    let mut acc = 0.0f32;
    for i in 0..q.len() {
        acc += (q[i] as f32) * x[i];
    }
    acc * scale
}

/// Fully-quantised int8×int8 dot with combined output scale: returns
/// `out_scale * sum_i (qw[i] * qx[i])`. The activation is **also**
/// int8 (each side carries its own per-tensor scale; `out_scale =
/// w_scale * x_scale` is folded in at the end). This is the
/// VNNI-friendly shape — see [`super::avx512::dot_int8_int8_avx512_vnni`]
/// — so the engine can route int8 activations through
/// `_mm512_dpbusd_epi32` and only spend one f32 multiply per dot at
/// the very end. The scalar reference here is the validation oracle.
#[inline]
pub fn dot_int8_int8(out_scale: f32, qw: &[i8], qx: &[i8]) -> f32 {
    debug_assert_eq!(qw.len(), qx.len());
    // Accumulate in i32 (saturating at i32::MAX is impossible for any
    // realistic length: max |qw[i] * qx[i]| = 127 * 128 = 16,256, so
    // even a 1 M-element row stays well under i32::MAX).
    let mut acc: i32 = 0;
    for i in 0..qw.len() {
        acc += (qw[i] as i32) * (qx[i] as i32);
    }
    (acc as f32) * out_scale
}

/// Reference SwiGLU FFN inner stage: `y[i] = silu(gate_w[i]·x) * (up_w[i]·x)`.
///
/// Used as the parity oracle for [`super::avx512::swiglu_f32_avx512`].
/// Writes into the caller-provided `y` (no allocation).
#[inline]
pub fn swiglu_f32(
    gate_w: &[f32],
    up_w: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    swiglu_f32_clamped(gate_w, up_w, x, rows, cols, y, None)
}

/// SwiGLU FFN inner stage with an optional gate clamp:
/// `y[i] = silu(clamp(gate_w[i]·x)) * (up_w[i]·x)`.
///
/// When `swiglu_limit` is `Some(limit)` the gate value `g` is clamped to
/// `[-limit, limit]` before the sigmoid — this is the GPT-OSS
/// `swiglu_limit` (e.g. `7.0`) behaviour. `None` reproduces the plain
/// SwiGLU used by every other architecture, with no per-element branch in
/// the unclamped loop. Writes into the caller-provided `y` (no allocation).
#[inline]
pub fn swiglu_f32_clamped(
    gate_w: &[f32],
    up_w: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
    swiglu_limit: Option<f32>,
) {
    debug_assert_eq!(gate_w.len(), rows * cols);
    debug_assert_eq!(up_w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    match swiglu_limit {
        Some(limit) => {
            for row in 0..rows {
                let off = row * cols;
                let g = dot_f32(&gate_w[off..off + cols], x).clamp(-limit, limit);
                let u = dot_f32(&up_w[off..off + cols], x);
                let silu_g = g / (1.0 + (-g).exp());
                y[row] = silu_g * u;
            }
        }
        None => {
            for row in 0..rows {
                let off = row * cols;
                let g = dot_f32(&gate_w[off..off + cols], x);
                let u = dot_f32(&up_w[off..off + cols], x);
                let silu_g = g / (1.0 + (-g).exp());
                y[row] = silu_g * u;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot_f32(&a, &b), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn dequant_int8_dot_basic() {
        let q = [1i8, -2, 3];
        let x = [1.0f32, 1.0, 1.0];
        assert!((dequant_int8_dot(0.5, &q, &x) - (1.0 - 2.0 + 3.0) * 0.5).abs() < 1e-6);
    }

    #[test]
    fn swiglu_f32_applies_limit() {
        // gate value of 100.0 should be clamped to 7.0 before silu
        let gate_w = [1.0f32]; // dot with x=[100.0] → g=100.0
        let up_w = [1.0f32]; // u = 100.0
        let x = [100.0f32];
        let mut y = [0.0f32];
        swiglu_f32_clamped(&gate_w, &up_w, &x, 1, 1, &mut y, Some(7.0));
        let g_clamped = 7.0f32;
        let silu_7 = g_clamped / (1.0 + (-g_clamped).exp());
        let expected = silu_7 * 100.0;
        assert!((y[0] - expected).abs() < 1e-5, "got {}, expected {}", y[0], expected);
    }

    #[test]
    fn swiglu_f32_no_limit_matches_reference() {
        let gate_w = [2.0f32];
        let up_w = [3.0f32];
        let x = [1.0f32];
        let g = gate_w[0] * x[0];
        let u = up_w[0] * x[0];
        let silu_g = g / (1.0 + (-g).exp());
        let expected = silu_g * u;

        let mut y1 = [0.0f32];
        swiglu_f32_clamped(&gate_w, &up_w, &x, 1, 1, &mut y1, None);
        assert!((y1[0] - expected).abs() < 1e-7);

        let mut y2 = [0.0f32];
        swiglu_f32(&gate_w, &up_w, &x, 1, 1, &mut y2);
        assert!((y2[0] - expected).abs() < 1e-7);
    }
}
