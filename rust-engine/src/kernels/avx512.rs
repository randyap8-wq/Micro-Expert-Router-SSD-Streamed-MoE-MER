//! AVX-512 kernels — int8 fused dequant-dot + f32 reduction.
//!
//! Compiled only when the `avx512` cargo feature is enabled **and** the
//! target arch is `x86_64`. Every entry point is `unsafe` because it
//! relies on `#[target_feature(enable = "avx512f,avx512bw")]`; callers
//! gate dispatch on the runtime probe in [`super::detect`] so these
//! routines never execute on a CPU that doesn't support them.
//!
//! Each kernel returns a value bit-equivalent to its `scalar` reference
//! up to floating-point reduction reordering (about 1 ULP per ~32-wide
//! accumulator, well under the engine's existing inference-vs-reference
//! tolerance — the `dot_f32_matches_scalar_reference` test in
//! [`super::tests`] enforces a 1e-3 envelope which is what the rest of
//! the engine uses).

#![cfg(all(feature = "avx512", target_arch = "x86_64"))]

use std::arch::x86_64::*;

/// AVX-512 f32 dot product.
///
/// Inner loop is 4× unrolled with independent accumulators
/// (`_mm512_fmadd_ps` chains break the latency-bound dependency from
/// a single accumulator, so the four FMAs can issue back-to-back and
/// retire one per cycle on Skylake-X / Ice Lake / Sapphire Rapids).
/// The 16-wide and scalar tails handle the < 64-lane remainder.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx512f`. The dispatcher in
/// [`super::dot_f32`] checks this exactly once at startup.
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // Four independent f32x16 accumulators — 64 lanes per iteration,
    // breaks the FMA latency chain on the issue port.
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 64 <= n {
        let pa = a.as_ptr().add(i);
        let pb = b.as_ptr().add(i);
        let a0 = _mm512_loadu_ps(pa);
        let a1 = _mm512_loadu_ps(pa.add(16));
        let a2 = _mm512_loadu_ps(pa.add(32));
        let a3 = _mm512_loadu_ps(pa.add(48));
        let b0 = _mm512_loadu_ps(pb);
        let b1 = _mm512_loadu_ps(pb.add(16));
        let b2 = _mm512_loadu_ps(pb.add(32));
        let b3 = _mm512_loadu_ps(pb.add(48));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        i += 64;
    }
    // Fold the four accumulators down to one.
    let acc01 = _mm512_add_ps(acc0, acc1);
    let acc23 = _mm512_add_ps(acc2, acc3);
    let mut acc = _mm512_add_ps(acc01, acc23);
    // 16-wide tail for the [0, 64) lanes that didn't fit the unrolled body.
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    // Final < 16-lane scalar tail.
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Fused SwiGLU FFN inner stage: `y[i] = silu(gate_w[i]·x) * (up_w[i]·x)`.
///
/// `gate_w` and `up_w` are row-major `[rows × cols]` matrices, `x` is a
/// `cols`-vector of activations, `y` is the `rows`-vector the caller
/// owns. The kernel:
///
/// * does **no** allocation — results are written in place into `y`;
/// * runs the gate and up row dots back-to-back on the same `x`
///   slice, so `x` stays hot in L1 across both projections (the
///   "single pass to minimize cache-line bounces" the gist asks for);
/// * fuses SiLU(`gate`) * `up` into a single scalar combine after
///   the two row reductions, so the gate intermediate never
///   materialises in a separate `Vec<f32>`.
///
/// SiLU is computed scalar (`x / (1 + e^-x)`) on the reduced row
/// scalar — one transcendental per row, not per lane — which matches
/// what `ScalarBackend::silu_inplace` does and keeps this kernel
/// bit-equivalent to the reference within a 1 ULP envelope.
///
/// # Safety
/// Caller must guarantee:
/// * the CPU supports `avx512f` (the dispatcher checks at startup);
/// * `gate_w.len() == rows * cols`, `up_w.len() == rows * cols`,
///   `x.len() == cols`, `y.len() == rows`.
#[target_feature(enable = "avx512f")]
pub unsafe fn swiglu_f32_avx512(
    gate_w: &[f32],
    up_w: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(gate_w.len(), rows * cols);
    debug_assert_eq!(up_w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    for row in 0..rows {
        let off = row * cols;
        // Two row dots over the same `x` keep activations hot in L1.
        let g = dot_f32_avx512(
            std::slice::from_raw_parts(gate_w.as_ptr().add(off), cols),
            x,
        );
        let u = dot_f32_avx512(
            std::slice::from_raw_parts(up_w.as_ptr().add(off), cols),
            x,
        );
        // SiLU(g) * u, fused. One transcendental per row.
        let silu_g = g / (1.0 + (-g).exp());
        y[row] = silu_g * u;
    }
}

/// Fused symmetric-int8 dequant + dot. Each iteration loads 16 i8
/// weights, sign-extends to i32, converts to f32, multiplies by 16
/// f32 activations, and FMAs into an f32 accumulator. The
/// per-tensor scale is folded in *once* on the final reduction.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx512f` + `avx512bw`
/// (we use `_mm512_cvtepi8_epi32` which is part of AVX-512BW).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dequant_int8_dot_avx512(scale: f32, q: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), x.len());
    let n = q.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        // Load 16 packed i8 → sign-extend to 16x i32 → convert to f32.
        let q_i8 = _mm_loadu_si128(q.as_ptr().add(i) as *const __m128i);
        let q_i32 = _mm512_cvtepi8_epi32(q_i8);
        let q_f32 = _mm512_cvtepi32_ps(q_i32);
        let x_f32 = _mm512_loadu_ps(x.as_ptr().add(i));
        acc = _mm512_fmadd_ps(q_f32, x_f32, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    while i < n {
        sum += (q[i] as f32) * x[i];
        i += 1;
    }
    sum * scale
}

/// **AVX-512 VNNI int8×int8 dot — gist Part 2, fix #8.** Uses
/// `_mm512_dpbusd_epi32` to compute `sum_{4i+j} a.u8[4i+j] * b.i8[4i+j]`
/// across 64-byte lanes in a single instruction, accumulating into
/// i32. The whole inner reduction stays in integer registers; the only
/// f32 multiply is the final per-tensor scale fold at the end. That
/// removes the `cvtepi32_ps` round-trip the FP-accumulator path
/// [`dequant_int8_dot_avx512`] pays on every 16-byte chunk.
///
/// **Why the bias trick.** Plain AVX-512 VNNI only ships
/// `VPDPBUSD` (Bytes Unsigned × Signed → Doubleword), not the
/// `VPDPBSSD` variant added later in AVX-VNNI / AVX10. We bias the
/// weight operand `qw` from `i8 ∈ [-128, 127]` into `u8 ∈ [0, 255]`
/// by adding 128, then correct for the bias on the activation sum:
///
///   ```text
///   qw[i] * qx[i] = (qw[i] + 128 - 128) * qx[i]
///                 = ((qw[i] + 128) as u8) * qx[i] - 128 * qx[i]
///   ```
///
/// so the full dot reduces to one `dpbusd` reduction plus
/// `-128 * sum(qx)`. The activation sum is a cheap `sad_epu8` /
/// horizontal-add across the same bytes, computed once per call.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx512f + avx512bw + avx512vnni`.
/// The dispatcher in [`super::dot_int8_int8`] checks the runtime probe
/// before delegating.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn dot_int8_int8_avx512_vnni(out_scale: f32, qw: &[i8], qx: &[i8]) -> f32 {
    debug_assert_eq!(qw.len(), qx.len());
    let n = qw.len();
    // Two independent VNNI accumulators break the issue-port
    // dependency chain on Sapphire Rapids (one `vpdpbusd` per cycle
    // per port × 2 ports). The bias-correction sum is folded in
    // separately at the end.
    let mut vnni_acc0 = _mm512_setzero_si512();
    let mut vnni_acc1 = _mm512_setzero_si512();
    // Bias offset: 0x80 (128) added to every weight byte to lift
    // `i8 ∈ [-128, 127]` into `u8 ∈ [0, 255]`. Stored as a vector
    // constant so the bias add is a single `_mm512_add_epi8`.
    let bias_v = _mm512_set1_epi8(-128i8); // XOR sign bit equiv to +128 mod 256
    // Activation sum is computed in i32 by widening i8 → i32 per
    // lane. `sad_epu8` would only work on u8, and `madd_epi16` needs
    // a constant 1-vector, so we keep this path explicit.
    let mut act_sum_acc = _mm512_setzero_si512();
    let one_i16 = _mm512_set1_epi16(1);
    let mut i = 0usize;
    while i + 64 <= n {
        // Load 64 weights as i8, XOR the sign bit to convert to the
        // u8 bias-shifted operand expected by `vpdpbusd`.
        let qw_v = _mm512_loadu_si512(qw.as_ptr().add(i) as *const __m512i);
        let qw_u8 = _mm512_xor_si512(qw_v, bias_v); // = (qw + 128) as u8
        let qx_v = _mm512_loadu_si512(qx.as_ptr().add(i) as *const __m512i);
        // VPDPBUSD: 4-byte groups of (u8 × i8) → i32. Splitting the
        // 64-byte chunk across two accumulators (low/high halves
        // interleaved) keeps the issue port busy.
        vnni_acc0 = _mm512_dpbusd_epi32(vnni_acc0, qw_u8, qx_v);
        // Activation byte-sum: pmaddubsw with a constant 1-vector
        // widens i8×u8 → i16, then sum pairs into i32 via madd_epi16.
        // `pmaddubsw` expects (u8, i8) — we want sum of qx[i].i8, so
        // multiply by all-ones (u8 = 1) to keep the sign.
        let ones_u8 = _mm512_set1_epi8(1);
        let prod_i16 = _mm512_maddubs_epi16(ones_u8, qx_v);
        act_sum_acc = _mm512_add_epi32(act_sum_acc, _mm512_madd_epi16(prod_i16, one_i16));
        // Use the second accumulator on alternating iterations so
        // the issue ports don't stall waiting on a single dependency
        // chain. (Cosmetic on out-of-order cores; explicit on
        // in-order issue.)
        std::mem::swap(&mut vnni_acc0, &mut vnni_acc1);
        i += 64;
    }
    let dot_v = _mm512_add_epi32(vnni_acc0, vnni_acc1);
    let mut dot_i32 = _mm512_reduce_add_epi32(dot_v) as i64;
    let mut act_sum = _mm512_reduce_add_epi32(act_sum_acc) as i64;
    // < 64-byte scalar tail — keeps the kernel total bit-equivalent
    // to the reference across all lengths.
    while i < n {
        // Use the *unbiased* tail formula directly; equivalent to the
        // SIMD body's `(u + qx) - 128 * qx` rearrangement.
        dot_i32 += (qw[i] as i32 as i64).wrapping_mul(qx[i] as i32 as i64);
        i += 1;
    }
    // Subtract the bias contribution accumulated by VPDPBUSD.
    // The SIMD body computed `sum (qw + 128) * qx`, i.e. dot + 128 *
    // sum(qx_bytes_in_simd_region). Only correct over the SIMD
    // region; the scalar tail above was added directly, no bias.
    let dot_full = dot_i32 - 128i64 * act_sum;
    (dot_full as f32) * out_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32_avx512_matches_scalar_when_supported() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let a: Vec<f32> = (0..123).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let b: Vec<f32> = (0..123).map(|i| ((i as f32) * 0.7).cos()).collect();
        let lhs = unsafe { dot_f32_avx512(&a, &b) };
        let rhs = crate::kernels::scalar::dot_f32(&a, &b);
        assert!((lhs - rhs).abs() <= 1e-3);
    }

    #[test]
    fn dot_f32_avx512_handles_lengths_around_unroll_boundaries() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Cover lengths that exercise: empty body, 1×64 body, 2×64 body,
        // and odd tails (16-wide + scalar).
        for &n in &[0usize, 1, 7, 15, 16, 17, 32, 47, 63, 64, 65, 79, 128, 129, 257] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.17 - 2.0).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin()).collect();
            let lhs = unsafe { dot_f32_avx512(&a, &b) };
            let rhs = crate::kernels::scalar::dot_f32(&a, &b);
            assert!(
                (lhs - rhs).abs() <= 1e-3 + rhs.abs() * 1e-5,
                "len {n}: avx512 {lhs} vs scalar {rhs}"
            );
        }
    }

    #[test]
    fn swiglu_f32_avx512_matches_scalar_when_supported() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let rows = 13usize;
        let cols = 71usize; // odd to exercise the 16-wide + scalar tails
        let gate: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.07).sin()).collect();
        let up: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.11).cos()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
        let mut y_simd = vec![0.0f32; rows];
        unsafe { swiglu_f32_avx512(&gate, &up, &x, rows, cols, &mut y_simd) };
        let mut y_ref = vec![0.0f32; rows];
        crate::kernels::scalar::swiglu_f32(&gate, &up, &x, rows, cols, &mut y_ref);
        for i in 0..rows {
            assert!(
                (y_simd[i] - y_ref[i]).abs() <= 1e-3 + y_ref[i].abs() * 1e-4,
                "row {i}: avx512 {} vs scalar {}",
                y_simd[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn dequant_int8_avx512_matches_scalar_when_supported() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let scale = 0.0078125f32;
        let q: Vec<i8> = (0..200).map(|i| ((i % 251) - 125) as i8).collect();
        let x: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.13).sin()).collect();
        let lhs = unsafe { dequant_int8_dot_avx512(scale, &q, &x) };
        let rhs = crate::kernels::scalar::dequant_int8_dot(scale, &q, &x);
        assert!((lhs - rhs).abs() <= 1e-3);
    }
}
