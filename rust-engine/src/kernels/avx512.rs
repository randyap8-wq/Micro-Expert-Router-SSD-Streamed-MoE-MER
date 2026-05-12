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

/// AVX-512 f32 dot product. 16-wide accumulator, scalar tail.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx512f`. The dispatcher in
/// [`super::dot_f32`] checks this exactly once at startup.
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
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
