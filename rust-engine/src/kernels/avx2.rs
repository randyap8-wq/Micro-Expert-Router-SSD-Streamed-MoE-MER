//! AVX2 + FMA kernels — feature-less auto-escalation path.
//!
//! Compiled unconditionally on `x86_64` (no cargo feature gate) so a
//! single binary deployed across a heterogeneous fleet automatically
//! benefits from AVX2 on any host that supports it. Every entry point
//! is `unsafe` because it relies on `#[target_feature(enable =
//! "avx2,fma")]`; callers gate dispatch on the runtime probe in
//! [`super::detect`] so these routines never execute on a CPU that
//! doesn't support them.
//!
//! Results are bit-equivalent to the [`super::scalar`] reference up
//! to floating-point reduction reordering (about 1 ULP per ~8-wide
//! accumulator, well under the engine's `1e-3` tolerance).

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// AVX2 f32 dot product. 8-wide FMA accumulator, scalar tail.
///
/// # Safety
///
/// Caller must guarantee the CPU supports `avx2 + fma`. The
/// dispatcher in [`super::dot_f32`] checks this exactly once at
/// startup via [`super::cpu_features`]. The kernel itself reads
/// through `_mm256_loadu_ps` (no alignment requirement on the
/// pointer), writes nothing, and uses a separate scalar loop for
/// the < 8 trailing elements, so no out-of-bounds access is possible
/// for any `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` guarantees the eight floats from offset
        // `i` are in bounds for both slices; `loadu_ps` has no
        // alignment requirement.
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    // Horizontal sum of the 8-wide accumulator.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    let mut sum = _mm_cvtss_f32(sums);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32_avx2_matches_scalar_when_supported() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let a: Vec<f32> = (0..123).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let b: Vec<f32> = (0..123).map(|i| ((i as f32) * 0.7).cos()).collect();
        // SAFETY: branch guarded above on the CPU feature probe.
        let lhs = unsafe { dot_f32_avx2(&a, &b) };
        let rhs = crate::kernels::scalar::dot_f32(&a, &b);
        assert!((lhs - rhs).abs() <= 1e-3);
    }
}
