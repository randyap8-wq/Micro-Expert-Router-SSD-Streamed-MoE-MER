//! Hardware-specific math dispatcher — gist Phase 2.
//!
//! At startup the engine probes the host CPU once and picks a kernel
//! backend, which is then exposed via [`current()`]:
//!
//! * [`KernelBackend::Scalar`] — pure Rust, always available, the
//!   fallback every other backend is benchmarked against.
//! * [`KernelBackend::Avx512`] — AVX-512F + AVX-512BW intrinsics that
//!   fuse int8 dequant with the dot product so weights never spill to
//!   a separate `Vec<f32>`. Compiled in only when the `avx512` cargo
//!   feature is enabled (off by default to keep portable builds
//!   buildable on any x86_64 toolchain without nightly).
//! * [`KernelBackend::Amx`] — Intel AMX tile-based BF16 matmul stub.
//!   AMX intrinsics are nightly-only as of Rust 1.84, so this module
//!   only carries a documented skeleton and the runtime detector;
//!   enabling the `amx` cargo feature builds the skeleton in but the
//!   active kernels still fall through to AVX-512 / scalar. The
//!   detection plumbing is wired so a follow-up PR (or a nightly
//!   build) can drop a real AMX kernel into [`amx`] without touching
//!   any call sites.
//!
//! The dispatcher only covers kernels where SIMD makes a meaningful
//! difference for the engine's hot path — namely the int8 dequant-dot
//! used by the streaming SwiGLU experts and a couple of helper
//! reductions. The dense `gate_up_swiglu` / `down_proj` matmuls in
//! [`crate::inference`] already route through `simd` / `blas` cargo
//! features; this module **does not** duplicate that path, it
//! supplements it for the quantised-weight code paths the scalar
//! float matmul can't accelerate by itself.

pub mod scalar;

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub mod avx512;

#[cfg(feature = "amx")]
pub mod amx;

use std::sync::OnceLock;

/// Identifier for the active kernel backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    Scalar,
    Avx512,
    /// AMX tile-based BF16. See module docs.
    Amx,
}

impl KernelBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            KernelBackend::Scalar => "scalar",
            KernelBackend::Avx512 => "avx512",
            KernelBackend::Amx => "amx",
        }
    }
}

static BACKEND: OnceLock<KernelBackend> = OnceLock::new();

/// Runtime CPU-feature probe.
///
/// Order of preference: AMX (when both the cargo feature and the CPU
/// support it), then AVX-512F+BW (cargo feature + CPU), then scalar.
/// `std::is_x86_feature_detected!` is stable on x86 since Rust 1.27,
/// so this needs no extra crate dependency.
pub fn detect() -> KernelBackend {
    #[cfg(all(feature = "amx", target_arch = "x86_64"))]
    {
        if amx::cpu_supports_amx() {
            return KernelBackend::Amx;
        }
    }
    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return KernelBackend::Avx512;
        }
    }
    KernelBackend::Scalar
}

/// Return the active backend, probing once on first call.
pub fn current() -> KernelBackend {
    *BACKEND.get_or_init(detect)
}

/// Log a one-line description of the selected backend. Safe to call
/// multiple times; only the first call probes.
pub fn log_backend() {
    let b = current();
    tracing::info!(backend = b.as_str(), "selected math kernel backend");
}

// -----------------------------------------------------------------------
// Public dispatch entry points.
//
// Each kernel returns the same value as its scalar reference. AVX-512 /
// AMX paths are unsafe wrappers around `#[target_feature]` intrinsics
// and are only entered when `current()` confirms the CPU supports them.
// -----------------------------------------------------------------------

/// Dot product over `f32` slices. The dense transformer matmul path
/// uses BLAS / `simd` directly; this helper exists for the quantised
/// expert kernels that need a one-off `f32` dot.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    match current() {
        #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
        KernelBackend::Avx512 => unsafe { avx512::dot_f32_avx512(a, b) },
        _ => scalar::dot_f32(a, b),
    }
}

/// `sum_i scale * q[i] * x[i]` — fused symmetric-int8 dequant + dot.
/// `q` is a row of int8 weights, `scale` is the per-tensor scale, `x`
/// is an `f32` activation row of the same length.
#[inline]
pub fn dequant_int8_dot(scale: f32, q: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), x.len());
    match current() {
        #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
        KernelBackend::Avx512 => unsafe { avx512::dequant_int8_dot_avx512(scale, q, x) },
        _ => scalar::dequant_int8_dot(scale, q, x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_stable_value() {
        let a = current();
        let b = current();
        assert_eq!(a, b);
    }

    #[test]
    fn dot_f32_matches_scalar_reference() {
        let a: Vec<f32> = (0..133).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let b: Vec<f32> = (0..133).map(|i| ((i as f32) * 0.25).sin()).collect();
        let lhs = dot_f32(&a, &b);
        let rhs = scalar::dot_f32(&a, &b);
        assert!((lhs - rhs).abs() <= 1e-3, "dot_f32 mismatch: {lhs} vs {rhs}");
    }

    #[test]
    fn dequant_int8_dot_matches_scalar_reference() {
        let scale = 0.0123f32;
        let q: Vec<i8> = (0..256).map(|i| ((i % 251) - 125) as i8).collect();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).cos()).collect();
        let lhs = dequant_int8_dot(scale, &q, &x);
        let rhs = scalar::dequant_int8_dot(scale, &q, &x);
        assert!((lhs - rhs).abs() <= 1e-3, "dequant_int8_dot mismatch");
    }

    #[test]
    fn backend_log_string_is_known() {
        let s = current().as_str();
        assert!(matches!(s, "scalar" | "avx512" | "amx"));
    }
}
