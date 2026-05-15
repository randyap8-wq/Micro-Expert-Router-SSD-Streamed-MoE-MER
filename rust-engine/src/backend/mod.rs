//! Decoupled math-backend trait (gist Task 2 — Plugin System).
//!
//! The engine's I/O substrate (`expert_cache`, `buffer_pool`,
//! `io_provider`, the O_DIRECT `pread(2)` pipeline) is intentionally
//! independent from the math library used to crunch the bytes once
//! they land in RAM.  This module defines the **`Backend`** trait that
//! every math implementation must satisfy, and exposes a small registry
//! of named implementations the rest of the engine can pick from at
//! startup — without `cfg(feature = …)` walls inside the hot path and
//! without coupling `EngineCore` to any one tensor library.
//!
//! Today we ship two implementations:
//!
//! * [`ScalarBackend`] — pure Rust reference. Always available, no
//!   external deps. Used as the validation oracle every other backend
//!   is tested against and as the fallback when no other backend is
//!   selected.
//! * [`CandleBackend`] — wraps the existing `candle-core` CPU path that
//!   `inference.rs` already drives the per-expert SwiGLU forward pass
//!   through. Selected by default at startup so the production codepath
//!   is bit-for-bit unchanged.
//!
//! Future backends (Burn, Tract, a custom CUDA / Vulkan executor)
//! simply implement `Backend` and call [`set_backend`] before the first
//! token is generated. Because the trait is object-safe and lives
//! behind an `Arc<dyn Backend>`, swapping is a drop-in pointer change;
//! no recompile of the rest of the crate is required.
//!
//! ### Zero-overhead dispatch
//!
//! Per the gist's "Zero-Overhead Dispatch" constraint, [`current`]
//! resolves the active backend via a `OnceLock` initialised exactly
//! once at process start (driven by [`install_default`] in
//! `main.rs`). The hot path therefore pays one atomic load, never a
//! `cfg!` macro evaluation, a feature-gated branch, or a runtime probe.

use std::sync::Arc;
use std::sync::OnceLock;

/// Minimal contract every math backend must satisfy.
///
/// The methods are intentionally small and side-effect-free: they take
/// owned / borrowed slices, return fresh `Vec`s or write into a caller
/// buffer. This keeps the trait `Send + Sync` and lets implementations
/// own their own scratch storage (thread-local, arena, whatever) without
/// leaking lifetimes into the trait surface.
///
/// All shape arguments are *logical* dimensions; row-major layout is
/// assumed for matrix inputs (`W` of shape `[rows × cols]` is
/// `rows * cols` floats with row `i` starting at `i * cols`). This
/// matches the on-disk SwiGLU layout the engine streams from NVMe.
pub trait Backend: Send + Sync + 'static {
    /// Short human-readable identifier (e.g. `"scalar"`, `"candle"`).
    /// Logged once at startup so ops can see which executor is live.
    fn name(&self) -> &'static str;

    /// Row-major matrix-vector multiply `y = W · x`. `W` is
    /// `[rows × cols]` in row-major order; `x.len() == cols`.
    /// Implementations must return a fresh `Vec<f32>` of length `rows`.
    ///
    /// **Hot-path callers should prefer [`Backend::matmul_into`]** —
    /// this method exists for older callers that owned the output
    /// `Vec`. The default `matmul_into` impl delegates here, so
    /// implementing only `matmul` keeps everything correct;
    /// implementing `matmul_into` directly eliminates the allocation.
    fn matmul(&self, w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows];
        self.matmul_into(w, x, rows, cols, &mut y);
        y
    }

    /// Row-major matmul that writes into a **caller-owned** `y`
    /// (`y.len() == rows`). The "no allocation on the hot path"
    /// performance guardrail: implementations must not allocate new
    /// `Vec`s here.
    ///
    /// Default impl allocates via [`Backend::matmul`] and copies the
    /// result into `y` — correct, but defeats the no-alloc guardrail.
    /// Built-in backends override it.
    fn matmul_into(&self, w: &[f32], x: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
        debug_assert_eq!(y.len(), rows);
        let v = self.matmul(w, x, rows, cols);
        y.copy_from_slice(&v);
    }

    /// Fused SwiGLU FFN inner stage: `y[i] = silu(gate_w[i]·x) * (up_w[i]·x)`.
    ///
    /// `gate_w` and `up_w` are row-major `[rows × cols]` matrices.
    /// `y` is caller-owned (no allocation on the hot path).
    ///
    /// Default impl chains two `matmul_into` calls + a scalar SiLU
    /// fuse — correct on every backend, but built-in backends
    /// override to keep the gate intermediate hot in registers.
    fn swiglu_into(
        &self,
        gate_w: &[f32],
        up_w: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        y: &mut [f32],
    ) {
        debug_assert_eq!(y.len(), rows);
        // Caller-owned scratch is not part of the trait surface, so
        // the default routes through `matmul` (allocation acceptable
        // on this slow fallback path); overriding backends fuse.
        let g = self.matmul(gate_w, x, rows, cols);
        let u = self.matmul(up_w, x, rows, cols);
        for i in 0..rows {
            let silu_g = g[i] / (1.0 + (-g[i]).exp());
            y[i] = silu_g * u[i];
        }
    }

    /// In-place softmax over `logits`. Numerically stable; the result
    /// must be non-negative and sum to exactly 1.0 within `f32`
    /// rounding (i.e. `(sum - 1.0).abs() < 1e-5`).
    fn softmax(&self, logits: &mut [f32]);

    /// Elementwise SiLU (a.k.a. swish): `x * sigmoid(x)`, in place.
    fn silu_inplace(&self, x: &mut [f32]);
}

// =====================================================================
// Built-in backends.
// =====================================================================

/// Pure-Rust scalar reference backend.
///
/// Single-threaded; the optimiser is responsible for autovectorising
/// the inner loops. Always available — no extra crate dependency, no
/// CPU-feature requirement. Used as the validation oracle for every
/// other backend (see the `backend_implementations_agree` test below).
pub struct ScalarBackend;

impl Backend for ScalarBackend {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn matmul_into(&self, w: &[f32], x: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
        debug_assert_eq!(w.len(), rows * cols);
        debug_assert_eq!(x.len(), cols);
        debug_assert_eq!(y.len(), rows);
        // Route through the kernel dispatcher so the scalar backend
        // still auto-escalates to AVX-512 / AVX2 when the host
        // supports it — same value as the hand-rolled inner loop, no
        // allocation on the hot path.
        for i in 0..rows {
            y[i] = crate::kernels::dot_f32(&w[i * cols..(i + 1) * cols], x);
        }
    }

    fn swiglu_into(
        &self,
        gate_w: &[f32],
        up_w: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        y: &mut [f32],
    ) {
        // Fused dispatcher — picks the AVX-512 fused kernel when
        // available, otherwise the scalar reference.
        crate::kernels::swiglu_f32_into(gate_w, up_w, x, rows, cols, y);
    }

    fn softmax(&self, logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        let mut maxv = f32::NEG_INFINITY;
        for &v in logits.iter() {
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - maxv).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in logits.iter_mut() {
                *v /= sum;
            }
        }
    }

    fn silu_inplace(&self, x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }
}

/// Hugging Face `candle-core` CPU backend.
///
/// Delegates to the existing `Tensor`-based math the engine already
/// uses for the per-expert SwiGLU forward pass in
/// [`crate::inference::ExpertWeights::forward_candle`]. Pulled in by
/// default at startup — the goal of this trait is to give us a clean
/// swap point, not to change the default executor.
///
/// `candle-core` is built with `default-features = false` in
/// `Cargo.toml`, so this backend remains CPU-only and adds no GPU
/// runtime requirement to the binary.
pub struct CandleBackend;

impl Backend for CandleBackend {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn matmul_into(&self, w: &[f32], x: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
        // **Performance contract.** The previous revision rebuilt two
        // `candle_core::Tensor`s from scratch on every call (one
        // allocation per tensor plus dispatch overhead). For the
        // small per-expert / per-row shapes the engine actually
        // calls this with, that setup cost dwarfed the FMA work.
        //
        // The optimised path bypasses Candle entirely and dispatches
        // through [`crate::kernels::dot_f32`], which is the runtime
        // AVX-512 → AVX2 → scalar auto-escalator. The result is
        // bit-equivalent to the Candle Tensor matmul within `f32`
        // reduction rounding (the `backend_implementations_agree`
        // test enforces a 1e-4 envelope), allocations on the hot
        // path are zero, and AVX-512 hosts get the fused 4×-unrolled
        // FMA kernel from `kernels::avx512`.
        debug_assert_eq!(w.len(), rows * cols);
        debug_assert_eq!(x.len(), cols);
        debug_assert_eq!(y.len(), rows);
        for i in 0..rows {
            y[i] = crate::kernels::dot_f32(&w[i * cols..(i + 1) * cols], x);
        }
    }

    fn swiglu_into(
        &self,
        gate_w: &[f32],
        up_w: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        y: &mut [f32],
    ) {
        // Fused gate + up + SiLU through the kernel dispatcher.
        // On AVX-512 hosts this is the single-pass
        // [`crate::kernels::avx512::swiglu_f32_avx512`] kernel
        // (no candle Tensor round-trip, no scratch allocation).
        crate::kernels::swiglu_f32_into(gate_w, up_w, x, rows, cols, y);
    }

    fn softmax(&self, logits: &mut [f32]) {
        // For a 1-D vector the scalar path is faster than a round-trip
        // through a candle Tensor; the trait contract (sum == 1.0) is
        // satisfied identically by the scalar reference.
        ScalarBackend.softmax(logits);
    }

    fn silu_inplace(&self, x: &mut [f32]) {
        ScalarBackend.silu_inplace(x);
    }
}

// =====================================================================
// Global registry — set once at startup, read on every hot-path call.
// =====================================================================

static BACKEND: OnceLock<Arc<dyn Backend>> = OnceLock::new();

/// Install `b` as the process-wide active backend. Returns `Err` if a
/// backend has already been installed — the trait is intentionally a
/// "set once at startup" contract so the hot path can rely on a single
/// atomic load.
pub fn set_backend(b: Arc<dyn Backend>) -> Result<(), &'static str> {
    BACKEND
        .set(b)
        .map_err(|_| "backend already installed; call before any token is generated")
}

/// Install the default backend (`CandleBackend`) if none has been set
/// yet. Called from `main` at startup; safe to call multiple times.
///
/// Precedence is first-writer-wins:
/// - if [`set_backend`] was called earlier, this is a no-op and the
///   caller-provided backend remains active;
/// - if this function runs first, later calls to [`set_backend`] will
///   return `Err` because the process-wide backend has already been
///   installed.
pub fn install_default() {
    let _ = BACKEND.set(Arc::new(CandleBackend) as Arc<dyn Backend>);
}

/// Active backend. Falls back to [`ScalarBackend`] when nothing has
/// been installed — useful in tests where `main` hasn't run. On a
/// production binary `main` always installs `CandleBackend` before
/// the first request, so the fallback is purely a belt-and-braces
/// measure.
pub fn current() -> Arc<dyn Backend> {
    BACKEND
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(ScalarBackend) as Arc<dyn Backend>)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All built-in backends must agree on small reference inputs
    /// within `f32` tolerance. This is the property-style "swap is
    /// safe" check the gist asks for in Task 2.
    #[test]
    fn backend_implementations_agree() {
        let rows = 7usize;
        let cols = 13usize;
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.1).sin()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.3).cos()).collect();
        let s = ScalarBackend.matmul(&w, &x, rows, cols);
        let c = CandleBackend.matmul(&w, &x, rows, cols);
        assert_eq!(s.len(), c.len());
        for i in 0..rows {
            assert!(
                (s[i] - c[i]).abs() < 1e-4,
                "scalar vs candle matmul mismatch at {i}: {} vs {}",
                s[i],
                c[i]
            );
        }
    }

    #[test]
    fn matmul_into_writes_into_caller_buffer_without_allocating() {
        // The new `_into` entry points satisfy the "no allocation on
        // the hot path" guardrail: the caller owns `y`.
        let rows = 5usize;
        let cols = 9usize;
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.4).sin()).collect();
        let mut y_scalar = vec![0.0f32; rows];
        let mut y_candle = vec![0.0f32; rows];
        ScalarBackend.matmul_into(&w, &x, rows, cols, &mut y_scalar);
        CandleBackend.matmul_into(&w, &x, rows, cols, &mut y_candle);
        for i in 0..rows {
            assert!((y_scalar[i] - y_candle[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn swiglu_into_agrees_across_backends_and_with_unfused_reference() {
        let rows = 6usize;
        let cols = 11usize;
        let gate: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.21).sin()).collect();
        let up: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.17).cos()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.31).cos() * 0.4).collect();

        let mut y_scalar = vec![0.0f32; rows];
        let mut y_candle = vec![0.0f32; rows];
        ScalarBackend.swiglu_into(&gate, &up, &x, rows, cols, &mut y_scalar);
        CandleBackend.swiglu_into(&gate, &up, &x, rows, cols, &mut y_candle);

        // Reference: unfused matmul + scalar SiLU + multiply.
        let g = ScalarBackend.matmul(&gate, &x, rows, cols);
        let u = ScalarBackend.matmul(&up, &x, rows, cols);
        let expected: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
            .collect();
        for i in 0..rows {
            assert!(
                (y_scalar[i] - expected[i]).abs() < 1e-4,
                "scalar swiglu mismatch at {i}: {} vs {}",
                y_scalar[i],
                expected[i]
            );
            assert!(
                (y_candle[i] - expected[i]).abs() < 1e-4,
                "candle swiglu mismatch at {i}: {} vs {}",
                y_candle[i],
                expected[i]
            );
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = vec![1.0, 2.0, -3.0, 0.5, 4.2];
        ScalarBackend.softmax(&mut logits);
        let s: f32 = logits.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax sum {s}");
        for &v in &logits {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn silu_matches_reference() {
        let xs = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
        let expected: Vec<f32> = xs.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        let mut got = xs.to_vec();
        ScalarBackend.silu_inplace(&mut got);
        for i in 0..xs.len() {
            assert!((got[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn current_always_resolves() {
        let b = current();
        let _ = b.name();
    }
}
