//! Hardware-specific math dispatcher — gist Task 1 (Hardware-Agnostic
//! "Auto-Escalation").
//!
//! At startup the engine probes the host CPU **once** and picks a
//! kernel backend, which is then exposed via [`current()`]:
//!
//! * [`KernelBackend::Scalar`] — pure Rust, always available, the
//!   fallback every other backend is benchmarked against.
//! * [`KernelBackend::Avx2`] — 256-bit FMA dot product. Compiled in
//!   unconditionally on `x86_64` (no cargo feature required); the
//!   `#[target_feature(enable = "avx2,fma")]` body is only entered
//!   when the runtime detector confirms the CPU supports it.
//! * [`KernelBackend::Avx512`] — AVX-512F + AVX-512BW intrinsics that
//!   fuse int8 dequant with the dot product so weights never spill to
//!   a separate `Vec<f32>`. Compiled in only when the `avx512` cargo
//!   feature is enabled (off by default so portable builds keep
//!   working on toolchains pinned to a non-AVX-512 baseline).
//! * [`KernelBackend::Amx`] — Intel AMX tile-based BF16 matmul stub.
//!   AMX intrinsics are nightly-only as of Rust 1.84, so this module
//!   only carries a documented skeleton and the runtime detector;
//!   enabling the `amx` cargo feature builds the skeleton in but the
//!   active kernels still fall through to AVX-512 / scalar. The
//!   detection plumbing is wired so a follow-up PR (or a nightly
//!   build) can drop a real AMX kernel into [`amx`] without touching
//!   any call sites.
//!
//! ### Why a feature-less auto-escalation path matters
//!
//! Before this module existed, opting into a SIMD matmul meant
//! recompiling with `--features simd` (or `--features blas`) and
//! shipping a different binary per CPU class. That conflicts with the
//! gist's **"Hardware-Agnostic Auto-Escalation"** requirement: a
//! single binary deployed across heterogeneous fleets must pick the
//! best local kernel by itself. The AVX2 path is therefore always
//! compiled (no cargo feature), and the runtime detector chooses
//! between AVX-512 (if the optional feature was built) → AVX2 → scalar
//! transparently. The selection is logged once at startup so ops can
//! see which path is live.
//!
//! ### Zero-overhead dispatch
//!
//! [`current()`] caches the detection result in a `OnceLock`, so the
//! hot path pays a single atomic load — never a `cfg!` evaluation,
//! never a re-probe of `is_x86_feature_detected!`. This matches the
//! gist's explicit "Zero-Overhead Dispatch" constraint.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub mod avx512;

#[cfg(feature = "amx")]
pub mod amx;

use std::sync::OnceLock;

/// Identifier for the active kernel backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    Scalar,
    /// 256-bit FMA dot product. Always compiled on x86_64; entered
    /// only when the CPU supports `avx2 + fma`.
    Avx2,
    Avx512,
    /// AMX tile-based BF16. See module docs.
    Amx,
}

impl KernelBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            KernelBackend::Scalar => "scalar",
            KernelBackend::Avx2 => "avx2",
            KernelBackend::Avx512 => "avx512",
            KernelBackend::Amx => "amx",
        }
    }
}

/// Detailed snapshot of the host CPU's capabilities, used both by the
/// kernel selector and by the startup log to give ops a single line
/// they can correlate with deployment fleets.
///
/// Filled in once at startup by [`probe_cpu`] and cached in a
/// `OnceLock`; the hot path never re-runs the probe. This matches the
/// gist's explicit "Zero-Overhead Dispatch" constraint.
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    /// Vendor string from `/proc/cpuinfo` (`"GenuineIntel"`, …) or
    /// `"unknown"` on non-Linux / parse failure.
    pub vendor: String,
    /// CPU model string (`model name` in `/proc/cpuinfo`).
    pub model: String,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    /// AVX-512 VNNI (Vector Neural Network Instructions): adds
    /// `VPDPBUSD` for fused (u8 × i8) → i32 reductions. Required by
    /// the int8×int8 dot kernel
    /// [`avx512::dot_int8_int8_avx512_vnni`] (gist Part 2, fix #8).
    pub avx512vnni: bool,
    pub amx_tile: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    /// True if the CPU model string contains a substring that strongly
    /// suggests this is a Sapphire Rapids (or newer Granite Rapids)
    /// Xeon — the chips on which the AMX 2-tile matmul path is the
    /// default-preferred kernel.
    pub sapphire_rapids_or_newer: bool,
}

impl CpuFeatures {
    fn unknown() -> Self {
        Self {
            vendor: "unknown".into(),
            model: "unknown".into(),
            avx2: false,
            fma: false,
            avx512f: false,
            avx512bw: false,
            avx512vnni: false,
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            sapphire_rapids_or_newer: false,
        }
    }
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
static BACKEND: OnceLock<KernelBackend> = OnceLock::new();

/// Cached host CPU probe. The first call performs the actual
/// detection; subsequent calls are a single atomic load.
pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(probe_cpu)
}

#[cfg(target_arch = "x86_64")]
fn probe_cpu() -> CpuFeatures {
    let mut f = CpuFeatures::unknown();
    f.avx2 = std::is_x86_feature_detected!("avx2");
    f.fma = std::is_x86_feature_detected!("fma");
    f.avx512f = std::is_x86_feature_detected!("avx512f");
    f.avx512bw = std::is_x86_feature_detected!("avx512bw");
    f.avx512vnni = std::is_x86_feature_detected!("avx512vnni");
    // `is_x86_feature_detected!("amx-tile")` is gated behind the
    // unstable `x86_amx_intrinsics` feature on stable Rust as of
    // 1.84, so we additionally consult /proc/cpuinfo on Linux.
    f.amx_tile = cpuinfo_has_flag("amx_tile");
    f.amx_int8 = cpuinfo_has_flag("amx_int8");
    f.amx_bf16 = cpuinfo_has_flag("amx_bf16");
    if let Some((vendor, model)) = cpuinfo_vendor_model() {
        f.sapphire_rapids_or_newer = is_sapphire_rapids_or_newer(&model);
        f.vendor = vendor;
        f.model = model;
    }
    f
}

#[cfg(not(target_arch = "x86_64"))]
fn probe_cpu() -> CpuFeatures {
    // AArch64 / other architectures: the engine still runs, just on
    // the scalar reference. NEON autovectorisation by the Rust
    // optimiser remains active in the scalar path.
    let (vendor, model) = cpuinfo_vendor_model().unwrap_or_else(|| ("unknown".into(), "unknown".into()));
    let mut f = CpuFeatures::unknown();
    f.vendor = vendor;
    f.model = model;
    f
}

/// Heuristic match for Sapphire-Rapids-class (or newer) Xeons that
/// are **expected to advertise AMX**. We don't ship a full CPUID
/// family / model table; instead we look for vendor-distributed
/// model strings that uniquely identify the generation. Updated as
/// Intel releases new SKUs.
///
/// The match is intentionally conservative: we require **both** the
/// `"xeon"` substring and a SKU-family token that's distinctive to
/// Sapphire Rapids or newer (e.g. `"platinum 84"`, `"gold 64"`,
/// `"xeon max"`, `"granite"`, `"emerald"`). Bare two-digit substrings
/// like `"64"` are too permissive — they match e.g. `"AMD EPYC
/// 7643"`. AMX is only ever advertised by Intel Xeon, so the `"xeon"`
/// gate alone already rules out AMD parts; the SKU tokens then narrow
/// the match to the Sapphire-Rapids generation specifically.
///
/// **Bronze SKUs are deliberately excluded.** The only Sapphire-Rapids
/// Bronze part at the time of writing (Bronze 3408U) does not
/// implement AMX, so matching `"bronze 34"` / `"bronze 35"` here
/// would yield a misleading `sapphire_rapids` log line on a chip
/// that will never escalate to the AMX kernel. The downstream
/// `detect()` separately requires `amx_tile && amx_int8` from
/// `/proc/cpuinfo`, so omitting Bronze tokens has no effect on
/// kernel selection — only on log accuracy.
fn is_sapphire_rapids_or_newer(model: &str) -> bool {
    let s = model.to_ascii_lowercase();
    if !s.contains("xeon") {
        return false;
    }
    // Sapphire Rapids (4th gen Xeon Scalable) and the Xeon W-2400 /
    // W-3400 workstation parts. Tokens are taken straight from Intel's
    // product pages.
    const SKU_TOKENS: &[&str] = &[
        "platinum 84", "platinum 85",
        "gold 64", "gold 65",
        "silver 44", "silver 45",
        // "max" — only the Xeon Max line carries this token; gated by
        // the outer `s.contains("xeon")` check above so it can't match
        // arbitrary non-Xeon model strings.
        " max ", "w-2400", "w-3400",
        // Granite Rapids (5th gen Xeon Scalable, AMX-capable) and
        // Emerald Rapids (refresh of Sapphire Rapids).
        "granite", "emerald",
    ];
    SKU_TOKENS.iter().any(|tok| s.contains(tok))
}

fn cpuinfo_has_flag(flag: &str) -> bool {
    let Some(s) = read_proc_cpuinfo() else { return false };
    s.lines()
        .filter(|l| l.starts_with("flags") || l.starts_with("Features"))
        .any(|l| l.split_whitespace().any(|tok| tok == flag))
}

fn cpuinfo_vendor_model() -> Option<(String, String)> {
    let s = read_proc_cpuinfo()?;
    let mut vendor = None;
    let mut model = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("vendor_id") {
            vendor = v.split(':').nth(1).map(|x| x.trim().to_string());
        } else if let Some(v) = line.strip_prefix("model name") {
            model = v.split(':').nth(1).map(|x| x.trim().to_string());
        }
        if vendor.is_some() && model.is_some() {
            break;
        }
    }
    Some((vendor.unwrap_or_else(|| "unknown".into()),
          model.unwrap_or_else(|| "unknown".into())))
}

fn read_proc_cpuinfo() -> Option<String> {
    use std::io::Read;
    let mut s = String::new();
    let mut f = std::fs::File::open("/proc/cpuinfo").ok()?;
    f.read_to_string(&mut s).ok()?;
    Some(s)
}

/// Runtime CPU-feature probe. Order of preference:
/// AMX (when both the cargo feature and the CPU support it, and the
/// CPU is Sapphire-Rapids-class or newer) → AVX-512F+BW (cargo
/// feature + CPU) → AVX2+FMA (always compiled on x86_64) → scalar.
pub fn detect() -> KernelBackend {
    let f = cpu_features();
    #[cfg(all(feature = "amx", target_arch = "x86_64"))]
    {
        if f.amx_tile && f.amx_int8 && f.sapphire_rapids_or_newer {
            return KernelBackend::Amx;
        }
    }
    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    {
        if f.avx512f && f.avx512bw {
            return KernelBackend::Avx512;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if f.avx2 && f.fma {
            return KernelBackend::Avx2;
        }
    }
    // Touch f so non-x86 builds still see it as used.
    let _ = f;
    KernelBackend::Scalar
}

/// Return the active backend, probing once on first call.
pub fn current() -> KernelBackend {
    *BACKEND.get_or_init(detect)
}

/// Log a one-line description of the selected backend and the salient
/// CPU features that drove the decision. Safe to call multiple times;
/// only the first call probes.
pub fn log_backend() {
    let b = current();
    let f = cpu_features();
    tracing::info!(
        backend = b.as_str(),
        vendor = %f.vendor,
        model = %f.model,
        avx2 = f.avx2,
        avx512 = f.avx512f && f.avx512bw,
        avx512vnni = f.avx512vnni,
        amx_int8 = f.amx_int8,
        sapphire_rapids = f.sapphire_rapids_or_newer,
        "auto-escalation selected math kernel backend"
    );
}

// -----------------------------------------------------------------------
// Public dispatch entry points.
//
// Each kernel returns the same value as its scalar reference. AVX-2 /
// AVX-512 / AMX paths are unsafe wrappers around `#[target_feature]`
// intrinsics and are only entered when `current()` confirms the CPU
// supports them.
// -----------------------------------------------------------------------

/// Dot product over `f32` slices. The dense transformer matmul path
/// uses BLAS / SIMD directly; this helper exists for the quantised
/// expert kernels that need a one-off `f32` dot.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    match current() {
        #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
        KernelBackend::Avx512 => {
            // SAFETY: the dispatcher confirmed the CPU supports
            // `avx512f` via the runtime probe in `cpu_features()`;
            // the AVX-512 kernel uses only safe slice indexing and a
            // 16-wide FMA accumulator, with the scalar tail handled
            // explicitly.
            unsafe { avx512::dot_f32_avx512(a, b) }
        }
        #[cfg(target_arch = "x86_64")]
        KernelBackend::Avx2 => {
            // SAFETY: the dispatcher confirmed the CPU supports
            // `avx2 + fma`. The kernel reads via `loadu_ps` (no
            // alignment requirement on the pointer), writes nothing,
            // and uses an explicit scalar tail.
            unsafe { avx2::dot_f32_avx2(a, b) }
        }
        _ => scalar::dot_f32(a, b),
    }
}

/// **Fully-quantised int8×int8 dot** — `out_scale * sum_i (qw[i] *
/// qx[i])` (gist Part 2, fix #8). When the host CPU supports AVX-512
/// VNNI (`vpdpbusd`), the inner reduction stays in i32 integer
/// registers via [`avx512::dot_int8_int8_avx512_vnni`] and only the
/// final `out_scale` fold spends an f32 multiply. Otherwise the call
/// falls through to the scalar reference (still i32 accumulation, so
/// the value is bit-equivalent).
///
/// Use this entry point when both the weight stream **and** the
/// activation row are already int8-quantised — the engine's existing
/// [`dequant_int8_dot`] entry point keeps the mixed f32-activation
/// shape used by the default streaming int8 path.
#[inline]
pub fn dot_int8_int8(out_scale: f32, qw: &[i8], qx: &[i8]) -> f32 {
    debug_assert_eq!(qw.len(), qx.len());
    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    {
        // This kernel is valid whenever the CPU exposes the exact
        // ISA features it requires, even if the global dispatcher has
        // selected `KernelBackend::Amx`. Gate directly on the cached
        // AVX-512 feature bits so AMX-capable hosts do not fall back
        // to scalar for int8×int8 dots.
        let f = cpu_features();
        if f.avx512f && f.avx512bw && f.avx512vnni {
            // SAFETY: cached CPU feature detection confirmed
            // `avx512f + avx512bw + avx512vnni`. The kernel reads via
            // unaligned `loadu_si512`, writes nothing, and handles
            // the scalar tail explicitly. The bias-trick correction
            // is documented on the kernel itself.
            return unsafe { avx512::dot_int8_int8_avx512_vnni(out_scale, qw, qx) };
        }
    }
    scalar::dot_int8_int8(out_scale, qw, qx)
}

/// `sum_i scale * q[i] * x[i]` — fused symmetric-int8 dequant + dot.
/// `q` is a row of int8 weights, `scale` is the per-tensor scale, `x`
/// is an `f32` activation row of the same length.
#[inline]
pub fn dequant_int8_dot(scale: f32, q: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), x.len());
    match current() {
        #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
        KernelBackend::Avx512 => {
            // SAFETY: the dispatcher confirmed the CPU supports
            // `avx512f + avx512bw` (the latter is required for
            // `_mm512_cvtepi8_epi32`). The kernel reads via
            // `loadu_si128` / `loadu_ps` (no alignment requirement),
            // writes nothing, and handles the scalar tail
            // explicitly.
            unsafe { avx512::dequant_int8_dot_avx512(scale, q, x) }
        }
        _ => scalar::dequant_int8_dot(scale, q, x),
    }
}

/// Fused SwiGLU FFN inner stage: `y[i] = silu(gate_w[i]·x) * (up_w[i]·x)`.
///
/// Writes into the caller-provided `y` (length `rows`) — **no
/// allocation on the hot path**, matching the gist's "performance
/// guardrail" rule. `gate_w` and `up_w` are row-major
/// `[rows × cols]` matrices, `x` is `cols`-long.
///
/// Auto-escalates: AVX-512 (when the cargo feature is compiled **and**
/// the CPU advertises `avx512f`) → scalar reference. The AVX-512 path
/// is the fused [`avx512::swiglu_f32_avx512`] kernel; the scalar
/// fallback is [`scalar::swiglu_f32`].
#[inline]
pub fn swiglu_f32_into(
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
    match current() {
        #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
        KernelBackend::Avx512 => {
            // SAFETY: dispatcher confirmed `avx512f`. The kernel only
            // reads via `loadu_ps` (no alignment requirement) and
            // writes through the caller's `&mut [f32]`.
            unsafe { avx512::swiglu_f32_avx512(gate_w, up_w, x, rows, cols, y) }
        }
        _ => scalar::swiglu_f32(gate_w, up_w, x, rows, cols, y),
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
    fn swiglu_f32_into_matches_scalar_reference() {
        let rows = 11usize;
        let cols = 73usize;
        let gate: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.07).sin()).collect();
        let up: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.11).cos()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
        let mut y_dispatch = vec![0.0f32; rows];
        swiglu_f32_into(&gate, &up, &x, rows, cols, &mut y_dispatch);
        let mut y_ref = vec![0.0f32; rows];
        scalar::swiglu_f32(&gate, &up, &x, rows, cols, &mut y_ref);
        for i in 0..rows {
            assert!(
                (y_dispatch[i] - y_ref[i]).abs() <= 1e-3 + y_ref[i].abs() * 1e-4,
                "swiglu_f32_into mismatch at {i}: {} vs {}",
                y_dispatch[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn dot_int8_int8_matches_scalar_reference() {
        // gist Part 2, fix #8 — VNNI kernel must be bit-equivalent to
        // the scalar reference (both accumulate in i32, then fold the
        // f32 scale at the very end, so the only floating-point error
        // is in that final multiply).
        let scale = 0.0078125f32; // exact in binary fp
        let qw: Vec<i8> = (0..193).map(|i| ((i % 251) - 125) as i8).collect();
        let qx: Vec<i8> = (0..193).map(|i| (((i * 7) % 197) - 98) as i8).collect();
        let lhs = dot_int8_int8(scale, &qw, &qx);
        let rhs = scalar::dot_int8_int8(scale, &qw, &qx);
        assert!(
            (lhs - rhs).abs() <= 1e-3 + rhs.abs() * 1e-4,
            "dot_int8_int8 mismatch: dispatch={lhs} scalar={rhs}"
        );
    }

    #[test]
    fn backend_log_string_is_known() {
        let s = current().as_str();
        assert!(matches!(s, "scalar" | "avx2" | "avx512" | "amx"));
    }

    #[test]
    fn cpu_features_probe_is_stable() {
        let a = cpu_features();
        let b = cpu_features();
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn sapphire_rapids_heuristic_recognises_obvious_models() {
        assert!(is_sapphire_rapids_or_newer("Intel(R) Xeon(R) Platinum 8480+"));
        assert!(is_sapphire_rapids_or_newer("Intel(R) Xeon(R) Gold 6448Y"));
        assert!(is_sapphire_rapids_or_newer("Intel(R) Xeon(R) Max 9468"));
        assert!(!is_sapphire_rapids_or_newer("AMD EPYC 7763 64-Core Processor"));
        assert!(!is_sapphire_rapids_or_newer("AMD EPYC 7643 48-Core Processor"));
        assert!(!is_sapphire_rapids_or_newer("Apple M1 Pro"));
        // Older Xeon generations must NOT match (they predate AMX).
        assert!(!is_sapphire_rapids_or_newer("Intel(R) Xeon(R) Platinum 8260"));
        assert!(!is_sapphire_rapids_or_newer("Intel(R) Xeon(R) Gold 6230"));
    }
}
