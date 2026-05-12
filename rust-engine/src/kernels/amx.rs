//! Intel AMX (Advanced Matrix Extensions) kernel skeleton.
//!
//! AMX intrinsics (`_tile_loadd`, `_tile_dpbssd`, `_tile_stored`, …)
//! are nightly-only on Rust as of 1.84 (tracked under
//! <https://github.com/rust-lang/rust/issues/126622>). This module
//! therefore deliberately keeps the runtime detector and the kernel
//! *shape* — so the dispatcher in [`super::detect`] can already
//! prefer AMX when available — while leaving the actual tile-based
//! matmul body as a documented stub that today routes back to the
//! scalar reference.
//!
//! Why include this at all? Two reasons:
//!
//! * The U.T.H. emitted by `gguf-convert` already carries `amx_tile_m
//!   /n/k` hints. Having a place to land those hints (`amx_tile_hint`)
//!   makes the contract testable today even though the executor is
//!   stubbed.
//! * A follow-up PR (or a downstream user on nightly) can drop a real
//!   tile-based body in here without touching call sites in
//!   [`crate::inference`] or [`crate::kernels`] — the dispatcher and
//!   the U.T.H. are forward-compatible.
//!
//! When the cargo feature `amx` is *not* set, this module is not
//! compiled at all; the dispatcher falls through to AVX-512 / scalar.

#![cfg(feature = "amx")]

/// Detect Intel AMX support at runtime.
///
/// Stable Rust exposes `is_x86_feature_detected!("amx-tile")` and
/// `"amx-int8"` on recent toolchains. On older toolchains we treat
/// AMX as unavailable rather than abort the build.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn cpu_supports_amx() -> bool {
    // `amx-tile` is the umbrella; `amx-int8` and `amx-bf16` are the
    // useful executors. We require at least `amx-tile + amx-int8`
    // because the int8 dequant kernel is the one we'd actually drop
    // in here first.
    // `is_x86_feature_detected!` on stable Rust does not recognise the
    // `"amx-tile"` / `"amx-int8"` strings until those features are
    // stabilised (they were added under the unstable
    // `x86_amx_intrinsics` gate). Until then, fall back to a runtime
    // /proc/cpuinfo probe so the `amx` cargo feature builds on stable.
    cpuinfo_has_flag("amx_tile") && cpuinfo_has_flag("amx_int8")
}

#[cfg(target_arch = "x86_64")]
fn cpuinfo_has_flag(flag: &str) -> bool {
    use std::io::Read;
    let mut s = String::new();
    let Ok(mut f) = std::fs::File::open("/proc/cpuinfo") else {
        return false;
    };
    if f.read_to_string(&mut s).is_err() {
        return false;
    }
    s.lines()
        .filter(|l| l.starts_with("flags") || l.starts_with("Features"))
        .any(|l| l.split_whitespace().any(|tok| tok == flag))
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn cpu_supports_amx() -> bool {
    false
}

/// Preferred tile shape. Mirrors the `amx_tile_hint_*` fields of the
/// [`crate::tensor_header::TensorHeader`] so a real kernel can pick
/// these up from the on-disk hint when one is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxTileHint {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Default for AmxTileHint {
    fn default() -> Self {
        // 16×16×64 is the canonical AMX_INT8 tile size on Sapphire
        // Rapids and Granite Rapids.
        Self { m: 16, n: 16, k: 64 }
    }
}
