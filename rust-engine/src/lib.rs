//! Thin library facade for the micro-expert-router engine.
//!
//! The crate is primarily a binary (`src/main.rs`) but a small set of
//! dependency-free modules is re-exposed as a `[lib]` so the
//! `tests/concurrency_stress.rs` integration benchmark can drive the
//! multi-tenant scheduling primitives (WRR fair-share, idle eviction,
//! pressure-aware speculation) without pulling in the full engine
//! graph.
//!
//! Adding new modules here is cheap, but please keep the surface
//! minimal: anything declared `pub mod` here gets compiled twice (once
//! for the bin, once for the lib), and modules with heavy
//! cross-module dependencies should be reached from
//! `tests/` via the binary itself rather than re-exported.

// Gist Task 3 — "Nightly AMX feature gating". Opt-in unstable feature
// flag that unlocks Rust's `stdarch_x86_amx` intrinsic surface
// (`_tile_loadd`, `_tile_dpbssd`, `_tile_stored`, …). When the
// `nightly-amx` cargo feature is OFF (the default), this attribute
// expands to nothing and the crate continues to build on stable Rust;
// the AMX dispatch path then falls back to the existing AVX-512
// kernel via [`crate::kernels::detect`]. Required on both crate
// roots (`lib.rs` and `main.rs`) because the bin and the lib are
// compiled as separate crates from this same source tree.
#![cfg_attr(feature = "nightly-amx", feature(stdarch_x86_amx))]
#![allow(dead_code)]

pub mod block_pool;
pub mod router;
