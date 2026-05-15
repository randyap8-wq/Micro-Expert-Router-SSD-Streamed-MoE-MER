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

#![allow(dead_code)]

pub mod block_pool;
pub mod router;
