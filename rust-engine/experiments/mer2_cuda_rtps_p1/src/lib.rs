//! Source-only partial diagnostic. No native backend is connected.
#![forbid(unsafe_code)]

pub mod contract;
pub mod expected;
pub mod fixture;
pub mod reference;
pub mod report;

pub const D: usize = 2048;
pub const DFF: usize = 768;
pub const BLOCK_VALUES: usize = 32;
pub const BLOCK_BYTES: usize = 18;
pub const PROJECTION_BYTES: usize = 884_736;
pub const Q4_BYTES: usize = 2_654_208;
pub const STAGE_LENGTHS: [usize; 5] = [DFF, DFF, DFF, DFF, D];
pub const STAGE_NAMES: [&str; 5] = ["gate", "up", "silu", "product", "down"];
pub const NATIVE_STOP: &str = "STOP_NATIVE_SYNTHETIC_STAGE_ADAPTER_UNBOUND";
pub const FIXTURE_ID: &str = "rtps-p1-rectangular-dyadic-v1";
pub const WORKSPACE_ID: &str = "rtps-p1-20261104-01a1a973-source-only";
pub const BASELINE_COMMIT: &str = "b55fd52d5043e7865a340c04f02433b0fa5dc974";
pub const W3_TREE: &str = "59e545e2d394c9eb159c3547cc826e41a5df9b97";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    Shape, Orientation, Stride, Tail, Bounds, Scale, FixtureBytes, Nonfinite,
    InputBits, StageOrder, Owner, RepeatedInvocation, MissingDriver,
    MissingResource, Enqueue, Fence, Readback, CleanupAmbiguous,
    CleanupRepeated, Terminal, NumericalBits,
}
pub type Result<T> = std::result::Result<T, Reject>;

/// There is deliberately no caller-supplied adapter, permit, PTX path, model
/// path or CPU fallback here. A real backend needs a separately reviewed patch.
pub fn native_once() -> report::TerminalReport {
    report::TerminalReport::native_unbound()
}

#[cfg(test)]
mod tests;
