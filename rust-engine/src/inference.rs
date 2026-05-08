//! Placeholder inference function.
//!
//! This stands in for the matrix multiplications a real MoE expert would
//! perform. We compute a trivial digest over the expert bytes so that:
//!
//! * The compiler can't optimise the read away (the buffer must really be
//!   touched in RAM).
//! * The cache-fill path is exercised end-to-end: pool → io_uring → DMA →
//!   inference.
//!
//! Replace with `tch::nn::Module::forward` / `candle::Tensor::matmul` /
//! `cudarc` kernels when wiring real weights.

use crate::expert_cache::ExpertResident;

#[derive(Debug, Clone, Copy)]
pub struct InferenceOutput {
    pub expert_id: u32,
    pub digest: u64,
}

pub fn run_inference(token_idx: u64, resident: &ExpertResident) -> InferenceOutput {
    // 64-bit FNV-1a over a strided window of the buffer. Touches every page
    // (so we observe the actual NVMe DMA), but stays cheap so we measure I/O,
    // not compute.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let bytes = resident.data();
    let mut digest = FNV_OFFSET ^ token_idx;
    // Stride by 4 KiB so we touch one byte per page; that's the work we'd
    // need to do anyway to fault every page in.
    let stride = 4096usize.min(bytes.len().max(1));
    let mut i = 0;
    while i < bytes.len() {
        digest ^= bytes[i] as u64;
        digest = digest.wrapping_mul(FNV_PRIME);
        i = i.saturating_add(stride);
    }
    InferenceOutput {
        expert_id: resident.id,
        digest,
    }
}
