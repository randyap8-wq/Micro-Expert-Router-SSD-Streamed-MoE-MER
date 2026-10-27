//! Draft A3N1 CPU-only mock child-process wire boundary.
//!
//! No GPU initialization, native CUDA adapter, Qwen model, and no assertion of
//! durable replay protection or last CUDA context destruction.
#![forbid(unsafe_code)]

pub mod contract;
pub mod wire;

pub const D: usize = 64;
pub const VECTOR_BYTES: usize = D * 4;
pub const MOCK_KIND: &str = "A3N1_CPU_MOCK_UNQUALIFIED";
pub const R4_BASE_COMMIT: &str = "2236db82a5f22c0d9793084e89de18647d7ebdbd";

pub type Result<T> = std::result::Result<T, &'static str>;

pub fn native_cuda_unavailable() -> Result<()> {
    Err("STOP_CUDA_NOT_AUTHORIZED")
}
