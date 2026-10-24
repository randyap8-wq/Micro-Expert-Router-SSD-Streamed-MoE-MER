//! MER2 G3-S0: independent synthetic CPU reference. No CUDA implementation.
#![forbid(unsafe_code)]

pub mod identity;
pub mod reference;

pub const BASE: &str = "ec8eea2edd6ea7a577d2f6de3e99d074462b00e1";
pub const SCHEMA: &str = "MER2-G3-S0-FIXTURE-v1";
pub const SEED: &str = "MER2-G3-S0-20261022-v1";
pub const VOCAB_ID: &str = "MER2-G3-SYNTH-VOCAB-v1";
pub const KIND: &str = "CPU_REFERENCE_ONLY";
pub const LAYERS: usize = 2;
pub const D: usize = 64;
pub const DFF: usize = 64;
pub const VOCAB: usize = 32;
pub const HEADS: usize = 4;
pub const HEAD_DIM: usize = 16;
pub const MAX_CONTEXT: usize = 4;
pub const EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
pub const Q4_BYTES: usize = 2304;
pub const RMS_EPS: f32 = 1e-5f32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stop {
    pub code: &'static str,
    pub detail: String,
}
pub type Result<T> = std::result::Result<T, Stop>;

impl Stop {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}
impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for Stop {}

pub fn require(ok: bool, code: &'static str, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Stop::new(code, why))
    }
}
pub fn finite(values: &[f32]) -> Result<()> {
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(Stop::new("STOP_NONFINITE", format!("scalar {i}")));
    }
    Ok(())
}
pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
pub fn cuda_unavailable() -> Result<()> {
    Err(Stop::new(
        "STOP_CUDA_UNAVAILABLE",
        "S0 has no CUDA feature, adapter, or CPU fallback for CUDA mode",
    ))
}
