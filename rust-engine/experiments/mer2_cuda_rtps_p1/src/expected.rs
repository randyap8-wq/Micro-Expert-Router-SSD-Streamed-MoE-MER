//! Pre-device mathematical oracle, authored without running fixture/reference.
//! Constants cover every stage element by index modulo four. They are not
//! generated from GPU results or by reusing the fixture encoder/CPU decoder.
use crate::{reference, Reject, Result, STAGE_LENGTHS};

pub const GATE: [u32; 4] = [0x4200_0000, 0x4280_0000, 0x42c0_0000, 0x4300_0000];
pub const UP: [u32; 4] = [0xc000_0000, 0x4080_0000, 0xc0c0_0000, 0x4100_0000];
pub const PRODUCT: [u32; 4] = [0xc280_0000, 0x4380_0000, 0xc410_0000, 0x4480_0000];
pub const DOWN: [u32; 4] = [0xc734_0000, 0x47b4_0000, 0xc807_0000, 0x4834_0000];

// All gate values are at least 32: round_f32(1 + exp(-gate)) = 1.
// Thus SiLU bits equal Gate bits for this fixture. This deliberately narrow
// geometry fixture CANNOT detect an identity substitution for the SiLU kernel.
// Native symbol/launch/order evidence and additional future numerical coverage
// are required; never promote this to general SiLU or P5D qualification.
pub fn bits(stage: usize, index: usize) -> Result<u32> {
    let len = *STAGE_LENGTHS.get(stage).ok_or(Reject::StageOrder)?;
    if index >= len { return Err(Reject::Bounds); }
    let pattern = match stage {
        0 | 2 => GATE,
        1 => UP,
        3 => PRODUCT,
        4 => DOWN,
        _ => return Err(Reject::StageOrder),
    };
    Ok(pattern[index % 4])
}

/// Exact bit equality is frozen for this dyadic fixture; no tolerance knob.
/// The scalar deltas are retained even when zero. +0/-0 remain different bits.
#[derive(Debug, PartialEq)]
pub struct Difference {
    pub expected_bits: u32,
    pub actual_bits: u32,
    pub signed_delta: f64,
    pub absolute_delta: f64,
    pub exact: bool,
}
pub fn difference(stage: usize, index: usize, actual: f32) -> Result<Difference> {
    if !actual.is_finite() { return Err(Reject::Nonfinite); }
    let expected_bits = bits(stage, index)?;
    let signed_delta = actual as f64 - f32::from_bits(expected_bits) as f64;
    Ok(Difference { expected_bits, actual_bits: actual.to_bits(), signed_delta,
        absolute_delta: signed_delta.abs(), exact: actual.to_bits() == expected_bits })
}

pub fn check_reference(stages: &reference::Stages) -> Result<()> {
    for (stage, values) in stages.values.iter().enumerate() {
        reference::finite(values, STAGE_LENGTHS[stage])?;
        for (index, value) in values.iter().enumerate() {
            if !difference(stage, index, *value)?.exact { return Err(Reject::NumericalBits); }
        }
    }
    Ok(())
}
