//! Independent CPU scalar decoder: no fixture writer, vendor code, CUDA,
//! model library, stage hashes, GPU outputs or fast-math reduction is used.
//! Ascending column order, distinct f32 product/add, no explicit fused mul_add.
use crate::{Reject, Result, D, DFF, PROJECTION_BYTES, Q4_BYTES, STAGE_LENGTHS};

pub struct Stages { pub values: [Vec<f32>; 5] }

/// Decode binary16 exactly via integer significand and powers of two.
pub fn half(bits: u16) -> Result<f32> {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    if exponent == 31 { return Err(Reject::Scale); }
    let magnitude = if exponent == 0 {
        (fraction as f32) * 2.0f32.powi(-24)
    } else {
        ((1024 + fraction) as f32) * 2.0f32.powi(exponent as i32 - 25)
    };
    Ok(if bits & 0x8000 != 0 { -magnitude } else { magnitude })
}

pub fn finite(values: &[f32], expected_len: usize) -> Result<()> {
    if values.len() != expected_len { return Err(Reject::Shape); }
    if values.iter().any(|v| !v.is_finite()) { return Err(Reject::Nonfinite); }
    Ok(())
}

pub fn scalar(matrix: &[u8], rows: usize, cols: usize, row: usize, col: usize) -> Result<f32> {
    if !matches!((rows, cols), (768, 2048) | (2048, 768)) { return Err(Reject::Shape); }
    if matrix.len() != PROJECTION_BYTES { return Err(Reject::Tail); }
    if row >= rows || col >= cols { return Err(Reject::Bounds); }
    let base = (row * (cols / 32) + col / 32) * 18;
    let scale = half(u16::from_le_bytes([matrix[base], matrix[base + 1]]))?;
    let lane = col % 32;
    let packed = matrix[base + 2 + lane % 16];
    let quant = if lane < 16 { packed & 15 } else { packed >> 4 };
    Ok((quant as i32 - 8) as f32 * scale)
}

pub fn matvec(matrix: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    if !matches!((rows, cols), (768, 2048) | (2048, 768)) { return Err(Reject::Shape); }
    if matrix.len() != PROJECTION_BYTES { return Err(Reject::Tail); }
    finite(input, cols)?;
    let mut output = vec![0.0f32; rows];
    for row in 0..rows {
        let mut sum = 0.0f32;
        for col in 0..cols {
            let product = scalar(matrix, rows, cols, row, col)? * input[col];
            sum += product;
            if !product.is_finite() || !sum.is_finite() { return Err(Reject::Nonfinite); }
        }
        output[row] = sum;
    }
    Ok(output)
}

/// Input bytes are decoded as three raw projections, with no 4096-byte header.
/// The caller must separately establish fixture identity; this is arithmetic,
/// not an authentication or native-execution interface.
pub fn evaluate(bytes: &[u8], input: &[f32]) -> Result<Stages> {
    if bytes.len() != Q4_BYTES { return Err(Reject::Tail); }
    finite(input, D)?;
    let gate = matvec(&bytes[..PROJECTION_BYTES], DFF, D, input)?;
    let up = matvec(&bytes[PROJECTION_BYTES..2 * PROJECTION_BYTES], DFF, D, input)?;
    let mut silu = Vec::with_capacity(DFF);
    let mut product = Vec::with_capacity(DFF);
    for index in 0..DFF {
        let exponential = (-gate[index]).exp();
        let denominator = 1.0f32 + exponential;
        if !exponential.is_finite() || !denominator.is_finite() { return Err(Reject::Nonfinite); }
        let activation = gate[index] / denominator;
        silu.push(activation);
        product.push(activation * up[index]);
    }
    finite(&silu, DFF)?;
    finite(&product, DFF)?;
    let down = matvec(&bytes[2 * PROJECTION_BYTES..], D, DFF, &product)?;
    let stages = Stages { values: [gate, up, silu, product, down] };
    for (values, len) in stages.values.iter().zip(STAGE_LENGTHS) { finite(values, len)?; }
    Ok(stages)
}
