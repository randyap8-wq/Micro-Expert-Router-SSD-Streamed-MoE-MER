use half::f16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[cfg(feature = "cuda")]
pub mod cuda;

pub const BASE: &str = "fec4d5f9e6304e0642990e9debe5d7449a8def38";
pub const KERNEL: &str = "dequantize_mul_mat_vec_q4_0_cuda";
pub const KERNEL_SHA256: &str = "510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972";
// Fixed before any GPU result. No command-line or environment overrides.
pub const ATOL: f64 = 1e-4;
pub const RTOL: f64 = 1e-5;

#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub weights: Vec<u8>,
    pub input: Vec<f32>,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Byte layout matches inference.rs::dequantize_q4_0_block at BASE.
pub fn dequantize(block: &[u8]) -> Result<[f32; 32], String> {
    if block.len() != 18 {
        return Err("Q4_0 requires exactly 18 bytes per block".into());
    }
    let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    if !scale.is_finite() {
        return Err("nonfinite f16 scale".into());
    }
    let mut out = [0.0; 32];
    for j in 0..16 {
        out[j] = scale * ((block[2 + j] & 15) as i32 - 8) as f32;
        out[j + 16] = scale * ((block[2 + j] >> 4) as i32 - 8) as f32;
    }
    Ok(out)
}

impl Fixture {
    pub fn validate(&self) -> Result<(), String> {
        // Deliberately tiny bounds also prevent C int indexing overflow.
        if self.rows == 0 || self.rows > 32 || self.cols == 0 || self.cols > 4096 {
            return Err("shape outside the synthetic experiment bounds".into());
        }
        if self.cols % 32 != 0
            || self.weights.len() != self.rows * (self.cols / 32) * 18
            || self.input.len() != self.cols
        {
            return Err("invalid packed matrix or input shape".into());
        }
        if self.input.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite input".into());
        }
        for block in self.weights.chunks_exact(18) {
            dequantize(block)?;
        }
        Ok(())
    }

    pub fn cuda_dimensions(&self) -> Result<(i32, i32), String> {
        self.validate()?;
        if !cfg!(target_endian = "little") {
            return Err("CUDA Q4_0 probe requires a little-endian host".into());
        }
        // quantized.cu uses a 64-column iteration with no tail predicate.
        if self.cols % 64 != 0 {
            return Err("CUDA kernel requires ncols divisible by 64".into());
        }
        Ok((self.cols as i32, self.rows as i32))
    }

    pub fn reference(&self) -> Result<Vec<f32>, String> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.rows);
        for row in self.weights.chunks_exact(self.cols / 32 * 18) {
            let mut sum = 0.0f32;
            for (b, block) in row.chunks_exact(18).enumerate() {
                for (j, weight) in dequantize(block)?.iter().enumerate() {
                    sum += weight * self.input[b * 32 + j];
                }
            }
            if !sum.is_finite() {
                return Err("nonfinite CPU reference".into());
            }
            out.push(sum);
        }
        Ok(out)
    }

    /// v1 bytes: domain + LE u32 rows/cols + packed weights + LE f32 input.
    pub fn fingerprint(&self) -> String {
        let mut bytes = b"MER2-CUDA-S1-fixture-v1\0".to_vec();
        bytes.extend((self.rows as u32).to_le_bytes());
        bytes.extend((self.cols as u32).to_le_bytes());
        bytes.extend(&self.weights);
        for x in &self.input {
            bytes.extend(x.to_le_bytes());
        }
        sha256(&bytes)
    }
}

/// Fixed synthetic inputs; no filesystem, model, random seed or device access.
pub fn fixtures() -> Vec<Fixture> {
    [
        ("zero", 1, 64),
        ("alternating", 3, 64),
        ("mixed", 5, 128),
        ("wide", 2, 192),
    ]
    .into_iter()
    .enumerate()
    .map(|(kind, (name, rows, cols))| {
        let mut weights = Vec::with_capacity(rows * cols / 32 * 18);
        for b in 0..rows * cols / 32 {
            let scale_bits: u16 = [0x3800, 0xb400, 0x3e00, 0xbc00, 0x3000][b % 5];
            weights.extend(scale_bits.to_le_bytes());
            for j in 0..16 {
                weights.push(match kind {
                    0 => 0x88,
                    1 if j % 2 == 0 => 0xf0,
                    1 => 0x0f,
                    _ => (((j * 3 + b * 5 + 1) % 16) as u8) << 4 | ((j * 7 + b * 3) % 16) as u8,
                });
            }
        }
        Fixture {
            name,
            rows,
            cols,
            weights,
            input: (0..cols)
                .map(|j| (((j * 7 + 3) % 29) as i32 - 14) as f32 / 13.0)
                .collect(),
        }
    })
    .collect()
}

/// Compare in f64 to avoid overflow in the acceptance calculation itself.
pub fn compare(reference: &[f32], actual: &[f32]) -> Result<f64, String> {
    if reference.is_empty() || reference.len() != actual.len() {
        return Err("empty or mismatched output shape".into());
    }
    let mut max_error = 0.0f64;
    for (row, (&r, &a)) in reference.iter().zip(actual).enumerate() {
        if !r.is_finite() || !a.is_finite() {
            return Err(format!("nonfinite output at row {row}"));
        }
        let error = (f64::from(a) - f64::from(r)).abs();
        let limit = ATOL + RTOL * f64::from(r).abs();
        max_error = max_error.max(error);
        if error > limit {
            return Err(format!("row {row}: abs_error={error} exceeds {limit}"));
        }
    }
    Ok(max_error)
}

pub fn source_identity() -> Result<Value, String> {
    let actual = sha256(include_bytes!(
        "../../../vendor/candle-kernels/src/quantized.cu"
    ));
    if actual != KERNEL_SHA256 {
        return Err("STOP_SOURCE_API: quantized.cu hash changed".into());
    }
    Ok(json!({
        "frozen_parent": BASE,
        "kernel": KERNEL,
        "kernel_source_sha256": actual,
        "candle_kernels_version": "0.10.2",
        "cudarc_version": "0.19.7",
        "experiment_sha256": {
            "Cargo.toml": sha256(include_bytes!("../Cargo.toml")),
            "Cargo.lock": sha256(include_bytes!("../Cargo.lock")),
            "src/lib.rs": sha256(include_bytes!("lib.rs")),
            "src/main.rs": sha256(include_bytes!("main.rs")),
            "src/cuda.rs": sha256(include_bytes!("cuda.rs")),
        },
    }))
}

pub fn preflight() -> Result<Value, String> {
    let mut cases = Vec::new();
    for fixture in fixtures() {
        fixture.cuda_dimensions()?;
        let reference = fixture.reference()?;
        cases.push(json!({
            "name": fixture.name,
            "rows": fixture.rows,
            "cols": fixture.cols,
            "fixture_sha256": fixture.fingerprint(),
            "reference": reference,
            "reference_f32_bits": reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        }));
    }
    Ok(json!({
        "schema": "MER2-CUDA-S1-v1",
        "source": source_identity()?,
        "atol": ATOL,
        "rtol": RTOL,
        "acceptance": "abs(actual-reference) <= atol + rtol*abs(reference), all finite",
        "fixtures": cases,
        "CUDA_GPU_TEST": "NOT_RUN",
    }))
}
