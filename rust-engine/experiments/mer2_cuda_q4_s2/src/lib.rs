//! Isolated synthetic Q4_0 SwiGLU contract; the default build is CPU-only.
pub use mer2_cuda_q4_s1::{compare, dequantize, sha256, ATOL, KERNEL, RTOL};
use serde_json::{json, Value};

#[cfg(feature = "cuda")]
pub mod cuda;
mod identity;

pub const BASE: &str = "be19b4217ba5e6b5bb963663bf3c3cf88566006f";
pub const SILU: &str = "usilu_f32";
pub const MUL: &str = "bmul_f32";

/// Encode exact signed Q4 levels, not a lossy float-to-Q4 quantizer.
/// The scale is already an IEEE f16 bit pattern; low j / high j+16.
pub fn pack_block(scale: u16, levels: &[i8; 32]) -> Result<Vec<u8>, String> {
    if levels.iter().any(|q| !(-8..=7).contains(q)) {
        return Err("Q4 level outside -8..=7".into());
    }
    let mut bytes = scale.to_le_bytes().to_vec();
    for j in 0..16 {
        bytes.push((levels[j] + 8) as u8 | (((levels[j + 16] + 8) as u8) << 4));
    }
    dequantize(&bytes)?;
    Ok(bytes)
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub packed: Vec<u8>,
}

fn dimension(n: usize) -> bool {
    matches!(n, 64 | 128)
}

impl Matrix {
    pub fn validate(&self) -> Result<(), String> {
        // Check bounds BEFORE any multiplication/cast; all C int indices fit.
        if !dimension(self.rows) || !dimension(self.cols) {
            return Err("synthetic dimensions must be 64 or 128".into());
        }
        if self.packed.len() != self.rows * (self.cols / 32) * 18 {
            return Err("incorrect row-major Q4_0 byte length".into());
        }
        for block in self.packed.chunks_exact(18) {
            dequantize(block)?;
        }
        Ok(())
    }

    pub fn reference(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.validate()?;
        if input.len() != self.cols || input.iter().any(|v| !v.is_finite()) {
            return Err("invalid projection input".into());
        }
        let mut output = Vec::with_capacity(self.rows);
        for row in self.packed.chunks_exact(self.cols / 32 * 18) {
            let mut sum = 0.0f32;
            for (b, block) in row.chunks_exact(18).enumerate() {
                for (j, w) in dequantize(block)?.iter().enumerate() {
                    sum += w * input[b * 32 + j];
                }
            }
            if !sum.is_finite() {
                return Err("nonfinite projection reference".into());
            }
            output.push(sum);
        }
        Ok(output)
    }

    // Conservative absolute bound, independent of cancellation/reduction order.
    fn bound(&self, input_bound: f64) -> Result<f64, String> {
        let mut maximum = 0.0f64;
        for row in self.packed.chunks_exact(self.cols / 32 * 18) {
            let mut l1 = 0.0f64;
            for block in row.chunks_exact(18) {
                for w in dequantize(block)? {
                    l1 += f64::from(w).abs();
                }
            }
            maximum = maximum.max(l1 * input_bound);
        }
        finite_bound(maximum)
    }
}

fn finite_bound(value: f64) -> Result<f64, String> {
    // Deliberate headroom for f32 sums; not a numerical acceptance tolerance.
    if !value.is_finite() || value > f64::from(f32::MAX) / 16.0 {
        Err("conservative f32 intermediate overflow guard".into())
    } else {
        Ok(value)
    }
}

pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    if gate.is_empty() || gate.len() != up.len() {
        return Err("invalid SwiGLU shape".into());
    }
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            if !g.is_finite() || !u.is_finite() {
                return Err("nonfinite SwiGLU input".into());
            }
            let h = (g / (1.0f32 + (-g).exp())) * u;
            if h.is_finite() {
                Ok(h)
            } else {
                Err("nonfinite SwiGLU result".into())
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
    pub input: Vec<f32>,
}

impl Fixture {
    pub fn validate(&self) -> Result<(), String> {
        for matrix in [&self.gate, &self.up, &self.down] {
            matrix.validate()?;
        }
        let (d_model, d_ff) = (self.gate.cols, self.gate.rows);
        if self.up.rows != d_ff
            || self.up.cols != d_model
            || self.down.rows != d_model
            || self.down.cols != d_ff
            || self.input.len() != d_model
        {
            return Err("incompatible three-matrix geometry".into());
        }
        if self.input.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite expert input".into());
        }
        let x = self
            .input
            .iter()
            .fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
        let g = self.gate.bound(x)?;
        let u = self.up.bound(x)?;
        // |SiLU(g)| <= |g|. Twice each stage bound covers accumulated rounding.
        let hidden = finite_bound((2.0 * g) * (2.0 * u))?;
        self.down.bound(2.0 * hidden)?;
        Ok(())
    }

    pub fn cuda_dimensions(&self) -> Result<(i32, i32), String> {
        self.validate()?;
        if !cfg!(target_endian = "little") || usize::BITS != 64 {
            return Err("reviewed CUDA ABI requires little-endian 64-bit host".into());
        }
        Ok((self.gate.cols as i32, self.gate.rows as i32))
    }

    /// Separate diagnostic oracle; never a down input on the CUDA path.
    pub fn reference(&self) -> Result<Vec<f32>, String> {
        self.validate()?;
        let gate = self.gate.reference(&self.input)?;
        let up = self.up.reference(&self.input)?;
        self.down.reference(&swiglu(&gate, &up)?)
    }

    /// Domain/NUL, each matrix's LE u32 rows/cols + packed bytes in gate/up/down
    /// order, then LE u32 input length and all LE f32 input bits. Name excluded.
    pub fn fingerprint(&self) -> String {
        let mut bytes = b"MER2-CUDA-S2-fixture-v1\0".to_vec();
        for m in [&self.gate, &self.up, &self.down] {
            bytes.extend((m.rows as u32).to_le_bytes());
            bytes.extend((m.cols as u32).to_le_bytes());
            bytes.extend(&m.packed);
        }
        bytes.extend((self.input.len() as u32).to_le_bytes());
        for x in &self.input {
            bytes.extend(x.to_le_bytes());
        }
        sha256(&bytes)
    }

    pub fn trace(&self) -> Result<Value, String> {
        self.cuda_dimensions()?;
        let (dm, df) = (self.gate.cols, self.gate.rows);
        let weights = self.gate.packed.len() + self.up.packed.len() + self.down.packed.len();
        let temporaries = 4 * df * 4;
        let sentinels = temporaries + dm * 4;
        Ok(json!({
            "kind": "EXPECTED_LOGICAL_BYTES_NOT_MEASURED",
            "h2d": {"gate_q4": self.gate.packed.len(), "up_q4": self.up.packed.len(),
                "down_q4": self.down.packed.len(), "input_f32": dm * 4,
                "nan_sentinel_f32": sentinels, "layout_u64": 24,
                "total": weights + dm * 4 + sentinels + 24},
            "device_temporaries": {"gate_f32": df * 4, "up_f32": df * 4,
                "silu_f32": df * 4, "hidden_f32": df * 4, "total": temporaries},
            "device_output_f32": dm * 4,
            "device_allocation_payload_total": weights + dm * 4 + sentinels + 24,
            "d2h": {"final_y_f32": dm * 4, "intermediates": 0},
            "launches": [KERNEL, KERNEL, SILU, MUL, KERNEL],
            "q4_grid_x": [df, df, dm], "q4_block": [32, 1, 1],
            "elementwise_grid": [1, 1, 1], "elementwise_block": [128, 1, 1],
            "layout_u64": [df, 1, 1], "num_dims_u64": 1,
            "stream": "single_default_stream", "diagnostic_d2h": false,
            "timing": "NOT_MEASURED", "vram": "NOT_MEASURED"
        }))
    }
}

pub fn fixtures() -> Vec<Fixture> {
    [
        ("zero", 64, 64),
        ("alternating", 128, 64),
        ("sign", 64, 128),
        ("mixed", 128, 128),
    ]
    .into_iter()
    .enumerate()
    .map(|(kind, (name, dm, df))| {
        let matrix = |rows: usize, cols: usize, projection: usize| {
            let mut packed = Vec::with_capacity(rows * cols / 32 * 18);
            for b in 0..rows * cols / 32 {
                let scale = [0x2400u16, 0xa000, 0x2800, 0xa400][(b + projection) % 4];
                let mut levels = [0i8; 32];
                for (j, q) in levels.iter_mut().enumerate() {
                    *q = match kind {
                        0 => 0,
                        1 => {
                            if (j + b + projection) % 2 == 0 {
                                -8
                            } else {
                                7
                            }
                        }
                        2 => {
                            if (j / 8 + b + projection) % 2 == 0 {
                                -5
                            } else {
                                3
                            }
                        }
                        _ => ((j * 7 + b * 3 + projection * 5 + j / 16) % 16) as i8 - 8,
                    };
                }
                packed.extend(pack_block(scale, &levels).expect("fixed exact Q4 levels"));
            }
            Matrix { rows, cols, packed }
        };
        Fixture {
            name,
            gate: matrix(df, dm, 0),
            up: matrix(df, dm, 1),
            down: matrix(dm, df, 2),
            input: (0..dm)
                .map(|j| (((j * 7 + 3) % 29) as i32 - 14) as f32 / 16.0)
                .collect(),
        }
    })
    .collect()
}

pub fn preflight() -> Result<Value, String> {
    let mut cases = Vec::new();
    for fixture in fixtures() {
        let reference = fixture.reference()?;
        cases.push(json!({
            "name": fixture.name, "d_model": fixture.gate.cols, "d_ff": fixture.gate.rows,
            "fixture_sha256": fixture.fingerprint(), "reference": reference,
            "reference_f32_bits": reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "trace": fixture.trace()?
        }));
    }
    Ok(json!({
        "schema": "MER2-CUDA-S2-v1", "source": identity::source_identity()?,
        "atol": ATOL, "rtol": RTOL,
        "acceptance": "all finite; abs(actual-reference) <= atol + rtol*abs(reference), f64 comparison",
        "fixtures": cases, "CUDA_GPU_TEST": "NOT_RUN", "CUDA_S2_GPU_TEST": "NOT_RUN",
        "CUDA_S2_COMPILATION": "NOT_RUN", "gpu_test_invocations": 0,
        "scope": "synthetic expert only; no full token; no performance claim"
    }))
}
