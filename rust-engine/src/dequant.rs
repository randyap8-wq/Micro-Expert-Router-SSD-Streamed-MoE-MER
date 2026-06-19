//! Standalone weight dequantisation kernels for the formats that do not
//! fit the GGUF block-quant family handled in [`crate::inference`] or the
//! DeepSeek FP8 E4M3 path in [`crate::mla`].
//!
//! This module covers:
//!
//! * **MXFP4 / OCP Microscaling FP4** (E2M1 elements + E8M0 block scales),
//!   the weight format used by `openai/gpt-oss-20b` / `gpt-oss-120b`.
//! * **FP8 E5M2** (`safetensors` `Dtype::F8_E5M2`), an activation-oriented
//!   8-bit float that some future checkpoints may use for weights.
//!
//! -------------------------------------------------------------------
//! ## Discrepancies vs the design spec (reality wins)
//!
//! The spec asked that the GPT-OSS `model.safetensors.index.json` and the
//! MiMo-V2-Flash config be fetched from HuggingFace before coding, to pin
//! down exact tensor names and dtypes. **HuggingFace (and the public
//! internet) is not reachable from this build sandbox** (`huggingface.co`
//! fails to resolve), so the implementation below follows the published
//! OCP Microscaling spec and the documented GPT-OSS layout:
//!
//! * GPT-OSS MoE weights ship as `safetensors` **`U8`** tensors. The
//!   packed 4-bit weights live in a `*_blocks` tensor and the per-block
//!   E8M0 scales in a sibling `*_scales` tensor (both `U8`). The spec's
//!   single `<name>_scale` suffix is therefore treated as one of several
//!   candidate suffixes (`_scale`, `_scales`, `.scale`, `_blocks` →
//!   `_scales`) when scanning the shards in [`crate::model`]. Whichever
//!   companion is found first and shape-matches `[rows, ceil(cols/32)]`
//!   wins.
//! * Attention / norm / embedding tensors in both GPT-OSS and
//!   MiMo-V2-Flash are **BF16**, which already decodes through
//!   [`crate::model::decode_safetensor_to_f32`]; the new
//!   [`crate::inference::WeightDtype::BF16`] variant only affects native
//!   expert `.bin` storage.
//! * Some `U8` tensors in GPT-OSS (e.g. attention-sink scale vectors) are
//!   **not** packed weight matrices; [`crate::model`] skips them with a
//!   `debug!` (not `warn!`) when their element count equals the full
//!   `rows * cols` rather than the packed `rows * cols / 2`.
//!
//! If/when the real weight index can be inspected, only the companion
//! scale-tensor name scan in [`crate::model`] should need adjusting — the
//! numeric kernels here are spec-exact and model-agnostic.

/// E2M1 (MXFP4) nibble → f32 decode table.
///
/// Each entry is a 4-bit value `[0, 15]` interpreted as a sign bit, two
/// exponent bits (bias 1) and one mantissa bit; zero exponent encodes a
/// subnormal. The 16 values are constant, so the mapping is a pure
/// lookup with **no runtime floating-point arithmetic**.
///
/// ```text
/// nibble  sign exp mantissa  value
/// 0x0      0   0     0        0.0
/// 0x1      0   0     1        0.5
/// 0x2      0   1     0        1.0
/// 0x3      0   1     1        1.5
/// 0x4      0   2     0        2.0
/// 0x5      0   2     1        3.0
/// 0x6      0   3     0        4.0
/// 0x7      0   3     1        6.0
/// 0x8      1   0     0       -0.0
/// 0x9      1   0     1       -0.5
/// 0xA      1   1     0       -1.0
/// 0xB      1   1     1       -1.5
/// 0xC      1   2     0       -2.0
/// 0xD      1   2     1       -3.0
/// 0xE      1   3     0       -4.0
/// 0xF      1   3     1       -6.0
/// ```
pub const MXFP4_E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one MXFP4 E2M1 nibble (a 4-bit value in `[0, 15]`) to f32.
///
/// Only the low four bits of `nibble` are consulted; any higher bits are
/// masked off so callers can pass a raw byte half without pre-masking.
#[inline]
pub const fn decode_mxfp4_nibble(nibble: u8) -> f32 {
    MXFP4_E2M1_TABLE[(nibble & 0x0F) as usize]
}

/// Decode one E8M0 block-scale byte to its f32 multiplier.
///
/// E8M0 is a pure power-of-two encoding: the byte `e` represents
/// `2^(e - 127)`, with `e == 0` reserved to mean "the whole block is
/// zero" (the multiplier is `0.0`). This matches the OCP Microscaling
/// scale format used by MXFP4.
#[inline]
pub fn decode_e8m0_scale(e: u8) -> f32 {
    if e == 0 {
        0.0
    } else {
        // For f32, exponent bits directly encode powers of two: 2^(e-127).
        f32::from_bits((e as u32) << 23)
    }
}

/// Dequantise an MXFP4 weight tensor to f32, row-major.
///
/// * `packed` — `rows * cols / 2` bytes, two E2M1 elements per byte,
///   **low nibble first** (element `2k` in bits `[3:0]`, element `2k+1`
///   in bits `[7:4]`). `cols` must be even, so every row begins on a
///   byte boundary.
/// * `scales` — `rows * ceil(cols / 32)` E8M0 scale bytes, row-major over
///   the block grid (one scale per 32 consecutive elements within a row).
/// * `rows`, `cols` — logical weight-matrix dimensions.
///
/// Returns the dequantised `rows * cols` f32 matrix, or an empty vector
/// when the inputs are inconsistent (odd `cols`, or `packed` / `scales`
/// lengths that don't match the dimensions) so callers can fall back to
/// seeded init rather than panic on a malformed checkpoint.
pub fn dequant_mxfp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    // `cols` must be even so every row begins on a byte boundary (two E2M1
    // elements per packed byte). Reject odd `cols` per the documented
    // contract rather than silently mis-packing the final nibble.
    if cols % 2 != 0 {
        return Vec::new();
    }
    let blocks_per_row = cols.div_ceil(32);
    let bytes_per_row = cols.div_ceil(2);
    let need_packed = rows.saturating_mul(bytes_per_row);
    let need_scales = rows.saturating_mul(blocks_per_row);
    if packed.len() != need_packed || scales.len() != need_scales {
        return Vec::new();
    }
    let mut out = vec![0.0f32; rows.saturating_mul(cols)];
    for r in 0..rows {
        let packed_row = r * bytes_per_row;
        let scale_row = r * blocks_per_row;
        for c in 0..cols {
            let byte = packed[packed_row + c / 2];
            let nibble = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let scale = decode_e8m0_scale(scales[scale_row + c / 32]);
            let n = r * cols + c;
            out[n] = decode_mxfp4_nibble(nibble) * scale;
        }
    }
    out
}

/// Decode one FP8 `e5m2` byte (1 sign, 5 exponent bits bias 15, 2
/// mantissa bits) to f32.
///
/// Unlike the `e4m3fn` format in [`crate::mla`], E5M2 follows IEEE-style
/// specials: exponent all-ones with a non-zero mantissa is NaN, all-ones
/// with a zero mantissa is `±Inf`, and a zero exponent encodes
/// subnormals `(-1)^s * 2^(-14) * (m / 4)`.
pub fn f8_e5m2_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let mant = (b & 0x03) as u32;
    if exp == 0x1F {
        // All-ones exponent: Inf (mant == 0) or NaN (mant != 0).
        if mant != 0 {
            return f32::NAN;
        }
        return sign * f32::INFINITY;
    }
    if exp == 0 {
        if mant == 0 {
            return sign * 0.0;
        }
        // Subnormal: value = (m / 4) * 2^(1 - bias), bias = 15.
        return sign * (mant as f32 / 4.0) * 2f32.powi(-14);
    }
    // Normal: value = (1 + m / 4) * 2^(exp - bias).
    sign * (1.0 + mant as f32 / 4.0) * 2f32.powi(exp - 15)
}

/// Element-wise dequantise an FP8 `e5m2` byte stream to f32.
///
/// E5M2 weights carry no companion block-scale tensor in current models
/// (the format is used for activations), so each byte is decoded
/// standalone via [`f8_e5m2_to_f32`].
pub fn dequant_fp8_e5m2(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| f8_e5m2_to_f32(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_nibble_matches_reference_table() {
        let expected = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        for (n, &want) in expected.iter().enumerate() {
            let got = decode_mxfp4_nibble(n as u8);
            assert_eq!(got.to_bits(), want.to_bits(), "nibble 0x{n:X}");
        }
        // High bits are masked off.
        assert_eq!(decode_mxfp4_nibble(0xF0).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_mxfp4_nibble(0xF7), 6.0);
    }

    #[test]
    fn e8m0_scale_decodes_powers_of_two() {
        assert_eq!(decode_e8m0_scale(0), 0.0); // reserved: block is zero
        assert_eq!(decode_e8m0_scale(127), 1.0); // 2^0
        assert_eq!(decode_e8m0_scale(128), 2.0); // 2^1
        assert_eq!(decode_e8m0_scale(126), 0.5); // 2^-1
        assert_eq!(decode_e8m0_scale(124), 0.125); // 2^-3
    }

    #[test]
    fn dequant_mxfp4_two_blocks_per_row() {
        // 2 rows x 64 cols => 2 blocks (of 32) per row.
        // Build packed bytes where, within each row, element index i
        // decodes to nibble (i % 16). cols/2 = 32 bytes per row.
        let rows = 2;
        let cols = 64;
        let mut packed = vec![0u8; rows * cols / 2];
        for r in 0..rows {
            for c in 0..cols {
                let n = r * cols + c;
                let nib = (c % 16) as u8;
                if n % 2 == 0 {
                    packed[n / 2] |= nib & 0x0F;
                } else {
                    packed[n / 2] |= (nib & 0x0F) << 4;
                }
            }
        }
        // Scales: row 0 => [2^0, 2^1], row 1 => [2^-1, 0 (zero block)].
        let scales = vec![127u8, 128u8, 126u8, 0u8];
        let out = dequant_mxfp4(&packed, &scales, rows, cols);
        assert_eq!(out.len(), rows * cols);

        for c in 0..cols {
            let want_base = MXFP4_E2M1_TABLE[c % 16];
            // Row 0, block 0 (c < 32) scale = 1.0; block 1 scale = 2.0.
            let s0 = if c < 32 { 1.0 } else { 2.0 };
            assert_eq!(out[c], want_base * s0, "row0 col{c}");
            // Row 1, block 0 scale = 0.5; block 1 scale = 0.0 (zeroed).
            let s1 = if c < 32 { 0.5 } else { 0.0 };
            assert_eq!(out[cols + c], want_base * s1, "row1 col{c}");
        }
    }

    #[test]
    fn dequant_mxfp4_rejects_inconsistent_shapes() {
        assert!(dequant_mxfp4(&[0u8; 4], &[127u8], 1, 7).is_empty()); // odd cols
        assert!(dequant_mxfp4(&[0u8; 3], &[127u8], 1, 8).is_empty()); // wrong packed len
        assert!(dequant_mxfp4(&[0u8; 4], &[127u8, 1], 1, 8).is_empty()); // wrong scale len
    }

    #[test]
    fn fp8_e5m2_normals_subnormals_and_specials() {
        // +0 / -0
        assert_eq!(f8_e5m2_to_f32(0x00), 0.0);
        assert_eq!(f8_e5m2_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
        // 1.0 = exp 15 (bias), mant 0 => 0.01111.00
        assert_eq!(f8_e5m2_to_f32(0b0_01111_00), 1.0);
        // 1.5 = (1 + 2/4) * 2^0 => mant = 0b10
        assert_eq!(f8_e5m2_to_f32(0b0_01111_10), 1.5);
        // 2.0 = exp 16 => 0b0_10000_00
        assert_eq!(f8_e5m2_to_f32(0b0_10000_00), 2.0);
        // -1.0
        assert_eq!(f8_e5m2_to_f32(0b1_01111_00), -1.0);
        // Smallest subnormal: exp 0, mant 1 => 2^-14 * (1/4) = 2^-16.
        assert_eq!(f8_e5m2_to_f32(0b0_00000_01), 2f32.powi(-16));
        // Largest subnormal: exp 0, mant 3 => 2^-14 * (3/4).
        assert_eq!(f8_e5m2_to_f32(0b0_00000_11), 2f32.powi(-14) * 0.75);
        // +Inf / -Inf: exp all ones, mant 0.
        assert!(f8_e5m2_to_f32(0b0_11111_00).is_infinite() && f8_e5m2_to_f32(0b0_11111_00) > 0.0);
        assert!(f8_e5m2_to_f32(0b1_11111_00).is_infinite() && f8_e5m2_to_f32(0b1_11111_00) < 0.0);
        // NaN: exp all ones, mant != 0.
        assert!(f8_e5m2_to_f32(0b0_11111_01).is_nan());
        assert!(f8_e5m2_to_f32(0b0_11111_11).is_nan());
    }

    #[test]
    fn dequant_fp8_e5m2_maps_each_byte() {
        let data = [0x00u8, 0b0_01111_00, 0b1_01111_00];
        let out = dequant_fp8_e5m2(&data);
        assert_eq!(out, vec![0.0, 1.0, -1.0]);
    }
}
