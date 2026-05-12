//! Real Mixtral / Llama-style **SwiGLU expert FFN** computed directly over
//! the bytes streamed from the SSD.
//!
//! Each expert file on disk is a flat blob of little-endian `f32` weights
//! laid out as three matrices, in this order:
//!
//! ```text
//!   gate_proj  : [d_ff   x d_model]   row-major
//!   up_proj    : [d_ff   x d_model]   row-major
//!   down_proj  : [d_model x d_ff  ]   row-major
//! ```
//!
//! The forward pass implemented here is the standard gated-MLP block used
//! by Mixtral, Llama-2/3, DeepSeek-V2/V3, Qwen-MoE, OLMoE, etc. — i.e. the
//! exact compute every routed expert in those models performs:
//!
//! ```text
//!   y = down_proj · ( silu(gate_proj · x)  ⊙  (up_proj · x) )
//! ```
//!
//! This is *the* per-expert kernel for sparse MoE transformers. Wiring it
//! here turns the engine from a pure I/O substrate into something that
//! actually executes a model-shaped computation over weights paged in from
//! the SSD — which is the entire point of streaming the weights in the
//! first place.
//!
//! ### Why pure Rust / scalar matmul?
//!
//! The point of this repository is to demonstrate that on modest hardware
//! you can keep the *active* parameter footprint streaming from SSD instead
//! of resident in DRAM. The compute kernel just has to be real enough to
//! exercise every byte that arrived from the drive. We therefore use a
//! straightforward scalar matmul + `silu` in `f32`, with no BLAS / SIMD
//! dependency: the resulting per-token cost is on the order of
//! `O(2 · d_model · d_ff)` MACs per expert, which is small enough on a
//! laptop that I/O remains observable but large enough that the compiler
//! cannot fold the read away. Swap this module for a `tch`/`candle`/`cudarc`
//! kernel when wiring real, larger weights — the I/O substrate around it
//! does not change.
//!
//! ### Why not mmap the bytes as `&[f32]` and hand them to a tensor lib?
//!
//! That is exactly the upgrade path. The buffers handed to this function
//! are page-aligned (the `O_DIRECT` invariant), so reinterpreting them as
//! `&[f32]` is sound — `align_of::<f32>() == 4` and we always allocate at
//! `≥ 4096`-byte alignment. See [`ExpertWeights::from_bytes`].

// The on-disk layout is documented as little-endian `f32`; we reinterpret
// the byte buffer as `&[f32]` in place, so the host's native endianness
// must match. Refuse to compile on big-endian targets rather than
// silently produce wrong weights.
#[cfg(not(target_endian = "little"))]
compile_error!(
    "inference module reinterprets on-disk little-endian f32 weights as \
     &[f32] in place; this only works on little-endian targets. Add an \
     explicit byte-swap conversion path before building for big-endian."
);

use crate::expert_cache::ExpertResident;
use std::fmt;

/// Bit-width with which expert weights are stored on disk.
///
/// `F32` is the legacy (and default) format: each weight is 4 bytes,
/// reinterpreted directly as `&[f32]`. `F16` halves bytes-per-parameter
/// (the dominant SSD-energy cost in this engine) and is dequantised into
/// an owned `Vec<f32>` at fetch time. `Int8` uses **per-tensor symmetric
/// quantization** with three `f32` scales (one each for `gate_proj`,
/// `up_proj`, `down_proj`) stored as a 12-byte header at the start of
/// every expert blob, followed by `i8` weights. This 4× compression
/// over F32 (and 2× over F16) is the dominant SSD-bandwidth win for
/// the Mixtral-scale workloads this engine targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightDtype {
    /// Little-endian `f32`, 4 bytes per weight.
    #[serde(alias = "F32", alias = "fp32")]
    F32,
    /// Little-endian IEEE-754 `f16` (`half::f16`), 2 bytes per weight.
    #[serde(alias = "F16", alias = "fp16", alias = "half")]
    F16,
    /// Per-tensor symmetric `int8`, 1 byte per weight, with a 12-byte
    /// header (`[gate_scale, up_scale, down_scale]: [f32; 3]`) at the
    /// start of every expert blob. Dequantised to `f32` at fetch time.
    #[serde(alias = "Int8", alias = "i8", alias = "q8")]
    Int8,
    /// **GGUF-style Q4_K_M block quantisation.** Each 256-weight
    /// super-block occupies 144 bytes: an `f16` super-scale `d`, an
    /// `f16` super-min `dmin`, 12 bytes of 6-bit packed sub-block
    /// scales and mins (8 of each), and 128 bytes of 4-bit packed
    /// weights. Effective bytes-per-weight = 144 / 256 = 0.5625, which
    /// roughly **doubles** the SSD-streamed expert capacity that fits
    /// in a given RAM budget versus `F16`. See [`Q4K_BLOCK_BYTES`] /
    /// [`Q4K_BLOCK_ELEMS`] for the layout constants and
    /// [`dequantize_q4k_block`] for the inverse kernel.
    #[serde(alias = "Q4K", alias = "Q4_K", alias = "Q4_K_M", alias = "q4_k", alias = "q4_k_m", alias = "q4km")]
    Q4K,
    /// **GGUF-style Q4_0 block quantisation.** Each 32-weight block
    /// occupies 18 bytes: an `f16` block scale `d` followed by 16
    /// bytes of symmetrically quantised 4-bit weights (low nibble
    /// first; each nibble decodes as `q - 8` ∈ `[-8, +7]`). The
    /// dequantised value is `d * (q - 8)`. Effective bytes-per-weight
    /// = 18 / 32 = 0.5625 (same density as Q4_K), but with no min
    /// channel — Q4_0 is the simplest and most widely-used 4-bit
    /// format and is the format requested by the predictive-controller
    /// design spec. See [`Q4_0_BLOCK_BYTES`] / [`Q4_0_BLOCK_ELEMS`] for
    /// the layout constants and [`dequantize_q4_0_block`] for the
    /// inverse kernel.
    #[serde(alias = "Q4_0", alias = "q40", alias = "q4")]
    Q4_0,
}

impl WeightDtype {
    /// Number of on-disk bytes per weight for this dtype. **Excludes**
    /// any per-tensor scale header — see [`Self::header_bytes`].
    ///
    /// For block-quantised dtypes (Q4K) this is **fractional** in
    /// reality (144 bytes per 256-weight block ≈ 0.5625 byte/weight);
    /// to keep the integer return type stable we round up to the
    /// nearest whole byte and let [`expert_weight_bytes_for`] do the
    /// exact accounting on the block boundary. Callers that need exact
    /// sizing should always go through [`expert_weight_bytes_for`].
    #[inline]
    pub const fn bytes_per_weight(self) -> usize {
        match self {
            WeightDtype::F32 => 4,
            WeightDtype::F16 => 2,
            WeightDtype::Int8 => 1,
            // 144 / 256 rounded up = 1; not used for sizing — see docs.
            WeightDtype::Q4K => 1,
            // 18 / 32 rounded up = 1; not used for sizing — see docs.
            WeightDtype::Q4_0 => 1,
        }
    }

    /// Number of constant-size header bytes prepended to every expert
    /// blob for this dtype, **before** the weight stream begins.
    /// `Int8` uses 12 bytes (`[f32; 3]` per-tensor scales); the
    /// floating-point dtypes use no header. Q4K and Q4_0 are both
    /// self-describing (the per-block scales already live inside each
    /// block) so they also use no global header.
    #[inline]
    pub const fn header_bytes(self) -> usize {
        match self {
            WeightDtype::F32
            | WeightDtype::F16
            | WeightDtype::Q4K
            | WeightDtype::Q4_0 => 0,
            WeightDtype::Int8 => INT8_HEADER_BYTES,
        }
    }

    /// Parse from CLI / metadata.json string. Returns `None` for unknown.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Some(WeightDtype::F32),
            "f16" | "fp16" | "half" => Some(WeightDtype::F16),
            "i8" | "int8" | "q8" => Some(WeightDtype::Int8),
            "q4k" | "q4_k" | "q4_k_m" | "q4km" => Some(WeightDtype::Q4K),
            "q4_0" | "q40" | "q4" => Some(WeightDtype::Q4_0),
            _ => None,
        }
    }

    /// Stable string form used in metadata.json / CLI flags.
    pub const fn as_str(self) -> &'static str {
        match self {
            WeightDtype::F32 => "f32",
            WeightDtype::F16 => "f16",
            WeightDtype::Int8 => "int8",
            WeightDtype::Q4K => "q4k",
            WeightDtype::Q4_0 => "q4_0",
        }
    }
}

/// Size of the per-expert INT8 scale header: three `f32` per-tensor
/// scales (`gate`, `up`, `down`).
pub const INT8_HEADER_BYTES: usize = 3 * 4;

/// Per-tensor symmetric-quantization scales for one expert's INT8
/// weights. Stored as the first 12 bytes of an `Int8` expert blob.
#[derive(Debug, Clone, Copy)]
pub struct Int8ExpertMeta {
    pub gate_scale: f32,
    pub up_scale: f32,
    pub down_scale: f32,
}

impl Int8ExpertMeta {
    /// Read the 12-byte header. Returns `None` if the buffer is too short.
    pub fn read_from(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < INT8_HEADER_BYTES {
            return None;
        }
        let g = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let u = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let d = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        Some(Self { gate_scale: g, up_scale: u, down_scale: d })
    }

    /// Serialise the header to its on-disk byte layout.
    pub fn to_bytes(self) -> [u8; INT8_HEADER_BYTES] {
        let mut out = [0u8; INT8_HEADER_BYTES];
        out[0..4].copy_from_slice(&self.gate_scale.to_le_bytes());
        out[4..8].copy_from_slice(&self.up_scale.to_le_bytes());
        out[8..12].copy_from_slice(&self.down_scale.to_le_bytes());
        out
    }
}

impl Default for WeightDtype {
    fn default() -> Self {
        WeightDtype::F32
    }
}

/// Dequantise a little-endian `f16` byte buffer into an owned `Vec<f32>`.
///
/// Each pair of bytes is interpreted as one little-endian `half::f16`
/// and converted to `f32`. `dst` is `clear()`ed first so the caller can
/// reuse a previously-allocated buffer.
pub fn dequantize_f16_to_f32(src: &[u8], dst: &mut Vec<f32>) {
    assert!(
        src.len() % 2 == 0,
        "f16 byte buffer length must be a multiple of 2, got {}",
        src.len()
    );
    let n = src.len() / 2;
    dst.clear();
    dst.reserve(n);
    // Manual LE conversion: avoids requiring a top-level `bytemuck`
    // dependency and works regardless of pointer alignment of `src`.
    for chunk in src.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        dst.push(half::f16::from_bits(bits).to_f32());
    }
}

// ---------------------------------------------------------------------
// Q4_K (GGUF Q4_K_M) block quantisation.
// ---------------------------------------------------------------------

/// Number of weights per Q4_K super-block.
pub const Q4K_BLOCK_ELEMS: usize = 256;
/// Bytes per Q4_K super-block on disk.
///
/// Layout:
/// ```text
///   d      : f16 (2 bytes)   — super-block scale
///   dmin   : f16 (2 bytes)   — super-block min
///   scales : 12 bytes        — 8x 6-bit sub-scales + 8x 6-bit sub-mins,
///                              packed exactly as in `ggml-quants.c`.
///   qs     : 128 bytes       — 256x 4-bit weights (low nibble first).
/// ```
/// Total: `2 + 2 + 12 + 128 = 144` bytes.
pub const Q4K_BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
/// Number of sub-blocks per Q4_K super-block (8 sub-blocks of 32 weights).
pub const Q4K_SUBBLOCKS: usize = 8;
/// Number of weights per Q4_K sub-block.
pub const Q4K_SUBBLOCK_ELEMS: usize = Q4K_BLOCK_ELEMS / Q4K_SUBBLOCKS; // 32

/// Decode the 12-byte `scales[12]` array from a Q4_K block into 8
/// sub-block (scale, min) 6-bit values.
///
/// This matches the bit-packing used by `llama.cpp`'s `ggml-quants.c`
/// `get_scale_min_k4`: for the first four sub-blocks, scale and min
/// are the low 6 bits of `scales[i]` and `scales[i+4]`; for the last
/// four, the high 2 bits live in the upper bits of `scales[0..4]` /
/// `scales[4..8]` and the low 4 bits live in the upper nibbles of
/// `scales[8..12]`. The output `(scale6, min6)` are integers in
/// `0..64`.
fn q4k_unpack_scales(scales: &[u8; 12]) -> [(u8, u8); Q4K_SUBBLOCKS] {
    // Reference implementation from `ggml-quants.c`:
    //
    //   if (j < 4) {
    //       *d = q[j] & 63;
    //       *m = q[j + 4] & 63;
    //   } else {
    //       *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
    //       *m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4);
    //   }
    let mut out = [(0u8, 0u8); Q4K_SUBBLOCKS];
    for j in 0..Q4K_SUBBLOCKS {
        let (d, m) = if j < 4 {
            let d = scales[j] & 0x3F;
            let m = scales[j + 4] & 0x3F;
            (d, m)
        } else {
            let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (d & 0x3F, m & 0x3F)
        };
        out[j] = (d, m);
    }
    out
}

/// Encode 8 `(scale6, min6)` pairs into a 12-byte `scales[12]` buffer
/// using the inverse of [`q4k_unpack_scales`]. Used by tests that need
/// to construct synthetic Q4_K blocks.
fn q4k_pack_scales(pairs: &[(u8, u8); Q4K_SUBBLOCKS]) -> [u8; 12] {
    // Inverse of the unpack above. Each input value must fit in 6 bits.
    let mut s = [0u8; 12];
    // First fill the low-6-bit slots for j in 0..4.
    for j in 0..4 {
        s[j] = pairs[j].0 & 0x3F; // scale[0..4]
        s[j + 4] = pairs[j].1 & 0x3F; // min[0..4]
    }
    // Now layer in the j in 4..8 entries:
    //   scale6_j = (s[j+4] & 0xF)  | ((s[j-4] >> 6) << 4)
    //   min6_j   = (s[j+4] >> 4)   | ((s[j  ] >> 6) << 4)
    // We need to (a) place the low 4 bits of scale_j into s[j+4][0..4],
    // (b) place the low 4 bits of min_j into s[j+4][4..8], (c) place
    // the high 2 bits of scale_j into s[j-4][6..8], (d) place the
    // high 2 bits of min_j into s[j][6..8].
    for j in 4..Q4K_SUBBLOCKS {
        let (scale_j, min_j) = pairs[j];
        let scale_j = scale_j & 0x3F;
        let min_j = min_j & 0x3F;
        s[j + 4] = (scale_j & 0x0F) | ((min_j & 0x0F) << 4);
        // Top 2 bits of scale_j go into the upper 2 bits of s[j-4].
        s[j - 4] = (s[j - 4] & 0x3F) | (((scale_j >> 4) & 0x03) << 6);
        // Top 2 bits of min_j go into the upper 2 bits of s[j].
        s[j] = (s[j] & 0x3F) | (((min_j >> 4) & 0x03) << 6);
    }
    s
}

/// Dequantise one Q4_K super-block into `dst` (must hold exactly
/// [`Q4K_BLOCK_ELEMS`] floats).
///
/// The inverse of the GGUF Q4_K_M quantiser:
/// ```text
///   for sub_j in 0..8:
///       scale_j = sub_scales[j] (6-bit)
///       min_j   = sub_mins[j]   (6-bit)
///       for i in 0..32:
///           q4 = qs[(j*32 + i) >> 1] >> ((i & 1) * 4) & 0xF      (low/high nibble)
///           dst[j*32 + i] = d * scale_j * q4 - dmin * min_j
/// ```
/// `d` and `dmin` are read as little-endian `f16` from the first four
/// bytes of `src`.
pub fn dequantize_q4k_block(src: &[u8], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        Q4K_BLOCK_BYTES,
        "Q4K block must be {} bytes",
        Q4K_BLOCK_BYTES
    );
    assert_eq!(
        dst.len(),
        Q4K_BLOCK_ELEMS,
        "Q4K dst must hold {} floats",
        Q4K_BLOCK_ELEMS
    );

    let d = half::f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([src[2], src[3]])).to_f32();
    let scales: [u8; 12] = src[4..16].try_into().expect("12-byte slice");
    let pairs = q4k_unpack_scales(&scales);
    let qs = &src[16..16 + 128];
    debug_assert_eq!(qs.len(), 128);

    for j in 0..Q4K_SUBBLOCKS {
        let (scale6, min6) = pairs[j];
        let s = d * scale6 as f32;
        let m = dmin * min6 as f32;
        // 32 weights per sub-block, packed as 16 bytes (low nibble first).
        let qs_off = j * (Q4K_SUBBLOCK_ELEMS / 2);
        for i in 0..Q4K_SUBBLOCK_ELEMS {
            let byte = qs[qs_off + (i >> 1)];
            let q4 = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            dst[j * Q4K_SUBBLOCK_ELEMS + i] = s * (q4 as f32) - m;
        }
    }
}

/// Dequantise a contiguous Q4_K stream of `n_weights` weights into
/// `dst`. The input `src` must contain exactly `ceil(n_weights / 256)`
/// blocks (`* 144` bytes); a tail block whose effective weight count
/// is < 256 is fully decoded and the unused floats at the end of the
/// last block are dropped.
///
/// Used by [`OwnedExpertWeights::from_bytes_q4k`] to decode the three
/// expert weight matrices in a single sweep before splitting them by
/// row.
pub fn dequantize_q4k_to_f32(src: &[u8], n_weights: usize, dst: &mut Vec<f32>) {
    let blocks = n_weights.div_ceil(Q4K_BLOCK_ELEMS);
    assert!(
        src.len() >= blocks * Q4K_BLOCK_BYTES,
        "Q4K source has {} bytes, need {} for {n_weights} weights",
        src.len(),
        blocks * Q4K_BLOCK_BYTES
    );
    dst.clear();
    dst.resize(blocks * Q4K_BLOCK_ELEMS, 0.0);
    for b in 0..blocks {
        let s = &src[b * Q4K_BLOCK_BYTES..(b + 1) * Q4K_BLOCK_BYTES];
        let d = &mut dst[b * Q4K_BLOCK_ELEMS..(b + 1) * Q4K_BLOCK_ELEMS];
        dequantize_q4k_block(s, d);
    }
    // Truncate any padding at the tail (for n_weights not divisible by 256).
    dst.truncate(n_weights);
}

// ---------------------------------------------------------------------
// Q4_0 (GGUF Q4_0) block quantisation.
// ---------------------------------------------------------------------

/// Number of weights per Q4_0 block. Matches `llama.cpp`'s
/// `QK4_0` / `ggml_tensor.block_size`. The design spec asks for
/// "every 32 weights share an f16 scale", which is exactly Q4_0.
pub const Q4_0_BLOCK_ELEMS: usize = 32;

/// Bytes per Q4_0 block on disk.
///
/// Layout:
/// ```text
///   d   : f16 (2 bytes)         — block scale
///   qs  : 16 bytes              — 32x 4-bit symmetric weights
///                                 (low nibble first; q ∈ 0..=15,
///                                 dequantised as `d * (q - 8)`).
/// ```
/// Total: `2 + 16 = 18` bytes. Effective bytes-per-weight = 18/32
/// = 0.5625, the same density as Q4_K but with a much simpler block
/// layout (one f16 scale, no min, no sub-blocks).
pub const Q4_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK_ELEMS / 2;

/// Dequantise one Q4_0 block into `dst` (must hold exactly
/// [`Q4_0_BLOCK_ELEMS`] floats).
///
/// Inverse of the GGUF Q4_0 quantiser:
/// ```text
///   d  = f16 super-scale (first 2 bytes, little-endian)
///   for i in 0..32:
///       q4 = qs[i >> 1] >> ((i & 1) * 4) & 0xF      (low/high nibble)
///       dst[i] = d * (q4 as i32 - 8) as f32
/// ```
pub fn dequantize_q4_0_block(src: &[u8], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        Q4_0_BLOCK_BYTES,
        "Q4_0 block must be {} bytes",
        Q4_0_BLOCK_BYTES
    );
    assert_eq!(
        dst.len(),
        Q4_0_BLOCK_ELEMS,
        "Q4_0 dst must hold {} floats",
        Q4_0_BLOCK_ELEMS
    );
    let d = half::f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32();
    let qs = &src[2..2 + Q4_0_BLOCK_ELEMS / 2];
    for i in 0..Q4_0_BLOCK_ELEMS {
        let byte = qs[i >> 1];
        let q4 = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        // Symmetric range: q4 ∈ 0..15 dequantises to q4-8 ∈ -8..+7.
        dst[i] = d * (q4 as i32 - 8) as f32;
    }
}

/// Quantise one block of up to [`Q4_0_BLOCK_ELEMS`] f32 weights into
/// the 18-byte Q4_0 layout. The block scale is `max_abs / 7.0` (so
/// the largest-magnitude weight maps to `q=15` ≡ `+7` after the bias
/// shift); zero-valued blocks store `d=0` and `q4=8` (decoding as 0).
/// Inputs shorter than [`Q4_0_BLOCK_ELEMS`] are zero-padded.
///
/// This matches the simple "max-abs symmetric" scheme used by
/// `llama.cpp`'s reference Q4_0 quantiser. Production GGUF quantisers
/// use a slightly more sophisticated rmin/rmax search, but the bit
/// layout and dequantisation are identical so any consumer of Q4_0
/// can read the output.
pub fn quantize_q4_0_block(src: &[f32], dst: &mut [u8]) {
    assert_eq!(
        dst.len(),
        Q4_0_BLOCK_BYTES,
        "Q4_0 dst must be {} bytes",
        Q4_0_BLOCK_BYTES
    );
    assert!(
        src.len() <= Q4_0_BLOCK_ELEMS,
        "Q4_0 src must hold at most {} floats, got {}",
        Q4_0_BLOCK_ELEMS,
        src.len()
    );
    let mut max_abs = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let d = max_abs / 7.0;
    // Write the f16 scale.
    let d_bits = half::f16::from_f32(d).to_bits();
    dst[0..2].copy_from_slice(&d_bits.to_le_bytes());
    // Initialise nibbles to the zero-encoding (`q4 = 8`, both nibbles).
    for byte in &mut dst[2..2 + Q4_0_BLOCK_ELEMS / 2] {
        *byte = 0x88;
    }
    if d == 0.0 {
        return;
    }
    let inv_d = 1.0 / d;
    let qs = &mut dst[2..2 + Q4_0_BLOCK_ELEMS / 2];
    for i in 0..Q4_0_BLOCK_ELEMS {
        let v = if i < src.len() { src[i] } else { 0.0 };
        // Round to nearest, shift +8 to map [-8,+7] → [0,15], clamp.
        let q = (v * inv_d).round() as i32 + 8;
        let q4 = q.clamp(0, 15) as u8;
        let byte = &mut qs[i >> 1];
        if i & 1 == 0 {
            *byte = (*byte & 0xF0) | (q4 & 0x0F);
        } else {
            *byte = (*byte & 0x0F) | ((q4 & 0x0F) << 4);
        }
    }
}

/// Dequantise a contiguous Q4_0 stream of `n_weights` weights into
/// `dst`. The input `src` must contain exactly `ceil(n_weights / 32)`
/// blocks (`* 18` bytes); a tail block whose effective weight count
/// is < 32 is fully decoded and the unused floats at the end are
/// truncated.
///
/// Mirrors [`dequantize_q4k_to_f32`] in shape so the call sites in
/// [`OwnedExpertWeights::from_bytes_q4_0`] can be a near-direct
/// translation of the Q4_K decoder.
pub fn dequantize_q4_0_to_f32(src: &[u8], n_weights: usize, dst: &mut Vec<f32>) {
    let blocks = n_weights.div_ceil(Q4_0_BLOCK_ELEMS);
    assert!(
        src.len() >= blocks * Q4_0_BLOCK_BYTES,
        "Q4_0 source has {} bytes, need {} for {n_weights} weights",
        src.len(),
        blocks * Q4_0_BLOCK_BYTES
    );
    dst.clear();
    dst.resize(blocks * Q4_0_BLOCK_ELEMS, 0.0);
    for b in 0..blocks {
        let s = &src[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
        let d = &mut dst[b * Q4_0_BLOCK_ELEMS..(b + 1) * Q4_0_BLOCK_ELEMS];
        dequantize_q4_0_block(s, d);
    }
    dst.truncate(n_weights);
}
pub type HiddenState = Vec<f32>;

/// Result of running one expert's FFN on a hidden state.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub expert_id: u32,
    /// 64-bit digest over the output activation. Cheap to log; lets us
    /// verify end-to-end that the bytes streamed from disk really did
    /// drive a deterministic computation.
    pub digest: u64,
    /// L2 norm of the output activation. A sanity signal that the matmul
    /// produced a non-degenerate result (NaN/Inf or all-zero would stand
    /// out immediately in the per-token logs).
    pub out_norm: f32,
}

/// Three-matrix view over a flat `&[f32]` blob with the layout described
/// at the top of this module.
pub struct ExpertWeights<'a> {
    pub d_model: usize,
    pub d_ff: usize,
    /// `gate_proj`, row-major `[d_ff x d_model]` (rows of `d_model` floats).
    pub gate: &'a [f32],
    /// `up_proj`, row-major `[d_ff x d_model]`.
    pub up: &'a [f32],
    /// `down_proj`, row-major `[d_model x d_ff]`.
    pub down: &'a [f32],
}

/// Number of `f32` weights an expert with these dimensions occupies.
///
/// Uses `saturating_mul` so absurdly large CLI-provided shapes don't
/// silently wrap in release mode — on overflow this returns
/// `usize::MAX`, which makes every downstream size check (the buffer
/// length comparison in [`ExpertWeights::from_bytes`], the
/// `expert_size` validation in `cmd_gen_data` / `generate_synthetic_experts`,
/// and the engine's startup check) reliably fail.
#[inline]
pub const fn expert_weight_count(d_model: usize, d_ff: usize) -> usize {
    // gate (d_ff * d_model) + up (d_ff * d_model) + down (d_model * d_ff)
    let one = d_model.saturating_mul(d_ff);
    one.saturating_mul(3)
}

/// Number of bytes an expert with these dimensions occupies on disk
/// (one `f32` is 4 bytes). Saturates to `usize::MAX` on overflow; see
/// [`expert_weight_count`].
#[inline]
pub const fn expert_weight_bytes(d_model: usize, d_ff: usize) -> usize {
    expert_weight_count(d_model, d_ff).saturating_mul(4)
}

/// Number of bytes an expert with these dimensions occupies on disk
/// when stored as little-endian `f16` (2 bytes per weight). Saturates to
/// `usize::MAX` on overflow.
#[inline]
pub const fn expert_weight_bytes_f16(d_model: usize, d_ff: usize) -> usize {
    expert_weight_count(d_model, d_ff).saturating_mul(2)
}

/// Number of bytes an expert with these dimensions occupies on disk
/// for the given dtype, **including** any per-expert header (e.g. the
/// 12-byte INT8 scale header).
///
/// For [`WeightDtype::Q4K`] the weight count is rounded **up** to a
/// multiple of [`Q4K_BLOCK_ELEMS`] (256) — Q4K only quantises whole
/// super-blocks, so a tail of < 256 weights still pays for one full
/// 144-byte block on disk. This matches the behaviour of every
/// production GGUF quantiser.
#[inline]
pub const fn expert_weight_bytes_for(d_model: usize, d_ff: usize, dtype: WeightDtype) -> usize {
    match dtype {
        WeightDtype::Q4K => {
            let weights = expert_weight_count(d_model, d_ff);
            let blocks = weights.div_ceil(Q4K_BLOCK_ELEMS);
            blocks.saturating_mul(Q4K_BLOCK_BYTES)
        }
        WeightDtype::Q4_0 => {
            // Per-tensor block alignment: each of the three matrices
            // (gate / up / down) is dequantised independently so a
            // tail of < 32 weights in any one matrix still consumes
            // one full 18-byte block. This matches
            // [`OwnedExpertWeights::from_bytes_q4_0`]'s expectation.
            let one = d_model.saturating_mul(d_ff);
            let one_blocks = one.div_ceil(Q4_0_BLOCK_ELEMS);
            let total_blocks = one_blocks.saturating_mul(3);
            total_blocks.saturating_mul(Q4_0_BLOCK_BYTES)
        }
        _ => {
            let payload = expert_weight_count(d_model, d_ff)
                .saturating_mul(dtype.bytes_per_weight());
            payload.saturating_add(dtype.header_bytes())
        }
    }
}

/// Errors produced when reinterpreting a raw byte buffer as expert
/// weights. These are the conditions that previously aborted the run
/// via `assert!`; the run path now logs them and skips the offending
/// expert instead of panicking on corrupt on-disk data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertWeightsError {
    /// The buffer is shorter than the SwiGLU weight blob requires.
    BufferTooSmall {
        have: usize,
        need: usize,
        d_model: usize,
        d_ff: usize,
    },
    /// The buffer's start address is not aligned for `f32` access.
    /// `BufferPool` always allocates page-aligned buffers, so this is
    /// a contract violation rather than something a corrupt file can
    /// trigger — but we surface it as an error rather than panicking.
    Misaligned { addr: usize, required: usize },
}

impl fmt::Display for ExpertWeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpertWeightsError::BufferTooSmall { have, need, d_model, d_ff } => write!(
                f,
                "expert buffer too small: have {have} bytes, need {need} for \
                 d_model={d_model}, d_ff={d_ff}"
            ),
            ExpertWeightsError::Misaligned { addr, required } => write!(
                f,
                "expert buffer is not f32-aligned: addr=0x{addr:x}, required={required}"
            ),
        }
    }
}

impl std::error::Error for ExpertWeightsError {}

impl<'a> ExpertWeights<'a> {
    /// Reinterpret a page-aligned byte buffer as the three weight matrices.
    ///
    /// Returns [`ExpertWeightsError`] if `bytes` is shorter than
    /// `expert_weight_bytes(d_model, d_ff)` or does not start at an
    /// address aligned to `align_of::<f32>()` (4 bytes). `BufferPool`
    /// allocates with `≥ 4096`-byte alignment, so the alignment branch
    /// is defensive — but the size branch can fire on a truncated /
    /// corrupt on-disk file, and surfacing it as a `Result` lets the
    /// engine log and skip the expert instead of aborting the whole run.
    ///
    /// Any trailing bytes (e.g. padding so the file size is a multiple of
    /// `block_align` for `O_DIRECT`) are ignored.
    pub fn from_bytes(
        bytes: &'a [u8],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let need_floats = expert_weight_count(d_model, d_ff);
        let need_bytes = need_floats.saturating_mul(4);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }
        // Buffers from `BufferPool` are page-aligned; `f32` only needs 4.
        let required_align = std::mem::align_of::<f32>();
        let addr = bytes.as_ptr() as usize;
        if addr % required_align != 0 {
            return Err(ExpertWeightsError::Misaligned {
                addr,
                required: required_align,
            });
        }

        // SAFETY:
        // * `bytes.as_ptr()` is non-null and verified above to be aligned
        //   to `align_of::<f32>()`.
        // * `need_floats * 4 <= bytes.len()`, so the resulting slice stays
        //   inside the original allocation.
        // * `f32` has no validity invariants: every 4-byte sequence is a
        //   valid `f32` (NaN / subnormal / Inf are all well-defined values
        //   for the type system; whether the *model* tolerates them is
        //   `gen-data`'s responsibility, and it writes finite weights).
        // * The on-disk layout is little-endian `f32`; the
        //   `compile_error!` at the top of this module ensures the host
        //   is also little-endian, so a byte-for-byte reinterpretation
        //   is correct.
        // * The lifetime of the resulting `&[f32]` is tied to `bytes`, so
        //   the borrow checker prevents mutation while the view exists.
        let floats: &'a [f32] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), need_floats)
        };

        let gate_len = d_ff * d_model;
        let up_len = d_ff * d_model;
        let down_len = d_model * d_ff;
        let (gate, rest) = floats.split_at(gate_len);
        let (up, rest) = rest.split_at(up_len);
        let (down, _trailing) = rest.split_at(down_len);

        Ok(Self { d_model, d_ff, gate, up, down })
    }

    /// Build the three-matrix view from a fully-materialised `&[f32]`
    /// slice (rather than from raw on-disk bytes). This is the shared
    /// helper used by both [`Self::from_bytes`] (zero-copy reinterpret)
    /// and [`OwnedExpertWeights::from_bytes_f16`] (dequantised owned
    /// `Vec<f32>`). Returns `BufferTooSmall` if `floats.len()` is short.
    pub fn from_floats(
        floats: &'a [f32],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let need_floats = expert_weight_count(d_model, d_ff);
        if floats.len() < need_floats {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: floats.len().saturating_mul(4),
                need: need_floats.saturating_mul(4),
                d_model,
                d_ff,
            });
        }
        let gate_len = d_ff * d_model;
        let up_len = d_ff * d_model;
        let down_len = d_model * d_ff;
        let (gate, rest) = floats[..need_floats].split_at(gate_len);
        let (up, rest) = rest.split_at(up_len);
        let (down, _trailing) = rest.split_at(down_len);
        Ok(Self { d_model, d_ff, gate, up, down })
    }


    /// Run the SwiGLU forward pass.
    ///
    /// Allocates one `Vec<f32>` for the gated intermediate (`d_ff`) and
    /// returns a `Vec<f32>` of length `d_model` for the FFN output.
    ///
    /// **Parallelism.** When the `simd` cargo feature is enabled and the
    /// matrices are large enough (`d_ff * d_model >= 8 KiB`), the gate /
    /// up projection and the down projection are split row-wise across
    /// threads using `std::thread::scope` (the same pattern as
    /// [`crate::transformer::matmul_row_major`] — no extra crate
    /// dependency). Each row of the output is independent, so no
    /// synchronisation is required. The scalar fallback is the default
    /// build.
    pub fn forward(&self, x: &[f32]) -> HiddenState {
        debug_assert_eq!(
            x.len(),
            self.d_model,
            "hidden state length must equal d_model"
        );

        // 1) Two parallel projections into d_ff: gate = W_g x, up = W_u x.
        //    Fuse them in the same loop so each row of (gate, up) reads
        //    `x` once — better cache behaviour than two separate matmuls.
        let mut gated = vec![0.0f32; self.d_ff];
        gate_up_swiglu(self.gate, self.up, x, &mut gated, self.d_model);

        // 2) Down projection: y = W_d · gated  -> length d_model.
        let mut y = vec![0.0f32; self.d_model];
        down_proj(self.down, &gated, &mut y, self.d_ff);
        y
    }
}

/// Fused gate / up SwiGLU projection: `gated[i] = silu(gate[i,:]·x) * (up[i,:]·x)`.
///
/// Row-parallel under the `simd` cargo feature; scalar otherwise. Each
/// row of `gated` is computed independently so the parallelisation is
/// embarrassingly safe.
#[inline]
fn gate_up_swiglu(gate: &[f32], up: &[f32], x: &[f32], gated: &mut [f32], d_model: usize) {
    debug_assert_eq!(gate.len(), gated.len() * d_model);
    debug_assert_eq!(up.len(), gated.len() * d_model);
    debug_assert_eq!(x.len(), d_model);

    #[cfg(feature = "blas")]
    {
        // BLAS-equivalent SGEMV via `matrixmultiply::sgemm`: split the
        // fused row pass into two tuned `(d_ff × d_model) × (d_model
        // × 1)` matrix-vector products followed by an elementwise
        // `silu(g)·u`. The crate's hand-tuned AVX2/NEON microkernel
        // (same one ndarray uses for `dot`) gives ~5–10× over the
        // scalar loop on dense f32 weights, which is the largest
        // win available in this codebase aside from quantising.
        //
        // We avoid two per-call `Vec<f32>` allocations on this
        // per-token / per-expert hot path:
        //   * `g` is written *directly* into the caller-supplied
        //     `gated` slot, then later overwritten in place with
        //     `silu(gated[i]) * u[i]`.
        //   * `u` is written into a thread-local scratch buffer that
        //     is grown to `d_ff` on first use and reused on every
        //     subsequent call.
        let d_ff = gated.len();
        thread_local! {
            static SGEMM_SCRATCH: std::cell::RefCell<Vec<f32>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        SGEMM_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < d_ff {
                scratch.resize(d_ff, 0.0);
            }
            let u_vec = &mut scratch[..d_ff];
            // Safety: SGEMM contract — row-major (m × k) · (k × n) → (m × n).
            // `gated` and `u_vec` are disjoint (one is the caller's
            // output buffer, the other is a thread-local scratch),
            // and neither aliases `gate`, `up`, or `x`.
            unsafe {
                matrixmultiply::sgemm(
                    d_ff, d_model, 1, 1.0,
                    gate.as_ptr(), d_model as isize, 1,
                    x.as_ptr(), 1, 1,
                    0.0, gated.as_mut_ptr(), 1, 1,
                );
                matrixmultiply::sgemm(
                    d_ff, d_model, 1, 1.0,
                    up.as_ptr(), d_model as isize, 1,
                    x.as_ptr(), 1, 1,
                    0.0, u_vec.as_mut_ptr(), 1, 1,
                );
            }
            for i in 0..d_ff {
                gated[i] = silu(gated[i]) * u_vec[i];
            }
        });
        return;
    }

    #[cfg(feature = "simd")]
    {
        let d_ff = gated.len();
        if d_ff * d_model >= 8 * 1024 {
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(d_ff.max(1));
            if nthreads > 1 {
                let chunk = d_ff.div_ceil(nthreads);
                std::thread::scope(|s| {
                    let mut handles = Vec::with_capacity(nthreads);
                    for (chunk_idx, out_chunk) in gated.chunks_mut(chunk).enumerate() {
                        let row_start = chunk_idx * chunk;
                        let g_slice = &gate[row_start * d_model
                            ..(row_start + out_chunk.len()) * d_model];
                        let u_slice = &up[row_start * d_model
                            ..(row_start + out_chunk.len()) * d_model];
                        let x_ref = x;
                        handles.push(s.spawn(move || {
                            for (i, slot) in out_chunk.iter_mut().enumerate() {
                                let g_row = &g_slice[i * d_model..(i + 1) * d_model];
                                let u_row = &u_slice[i * d_model..(i + 1) * d_model];
                                let mut g = 0.0f32;
                                let mut u = 0.0f32;
                                for j in 0..d_model {
                                    g += g_row[j] * x_ref[j];
                                    u += u_row[j] * x_ref[j];
                                }
                                *slot = silu(g) * u;
                            }
                        }));
                    }
                    for h in handles {
                        // Propagate worker panics — a silent failure here
                        // would leave `gated` partially written.
                        h.join().expect("expert gate/up matmul worker panicked");
                    }
                });
                return;
            }
        }
    }
    let d_ff = gated.len();
    for i in 0..d_ff {
        let row = i * d_model;
        let g_row = &gate[row..row + d_model];
        let u_row = &up[row..row + d_model];
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for j in 0..d_model {
            g += g_row[j] * x[j];
            u += u_row[j] * x[j];
        }
        gated[i] = silu(g) * u;
    }
}

/// Down projection `y = W_d · gated`. Row-parallel under `simd`.
#[inline]
fn down_proj(down: &[f32], gated: &[f32], y: &mut [f32], d_ff: usize) {
    debug_assert_eq!(down.len(), y.len() * d_ff);
    debug_assert_eq!(gated.len(), d_ff);

    #[cfg(feature = "blas")]
    {
        // Pure SGEMV: `y[d_model] = down[d_model × d_ff] · gated[d_ff]`.
        // Safety: SGEMM contract — row-major (m × k) · (k × n) → (m × n).
        let d_model = y.len();
        unsafe {
            matrixmultiply::sgemm(
                d_model, d_ff, 1, 1.0,
                down.as_ptr(), d_ff as isize, 1,
                gated.as_ptr(), 1, 1,
                0.0, y.as_mut_ptr(), 1, 1,
            );
        }
        return;
    }

    #[cfg(feature = "simd")]
    {
        let d_model = y.len();
        if d_model * d_ff >= 8 * 1024 {
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(d_model.max(1));
            if nthreads > 1 {
                let chunk = d_model.div_ceil(nthreads);
                std::thread::scope(|s| {
                    let mut handles = Vec::with_capacity(nthreads);
                    for (chunk_idx, out_chunk) in y.chunks_mut(chunk).enumerate() {
                        let row_start = chunk_idx * chunk;
                        let d_slice = &down[row_start * d_ff
                            ..(row_start + out_chunk.len()) * d_ff];
                        let g_ref = gated;
                        handles.push(s.spawn(move || {
                            for (i, slot) in out_chunk.iter_mut().enumerate() {
                                let d_row = &d_slice[i * d_ff..(i + 1) * d_ff];
                                let mut acc = 0.0f32;
                                for j in 0..d_ff {
                                    acc += d_row[j] * g_ref[j];
                                }
                                *slot = acc;
                            }
                        }));
                    }
                    for h in handles {
                        // Propagate worker panics — a silent failure here
                        // would leave `y` partially written.
                        h.join().expect("expert down-proj matmul worker panicked");
                    }
                });
                return;
            }
        }
    }
    let d_model = y.len();
    for i in 0..d_model {
        let row = i * d_ff;
        let d_row = &down[row..row + d_ff];
        let mut acc = 0.0f32;
        for j in 0..d_ff {
            acc += d_row[j] * gated[j];
        }
        y[i] = acc;
    }
}

/// SiLU / swish activation: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Owned variant of [`ExpertWeights`] backed by a `Vec<f32>` rather than
/// borrowed bytes. Returned by [`OwnedExpertWeights::from_bytes_f16`] and
/// [`OwnedExpertWeights::from_bytes_partial`], where dequantisation /
/// column repacking forces materialising fresh f32 storage.
pub struct OwnedExpertWeights {
    pub d_model: usize,
    pub d_ff: usize,
    /// `gate_proj` row-major. For partial-load this is `[d_ff x M]`,
    /// where `M = col_indices.len()`; otherwise `[d_ff x d_model]`.
    pub gate: Vec<f32>,
    /// `up_proj` row-major (same shape conventions as `gate`).
    pub up: Vec<f32>,
    /// `down_proj` row-major. For full / f16 weights: `[d_model x d_ff]`.
    /// For partial-load: still `[d_model x d_ff]` (down_proj rows are not
    /// reduced) but only computed against a packed gated vector of
    /// length `d_ff`.
    pub down: Vec<f32>,
    /// For partial-load: the column indices of `gate`/`up` in the
    /// original `[d_ff x d_model]` matrix that were actually loaded.
    /// `None` means full load (all `d_model` columns).
    pub col_indices: Option<Vec<usize>>,
}

impl OwnedExpertWeights {
    /// Build an owned weight set by dequantising a **Q4_K** byte buffer
    /// (GGUF Q4_K_M layout) into a fresh `Vec<f32>`. The buffer is
    /// expected to be a contiguous stream of [`Q4K_BLOCK_BYTES`]-sized
    /// super-blocks covering the three weight matrices in the same
    /// partitioned order as [`ExpertWeights::from_bytes`]:
    ///
    /// ```text
    ///   gate_proj   (ceil(d_ff*d_model / 256) * 144 bytes)
    ///   up_proj     (ceil(d_ff*d_model / 256) * 144 bytes)
    ///   down_proj   (ceil(d_model*d_ff / 256) * 144 bytes)
    /// ```
    ///
    /// Each tensor is dequantised independently (sub-block scales /
    /// mins do not cross matrix boundaries), so a tail of < 256
    /// weights in any one matrix still consumes one full super-block.
    /// The total on-disk size is given by
    /// [`expert_weight_bytes_for(d_model, d_ff, WeightDtype::Q4K)`](expert_weight_bytes_for).
    pub fn from_bytes_q4k(
        bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let gate_n = d_ff.saturating_mul(d_model);
        let up_n = d_ff.saturating_mul(d_model);
        let down_n = d_model.saturating_mul(d_ff);
        let gate_blocks = gate_n.div_ceil(Q4K_BLOCK_ELEMS);
        let up_blocks = up_n.div_ceil(Q4K_BLOCK_ELEMS);
        let down_blocks = down_n.div_ceil(Q4K_BLOCK_ELEMS);
        let need_bytes = (gate_blocks + up_blocks + down_blocks)
            .saturating_mul(Q4K_BLOCK_BYTES);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }

        let mut off = 0;
        let mut gate_buf: Vec<f32> = Vec::new();
        dequantize_q4k_to_f32(
            &bytes[off..off + gate_blocks * Q4K_BLOCK_BYTES],
            gate_n,
            &mut gate_buf,
        );
        off += gate_blocks * Q4K_BLOCK_BYTES;
        let mut up_buf: Vec<f32> = Vec::new();
        dequantize_q4k_to_f32(
            &bytes[off..off + up_blocks * Q4K_BLOCK_BYTES],
            up_n,
            &mut up_buf,
        );
        off += up_blocks * Q4K_BLOCK_BYTES;
        let mut down_buf: Vec<f32> = Vec::new();
        dequantize_q4k_to_f32(
            &bytes[off..off + down_blocks * Q4K_BLOCK_BYTES],
            down_n,
            &mut down_buf,
        );

        Ok(Self {
            d_model,
            d_ff,
            gate: gate_buf,
            up: up_buf,
            down: down_buf,
            col_indices: None,
        })
    }

    /// Build an owned weight set by dequantising a **Q4_0** byte buffer
    /// (GGUF Q4_0 layout) into a fresh `Vec<f32>`. The buffer is
    /// expected to be a contiguous stream of [`Q4_0_BLOCK_BYTES`]-sized
    /// blocks covering the three weight matrices in the same
    /// partitioned order as [`ExpertWeights::from_bytes`]:
    ///
    /// ```text
    ///   gate_proj   (ceil(d_ff*d_model / 32) * 18 bytes)
    ///   up_proj     (ceil(d_ff*d_model / 32) * 18 bytes)
    ///   down_proj   (ceil(d_model*d_ff / 32) * 18 bytes)
    /// ```
    ///
    /// Each tensor is dequantised independently (block scales do not
    /// cross matrix boundaries), so a tail of < 32 weights in any one
    /// matrix still consumes one full 18-byte block. The total
    /// on-disk size is given by
    /// [`expert_weight_bytes_for(d_model, d_ff, WeightDtype::Q4_0)`](expert_weight_bytes_for).
    ///
    /// This is the inverse of the GGUF Q4_0 quantiser specified in the
    /// "Omniscient Predictive Architecture" design spec — every 32
    /// weights share an `f16` scale, and dequantisation is `d * (q-8)`
    /// for each 4-bit nibble. Decoded buffers feed directly into the
    /// existing scalar SwiGLU forward pass on
    /// [`OwnedExpertWeights::forward`].
    pub fn from_bytes_q4_0(
        bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let gate_n = d_ff.saturating_mul(d_model);
        let up_n = d_ff.saturating_mul(d_model);
        let down_n = d_model.saturating_mul(d_ff);
        let gate_blocks = gate_n.div_ceil(Q4_0_BLOCK_ELEMS);
        let up_blocks = up_n.div_ceil(Q4_0_BLOCK_ELEMS);
        let down_blocks = down_n.div_ceil(Q4_0_BLOCK_ELEMS);
        let need_bytes = (gate_blocks + up_blocks + down_blocks)
            .saturating_mul(Q4_0_BLOCK_BYTES);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }

        let mut off = 0;
        let mut gate_buf: Vec<f32> = Vec::new();
        dequantize_q4_0_to_f32(
            &bytes[off..off + gate_blocks * Q4_0_BLOCK_BYTES],
            gate_n,
            &mut gate_buf,
        );
        off += gate_blocks * Q4_0_BLOCK_BYTES;
        let mut up_buf: Vec<f32> = Vec::new();
        dequantize_q4_0_to_f32(
            &bytes[off..off + up_blocks * Q4_0_BLOCK_BYTES],
            up_n,
            &mut up_buf,
        );
        off += up_blocks * Q4_0_BLOCK_BYTES;
        let mut down_buf: Vec<f32> = Vec::new();
        dequantize_q4_0_to_f32(
            &bytes[off..off + down_blocks * Q4_0_BLOCK_BYTES],
            down_n,
            &mut down_buf,
        );

        Ok(Self {
            d_model,
            d_ff,
            gate: gate_buf,
            up: up_buf,
            down: down_buf,
            col_indices: None,
        })
    }

    /// Build an owned weight set by dequantising a per-tensor symmetric
    /// **INT8** byte buffer into a fresh `Vec<f32>`. The buffer layout
    /// is: 12-byte [`Int8ExpertMeta`] header (`[gate, up, down]: [f32; 3]`
    /// scales), followed by `i8` weights in the same partitioned order
    /// as [`ExpertWeights::from_bytes`]:
    ///
    /// ```text
    ///   header        (12 bytes)
    ///   gate_proj     (d_ff * d_model bytes, i8)
    ///   up_proj       (d_ff * d_model bytes, i8)
    ///   down_proj     (d_model * d_ff bytes, i8)
    /// ```
    ///
    /// Each tensor is dequantised by `f32_value = i8_value * tensor_scale`.
    /// This is the inverse of [`crate::main::cmd_gen_data`]'s INT8
    /// emitter and matches the reference scheme used by every
    /// production INT8 inference kernel (Mixtral-INT8, AWQ, GPTQ).
    pub fn from_bytes_int8(
        bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let need_floats = expert_weight_count(d_model, d_ff);
        let need_bytes = need_floats.saturating_add(INT8_HEADER_BYTES);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }
        let meta = Int8ExpertMeta::read_from(bytes).expect("header byte length pre-checked");
        let payload = &bytes[INT8_HEADER_BYTES..need_bytes];
        let gate_len = d_ff * d_model;
        let up_len = d_ff * d_model;
        let down_len = d_model * d_ff;
        debug_assert_eq!(gate_len + up_len + down_len, payload.len());

        let dequant = |src: &[u8], scale: f32| -> Vec<f32> {
            // Two's-complement reinterpret: `i8` is a single byte cast.
            src.iter().map(|&b| (b as i8) as f32 * scale).collect()
        };
        let gate = dequant(&payload[..gate_len], meta.gate_scale);
        let up = dequant(&payload[gate_len..gate_len + up_len], meta.up_scale);
        let down = dequant(
            &payload[gate_len + up_len..gate_len + up_len + down_len],
            meta.down_scale,
        );
        Ok(Self { d_model, d_ff, gate, up, down, col_indices: None })
    }

    /// Build an owned weight set by dequantising a little-endian `f16`
    /// byte buffer into a fresh `Vec<f32>`. The resulting buffer is
    /// partitioned the same way as [`ExpertWeights::from_bytes`].
    pub fn from_bytes_f16(
        bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, ExpertWeightsError> {
        let need_floats = expert_weight_count(d_model, d_ff);
        let need_bytes = need_floats.saturating_mul(2);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }
        // Only dequantise exactly the bytes we need; trailing padding
        // (added so the file size is a multiple of `block_align`) is
        // ignored, matching `from_bytes`.
        let mut floats: Vec<f32> = Vec::new();
        dequantize_f16_to_f32(&bytes[..need_bytes], &mut floats);

        // Split into the three matrices using the f32 helper.
        // `from_floats` borrows from `floats`; we copy each region into
        // its own owned `Vec` so the resulting struct can outlive the
        // staging buffer without retaining the whole blob.
        let view = ExpertWeights::from_floats(&floats, d_model, d_ff)?;
        Ok(Self {
            d_model,
            d_ff,
            gate: view.gate.to_vec(),
            up: view.up.to_vec(),
            down: view.down.to_vec(),
            col_indices: None,
        })
    }

    /// Build an owned weight set from the **packed-column** byte format
    /// written by `read_expert_columns`: only the `M` columns listed in
    /// `col_indices` are present for `gate_proj` and `up_proj`; the full
    /// `down_proj` is present (no row reduction is done — `forward_partial`
    /// just zeros out the unloaded gated coordinates).
    ///
    /// Layout (after dequantisation if `dtype == F16`):
    /// `gate_packed [d_ff x M]  ||  up_packed [d_ff x M]  ||  down [d_model x d_ff]`.
    #[allow(dead_code)]
    pub fn from_bytes_partial(
        bytes: &[u8],
        col_indices: &[usize],
        d_model: usize,
        d_ff: usize,
        dtype: WeightDtype,
    ) -> Result<Self, ExpertWeightsError> {
        let m = col_indices.len();
        for &c in col_indices {
            assert!(c < d_model, "col index {c} out of range for d_model={d_model}");
        }
        let packed_floats = d_ff
            .saturating_mul(m)
            .saturating_mul(2)
            .saturating_add(d_model.saturating_mul(d_ff));
        let bpw = dtype.bytes_per_weight();
        let need_bytes = packed_floats.saturating_mul(bpw);
        if bytes.len() < need_bytes {
            return Err(ExpertWeightsError::BufferTooSmall {
                have: bytes.len(),
                need: need_bytes,
                d_model,
                d_ff,
            });
        }
        // Materialise a single f32 buffer covering the packed blob.
        let floats: Vec<f32> = match dtype {
            WeightDtype::F32 => {
                let mut v = Vec::with_capacity(packed_floats);
                for chunk in bytes[..packed_floats * 4].chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                v
            }
            WeightDtype::F16 => {
                let mut v = Vec::new();
                dequantize_f16_to_f32(&bytes[..packed_floats * 2], &mut v);
                v
            }
            WeightDtype::Int8 | WeightDtype::Q4K | WeightDtype::Q4_0 => {
                // Partial-load + INT8 / Q4K / Q4_0 isn't supported
                // (the partial-load packing is column-major and the
                // block / per-tensor scales are not per-column, so
                // the dequant rounding would shift). Bail out
                // cleanly so callers can fall back to the full-load
                // path.
                return Err(ExpertWeightsError::BufferTooSmall {
                    have: bytes.len(),
                    need: usize::MAX,
                    d_model,
                    d_ff,
                });
            }
        };
        let gate_len = d_ff * m;
        let up_len = d_ff * m;
        let down_len = d_model * d_ff;
        let gate = floats[..gate_len].to_vec();
        let up = floats[gate_len..gate_len + up_len].to_vec();
        let down = floats[gate_len + up_len..gate_len + up_len + down_len].to_vec();
        Ok(Self {
            d_model,
            d_ff,
            gate,
            up,
            down,
            col_indices: Some(col_indices.to_vec()),
        })
    }

    /// Run the SwiGLU FFN forward pass on the owned weights. Behaves
    /// identically to [`ExpertWeights::forward`] when `col_indices` is
    /// `None`; otherwise dispatches to [`Self::forward_partial`].
    pub fn forward(&self, x: &[f32]) -> HiddenState {
        if self.col_indices.is_some() {
            self.forward_partial(x)
        } else {
            // Build a borrowed view and reuse the existing forward.
            ExpertWeights {
                d_model: self.d_model,
                d_ff: self.d_ff,
                gate: &self.gate,
                up: &self.up,
                down: &self.down,
            }
            .forward(x)
        }
    }

    /// Forward pass using only the columns listed in `col_indices`.
    /// `gate`/`up` are stored as `[d_ff x M]` packed (column j of the
    /// packed matrix corresponds to original column `col_indices[j]`).
    /// The dot products sum only over the loaded columns of `x`; this
    /// trades a tiny bit of accuracy for proportionally fewer SSD bytes.
    pub fn forward_partial(&self, x: &[f32]) -> HiddenState {
        debug_assert_eq!(x.len(), self.d_model);
        let cols = self
            .col_indices
            .as_ref()
            .expect("forward_partial requires col_indices");
        let m = cols.len();
        debug_assert_eq!(self.gate.len(), self.d_ff * m);
        debug_assert_eq!(self.up.len(), self.d_ff * m);

        // 1) gate / up projections, each summed only over the loaded columns.
        let mut gated = vec![0.0f32; self.d_ff];
        for i in 0..self.d_ff {
            let row_off = i * m;
            let g_row = &self.gate[row_off..row_off + m];
            let u_row = &self.up[row_off..row_off + m];
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for (j, &orig_col) in cols.iter().enumerate() {
                g += g_row[j] * x[orig_col];
                u += u_row[j] * x[orig_col];
            }
            gated[i] = silu(g) * u;
        }

        // 2) Down projection over the full `gated` vector. (Unloaded
        // input columns of x affect only how `gated` was computed, not
        // the down-projection structure.)
        let mut y = vec![0.0f32; self.d_model];
        down_proj(&self.down, &gated, &mut y, self.d_ff);
        y
    }
}

/// Generate the per-token hidden-state vector that flows into the FFN.
///
/// In a real model this would be the residual-stream activation produced
/// by the previous layer's attention block. Here we synthesise it
/// deterministically from `(token_idx, seed)` so a run is replayable and
/// every routed expert sees a non-trivial input. Values are bounded in
/// `[-1, 1]` so the matmul stays numerically tame.
pub fn synth_hidden_state(token_idx: u64, d_model: usize, seed: u64) -> HiddenState {
    let mut s = token_idx
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(seed.wrapping_mul(0xBF58476D1CE4E5B9));
    let mut out = Vec::with_capacity(d_model);
    for _ in 0..d_model {
        // splitmix64 step
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map the top 24 bits to a float in [-1, 1).
        let u = (z >> 40) as u32; // 24 bits
        let f = (u as f32) / ((1u32 << 23) as f32) - 1.0;
        out.push(f);
    }
    out
}

/// Compute the (digest, out_norm) summary fields shared by `run_inference`
/// and its f16 / partial variants. Folded over `f32::to_bits` so the
/// digest is exactly reproducible bit-for-bit between runs.
fn summarise_output(token_idx: u64, expert_id: u32, y: &[f32]) -> InferenceOutput {
    let mut sum_sq = 0.0f64;
    for &v in y {
        sum_sq += (v as f64) * (v as f64);
    }
    let out_norm = sum_sq.sqrt() as f32;
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut digest = FNV_OFFSET ^ token_idx ^ (expert_id as u64);
    for &v in y {
        let bits = v.to_bits() as u64;
        digest ^= bits;
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    InferenceOutput { expert_id, digest, out_norm }
}

/// Run one expert's FFN on the hidden state. The buffer behind `resident`
/// is the bytes that came directly off the SSD via `O_DIRECT`. Returns
/// both the activation vector (for combining with other experts) and an
/// [`InferenceOutput`] summary suitable for logging, or an
/// [`ExpertWeightsError`] if the resident buffer can't be reinterpreted
/// as a valid SwiGLU weight blob (e.g. a truncated / corrupt file).
pub fn run_inference(
    token_idx: u64,
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = ExpertWeights::from_bytes(resident.data(), d_model, d_ff)?;
    let y = weights.forward(x);
    let out = summarise_output(token_idx, resident.id, &y);
    Ok((out, y))
}

/// f16 counterpart of [`run_inference`]: dequantises the resident bytes
/// into an owned `Vec<f32>` (via [`OwnedExpertWeights::from_bytes_f16`])
/// and runs the same SwiGLU forward pass. Used when the on-disk dtype
/// is [`WeightDtype::F16`], halving SSD bytes per expert read.
pub fn run_inference_f16(
    token_idx: u64,
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = OwnedExpertWeights::from_bytes_f16(resident.data(), d_model, d_ff)?;
    let y = weights.forward(x);
    let out = summarise_output(token_idx, resident.id, &y);
    Ok((out, y))
}

/// INT8 counterpart of [`run_inference`]: dequantises the resident bytes
/// (12-byte scale header + per-tensor symmetric `i8` weights) into an
/// owned `Vec<f32>` (via [`OwnedExpertWeights::from_bytes_int8`]) and
/// runs the same SwiGLU forward pass. Used when the on-disk dtype is
/// [`WeightDtype::Int8`], **quartering** SSD bytes per expert read
/// versus F32 — the dominant SSD-bandwidth optimisation in this engine.
pub fn run_inference_int8(
    token_idx: u64,
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = OwnedExpertWeights::from_bytes_int8(resident.data(), d_model, d_ff)?;
    let y = weights.forward(x);
    let out = summarise_output(token_idx, resident.id, &y);
    Ok((out, y))
}

/// Q4_K counterpart of [`run_inference`]: dequantises the resident
/// bytes (a stream of GGUF Q4_K_M super-blocks) into an owned
/// `Vec<f32>` (via [`OwnedExpertWeights::from_bytes_q4k`]) and runs
/// the same SwiGLU forward pass. Used when the on-disk dtype is
/// [`WeightDtype::Q4K`], roughly **doubling** the on-DRAM hot-expert
/// capacity that fits in a given memory budget vs `F16`.
pub fn run_inference_q4k(
    token_idx: u64,
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = OwnedExpertWeights::from_bytes_q4k(resident.data(), d_model, d_ff)?;
    let y = weights.forward(x);
    let out = summarise_output(token_idx, resident.id, &y);
    Ok((out, y))
}

/// Q4_0 counterpart of [`run_inference`]: dequantises the resident
/// bytes (a stream of GGUF Q4_0 18-byte blocks, each holding an `f16`
/// scale and 32 symmetric 4-bit weights) into an owned `Vec<f32>`
/// (via [`OwnedExpertWeights::from_bytes_q4_0`]) and runs the same
/// SwiGLU forward pass. This is the path that lights up when the
/// expert files on disk were produced with the **Q4_0** dtype called
/// out in the "Omniscient Predictive Architecture" design spec —
/// the dequant happens inside the RAM buffer immediately after the
/// `pread(2)` / `io_uring` fetch completes, seamlessly handing off
/// to the existing scalar matmul kernel.
pub fn run_inference_q4_0(
    token_idx: u64,
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = OwnedExpertWeights::from_bytes_q4_0(resident.data(), d_model, d_ff)?;
    let y = weights.forward(x);
    let out = summarise_output(token_idx, resident.id, &y);
    Ok((out, y))
}

/// Partial-load counterpart of [`run_inference`]: reconstructs the
/// expert from a packed-column blob (produced by `read_expert_columns`)
/// and runs [`OwnedExpertWeights::forward_partial`].
#[allow(dead_code)]
pub fn run_inference_partial(
    token_idx: u64,
    resident: &ExpertResident,
    col_indices: &[usize],
    x: &[f32],
    d_model: usize,
    d_ff: usize,
    dtype: WeightDtype,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    let weights = OwnedExpertWeights::from_bytes_partial(
        resident.data(),
        col_indices,
        d_model,
        d_ff,
        dtype,
    )?;
    let y = weights.forward_partial(x);
    let out = summarise_output(token_idx, resident.id, &y);
    Ok((out, y))
}

/// Fold the top-K expert outputs together with a **softmax-gated weighted
/// sum** — the standard Mixtral / Llama-MoE combiner:
///
/// ```text
///   y = sum_i ( scores[i] * outputs[i] )
/// ```
///
/// `scores` must already be normalised over the chosen top-K experts
/// (the caller is responsible for the softmax + re-normalisation —
/// [`crate::gating::LinearGate::route`] does it for the real-transformer
/// path; the benchmark / synthetic path passes uniform `1/k` weights so
/// behaviour is unchanged when no real gate weights are available).
///
/// `outputs` and `scores` must have the same length; if they're empty,
/// the returned vector is empty (the caller is expected to handle that
/// case — the real-transformer path filters out experts that failed to
/// materialise on disk *from both* slices upstream so the weighted sum
/// stays well-defined).
pub fn combine_outputs(outputs: &[HiddenState], scores: &[f32]) -> HiddenState {
    debug_assert_eq!(
        outputs.len(),
        scores.len(),
        "combine_outputs: outputs and scores must have the same length"
    );
    if outputs.is_empty() {
        return Vec::new();
    }
    let d = outputs[0].len();
    let mut out = vec![0.0f32; d];
    for (vec, &s) in outputs.iter().zip(scores.iter()) {
        debug_assert_eq!(vec.len(), d);
        for (o, v) in out.iter_mut().zip(vec.iter()) {
            *o += s * *v;
        }
    }
    out
}

/// Helper: build a uniform `[1/k; k]` score vector for the synthetic /
/// benchmark path that has no real gating network. With these scores the
/// new [`combine_outputs`] reproduces the legacy "uniform average"
/// behaviour exactly.
#[inline]
pub fn uniform_scores(k: usize) -> Vec<f32> {
    if k == 0 {
        Vec::new()
    } else {
        let s = 1.0 / k as f32;
        vec![s; k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned_buffer::AlignedBuffer;

    /// Build a buffer holding deterministic, finite f32 weights so we can
    /// exercise `ExpertWeights::from_bytes` and `forward` without going
    /// through the full storage layer.
    fn make_weights_buffer(d_model: usize, d_ff: usize, fill: f32) -> AlignedBuffer {
        let bytes_needed = expert_weight_bytes(d_model, d_ff);
        // round up to a multiple of 4096 to satisfy AlignedBuffer invariants
        let padded = bytes_needed.div_ceil(4096) * 4096;
        let mut buf = AlignedBuffer::new(padded, 4096);
        for chunk in buf.as_mut_slice().chunks_exact_mut(4) {
            chunk.copy_from_slice(&fill.to_le_bytes());
        }
        buf
    }

    #[test]
    fn weights_view_partitions_buffer_correctly() {
        let d_model = 4;
        let d_ff = 8;
        let buf = make_weights_buffer(d_model, d_ff, 0.25);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        assert_eq!(w.gate.len(), d_ff * d_model);
        assert_eq!(w.up.len(), d_ff * d_model);
        assert_eq!(w.down.len(), d_model * d_ff);
        assert!(w.gate.iter().all(|&v| v == 0.25));
        assert!(w.down.iter().all(|&v| v == 0.25));
    }

    #[test]
    fn forward_produces_finite_output_of_correct_shape() {
        let d_model = 16;
        let d_ff = 32;
        // Use small weights so silu*x*x stays in a well-behaved range.
        let buf = make_weights_buffer(d_model, d_ff, 0.05);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(7, d_model, 1234);
        let y = w.forward(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()), "got non-finite output: {y:?}");
    }

    #[test]
    fn forward_is_deterministic() {
        let d_model = 8;
        let d_ff = 16;
        let buf = make_weights_buffer(d_model, d_ff, 0.1);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(42, d_model, 99);
        let a = w.forward(&x);
        let b = w.forward(&x);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_weights_yield_zero_output() {
        let d_model = 4;
        let d_ff = 4;
        let buf = make_weights_buffer(d_model, d_ff, 0.0);
        let w = ExpertWeights::from_bytes(buf.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(1, d_model, 1);
        let y = w.forward(&x);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn from_bytes_rejects_truncated_buffer() {
        let d_model = 4;
        let d_ff = 8;
        let buf = make_weights_buffer(d_model, d_ff, 0.0);
        let need = expert_weight_bytes(d_model, d_ff);
        // Hand it a slice that is one f32 short of the requirement.
        let truncated = &buf.as_slice()[..need - 4];
        let err = ExpertWeights::from_bytes(truncated, d_model, d_ff)
            .err()
            .expect("expected an error from a truncated buffer");
        match err {
            ExpertWeightsError::BufferTooSmall { have, need: n, .. } => {
                assert_eq!(have, need - 4);
                assert_eq!(n, need);
            }
            other => panic!("expected BufferTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn expert_weight_count_saturates_on_overflow() {
        // d_model * d_ff would overflow usize on every supported target.
        // saturating_mul must clamp to usize::MAX so downstream size
        // checks (which compare against finite buffer lengths) reliably
        // fail rather than silently wrap.
        let huge = usize::MAX;
        assert_eq!(expert_weight_count(huge, 2), usize::MAX);
        assert_eq!(expert_weight_bytes(huge, 2), usize::MAX);
    }

    #[test]
    fn synth_hidden_state_is_bounded_and_deterministic() {
        let a = synth_hidden_state(123, 32, 7);
        let b = synth_hidden_state(123, 32, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
        assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn silu_basic_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(x) -> x for large positive x
        assert!((silu(20.0) - 20.0).abs() < 1e-3);
        // silu(x) -> 0 for large negative x
        assert!(silu(-20.0).abs() < 1e-3);
    }

    #[test]
    fn combine_outputs_uniform_scores_averages() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![3.0, 2.0, 1.0];
        let c = combine_outputs(&[a, b], &uniform_scores(2));
        assert_eq!(c, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn from_bytes_int8_dequantizes_with_per_tensor_scales() {
        // Build an INT8 expert blob by hand and verify dequantisation
        // produces the expected f32 weights (within rounding).
        let d_model = 4;
        let d_ff = 4;
        let count = expert_weight_count(d_model, d_ff);
        // Use distinct scales per tensor so a bug that confuses them
        // would visibly break the assertions.
        let meta = Int8ExpertMeta {
            gate_scale: 0.01,
            up_scale: 0.02,
            down_scale: 0.04,
        };
        let mut bytes = Vec::with_capacity(INT8_HEADER_BYTES + count);
        bytes.extend_from_slice(&meta.to_bytes());
        let gate_len = d_ff * d_model;
        let up_len = d_ff * d_model;
        let down_len = d_model * d_ff;
        // Fill with deterministic int8 values.
        for i in 0..gate_len {
            bytes.push((((i as i32) % 7) - 3) as i8 as u8);
        }
        for i in 0..up_len {
            bytes.push((((i as i32) % 5) - 2) as i8 as u8);
        }
        for i in 0..down_len {
            bytes.push((((i as i32) % 9) - 4) as i8 as u8);
        }
        let w = OwnedExpertWeights::from_bytes_int8(&bytes, d_model, d_ff).unwrap();
        // Spot-check: gate[0] = (0 % 7 - 3) * 0.01 = -0.03
        assert!((w.gate[0] - (-0.03)).abs() < 1e-6);
        assert!((w.up[0] - (-0.04)).abs() < 1e-6);
        assert!((w.down[0] - (-0.16)).abs() < 1e-6);
        // Forward must produce a finite vector (smoke test).
        let x = vec![0.1; d_model];
        let y = w.forward(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn int8_short_buffer_returns_error() {
        let bytes = vec![0u8; INT8_HEADER_BYTES + 4]; // way too small
        let res = OwnedExpertWeights::from_bytes_int8(&bytes, 8, 8);
        assert!(matches!(res, Err(ExpertWeightsError::BufferTooSmall { .. })));
    }

    #[test]
    fn combine_outputs_weighted_sum_uses_scores() {
        let d = 4;
        let outs = vec![vec![1.0; d], vec![2.0; d], vec![4.0; d]];
        // 0.5*1 + 0.25*2 + 0.25*4 = 2.0
        let scores = vec![0.5, 0.25, 0.25];
        let y = combine_outputs(&outs, &scores);
        for v in y {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn combine_outputs_empty_inputs() {
        let y = combine_outputs(&[], &[]);
        assert!(y.is_empty());
    }

    #[test]
    fn uniform_scores_sums_to_one() {
        let s = uniform_scores(4);
        assert_eq!(s.len(), 4);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(uniform_scores(0), Vec::<f32>::new());
    }

    /// Build a `Vec<u8>` containing `expert_weight_count` weights as
    /// little-endian f16 with the specified fill value.
    fn make_f16_buffer(d_model: usize, d_ff: usize, fill: f32) -> Vec<u8> {
        let n = expert_weight_count(d_model, d_ff);
        let h = half::f16::from_f32(fill);
        let mut v = Vec::with_capacity(n * 2);
        for _ in 0..n {
            v.extend_from_slice(&h.to_bits().to_le_bytes());
        }
        v
    }

    #[test]
    fn dequantize_f16_round_trips_known_values() {
        let src: Vec<u8> = [1.0_f32, -0.5, 0.25, 2.0]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let mut dst = Vec::new();
        dequantize_f16_to_f32(&src, &mut dst);
        assert_eq!(dst.len(), 4);
        assert!((dst[0] - 1.0).abs() < 1e-3);
        assert!((dst[1] - (-0.5)).abs() < 1e-3);
        assert!((dst[2] - 0.25).abs() < 1e-3);
        assert!((dst[3] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn from_bytes_f16_round_trips() {
        // f16 path must produce a finite output of the right shape, and
        // it must agree with the f32 forward computed on the same fill
        // value to within f16 quantisation noise.
        let d_model = 16;
        let d_ff = 32;
        let fill = 0.05_f32;

        let f16_bytes = make_f16_buffer(d_model, d_ff, fill);
        let weights16 = OwnedExpertWeights::from_bytes_f16(&f16_bytes, d_model, d_ff).unwrap();
        let x = synth_hidden_state(7, d_model, 1234);
        let y16 = weights16.forward(&x);
        assert_eq!(y16.len(), d_model);
        assert!(y16.iter().all(|v| v.is_finite()));

        let f32_buf = make_weights_buffer(d_model, d_ff, fill);
        let w32 = ExpertWeights::from_bytes(f32_buf.as_slice(), d_model, d_ff).unwrap();
        let y32 = w32.forward(&x);
        for (a, b) in y16.iter().zip(y32.iter()) {
            assert!(
                (a - b).abs() < 1e-2,
                "f16 forward diverged from f32: {a} vs {b}"
            );
        }
    }

    #[test]
    fn f16_bytes_helper_is_half_of_f32() {
        let d_model = 32;
        let d_ff = 64;
        assert_eq!(
            expert_weight_bytes_f16(d_model, d_ff) * 2,
            expert_weight_bytes(d_model, d_ff)
        );
    }

    /// Build a packed-column buffer for partial loading. With `m == d_model`
    /// the packed layout is identical to the full f32 layout (column ids
    /// are [0..d_model]), so partial forward must equal full forward.
    fn make_packed_partial_buffer(
        d_model: usize,
        d_ff: usize,
        cols: &[usize],
        fill: f32,
    ) -> Vec<u8> {
        let m = cols.len();
        // Full gate / up matrices we'd produce, then pick columns out.
        let _ = (d_model, d_ff, fill);
        // For a constant fill, every column is `fill`, so packed = fill repeated.
        let mut out = Vec::new();
        for _ in 0..(d_ff * m) {
            out.extend_from_slice(&fill.to_le_bytes());
        }
        for _ in 0..(d_ff * m) {
            out.extend_from_slice(&fill.to_le_bytes());
        }
        for _ in 0..(d_model * d_ff) {
            out.extend_from_slice(&fill.to_le_bytes());
        }
        out
    }

    #[test]
    fn partial_forward_matches_full_for_large_fraction() {
        // With M == d_model (fraction 1.0) and column indices [0..d_model],
        // partial forward and full forward must be bit-for-bit equal.
        let d_model = 8;
        let d_ff = 16;
        let fill = 0.1;
        let cols: Vec<usize> = (0..d_model).collect();
        let packed = make_packed_partial_buffer(d_model, d_ff, &cols, fill);
        let owned = OwnedExpertWeights::from_bytes_partial(
            &packed,
            &cols,
            d_model,
            d_ff,
            WeightDtype::F32,
        )
        .unwrap();

        let full = make_weights_buffer(d_model, d_ff, fill);
        let full_view = ExpertWeights::from_bytes(full.as_slice(), d_model, d_ff).unwrap();
        let x = synth_hidden_state(3, d_model, 9);
        let y_partial = owned.forward_partial(&x);
        let y_full = full_view.forward(&x);
        for (a, b) in y_partial.iter().zip(y_full.iter()) {
            assert!((a - b).abs() < 1e-5, "partial vs full: {a} vs {b}");
        }
    }

    #[test]
    fn partial_forward_produces_finite_output() {
        let d_model = 8;
        let d_ff = 16;
        let fill = 0.05;
        // M = d_model / 2 — partial-load fraction 0.5.
        let cols: Vec<usize> = (0..d_model).step_by(2).collect();
        assert_eq!(cols.len(), 4);
        let packed = make_packed_partial_buffer(d_model, d_ff, &cols, fill);
        let owned = OwnedExpertWeights::from_bytes_partial(
            &packed,
            &cols,
            d_model,
            d_ff,
            WeightDtype::F32,
        )
        .unwrap();
        let x = synth_hidden_state(0, d_model, 1);
        let y = owned.forward_partial(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn weight_dtype_round_trips_through_string() {
        assert_eq!(WeightDtype::from_str_opt("f32"), Some(WeightDtype::F32));
        assert_eq!(WeightDtype::from_str_opt("F16"), Some(WeightDtype::F16));
        assert_eq!(WeightDtype::from_str_opt("fp16"), Some(WeightDtype::F16));
        assert_eq!(WeightDtype::from_str_opt("bogus"), None);
        assert_eq!(WeightDtype::F32.as_str(), "f32");
        assert_eq!(WeightDtype::F16.as_str(), "f16");
        assert_eq!(WeightDtype::from_str_opt("q4k"), Some(WeightDtype::Q4K));
        assert_eq!(WeightDtype::from_str_opt("Q4_K_M"), Some(WeightDtype::Q4K));
        assert_eq!(WeightDtype::Q4K.as_str(), "q4k");
    }

    // ------------------------- Q4_K tests ------------------------------

    /// Build one synthetic Q4_K block whose every weight equals
    /// `d * scale6 * q - dmin * min6` for the given parameters. Returns
    /// the 144-byte block and the expected dequantised float vector.
    fn make_q4k_block(
        d: f32,
        dmin: f32,
        sub_pairs: [(u8, u8); Q4K_SUBBLOCKS],
        nibble_fill: u8,
    ) -> ([u8; Q4K_BLOCK_BYTES], [f32; Q4K_BLOCK_ELEMS]) {
        let mut blk = [0u8; Q4K_BLOCK_BYTES];
        let d16 = half::f16::from_f32(d).to_bits().to_le_bytes();
        let dm16 = half::f16::from_f32(dmin).to_bits().to_le_bytes();
        blk[0..2].copy_from_slice(&d16);
        blk[2..4].copy_from_slice(&dm16);
        let s = q4k_pack_scales(&sub_pairs);
        blk[4..16].copy_from_slice(&s);
        // Pack the same 4-bit nibble (low and high) in every byte.
        let q = nibble_fill & 0x0F;
        let byte = q | (q << 4);
        for i in 16..16 + 128 {
            blk[i] = byte;
        }
        let d_decoded = half::f16::from_f32(d).to_f32();
        let dmin_decoded = half::f16::from_f32(dmin).to_f32();
        let mut expected = [0.0f32; Q4K_BLOCK_ELEMS];
        for j in 0..Q4K_SUBBLOCKS {
            let (sc, mn) = sub_pairs[j];
            let s = d_decoded * sc as f32;
            let m = dmin_decoded * mn as f32;
            for i in 0..Q4K_SUBBLOCK_ELEMS {
                expected[j * Q4K_SUBBLOCK_ELEMS + i] = s * (q as f32) - m;
            }
        }
        (blk, expected)
    }

    #[test]
    fn q4k_pack_unpack_round_trips_random_pairs() {
        // Every (scale6, min6) pair in 0..64 must survive a pack/unpack.
        let pairs: [(u8, u8); Q4K_SUBBLOCKS] = [
            (0, 0),
            (1, 63),
            (15, 32),
            (33, 7),
            (63, 1),
            (16, 16),
            (47, 48),
            (8, 9),
        ];
        let packed = q4k_pack_scales(&pairs);
        let unpacked = q4k_unpack_scales(&packed);
        assert_eq!(unpacked, pairs, "packed bytes: {:?}", packed);
    }

    #[test]
    fn dequantize_q4k_block_matches_reference_formula() {
        let pairs: [(u8, u8); Q4K_SUBBLOCKS] = [
            (10, 5),
            (12, 7),
            (14, 9),
            (16, 11),
            (18, 13),
            (20, 15),
            (22, 17),
            (24, 19),
        ];
        let (blk, expected) = make_q4k_block(0.25, 0.125, pairs, 0x07);
        let mut dst = vec![0.0f32; Q4K_BLOCK_ELEMS];
        dequantize_q4k_block(&blk, &mut dst);
        for (a, b) in dst.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "Q4K dequant diverged: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn dequantize_q4k_to_f32_truncates_to_exact_count() {
        // Dequantise 1.5 blocks worth of weights.
        let pairs = [(8u8, 4u8); Q4K_SUBBLOCKS];
        let (blk, _) = make_q4k_block(0.5, 0.25, pairs, 3);
        let mut src = Vec::new();
        src.extend_from_slice(&blk);
        src.extend_from_slice(&blk);
        let mut dst = Vec::new();
        dequantize_q4k_to_f32(&src, 256 + 100, &mut dst);
        assert_eq!(dst.len(), 256 + 100);
        // First 256 must equal the per-element formula.
        let d = half::f16::from_f32(0.5).to_f32();
        let dmin = half::f16::from_f32(0.25).to_f32();
        let expected = d * 8.0 * 3.0 - dmin * 4.0;
        for v in &dst[..256] {
            assert!((v - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn q4k_expert_bytes_round_to_block() {
        // Choose dimensions so neither tensor lands exactly on 256.
        let d_model = 32; // gate/up = 32*64=2048 = 8 blocks; down = 32*64 = 8.
        let d_ff = 64;
        let total = expert_weight_bytes_for(d_model, d_ff, WeightDtype::Q4K);
        // 3 matrices × 8 blocks × 144 bytes = 3456.
        assert_eq!(total, 3 * 8 * Q4K_BLOCK_BYTES);
    }

    #[test]
    fn from_bytes_q4k_round_trips_to_owned_weights() {
        // Build a tiny expert with constant Q4_K weights and verify
        // forward produces a finite output of the right shape.
        let d_model: usize = 16;
        let d_ff: usize = 32; // 16*32 = 512 = 2 blocks per matrix; 6 blocks total.
        let pairs = [(4u8, 2u8); Q4K_SUBBLOCKS];
        let (blk, _expected) = make_q4k_block(0.1, 0.05, pairs, 5);
        let blocks_per_matrix = (d_model * d_ff).div_ceil(Q4K_BLOCK_ELEMS as usize);
        let mut bytes = Vec::new();
        for _ in 0..(3 * blocks_per_matrix) {
            bytes.extend_from_slice(&blk);
        }
        let w = OwnedExpertWeights::from_bytes_q4k(&bytes, d_model, d_ff).unwrap();
        // All weights are constant; forward must produce a finite vector.
        assert_eq!(w.gate.len(), d_model * d_ff);
        assert_eq!(w.up.len(), d_model * d_ff);
        assert_eq!(w.down.len(), d_model * d_ff);
        let x = synth_hidden_state(0, d_model, 1);
        let y = w.forward(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn from_bytes_q4k_rejects_short_buffer() {
        let d_model = 16;
        let d_ff = 32;
        let bytes = vec![0u8; 100]; // way too small
        let res = OwnedExpertWeights::from_bytes_q4k(&bytes, d_model, d_ff);
        assert!(matches!(res, Err(ExpertWeightsError::BufferTooSmall { .. })));
    }

    // -----------------------------------------------------------------
    // Q4_0 tests.
    // -----------------------------------------------------------------

    #[test]
    fn q4_0_dtype_round_trips_through_str() {
        assert_eq!(WeightDtype::from_str_opt("q4_0"), Some(WeightDtype::Q4_0));
        assert_eq!(WeightDtype::from_str_opt("Q4_0"), Some(WeightDtype::Q4_0));
        assert_eq!(WeightDtype::from_str_opt("q40"), Some(WeightDtype::Q4_0));
        assert_eq!(WeightDtype::from_str_opt("q4"), Some(WeightDtype::Q4_0));
        assert_eq!(WeightDtype::Q4_0.as_str(), "q4_0");
    }

    #[test]
    fn q4_0_block_constants_match_spec() {
        // The design spec calls out "every 32 weights share an f16
        // scale". Enforce both halves of that contract here so a future
        // refactor can't silently change the on-disk layout.
        assert_eq!(Q4_0_BLOCK_ELEMS, 32);
        assert_eq!(Q4_0_BLOCK_BYTES, 2 + 16); // f16 + 16 nibble bytes
    }

    #[test]
    fn q4_0_quantize_dequantize_round_trips_to_low_error() {
        // Random-ish but deterministic block of 32 weights. After
        // round-trip through Q4_0 we expect low absolute error
        // (bounded by `d/2 = max_abs/14`, since 4-bit symmetric
        // quantisation has 16 levels covering 2*max_abs).
        let mut src = [0.0f32; Q4_0_BLOCK_ELEMS];
        let mut state: u64 = 0xCAFEBABE;
        let mut max_abs = 0.0f32;
        for slot in src.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 40) as u32;
            let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
            *slot = unit * 2.5; // arbitrary scale
            max_abs = max_abs.max(slot.abs());
        }
        let mut blk = [0u8; Q4_0_BLOCK_BYTES];
        quantize_q4_0_block(&src, &mut blk);
        let mut decoded = [0.0f32; Q4_0_BLOCK_ELEMS];
        dequantize_q4_0_block(&blk, &mut decoded);
        // Per-element error <= d (= max_abs/7). Sum-of-squares norm
        // bound is the per-element bound times sqrt(N).
        let d = max_abs / 7.0;
        for (a, b) in src.iter().zip(decoded.iter()) {
            let err = (a - b).abs();
            assert!(err <= d * 1.01, "err {err} exceeds bound {d}");
        }
    }

    #[test]
    fn q4_0_zero_block_round_trips_to_zero() {
        // All-zero weights must encode to d=0, q4=8 and decode to 0.
        let src = [0.0f32; Q4_0_BLOCK_ELEMS];
        let mut blk = [0u8; Q4_0_BLOCK_BYTES];
        quantize_q4_0_block(&src, &mut blk);
        // Scale field == 0.
        assert_eq!(u16::from_le_bytes([blk[0], blk[1]]), 0);
        let mut decoded = [0.0f32; Q4_0_BLOCK_ELEMS];
        dequantize_q4_0_block(&blk, &mut decoded);
        assert!(decoded.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_0_expert_bytes_round_to_block_per_tensor() {
        // d_model=8, d_ff=8 → each tensor = 64 weights = 2 blocks.
        // 3 tensors × 2 blocks × 18 bytes = 108 bytes.
        let total = expert_weight_bytes_for(8, 8, WeightDtype::Q4_0);
        assert_eq!(total, 3 * 2 * Q4_0_BLOCK_BYTES);

        // d_model=10, d_ff=10 → each tensor = 100 weights = 4 blocks
        // (100 / 32 = 3.125, ceil = 4). 3 × 4 × 18 = 216 bytes.
        let total2 = expert_weight_bytes_for(10, 10, WeightDtype::Q4_0);
        assert_eq!(total2, 3 * 4 * Q4_0_BLOCK_BYTES);
    }

    #[test]
    fn from_bytes_q4_0_round_trips_to_owned_weights() {
        // Tiny expert with constant Q4_0 weights → forward produces
        // a finite vector of the right shape, and the Q4_0 dequant
        // yields the same numeric tensor as the round-trip from
        // `quantize_q4_0_block`.
        let d_model: usize = 8;
        let d_ff: usize = 8; // each tensor = 64 weights = 2 blocks.
        let blocks_per_matrix = (d_model * d_ff).div_ceil(Q4_0_BLOCK_ELEMS);
        // Encode a small constant block: every weight = 1.0.
        let src = [1.0f32; Q4_0_BLOCK_ELEMS];
        let mut blk = [0u8; Q4_0_BLOCK_BYTES];
        quantize_q4_0_block(&src, &mut blk);
        let mut bytes = Vec::new();
        for _ in 0..(3 * blocks_per_matrix) {
            bytes.extend_from_slice(&blk);
        }
        let w = OwnedExpertWeights::from_bytes_q4_0(&bytes, d_model, d_ff).unwrap();
        assert_eq!(w.gate.len(), d_model * d_ff);
        assert_eq!(w.up.len(), d_model * d_ff);
        assert_eq!(w.down.len(), d_model * d_ff);
        // Every dequantised weight is approximately 1.0 (bounded by
        // the Q4_0 quantisation error, ≈ max_abs/7 = 1/7).
        for &v in w.gate.iter().chain(w.up.iter()).chain(w.down.iter()) {
            assert!((v - 1.0).abs() < 0.2, "weight {v} too far from 1.0");
        }
        let x = synth_hidden_state(0, d_model, 1);
        let y = w.forward(&x);
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn from_bytes_q4_0_rejects_short_buffer() {
        let d_model = 16;
        let d_ff = 32;
        let bytes = vec![0u8; 16]; // way too small
        let res = OwnedExpertWeights::from_bytes_q4_0(&bytes, d_model, d_ff);
        assert!(matches!(res, Err(ExpertWeightsError::BufferTooSmall { .. })));
    }

    /// Scalar reference for `gate_up_swiglu`, deliberately independent
    /// of the feature-gated SIMD / BLAS paths so the comparison
    /// remains meaningful when those features are enabled.
    fn gate_up_swiglu_reference(
        gate: &[f32],
        up: &[f32],
        x: &[f32],
        gated: &mut [f32],
        d_model: usize,
    ) {
        let d_ff = gated.len();
        for i in 0..d_ff {
            let g_row = &gate[i * d_model..(i + 1) * d_model];
            let u_row = &up[i * d_model..(i + 1) * d_model];
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for j in 0..d_model {
                g += g_row[j] * x[j];
                u += u_row[j] * x[j];
            }
            gated[i] = silu(g) * u;
        }
    }

    /// The BLAS branch of `gate_up_swiglu` writes `g` directly into the
    /// caller's `gated` slot and uses a thread-local scratch buffer
    /// for `u`. This test, gated on the `blas` cargo feature, asserts
    /// that:
    ///   1. Outputs match the scalar reference within f32 tolerance
    ///      (i.e. the in-place rewrite of `gated[i] = silu(gated[i]) * u[i]`
    ///      is correct).
    ///   2. The thread-local scratch is reused correctly across
    ///      successive calls, including a call that grows `d_ff`
    ///      (which forces a `resize`) followed by a call that shrinks
    ///      it back (which must still produce correct results from the
    ///      first `d_ff` elements of the now-larger scratch).
    #[cfg(feature = "blas")]
    #[test]
    fn gate_up_swiglu_blas_matches_scalar_reference_and_reuses_scratch() {
        // Deterministic small weights / inputs.
        fn fill_deterministic(buf: &mut [f32], seed: u64) {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for v in buf.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Map to small range so silu * x stays well-behaved.
                *v = ((s as i32 as f32) / i32::MAX as f32) * 0.1;
            }
        }

        let cases = [
            (8usize, 16usize),   // small
            (8, 32),             // grow scratch (d_ff bigger than first call)
            (8, 16),             // shrink back — scratch len stays >= d_ff
            (4, 4),              // tiny, exercises d_ff == d_model
        ];

        for (idx, &(d_model, d_ff)) in cases.iter().enumerate() {
            let mut gate = vec![0.0f32; d_ff * d_model];
            let mut up = vec![0.0f32; d_ff * d_model];
            let mut x = vec![0.0f32; d_model];
            fill_deterministic(&mut gate, 1 + idx as u64);
            fill_deterministic(&mut up, 1001 + idx as u64);
            fill_deterministic(&mut x, 2003 + idx as u64);

            let mut out_blas = vec![0.0f32; d_ff];
            gate_up_swiglu(&gate, &up, &x, &mut out_blas, d_model);

            let mut out_ref = vec![0.0f32; d_ff];
            gate_up_swiglu_reference(&gate, &up, &x, &mut out_ref, d_model);

            assert_eq!(out_blas.len(), out_ref.len());
            for i in 0..d_ff {
                assert!(
                    out_blas[i].is_finite() && out_ref[i].is_finite(),
                    "non-finite output at case {idx}, index {i}"
                );
                let diff = (out_blas[i] - out_ref[i]).abs();
                let tol = 1e-4 * out_ref[i].abs().max(1.0);
                assert!(
                    diff <= tol,
                    "blas vs scalar mismatch at case {idx}, idx {i}: \
                     blas={} ref={} diff={diff} tol={tol}",
                    out_blas[i], out_ref[i]
                );
            }
        }
    }
}
