//! **Unified Tensor Header (U.T.H.)** — gist Phase 3.
//!
//! Every expert blob emitted by `gguf-convert` can be prefixed with a
//! fixed-size 64-byte header that tells the loader, *before the first
//! byte of weights even arrives*, exactly which dequant / SIMD kernel
//! to dispatch to. The header is **page-padded** so the first weight
//! byte still lands at a 4 KiB offset — preserving the `O_DIRECT`
//! invariants the engine relies on.
//!
//! ```text
//!   offset  size  field                  notes
//!   ------  ----  ---------------------  -------------------------------------
//!     0      4    magic = b"UTH1"        recognise/skip on read
//!     4      2    version (u16 LE)       starts at 1
//!     6      1    dtype_id (u8)          mirrors WeightDtype discriminant
//!     7      1    shape_rank (u8)        1..=4
//!     8     16    shape[4] (u32 LE)      0-padded; row-major
//!    24      4    quant_scale_offset     bytes from start of payload to scales
//!    28      4    quant_scale_count      number of f32 scales, 0 if none/inline
//!    32      4    amx_tile_hint_m (u32)  preferred AMX tile dims
//!    36      4    amx_tile_hint_n (u32)
//!    40      4    amx_tile_hint_k (u32)
//!    44      4    flags (u32 LE)         bit0: page-aligned-after-header
//!    48     16    reserved (zero)
//!   ------------
//!     64    bytes total
//! ```
//!
//! Why 64 bytes and not the more natural 4 KiB? Two reasons:
//!
//! * 64 bytes is one cache line on every CPU the engine targets, so the
//!   header fits in the L1d line that brings in the first weights and
//!   the kernel dispatcher never pays a separate fetch.
//! * The disk-side **page alignment** invariant required by `O_DIRECT`
//!   is preserved by padding the header *region* up to the configured
//!   block alignment (default 4 KiB) — the writer in `gguf_loader`
//!   does this transparently. The header itself is therefore just a
//!   self-describing prefix of the same 4 KiB block, with no extra I/O.
//!
//! The reader is **best-effort and backwards-compatible**: blobs without
//! the `UTH1` magic at byte 0 are returned unchanged, so older
//! `expert_<id>.bin` files keep working with no flag.

use crate::inference::WeightDtype;
use std::fmt;

/// 4-byte ASCII magic at the start of every U.T.H. blob.
pub const UTH_MAGIC: [u8; 4] = *b"UTH1";

/// 4-byte ASCII magic for the mixed-projection expert header.
pub const UTH2_MAGIC: [u8; 4] = *b"UTH2";

/// Total on-disk size of the header itself (excluding any page padding
/// that may follow it).
pub const UTH_BYTES: usize = 64;

/// Current header version. Incremented on incompatible layout changes.
pub const UTH_VERSION: u16 = 1;

/// Current mixed-projection header version.
pub const UTH2_VERSION: u16 = 2;

/// Fixed on-disk size of the UTH2 header itself, excluding page padding.
pub const UTH2_BYTES: usize = 128;

/// `flags` bit indicating that the payload (post-header) is itself
/// page-aligned (i.e. the writer padded the header region up to the
/// engine's block alignment before emitting weights).
pub const UTH_FLAG_PAGE_ALIGNED_PAYLOAD: u32 = 1 << 0;

/// Maximum supported shape rank — fits in the 16 `shape` bytes.
pub const UTH_MAX_RANK: usize = 4;

/// Wire-stable dtype tag. The mapping is a contract; do **not**
/// reorder these without bumping `UTH_VERSION`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UthDtypeId {
    F32 = 0,
    F16 = 1,
    Int8 = 2,
    Q4K = 3,
    Q4_0 = 4,
    Q8_0 = 5,
    BF16 = 6,
    MXFP4 = 7,
    Q5K = 8,
    Q6K = 9,
    Mixed = 10,
}

impl UthDtypeId {
    pub fn from_weight(d: WeightDtype) -> Self {
        match d {
            WeightDtype::F32 => UthDtypeId::F32,
            WeightDtype::F16 => UthDtypeId::F16,
            WeightDtype::Int8 => UthDtypeId::Int8,
            WeightDtype::Q4K => UthDtypeId::Q4K,
            WeightDtype::Q4_0 => UthDtypeId::Q4_0,
            WeightDtype::Q8_0 => UthDtypeId::Q8_0,
            WeightDtype::BF16 => UthDtypeId::BF16,
            WeightDtype::MXFP4 => UthDtypeId::MXFP4,
            WeightDtype::Q5K => UthDtypeId::Q5K,
            WeightDtype::Q6K => UthDtypeId::Q6K,
            WeightDtype::Mixed => UthDtypeId::Mixed,
        }
    }

    pub fn to_weight(self) -> WeightDtype {
        match self {
            UthDtypeId::F32 => WeightDtype::F32,
            UthDtypeId::F16 => WeightDtype::F16,
            UthDtypeId::Int8 => WeightDtype::Int8,
            UthDtypeId::Q4K => WeightDtype::Q4K,
            UthDtypeId::Q4_0 => WeightDtype::Q4_0,
            UthDtypeId::Q8_0 => WeightDtype::Q8_0,
            UthDtypeId::BF16 => WeightDtype::BF16,
            UthDtypeId::MXFP4 => WeightDtype::MXFP4,
            UthDtypeId::Q5K => WeightDtype::Q5K,
            UthDtypeId::Q6K => WeightDtype::Q6K,
            UthDtypeId::Mixed => WeightDtype::Mixed,
        }
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(UthDtypeId::F32),
            1 => Some(UthDtypeId::F16),
            2 => Some(UthDtypeId::Int8),
            3 => Some(UthDtypeId::Q4K),
            4 => Some(UthDtypeId::Q4_0),
            5 => Some(UthDtypeId::Q8_0),
            6 => Some(UthDtypeId::BF16),
            7 => Some(UthDtypeId::MXFP4),
            8 => Some(UthDtypeId::Q5K),
            9 => Some(UthDtypeId::Q6K),
            10 => Some(UthDtypeId::Mixed),
            _ => None,
        }
    }
}

/// One projection range inside a mixed UTH2 expert payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionRange {
    pub dtype: UthDtypeId,
    pub offset: u64,
    pub len: u64,
    pub weights: u32,
}

impl ProjectionRange {
    pub fn end(self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }
}

/// Parsed UTH2 mixed expert header. Offsets are relative to the payload
/// start, not the start of the file, so page padding after the header is
/// never part of any projection range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixedExpertHeader {
    pub version: u16,
    pub d_model: u32,
    pub d_ff: u32,
    pub flags: u32,
    pub gate: ProjectionRange,
    pub up: ProjectionRange,
    pub down: ProjectionRange,
}

impl MixedExpertHeader {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        gate: ProjectionRange,
        up: ProjectionRange,
        down: ProjectionRange,
    ) -> Self {
        Self {
            version: UTH2_VERSION,
            d_model: d_model as u32,
            d_ff: d_ff as u32,
            flags: UTH_FLAG_PAGE_ALIGNED_PAYLOAD,
            gate,
            up,
            down,
        }
    }

    #[cfg(test)]
    pub fn projection_payload_len(&self) -> Option<u64> {
        [self.gate, self.up, self.down]
            .into_iter()
            .map(ProjectionRange::end)
            .try_fold(0u64, |max_end, end| end.map(|e| max_end.max(e)))
    }

    pub fn validate(&self, payload_len: u64) -> Result<(), String> {
        if self.version != UTH2_VERSION {
            return Err(format!("unsupported UTH2 version {}", self.version));
        }
        if self.d_model == 0 || self.d_ff == 0 {
            return Err("UTH2 d_model and d_ff must be non-zero".to_string());
        }
        let ranges = [("gate", self.gate), ("up", self.up), ("down", self.down)];
        for (name, r) in ranges {
            if r.dtype == UthDtypeId::Mixed {
                return Err(format!("UTH2 {name} projection cannot use mixed dtype id"));
            }
            if r.len == 0 {
                return Err(format!("UTH2 {name} projection has zero length"));
            }
            let end = r
                .end()
                .ok_or_else(|| format!("UTH2 {name} range overflows u64"))?;
            if end > payload_len {
                return Err(format!(
                    "UTH2 {name} range {}..{} exceeds payload length {}",
                    r.offset, end, payload_len
                ));
            }
        }
        let mut sorted = [
            (
                "gate",
                self.gate.offset,
                self.gate.end().unwrap_or(u64::MAX),
            ),
            ("up", self.up.offset, self.up.end().unwrap_or(u64::MAX)),
            (
                "down",
                self.down.offset,
                self.down.end().unwrap_or(u64::MAX),
            ),
        ];
        sorted.sort_by_key(|(_, start, _)| *start);
        for pair in sorted.windows(2) {
            let (left_name, _left_start, left_end) = pair[0];
            let (right_name, right_start, _right_end) = pair[1];
            if left_end > right_start {
                return Err(format!(
                    "UTH2 projection ranges overlap: {left_name} and {right_name}"
                ));
            }
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> [u8; UTH2_BYTES] {
        let mut buf = [0u8; UTH2_BYTES];
        buf[0..4].copy_from_slice(&UTH2_MAGIC);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&3u16.to_le_bytes());
        buf[8..12].copy_from_slice(&self.d_model.to_le_bytes());
        buf[12..16].copy_from_slice(&self.d_ff.to_le_bytes());
        buf[16..20].copy_from_slice(&self.flags.to_le_bytes());

        for (i, (role, p)) in [(0u8, self.gate), (1u8, self.up), (2u8, self.down)]
            .into_iter()
            .enumerate()
        {
            let off = 24 + i * 24;
            buf[off] = p.dtype as u8;
            buf[off + 1] = role;
            buf[off + 4..off + 12].copy_from_slice(&p.offset.to_le_bytes());
            buf[off + 12..off + 20].copy_from_slice(&p.len.to_le_bytes());
            buf[off + 20..off + 24].copy_from_slice(&p.weights.to_le_bytes());
        }
        buf
    }

    pub fn write_padded(&self, block_align: usize, dst: &mut Vec<u8>) {
        let start = dst.len();
        dst.extend_from_slice(&self.to_bytes());
        let after = dst.len() - start;
        let pad = if block_align > 0 {
            (block_align - (after % block_align)) % block_align
        } else {
            0
        };
        dst.resize(dst.len() + pad, 0);
    }

    pub fn probe(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < UTH2_BYTES || bytes[0..4] != UTH2_MAGIC {
            return None;
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != UTH2_VERSION {
            return None;
        }
        let projection_count = u16::from_le_bytes([bytes[6], bytes[7]]);
        if projection_count != 3 {
            return None;
        }
        let d_model = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let d_ff = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let flags = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

        let read_projection = |idx: usize, expected_role: u8| -> Option<ProjectionRange> {
            let off = 24 + idx * 24;
            if bytes[off + 1] != expected_role {
                return None;
            }
            let dtype = UthDtypeId::from_u8(bytes[off])?;
            let offset = u64::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
                bytes[off + 8],
                bytes[off + 9],
                bytes[off + 10],
                bytes[off + 11],
            ]);
            let len = u64::from_le_bytes([
                bytes[off + 12],
                bytes[off + 13],
                bytes[off + 14],
                bytes[off + 15],
                bytes[off + 16],
                bytes[off + 17],
                bytes[off + 18],
                bytes[off + 19],
            ]);
            let weights = u32::from_le_bytes([
                bytes[off + 20],
                bytes[off + 21],
                bytes[off + 22],
                bytes[off + 23],
            ]);
            Some(ProjectionRange {
                dtype,
                offset,
                len,
                weights,
            })
        };

        Some(Self {
            version,
            d_model,
            d_ff,
            flags,
            gate: read_projection(0, 0)?,
            up: read_projection(1, 1)?,
            down: read_projection(2, 2)?,
        })
    }

    pub fn strip<'a>(bytes: &'a [u8], block_align: usize) -> Option<(Self, &'a [u8])> {
        let h = Self::probe(bytes)?;
        let prefix = if (h.flags & UTH_FLAG_PAGE_ALIGNED_PAYLOAD) != 0 && block_align > 0 {
            let pad = (block_align - (UTH2_BYTES % block_align)) % block_align;
            UTH2_BYTES + pad
        } else {
            UTH2_BYTES
        };
        if prefix > bytes.len() {
            return None;
        }
        let payload = &bytes[prefix..];
        h.validate(payload.len() as u64).ok()?;
        Some((h, payload))
    }
}

/// Parsed Unified Tensor Header. Use [`TensorHeader::probe`] to read
/// from a byte slice without committing to it, [`TensorHeader::write_to`]
/// to serialise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorHeader {
    pub version: u16,
    pub dtype: UthDtypeId,
    pub shape: [u32; UTH_MAX_RANK],
    pub shape_rank: u8,
    pub quant_scale_offset: u32,
    pub quant_scale_count: u32,
    pub amx_tile_m: u32,
    pub amx_tile_n: u32,
    pub amx_tile_k: u32,
    pub flags: u32,
}

impl TensorHeader {
    /// Build a header for a SwiGLU expert blob.
    ///
    /// The shape is recorded as `[d_ff, d_model, 3, 0]` — three
    /// `[d_ff × d_model]` (or `[d_model × d_ff]` for `down`) matrices
    /// stacked in the canonical `gate || up || down` order.
    ///
    /// `quant_scale_offset` addresses the **scale region** (not the
    /// start of weights). For `Int8`, the three f32 per-tensor scales
    /// live in the first 12 bytes of the payload (see
    /// [`WeightDtype::header_bytes`]), so the offset is `0` and the
    /// count is `3`. For the float dtypes there are no scales; for
    /// `Q4K` / `Q4_0` the scales are kept inline per-block and cannot
    /// be addressed by a single offset, so the count is reported as
    /// `0` and readers must inspect blocks directly. In both of these
    /// no-global-scales cases the offset value is meaningless and is
    /// left as `0`.
    pub fn for_swiglu_expert(dtype: WeightDtype, d_model: usize, d_ff: usize) -> Self {
        // Tile hints are advisory; pick the AMX-tile-friendly default
        // 16×16×64 BF16 hint (a no-op on non-AMX backends).
        let (quant_scale_offset, quant_scale_count) = match dtype {
            // Int8: three f32 per-tensor scales sit at the very start
            // of the payload (offset 0), followed by the weight stream
            // at offset `dtype.header_bytes()`.
            WeightDtype::Int8 => (0u32, 3u32),
            // No global scale region — `quant_scale_offset` is unused
            // when `quant_scale_count == 0`. (MXFP4 keeps its E8M0 block
            // scales inline per projection, so like Q4_0/Q4K they cannot
            // be addressed by a single offset.)
            WeightDtype::F32
            | WeightDtype::F16
            | WeightDtype::Q4K
            | WeightDtype::Q4_0
            | WeightDtype::Q8_0
            | WeightDtype::Q5K
            | WeightDtype::Q6K
            | WeightDtype::BF16
            | WeightDtype::MXFP4
            | WeightDtype::Mixed => (0, 0),
        };
        Self {
            version: UTH_VERSION,
            dtype: UthDtypeId::from_weight(dtype),
            shape: [d_ff as u32, d_model as u32, 3, 0],
            shape_rank: 3,
            quant_scale_offset,
            quant_scale_count,
            amx_tile_m: 16,
            amx_tile_n: 16,
            amx_tile_k: 64,
            flags: UTH_FLAG_PAGE_ALIGNED_PAYLOAD,
        }
    }

    /// Serialise the header into a 64-byte array.
    pub fn to_bytes(&self) -> [u8; UTH_BYTES] {
        let mut buf = [0u8; UTH_BYTES];
        buf[0..4].copy_from_slice(&UTH_MAGIC);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6] = self.dtype as u8;
        buf[7] = self.shape_rank;
        for i in 0..UTH_MAX_RANK {
            let off = 8 + i * 4;
            buf[off..off + 4].copy_from_slice(&self.shape[i].to_le_bytes());
        }
        buf[24..28].copy_from_slice(&self.quant_scale_offset.to_le_bytes());
        buf[28..32].copy_from_slice(&self.quant_scale_count.to_le_bytes());
        buf[32..36].copy_from_slice(&self.amx_tile_m.to_le_bytes());
        buf[36..40].copy_from_slice(&self.amx_tile_n.to_le_bytes());
        buf[40..44].copy_from_slice(&self.amx_tile_k.to_le_bytes());
        buf[44..48].copy_from_slice(&self.flags.to_le_bytes());
        // bytes 48..64 reserved, already zero.
        buf
    }

    /// Write the header followed by enough zero padding to push the
    /// payload start to a multiple of `block_align`.
    pub fn write_padded(&self, block_align: usize, dst: &mut Vec<u8>) {
        let start = dst.len();
        dst.extend_from_slice(&self.to_bytes());
        let after = dst.len() - start; // == UTH_BYTES
        let pad = if block_align > 0 {
            (block_align - (after % block_align)) % block_align
        } else {
            0
        };
        dst.resize(dst.len() + pad, 0);
    }

    /// Try to parse a header from the start of `bytes`. Returns `None`
    /// if the magic does not match — callers should treat such blobs
    /// as legacy (no UTH) and proceed unchanged.
    pub fn probe(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < UTH_BYTES {
            return None;
        }
        if bytes[0..4] != UTH_MAGIC {
            return None;
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version == 0 || version > UTH_VERSION {
            return None;
        }
        let dtype = UthDtypeId::from_u8(bytes[6])?;
        let shape_rank = bytes[7];
        if shape_rank == 0 || shape_rank as usize > UTH_MAX_RANK {
            return None;
        }
        let mut shape = [0u32; UTH_MAX_RANK];
        for i in 0..UTH_MAX_RANK {
            let off = 8 + i * 4;
            shape[i] =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        let quant_scale_offset = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        let quant_scale_count = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let amx_tile_m = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let amx_tile_n = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        let amx_tile_k = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        let flags = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        Some(Self {
            version,
            dtype,
            shape,
            shape_rank,
            quant_scale_offset,
            quant_scale_count,
            amx_tile_m,
            amx_tile_n,
            amx_tile_k,
            flags,
        })
    }

    /// If `bytes` starts with a valid U.T.H., return the *payload*
    /// (post-header, post-padding) slice paired with the parsed header.
    /// Otherwise return `(None, bytes)` so the caller can treat the
    /// blob as a legacy (no-UTH) expert file.
    ///
    /// `block_align` is the on-disk padding the writer used (usually
    /// 4096 — see [`UTH_FLAG_PAGE_ALIGNED_PAYLOAD`]). Pass 0 to read
    /// the payload starting immediately after the 64-byte header.
    pub fn strip<'a>(bytes: &'a [u8], block_align: usize) -> (Option<TensorHeader>, &'a [u8]) {
        match Self::probe(bytes) {
            None => (None, bytes),
            Some(h) => {
                let prefix = if (h.flags & UTH_FLAG_PAGE_ALIGNED_PAYLOAD) != 0 && block_align > 0 {
                    let pad = (block_align - (UTH_BYTES % block_align)) % block_align;
                    UTH_BYTES + pad
                } else {
                    UTH_BYTES
                };
                if prefix > bytes.len() {
                    (None, bytes)
                } else {
                    (Some(h), &bytes[prefix..])
                }
            }
        }
    }
}

impl fmt::Display for TensorHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UTH1 v{} dtype={:?} shape={:?}/rank={} qoff={} qcnt={} tile={}x{}x{} flags=0x{:x}",
            self.version,
            self.dtype,
            &self.shape[..self.shape_rank as usize],
            self.shape_rank,
            self.quant_scale_offset,
            self.quant_scale_count,
            self.amx_tile_m,
            self.amx_tile_n,
            self.amx_tile_k,
            self.flags,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_swiglu_header() {
        for dtype in [
            WeightDtype::F32,
            WeightDtype::F16,
            WeightDtype::Int8,
            WeightDtype::Q4K,
            WeightDtype::Q4_0,
            WeightDtype::Q8_0,
            WeightDtype::Q5K,
            WeightDtype::Q6K,
            WeightDtype::Mixed,
        ] {
            let h = TensorHeader::for_swiglu_expert(dtype, 512, 2048);
            let bytes = h.to_bytes();
            assert_eq!(bytes.len(), UTH_BYTES);
            assert_eq!(&bytes[0..4], &UTH_MAGIC);
            let parsed = TensorHeader::probe(&bytes).expect("probe round-trip");
            assert_eq!(parsed, h);
            assert_eq!(parsed.dtype.to_weight(), dtype);
        }
    }

    #[test]
    fn swiglu_header_quant_scale_metadata_per_dtype() {
        // Int8 stores three f32 per-tensor scales at the start of the
        // payload, so the header must advertise offset 0 / count 3.
        let h = TensorHeader::for_swiglu_expert(WeightDtype::Int8, 512, 2048);
        assert_eq!(h.quant_scale_offset, 0);
        assert_eq!(h.quant_scale_count, 3);

        // All other dtypes have no global scale region — count must be
        // 0 and offset is unused (left as 0 by convention).
        for dtype in [
            WeightDtype::F32,
            WeightDtype::F16,
            WeightDtype::Q4K,
            WeightDtype::Q4_0,
            WeightDtype::Q8_0,
            WeightDtype::Q5K,
            WeightDtype::Q6K,
            WeightDtype::Mixed,
        ] {
            let h = TensorHeader::for_swiglu_expert(dtype, 512, 2048);
            assert_eq!(
                (h.quant_scale_offset, h.quant_scale_count),
                (0, 0),
                "unexpected quant scale metadata for {:?}",
                dtype
            );
        }
    }

    #[test]
    fn write_padded_aligns_payload() {
        let h = TensorHeader::for_swiglu_expert(WeightDtype::F32, 8, 16);
        let mut buf = Vec::new();
        h.write_padded(4096, &mut buf);
        assert_eq!(buf.len(), 4096);
        assert!(buf[UTH_BYTES..].iter().all(|&b| b == 0));
        let parsed = TensorHeader::probe(&buf).expect("probe");
        assert_eq!(parsed, h);
    }

    #[test]
    fn strip_skips_padded_prefix() {
        let h = TensorHeader::for_swiglu_expert(WeightDtype::F32, 8, 16);
        let mut buf = Vec::new();
        h.write_padded(4096, &mut buf);
        buf.extend_from_slice(&[0xAB; 64]); // fake "weights"
        let (parsed, payload) = TensorHeader::strip(&buf, 4096);
        assert!(parsed.is_some());
        assert_eq!(payload.len(), 64);
        assert!(payload.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn strip_returns_unchanged_on_legacy_blob() {
        let buf = vec![0xCDu8; 128];
        let (parsed, payload) = TensorHeader::strip(&buf, 4096);
        assert!(parsed.is_none());
        assert_eq!(payload.as_ptr(), buf.as_ptr());
        assert_eq!(payload.len(), buf.len());
    }

    #[test]
    fn mixed_uth2_roundtrip_and_strip() {
        let h = MixedExpertHeader::new(
            64,
            128,
            ProjectionRange {
                dtype: UthDtypeId::Q4_0,
                offset: 0,
                len: 18,
                weights: 32,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q4K,
                offset: 18,
                len: 144,
                weights: 256,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q5K,
                offset: 162,
                len: 176,
                weights: 256,
            },
        );
        let bytes = h.to_bytes();
        assert_eq!(&bytes[..4], &UTH2_MAGIC);
        assert_eq!(MixedExpertHeader::probe(&bytes), Some(h));

        let mut file = Vec::new();
        h.write_padded(4096, &mut file);
        assert_eq!(file.len(), 4096);
        file.extend_from_slice(&vec![0xA5; 338]);
        file.extend_from_slice(&vec![0; 4096 - 338]);
        let (parsed, payload) = MixedExpertHeader::strip(&file, 4096).expect("strip UTH2");
        assert_eq!(parsed, h);
        assert_eq!(payload[0], 0xA5);
        assert_eq!(parsed.projection_payload_len(), Some(338));
    }

    #[test]
    fn mixed_uth2_rejects_unknown_or_malformed_ranges() {
        let h = MixedExpertHeader::new(
            64,
            128,
            ProjectionRange {
                dtype: UthDtypeId::Q4_0,
                offset: 0,
                len: 18,
                weights: 32,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q4K,
                offset: 18,
                len: 144,
                weights: 256,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q5K,
                offset: 162,
                len: 176,
                weights: 256,
            },
        );
        let mut future = h.to_bytes();
        future[4..6].copy_from_slice(&3u16.to_le_bytes());
        assert!(MixedExpertHeader::probe(&future).is_none());

        let overlapping = MixedExpertHeader::new(
            64,
            128,
            ProjectionRange {
                dtype: UthDtypeId::Q4_0,
                offset: 0,
                len: 18,
                weights: 32,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q4K,
                offset: 8,
                len: 144,
                weights: 256,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q5K,
                offset: 162,
                len: 176,
                weights: 256,
            },
        );
        assert!(overlapping.validate(4096).is_err());

        let past_end = MixedExpertHeader::new(
            64,
            128,
            ProjectionRange {
                dtype: UthDtypeId::Q4_0,
                offset: 0,
                len: 18,
                weights: 32,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q4K,
                offset: 18,
                len: 144,
                weights: 256,
            },
            ProjectionRange {
                dtype: UthDtypeId::Q5K,
                offset: 162,
                len: 176,
                weights: 256,
            },
        );
        assert!(past_end.validate(200).is_err());
    }

    #[test]
    fn rejects_truncated_or_bad_magic() {
        assert!(TensorHeader::probe(&[]).is_none());
        assert!(TensorHeader::probe(&[0u8; 32]).is_none());
        let mut bad = [0u8; UTH_BYTES];
        bad[..4].copy_from_slice(b"NOPE");
        assert!(TensorHeader::probe(&bad).is_none());
    }

    // -----------------------------------------------------------------
    // Property-based fuzz tests (gist Task 4 — Formal Verification
    // Readiness).
    //
    // We don't pull in `proptest` / `quickcheck` (extra build-time
    // dep, slows CI on the engine's already-large compile budget).
    // Instead a deterministic xorshift PRNG enumerates thousands of
    // pseudo-random byte buffers and asserts the parser:
    //   * never panics,
    //   * never reads past the end of the input (we feed lengths
    //     spanning 0 .. 4*UTH_BYTES so any out-of-bounds read would
    //     be caught by the slice machinery),
    //   * always agrees with the round-trip property when the input
    //     happens to start with valid UTH bytes.
    //
    // Run with `cargo test --release tensor_header::tests::fuzz`.
    // -----------------------------------------------------------------

    fn xorshift_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn fill_random(buf: &mut [u8], seed: u64) {
        let mut state = seed.max(1);
        for chunk in buf.chunks_mut(8) {
            let v = xorshift_next(&mut state);
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (v >> (8 * i)) as u8;
            }
        }
    }

    #[test]
    fn fuzz_probe_never_panics_on_random_input() {
        // 16 K iterations × up to 256 B input ≈ 4 MiB of work, well
        // under the test budget even in debug mode.
        for trial in 0..16384u64 {
            let len =
                ((trial.wrapping_mul(0x9E37_79B9_7F4A_7C15)) % (4 * UTH_BYTES as u64 + 1)) as usize;
            let mut buf = vec![0u8; len];
            fill_random(&mut buf, trial.wrapping_mul(0x1234_5678) ^ 0xDEAD_BEEF);
            // Property: probe must return either `None` or a fully-
            // populated `TensorHeader` whose magic round-trips. It
            // must NEVER panic, NEVER OOB-read, NEVER UB.
            match TensorHeader::probe(&buf) {
                None => {}
                Some(h) => {
                    let re = h.to_bytes();
                    assert_eq!(re.len(), UTH_BYTES);
                    assert_eq!(&re[0..4], &UTH_MAGIC);
                    // Re-probe of the canonical encoding must match.
                    let h2 = TensorHeader::probe(&re).expect("re-probe");
                    assert_eq!(h, h2);
                }
            }
        }
    }

    #[test]
    fn fuzz_probe_with_valid_magic_random_tail() {
        // Force the first 4 bytes to the UTH magic, then randomise
        // the rest. The parser may accept or reject depending on
        // dtype validity, but it must never panic.
        for trial in 0..4096u64 {
            let mut buf = vec![0u8; UTH_BYTES];
            buf[..4].copy_from_slice(&UTH_MAGIC);
            fill_random(&mut buf[4..], trial.wrapping_mul(0xC0FF_EE12_3456_789A));
            let _ = TensorHeader::probe(&buf);
        }
    }

    #[test]
    fn fuzz_strip_never_panics_on_random_inputs() {
        // `strip` is the public entry point downstream kernels call;
        // a panic here would crash the engine on a malformed
        // expert_<id>.bin.
        for trial in 0..4096u64 {
            let len =
                ((trial.wrapping_mul(0x517C_C1B7_2722_0A95)) % (3 * UTH_BYTES as u64 + 1)) as usize;
            let mut buf = vec![0u8; len];
            fill_random(&mut buf, trial.wrapping_add(0x0123_4567_89AB_CDEF));
            for block in [16usize, 64, 512, 4096] {
                let (_h, payload) = TensorHeader::strip(&buf, block);
                // The returned payload must be a sub-slice of buf;
                // its pointer + length must lie within buf.
                //
                // Skip the pointer-range assertion for empty buffers:
                // `Vec::as_ptr()` may be a dangling-but-aligned sentinel
                // for zero-length allocations, so comparing raw pointer
                // values is not meaningful in that case.
                if !buf.is_empty() {
                    let buf_range = buf.as_ptr() as usize..buf.as_ptr() as usize + buf.len();
                    let pay_start = payload.as_ptr() as usize;
                    let pay_end = pay_start + payload.len();
                    assert!(pay_start >= buf_range.start && pay_end <= buf_range.end);
                }
            }
        }
    }
}
