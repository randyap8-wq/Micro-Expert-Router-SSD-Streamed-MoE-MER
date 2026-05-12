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
//!    24      4    quant_scale_offset     bytes from start of payload
//!    28      4    quant_scale_count      number of scales (u32 each), 0 if none
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

/// Total on-disk size of the header itself (excluding any page padding
/// that may follow it).
pub const UTH_BYTES: usize = 64;

/// Current header version. Incremented on incompatible layout changes.
pub const UTH_VERSION: u16 = 1;

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
}

impl UthDtypeId {
    pub fn from_weight(d: WeightDtype) -> Self {
        match d {
            WeightDtype::F32 => UthDtypeId::F32,
            WeightDtype::F16 => UthDtypeId::F16,
            WeightDtype::Int8 => UthDtypeId::Int8,
            WeightDtype::Q4K => UthDtypeId::Q4K,
            WeightDtype::Q4_0 => UthDtypeId::Q4_0,
        }
    }

    pub fn to_weight(self) -> WeightDtype {
        match self {
            UthDtypeId::F32 => WeightDtype::F32,
            UthDtypeId::F16 => WeightDtype::F16,
            UthDtypeId::Int8 => WeightDtype::Int8,
            UthDtypeId::Q4K => WeightDtype::Q4K,
            UthDtypeId::Q4_0 => WeightDtype::Q4_0,
        }
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(UthDtypeId::F32),
            1 => Some(UthDtypeId::F16),
            2 => Some(UthDtypeId::Int8),
            3 => Some(UthDtypeId::Q4K),
            4 => Some(UthDtypeId::Q4_0),
            _ => None,
        }
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
    pub fn for_swiglu_expert(dtype: WeightDtype, d_model: usize, d_ff: usize) -> Self {
        // Tile hints are advisory; pick the AMX-tile-friendly default
        // 16×16×64 BF16 hint (a no-op on non-AMX backends).
        Self {
            version: UTH_VERSION,
            dtype: UthDtypeId::from_weight(dtype),
            shape: [d_ff as u32, d_model as u32, 3, 0],
            shape_rank: 3,
            quant_scale_offset: dtype.header_bytes() as u32,
            quant_scale_count: match dtype {
                // Int8 stores 3 f32 scales (gate, up, down) in its 12-byte header.
                WeightDtype::Int8 => 3,
                // Q4K / Q4_0 keep scales inline per-block — not addressed by a
                // single offset. Report 0 here; readers must inspect blocks.
                _ => 0,
            },
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
        if shape_rank as usize > UTH_MAX_RANK {
            return None;
        }
        let mut shape = [0u32; UTH_MAX_RANK];
        for i in 0..UTH_MAX_RANK {
            let off = 8 + i * 4;
            shape[i] = u32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]);
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

    /// Returns true if `bytes` starts with the U.T.H. magic.
    pub fn has_magic(bytes: &[u8]) -> bool {
        bytes.len() >= 4 && bytes[0..4] == UTH_MAGIC
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
    fn rejects_truncated_or_bad_magic() {
        assert!(TensorHeader::probe(&[]).is_none());
        assert!(TensorHeader::probe(&[0u8; 32]).is_none());
        let mut bad = [0u8; UTH_BYTES];
        bad[..4].copy_from_slice(b"NOPE");
        assert!(TensorHeader::probe(&bad).is_none());
    }
}
