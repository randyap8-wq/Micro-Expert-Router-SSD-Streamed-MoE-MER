//! Minimal GGUF reader (gist Phase 2).
//!
//! GGUF is the on-disk container used by `llama.cpp` / Ollama / public
//! Mixtral quantised downloads. The spec is at
//! <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>.
//!
//! This module implements only the subset the engine needs to migrate a
//! Mixtral checkpoint into the engine's per-expert binary format:
//!
//! * Magic / version validation (`GGUF`, versions 1, 2, 3).
//! * Metadata key/value table (strings, primitive scalars, arrays).
//! * Tensor info table (name, shape, dtype, offset).
//! * On-demand tensor data access — returns a `&[u8]` slice that points
//!   into a buffer the parser owns. No `mmap` dependency: the whole file
//!   is read once into memory at `open()` time, which is simpler, has no
//!   `unsafe`, and is fine for the offline `gguf-convert` subcommand
//!   (the engine never opens GGUFs on the inference hot path).
//!
//! The dtype map covers the five GGUF types the engine cares about:
//! `F32(0)`, `F16(1)`, `Q4_0(2)`, `Q4_K(12)`, `Q6_K(14)`. Of those,
//! `F32`/`F16`/`Q4_0`/`Q4_K` map onto the engine's [`WeightDtype`]; `Q6_K`
//! is recognised so [`GgufFile::open`] doesn't fail on a Q6_K-quantised
//! source, but tensors of that dtype are surfaced as `None` from
//! [`GgufFile::tensor_dtype`] (and the loader falls back to seeded init
//! for those tensors).

use crate::inference::{
    WeightDtype, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// The 4-byte ASCII magic at the start of every GGUF file: `"GGUF"`.
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF metadata value type tags, from the spec.
#[allow(non_camel_case_types, dead_code)]
#[repr(u32)]
enum GgufType {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
}

/// GGML tensor dtype codes used in the GGUF tensor-info table.
///
/// Only the ones the engine cares about are named — everything else is
/// represented as the raw `u32` code in [`GgufTensorInfo::ggml_dtype`].
#[allow(non_camel_case_types, dead_code)]
pub mod ggml_dtype {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q8_1: u32 = 9;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const Q8_K: u32 = 15;
    /// GGML BF16 dtype code (added in later llama.cpp releases).
    pub const BF16: u32 = 32;
}

/// A metadata value. Arrays are flattened into typed `Vec`s for the
/// element types the loader actually consults; mixed-type arrays (rare
/// in real GGUF files) are stored as raw bytes.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    /// Generic array. The inner `Vec` stores one boxed value per element.
    /// Cheap to construct, cheap to iterate, exact-typed enough for the
    /// extractor's needs (`as_u64()` / `as_string()` helpers below).
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Best-effort cast to `u64`. Used for hyperparameter metadata
    /// (`llama.block_count`, etc.) which is written as `U32` or `U64`
    /// depending on the producer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) if v >= 0 => Some(v as u64),
            GgufValue::I16(v) if v >= 0 => Some(v as u64),
            GgufValue::I32(v) if v >= 0 => Some(v as u64),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Best-effort cast to `&str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Description of a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in GGML order (innermost-first). Common shapes:
    /// `[d_model]` (1-D), `[d_model, vocab]` (2-D), `[d_model, d_ff,
    /// num_experts]` (3-D Mixtral expert tensor).
    pub shape: Vec<u64>,
    /// Raw GGML dtype code (see [`ggml_dtype`]).
    pub ggml_dtype: u32,
    /// Byte offset from the start of the tensor-data section.
    pub offset: u64,
    /// Computed byte length of this tensor's data.
    pub byte_len: u64,
}

impl GgufTensorInfo {
    /// Total element count = product of shape dimensions.
    pub fn elem_count(&self) -> u64 {
        self.shape.iter().copied().product()
    }
}

/// Parsed GGUF file. Owns the raw file bytes; tensor data is returned
/// as `&[u8]` slices into that buffer.
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// File bytes. Tensor `offset`s are relative to `tensor_data_start`.
    bytes: Vec<u8>,
    tensor_data_start: usize,
}

impl GgufFile {
    /// Open and fully parse a GGUF file. Reads the file into memory.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        Self::parse(bytes)
    }

    /// Borrow the tensor's raw bytes. `None` if the tensor or its
    /// declared byte range is invalid.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.tensor_data_start.checked_add(info.offset as usize)?;
        let end = start.checked_add(info.byte_len as usize)?;
        if end > self.bytes.len() {
            return None;
        }
        Some(&self.bytes[start..end])
    }

    /// Map this tensor's GGML dtype onto the engine's [`WeightDtype`].
    /// `None` for dtypes the engine doesn't currently support (e.g.
    /// Q6_K, Q5_*, Q8_*, the *_K variants other than Q4_K).
    pub fn tensor_dtype(&self, name: &str) -> Option<WeightDtype> {
        let info = self.tensors.get(name)?;
        ggml_to_weight_dtype(info.ggml_dtype)
    }

    /// Architecture string from `general.architecture` (e.g.
    /// `"llama"`, `"mixtral"`), if present.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    /// Whether the named tensor exists in this GGUF.
    #[allow(dead_code)]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    // -------------------------- parser -----------------------------

    fn parse(bytes: Vec<u8>) -> io::Result<Self> {
        let mut cur = Cursor::new(&bytes);
        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not a GGUF file (magic = {magic:?})"),
            ));
        }
        let version = cur.read_u32()?;
        if version == 0 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version {version}; supported = 1, 2, 3"),
            ));
        }

        // Counts: v1 uses u32, v2+ uses u64. The reader transparently
        // promotes so all downstream code sees u64.
        let tensor_count = read_count(&mut cur, version)?;
        let kv_count = read_count(&mut cur, version)?;

        let mut metadata: HashMap<String, GgufValue> = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_string(&mut cur, version)?;
            let val = read_value(&mut cur, version)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let alignment = alignment.max(1);

        let mut tensors: HashMap<String, GgufTensorInfo> =
            HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut cur, version)?;
            let n_dims = cur.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let d = if version == 1 {
                    cur.read_u32()? as u64
                } else {
                    cur.read_u64()?
                };
                shape.push(d);
            }
            let ggml_dtype = cur.read_u32()?;
            let offset = cur.read_u64()?;
            let byte_len = ggml_tensor_bytes(ggml_dtype, &shape).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("cannot compute byte length for tensor {name} (dtype {ggml_dtype})"),
                )
            })?;
            tensors.insert(
                name.clone(),
                GgufTensorInfo { name, shape, ggml_dtype, offset, byte_len },
            );
        }

        // Tensor data is padded to `alignment` from the start of the file.
        let after_header = cur.pos();
        let pad = (alignment - (after_header % alignment)) % alignment;
        let tensor_data_start = after_header + pad;
        if tensor_data_start > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "GGUF tensor-data section starts past end of file",
            ));
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            bytes,
            tensor_data_start,
        })
    }
}

/// Map a GGML dtype code to the engine's [`WeightDtype`].
pub fn ggml_to_weight_dtype(code: u32) -> Option<WeightDtype> {
    match code {
        ggml_dtype::F32 => Some(WeightDtype::F32),
        ggml_dtype::F16 => Some(WeightDtype::F16),
        ggml_dtype::BF16 => Some(WeightDtype::BF16),
        ggml_dtype::Q4_0 => Some(WeightDtype::Q4_0),
        ggml_dtype::Q4_K => Some(WeightDtype::Q4K),
        ggml_dtype::Q8_0 => Some(WeightDtype::Q8_0),
        _ => None,
    }
}

/// Byte length of a GGML tensor with the given dtype and shape, or
/// `None` if the dtype is not understood. Quant dtypes round up to a
/// whole block (this matches the layout `ggml-quants.c` writes).
fn ggml_tensor_bytes(code: u32, shape: &[u64]) -> Option<u64> {
    let elems: u64 = shape.iter().copied().product();
    let bytes = match code {
        ggml_dtype::F32 => elems.checked_mul(4)?,
        ggml_dtype::F16 => elems.checked_mul(2)?,
        ggml_dtype::BF16 => elems.checked_mul(2)?,
        ggml_dtype::Q4_0 => {
            // 32-element blocks of 18 bytes each.
            let blocks = elems.div_ceil(Q4_0_BLOCK_ELEMS as u64);
            blocks.checked_mul(Q4_0_BLOCK_BYTES as u64)?
        }
        ggml_dtype::Q4_K => {
            // 256-element super-blocks of 144 bytes each.
            let blocks = elems.div_ceil(Q4K_BLOCK_ELEMS as u64);
            blocks.checked_mul(Q4K_BLOCK_BYTES as u64)?
        }
        ggml_dtype::Q8_0 => {
            // 32-element blocks of 34 bytes each (1 f16 scale + 32 i8).
            let blocks = elems.div_ceil(32);
            blocks.checked_mul(34)?
        }
        ggml_dtype::Q6_K => {
            // 256-element super-blocks of 210 bytes each.
            let blocks = elems.div_ceil(256);
            blocks.checked_mul(210)?
        }
        _ => return None,
    };
    Some(bytes)
}

// ------------------------ value / cursor helpers --------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn advance(&mut self, n: usize) -> io::Result<()> {
        if self.pos.checked_add(n).map_or(true, |p| p > self.buf.len()) {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "GGUF: unexpected end of buffer",
            ));
        }
        self.pos += n;
        Ok(())
    }
    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let start = self.pos;
        self.advance(n)?;
        Ok(&self.buf[start..start + n])
    }
    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_u16(&mut self) -> io::Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

fn read_count<'a>(cur: &mut Cursor<'a>, version: u32) -> io::Result<u64> {
    if version == 1 {
        Ok(cur.read_u32()? as u64)
    } else {
        cur.read_u64()
    }
}

fn read_string<'a>(cur: &mut Cursor<'a>, version: u32) -> io::Result<String> {
    let len = read_count(cur, version)? as usize;
    let bytes = cur.read_bytes(len)?;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn read_value<'a>(cur: &mut Cursor<'a>, version: u32) -> io::Result<GgufValue> {
    let ty = cur.read_u32()?;
    read_value_typed(cur, version, ty)
}

fn read_value_typed<'a>(cur: &mut Cursor<'a>, version: u32, ty: u32) -> io::Result<GgufValue> {
    Ok(match ty {
        x if x == GgufType::UINT8 as u32 => GgufValue::U8(cur.read_u8()?),
        x if x == GgufType::INT8 as u32 => GgufValue::I8(cur.read_u8()? as i8),
        x if x == GgufType::UINT16 as u32 => GgufValue::U16(cur.read_u16()?),
        x if x == GgufType::INT16 as u32 => GgufValue::I16(cur.read_u16()? as i16),
        x if x == GgufType::UINT32 as u32 => GgufValue::U32(cur.read_u32()?),
        x if x == GgufType::INT32 as u32 => GgufValue::I32(cur.read_u32()? as i32),
        x if x == GgufType::UINT64 as u32 => GgufValue::U64(cur.read_u64()?),
        x if x == GgufType::INT64 as u32 => GgufValue::I64(cur.read_u64()? as i64),
        x if x == GgufType::FLOAT32 as u32 => {
            GgufValue::F32(f32::from_bits(cur.read_u32()?))
        }
        x if x == GgufType::FLOAT64 as u32 => {
            GgufValue::F64(f64::from_bits(cur.read_u64()?))
        }
        x if x == GgufType::BOOL as u32 => GgufValue::Bool(cur.read_u8()? != 0),
        x if x == GgufType::STRING as u32 => GgufValue::String(read_string(cur, version)?),
        x if x == GgufType::ARRAY as u32 => {
            let inner_ty = cur.read_u32()?;
            let n = read_count(cur, version)? as usize;
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(read_value_typed(cur, version, inner_ty)?);
            }
            GgufValue::Array(out)
        }
        other => {
            // Unknown well-typed value. Without knowing the width we
            // can't safely skip and continue, so surface as an error.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GGUF: unknown value type code {other}"),
            ));
        }
    })
}

// -----------------------------------------------------------------------
// Streaming reader.
// -----------------------------------------------------------------------

/// A GGUF reader that only parses the **header** (magic + KV table + tensor
/// info table) into memory and reads tensor bodies on demand by seeking
/// the still-open `File`. The eager [`GgufFile::open`] entry point is
/// retained for tests and for very small fixtures where slurping the
/// whole file is genuinely simpler; everything that operates on
/// production-scale Mixtral checkpoints (10s of GB to >100 GB) should
/// use this type instead — see [`crate::gguf_loader`].
///
/// The implementation is deliberately straightforward: it does a single
/// streaming read of the header (which is bounded — typically a few
/// hundred KB even on a 100 GB GGUF, since tensor *bodies* dominate),
/// then satisfies each `read_tensor_data` call with one
/// `seek` + `read_exact` on the original `File`. There is no `mmap`
/// dependency and no `unsafe`. Concurrent reads are guarded by a
/// `Mutex<File>` so the reader is `Send + Sync`.
pub struct GgufStreamReader {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// File handle, kept open for tensor-data seeks.
    file: parking_lot::Mutex<File>,
    /// Byte offset (from the start of the file) of the tensor-data
    /// region. Tensor `offset` fields are relative to this.
    tensor_data_start: u64,
    /// Total file size in bytes (cached at open).
    file_len: u64,
}

impl GgufStreamReader {
    /// Open `path` and parse only the header. The tensor data is **not**
    /// read into memory; call [`Self::read_tensor_data`] for each
    /// tensor you actually need.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let file_len = f.metadata()?.len();

        // Step 1: read the fixed header preamble (magic + version + counts).
        let mut hdr = [0u8; 24];
        f.read_exact(&mut hdr[..4])?;
        if &hdr[..4] != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not a GGUF file (magic = {:?})", &hdr[..4]),
            ));
        }
        f.read_exact(&mut hdr[4..8])?;
        let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if version == 0 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version {version}; supported = 1, 2, 3"),
            ));
        }
        let (tensor_count_bytes, kv_count_bytes) = if version == 1 { (4, 4) } else { (8, 8) };
        let mut count_buf = [0u8; 16];
        f.read_exact(&mut count_buf[..tensor_count_bytes + kv_count_bytes])?;
        let tensor_count = read_count_bytes(&count_buf[..tensor_count_bytes], version);
        let kv_count = read_count_bytes(
            &count_buf[tensor_count_bytes..tensor_count_bytes + kv_count_bytes],
            version,
        );

        // Step 2: read the rest of the header into memory. The header
        // size is bounded by `tensor_count * (string + shape + 12 bytes)`
        // plus the KV table — both are tiny relative to the body.
        // We do this by streaming the file from its current position
        // until we have enough to parse the tensor table; because we
        // don't know the exact size ahead of time we read in 4 MiB
        // chunks until parsing succeeds, then stop. In practice one
        // chunk is enough for every published Mixtral checkpoint.
        let header_preamble_len = (4 + 4 + tensor_count_bytes + kv_count_bytes) as u64;
        let mut header_bytes = Vec::with_capacity(4 * 1024 * 1024);
        header_bytes.extend_from_slice(&hdr[..8]);
        header_bytes.extend_from_slice(&count_buf[..tensor_count_bytes + kv_count_bytes]);
        let mut chunk = vec![0u8; 4 * 1024 * 1024];
        loop {
            let n = f.read(&mut chunk)?;
            if n == 0 {
                // Reached EOF without finding the tensor-data section —
                // surfaces below as an `UnexpectedEof` error.
                break;
            }
            header_bytes.extend_from_slice(&chunk[..n]);
            // Try to parse the header from the bytes accumulated so far.
            // On success we'll know `tensor_data_start` and can stop
            // pulling header bytes.
            if let Some(parsed) = Self::try_parse_header(
                &header_bytes,
                version,
                tensor_count,
                kv_count,
                header_preamble_len,
            ) {
                return Ok(Self {
                    version: parsed.version,
                    metadata: parsed.metadata,
                    tensors: parsed.tensors,
                    file: parking_lot::Mutex::new(f),
                    tensor_data_start: parsed.tensor_data_start,
                    file_len,
                });
            }
            // Bail if the header is patently malformed — refuse to grow
            // the buffer unboundedly on a corrupt file. Real GGUF
            // headers are at most a few hundred MiB even at extreme
            // tensor counts.
            if header_bytes.len() > 10 * 1024 * 1024 * 1024 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF header exceeds 10 GiB; refusing to grow further",
                ));
            }
        }
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "GGUF file ended before the tensor-data section was reachable",
        ))
    }

    /// Read the raw bytes of a tensor by name into an owned `Vec<u8>`.
    /// Returns `None` if the tensor is not present in this file.
    pub fn read_tensor_data(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
        let info = match self.tensors.get(name) {
            Some(i) => i.clone(),
            None => return Ok(None),
        };
        let start = self
            .tensor_data_start
            .checked_add(info.offset)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor offset overflow"))?;
        let end = start
            .checked_add(info.byte_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor end overflow"))?;
        if end > self.file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} declares range {start}..{end} past end of file ({})",
                    self.file_len
                ),
            ));
        }
        let byte_len: usize = info.byte_len.try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} byte length {} is too large for this platform",
                    info.byte_len
                ),
            )
        })?;
        let mut out = vec![0u8; byte_len];
        let mut f = self.file.lock();
        use std::io::{Seek, SeekFrom};
        f.seek(SeekFrom::Start(start))?;
        f.read_exact(&mut out)?;
        Ok(Some(out))
    }

    /// Look up tensor info (shape, dtype, offset) by name without
    /// reading any body bytes.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// Map this tensor's GGML dtype onto the engine's [`WeightDtype`].
    pub fn tensor_dtype(&self, name: &str) -> Option<WeightDtype> {
        let info = self.tensors.get(name)?;
        ggml_to_weight_dtype(info.ggml_dtype)
    }

    /// `general.architecture`, if present.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    /// Try to parse the full header out of `bytes`. Returns `None` if
    /// `bytes` is too short to contain the full header yet — the caller
    /// should pull more bytes and retry.
    fn try_parse_header(
        bytes: &[u8],
        _version: u32,
        _tensor_count: u64,
        _kv_count: u64,
        _preamble_len: u64,
    ) -> Option<ParsedGgufHeader> {
        // We can't easily know "is this enough?" without partially
        // parsing — so reuse the existing eager parser. If it succeeds
        // we extract the header fields and discard the body bytes the
        // caller accidentally pulled in (they were a small overshoot
        // bounded by the 4 MiB read chunk).
        //
        // The eager parser takes ownership of `Vec<u8>`. Clone the
        // accumulated header buffer here — the cost is O(header size),
        // which is tiny relative to a real GGUF body. Doing this once
        // when we converge on a parseable header is fine.
        let parsed = match GgufFile::parse(bytes.to_vec()) {
            Ok(p) => p,
            Err(_) => return None,
        };
        Some(ParsedGgufHeader {
            version: parsed.version,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            tensor_data_start: parsed.tensor_data_start as u64,
        })
    }
}

struct ParsedGgufHeader {
    version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    tensor_data_start: u64,
}

fn read_count_bytes(bytes: &[u8], version: u32) -> u64 {
    if version == 1 {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64
    } else {
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

// -----------------------------------------------------------------------
// `GgufSource` — uniform abstraction over the eager and streaming readers.
// -----------------------------------------------------------------------

/// Read-only view over a GGUF file. The `gguf_loader` module is
/// parametrised over this trait so it can drive either the eager
/// in-memory [`GgufFile`] (used by the tests and small fixtures) or
/// the on-disk streaming [`GgufStreamReader`] (used by `gguf-convert`
/// at production scale).
pub trait GgufSource {
    fn metadata(&self) -> &HashMap<String, GgufValue>;
    fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo>;
    /// Read a tensor's body into an owned `Vec<u8>`. Returns `Ok(None)`
    /// when the tensor is not present (matches the legacy
    /// `GgufFile::tensor_data` behaviour for the missing case).
    fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>>;
    /// Map this tensor's GGML dtype onto the engine's [`WeightDtype`].
    fn tensor_dtype(&self, name: &str) -> Option<WeightDtype> {
        let info = self.tensor_info(name)?;
        ggml_to_weight_dtype(info.ggml_dtype)
    }
    fn architecture(&self) -> Option<&str> {
        self.metadata().get("general.architecture").and_then(|v| v.as_str())
    }
    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_info(name).is_some()
    }
}

impl GgufSource for GgufFile {
    fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }
    fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }
    fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
        Ok(self.tensor_data(name).map(<[u8]>::to_vec))
    }
}

impl GgufSource for GgufStreamReader {
    fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }
    fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }
    fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
        self.read_tensor_data(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny in-memory GGUF v3 file with one F32 metadata value
    /// and one 4-element F32 tensor named `"t"`, then parse it back.
    fn synth_gguf() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        out.extend_from_slice(&2u64.to_le_bytes()); // kv_count

        // kv 1: general.alignment = 32 (u32)
        let key = b"general.alignment";
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&(GgufType::UINT32 as u32).to_le_bytes());
        out.extend_from_slice(&32u32.to_le_bytes());

        // kv 2: general.architecture = "llama"
        let key = b"general.architecture";
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&(GgufType::STRING as u32).to_le_bytes());
        let s = b"llama";
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s);

        // tensor info: name="t", 1D shape [4], dtype F32, offset 0
        let tname = b"t";
        out.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        out.extend_from_slice(tname);
        out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        out.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
        out.extend_from_slice(&0u32.to_le_bytes()); // dtype = F32
        out.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // pad to alignment
        while out.len() % 32 != 0 {
            out.push(0);
        }
        // tensor data: 4 f32s = [1.0, 2.0, 3.0, 4.0]
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    #[test]
    fn parse_synthetic_gguf_round_trips_tensor_data() {
        let bytes = synth_gguf();
        let gguf = GgufFile::parse(bytes).expect("parse");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(gguf.tensors.len(), 1);
        let info = gguf.tensors.get("t").unwrap();
        assert_eq!(info.shape, vec![4]);
        assert_eq!(info.ggml_dtype, ggml_dtype::F32);
        assert_eq!(info.byte_len, 16);
        let data = gguf.tensor_data("t").unwrap();
        assert_eq!(data.len(), 16);
        let v: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gguf.tensor_dtype("t"), Some(WeightDtype::F32));
    }

    #[test]
    fn rejects_non_gguf_magic() {
        let bytes = b"NOPE\0\0\0\0".to_vec();
        let err = GgufFile::parse(bytes).err().expect("expected parse failure");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn ggml_dtype_mapping_covers_supported_codes() {
        assert_eq!(ggml_to_weight_dtype(ggml_dtype::F32), Some(WeightDtype::F32));
        assert_eq!(ggml_to_weight_dtype(ggml_dtype::F16), Some(WeightDtype::F16));
        assert_eq!(ggml_to_weight_dtype(ggml_dtype::Q4_0), Some(WeightDtype::Q4_0));
        assert_eq!(ggml_to_weight_dtype(ggml_dtype::Q4_K), Some(WeightDtype::Q4K));
        assert_eq!(ggml_to_weight_dtype(ggml_dtype::Q8_0), Some(WeightDtype::Q8_0));
        // Unsupported: Q6_K, Q5_K, etc. — surface as None so the loader
        // can skip / fall back to seeded init.
        assert_eq!(ggml_to_weight_dtype(ggml_dtype::Q6_K), None);
    }

    #[test]
    fn stream_reader_parses_header_and_reads_body_on_demand() {
        // Write the synthetic GGUF to a tempfile and open it
        // through `GgufStreamReader`. The streaming path must
        // surface the same metadata, tensor info, and body bytes
        // as the eager `GgufFile::parse` path.
        let bytes = synth_gguf();
        let dir = std::env::temp_dir().join(format!(
            "gguf-stream-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let r = GgufStreamReader::open(&path).expect("stream open");
        assert_eq!(r.version, 3);
        assert_eq!(r.architecture(), Some("llama"));
        let info = r.tensor_info("t").expect("tensor info");
        assert_eq!(info.shape, vec![4]);
        assert_eq!(info.byte_len, 16);
        assert_eq!(r.tensor_dtype("t"), Some(WeightDtype::F32));

        let body = r.read_tensor_data("t").unwrap().expect("body present");
        assert_eq!(body.len(), 16);
        let v: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);

        // Missing tensor → Ok(None), not an error.
        assert!(r.read_tensor_data("missing").unwrap().is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // -----------------------------------------------------------------
    // Property-based fuzz tests (gist Task 4 — Formal Verification
    // Readiness).
    //
    // The GGUF parser is exposed to arbitrary byte streams in
    // production (`gguf-convert` consumes whatever the user points at).
    // These tests verify the eager parser never panics, OOB-reads, or
    // hangs on a malformed file, irrespective of how the bytes were
    // produced.
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
    fn fuzz_parse_never_panics_on_random_bytes() {
        // 4096 random inputs × up to 4 KiB; the parser must always
        // return Err cleanly, never panic / unwrap / OOB-read.
        for trial in 0..4096u64 {
            let len = (trial.wrapping_mul(0xA3C5_9B27) % 4096) as usize;
            let mut buf = vec![0u8; len];
            fill_random(&mut buf, trial.wrapping_mul(0xDEAD_BEEF));
            // The eager parser is the public consumer of arbitrary
            // byte streams; the streaming reader needs a file path,
            // which the fuzzer can't conjure into existence here.
            let _ = GgufFile::parse(buf);
        }
    }

    #[test]
    fn fuzz_parse_with_valid_magic_random_tail() {
        // Stamp the first 4 bytes with the GGUF magic and randomise
        // the rest. The parser must still reject every invalid
        // metadata block without panicking.
        for trial in 0..2048u64 {
            let len = 8 + (trial.wrapping_mul(0x9E37_79B9) % 1024) as usize;
            let mut buf = vec![0u8; len];
            buf[..4].copy_from_slice(GGUF_MAGIC);
            fill_random(&mut buf[4..], trial.wrapping_add(0xC0FF_EE12));
            let _ = GgufFile::parse(buf);
        }
    }
}
