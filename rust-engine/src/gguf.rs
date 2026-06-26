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
use std::path::{Path, PathBuf};

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
            )? {
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

    /// Try to parse the full header out of `bytes`. Returns `Ok(None)`
    /// only if `bytes` is too short to contain the full header yet — the
    /// caller should pull more bytes and retry.
    fn try_parse_header(
        bytes: &[u8],
        _version: u32,
        _tensor_count: u64,
        _kv_count: u64,
        _preamble_len: u64,
    ) -> io::Result<Option<ParsedGgufHeader>> {
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
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(Some(ParsedGgufHeader {
            version: parsed.version,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            tensor_data_start: parsed.tensor_data_start as u64,
        }))
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
// Native GGUF shard sets.
// -----------------------------------------------------------------------

/// Read-only view over a standard tensor-level GGUF shard set.
///
/// Tensor offsets remain local to the owning shard. The unified tensor
/// table only records which shard owns each tensor name so body reads can
/// dispatch back to that shard's streaming reader without merging files.
pub struct GgufShardSet {
    readers: Vec<GgufStreamReader>,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    owners: HashMap<String, usize>,
}

impl GgufShardSet {
    /// Open a standard `*-00001-of-00005.gguf` style shard set from any
    /// shard path in the set.
    pub fn open(path: &Path) -> io::Result<Self> {
        let spec = parse_shard_filename(path)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} does not look like a standard GGUF shard name (*-00001-of-00005.gguf)",
                    path.display()
                ),
            )
        })?;
        Self::open_from_spec(spec)
    }

    fn open_from_spec(spec: GgufShardSpec) -> io::Result<Self> {
        let mut readers = Vec::with_capacity(spec.count);
        for shard_number in 1..=spec.count {
            let shard_path = spec.path_for(shard_number);
            if !shard_path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "missing GGUF shard {shard_number} of {}: {}",
                        spec.count,
                        shard_path.display()
                    ),
                ));
            }
            let reader = GgufStreamReader::open(&shard_path).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!(
                        "failed to open GGUF shard {shard_number} of {} ({}): {e}",
                        spec.count,
                        shard_path.display()
                    ),
                )
            })?;
            readers.push(reader);
        }

        validate_shards(&readers, spec.count)?;

        let metadata = readers[0].metadata.clone();
        let mut tensors = HashMap::new();
        let mut owners = HashMap::new();
        for (owner, reader) in readers.iter().enumerate() {
            for (name, info) in &reader.tensors {
                if let Some(previous_owner) = owners.get(name) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "duplicate tensor name `{name}` in GGUF shard {} and shard {}",
                            *previous_owner + 1,
                            owner + 1
                        ),
                    ));
                }
                tensors.insert(name.clone(), info.clone());
                owners.insert(name.clone(), owner);
            }
        }

        if let Some(declared) = declared_split_tensor_count(&readers)? {
            let actual = tensors.len() as u64;
            if declared != actual {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GGUF split.tensors.count={declared} disagrees with unified tensor count {actual}"
                    ),
                ));
            }
        }

        Ok(Self {
            readers,
            metadata,
            tensors,
            owners,
        })
    }
}

/// Open either a normal single GGUF file or a standard tensor-level GGUF
/// shard set. Discovery is based on the filename convention; split
/// metadata, when present, is used as validation.
pub fn open_gguf_source(path: &Path, legacy_eager: bool) -> io::Result<Box<dyn GgufSource>> {
    if let Some(spec) = parse_shard_filename(path)? {
        return Ok(Box::new(GgufShardSet::open_from_spec(spec)?));
    }

    if legacy_eager {
        Ok(Box::new(GgufFile::open(path)?))
    } else {
        Ok(Box::new(GgufStreamReader::open(path)?))
    }
}

#[derive(Debug, Clone)]
struct GgufShardSpec {
    parent: PathBuf,
    prefix: String,
    count: usize,
    index_width: usize,
    count_width: usize,
}

impl GgufShardSpec {
    fn path_for(&self, shard_number: usize) -> PathBuf {
        let file_name = format!(
            "{}-{shard_number:0index_width$}-of-{count:0count_width$}.gguf",
            self.prefix,
            count = self.count,
            index_width = self.index_width,
            count_width = self.count_width
        );
        self.parent.join(file_name)
    }
}

fn parse_shard_filename(path: &Path) -> io::Result<Option<GgufShardSpec>> {
    let Some(file_name) = path.file_name().and_then(|f| f.to_str()) else {
        return Ok(None);
    };
    let Some(stem) = file_name.strip_suffix(".gguf") else {
        return Ok(None);
    };
    let Some((before_count, count_str)) = stem.rsplit_once("-of-") else {
        return Ok(None);
    };
    let Some((prefix, index_str)) = before_count.rsplit_once('-') else {
        return Ok(None);
    };
    if prefix.is_empty()
        || index_str.is_empty()
        || count_str.is_empty()
        || !index_str.bytes().all(|b| b.is_ascii_digit())
        || !count_str.bytes().all(|b| b.is_ascii_digit())
    {
        return Ok(None);
    }

    let index = index_str.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("GGUF shard index `{index_str}` in {file_name} is too large"),
        )
    })?;
    let count = count_str.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("GGUF shard count `{count_str}` in {file_name} is too large"),
        )
    })?;
    if count == 0 || index == 0 || index > count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid GGUF shard numbering in {file_name}: shard {index} of {count}"),
        ));
    }

    Ok(Some(GgufShardSpec {
        parent: path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        prefix: prefix.to_owned(),
        count,
        index_width: index_str.len(),
        count_width: count_str.len(),
    }))
}

fn validate_shards(readers: &[GgufStreamReader], expected_count: usize) -> io::Result<()> {
    if readers.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF shard set is empty",
        ));
    }

    let expected_version = readers[0].version;
    let expected_arch =
        metadata_str(&readers[0].metadata, "general.architecture", 1)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF shard 1 missing required metadata `general.architecture`",
            )
        })?;

    for (idx, reader) in readers.iter().enumerate() {
        let shard_number = idx + 1;
        if reader.version != expected_version {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF shard {shard_number} version {} disagrees with shard 1 version {expected_version}",
                    reader.version
                ),
            ));
        }

        let arch = metadata_str(&reader.metadata, "general.architecture", shard_number)?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GGUF shard {shard_number} missing required metadata `general.architecture`"
                    ),
                )
            })?;
        if arch != expected_arch {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF shard {shard_number} general.architecture `{arch}` disagrees with shard 1 `{expected_arch}`"
                ),
            ));
        }

        if let Some(split_no) = metadata_u64(&reader.metadata, "split.no", shard_number)? {
            let expected_split_no = idx as u64;
            if split_no != expected_split_no {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GGUF shard {shard_number} split.no={split_no} disagrees with filename order; expected {expected_split_no}"
                    ),
                ));
            }
        }

        if let Some(split_count) = metadata_u64(&reader.metadata, "split.count", shard_number)? {
            if split_count != expected_count as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GGUF shard {shard_number} split.count={split_count} disagrees with filename count {expected_count}"
                    ),
                ));
            }
        }
    }

    Ok(())
}

fn declared_split_tensor_count(readers: &[GgufStreamReader]) -> io::Result<Option<u64>> {
    let mut declared = None;
    for (idx, reader) in readers.iter().enumerate() {
        let shard_number = idx + 1;
        if let Some(count) = metadata_u64(&reader.metadata, "split.tensors.count", shard_number)? {
            if let Some(previous) = declared {
                if previous != count {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "GGUF shard {shard_number} split.tensors.count={count} disagrees with earlier declaration {previous}"
                        ),
                    ));
                }
            } else {
                declared = Some(count);
            }
        }
    }
    Ok(declared)
}

fn metadata_u64(
    metadata: &HashMap<String, GgufValue>,
    key: &str,
    shard_number: usize,
) -> io::Result<Option<u64>> {
    metadata
        .get(key)
        .map(|v| {
            v.as_u64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("GGUF shard {shard_number} metadata `{key}` is not an integer"),
                )
            })
        })
        .transpose()
}

fn metadata_str<'a>(
    metadata: &'a HashMap<String, GgufValue>,
    key: &str,
    shard_number: usize,
) -> io::Result<Option<&'a str>> {
    metadata
        .get(key)
        .map(|v| {
            v.as_str().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("GGUF shard {shard_number} metadata `{key}` is not a string"),
                )
            })
        })
        .transpose()
}

// -----------------------------------------------------------------------
// `GgufSource` — uniform abstraction over the eager and streaming readers.
// -----------------------------------------------------------------------

/// Read-only view over GGUF tensors. The `gguf_loader` module is
/// parametrised over this trait so it can drive either the eager
/// in-memory [`GgufFile`] (used by the tests and small fixtures) or
/// the on-disk streaming [`GgufStreamReader`] (used by `gguf-convert`
/// at production scale) or a unified [`GgufShardSet`].
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

impl GgufSource for GgufShardSet {
    fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }
    fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }
    fn read_tensor_owned(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
        let Some(owner) = self.owners.get(name).copied() else {
            return Ok(None);
        };
        self.readers[owner].read_tensor_data(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny in-memory GGUF v3 file with one F32 metadata value
    /// and one 4-element F32 tensor named `"t"`, then parse it back.
    fn synth_gguf() -> Vec<u8> {
        synth_gguf_with_dtype(ggml_dtype::F32)
    }

    fn synth_gguf_with_dtype(dtype: u32) -> Vec<u8> {
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
        out.extend_from_slice(&dtype.to_le_bytes());
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

    struct SynthTensor {
        name: String,
        shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
    }

    fn synth_f32_tensor(name: &str, values: &[f32]) -> SynthTensor {
        SynthTensor {
            name: name.to_owned(),
            shape: vec![values.len() as u64],
            dtype: ggml_dtype::F32,
            data: f32_bytes(values),
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn push_gguf_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_gguf_value(out: &mut Vec<u8>, value: &GgufValue) {
        match value {
            GgufValue::U32(v) => {
                out.extend_from_slice(&(GgufType::UINT32 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::U64(v) => {
                out.extend_from_slice(&(GgufType::UINT64 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::String(s) => {
                out.extend_from_slice(&(GgufType::STRING as u32).to_le_bytes());
                push_gguf_string(out, s);
            }
            other => panic!("test helper cannot encode metadata value {other:?}"),
        }
    }

    fn push_metadata(out: &mut Vec<u8>, key: &str, value: &GgufValue) {
        push_gguf_string(out, key);
        push_gguf_value(out, value);
    }

    fn synth_gguf_custom(
        arch: &str,
        extra_metadata: &[(&str, GgufValue)],
        tensors: &[SynthTensor],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(2u64 + extra_metadata.len() as u64).to_le_bytes());

        push_metadata(&mut out, "general.alignment", &GgufValue::U32(32));
        push_metadata(
            &mut out,
            "general.architecture",
            &GgufValue::String(arch.to_owned()),
        );
        for (key, value) in extra_metadata {
            push_metadata(&mut out, key, value);
        }

        let mut offset = 0u64;
        for tensor in tensors {
            push_gguf_string(&mut out, &tensor.name);
            out.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in &tensor.shape {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&tensor.dtype.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset += ggml_tensor_bytes(tensor.dtype, &tensor.shape).unwrap();
        }

        while out.len() % 32 != 0 {
            out.push(0);
        }
        for tensor in tensors {
            out.extend_from_slice(&tensor.data);
        }
        out
    }

    fn temp_test_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{label}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn shard_path(dir: &Path, prefix: &str, idx: usize, count: usize) -> PathBuf {
        dir.join(format!("{prefix}-{idx:05}-of-{count:05}.gguf"))
    }

    fn write_synth_shard(
        dir: &Path,
        prefix: &str,
        idx: usize,
        count: usize,
        arch: &str,
        tensors: &[SynthTensor],
        split_count: u64,
        split_tensors_count: u64,
    ) -> PathBuf {
        let path = shard_path(dir, prefix, idx, count);
        let metadata = [
            ("split.no", GgufValue::U32((idx - 1) as u32)),
            ("split.count", GgufValue::U64(split_count)),
            ("split.tensors.count", GgufValue::U64(split_tensors_count)),
        ];
        let bytes = synth_gguf_custom(arch, &metadata, tensors);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn write_basic_two_shards(dir: &Path) -> (PathBuf, PathBuf) {
        let shard1_tensors = [synth_f32_tensor("left", &[1.0, 2.0, 3.0, 4.0])];
        let shard2_tensors = [synth_f32_tensor("right", &[9.0, 10.0, 11.0, 12.0])];
        let shard1 = write_synth_shard(dir, "Model-Q4_K_M", 1, 2, "llama", &shard1_tensors, 2, 2);
        let shard2 = write_synth_shard(dir, "Model-Q4_K_M", 2, 2, "llama", &shard2_tensors, 2, 2);
        (shard1, shard2)
    }

    fn open_err(path: &Path) -> String {
        match open_gguf_source(path, false) {
            Ok(_) => panic!("expected open_gguf_source to fail for {}", path.display()),
            Err(e) => e.to_string(),
        }
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

    #[test]
    fn open_factory_single_file_streaming_and_eager_remain_unchanged() {
        let dir = temp_test_dir("gguf-source-single");
        let path = dir.join("single.gguf");
        std::fs::write(&path, synth_gguf()).unwrap();

        for legacy_eager in [false, true] {
            let source = open_gguf_source(&path, legacy_eager).expect("open source");
            assert_eq!(source.architecture(), Some("llama"));
            let info = source.tensor_info("t").expect("tensor info");
            assert_eq!(info.shape, vec![4]);
            assert_eq!(info.offset, 0);
            assert_eq!(source.tensor_dtype("t"), Some(WeightDtype::F32));
            let body = source.read_tensor_owned("t").unwrap().expect("body");
            assert_eq!(body, f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shard_set_exposes_tensors_from_all_files() {
        let dir = temp_test_dir("gguf-shards-expose");
        let (shard1, _) = write_basic_two_shards(&dir);

        let source = open_gguf_source(&shard1, false).expect("open shard set");
        assert_eq!(source.architecture(), Some("llama"));
        assert!(source.tensor_info("left").is_some());
        assert!(source.tensor_info("right").is_some());
        assert_eq!(source.tensor_info("left").unwrap().offset, 0);
        assert_eq!(source.tensor_info("right").unwrap().offset, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shard_set_reads_tensor_from_owning_shard() {
        let dir = temp_test_dir("gguf-shards-read");
        let (shard1, _) = write_basic_two_shards(&dir);

        let source = open_gguf_source(&shard1, false).expect("open shard set");
        let right = source
            .read_tensor_owned("right")
            .unwrap()
            .expect("right tensor");
        assert_eq!(right, f32_bytes(&[9.0, 10.0, 11.0, 12.0]));
        let left = source
            .read_tensor_owned("left")
            .unwrap()
            .expect("left tensor");
        assert_eq!(left, f32_bytes(&[1.0, 2.0, 3.0, 4.0]));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opening_second_shard_discovers_the_complete_set() {
        let dir = temp_test_dir("gguf-shards-second");
        let (_, shard2) = write_basic_two_shards(&dir);

        let source = open_gguf_source(&shard2, false).expect("open from shard 2");
        assert!(source.tensor_info("left").is_some());
        assert!(source.tensor_info("right").is_some());
        let left = source
            .read_tensor_owned("left")
            .unwrap()
            .expect("left tensor");
        assert_eq!(left, f32_bytes(&[1.0, 2.0, 3.0, 4.0]));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_shard_fails_clearly() {
        let dir = temp_test_dir("gguf-shards-missing");
        let shard2_tensors = [synth_f32_tensor("right", &[9.0, 10.0, 11.0, 12.0])];
        let shard2 = write_synth_shard(&dir, "Model-Q4_K_M", 2, 2, "llama", &shard2_tensors, 2, 2);

        let err = open_err(&shard2);
        assert!(err.contains("missing GGUF shard 1 of 2"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_tensor_name_fails() {
        let dir = temp_test_dir("gguf-shards-duplicate");
        let shard1_tensors = [synth_f32_tensor("dup", &[1.0, 2.0, 3.0, 4.0])];
        let shard2_tensors = [synth_f32_tensor("dup", &[9.0, 10.0, 11.0, 12.0])];
        let shard1 = write_synth_shard(&dir, "Model-Q4_K_M", 1, 2, "llama", &shard1_tensors, 2, 2);
        write_synth_shard(&dir, "Model-Q4_K_M", 2, 2, "llama", &shard2_tensors, 2, 2);

        let err = open_err(&shard1);
        assert!(err.contains("duplicate tensor name `dup`"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inconsistent_shard_count_fails() {
        let dir = temp_test_dir("gguf-shards-count");
        let shard1_tensors = [synth_f32_tensor("left", &[1.0, 2.0, 3.0, 4.0])];
        let shard2_tensors = [synth_f32_tensor("right", &[9.0, 10.0, 11.0, 12.0])];
        let shard1 = write_synth_shard(&dir, "Model-Q4_K_M", 1, 2, "llama", &shard1_tensors, 3, 2);
        write_synth_shard(&dir, "Model-Q4_K_M", 2, 2, "llama", &shard2_tensors, 2, 2);

        let err = open_err(&shard1);
        assert!(err.contains("split.count=3"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_mismatch_fails() {
        let dir = temp_test_dir("gguf-shards-arch");
        let shard1_tensors = [synth_f32_tensor("left", &[1.0, 2.0, 3.0, 4.0])];
        let shard2_tensors = [synth_f32_tensor("right", &[9.0, 10.0, 11.0, 12.0])];
        let shard1 = write_synth_shard(&dir, "Model-Q4_K_M", 1, 2, "llama", &shard1_tensors, 2, 2);
        write_synth_shard(
            &dir,
            "Model-Q4_K_M",
            2,
            2,
            "qwen2moe",
            &shard2_tensors,
            2,
            2,
        );

        let err = open_err(&shard1);
        assert!(
            err.contains("general.architecture `qwen2moe` disagrees"),
            "{err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_tensor_count_mismatch_fails() {
        let dir = temp_test_dir("gguf-shards-tensor-count");
        let shard1_tensors = [synth_f32_tensor("left", &[1.0, 2.0, 3.0, 4.0])];
        let shard2_tensors = [synth_f32_tensor("right", &[9.0, 10.0, 11.0, 12.0])];
        let shard1 = write_synth_shard(&dir, "Model-Q4_K_M", 1, 2, "llama", &shard1_tensors, 2, 3);
        write_synth_shard(&dir, "Model-Q4_K_M", 2, 2, "llama", &shard2_tensors, 2, 3);

        let err = open_err(&shard1);
        assert!(err.contains("split.tensors.count=3"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shard_set_open_does_not_require_tensor_bodies() {
        let dir = temp_test_dir("gguf-shards-header-only");
        let shard1_tensors = [SynthTensor {
            name: "huge_left".to_owned(),
            shape: vec![2_000_000],
            dtype: ggml_dtype::F32,
            data: Vec::new(),
        }];
        let shard2_tensors = [SynthTensor {
            name: "huge_right".to_owned(),
            shape: vec![2_000_000],
            dtype: ggml_dtype::F32,
            data: Vec::new(),
        }];
        let shard1 = write_synth_shard(&dir, "Model-Q4_K_M", 1, 2, "llama", &shard1_tensors, 2, 2);
        write_synth_shard(&dir, "Model-Q4_K_M", 2, 2, "llama", &shard2_tensors, 2, 2);

        let source = open_gguf_source(&shard1, false).expect("open shard set from headers");
        assert!(source.tensor_info("huge_left").is_some());
        assert!(source.tensor_info("huge_right").is_some());
        let err = source
            .read_tensor_owned("huge_right")
            .err()
            .expect("body read should validate the local shard range");
        assert!(err.to_string().contains("past end of file"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stream_header_parse_reports_incomplete_on_truncated_header() {
        let bytes = synth_gguf();
        let parsed = GgufStreamReader::try_parse_header(&bytes[..24], 3, 1, 2, 24)
            .expect("truncated header should not be a permanent error");
        assert!(parsed.is_none());
    }

    #[test]
    fn stream_reader_rejects_unsupported_dtype_without_reading_to_eof() {
        let mut bytes = synth_gguf_with_dtype(ggml_dtype::Q2_K);
        bytes.resize(bytes.len() + 5 * 1024 * 1024, 0);

        let dir = std::env::temp_dir().join(format!(
            "gguf-stream-invalid-dtype-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unsupported-dtype.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let err = GgufStreamReader::open(&path)
            .err()
            .expect("unsupported dtype should fail");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(msg.contains("tensor t"), "{msg}");
        assert!(msg.contains("dtype 10"), "{msg}");

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
