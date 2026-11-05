//! In-memory synthetic payload writer. Never reads GGUF, UTH1, SSD or a model.
//! Q4 byte 2+j stores columns j (low nibble) and j+16 (high nibble), not
//! adjacent columns. Scales are IEEE binary16, little endian.
use crate::{Reject, Result, BLOCK_BYTES, D, DFF, PROJECTION_BYTES, Q4_BYTES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection { Gate, Up, Down }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation { OutputRowsInputColumns, Transposed }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub projection: Projection,
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub offset_bytes: usize,
    pub orientation: Orientation,
}

impl Layout {
    pub fn exact(projection: Projection) -> Self {
        let (rows, cols, offset_bytes) = match projection {
            Projection::Gate => (DFF, D, 0),
            Projection::Up => (DFF, D, PROJECTION_BYTES),
            Projection::Down => (D, DFF, 2 * PROJECTION_BYTES),
        };
        Self { projection, rows, cols, row_stride_bytes: (cols / 32) * 18,
            offset_bytes, orientation: Orientation::OutputRowsInputColumns }
    }

    pub fn validate(self, total: usize) -> Result<()> {
        let expected = Self::exact(self.projection);
        if self.rows != expected.rows || self.cols != expected.cols || self.cols % 64 != 0 {
            return Err(Reject::Shape);
        }
        if self.orientation != expected.orientation { return Err(Reject::Orientation); }
        if self.row_stride_bytes != expected.row_stride_bytes { return Err(Reject::Stride); }
        if self.offset_bytes != expected.offset_bytes { return Err(Reject::Orientation); }
        if total != Q4_BYTES { return Err(Reject::Tail); }
        let end = self.rows.checked_mul(self.row_stride_bytes)
            .and_then(|size| self.offset_bytes.checked_add(size)).ok_or(Reject::Bounds)?;
        if end > total { return Err(Reject::Bounds); }
        Ok(())
    }
}

/// Writer policy, independent of reference.rs's scalar decoder and expected.rs.
/// Every Q4 weight is nonzero. Low/high half signs differ in Gate and Up.
fn block(projection: Projection, row: usize) -> [u8; BLOCK_BYTES] {
    let magnitude = (row % 4 + 1) as u8;
    let mut out = [0u8; BLOCK_BYTES];
    let scale: u16 = match projection {
        Projection::Gate => 0x3800, // 1/2
        Projection::Up => 0x2800,   // 1/32
        Projection::Down => 0x3000, // 1/8
    };
    out[..2].copy_from_slice(&scale.to_le_bytes());
    for j in 0..16 {
        let (low, high) = match projection {
            Projection::Gate => (8 + magnitude, 8 - magnitude),
            Projection::Up if row % 2 == 0 => (8 - magnitude, 8 + magnitude),
            Projection::Up => (8 + magnitude, 8 - magnitude),
            Projection::Down => {
                // Product sign alternates by column. Down row sign alternates
                // by row; both packed halves have the same parity.
                let q = if (row + j) % 2 == 0 { 8 + magnitude } else { 8 - magnitude };
                (q, q)
            }
        };
        out[j + 2] = low | (high << 4);
    }
    out
}

pub struct Fixture {
    bytes: Vec<u8>,
    input: Vec<f32>,
}
impl Fixture {
    /// Authored, NOT EXECUTED in P1. Allocates only fixed-size CPU buffers.
    pub fn build() -> Self {
        let mut bytes = vec![0u8; Q4_BYTES];
        for projection in [Projection::Gate, Projection::Up, Projection::Down] {
            let layout = Layout::exact(projection);
            for row in 0..layout.rows {
                let image = block(projection, row);
                for b in 0..layout.cols / 32 {
                    let offset = layout.offset_bytes + row * layout.row_stride_bytes + b * 18;
                    bytes[offset..offset + 18].copy_from_slice(&image);
                }
            }
        }
        let input = (0..D).map(|column| f32::from_bits(
            if column % 32 < 16 { 0x3d00_0000 } else { 0xbd00_0000 }
        )).collect();
        Self { bytes, input }
    }
    pub fn bytes(&self) -> &[u8] { &self.bytes }
    pub fn input(&self) -> &[f32] { &self.input }
    pub fn validate(&self) -> Result<()> {
        validate_payload(&self.bytes)?;
        validate_input(&self.input)
    }
}

pub fn validate_input(input: &[f32]) -> Result<()> {
    if input.len() != D { return Err(Reject::Shape); }
    for (column, value) in input.iter().enumerate() {
        if !value.is_finite() { return Err(Reject::Nonfinite); }
        let bits = if column % 32 < 16 { 0x3d00_0000 } else { 0xbd00_0000 };
        if value.to_bits() != bits { return Err(Reject::InputBits); }
    }
    Ok(())
}

pub fn matrix_span(bytes: &[u8], layout: Layout) -> Result<&[u8]> {
    layout.validate(bytes.len())?;
    bytes.get(layout.offset_bytes..layout.offset_bytes + PROJECTION_BYTES).ok_or(Reject::Bounds)
}

pub fn validate_payload(bytes: &[u8]) -> Result<()> {
    if bytes.len() != Q4_BYTES { return Err(Reject::Tail); }
    for projection in [Projection::Gate, Projection::Up, Projection::Down] {
        let layout = Layout::exact(projection);
        let matrix = matrix_span(bytes, layout)?;
        for (index, candidate) in matrix.chunks_exact(BLOCK_BYTES).enumerate() {
            let row = index / (layout.cols / 32);
            let expected = block(projection, row);
            let scale = u16::from_le_bytes([candidate[0], candidate[1]]);
            if scale & 0x7c00 == 0x7c00 || candidate[..2] != expected[..2] {
                return Err(Reject::Scale);
            }
            if candidate != expected.as_slice() { return Err(Reject::FixtureBytes); }
        }
    }
    Ok(())
}
