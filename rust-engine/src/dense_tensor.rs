//! Resident dense weight storage for the real-transformer path.
//!
//! Legacy converted checkpoints store every resident dense tensor as raw
//! little-endian f32. Native GGUF conversions can keep selected tensors in
//! their original block-quantized layout; this module hides that difference
//! behind row lookup and matrix-vector APIs used by embeddings, attention
//! projections, router gates, and the LM head.

use crate::inference::{
    dequantize_q8_0_block, dequantize_q8_0_to_f32, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseDType {
    F32,
    Q8_0,
}

impl DenseDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => Some(Self::F32),
            "q8_0" | "q8-0" | "ggml-q8_0" => Some(Self::Q8_0),
            _ => None,
        }
    }
}

impl fmt::Display for DenseDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseTensorManifest {
    pub format_version: u32,
    pub tensors: Vec<DenseTensorManifestEntry>,
}

impl DenseTensorManifest {
    pub fn find_alias(&self, name: &str) -> Option<&DenseTensorManifestEntry> {
        self.tensors.iter().find(|entry| {
            entry.canonical_name == name || entry.aliases.iter().any(|alias| alias == name)
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseTensorManifestEntry {
    pub canonical_name: String,
    pub file: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub dtype: DenseDType,
    pub dims: Vec<usize>,
    pub byte_len: usize,
    #[serde(default)]
    pub checksum: Option<String>,
    #[serde(default)]
    pub tied_to: Option<String>,
}

pub fn dense_checksum(bytes: &[u8]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{h:016x}")
}

#[derive(Clone, Debug, PartialEq)]
pub enum DenseWeight {
    F32 {
        values: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Q8_0 {
        bytes: Arc<[u8]>,
        rows: usize,
        cols: usize,
    },
}

impl DenseWeight {
    pub fn from_f32(values: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            values.len(),
            rows.saturating_mul(cols),
            "dense f32 weight must be [{rows}, {cols}]"
        );
        Self::F32 { values, rows, cols }
    }

    pub fn from_q8_0_bytes(
        bytes: Vec<u8>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, DenseWeightError> {
        let weights = rows
            .checked_mul(cols)
            .ok_or(DenseWeightError::ShapeOverflow { rows, cols })?;
        let expected = q8_0_bytes_for(weights)?;
        if bytes.len() != expected {
            return Err(DenseWeightError::ByteLength {
                dtype: DenseDType::Q8_0,
                rows,
                cols,
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self::Q8_0 {
            bytes: Arc::<[u8]>::from(bytes),
            rows,
            cols,
        })
    }

    #[inline]
    pub fn dtype(&self) -> DenseDType {
        match self {
            Self::F32 { .. } => DenseDType::F32,
            Self::Q8_0 { .. } => DenseDType::Q8_0,
        }
    }

    #[inline]
    pub fn dtype_name(&self) -> &'static str {
        self.dtype().as_str()
    }

    #[inline]
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::Q8_0 { rows, .. } => *rows,
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Q8_0 { cols, .. } => *cols,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::F32 { values, .. } => values.len() * std::mem::size_of::<f32>(),
            Self::Q8_0 { bytes, .. } => bytes.len(),
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::F32 { values, .. } => Some(values.as_slice()),
            Self::Q8_0 { .. } => None,
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::F32 { values, .. } => values.clone(),
            Self::Q8_0 { bytes, .. } => {
                let mut out = Vec::with_capacity(self.len());
                dequantize_q8_0_to_f32(bytes, self.len(), &mut out);
                out
            }
        }
    }

    pub fn iter(&self) -> DenseWeightIter<'_> {
        DenseWeightIter {
            weight: self,
            offset: 0,
            decoded: [0.0; Q8_0_BLOCK_ELEMS],
            decoded_block: None,
        }
    }

    pub fn row_dequant_into(&self, row: usize, out: &mut Vec<f32>) {
        assert!(row < self.rows(), "dense row {row} out of range");
        let cols = self.cols();
        out.clear();
        out.resize(cols, 0.0);
        match self {
            Self::F32 { values, .. } => {
                let start = row * cols;
                out.copy_from_slice(&values[start..start + cols]);
            }
            Self::Q8_0 { bytes, .. } => {
                q8_0_copy_range(bytes, row * cols, cols, out);
            }
        }
    }

    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.rows()];
        self.matvec_into(x, &mut y);
        y
    }

    pub fn matvec_into(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols(), "dense matvec input length mismatch");
        assert_eq!(y.len(), self.rows(), "dense matvec output length mismatch");
        match self {
            Self::F32 { values, rows, cols } => {
                crate::transformer::matmul_row_major_into(values, x, y, *rows, *cols);
            }
            Self::Q8_0 { bytes, rows, cols } => {
                if *rows == 0 || *cols == 0 {
                    y.fill(0.0);
                    return;
                }
                crate::parallel::par_row_chunks(y, *cols, |row_start, out| {
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = q8_0_row_dot(bytes, row_start + i, *cols, x);
                    }
                });
            }
        }
    }

    pub fn matmat_into(&self, x: &[f32], x_rows: usize, out: &mut [f32]) {
        let cols = self.cols();
        let rows = self.rows();
        assert_eq!(x.len(), x_rows * cols, "dense matmat input length mismatch");
        assert_eq!(
            out.len(),
            x_rows * rows,
            "dense matmat output length mismatch"
        );
        for i in 0..x_rows {
            let input = &x[i * cols..(i + 1) * cols];
            let output = &mut out[i * rows..(i + 1) * rows];
            self.matvec_into(input, output);
        }
    }

    pub fn greedy_argmax(&self, x: &[f32]) -> u32 {
        assert_eq!(x.len(), self.cols(), "dense argmax input length mismatch");
        if let Some((start, end, tasks)) = self.parallel_row_work() {
            let (best_id, _) = (0..tasks)
                .into_par_iter()
                .map(|task| {
                    let chunk_start = start + task * (end - start).div_ceil(tasks);
                    let chunk_end = (chunk_start + (end - start).div_ceil(tasks)).min(end);
                    self.greedy_argmax_range(x, chunk_start, chunk_end)
                })
                .reduce(
                    || (usize::MAX, f32::NEG_INFINITY),
                    |a, b| dense_best_pair(a, b),
                );
            return if best_id == usize::MAX {
                0
            } else {
                best_id as u32
            };
        }
        let (best_id, _) = self.greedy_argmax_range(x, 0, self.rows());
        if best_id == usize::MAX {
            0
        } else {
            best_id as u32
        }
    }

    pub fn top_k_logits(&self, x: &[f32], k: usize) -> Vec<(usize, f32)> {
        assert_eq!(x.len(), self.cols(), "dense top-k input length mismatch");
        if k == 0 || self.rows() == 0 {
            return Vec::new();
        }
        let limit = k.min(self.rows());
        if let Some((start, end, tasks)) = self.parallel_row_work() {
            return (0..tasks)
                .into_par_iter()
                .map(|task| {
                    let chunk_rows = (end - start).div_ceil(tasks);
                    let chunk_start = start + task * chunk_rows;
                    let chunk_end = (chunk_start + chunk_rows).min(end);
                    self.top_k_logits_range(x, limit, chunk_start, chunk_end)
                })
                .reduce(Vec::new, |mut a, b| {
                    for (row, score) in b {
                        insert_top_candidate(&mut a, limit, row, score);
                    }
                    a
                });
        }
        self.top_k_logits_range(x, limit, 0, self.rows())
    }

    fn parallel_row_work(&self) -> Option<(usize, usize, usize)> {
        let rows = self.rows();
        let cols = self.cols();
        let total = rows.checked_mul(cols)?;
        let nthreads = crate::parallel::num_threads();
        if rows <= 1
            || nthreads <= 1
            || total < crate::parallel::MIN_TOTAL_FOR_PARALLEL
            || crate::parallel::in_rayon_worker()
        {
            return None;
        }
        let work_limited = (total / crate::parallel::MIN_ELEMS_PER_TASK).max(1);
        let tasks = nthreads.min(work_limited).min(rows).max(1);
        Some((0, rows, tasks))
    }

    fn greedy_argmax_range(&self, x: &[f32], start: usize, end: usize) -> (usize, f32) {
        let mut best_id = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for row in start..end {
            let score = self.row_dot(row, x);
            if score.total_cmp(&best_score).is_gt()
                || (score.total_cmp(&best_score).is_eq() && row < best_id)
            {
                best_id = row;
                best_score = score;
            }
        }
        (best_id, best_score)
    }

    fn top_k_logits_range(
        &self,
        x: &[f32],
        limit: usize,
        start: usize,
        end: usize,
    ) -> Vec<(usize, f32)> {
        let mut top = Vec::with_capacity(limit);
        for row in start..end {
            let score = self.row_dot(row, x);
            insert_top_candidate(&mut top, limit, row, score);
        }
        top
    }

    fn row_dot(&self, row: usize, x: &[f32]) -> f32 {
        assert!(row < self.rows(), "dense row {row} out of range");
        match self {
            Self::F32 { values, cols, .. } => {
                let start = row * *cols;
                crate::kernels::dot_f32(&values[start..start + *cols], x)
            }
            Self::Q8_0 { bytes, cols, .. } => q8_0_row_dot(bytes, row, *cols, x),
        }
    }
}

impl Default for DenseWeight {
    fn default() -> Self {
        Self::F32 {
            values: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }
}

impl PartialEq<Vec<f32>> for DenseWeight {
    fn eq(&self, other: &Vec<f32>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == *b)
    }
}

pub struct DenseWeightIter<'a> {
    weight: &'a DenseWeight,
    offset: usize,
    decoded: [f32; Q8_0_BLOCK_ELEMS],
    decoded_block: Option<usize>,
}

impl Iterator for DenseWeightIter<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.weight.len() {
            return None;
        }
        let value = match self.weight {
            DenseWeight::F32 { values, .. } => values[self.offset],
            DenseWeight::Q8_0 { bytes, .. } => {
                let block = self.offset / Q8_0_BLOCK_ELEMS;
                if self.decoded_block != Some(block) {
                    let start = block * Q8_0_BLOCK_BYTES;
                    dequantize_q8_0_block(
                        &bytes[start..start + Q8_0_BLOCK_BYTES],
                        &mut self.decoded,
                    );
                    self.decoded_block = Some(block);
                }
                self.decoded[self.offset % Q8_0_BLOCK_ELEMS]
            }
        };
        self.offset += 1;
        Some(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseWeightError {
    ShapeOverflow {
        rows: usize,
        cols: usize,
    },
    ByteLength {
        dtype: DenseDType,
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for DenseWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow { rows, cols } => {
                write!(f, "dense shape [{rows}, {cols}] overflows usize")
            }
            Self::ByteLength {
                dtype,
                rows,
                cols,
                expected,
                actual,
            } => write!(
                f,
                "dense {dtype} weight [{rows}, {cols}] has {actual} bytes, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for DenseWeightError {}

fn q8_0_bytes_for(weights: usize) -> Result<usize, DenseWeightError> {
    weights
        .div_ceil(Q8_0_BLOCK_ELEMS)
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or(DenseWeightError::ShapeOverflow {
            rows: weights,
            cols: Q8_0_BLOCK_BYTES,
        })
}

fn q8_0_copy_range(bytes: &[u8], start_weight: usize, len: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), len);
    if len == 0 {
        return;
    }
    let first_block = start_weight / Q8_0_BLOCK_ELEMS;
    let last_block = (start_weight + len - 1) / Q8_0_BLOCK_ELEMS;
    let mut decoded = [0.0f32; Q8_0_BLOCK_ELEMS];
    for block in first_block..=last_block {
        let block_start = block * Q8_0_BLOCK_ELEMS;
        let start = block * Q8_0_BLOCK_BYTES;
        dequantize_q8_0_block(&bytes[start..start + Q8_0_BLOCK_BYTES], &mut decoded);
        let copy_start = start_weight.max(block_start);
        let copy_end = (start_weight + len).min(block_start + Q8_0_BLOCK_ELEMS);
        for flat in copy_start..copy_end {
            out[flat - start_weight] = decoded[flat - block_start];
        }
    }
}

fn q8_0_row_dot(bytes: &[u8], row: usize, cols: usize, x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), cols);
    if cols == 0 {
        return 0.0;
    }
    let row_start = row * cols;
    let first_block = row_start / Q8_0_BLOCK_ELEMS;
    let last_block = (row_start + cols - 1) / Q8_0_BLOCK_ELEMS;
    let mut decoded = [0.0f32; Q8_0_BLOCK_ELEMS];
    let mut acc = 0.0f32;
    for block in first_block..=last_block {
        let block_start = block * Q8_0_BLOCK_ELEMS;
        let start = block * Q8_0_BLOCK_BYTES;
        dequantize_q8_0_block(&bytes[start..start + Q8_0_BLOCK_BYTES], &mut decoded);
        let dot_start = row_start.max(block_start);
        let dot_end = (row_start + cols).min(block_start + Q8_0_BLOCK_ELEMS);
        for flat in dot_start..dot_end {
            acc += decoded[flat - block_start] * x[flat - row_start];
        }
    }
    acc
}

#[inline]
fn dense_score_is_better(candidate_score: f32, candidate_id: usize, score: f32, id: usize) -> bool {
    match candidate_score.total_cmp(&score) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => candidate_id < id,
        std::cmp::Ordering::Less => false,
    }
}

fn dense_best_pair(a: (usize, f32), b: (usize, f32)) -> (usize, f32) {
    if dense_score_is_better(a.1, a.0, b.1, b.0) {
        a
    } else {
        b
    }
}

fn insert_top_candidate(top: &mut Vec<(usize, f32)>, limit: usize, row: usize, score: f32) {
    let pos = top
        .iter()
        .position(|&(id, s)| dense_score_is_better(score, row, s, id));
    match pos {
        Some(pos) => {
            top.insert(pos, (row, score));
            if top.len() > limit {
                top.pop();
            }
        }
        None if top.len() < limit => top.push((row, score)),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::quantize_q8_0_block;

    fn quantize_q8(values: &[f32]) -> Vec<u8> {
        let blocks = values.len().div_ceil(Q8_0_BLOCK_ELEMS);
        let mut out = vec![0u8; blocks * Q8_0_BLOCK_BYTES];
        for block in 0..blocks {
            let start = block * Q8_0_BLOCK_ELEMS;
            let end = (start + Q8_0_BLOCK_ELEMS).min(values.len());
            quantize_q8_0_block(
                &values[start..end],
                &mut out[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES],
            );
        }
        out
    }

    #[test]
    fn q8_row_dequant_handles_rows_that_cross_blocks() {
        let values: Vec<f32> = (0..105).map(|i| (i as f32 - 48.0) / 8.0).collect();
        let weight = DenseWeight::from_q8_0_bytes(quantize_q8(&values), 3, 35).unwrap();
        let mut row = Vec::new();
        weight.row_dequant_into(1, &mut row);
        let expected = &weight.to_f32_vec()[35..70];
        assert_eq!(row, expected);
    }

    #[test]
    fn q8_matvec_matches_dequantized_reference() {
        let values: Vec<f32> = (0..105).map(|i| ((i % 17) as f32 - 8.0) / 6.0).collect();
        let weight = DenseWeight::from_q8_0_bytes(quantize_q8(&values), 3, 35).unwrap();
        let x: Vec<f32> = (0..35).map(|i| (i as f32 + 1.0) / 19.0).collect();
        let got = weight.matvec(&x);
        let f32_weight = DenseWeight::from_f32(weight.to_f32_vec(), 3, 35);
        let expected = f32_weight.matvec(&x);
        assert_eq!(got, expected);
    }

    #[test]
    fn q8_greedy_argmax_matches_dequantized_reference_with_ties() {
        let rows = 4;
        let cols = 33;
        let mut values = vec![0.0f32; rows * cols];
        values[2 * cols] = 4.0;
        values[3 * cols] = 4.0;
        let x = vec![1.0f32; cols];
        let q8 = DenseWeight::from_q8_0_bytes(quantize_q8(&values), rows, cols).unwrap();
        let f32_weight = DenseWeight::from_f32(q8.to_f32_vec(), rows, cols);
        assert_eq!(q8.greedy_argmax(&x), f32_weight.greedy_argmax(&x));
        assert_eq!(q8.greedy_argmax(&x), 2);
    }

    #[test]
    fn q8_parallel_lm_head_candidates_match_dequantized_reference() {
        let rows = 8192;
        let cols = 33;
        let mut values = vec![0.0f32; rows * cols];
        for row in 0..rows {
            values[row * cols] = row as f32 / 10_000.0;
        }
        values[7000 * cols] = 100.0;
        values[7001 * cols] = 100.0;
        let mut x = vec![0.0f32; cols];
        x[0] = 1.0;

        let q8 = DenseWeight::from_q8_0_bytes(quantize_q8(&values), rows, cols).unwrap();
        let f32_weight = DenseWeight::from_f32(q8.to_f32_vec(), rows, cols);

        assert_eq!(q8.greedy_argmax(&x), f32_weight.greedy_argmax(&x));
        assert_eq!(q8.greedy_argmax(&x), 7000);
        assert_eq!(q8.top_k_logits(&x, 8), f32_weight.top_k_logits(&x, 8));
    }

    #[test]
    fn q8_rejects_malformed_byte_len() {
        let err = DenseWeight::from_q8_0_bytes(vec![0u8; Q8_0_BLOCK_BYTES - 1], 1, 32).unwrap_err();
        assert!(matches!(err, DenseWeightError::ByteLength { .. }));
    }
}
