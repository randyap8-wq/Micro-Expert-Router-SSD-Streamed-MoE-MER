//! **Tier 2 — packed expert storage (single-blob layout + coalesced reads).**
//!
//! The default [`crate::io_provider::NvmeStorage`] keeps **one file per
//! expert** (`expert_<id>.bin`). That layout is simple and robust, but it has
//! two structural costs on the SSD-streaming hot path:
//!
//! 1. **One fd per distinct expert.** A low-hit-rate run streams thousands of
//!    distinct experts past the bounded fd cache, so every miss can pay an
//!    `open(2)` syscall and thrash the descriptor LRU.
//! 2. **No read coalescing.** When a single token misses on `K` experts the
//!    engine issues `K` independent `pread(2)`s. Even dispatched concurrently
//!    the NVMe can only overlap them up to its queue depth, and each one is a
//!    separate command with its own submission overhead.
//!
//! The **packed layout** concatenates every expert payload into a single blob
//! file, each occupying one block-aligned `expert_size` slot, in an order
//! chosen *offline* (by routing-frequency profile or co-firing affinity — see
//! the `repack` CLI subcommand). A small JSON [`PackedManifest`] records each
//! expert's byte `(offset, len)` within the blob.
//!
//! Two wins fall out:
//!
//! * **One fd for everything.** All reads target the same blob descriptor — no
//!   per-expert `open()`, no fd-cache pressure.
//! * **Coalesced vectored reads.** Experts placed in adjacent slots are
//!   physically contiguous, so a run of them can be fetched in a **single
//!   `preadv(2)`** that scatters straight into the per-expert
//!   [`crate::buffer_pool::PooledBuffer`]s — one syscall, one seek, full
//!   device queue depth — instead of `K` separate reads. See
//!   [`NvmeStorage::read_experts_batch`](crate::io_provider::NvmeStorage::read_experts_batch).
//!
//! The whole layer is **opt-in**: with no `[storage] packed_blob` configured
//! the engine uses the original one-file-per-expert path bit-for-bit.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::io_provider::open_expert_file;

/// One expert's byte location within the packed blob.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PackedEntry {
    /// Byte offset of the expert's payload from the start of the blob.
    /// Always a multiple of [`PackedManifest::block_align`].
    pub offset: u64,
    /// Number of payload bytes. Equal to [`PackedManifest::expert_size`]
    /// for the uniform-slot layout the engine produces.
    pub len: u64,
}

/// On-disk index mapping every expert id to its `(offset, len)` slot in a
/// single packed blob. Serialised as JSON beside the blob (the `repack`
/// subcommand emits both; the engine loads them via [`PackedBlob::open`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedManifest {
    /// Block alignment every offset and slot is padded to (matches
    /// `StorageConfig::block_align`, typically 4096). Guarantees each
    /// slot start is `O_DIRECT`-safe.
    pub block_align: u64,
    /// Uniform padded slot size for each expert, in bytes. Equal to
    /// `StorageConfig::expert_size`; the engine asserts the two agree at
    /// wiring time so a packed read fills a whole `PooledBuffer`.
    pub expert_size: u64,
    /// `id -> (offset, len)`.
    pub entries: HashMap<u32, PackedEntry>,
    /// Expert ids in ascending blob-offset order. Persisted so a re-pack
    /// or a human can inspect / reproduce the chosen physical layout.
    pub order: Vec<u32>,
}

impl PackedManifest {
    /// Build a manifest for `order` (the physical layout, front to back)
    /// with a uniform `expert_size` slot per expert. Offsets are assigned
    /// densely: slot `i` lives at `i * expert_size`.
    pub fn uniform(order: Vec<u32>, expert_size: u64, block_align: u64) -> Self {
        let mut entries = HashMap::with_capacity(order.len());
        for (i, &id) in order.iter().enumerate() {
            entries.insert(
                id,
                PackedEntry {
                    offset: i as u64 * expert_size,
                    len: expert_size,
                },
            );
        }
        Self {
            block_align,
            expert_size,
            entries,
            order,
        }
    }

    /// Total number of expert slots in the blob.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the manifest indexes no experts.
    #[allow(dead_code)] // public API symmetry with `len`.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Expected blob length in bytes (last slot end). `0` for an empty
    /// manifest.
    pub fn blob_len(&self) -> u64 {
        self.entries
            .values()
            .map(|e| e.offset + e.len)
            .max()
            .unwrap_or(0)
    }

    /// Validate that each manifest entry matches the uniform-slot layout used
    /// by the packed read path.
    pub fn validate(&self) -> Result<(), String> {
        if self.entries.is_empty() {
            return Ok(());
        }
        if self.expert_size == 0 {
            return Err("packed manifest expert_size must be non-zero".to_string());
        }
        for (&id, entry) in &self.entries {
            if entry.len != self.expert_size {
                return Err(format!(
                    "packed manifest entry for expert {id} has len {} but expert_size is {}",
                    entry.len, self.expert_size
                ));
            }
            if entry.offset % self.expert_size != 0 {
                return Err(format!(
                    "packed manifest entry for expert {id} has offset {} which is not a multiple of expert_size {}",
                    entry.offset, self.expert_size
                ));
            }
        }
        Ok(())
    }

    /// Serialise to pretty JSON.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }

    /// Write the manifest to `path` as JSON.
    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let json = self
            .to_json()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a manifest from a JSON file at `path`.
    pub fn load_from(path: &Path) -> io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Self::from_json(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// An opened packed blob: the shared blob file descriptor plus its
/// [`PackedManifest`]. Cheap to clone into background prefetch tasks via the
/// `Arc<File>` handle.
#[derive(Debug)]
pub struct PackedBlob {
    file: Arc<File>,
    manifest: PackedManifest,
    blob_path: PathBuf,
}

impl PackedBlob {
    /// Open the blob at `blob_path` and load its manifest from
    /// `manifest_path`. `use_direct_io` mirrors the engine's
    /// `StorageConfig::use_direct_io` so the packed path honours the same
    /// page-cache-bypass policy as the per-file path.
    pub fn open(blob_path: &Path, manifest_path: &Path, use_direct_io: bool) -> io::Result<Self> {
        let manifest = PackedManifest::load_from(manifest_path)?;
        let file = open_expert_file(blob_path, use_direct_io)?;
        let blob = Self {
            file: Arc::new(file),
            manifest,
            blob_path: blob_path.to_path_buf(),
        };
        // Reject malformed/stale manifests at open time so direct callers
        // (not just the attach path) never drive wrong offsets into reads.
        blob.validate()?;
        Ok(blob)
    }

    /// The shared blob file descriptor.
    pub fn file(&self) -> &Arc<File> {
        &self.file
    }

    /// The blob's manifest.
    pub fn manifest(&self) -> &PackedManifest {
        &self.manifest
    }

    /// Look up an expert's slot, if it is packed in this blob.
    pub fn entry(&self, id: u32) -> Option<PackedEntry> {
        self.manifest.entries.get(&id).copied()
    }

    /// Whether `id` is present in this blob.
    #[allow(dead_code)] // public lookup API; used by external callers / tests.
    pub fn contains(&self, id: u32) -> bool {
        self.manifest.entries.contains_key(&id)
    }

    /// Number of experts packed in this blob.
    pub fn len(&self) -> usize {
        self.manifest.len()
    }

    /// Whether the blob is empty.
    #[allow(dead_code)] // public API symmetry with `len`.
    pub fn is_empty(&self) -> bool {
        self.manifest.is_empty()
    }

    /// Filesystem path of the blob (for diagnostics).
    #[allow(dead_code)] // surfaced for diagnostics / external callers.
    pub fn path(&self) -> &Path {
        &self.blob_path
    }

    /// Validate manifest invariants and ensure the blob file is long enough
    /// for the manifest's declared slots.
    pub fn validate(&self) -> io::Result<()> {
        self.manifest
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let actual_len = self.file.metadata()?.len();
        let expected_len = self.manifest.blob_len();
        if actual_len < expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "packed blob {} is truncated: length {actual_len} bytes but manifest requires {expected_len} bytes",
                    self.blob_path.display()
                ),
            ));
        }
        Ok(())
    }
}

/// A maximal run of experts whose blob slots are physically contiguous, i.e.
/// `slot[i+1].offset == slot[i].offset + slot[i].len`. A run can be fetched
/// with a single coalesced `preadv`.
///
/// `members` lists the experts in ascending-offset order; each tuple is the
/// expert id and the index into the caller's original request slice (so the
/// scattered bytes land in the right destination buffer).
#[derive(Debug, Clone)]
#[allow(dead_code)] // produced by `coalesce_runs` on the batched packed-read path.
pub struct ContiguousRun {
    /// Byte offset of the run's first slot in the blob.
    pub start_offset: u64,
    /// Total bytes the run spans (sum of member lens).
    pub total_len: u64,
    /// `(expert_id, original_request_index)` in ascending-offset order.
    pub members: Vec<(u32, usize)>,
}

/// Group a set of requested experts into maximal contiguous runs over their
/// packed-blob slots.
///
/// `requested` is `(expert_id, original_index, entry)`. The result is sorted
/// by `start_offset`; each [`ContiguousRun`] holds the members that a single
/// `preadv` can satisfy. Experts with no adjacent neighbour come back as
/// singleton runs, which the caller services with the ordinary per-expert
/// fault-tolerant read.
#[allow(dead_code)] // entry point of the batched packed-read path (see io_provider).
pub fn coalesce_runs(mut requested: Vec<(u32, usize, PackedEntry)>) -> Vec<ContiguousRun> {
    if requested.is_empty() {
        return Vec::new();
    }
    requested.sort_by_key(|(_, _, e)| e.offset);

    let mut runs: Vec<ContiguousRun> = Vec::new();
    let mut cur = ContiguousRun {
        start_offset: requested[0].2.offset,
        total_len: requested[0].2.len,
        members: vec![(requested[0].0, requested[0].1)],
    };
    let mut cur_end = requested[0].2.offset + requested[0].2.len;

    for (id, idx, entry) in requested.into_iter().skip(1) {
        if entry.offset == cur_end {
            // Physically adjacent — extend the current run.
            cur.total_len += entry.len;
            cur.members.push((id, idx));
            cur_end = entry.offset + entry.len;
        } else {
            // Gap (or duplicate offset) — close the run and start a new one.
            runs.push(std::mem::replace(
                &mut cur,
                ContiguousRun {
                    start_offset: entry.offset,
                    total_len: entry.len,
                    members: vec![(id, idx)],
                },
            ));
            cur_end = entry.offset + entry.len;
        }
    }
    runs.push(cur);
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(offset: u64, len: u64) -> PackedEntry {
        PackedEntry { offset, len }
    }

    #[test]
    fn uniform_manifest_assigns_dense_offsets() {
        let m = PackedManifest::uniform(vec![7, 3, 9], 4096, 4096);
        assert_eq!(m.entries[&7].offset, 0);
        assert_eq!(m.entries[&3].offset, 4096);
        assert_eq!(m.entries[&9].offset, 8192);
        assert_eq!(m.entries[&7].len, 4096);
        assert_eq!(m.blob_len(), 3 * 4096);
        assert_eq!(m.order, vec![7, 3, 9]);
    }

    #[test]
    fn packed_manifest_validate_accepts_uniform_slots() {
        let m = PackedManifest::uniform(vec![7, 3, 9], 4096, 4096);
        assert!(m.validate().is_ok());
    }

    #[test]
    fn packed_manifest_validate_rejects_len_mismatch() {
        let mut m = PackedManifest::uniform(vec![7, 3], 4096, 4096);
        m.entries.get_mut(&3).unwrap().len = 2048;
        let err = m.validate().unwrap_err();
        assert!(err.contains("expert 3"), "unexpected error: {err}");
        assert!(err.contains("len 2048"), "unexpected error: {err}");
    }

    #[test]
    fn packed_manifest_validate_rejects_unaligned_offset() {
        let mut m = PackedManifest::uniform(vec![7, 3], 4096, 4096);
        m.entries.get_mut(&3).unwrap().offset = 512;
        let err = m.validate().unwrap_err();
        assert!(err.contains("expert 3"), "unexpected error: {err}");
        assert!(err.contains("offset 512"), "unexpected error: {err}");
    }

    #[test]
    fn manifest_json_round_trips() {
        let m = PackedManifest::uniform(vec![0, 1, 2, 5], 2 * 1024 * 1024, 4096);
        let json = m.to_json().unwrap();
        let back = PackedManifest::from_json(&json).unwrap();
        assert_eq!(back.expert_size, m.expert_size);
        assert_eq!(back.block_align, m.block_align);
        assert_eq!(back.order, m.order);
        for id in [0u32, 1, 2, 5] {
            assert_eq!(back.entries[&id].offset, m.entries[&id].offset);
            assert_eq!(back.entries[&id].len, m.entries[&id].len);
        }
    }

    #[test]
    fn coalesce_merges_only_adjacent_slots() {
        // Slots: id10@0, id11@100, id12@200 are contiguous (len 100);
        // id20@500 is isolated.
        let req = vec![
            (12u32, 0usize, entry(200, 100)),
            (10u32, 1usize, entry(0, 100)),
            (20u32, 2usize, entry(500, 100)),
            (11u32, 3usize, entry(100, 100)),
        ];
        let runs = coalesce_runs(req);
        assert_eq!(runs.len(), 2, "one 3-wide run + one singleton");

        let big = &runs[0];
        assert_eq!(big.start_offset, 0);
        assert_eq!(big.total_len, 300);
        assert_eq!(
            big.members,
            vec![(10, 1), (11, 3), (12, 0)],
            "members carry original request indices in offset order"
        );

        let small = &runs[1];
        assert_eq!(small.start_offset, 500);
        assert_eq!(small.total_len, 100);
        assert_eq!(small.members, vec![(20, 2)]);
    }

    #[test]
    fn coalesce_handles_gap_then_resume() {
        // 0..100 and 100..200 contiguous, gap, then 4096..4196 alone.
        let req = vec![
            (1u32, 0usize, entry(0, 100)),
            (2u32, 1usize, entry(100, 100)),
            (3u32, 2usize, entry(4096, 100)),
        ];
        let runs = coalesce_runs(req);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].members.len(), 2);
        assert_eq!(runs[1].members.len(), 1);
        assert_eq!(runs[1].start_offset, 4096);
    }

    #[test]
    fn coalesce_empty_is_empty() {
        assert!(coalesce_runs(Vec::new()).is_empty());
    }

    #[test]
    fn open_rejects_truncated_blob() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let blob_path = dir.join(format!("trunc_{pid}.blob"));
        let manifest_path = dir.join(format!("trunc_{pid}.manifest.json"));
        // Manifest declares two 4096-byte slots (blob_len = 8192) but the
        // blob file holds only one slot's worth of bytes.
        let m = PackedManifest::uniform(vec![0, 1], 4096, 4096);
        m.write_to(&manifest_path).unwrap();
        std::fs::write(&blob_path, vec![0u8; 4096]).unwrap();
        let err = PackedBlob::open(&blob_path, &manifest_path, false).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&blob_path);
        let _ = std::fs::remove_file(&manifest_path);
    }
}
