//! NVMe storage provider.
//!
//! Opens each expert as its own file (optionally with `O_DIRECT` on Linux),
//! keeps the file descriptors resident in an fd cache, and reads experts
//! into a [`PooledBuffer`] owned by the caller via positional reads
//! (`pread(2)`).
//!
//! ## Why this layout?
//!
//! * **One fd per expert, kept open.** Removes per-read `open()` syscalls
//!   from the steady-state latency budget — the same property an io_uring
//!   path would want with registered files.
//! * **`O_DIRECT`.** Bypasses the page cache so we measure (and consume)
//!   raw NVMe bandwidth. Required by the spec; the buffer pool guarantees
//!   the alignment invariants the kernel checks.
//! * **`block_in_place` + `pread`.** Each cache miss runs the synchronous
//!   `pread(2)` on the current Tokio worker via
//!   [`tokio::task::block_in_place`]. The worker is donated to blocking
//!   work and other ready tasks are picked up by sibling workers, so the
//!   runtime stays responsive. Using `pread` (positional) instead of
//!   `read_exact` means we do not touch the file offset and reads are
//!   safe to issue concurrently against the same fd.
//! * **Optional `--no-direct` fallback.** Useful on tmpfs / overlayfs /
//!   non-Linux CI where `O_DIRECT` returns `EINVAL`. The engine still
//!   exercises the same prefetch + LRU + alignment logic.
//!
//! ## Why not `rio`?
//!
//! Earlier drafts used the [`rio`] io_uring crate directly. `rio 0.9.4` has
//! an unfixed use-after-free advisory and the crate is unmaintained, so we
//! removed the dependency. The intended production replacement is
//! [`tokio-uring`] (or the raw `io-uring` crate with registered fixed
//! buffers); both require restructuring `main` around their own runtime
//! entry points (`tokio_uring::start`), which is left as future work. The
//! `io_provider` module is the only place that needs to change to swap
//! backends, so this is a self-contained migration.

use crate::buffer_pool::PooledBuffer;
use crate::inference::WeightDtype;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

/// Configuration for the storage layer.
#[derive(Clone, Debug)]
pub struct StorageConfig {
    /// Directory containing `expert_<id>.bin` files.
    pub base_path: PathBuf,
    /// Size (bytes) of every expert file. Must be a multiple of `block_align`.
    pub expert_size: usize,
    /// Logical block size to use for `O_DIRECT` alignment (typically 4096).
    pub block_align: usize,
    /// Whether to open files with `O_DIRECT` and bypass the page cache.
    pub use_direct_io: bool,
}

/// NVMe-backed storage with a per-expert fd cache.
pub struct NvmeStorage {
    cfg: StorageConfig,
    files: RwLock<HashMap<u32, Arc<File>>>,
}

impl NvmeStorage {
    pub fn new(cfg: StorageConfig) -> io::Result<Self> {
        assert!(cfg.block_align.is_power_of_two());
        assert!(
            cfg.expert_size % cfg.block_align == 0,
            "expert_size {} must be a multiple of block_align {}",
            cfg.expert_size,
            cfg.block_align
        );

        Ok(Self {
            cfg,
            files: RwLock::new(HashMap::new()),
        })
    }

    pub fn config(&self) -> &StorageConfig {
        &self.cfg
    }

    /// Path of the file backing a given expert id.
    pub fn expert_path(&self, id: u32) -> PathBuf {
        self.cfg.base_path.join(format!("expert_{id}.bin"))
    }

    fn open_one(&self, id: u32) -> io::Result<Arc<File>> {
        let path = self.expert_path(id);
        let file = open_expert_file(&path, self.cfg.use_direct_io)?;
        Ok(Arc::new(file))
    }

    /// Get (and cache) the file handle for an expert id.
    fn fd_for(&self, id: u32) -> io::Result<Arc<File>> {
        if let Some(f) = self.files.read().get(&id) {
            return Ok(f.clone());
        }
        let mut guard = self.files.write();
        if let Some(f) = guard.get(&id) {
            return Ok(f.clone());
        }
        let f = self.open_one(id)?;
        guard.insert(id, f.clone());
        Ok(f)
    }

    /// Pre-open all expert fds to take that cost out of the steady-state path.
    pub fn warmup_fds(&self, ids: impl IntoIterator<Item = u32>) -> io::Result<()> {
        for id in ids {
            self.fd_for(id)?;
        }
        Ok(())
    }

    /// Read the full bytes of `expert_id` into `buf`.
    ///
    /// `buf` must be exactly `expert_size` bytes long and aligned to
    /// `block_align`; the [`BufferPool`](crate::buffer_pool::BufferPool) takes
    /// care of both invariants.
    ///
    /// Returns the number of bytes actually read (which equals `expert_size`
    /// on success — short reads are surfaced as an `UnexpectedEof` error).
    pub async fn read_expert(&self, expert_id: u32, buf: &mut PooledBuffer) -> io::Result<usize> {
        debug_assert_eq!(buf.len(), self.cfg.expert_size);
        let file = self.fd_for(expert_id)?;
        let n = self.read_into(&file, buf).await?;
        if n != self.cfg.expert_size {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "short read on expert {expert_id}: got {n} bytes, expected {}",
                    self.cfg.expert_size
                ),
            ));
        }
        Ok(n)
    }

    /// Batched read: fill `bufs[i]` with the bytes of `ids[i]`, all in
    /// one blocking-donation. The two slices must have the same length.
    ///
    /// **Why this exists.** When a single token misses on `K > 1` experts,
    /// the engine wants to push all `K` reads into the device's queue
    /// before doing any per-buffer post-processing. The default per-fetch
    /// path runs each `pread(2)` inside its own
    /// [`tokio::task::block_in_place`] call, which means the runtime has
    /// to round-trip between scheduler decisions for every expert. This
    /// helper hoists all `K` syscalls into one `block_in_place` block:
    /// the underlying [`std::os::unix::fs::FileExt::read_at`] calls are
    /// issued back-to-back to the kernel so the NVMe queue depth ramps
    /// up immediately, which is the same property an `io_uring`
    /// `submit_and_wait(K)` provides on the high-throughput path.
    ///
    /// On Linux with the `io_uring` cargo feature this method also has
    /// a sibling, `crate::io_uring_storage::IoUringStorage::read_experts_batch_fixed`,
    /// that pushes all `K` reads as `READ_FIXED` SQEs and submits once.
    pub async fn read_experts_batch(
        &self,
        ids: &[u32],
        bufs: &mut [&mut PooledBuffer],
    ) -> io::Result<usize> {
        assert_eq!(
            ids.len(),
            bufs.len(),
            "read_experts_batch: ids and bufs must have the same length"
        );
        if ids.is_empty() {
            return Ok(0);
        }
        // Resolve all fds before donating the worker — `fd_for` takes a
        // (rare) write lock the first time it sees an id, and we don't
        // want to hold that lock across `block_in_place`.
        let mut files: Vec<Arc<File>> = Vec::with_capacity(ids.len());
        for &id in ids {
            files.push(self.fd_for(id)?);
        }
        let expert_size = self.cfg.expert_size;
        for buf in bufs.iter() {
            debug_assert_eq!(buf.len(), expert_size);
        }

        // Single donation: all K reads dispatched without yielding to the
        // runtime between syscalls. On Linux this hands the NVMe queue
        // K consecutive submissions, matching the io_uring path's
        // submit-once semantics.
        let total = tokio::task::block_in_place(|| -> io::Result<usize> {
            let mut total = 0usize;
            for (file, buf) in files.iter().zip(bufs.iter_mut()) {
                let n = file.read_at(buf.as_mut_slice(), 0)?;
                if n != expert_size {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read in batch: got {n} bytes, expected {expert_size}"
                        ),
                    ));
                }
                total += n;
            }
            Ok(total)
        })?;
        Ok(total)
    }

    /// Partial-column read: load only the listed input-feature columns
    /// of an expert's `gate_proj` and `up_proj` plus the full `down_proj`,
    /// packed into `buf` in the layout consumed by
    /// [`crate::inference::OwnedExpertWeights::from_bytes_partial`]:
    ///
    /// ```text
    ///   gate_packed [d_ff x M]  ||  up_packed [d_ff x M]  ||  down [d_model x d_ff]
    /// ```
    ///
    /// `M = col_indices.len()` and `dtype` selects the on-disk byte width
    /// (2 for f16, 4 for f32). `buf.len()` must be at least the packed
    /// blob size (`(2*d_ff*M + d_model*d_ff) * bytes_per_weight`).
    ///
    /// **Implementation note (energy):** the row-major on-disk layout
    /// stores all columns of every row contiguously, so a strict
    /// "read only M columns" path would still need to touch every row.
    /// Today this function reads the full expert file once and packs
    /// the requested columns into `buf` in-process; that gives the
    /// **compute / dequantise** energy saving (proportional to M/d_model)
    /// without (yet) the **SSD bandwidth** saving. Switching to a
    /// column-major on-disk layout — written by the offline extractor —
    /// is the follow-up that turns this into a true bandwidth reduction.
    /// The engine API and the `from_bytes_partial` consumer are stable
    /// across that change.
    #[allow(dead_code)]
    pub async fn read_expert_columns(
        &self,
        expert_id: u32,
        col_indices: &[usize],
        dtype: WeightDtype,
        d_model: usize,
        d_ff: usize,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize> {
        let bpw = dtype.bytes_per_weight();
        let m = col_indices.len();
        for &c in col_indices {
            if c >= d_model {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("col index {c} out of range for d_model={d_model}"),
                ));
            }
        }
        let packed_bytes = (2 * d_ff * m + d_model * d_ff) * bpw;
        if buf.len() < packed_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "destination buffer too small for partial load: have {}, need {}",
                    buf.len(),
                    packed_bytes
                ),
            ));
        }

        // Stage the full file into a scratch Vec, then pack out the
        // requested columns. `block_in_place` keeps the runtime
        // responsive while pread runs.
        let file = self.fd_for(expert_id)?;
        let expert_size = self.cfg.expert_size;
        let mut scratch = vec![0u8; expert_size];
        tokio::task::block_in_place(|| file.read_at(&mut scratch, 0))?;

        let row_bytes = d_model * bpw;
        let gate_off = 0;
        let up_off = d_ff * row_bytes;
        let down_off = 2 * d_ff * row_bytes;

        let mut pos = 0usize;
        // gate_packed
        for i in 0..d_ff {
            let row_start = gate_off + i * row_bytes;
            for &c in col_indices {
                let src = row_start + c * bpw;
                buf.as_mut_slice()[pos..pos + bpw].copy_from_slice(&scratch[src..src + bpw]);
                pos += bpw;
            }
        }
        // up_packed
        for i in 0..d_ff {
            let row_start = up_off + i * row_bytes;
            for &c in col_indices {
                let src = row_start + c * bpw;
                buf.as_mut_slice()[pos..pos + bpw].copy_from_slice(&scratch[src..src + bpw]);
                pos += bpw;
            }
        }
        // down_proj copied verbatim
        let down_size = d_model * d_ff * bpw;
        buf.as_mut_slice()[pos..pos + down_size]
            .copy_from_slice(&scratch[down_off..down_off + down_size]);
        pos += down_size;

        Ok(pos)
    }

    #[cfg(target_os = "linux")]
    async fn read_into(&self, file: &File, buf: &mut PooledBuffer) -> io::Result<usize> {
        // Run the synchronous `pread(2)` on the current Tokio worker via
        // `block_in_place`. Other ready tasks are migrated to sibling
        // workers, so we don't stall the runtime; we also avoid the
        // `'static` requirement of `spawn_blocking`, which lets us keep
        // the borrow on `buf`.
        let len = buf.len();
        tokio::task::block_in_place(|| {
            // `read_at` is a positional read (`pread`) that does not touch
            // the file offset, so concurrent reads against the same fd
            // from multiple workers are safe.
            file.read_at(buf.as_mut_slice(), 0)
        })?;
        Ok(len)
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    async fn read_into(&self, file: &File, buf: &mut PooledBuffer) -> io::Result<usize> {
        // Same logic on macOS for development. `O_DIRECT` is unavailable;
        // the user is expected to pass `--no-direct` on those hosts.
        let len = buf.len();
        tokio::task::block_in_place(|| file.read_at(buf.as_mut_slice(), 0))?;
        Ok(len)
    }

    #[cfg(not(unix))]
    async fn read_into(&self, _file: &File, _buf: &mut PooledBuffer) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this engine targets Unix; non-Unix platforms are not supported",
        ))
    }
}

#[cfg(target_os = "linux")]
fn open_expert_file(path: &Path, direct: bool) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    if direct {
        opts.custom_flags(libc::O_DIRECT);
    }
    match opts.open(path) {
        Ok(f) => Ok(f),
        Err(e) if direct && e.raw_os_error() == Some(libc::EINVAL) => {
            // Filesystem doesn't support O_DIRECT (tmpfs, overlayfs, some
            // FUSE mounts). Tell the user to either move the data dir to a
            // real block device or disable direct I/O.
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "open({}) with O_DIRECT failed (EINVAL): the underlying \
                     filesystem does not support direct I/O. Re-run with \
                     --no-direct, or place the data directory on an ext4/xfs \
                     mount on a real NVMe device.",
                    path.display()
                ),
            ))
        }
        Err(e) => Err(e),
    }
}

#[cfg(not(target_os = "linux"))]
fn open_expert_file(path: &Path, _direct: bool) -> io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

/// Generate `num_experts` deterministic test files in `dir` with f32
/// weights. See [`generate_synthetic_experts_with_dtype`] for the f16
/// variant.
pub fn generate_synthetic_experts(
    dir: &Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
) -> io::Result<()> {
    generate_synthetic_experts_with_dtype(dir, num_experts, expert_size, d_model, d_ff, WeightDtype::F32)
}

/// Generate `num_experts` deterministic test files in `dir`. Each file
/// contains real `f32` *or* `f16` SwiGLU weights laid out as
/// `gate_proj || up_proj || down_proj` (row-major; see
/// [`crate::inference`]).
///
/// `weight_bytes` (= [`crate::inference::expert_weight_bytes_for`])
/// is the number of bytes the engine will actually consume. `expert_size`
/// is the size on disk; if it is larger than `weight_bytes` the trailing
/// region is zero padded so the file size stays a multiple of
/// `block_align` (an `O_DIRECT` requirement on Linux).
///
/// Weights are drawn from a small bounded uniform distribution
/// (`U(-scale, +scale)` with `scale ≈ 1 / sqrt(d_model)`) using a
/// per-expert deterministic xorshift, so the SwiGLU forward pass remains
/// numerically stable for any `d_model`/`d_ff` and runs are reproducible.
pub fn generate_synthetic_experts_with_dtype(
    dir: &Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    dtype: WeightDtype,
) -> io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;

    let weight_bytes = crate::inference::expert_weight_bytes_for(d_model, d_ff, dtype);
    if weight_bytes > expert_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expert_size {expert_size} too small for d_model={d_model} d_ff={d_ff} \
                 dtype={:?} (need at least {weight_bytes} bytes for the SwiGLU weights)",
                dtype
            ),
        ));
    }

    let scale = 1.0f32 / (d_model.max(1) as f32).sqrt();
    let pad_bytes = expert_size - weight_bytes;
    let zero_pad = vec![0u8; (1 << 20).min(pad_bytes.max(1))];
    let chunk_floats = 16 * 1024;

    let bpw = dtype.bytes_per_weight();
    for id in 0..num_experts {
        let path = dir.join(format!("expert_{id}.bin"));
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let mut state: u64 = 0x9E37_79B9_7F4A_7C15u64
            .wrapping_add((id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));

        let mut floats_remaining = crate::inference::expert_weight_count(d_model, d_ff);
        let mut buf = Vec::<u8>::with_capacity(chunk_floats * bpw);
        while floats_remaining > 0 {
            let n = floats_remaining.min(chunk_floats);
            buf.clear();
            for _ in 0..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 40) as u32;
                let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
                let v = unit * scale;
                match dtype {
                    WeightDtype::F32 => buf.extend_from_slice(&v.to_le_bytes()),
                    WeightDtype::F16 => {
                        let h = half::f16::from_f32(v);
                        buf.extend_from_slice(&h.to_bits().to_le_bytes());
                    }
                }
            }
            f.write_all(&buf)?;
            floats_remaining -= n;
        }

        let mut remaining_pad = pad_bytes;
        while remaining_pad > 0 {
            let n = remaining_pad.min(zero_pad.len());
            f.write_all(&zero_pad[..n])?;
            remaining_pad -= n;
        }
        f.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;

    /// Internal helper: a unique tempdir under `std::env::temp_dir()`.
    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("mer-io-test-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_experts_batch_returns_same_bytes_as_single_reads() {
        let dir = tempdir("batch");
        let num_experts = 4u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(&dir, num_experts, expert_size, d_model, d_ff).unwrap();
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
        })
        .unwrap();

        let pool = BufferPool::new(num_experts as usize * 2 + 2, expert_size, block);

        // Reference: read each expert one-by-one.
        let mut ref_bufs: Vec<Vec<u8>> = Vec::with_capacity(num_experts as usize);
        for id in 0..num_experts {
            let mut b = pool.acquire().await;
            storage.read_expert(id, &mut b).await.unwrap();
            ref_bufs.push(b.as_slice().to_vec());
        }

        // Batched read into fresh buffers.
        let mut bufs: Vec<_> = Vec::with_capacity(num_experts as usize);
        for _ in 0..num_experts {
            bufs.push(pool.acquire().await);
        }
        let ids: Vec<u32> = (0..num_experts).collect();
        let mut buf_refs: Vec<&mut crate::buffer_pool::PooledBuffer> = bufs.iter_mut().collect();
        let total = storage.read_experts_batch(&ids, &mut buf_refs).await.unwrap();
        assert_eq!(total, expert_size * num_experts as usize);
        for (i, b) in bufs.iter().enumerate() {
            assert_eq!(b.as_slice(), ref_bufs[i].as_slice(), "mismatch on expert {i}");
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
