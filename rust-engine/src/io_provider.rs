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
    /// Directory containing `expert_<id>.bin` files (or
    /// `expert_<layer>_<local_id>.bin` when [`Self::num_experts_per_layer`]
    /// is set; see also [`NvmeStorage::expert_path`]).
    pub base_path: PathBuf,
    /// Size (bytes) of every expert file. Must be a multiple of `block_align`.
    pub expert_size: usize,
    /// Logical block size to use for `O_DIRECT` alignment (typically 4096).
    pub block_align: usize,
    /// Whether to open files with `O_DIRECT` and bypass the page cache.
    pub use_direct_io: bool,
    /// Optional: number of experts per MoE layer in a multi-layer
    /// model. When `Some(n)` and `n > 0`, [`NvmeStorage::expert_path`]
    /// resolves the global expert id `g` to
    /// `expert_<g / n>_<g % n>.bin` whenever the legacy single-namespace
    /// `expert_<g>.bin` file does not exist. This makes the storage
    /// layer compatible with both the legacy GGUF-converter naming
    /// (single global namespace) **and** the multi-layer HF extractor
    /// naming (per-layer namespace) without requiring a second copy
    /// of the weight files. `None` (default) preserves the original
    /// single-namespace behaviour exactly.
    pub num_experts_per_layer: Option<u32>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::new(),
            expert_size: 0,
            block_align: 4096,
            use_direct_io: false,
            num_experts_per_layer: None,
        }
    }
}

/// NVMe-backed storage with a per-expert fd cache.
pub struct NvmeStorage {
    cfg: StorageConfig,
    files: RwLock<HashMap<u32, Arc<File>>>,
    /// Optional multi-drive layout. When non-empty, expert `id` lives at
    /// `extra_paths[id as usize % extra_paths.len()] / expert_<id>.bin`
    /// (with `cfg.base_path` *included* as `extra_paths[0]`). When
    /// empty, only `cfg.base_path` is consulted — the legacy single-
    /// drive layout. Gist Phase 4 (multi-drive striping).
    striped_paths: Vec<PathBuf>,
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
            striped_paths: Vec::new(),
        })
    }

    /// Construct a multi-drive striped storage. Experts are distributed
    /// across `dirs` by `id % dirs.len()`, so cache-miss reads issued
    /// concurrently for distinct expert ids hit *different* NVMe
    /// devices in the common case — the queue-depth advantage scales
    /// linearly with `dirs.len()` until the host PCIe link saturates.
    ///
    /// Layout invariants the rest of the engine relies on are
    /// preserved: every expert file is still `expert_size` bytes
    /// aligned to `block_align`, and `read_expert` returns the same
    /// `PooledBuffer` shape. The fd cache is shared across all drives.
    ///
    /// Compatible single-drive behaviour: when `dirs.len() == 1`, this
    /// is equivalent to [`Self::new`] with `cfg.base_path =
    /// dirs[0]`.
    pub fn striped(cfg: StorageConfig, dirs: Vec<PathBuf>) -> io::Result<Self> {
        if dirs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "NvmeStorage::striped requires at least one directory",
            ));
        }
        let mut cfg = cfg;
        cfg.base_path = dirs[0].clone();
        let mut s = Self::new(cfg)?;
        s.striped_paths = dirs;
        Ok(s)
    }

    /// Number of drives this storage is striped across. `1` for the
    /// legacy single-drive layout.
    pub fn num_drives(&self) -> usize {
        self.striped_paths.len().max(1)
    }

    pub fn config(&self) -> &StorageConfig {
        &self.cfg
    }

    /// Path of the file backing a given expert id.
    ///
    /// Resolution order:
    ///
    /// 1. The legacy single-namespace path
    ///    `<dir>/expert_<id>.bin` (compatible with the GGUF
    ///    converter and the engine's synthetic generators).
    /// 2. If [`StorageConfig::num_experts_per_layer`] is set and the
    ///    legacy file does not exist, the multi-layer extractor path
    ///    `<dir>/expert_<id / n>_<id % n>.bin` (matching
    ///    `scripts/extract_mixtral_experts.py`'s multi-layer dump).
    ///
    /// `<dir>` is `cfg.base_path` for the single-drive layout, or
    /// `striped_paths[id % n_drives]` for striped multi-drive layouts.
    pub fn expert_path(&self, id: u32) -> PathBuf {
        let dir = if self.striped_paths.is_empty() {
            self.cfg.base_path.clone()
        } else {
            let n = self.striped_paths.len();
            self.striped_paths[(id as usize) % n].clone()
        };
        let primary = dir.join(format!("expert_{id}.bin"));
        // Fast path: the legacy single-namespace file exists. Use
        // metadata() rather than try_exists() so we behave the same
        // as `open_one` would on permission / I/O errors (those still
        // bubble up to the caller).
        if std::fs::metadata(&primary).is_ok() {
            return primary;
        }
        // Multi-layer fallback: resolve to `expert_<layer>_<local>.bin`
        // when the operator told us the per-layer count.
        if let Some(n) = self.cfg.num_experts_per_layer {
            if n > 0 {
                let layer = id / n;
                let local = id % n;
                return dir.join(format!("expert_{layer}_{local}.bin"));
            }
        }
        primary
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
        //
        // We *must* return the byte count `read_at` reports, not
        // `buf.len()`: a truncated expert file (or any short read on a
        // network-mounted FS) would otherwise look like a full read and
        // the caller's "got `n` bytes, expected …" check would never
        // fire. See `read_expert` / `read_experts_batch` for the
        // surface-level validation that depends on this.
        tokio::task::block_in_place(|| {
            // `read_at` is a positional read (`pread`) that does not touch
            // the file offset, so concurrent reads against the same fd
            // from multiple workers are safe.
            file.read_at(buf.as_mut_slice(), 0)
        })
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    async fn read_into(&self, file: &File, buf: &mut PooledBuffer) -> io::Result<usize> {
        // Same logic on macOS for development. `O_DIRECT` is unavailable;
        // the user is expected to pass `--no-direct` on those hosts.
        // As on Linux, return the actual count from `read_at` so short
        // reads are surfaced — see the Linux branch's note.
        tokio::task::block_in_place(|| file.read_at(buf.as_mut_slice(), 0))
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

    for id in 0..num_experts {
        let path = dir.join(format!("expert_{id}.bin"));
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let mut state: u64 = 0x9E37_79B9_7F4A_7C15u64
            .wrapping_add((id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));

        // For INT8, write the 12-byte per-tensor scale header first.
        // Synthetic weights are drawn from `U(-scale, +scale)`, so the
        // per-tensor max-abs is bounded by `scale` and the symmetric
        // quantizer divisor is `scale / 127.0`. Real INT8-quantised
        // weights would compute these from the actual tensor maxima
        // during conversion (see `python/quantize_int8.py`).
        if matches!(dtype, WeightDtype::Int8) {
            let q = scale / 127.0;
            let meta = crate::inference::Int8ExpertMeta {
                gate_scale: q,
                up_scale: q,
                down_scale: q,
            };
            f.write_all(&meta.to_bytes())?;
        }

        let mut floats_remaining = crate::inference::expert_weight_count(d_model, d_ff);
        // Q4K writes per-block (144 bytes for 256 weights). For other
        // dtypes we write per-weight. Keep the per-weight RNG fed
        // through `state` so synthetic experts remain deterministic
        // across dtype choices for tests / golden runs.
        if matches!(dtype, WeightDtype::Q4K) {
            use crate::inference::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS};
            let mut block_floats = vec![0.0f32; Q4K_BLOCK_ELEMS];
            let mut block_bytes = vec![0u8; Q4K_BLOCK_BYTES];
            while floats_remaining > 0 {
                let n = floats_remaining.min(Q4K_BLOCK_ELEMS);
                for slot in block_floats.iter_mut().take(n) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 40) as u32;
                    let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
                    *slot = unit * scale;
                }
                // Pad the tail of the last block with zeros so its
                // dequant produces zeros for the unused tail slots.
                for slot in block_floats.iter_mut().skip(n) {
                    *slot = 0.0;
                }
                quantize_q4k_block_min_max(&block_floats, &mut block_bytes);
                f.write_all(&block_bytes)?;
                floats_remaining = floats_remaining.saturating_sub(n);
            }
        } else if matches!(dtype, WeightDtype::Q4_0) {
            // Q4_0 writes per-block (18 bytes for 32 weights), per-tensor.
            // Each of gate / up / down is rounded up to a 32-weight
            // boundary independently to match
            // [`expert_weight_bytes_for(_, _, Q4_0)`] and
            // [`OwnedExpertWeights::from_bytes_q4_0`].
            use crate::inference::{quantize_q4_0_block, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};
            let mut block_floats = vec![0.0f32; Q4_0_BLOCK_ELEMS];
            let mut block_bytes = vec![0u8; Q4_0_BLOCK_BYTES];
            // Three tensors of `d_model * d_ff` weights each.
            let one = d_model.saturating_mul(d_ff);
            for _tensor in 0..3 {
                let mut t_remaining = one;
                while t_remaining > 0 {
                    let n = t_remaining.min(Q4_0_BLOCK_ELEMS);
                    for slot in block_floats.iter_mut().take(n) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let u = (state >> 40) as u32;
                        let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
                        *slot = unit * scale;
                    }
                    for slot in block_floats.iter_mut().skip(n) {
                        *slot = 0.0;
                    }
                    quantize_q4_0_block(&block_floats[..Q4_0_BLOCK_ELEMS], &mut block_bytes);
                    f.write_all(&block_bytes)?;
                    t_remaining = t_remaining.saturating_sub(n);
                }
            }
            // The per-weight loop below writes nothing because
            // floats_remaining was set assuming the per-weight format;
            // null it out so we don't double-write.
            floats_remaining = 0;
        } else {
            let bpw = dtype.bytes_per_weight();
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
                        WeightDtype::Int8 => {
                            // Per-tensor symmetric quant. With the synthetic
                            // distribution and `q = scale/127.0`, `v / q`
                            // is in `[-127, +127]` so no clamp loss occurs;
                            // we still clamp defensively for robustness.
                            let q = scale / 127.0;
                            let qv = (v / q).round().clamp(-127.0, 127.0) as i8;
                            buf.push(qv as u8);
                        }
                        WeightDtype::Q4K => unreachable!("Q4K handled above"),
                        WeightDtype::Q4_0 => unreachable!("Q4_0 handled above"),
                    }
                }
                f.write_all(&buf)?;
                floats_remaining -= n;
            }
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

/// Quantise a 256-element block to GGUF Q4_K_M layout, writing 144
/// bytes into `dst`. The encoder uses simple per-sub-block min/max
/// clipping followed by 4-bit linear quantisation:
///
/// * super-block range `(lo, hi) = (min(x), max(x))`;
/// * `d = (hi - lo) / 63 / 15`,  `dmin = -lo / 63`;
/// * each sub-block's `scale6` is the 6-bit value that minimises the
///   sub-block's quantisation error against `d` (here we use the
///   block-wide max of |x - lo| -> 63 mapping for simplicity, which
///   is what the reference `ggml_quantize_q4_K_reference` does for a
///   minimum-effort encoder when no statistics are available);
/// * each sub-block's `min6` is `0` (no per-sub-block offset beyond
///   the global `dmin`).
///
/// This is a faithful inverse of [`crate::inference::dequantize_q4k_block`]
/// for the synthetic-weight regime: every weight produced by the
/// generator is bounded in `[-scale, +scale]`, so the simple
/// per-block fitting suffices and no per-sub-block bias correction
/// is needed for tests / golden runs to round-trip cleanly. A
/// production encoder (e.g. `python/quantize_q4k.py`) would solve
/// the per-sub-block 2-D least-squares problem; we don't do that
/// here because the synthetic distribution is uniform.
fn quantize_q4k_block_min_max(src: &[f32], dst: &mut [u8]) {
    use crate::inference::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q4K_SUBBLOCKS, Q4K_SUBBLOCK_ELEMS};
    debug_assert_eq!(src.len(), Q4K_BLOCK_ELEMS);
    debug_assert_eq!(dst.len(), Q4K_BLOCK_BYTES);

    // Find the super-block range.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in src {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        // Degenerate block: emit all zeros (which dequantises to 0).
        for b in dst.iter_mut() {
            *b = 0;
        }
        return;
    }
    // d * 63 * 15 = (hi - lo)  =>  d = (hi - lo) / 945.
    let denom = (hi - lo) / 945.0f32;
    let d_f16 = half::f16::from_f32(denom);
    // dmin * 63 = -lo  =>  dmin = -lo / 63.
    let dmin_f16 = half::f16::from_f32(-lo / 63.0);

    dst[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    dst[2..4].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());

    // Use a constant per-sub-block scale6 = 63 and min6 = 63 so the
    // dequant maps q4 in 0..15 to lo + (q4 / 15) * (hi - lo) — i.e.
    // the standard 4-bit linear quantiser scaled across the whole
    // super-block. (sub_scale=63 ensures q4=15 -> hi; min6=63 with
    // dmin = -lo/63 contributes -dmin * 63 = lo.)
    //
    // Pack the 8 (scale6, min6) = (63, 63) pairs.
    let pairs = [(63u8, 63u8); Q4K_SUBBLOCKS];
    let s = q4k_pack_scales_local(&pairs);
    dst[4..16].copy_from_slice(&s);

    // Quantise each weight: q4 = round((v - lo) / (hi - lo) * 15).
    // (Using the block-wide range, since every sub_scale equals 63
    // and dmin*min6 == lo.)
    let inv_range = 15.0f32 / (hi - lo);
    let qs = &mut dst[16..16 + 128];
    for j in 0..Q4K_SUBBLOCKS {
        let qs_off = j * (Q4K_SUBBLOCK_ELEMS / 2);
        for i in 0..Q4K_SUBBLOCK_ELEMS {
            let v = src[j * Q4K_SUBBLOCK_ELEMS + i];
            let q = ((v - lo) * inv_range).round().clamp(0.0, 15.0) as u8;
            let byte_idx = qs_off + (i >> 1);
            if i & 1 == 0 {
                qs[byte_idx] = (qs[byte_idx] & 0xF0) | (q & 0x0F);
            } else {
                qs[byte_idx] = (qs[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
            }
        }
    }
}

/// Local copy of the inference module's q4k scale packer; kept here to
/// avoid making the inference helper `pub`. Mirrors the bit layout
/// described in [`crate::inference::dequantize_q4k_block`].
fn q4k_pack_scales_local(pairs: &[(u8, u8); 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..4 {
        s[j] = pairs[j].0 & 0x3F;
        s[j + 4] = pairs[j].1 & 0x3F;
    }
    for j in 4..8 {
        let (scale_j, min_j) = pairs[j];
        let scale_j = scale_j & 0x3F;
        let min_j = min_j & 0x3F;
        s[j + 4] = (scale_j & 0x0F) | ((min_j & 0x0F) << 4);
        s[j - 4] = (s[j - 4] & 0x3F) | (((scale_j >> 4) & 0x03) << 6);
        s[j] = (s[j] & 0x3F) | (((min_j >> 4) & 0x03) << 6);
    }
    s
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

    /// Striping smoke test: when `NvmeStorage::striped` is constructed
    /// with N directories, `expert_path(id)` selects directory
    /// `id % N`, and `read_expert` returns the same bytes the file
    /// behind that path contains.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn striped_storage_shards_by_id_modulo_drives() {
        let d0 = tempdir("stripe-a");
        let d1 = tempdir("stripe-b");
        let num_experts = 4u32;
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = weight_bytes.div_ceil(block) * block;
        // Even ids -> d0; odd ids -> d1.
        for id in 0..num_experts {
            let dir = if id % 2 == 0 { &d0 } else { &d1 };
            let path = dir.join(format!("expert_{id}.bin"));
            // Distinct fill byte per expert so reads can be verified.
            let mut blob = vec![0u8; expert_size];
            for b in blob.iter_mut() {
                *b = (id as u8).wrapping_add(0x10);
            }
            std::fs::write(&path, &blob).unwrap();
        }
        let storage = NvmeStorage::striped(
            StorageConfig {
                base_path: d0.clone(),
                expert_size,
                block_align: block,
                use_direct_io: false, num_experts_per_layer: None,
            },
            vec![d0.clone(), d1.clone()],
        )
        .unwrap();
        assert_eq!(storage.num_drives(), 2);
        // expert_0 / expert_2 must resolve to d0; expert_1 / expert_3 to d1.
        assert_eq!(storage.expert_path(0), d0.join("expert_0.bin"));
        assert_eq!(storage.expert_path(1), d1.join("expert_1.bin"));
        assert_eq!(storage.expert_path(2), d0.join("expert_2.bin"));
        assert_eq!(storage.expert_path(3), d1.join("expert_3.bin"));

        let pool = BufferPool::new(num_experts as usize, expert_size, block);
        for id in 0..num_experts {
            let mut buf = pool.acquire().await;
            storage.read_expert(id, &mut buf).await.unwrap();
            assert_eq!(buf.as_slice()[0], (id as u8).wrapping_add(0x10));
        }
        let _ = std::fs::remove_dir_all(&d0);
        let _ = std::fs::remove_dir_all(&d1);
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
            use_direct_io: false, num_experts_per_layer: None,
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_layer_naming_fallback_resolves_layered_files() {
        // The multi-layer HF extractor writes `expert_<layer>_<id>.bin`;
        // the storage layer must transparently resolve a global expert
        // id to the corresponding `(layer, local)` file when the
        // legacy single-namespace file is missing.
        let dir = tempdir("multilayer");
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        // 2 layers × 3 experts. Write only the layered names — the
        // legacy `expert_<global>.bin` files do *not* exist.
        let n = 3u32;
        for layer in 0..2u32 {
            for local in 0..n {
                let path = dir.join(format!("expert_{layer}_{local}.bin"));
                let mut blob = vec![0u8; expert_size];
                for b in blob.iter_mut() {
                    *b = ((layer * n + local) as u8).wrapping_add(0x40);
                }
                std::fs::write(&path, &blob).unwrap();
            }
        }
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: Some(n),
        })
        .unwrap();
        // Global id → (id/n, id%n).
        assert_eq!(storage.expert_path(0), dir.join("expert_0_0.bin"));
        assert_eq!(storage.expert_path(2), dir.join("expert_0_2.bin"));
        assert_eq!(storage.expert_path(3), dir.join("expert_1_0.bin"));
        assert_eq!(storage.expert_path(5), dir.join("expert_1_2.bin"));
        // And the bytes round-trip through `read_expert`, which is
        // what the engine's miss path actually calls.
        let pool = BufferPool::new(2, expert_size, block);
        let mut buf = pool.acquire().await;
        let bytes = storage.read_expert(4, &mut buf).await.unwrap();
        assert_eq!(bytes, expert_size);
        // Layer=1, local=1 → fill byte = (1*3 + 1) + 0x40 = 0x44.
        assert_eq!(buf.as_slice()[0], 0x44);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
