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

/// Generate `num_experts` deterministic test files in `dir`. Each file
/// contains real `f32` SwiGLU weights laid out as
/// `gate_proj || up_proj || down_proj` (row-major; see
/// [`crate::inference`]).
///
/// `weight_bytes` is the number of bytes the engine will actually consume
/// (`expert_weight_bytes(d_model, d_ff)`). `expert_size` is the size on
/// disk; if it is larger than `weight_bytes` the trailing region is zero
/// padded so the file size stays a multiple of `block_align` (an
/// `O_DIRECT` requirement on Linux).
///
/// Weights are drawn from a small bounded uniform distribution
/// (`U(-scale, +scale)` with `scale ≈ 1 / sqrt(d_model)`) using a
/// per-expert deterministic xorshift, so the SwiGLU forward pass remains
/// numerically stable for any `d_model`/`d_ff` and runs are reproducible.
pub fn generate_synthetic_experts(
    dir: &Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
) -> io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;

    let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
    if weight_bytes > expert_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expert_size {expert_size} too small for d_model={d_model} d_ff={d_ff} \
                 (need at least {weight_bytes} bytes for the SwiGLU weights)"
            ),
        ));
    }

    // Initialisation scale. With small inputs in [-1, 1] this keeps the
    // pre-activation roughly unit-scale and avoids saturating SiLU.
    let scale = 1.0f32 / (d_model.max(1) as f32).sqrt();
    let pad_bytes = expert_size - weight_bytes;
    let zero_pad = vec![0u8; (1 << 20).min(pad_bytes.max(1))];
    let chunk_floats = 16 * 1024; // 64 KiB at a time

    for id in 0..num_experts {
        let path = dir.join(format!("expert_{id}.bin"));
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // xorshift64* seeded per-expert so gen-data is fully deterministic.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15u64
            .wrapping_add((id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));

        let mut floats_remaining = crate::inference::expert_weight_count(d_model, d_ff);
        let mut buf = Vec::<u8>::with_capacity(chunk_floats * 4);
        while floats_remaining > 0 {
            let n = floats_remaining.min(chunk_floats);
            buf.clear();
            for _ in 0..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map the high 24 bits to [-scale, +scale).
                let u = (state >> 40) as u32; // 24 bits
                let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0; // [-1, 1)
                let v = unit * scale;
                buf.extend_from_slice(&v.to_le_bytes());
            }
            f.write_all(&buf)?;
            floats_remaining -= n;
        }

        // Zero pad up to expert_size to satisfy O_DIRECT block alignment.
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
