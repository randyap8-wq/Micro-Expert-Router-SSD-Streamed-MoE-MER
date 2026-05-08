//! NVMe storage provider.
//!
//! On Linux this opens each expert file (optionally with `O_DIRECT`) at
//! startup, keeps the file descriptors resident, and submits async reads
//! through an [`rio`] io_uring instance into a [`PooledBuffer`] owned by the
//! caller. On non-Linux platforms it falls back to blocking `pread` on a
//! Tokio blocking thread, which keeps the rest of the engine portable for
//! development on macOS / WSL even though the hot path requires Linux.
//!
//! ## Why this layout?
//!
//! * **One ring, many fds.** A single io_uring SQ can drive thousands of
//!   in-flight reads. Opening files once removes per-read open() syscalls
//!   from the latency budget.
//! * **`O_DIRECT`.** Bypasses the page cache so we measure (and consume) raw
//!   NVMe bandwidth. Required by the spec; the buffer pool guarantees the
//!   alignment invariants the kernel checks.
//! * **Optional fallback.** `--no-direct` reverts to buffered reads, which is
//!   useful on tmpfs / overlayfs / non-Linux CI where `O_DIRECT` returns
//!   `EINVAL`. The engine still exercises the same prefetch + LRU logic.

use crate::buffer_pool::PooledBuffer;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
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

/// NVMe-backed storage with an io_uring backend on Linux.
pub struct NvmeStorage {
    cfg: StorageConfig,
    files: RwLock<HashMap<u32, Arc<File>>>,
    #[cfg(target_os = "linux")]
    ring: rio::Rio,
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

        #[cfg(target_os = "linux")]
        let ring = rio::new()?;

        Ok(Self {
            cfg,
            files: RwLock::new(HashMap::new()),
            #[cfg(target_os = "linux")]
            ring,
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
        // rio's `read_at` takes `&B` where `B: AsMut<[u8]>` (via `AsIoVecMut`)
        // and writes through the slice's raw pointer. We hold `&mut buf`
        // exclusively for the duration of the await, so no aliasing can
        // occur. The buffer is page-aligned and its length is a multiple of
        // the block size, which satisfies the `O_DIRECT` kernel preconditions.
        let completion = self.ring.read_at(file, &*buf, 0);
        completion.await
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    async fn read_into(&self, file: &File, buf: &mut PooledBuffer) -> io::Result<usize> {
        // Portable Unix fallback (macOS): synchronous `pread` directly.
        // This blocks the executor briefly and is *not* the high-performance
        // path — it exists so the engine still builds and runs end-to-end
        // on macOS for development.
        use std::os::unix::fs::FileExt;
        let len = buf.len();
        file.read_at(buf.as_mut_slice(), 0)?;
        Ok(len)
    }

    #[cfg(not(unix))]
    async fn read_into(&self, _file: &File, _buf: &mut PooledBuffer) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this engine targets Linux (io_uring) with a Unix fallback; \
             non-Unix platforms are not supported",
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

/// Generate `num_experts` deterministic test files in `dir`. Each file has
/// `expert_size` bytes and is filled with a per-expert byte pattern so reads
/// can be verified.
pub fn generate_synthetic_experts(
    dir: &Path,
    num_experts: u32,
    expert_size: usize,
) -> io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    // Write 1 MiB at a time to keep memory use bounded for large experts.
    let chunk_size = (1 << 20).min(expert_size);
    for id in 0..num_experts {
        let path = dir.join(format!("expert_{id}.bin"));
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let pattern = (id & 0xFF) as u8;
        let chunk = vec![pattern; chunk_size];
        let mut remaining = expert_size;
        while remaining > 0 {
            let n = remaining.min(chunk.len());
            f.write_all(&chunk[..n])?;
            remaining -= n;
        }
        f.flush()?;
    }
    Ok(())
}
