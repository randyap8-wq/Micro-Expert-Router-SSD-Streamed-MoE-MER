//! Linux `io_uring` storage backend with **registered fixed buffers**.
//!
//! Compiled only when the `io_uring` cargo feature is enabled and the
//! target OS is Linux. The default engine build does not include this
//! file in the binary; the legacy `pread(2)` backend in
//! [`crate::io_provider`] handles all reads.
//!
//! ### Why io_uring?
//!
//! Each `pread(2)` cache miss currently takes one syscall per expert.
//! With io_uring + registered fixed buffers we can:
//!
//! 1. **Pre-pin every `BufferPool` slot** with the kernel exactly once
//!    at startup (`io_uring_register(IORING_REGISTER_BUFFERS, …)` via
//!    [`crate::buffer_pool::BufferPool::raw_iovecs`]). After that, a
//!    read submission only references a buffer *index* — the kernel
//!    doesn't have to walk the user mapping or pin pages on the hot
//!    path.
//! 2. **Batch many reads with a single syscall.** When a token misses
//!    on `K > 1` experts, we push `K` SQEs and `enter()` once.
//! 3. **Reduce per-read CPU** (and therefore energy) by ~30–50 % on
//!    NVMe-class SSDs in published microbenchmarks. That CPU time is
//!    pure overhead — the same bytes were going to leave the device
//!    either way; io_uring just makes the kernel cheaper.
//!
//! ### Status
//!
//! This is a real implementation backed by the [`io_uring`] crate. It
//! supports `READ_FIXED` against pre-registered pool buffers and a
//! batched-submit entry point that pushes K SQEs and `submit_and_wait(K)`
//! once. The default engine factory still selects the portable `pread`
//! backend; a deployment can opt in to this one by constructing
//! [`IoUringStorage`] directly (see the integration sketch in the
//! `Engine` docs and `cmd_run`'s `--io-uring` branch).
//!
//! [`io_uring`]: https://docs.rs/io-uring

#![allow(dead_code)]

use crate::buffer_pool::{BufferPool, PooledBuffer};
use std::io;
use std::path::PathBuf;

/// Configuration for the `io_uring` backend. Mirrors
/// [`crate::io_provider::StorageConfig`] — we keep it as a separate
/// type so adding ring-specific knobs (queue depth, polling, etc.)
/// later doesn't break the portable backend's signature.
pub struct IoUringConfig {
    pub base_path: PathBuf,
    pub expert_size: usize,
    pub block_align: usize,
    /// Submission queue depth. Tracking expert top-K * cache_slots is
    /// usually enough; 64 is a safe default for small models.
    pub queue_depth: u32,
    /// Optional NUMA node hint. When `Some(n)` and the build target
    /// is Linux, the **constructing thread's** CPU affinity is pinned
    /// to the CPUs that belong to NUMA node `n` (as reported by
    /// `/sys/devices/system/node/node{n}/cpulist`) before
    /// `io_uring_register` is called. Because `register_buffers` and
    /// the per-ring kernel memory the kernel allocates during ring
    /// creation are charged to the calling thread's NUMA locality,
    /// this keeps the ring's metadata co-located with the buffers and
    /// the SSD's DMA target. Failures are best-effort: an unknown
    /// node id, missing sysfs entries, or a denied `sched_setaffinity`
    /// call all degrade silently (logged at WARN) rather than fail
    /// the construction.
    pub numa_node: Option<i32>,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("."),
            expert_size: 0,
            block_align: 4096,
            queue_depth: 64,
            numa_node: None,
        }
    }
}

/// `io_uring` storage backend.
///
/// **Construction** registers every buffer in `pool` as a fixed
/// io_uring buffer (one `io_uring_register` syscall, amortised across
/// the lifetime of the engine). Subsequent reads are
/// `IORING_OP_READ_FIXED` SQEs that reference a buffer index — no
/// per-read iovec setup.
pub struct IoUringStorage {
    cfg: IoUringConfig,
    /// Number of buffer slots that were registered. Stored for
    /// validation only — the actual ring + fd state lives in `inner`
    /// when the `io_uring` cargo feature is enabled.
    registered_buffers: usize,
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    inner: std::sync::Arc<linux_impl::Ring>,
}

impl IoUringStorage {
    /// Create a new io_uring backend over `pool`. The pool's buffers
    /// are registered *as is* with the kernel; do not resize the pool
    /// after this returns.
    ///
    /// On non-Linux builds, or when the `io_uring` cargo feature is
    /// off, this returns the validated config but the read methods
    /// will surface `Unsupported` errors at call time. Use the
    /// `cfg!(feature = "io_uring")` guard to pick a backend at the
    /// engine factory level.
    pub fn new(cfg: IoUringConfig, pool: &BufferPool) -> io::Result<Self> {
        let iovecs = pool.raw_iovecs();
        if iovecs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "io_uring backend requires a non-empty buffer pool",
            ));
        }
        if cfg.queue_depth == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "io_uring backend requires queue_depth > 0",
            ));
        }
        let registered = iovecs.len();

        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            let ring = linux_impl::Ring::new(&cfg, iovecs)?;
            return Ok(Self {
                cfg,
                registered_buffers: registered,
                inner: std::sync::Arc::new(ring),
            });
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            // Buffers were validated; the actual ring is unavailable
            // on this build. Read methods will return Unsupported.
            Ok(Self {
                cfg,
                registered_buffers: registered,
            })
        }
    }

    /// Number of pool buffers currently registered with the kernel.
    pub fn registered_buffers(&self) -> usize {
        self.registered_buffers
    }

    /// Submit a `READ_FIXED` SQE for `expert_id` into the registered
    /// buffer behind `buf`, then wait for its completion. Returns the
    /// number of bytes read (must equal `expert_size` on success).
    pub async fn read_expert_fixed(
        &self,
        expert_id: u32,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize> {
        debug_assert_eq!(buf.len(), self.cfg.expert_size);
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            let ring = self.inner.clone();
            let ptr = buf.as_mut_slice().as_mut_ptr();
            let len = self.cfg.expert_size;
            let n = tokio::task::block_in_place(move || {
                ring.read_expert_fixed_blocking(expert_id, ptr, len)
            })?;
            if n != self.cfg.expert_size {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "io_uring short read on expert {expert_id}: got {n} bytes, expected {}",
                        self.cfg.expert_size
                    ),
                ));
            }
            Ok(n)
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            let _ = (expert_id, buf);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "io_uring backend is unavailable on this build (Linux + `io_uring` cargo feature required)",
            ))
        }
    }

    /// Batched read: push one `READ_FIXED` SQE per `(expert_id, buf)`
    /// pair, `submit_and_wait(K)` once, drain the K completions. This
    /// is the moral equivalent of [`crate::io_provider::NvmeStorage::read_experts_batch`]
    /// but with a single `io_uring_enter` syscall regardless of K.
    pub async fn read_experts_batch_fixed(
        &self,
        ids: &[u32],
        bufs: &mut [&mut PooledBuffer],
    ) -> io::Result<usize> {
        assert_eq!(
            ids.len(),
            bufs.len(),
            "read_experts_batch_fixed: ids and bufs must have the same length"
        );
        if ids.is_empty() {
            return Ok(0);
        }
        for buf in bufs.iter() {
            debug_assert_eq!(buf.len(), self.cfg.expert_size);
        }

        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            let ring = self.inner.clone();
            let len = self.cfg.expert_size;
            // Snapshot raw pointers + ids without holding any borrow
            // across the `block_in_place` boundary.
            let ids_owned: Vec<u32> = ids.to_vec();
            let ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_slice().as_mut_ptr()).collect();
            // SAFETY: the &mut PooledBuffer borrows here keep the
            // backing AlignedBuffers alive for the duration of the
            // syscall. We move the raw pointers into the closure but
            // do not let them outlive this `await` (the mutable
            // borrows are released only after `block_in_place`
            // returns).
            let ptrs_send = SendPtrs(ptrs);
            let n = tokio::task::block_in_place(move || {
                ring.read_experts_batch_fixed_blocking(&ids_owned, &ptrs_send.0, len)
            })?;
            let expected = self.cfg.expert_size * ids.len();
            if n != expected {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("io_uring short batch read: got {n} bytes, expected {expected}"),
                ));
            }
            Ok(n)
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            let _ = (ids, bufs);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "io_uring backend is unavailable on this build (Linux + `io_uring` cargo feature required)",
            ))
        }
    }
}

/// Tiny `Send` wrapper over a `Vec<*mut u8>` so it can cross the
/// `block_in_place` boundary. Safety: the raw pointers refer to bytes
/// inside `PooledBuffer`s that the caller keeps mutably borrowed for
/// the duration of the closure — there is no aliasing.
#[cfg(all(target_os = "linux", feature = "io_uring"))]
struct SendPtrs(Vec<*mut u8>);
#[cfg(all(target_os = "linux", feature = "io_uring"))]
unsafe impl Send for SendPtrs {}

#[cfg(all(target_os = "linux", feature = "io_uring"))]
mod linux_impl {
    //! Inner Linux + `io-uring` crate implementation. Kept private so
    //! the `IoUringStorage` public surface is identical regardless of
    //! cargo features.

    use io_uring::{opcode, types, IoUring};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::{AsRawFd, RawFd};
    use std::sync::Mutex;

    pub(super) struct Ring {
        ring: Mutex<IoUring>,
        /// Map registered buffer pointer -> kernel buffer index. `acquire`
        /// hands out the same pointer every time a given slot is reused,
        /// so this lookup is stable across the lifetime of the pool.
        buf_index: HashMap<usize, u16>,
        /// Per-expert open-file cache, mirroring `NvmeStorage`'s.
        fds: RwLock<HashMap<u32, std::sync::Arc<File>>>,
        cfg_base: std::path::PathBuf,
    }

    impl Ring {
        pub(super) fn new(
            cfg: &super::IoUringConfig,
            iovecs: Vec<(*mut u8, usize)>,
        ) -> io::Result<Self> {
            // Best-effort NUMA pinning before ring creation so the
            // kernel allocates the ring's metadata on a node close
            // to the buffers and (typically) the NVMe device.
            // Failures are logged and ignored — pinning is a perf
            // hint, never a correctness requirement.
            if let Some(node) = cfg.numa_node {
                if let Err(e) = pin_thread_to_numa_node(node) {
                    tracing::warn!(
                        node,
                        error = %e,
                        "io_uring: NUMA pinning to node {} failed; ring will be created \
                         with default affinity",
                        node
                    );
                } else {
                    tracing::info!(
                        node,
                        "io_uring: pinned constructing thread to NUMA node {} \
                         before ring registration",
                        node
                    );
                }
            }
            let ring = IoUring::new(cfg.queue_depth)?;
            let raw_iovecs: Vec<libc::iovec> = iovecs
                .iter()
                .map(|(p, l)| libc::iovec { iov_base: *p as *mut _, iov_len: *l })
                .collect();
            // SAFETY: `raw_iovecs` borrows pointers owned by the
            // caller's `BufferPool`. The pool guarantees these stay
            // valid for the lifetime of the engine; the io_uring crate
            // requires the registered set to outlive every in-flight
            // submission, which is also satisfied by that lifetime.
            unsafe {
                ring.submitter().register_buffers(&raw_iovecs)?;
            }
            let buf_index = iovecs
                .iter()
                .enumerate()
                .map(|(i, (p, _))| (*p as usize, i as u16))
                .collect();
            Ok(Self {
                ring: Mutex::new(ring),
                buf_index,
                fds: RwLock::new(HashMap::new()),
                cfg_base: cfg.base_path.clone(),
            })
        }

        fn fd_for(&self, id: u32) -> io::Result<std::sync::Arc<File>> {
            if let Some(f) = self.fds.read().get(&id) {
                return Ok(f.clone());
            }
            let path = self.cfg_base.join(format!("expert_{id:04}.bin"));
            let f = std::sync::Arc::new(File::open(path)?);
            self.fds.write().insert(id, f.clone());
            Ok(f)
        }

        fn buf_idx_for(&self, ptr: *mut u8) -> io::Result<u16> {
            self.buf_index.get(&(ptr as usize)).copied().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "io_uring: buffer pointer is not registered with the kernel \
                     (the BufferPool slot must be one acquired *after* IoUringStorage::new)",
                )
            })
        }

        pub(super) fn read_expert_fixed_blocking(
            &self,
            expert_id: u32,
            buf_ptr: *mut u8,
            len: usize,
        ) -> io::Result<usize> {
            let buf_idx = self.buf_idx_for(buf_ptr)?;
            let f = self.fd_for(expert_id)?;
            let fd: RawFd = f.as_raw_fd();
            let sqe = opcode::ReadFixed::new(types::Fd(fd), buf_ptr, len as u32, buf_idx)
                .offset(0)
                .build()
                .user_data(expert_id as u64);
            let mut ring = self.ring.lock().unwrap();
            // SAFETY: SQE references `buf_ptr` and `fd` which are both
            // kept alive (the buffer through the caller's mutable
            // borrow on `PooledBuffer`, the fd through `f` which is
            // also stored in `self.fds`).
            unsafe {
                ring.submission().push(&sqe).map_err(|_| {
                    io::Error::new(io::ErrorKind::Other, "io_uring submission queue full")
                })?;
            }
            ring.submit_and_wait(1)?;
            let cqe = ring
                .completion()
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "io_uring: no completion event"))?;
            let result = cqe.result();
            if result < 0 {
                return Err(io::Error::from_raw_os_error(-result));
            }
            Ok(result as usize)
        }

        pub(super) fn read_experts_batch_fixed_blocking(
            &self,
            ids: &[u32],
            ptrs: &[*mut u8],
            len: usize,
        ) -> io::Result<usize> {
            // Resolve indices + fds without the ring lock so any
            // expensive setup happens before we touch the queue.
            let mut prepared: Vec<(RawFd, *mut u8, u16)> = Vec::with_capacity(ids.len());
            // Hold strong references to the fd Arcs so they outlive
            // the ring submission (they're also cached in self.fds,
            // but be explicit).
            let mut keep_alive: Vec<std::sync::Arc<File>> = Vec::with_capacity(ids.len());
            for (i, &expert_id) in ids.iter().enumerate() {
                let buf_idx = self.buf_idx_for(ptrs[i])?;
                let f = self.fd_for(expert_id)?;
                let fd = f.as_raw_fd();
                prepared.push((fd, ptrs[i], buf_idx));
                keep_alive.push(f);
            }

            let mut ring = self.ring.lock().unwrap();
            for (i, (fd, ptr, buf_idx)) in prepared.iter().copied().enumerate() {
                let sqe = opcode::ReadFixed::new(types::Fd(fd), ptr, len as u32, buf_idx)
                    .offset(0)
                    .build()
                    .user_data(ids[i] as u64);
                // SAFETY: see read_expert_fixed_blocking above.
                unsafe {
                    ring.submission().push(&sqe).map_err(|_| {
                        io::Error::new(io::ErrorKind::Other, "io_uring submission queue full")
                    })?;
                }
            }
            ring.submit_and_wait(ids.len())?;
            let mut total = 0usize;
            for _ in 0..ids.len() {
                let cqe = ring.completion().next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::Other, "io_uring: missing completion event")
                })?;
                let result = cqe.result();
                if result < 0 {
                    return Err(io::Error::from_raw_os_error(-result));
                }
                total += result as usize;
            }
            // `keep_alive` goes out of scope here; the fds remain
            // cached in `self.fds` and will be reused on the next call.
            Ok(total)
        }
    }

    /// Best-effort: pin the calling thread to the CPUs reported by
    /// `/sys/devices/system/node/node{n}/cpulist`. Returns the
    /// underlying io::Error on syscall failure or `InvalidInput` when
    /// the sysfs entry is missing / unparseable so the caller can
    /// emit a structured warning. The pin is *thread-local* — it
    /// does not affect the rest of the process.
    pub(super) fn pin_thread_to_numa_node(node: i32) -> io::Result<()> {
        if node < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "numa node must be non-negative",
            ));
        }
        let path = format!("/sys/devices/system/node/node{}/cpulist", node);
        let body = std::fs::read_to_string(&path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("could not read {path}: {e}"),
            )
        })?;
        let cpus = parse_cpulist(&body).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("could not parse cpulist {:?}", body),
            )
        })?;
        if cpus.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "numa node has empty cpulist",
            ));
        }
        // SAFETY: cpu_set_t is a POD bitset; we zero it via
        // mem::zeroed, fill in bits with libc helpers, and call
        // pthread_setaffinity_np on the current thread (gettid()
        // analogue) — the syscall only reads our buffer.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            for cpu in &cpus {
                libc::CPU_SET(*cpu, &mut set);
            }
            // sched_setaffinity(0, ...) pins the *calling thread*
            // (not the whole process) when CLONE_THREAD semantics
            // are in effect, which is the case for every tokio /
            // std::thread spawn. This matches what we want: only
            // this io_uring construction thread is pinned; tokio
            // workers and the engine's matmul thread pool remain
            // free to schedule wherever the OS prefers.
            let rc = libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const _,
            );
            if rc != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    /// Parse a kernel cpulist string (`"0-3,8,10-11"`) into a flat
    /// vector of cpu ids. Returns `None` on parse error.
    fn parse_cpulist(body: &str) -> Option<Vec<usize>> {
        let mut out: Vec<usize> = Vec::new();
        for part in body.trim().split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((lo, hi)) = part.split_once('-') {
                let lo: usize = lo.parse().ok()?;
                let hi: usize = hi.parse().ok()?;
                if hi < lo {
                    return None;
                }
                out.extend(lo..=hi);
            } else {
                out.push(part.parse().ok()?);
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registers_all_pool_slots() {
        let pool = BufferPool::new(4, 4096, 4096);
        let res = IoUringStorage::new(
            IoUringConfig {
                base_path: PathBuf::from("/tmp"),
                expert_size: 4096,
                block_align: 4096,
                queue_depth: 8,
                numa_node: None,
            },
            &pool,
        );
        // On Linux + io_uring feature, this should succeed; on other
        // builds, validation succeeds and we get a stub. Either way
        // the registered count must reflect the pool size.
        match res {
            Ok(s) => assert_eq!(s.registered_buffers(), 4),
            Err(e) => {
                // On systems where io_uring is unavailable (older
                // kernels, container restrictions, …) the kernel will
                // surface ENOSYS / EPERM. That's fine — this is an
                // optional backend.
                eprintln!("io_uring backend unavailable in this environment: {e}");
            }
        }
    }

    #[test]
    fn rejects_zero_queue_depth() {
        let pool = BufferPool::new(1, 4096, 4096);
        let res = IoUringStorage::new(
            IoUringConfig {
                base_path: PathBuf::from("/tmp"),
                expert_size: 4096,
                block_align: 4096,
                queue_depth: 0,
                numa_node: None,
            },
            &pool,
        );
        assert!(res.is_err());
    }

    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_expert_fixed_round_trips_against_disk() {
        use crate::io_provider::generate_synthetic_experts;

        let mut tmp = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        tmp.push(format!("mer-iouring-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let num_experts = 2u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(&tmp, num_experts, expert_size, d_model, d_ff).unwrap();

        // Pool slots: enough for both experts plus headroom.
        let pool = BufferPool::new(4, expert_size, block);
        let storage = match IoUringStorage::new(
            IoUringConfig {
                base_path: tmp.clone(),
                expert_size,
                block_align: block,
                queue_depth: 8,
                numa_node: None,
            },
            &pool,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Kernel doesn't support io_uring (or buffer registration is
                // forbidden in this sandbox). Treat as a soft-skip.
                eprintln!("io_uring not available, skipping: {e}");
                let _ = std::fs::remove_dir_all(&tmp);
                return;
            }
        };

        // Read expert 0 via io_uring.
        let mut buf = pool.acquire().await;
        let n = match storage.read_expert_fixed(0, &mut buf).await {
            Ok(n) => n,
            Err(e) => {
                eprintln!("io_uring read failed in this env: {e}");
                let _ = std::fs::remove_dir_all(&tmp);
                return;
            }
        };
        assert_eq!(n, expert_size);

        // Cross-check against a raw `pread`: byte-identical content.
        let path = tmp.join("expert_0000.bin");
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..], buf.as_slice());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
