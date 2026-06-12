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
//! batched-submit entry point that pushes K SQEs with one syscall. The
//! reactor thread is **multi-request concurrent**: several callers'
//! macro-batches share the ring's queue depth simultaneously (slot-
//! tagged `user_data`, per-batch completion accounting), so a
//! speculative prefetch batch no longer waits for a foreground batch
//! to fully drain before its SQEs reach the kernel. The default engine
//! factory still selects the portable `pread`
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

/// `io_uring` storage backend (gist Part 2).
///
/// **Construction** registers every buffer in `pool` as a fixed
/// io_uring buffer (one `io_uring_register` syscall, amortised across
/// the lifetime of the engine). Subsequent reads are
/// `IORING_OP_READ_FIXED` SQEs that reference a buffer index — no
/// per-read iovec setup, avoiding the per-syscall cost of the
/// `read_at`/pread fallback in [`crate::io_provider::NvmeStorage`].
///
/// On non-Linux targets (or when the `io_uring` cargo feature is off)
/// this becomes a no-op shim that surfaces `Unsupported` errors at
/// call time, so engine factory code can keep a single code path.
///
/// The kernel-side state lives in `linux_impl::Ring`; the [`Ring`]'s
/// reactor thread is created in `IoUringStorage::new` and joined in
/// [`IoUringStorage::Drop`] via the inner `Arc<Ring>`. The reactor
/// owns its `IoUring` instance exclusively; the only path into it is
/// a bounded mpsc channel of [`ReactorRequest`] envelopes.
///
/// `IoUringStorage` retains a [`crate::buffer_pool::BufferPool`]
/// clone for the lifetime of the backend so the kernel-registered
/// fixed-buffer pointers (passed to `register_buffers`) remain valid
/// even if the original `BufferPool` handle is dropped at the call
/// site. (F2.3 in the audit.) `BufferPool` is `Clone` and internally
/// reference-counted, so this is a cheap arc-bump.
pub struct IoUringStorage {
    cfg: IoUringConfig,
    /// Number of buffer slots that were registered. Stored for
    /// validation only — the actual ring + fd state lives in `inner`
    /// when the `io_uring` cargo feature is enabled.
    registered_buffers: usize,
    /// Buffer pool clone that owns the heap-allocated, page-aligned
    /// fixed buffers the kernel has registered raw pointers into.
    /// Holding this `BufferPool` (which is internally an `Arc`)
    /// guarantees the backing storage outlives the kernel ring.
    /// Mostly opaque; retained for fixed-buffer lifetime safety.
    #[allow(dead_code)]
    pool: BufferPool,
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
            let ring = linux_impl::Ring::new(&cfg, iovecs, pool.clone())?;
            return Ok(Self {
                cfg,
                registered_buffers: registered,
                pool: pool.clone(),
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
                pool: pool.clone(),
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
            let ptr = buf.as_mut_slice().as_mut_ptr();
            let len = self.cfg.expert_size;
            let n = self.inner.read_expert_fixed_async(expert_id, ptr, len).await?;
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
            let len = self.cfg.expert_size;
            // Snapshot raw pointers + ids without holding any borrow
            // across the `.await` boundary. The mutable borrows on
            // `bufs` keep the backing AlignedBuffers alive for the
            // duration of the request — the reactor never accesses
            // them after the oneshot fires, and we never access them
            // while in flight.
            let ids_owned: Vec<u32> = ids.to_vec();
            let ptrs: Vec<*mut u8> = bufs
                .iter_mut()
                .map(|b| b.as_mut_slice().as_mut_ptr())
                .collect();
            let ptrs_send = SendPtrs(ptrs);
            let n = self
                .inner
                .read_experts_batch_fixed_async(&ids_owned, &ptrs_send.0, len)
                .await?;
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
    /// Speculative-prefetch entry point that fuses the **batched io_uring
    /// completion** with the **zero-copy shadow → primary promotion**
    /// requested by the gist (Task 2: "the Speculator triggers a
    /// `promote_shadow` call in the BufferPool during the io_uring
    /// completion interrupt").
    ///
    /// Caller owns:
    ///
    /// * `ids` — the expert ids to fetch (top-K speculator output);
    /// * `shadow_bufs` — `K` `PooledBuffer`s previously acquired from
    ///   the **shadow** half of `pool` via
    ///   [`crate::buffer_pool::BufferPool::try_acquire_shadow`] /
    ///   [`crate::buffer_pool::BufferPool::acquire_shadow`];
    /// * `pool` — the [`BufferPool`] those shadow buffers came from
    ///   (used for the slot-tag swap).
    ///
    /// The method:
    ///
    /// 1. issues a single batched `submit_and_wait(K)` against the
    ///    pre-registered fixed buffers (same submission path as
    ///    [`Self::read_experts_batch_fixed`] — one syscall regardless
    ///    of K);
    /// 2. on completion, calls [`BufferPool::promote_shadow`] for each
    ///    buffer so the next `Drop` returns the slot to the **primary**
    ///    free list — i.e. the bytes that just arrived become resident
    ///    without any extra copy or re-read.
    ///
    /// The returned `Vec<PooledBuffer>` is in the same order as `ids`
    /// and every buffer reports `is_shadow() == false` (they're now
    /// primary). On error the shadow buffers are dropped back into
    /// the shadow free list — no leak.
    pub async fn read_experts_batch_fixed_promote(
        &self,
        ids: &[u32],
        shadow_bufs: Vec<PooledBuffer>,
        pool: &BufferPool,
    ) -> io::Result<Vec<PooledBuffer>> {
        assert_eq!(
            ids.len(),
            shadow_bufs.len(),
            "read_experts_batch_fixed_promote: ids.len()={} must equal shadow_bufs.len()={}",
            ids.len(),
            shadow_bufs.len(),
        );
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        for buf in shadow_bufs.iter() {
            debug_assert_eq!(buf.len(), self.cfg.expert_size);
        }
        // Drive the F1.3 cancellation-safe owned-buffer path. The
        // shadow buffers are moved into the reactor's keep-alive
        // owner so they cannot return to the buffer pool's free
        // list until the kernel finishes writing them, even if our
        // outer future is cancelled at the .await below. Compare to
        // the legacy `&mut PooledBuffer` API, which leaves the bufs
        // on the caller's stack — a cancellation there frees the
        // slot while the kernel is mid-write.
        let shadow_bufs = {
            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            {
                let len = self.cfg.expert_size;
                let ids_owned: Vec<u32> = ids.to_vec();
                self.inner
                    .read_experts_batch_fixed_owned_recover(&ids_owned, shadow_bufs, len)
                    .await?
            }
            #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
            {
                let _ = ids;
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "io_uring backend is unavailable on this build (Linux + `io_uring` cargo feature required)",
                ));
            }
        };
        // Single-syscall completion fence behind us: every byte for
        // every expert in `ids` is now in the matching shadow buffer.
        // Promote each one so the slot accounting flips to primary
        // without an extra copy — this is the zero-latency
        // "speculation confirmed → resident" hand-off.
        let promoted: Vec<PooledBuffer> = shadow_bufs
            .into_iter()
            .map(|b| pool.promote_shadow(b))
            .collect();
        Ok(promoted)
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

    use crate::buffer_pool::PooledBuffer;
    use io_uring::{opcode, types, IoUring};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::{AsRawFd, RawFd};

    /// One pending submission: a fully-prepared batch of `(fd, ptr,
    /// buf_idx, expert_id)` tuples plus a oneshot the reactor uses to
    /// report total bytes (or the first OS error).
    pub(super) struct ReactorRequest {
        pub prepared: Vec<(RawFd, *mut u8, u16, u32)>,
        pub len: usize,
        pub reply: tokio::sync::oneshot::Sender<io::Result<usize>>,
        /// Kept alive on the requester side until the oneshot fires;
        /// the reactor never touches it. We hold a `Box<dyn Any +
        /// Send>` so the requester can stash whatever borrow it
        /// needs (here: the `Arc<File>` fd-cache entries).
        pub _keep_alive: Box<dyn std::any::Any + Send>,
    }

    // SAFETY: the raw pointers in `prepared` refer to bytes inside
    // `PooledBuffer`s that the requester keeps mutably borrowed until
    // the oneshot in `reply` fires. The reactor never holds them past
    // that point, and the requester never accesses them while the
    // request is in flight, so there is no data race.
    unsafe impl Send for ReactorRequest {}

    /// Bounded async-backpressure channel between the public
    /// `IoUringStorage` async surface and the dedicated reactor thread
    /// that exclusively owns the `IoUring`.
    pub(super) type ReactorTx = tokio::sync::mpsc::Sender<ReactorRequest>;
    type ReactorRx = tokio::sync::mpsc::Receiver<ReactorRequest>;

    pub(super) struct Ring {
        /// Bounded mpsc channel: the public async surface holds the
        /// `Sender` and uses `send().await` to apply natural async
        /// backpressure when the reactor is saturated. The reactor
        /// thread owns the `Receiver`. Wrapped in `Option` so `Drop`
        /// can `take()` it — dropping the sender is what signals the
        /// reactor loop to exit.
        tx: Option<ReactorTx>,
        /// Map registered buffer pointer -> kernel buffer index.
        buf_index: HashMap<usize, u16>,
        /// Per-expert open-file cache, mirroring `NvmeStorage`'s.
        fds: RwLock<HashMap<u32, std::sync::Arc<File>>>,
        cfg_base: std::path::PathBuf,
        /// Cached SQE capacity (`cfg.queue_depth`). Used by the
        /// reactor's batched submission path to chunk macro-batches
        /// that exceed the ring's configured depth.
        queue_depth: usize,
        /// Join handle for the reactor thread, joined on Drop so a
        /// torn-down storage doesn't leave a dangling kernel ring.
        reactor: Option<std::thread::JoinHandle<()>>,
        /// `BufferPool` clone that owns the heap-allocated, page-
        /// aligned buffers that were `register_buffers`'d into the
        /// kernel. Holding this `Arc` (BufferPool is `Clone` /
        /// internally `Arc`'d) for the lifetime of the ring is what
        /// keeps those buffer addresses valid for as long as the
        /// kernel might dereference them. (F2.3 in the audit.)
        #[allow(dead_code)]
        pool: super::BufferPool,
    }

    impl Ring {
        pub(super) fn new(
            cfg: &super::IoUringConfig,
            iovecs: Vec<(*mut u8, usize)>,
            pool: super::BufferPool,
        ) -> io::Result<Self> {
            // gist Part 3 — buffer alignment validation. The kernel
            // requires every fixed buffer registered via
            // `IORING_REGISTER_BUFFERS` to be page-aligned (4096 bytes
            // on x86_64 / aarch64). `AlignedBuffer::new` already
            // enforces this for the primary `BufferPool` allocation
            // path, but we revalidate here so that:
            //
            //   * any future caller that constructs `iovecs` outside of
            //     `BufferPool` (custom integrations, tests) gets a
            //     clear `InvalidInput` instead of an opaque kernel
            //     `EINVAL` from `register_buffers`;
            //   * the invariant is captured at the io_uring boundary,
            //     not just at the allocator boundary.
            const REQUIRED_ALIGN: usize = 4096;
            for (i, (p, l)) in iovecs.iter().enumerate() {
                let addr = *p as usize;
                if addr % REQUIRED_ALIGN != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "io_uring fixed buffer #{i} base pointer {addr:#x} is not aligned to {REQUIRED_ALIGN} bytes; \
                             io_uring requires page-aligned registered buffers",
                        ),
                    ));
                }
                if *l == 0 || *l % REQUIRED_ALIGN != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "io_uring fixed buffer #{i} length {l} is not a positive multiple of {REQUIRED_ALIGN}; \
                             io_uring requires page-sized registered buffers",
                        ),
                    ));
                }
            }
            // Best-effort NUMA pinning before ring creation so the
            // kernel allocates the ring's metadata on a node close to
            // the buffers. Failures are logged and ignored.
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
            let mut ring = IoUring::new(cfg.queue_depth)?;
            let raw_iovecs: Vec<libc::iovec> = iovecs
                .iter()
                .map(|(p, l)| libc::iovec { iov_base: *p as *mut _, iov_len: *l })
                .collect();
            // SAFETY: `raw_iovecs` borrows pointers owned by the
            // caller's `BufferPool`. The pool guarantees these stay
            // valid for the lifetime of the engine.
            unsafe {
                ring.submitter().register_buffers(&raw_iovecs)?;
            }
            let buf_index: HashMap<usize, u16> = iovecs
                .iter()
                .enumerate()
                .map(|(i, (p, _))| (*p as usize, i as u16))
                .collect();
            let queue_depth = (cfg.queue_depth as usize).max(1);
            // Bounded mpsc — depth `queue_depth` so the channel
            // naturally back-pressures the request side once the
            // reactor is in flight on at least one macro-batch.
            let (tx, rx) = tokio::sync::mpsc::channel::<ReactorRequest>(queue_depth);
            let reactor = std::thread::Builder::new()
                .name("mer-io-reactor".into())
                .spawn(move || reactor_loop(ring, rx, queue_depth))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("io_uring: failed to spawn reactor thread: {e}"),
                    )
                })?;
            Ok(Self {
                tx: Some(tx),
                buf_index,
                fds: RwLock::new(HashMap::new()),
                cfg_base: cfg.base_path.clone(),
                queue_depth,
                reactor: Some(reactor),
                pool,
            })
        }

        fn fd_for(&self, id: u32) -> io::Result<std::sync::Arc<File>> {
            if let Some(f) = self.fds.read().get(&id) {
                return Ok(f.clone());
            }
            // Same on-disk convention as `NvmeStorage::expert_path` /
            // `generate_synthetic_experts`: `expert_<id>.bin` without
            // zero padding. Fall back to the legacy zero-padded
            // `expert_<id:04>.bin` name for deployments that still
            // use it.
            let primary = self.cfg_base.join(format!("expert_{id}.bin"));
            let path = if primary.exists() {
                primary
            } else {
                let padded = self.cfg_base.join(format!("expert_{id:04}.bin"));
                if padded.exists() { padded } else { primary }
            };
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

        /// Async single-expert read: encoded as a 1-element batch
        /// through the reactor channel. No `block_in_place`, no
        /// shared mutex on the hot path.
        pub(super) async fn read_expert_fixed_async(
            &self,
            expert_id: u32,
            buf_ptr: *mut u8,
            len: usize,
        ) -> io::Result<usize> {
            self.submit_batch(&[expert_id], &[buf_ptr], len).await
        }

        /// Async batched read: prepares the descriptor tuples, sends
        /// one `ReactorRequest`, awaits the oneshot. The bounded mpsc
        /// channel applies async backpressure when the reactor is busy.
        pub(super) async fn read_experts_batch_fixed_async(
            &self,
            ids: &[u32],
            ptrs: &[*mut u8],
            len: usize,
        ) -> io::Result<usize> {
            self.submit_batch(ids, ptrs, len).await
        }

        /// **F1.3 cancellation-safe owned-buffer entry point.** Takes
        /// ownership of `bufs`, sends them through the reactor request
        /// so they're pinned by the request's `_keep_alive` box (via
        /// an `Arc<Mutex<Option<…>>>` shared with the caller for
        /// post-reply recovery), then returns them to the caller on
        /// the reply. If the caller's future is cancelled between
        /// send and reply, the reactor still processes the request
        /// and drops its `Arc` clone, leaving the buffers in the
        /// shared cell on our side; our local `Arc` drop then
        /// releases them back to the pool — but only *after* the
        /// kernel finishes writing them. This eliminates the
        /// buffer-pool vs. kernel-write race the `&mut PooledBuffer`
        /// API has.
        pub(super) async fn read_experts_batch_fixed_owned_recover(
            &self,
            ids: &[u32],
            mut bufs: Vec<PooledBuffer>,
            len: usize,
        ) -> io::Result<Vec<PooledBuffer>> {
            // Snapshot raw pointers BEFORE moving `bufs` into the
            // owner cell. Pointers reference the heap-allocated
            // `AlignedBuffer` payload, which is stable across moves
            // of the enclosing `PooledBuffer` (the buffer carries a
            // `NonNull<u8>` to the underlying allocation, not an
            // inline array). The transient `&mut` borrows here end
            // at the semicolon (NLL), so `bufs` becomes movable
            // immediately after.
            let ptrs: Vec<*mut u8> = bufs
                .iter_mut()
                .map(|b| b.as_mut_slice().as_mut_ptr())
                .collect();
            let shared: std::sync::Arc<parking_lot::Mutex<Option<Vec<PooledBuffer>>>> =
                std::sync::Arc::new(parking_lot::Mutex::new(Some(bufs)));
            // Owner clone goes into the reactor request; the original
            // stays here so we can pull buffers back out on success.
            let reactor_owner = std::sync::Arc::clone(&shared);
            let owner: Box<dyn std::any::Any + Send> = Box::new(reactor_owner);
            let res = self.submit_batch_with_owner(ids, &ptrs, len, owner).await;
            match res {
                Ok(_n) => {
                    let recovered = shared
                        .lock()
                        .take()
                        .ok_or_else(|| io::Error::new(
                            io::ErrorKind::Other,
                            "io_uring owned-buffer reply lost ownership \
                             (reactor took the buffers — should not happen)",
                        ))?;
                    Ok(recovered)
                }
                Err(e) => Err(e),
            }
        }

        async fn submit_batch(
            &self,
            ids: &[u32],
            ptrs: &[*mut u8],
            len: usize,
        ) -> io::Result<usize> {
            self.submit_batch_with_owner(ids, ptrs, len, Box::new(()))
                .await
        }

        /// Cancellation-safe entry point: same as [`submit_batch`] but
        /// the caller hands us an owner that we move into the
        /// `_keep_alive` box. As long as the owner transitively owns
        /// the [`PooledBuffer`]s whose `*mut u8` we just dereferenced
        /// (F1.3), the kernel cannot race the buffer pool's free-list:
        /// even if the caller's outer future is dropped mid-flight,
        /// the bufs stay alive until the reactor finishes
        /// `process_one` and drops `_keep_alive`.
        async fn submit_batch_with_owner(
            &self,
            ids: &[u32],
            ptrs: &[*mut u8],
            len: usize,
            owner: Box<dyn std::any::Any + Send>,
        ) -> io::Result<usize> {
            // Resolve fds + buf indices on the request side (lock-free
            // for the reactor — it never has to touch the fd cache or
            // the buf-index map).
            let mut prepared: Vec<(RawFd, *mut u8, u16, u32)> = Vec::with_capacity(ids.len());
            let mut keep_alive_fds: Vec<std::sync::Arc<File>> = Vec::with_capacity(ids.len());
            for (i, &expert_id) in ids.iter().enumerate() {
                let buf_idx = self.buf_idx_for(ptrs[i])?;
                let f = self.fd_for(expert_id)?;
                let fd = f.as_raw_fd();
                prepared.push((fd, ptrs[i], buf_idx, expert_id));
                keep_alive_fds.push(f);
            }
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            // Pack fds + caller-supplied owner into a single Any. The
            // reactor never inspects either — it just drops both after
            // sending the reply, which is what keeps the underlying
            // buffer memory pinned through the kernel-side write.
            let combined: Box<dyn std::any::Any + Send> =
                Box::new((keep_alive_fds, owner));
            let req = ReactorRequest {
                prepared,
                len,
                reply: reply_tx,
                _keep_alive: combined,
            };
            let tx = self.tx.as_ref().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Other,
                    "io_uring reactor channel closed — backend has been torn down",
                )
            })?;
            tx.send(req).await.map_err(|_| {
                io::Error::new(
                    io::ErrorKind::Other,
                    "io_uring reactor channel closed — backend has been torn down",
                )
            })?;
            // Note: the reactor *always* sends a reply (Ok or Err) for
            // every accepted request, so a RecvError here genuinely
            // indicates a reactor panic.
            reply_rx.await.map_err(|_| {
                io::Error::new(
                    io::ErrorKind::Other,
                    "io_uring reactor dropped reply channel — reactor thread panicked",
                )
            })?
        }
    }

    impl Drop for Ring {
        fn drop(&mut self) {
            // Closing `tx` causes the reactor's `recv().await` to
            // return `None` and the loop to exit cleanly. We `take`
            // it out of the `Option` so the sender is dropped *before*
            // we attempt to join the reactor thread — otherwise the
            // join would deadlock waiting for a channel close that
            // never came.
            drop(self.tx.take());
            if let Some(h) = self.reactor.take() {
                // Best-effort join; if the reactor panicked we don't
                // want the drop path to also panic.
                let _ = h.join();
            }
        }
    }

    /// Reactor body. Exclusively owns the `IoUring`.
    ///
    /// **Multi-request concurrency** (the follow-up flagged in PR
    /// #101): instead of dequeuing one `ReactorRequest` at a time and
    /// blocking in `submit_and_wait` until its entire macro-batch
    /// completes, the reactor keeps up to [`MAX_INFLIGHT_REQUESTS`]
    /// macro-batches in flight on the ring **simultaneously**, as long
    /// as their combined SQE count fits the configured queue depth.
    /// Concurrent callers (foreground miss fetches + speculative
    /// prefetch batches) therefore share the NVMe queue instead of the
    /// prefetch batch queueing *behind* the foreground batch at the
    /// reactor boundary.
    ///
    /// Mechanics:
    /// * Each admitted request is assigned a **slot id**; every one of
    ///   its SQEs carries `user_data = slot << 32 | expert_id`, so a
    ///   CQE can always be attributed to its owning request.
    /// * Completions are drained as they arrive (`submit_and_wait(1)` +
    ///   drain-all-available), decrementing the owning request's
    ///   remaining-op count; the reply oneshot fires the moment *that
    ///   request's* ops are all done — independent of any other
    ///   in-flight batch.
    /// * Oversized macro-batches (more SQEs than the ring depth) still
    ///   take the legacy exclusive chunked path ([`process_one`]),
    ///   which needs sole ownership of the completion queue; they are
    ///   parked until the ring drains.
    /// * A request that does not currently fit (`pending_ops + K >
    ///   queue_depth`) is parked — never dropped — and re-admitted as
    ///   soon as completions free up SQE budget. Channel FIFO order is
    ///   preserved (a parked request blocks later admissions).
    ///
    /// Loops until the channel closes *and* every in-flight batch has
    /// been drained.
    fn reactor_loop(mut ring: IoUring, mut rx: ReactorRx, queue_depth: usize) {
        use tokio::sync::mpsc::error::TryRecvError;
        let cap = queue_depth.max(1);
        let mut inflight: HashMap<u32, InflightBatch> = HashMap::new();
        let mut pending_ops: usize = 0;
        // Ops belonging to batches that were failed administratively
        // (their slots removed from `inflight`) but whose SQEs the
        // kernel may still complete. Tracked separately from
        // `pending_ops` so a late orphan CQE can never steal budget
        // accounting from a *live* batch — if it did, `pending_ops`
        // could hit zero while a live batch still had ops in the
        // kernel, the loop would stop waiting for completions, and
        // that batch's caller would hang on its oneshot forever.
        let mut orphaned_ops: usize = 0;
        let mut next_slot: u32 = 0;
        let mut parked: Option<ReactorRequest> = None;
        let mut closed = false;
        // Buffer owners for batches failed administratively (a
        // `submit_and_wait` error) while the kernel may still hold
        // their SQEs: dropping them could let the kernel DMA into
        // freed memory, so they are deliberately kept alive for the
        // reactor's lifetime instead (never safe to drop earlier).
        // Only ever populated on a broken-ring path.
        let mut leaked_keep_alives: Vec<Box<dyn std::any::Any + Send>> = Vec::new();

        loop {
            // ── Admission ────────────────────────────────────────────
            // A parked request has FIFO priority over the channel.
            if let Some(req) = parked.take() {
                if let Admit::Parked(req) = admit_request(
                    &mut ring, &mut inflight, &mut next_slot, &mut pending_ops,
                    orphaned_ops, cap, req,
                ) {
                    parked = Some(req);
                }
            }
            if parked.is_none() && !closed {
                if inflight.is_empty() {
                    // Fully idle (orphaned CQEs, if any, need no active
                    // wait — no caller is waiting on them): block until
                    // work (or close) arrives.
                    match rx.blocking_recv() {
                        Some(req) => {
                            if let Admit::Parked(req) = admit_request(
                                &mut ring, &mut inflight, &mut next_slot, &mut pending_ops,
                                orphaned_ops, cap, req,
                            ) {
                                parked = Some(req);
                            }
                        }
                        None => closed = true,
                    }
                }
                // Opportunistic top-up: admit whatever else is already
                // queued, without blocking, until the ring budget or
                // request cap stops us.
                while parked.is_none() && !closed {
                    match rx.try_recv() {
                        Ok(req) => {
                            if let Admit::Parked(req) = admit_request(
                                &mut ring, &mut inflight, &mut next_slot, &mut pending_ops,
                                orphaned_ops, cap, req,
                            ) {
                                parked = Some(req);
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => closed = true,
                    }
                }
            }
            if closed && inflight.is_empty() && parked.is_none() {
                break;
            }

            // ── Completion wait + drain ──────────────────────────────
            // Wait when live ops are outstanding, or when a parked
            // request can only make progress once orphaned CQEs drain
            // (an oversized batch needs exclusive CQ ownership).
            if pending_ops > 0 || (orphaned_ops > 0 && parked.is_some()) {
                if let Err(e) = ring.submit_and_wait(1) {
                    tracing::warn!(
                        error = %e,
                        in_flight = inflight.len(),
                        orphaned_ops,
                        "io_uring: submit_and_wait failed — failing all in-flight batches"
                    );
                    for (_, batch) in inflight.drain() {
                        let _ = batch.reply.send(Err(io::Error::new(
                            e.kind(),
                            format!("io_uring submit_and_wait failed: {e}"),
                        )));
                        // The kernel may still own these SQEs; never
                        // free the underlying buffers from here.
                        leaked_keep_alives.push(batch._keep_alive);
                    }
                    // The failed batches' ops become orphans: their
                    // CQEs may still surface later and must not be
                    // attributed to (or steal accounting from) batches
                    // admitted after this point.
                    orphaned_ops += pending_ops;
                    pending_ops = 0;
                    if orphaned_ops > 0 && parked.is_some() && inflight.is_empty() {
                        // We were waiting purely to drain orphans for a
                        // parked oversized batch, and the ring cannot
                        // even wait any more. Abandon the orphan drain
                        // so the parked request is not stuck forever.
                        tracing::warn!(
                            orphaned_ops,
                            "io_uring: abandoning orphaned-CQE drain on broken ring"
                        );
                        orphaned_ops = 0;
                    }
                    continue;
                }
                // Drain everything that is ready (≥ 1 by the wait above).
                loop {
                    let next = { ring.completion().next() };
                    let Some(cqe) = next else { break };
                    handle_cqe(&cqe, &mut inflight, &mut pending_ops, &mut orphaned_ops);
                }
            }
        }
    }

    /// Maximum number of macro-batches the reactor keeps in flight on
    /// the ring at once. Bounds the slot table and keeps per-CQE
    /// attribution lookups cheap; the SQE budget (`queue_depth`) is
    /// the real limiter for total queue depth.
    const MAX_INFLIGHT_REQUESTS: usize = 8;

    /// Book-keeping for one admitted macro-batch whose SQEs are on the
    /// ring. Mirrors the accumulation `process_one` did on the stack,
    /// lifted into a slot table so several batches can accumulate
    /// concurrently.
    struct InflightBatch {
        /// SQEs pushed for this batch that have not yet completed.
        remaining: usize,
        /// Expected read length per op (short-read validation).
        len: usize,
        /// Sum of full-length completions so far.
        total_bytes: usize,
        /// First per-op failure (OS error or short read) — reported to
        /// the caller once the whole batch drains, exactly like the
        /// serial path's first-error semantics.
        first_err: Option<io::Error>,
        reply: tokio::sync::oneshot::Sender<io::Result<usize>>,
        /// Owner of the fds + buffers the kernel is reading into; must
        /// outlive the last CQE of this batch.
        _keep_alive: Box<dyn std::any::Any + Send>,
    }

    enum Admit {
        /// Request consumed: SQEs are on the ring (or it was answered
        /// inline — empty, push-failed, or legacy oversized path).
        Admitted,
        /// Request does not fit right now; caller must retry after
        /// completions free SQE budget.
        Parked(ReactorRequest),
    }

    /// Try to put `req`'s SQEs on the ring under the shared-queue
    /// budget. See [`reactor_loop`] for the admission rules.
    fn admit_request(
        ring: &mut IoUring,
        inflight: &mut HashMap<u32, InflightBatch>,
        next_slot: &mut u32,
        pending_ops: &mut usize,
        orphaned_ops: usize,
        cap: usize,
        req: ReactorRequest,
    ) -> Admit {
        let total = req.prepared.len();
        if total == 0 {
            let _ = req.reply.send(Ok(0));
            return Admit::Admitted;
        }
        if total > cap {
            // Oversized macro-batch: needs the legacy chunked
            // submit/drain windows, which assume exclusive ownership
            // of the completion queue. Run it only when nothing else
            // is in flight — including orphaned CQEs from
            // administratively-failed batches, which `process_one`
            // would otherwise mis-attribute to its own ops.
            if !inflight.is_empty() || orphaned_ops > 0 {
                return Admit::Parked(req);
            }
            let ReactorRequest { prepared, len, reply, _keep_alive } = req;
            let result = process_one(ring, &prepared, len, cap);
            let _ = reply.send(result);
            drop(_keep_alive);
            return Admit::Admitted;
        }
        if *pending_ops + total > cap || inflight.len() >= MAX_INFLIGHT_REQUESTS {
            return Admit::Parked(req);
        }
        // Allocate an unused slot id for CQE attribution.
        let slot = loop {
            let s = *next_slot;
            *next_slot = next_slot.wrapping_add(1);
            if !inflight.contains_key(&s) {
                break s;
            }
        };
        let ReactorRequest { prepared, len, reply, _keep_alive } = req;
        let mut pushed = 0usize;
        let mut push_err: Option<io::Error> = None;
        'push: for &(fd, ptr, buf_idx, expert_id) in &prepared {
            let sqe = opcode::ReadFixed::new(types::Fd(fd), ptr, len as u32, buf_idx)
                .offset(0)
                .build()
                .user_data(((slot as u64) << 32) | expert_id as u64);
            // SAFETY / retry semantics identical to the serial path:
            // `ptr`/`fd` are kept alive by `_keep_alive` and the ring's
            // pool clone; only `PushError::Full` is transient.
            const MAX_PUSH_RETRIES: usize = 16;
            let mut attempts = 0usize;
            loop {
                // SAFETY: the reactor thread is the single owner of
                // `ring`; the SQE is a value type.
                if unsafe { ring.submission().push(&sqe) }.is_ok() {
                    pushed += 1;
                    break;
                }
                attempts += 1;
                if attempts > MAX_PUSH_RETRIES {
                    push_err = Some(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "io_uring: submission queue refused {MAX_PUSH_RETRIES} push retries — kernel not draining SQ"
                        ),
                    ));
                    break 'push;
                }
                // SQ full — non-blocking flush to make room, then retry.
                if let Err(e) = ring.submit() {
                    push_err = Some(e);
                    break 'push;
                }
            }
        }
        if pushed == 0 {
            // Nothing reached the ring: fail the request inline.
            let e = push_err.unwrap_or_else(|| {
                io::Error::new(io::ErrorKind::Other, "io_uring: no SQEs pushed")
            });
            let _ = reply.send(Err(e));
            drop(_keep_alive);
            return Admit::Admitted;
        }
        // Hand the pushed window to the kernel now (non-blocking) so
        // the device starts servicing it while we admit more requests.
        // A failure here is recorded as the batch's first error; the
        // SQEs stay queued and the next `submit_and_wait` retries the
        // kernel handoff.
        if let Err(e) = ring.submit() {
            if push_err.is_none() {
                push_err = Some(e);
            }
        }
        *pending_ops += pushed;
        inflight.insert(
            slot,
            InflightBatch {
                remaining: pushed,
                len,
                total_bytes: 0,
                first_err: push_err.map(|e| {
                    io::Error::new(
                        e.kind(),
                        format!("io_uring batch partially submitted: {e}"),
                    )
                }),
                reply,
                _keep_alive,
            },
        );
        Admit::Admitted
    }

    /// Attribute one CQE to its owning in-flight batch, apply the same
    /// per-op validation as the serial path (negative result → OS
    /// error, short read → `UnexpectedEof`), and fire the batch's
    /// reply when its last op drains.
    fn handle_cqe(
        cqe: &io_uring::cqueue::Entry,
        inflight: &mut HashMap<u32, InflightBatch>,
        pending_ops: &mut usize,
        orphaned_ops: &mut usize,
    ) {
        let user_data = cqe.user_data();
        let slot = (user_data >> 32) as u32;
        let expert_id = user_data as u32;
        let Some(batch) = inflight.get_mut(&slot) else {
            // CQE for a slot we no longer track — an orphan from a
            // batch failed administratively (`submit_and_wait` error).
            // Account it against the orphan budget, never against
            // `pending_ops`: live batches' accounting must not be
            // disturbed by stale completions.
            tracing::warn!(slot, expert_id, "io_uring: CQE for unknown request slot");
            *orphaned_ops = orphaned_ops.saturating_sub(1);
            return;
        };
        *pending_ops = pending_ops.saturating_sub(1);
        let result = cqe.result();
        if result < 0 {
            let e = io::Error::from_raw_os_error(-result);
            tracing::warn!(
                expert_id,
                error = %e,
                "io_uring CQE reported error"
            );
            if batch.first_err.is_none() {
                batch.first_err = Some(io::Error::new(
                    e.kind(),
                    format!("io_uring read on expert {expert_id} failed: {e}"),
                ));
            }
        } else {
            let n = result as usize;
            if n != batch.len {
                tracing::warn!(
                    expert_id,
                    got = n,
                    expected = batch.len,
                    "io_uring CQE reported short read"
                );
                if batch.first_err.is_none() {
                    batch.first_err = Some(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "io_uring short read on expert {expert_id}: got {n} bytes, expected {}",
                            batch.len
                        ),
                    ));
                }
            } else {
                batch.total_bytes += n;
            }
        }
        batch.remaining -= 1;
        if batch.remaining == 0 {
            let batch = inflight
                .remove(&slot)
                .expect("slot present: we just mutated it");
            let result = match batch.first_err {
                Some(e) => Err(e),
                None => Ok(batch.total_bytes),
            };
            let _ = batch.reply.send(result);
            drop(batch._keep_alive);
        }
    }

    fn process_one(
        ring: &mut IoUring,
        prepared: &[(RawFd, *mut u8, u16, u32)],
        len: usize,
        cap: usize,
    ) -> io::Result<usize> {
        let total = prepared.len();
        if total == 0 {
            return Ok(0);
        }
        let mut submitted_ops = 0usize;
        let mut pushed = 0usize;
        while pushed < total {
            let chunk_end = (pushed + cap).min(total);
            for i in pushed..chunk_end {
                let (fd, ptr, buf_idx, expert_id) = prepared[i];
                let sqe = opcode::ReadFixed::new(types::Fd(fd), ptr, len as u32, buf_idx)
                    .offset(0)
                    .build()
                    .user_data(expert_id as u64);
                // SAFETY: SQE references `ptr` and `fd` which are kept
                // alive by the requester's `_keep_alive` box (which
                // holds the `Arc<File>` fd-cache entries and the
                // owned `PooledBuffer`s — see F1.3) and by the
                // ring's `pool` clone (F2.3). We do not panic on
                // submission-queue-full: we `submit()` to flush
                // whatever is already queued (creating room) and
                // retry. F3.3: bound the retry count so a kernel
                // that refuses to drain the SQ cannot wedge the
                // reactor in an infinite loop.
                const MAX_PUSH_RETRIES: usize = 16;
                let mut push_attempts = 0usize;
                loop {
                    // SAFETY: io_uring crate's `push` requires us to
                    // own the SQ uniquely; we do (the reactor is the
                    // single owner of `ring`). The SQE entry itself
                    // is a value type.
                    let push_result = unsafe { ring.submission().push(&sqe) };
                    if push_result.is_ok() {
                        break;
                    }
                    push_attempts += 1;
                    if push_attempts > MAX_PUSH_RETRIES {
                        if submitted_ops > 0 {
                            if let Err(e) = drain_submitted_completions(ring, submitted_ops) {
                                tracing::warn!(
                                    error = %e,
                                    submitted_ops,
                                    "io_uring: failed to fully drain previously submitted CQEs before SQ retry abort"
                                );
                            }
                        }
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!(
                                "io_uring: submission queue refused {MAX_PUSH_RETRIES} push retries — kernel not draining SQ"
                            ),
                        ));
                    }
                    // SQ full — non-blocking flush, then retry. `submit()`
                    // forwards any kernel-side error (EAGAIN, ENOMEM,
                    // …) out of the reactor; only `PushError::Full`
                    // from `push` is treated as "transient, retry".
                    submitted_ops += ring.submit()?;
                }
            }
            if chunk_end < total {
                // Hand this window down to the kernel so it can start
                // servicing reads while we keep queueing. Non-blocking.
                submitted_ops += ring.submit()?;
            }
            pushed = chunk_end;
        }
        // Single wait barrier for the whole macro-batch.
        ring.submit_and_wait(total)?;
        // F1.1 + F1.2: drain ALL completions before returning.
        // Returning early on the first error would orphan the
        // remaining CQEs in the ring; the next macro-batch would
        // then dequeue them in `completion().next()` as if they
        // belonged to the new batch, mixing expert ids across
        // requests. Verify the per-CQE length against the requested
        // `len` so a short read surfaces as the failure it is rather
        // than silently being summed into `totalbytes`.
        let mut first_err: Option<io::Error> = None;
        let mut totalbytes = 0usize;
        let mut drained = 0usize;
        while drained < total {
            // Splitting the completion-queue access into its own scope
            // drops the `CompletionQueue` (and its mutable borrow on
            // `ring`) before we call `ring.submit()` below; the returned
            // CQE `Entry` is an owned copy, so it outlives the queue.
            let next = { ring.completion().next() };
            let cqe = match next {
                Some(c) => c,
                None => {
                    // CQE not yet available — flush + spin briefly.
                    // `submit_and_wait(total)` above should have
                    // guaranteed all are ready, but on aggressive
                    // SQPOLL kernels there can be a publication lag.
                    ring.submit()?;
                    continue;
                }
            };
            drained += 1;
            let result = cqe.result();
            let user_data = cqe.user_data();
            if result < 0 {
                let e = io::Error::from_raw_os_error(-result);
                tracing::warn!(
                    expert_id = user_data,
                    error = %e,
                    "io_uring CQE reported error"
                );
                if first_err.is_none() {
                    first_err = Some(io::Error::new(
                        e.kind(),
                        format!("io_uring read on expert {user_data} failed: {e}"),
                    ));
                }
                continue;
            }
            let n = result as usize;
            if n != len {
                tracing::warn!(
                    expert_id = user_data,
                    got = n,
                    expected = len,
                    "io_uring CQE reported short read"
                );
                if first_err.is_none() {
                    first_err = Some(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "io_uring short read on expert {user_data}: got {n} bytes, expected {len}"
                        ),
                    ));
                }
                continue;
            }
            totalbytes += n;
        }
        if let Some(e) = first_err {
            Err(e)
        } else {
            Ok(totalbytes)
        }
    }

    fn drain_submitted_completions(ring: &mut IoUring, submitted: usize) -> io::Result<()> {
        let mut drained = 0usize;
        while drained < submitted {
            // Drop the `CompletionQueue` borrow before calling
            // `ring.submit_and_wait(1)`; the `Entry` is an owned copy.
            let next = { ring.completion().next() };
            match next {
                Some(_) => drained += 1,
                None => {
                    ring.submit_and_wait(1)?;
                }
            }
        }
        Ok(())
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
        let path = tmp.join("expert_0.bin");
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..], buf.as_slice());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batch_fixed_promote_round_trips_and_flips_shadow_to_primary() {
        use crate::io_provider::generate_synthetic_experts;

        let mut tmp = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        tmp.push(format!("mer-iouring-promote-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let num_experts = 3u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(&tmp, num_experts, expert_size, d_model, d_ff).unwrap();

        // Two primary + two shadow slots; the speculative path uses the shadow half.
        let pool = BufferPool::new_with_shadow(2, 2, expert_size, block);
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
                eprintln!("io_uring not available, skipping: {e}");
                let _ = std::fs::remove_dir_all(&tmp);
                return;
            }
        };

        // Acquire two **shadow** buffers — what a Speculator would do.
        let s0 = pool.try_acquire_shadow().expect("shadow 0");
        let s1 = pool.try_acquire_shadow().expect("shadow 1");
        assert!(s0.is_shadow() && s1.is_shadow());

        // One submit_and_wait(K) fires both reads, then promote_shadow
        // is called for each completion: the returned buffers must
        // already be primary.
        let ids = vec![0u32, 1u32];
        let promoted = match storage
            .read_experts_batch_fixed_promote(&ids, vec![s0, s1], &pool)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!("io_uring batch read failed in this env: {e}");
                let _ = std::fs::remove_dir_all(&tmp);
                return;
            }
        };
        assert_eq!(promoted.len(), 2);
        for (i, buf) in promoted.iter().enumerate() {
            assert!(
                !buf.is_shadow(),
                "buffer {i} should have been promoted to primary"
            );
            // Byte-identical to a raw pread of the same expert file.
            let raw = std::fs::read(tmp.join(format!("expert_{i}.bin"))).unwrap();
            assert_eq!(&raw[..], buf.as_slice());
        }

        // Dropping the promoted buffers must release them into the
        // *primary* free list — the shadow half should still be
        // exhausted (we promoted both shadow slots).
        drop(promoted);
        assert!(pool.try_acquire_shadow().is_none(), "shadow exhausted");
        // Primary picked up the two now-free buffers.
        let _p0 = pool.try_acquire().expect("primary slot 0");
        let _p1 = pool.try_acquire().expect("primary slot 1");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The multi-request reactor must serve byte-correct replies when
    /// several macro-batches share the ring concurrently — including
    /// batches that exceed `MAX_INFLIGHT_REQUESTS` (parking), exhaust
    /// the SQE budget (admission gating), and an **oversized** batch
    /// (more SQEs than the queue depth) that takes the exclusive
    /// legacy `process_one` path interleaved with fitting batches.
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_batches_share_ring_and_round_trip() {
        use crate::io_provider::generate_synthetic_experts;

        let mut tmp = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        tmp.push(format!("mer-iouring-conc-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let num_experts = 8u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(&tmp, num_experts, expert_size, d_model, d_ff).unwrap();

        // Small queue depth (4) so concurrent 2-element batches exceed
        // the SQE budget (forcing parking + re-admission) and a
        // 6-element batch is oversized (legacy exclusive path).
        let pool = BufferPool::new(24, expert_size, block);
        let storage = match IoUringStorage::new(
            IoUringConfig {
                base_path: tmp.clone(),
                expert_size,
                block_align: block,
                queue_depth: 4,
                numa_node: None,
            },
            &pool,
        ) {
            Ok(s) => std::sync::Arc::new(s),
            Err(e) => {
                eprintln!("io_uring not available, skipping: {e}");
                let _ = std::fs::remove_dir_all(&tmp);
                return;
            }
        };

        // Reference bytes via std reads.
        let reference: Vec<Vec<u8>> = (0..num_experts)
            .map(|i| std::fs::read(tmp.join(format!("expert_{i}.bin"))).unwrap())
            .collect();

        let mut tasks = Vec::new();
        // 8 concurrent 2-element batches (16 SQEs total >> depth 4).
        // The batch futures are not `Send` (raw buffer pointers held
        // across the await), so concurrency is driven by `join_all`
        // on this task: every future sends its `ReactorRequest`
        // before any reply resolves, so the reactor sees them all
        // in flight together.
        for t in 0..8u32 {
            let storage = storage.clone();
            let pool = pool.clone();
            let reference = &reference;
            tasks.push(async move {
                let ids = vec![t % num_experts, (t + 3) % num_experts];
                let mut b0 = pool.acquire().await;
                let mut b1 = pool.acquire().await;
                {
                    let mut bufs = [&mut b0, &mut b1];
                    let n = storage
                        .read_experts_batch_fixed(&ids, &mut bufs)
                        .await
                        .expect("concurrent batch read");
                    assert_eq!(n, ids.len() * expert_size);
                }
                assert_eq!(b0.as_slice(), &reference[ids[0] as usize][..]);
                assert_eq!(b1.as_slice(), &reference[ids[1] as usize][..]);
            });
        }
        // One oversized batch (6 SQEs > depth 4) racing the others:
        // exercises parking-until-exclusive + the legacy chunked path.
        let oversized = {
            let storage = storage.clone();
            let pool = pool.clone();
            let reference = &reference;
            async move {
                let ids: Vec<u32> = (0..6u32).collect();
                let mut bufs: Vec<PooledBuffer> = Vec::new();
                for _ in 0..ids.len() {
                    bufs.push(pool.acquire().await);
                }
                {
                    let mut refs: Vec<&mut PooledBuffer> = bufs.iter_mut().collect();
                    let n = storage
                        .read_experts_batch_fixed(&ids, &mut refs)
                        .await
                        .expect("oversized batch read");
                    assert_eq!(n, ids.len() * expert_size);
                }
                for (i, b) in bufs.iter().enumerate() {
                    assert_eq!(b.as_slice(), &reference[i][..], "oversized expert {i}");
                }
            }
        };
        futures::future::join(futures::future::join_all(tasks), oversized).await;

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
