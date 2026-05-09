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
//!    [`BufferPool::raw_iovecs`]). After that, a read submission only
//!    references a buffer *index* — the kernel doesn't have to walk the
//!    user mapping or pin pages on the hot path.
//! 2. **Batch many reads with a single syscall.** When a token misses
//!    on `K > 1` experts, we push `K` SQEs and `enter()` once.
//! 3. **Reduce per-read CPU** (and therefore energy) by ~30–50 % on
//!    NVMe-class SSDs in published microbenchmarks. That CPU time is
//!    pure overhead — the same bytes were going to leave the device
//!    either way; io_uring just makes the kernel cheaper.
//!
//! ### Status
//!
//! This module declares the API surface (`IoUringStorage::new`,
//! `IoUringStorage::read_expert_fixed`) so a follow-up PR can drop in
//! a complete `io_uring` crate-backed implementation without touching
//! the public engine. The current bodies are `unimplemented!()` — the
//! engine selects the `pread` backend, and `--io-uring` only logs a
//! note. Wiring the real ring is straightforward once a Linux test
//! environment is available; see the doc comments below for the
//! intended sequence.

#![allow(dead_code)]

use crate::buffer_pool::BufferPool;
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
}

/// `io_uring` storage backend.
///
/// **Construction** registers every buffer in `pool` as a fixed
/// io_uring buffer (one `io_uring_register` syscall, amortised across
/// the lifetime of the engine). Subsequent reads are `IORING_OP_READ_FIXED`
/// SQEs that reference a buffer index — no per-read iovec setup.
pub struct IoUringStorage {
    cfg: IoUringConfig,
    /// Number of buffer slots that were registered. Stored for
    /// validation only — the actual ring + fd state lives in the
    /// follow-up wiring.
    registered_buffers: usize,
}

impl IoUringStorage {
    /// Create a new io_uring backend over `pool`. The pool's buffers
    /// are registered *as is* with the kernel; do not resize the pool
    /// after this returns.
    pub fn new(cfg: IoUringConfig, pool: &BufferPool) -> io::Result<Self> {
        // Snapshot the iovecs so the caller's safety contract on
        // `BufferPool::raw_iovecs` is observable. The real ring would
        // hand these to `io_uring_register` here.
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
        Ok(Self {
            cfg,
            registered_buffers: iovecs.len(),
        })
    }

    /// Number of pool buffers currently registered with the kernel.
    pub fn registered_buffers(&self) -> usize {
        self.registered_buffers
    }

    /// Submit a `READ_FIXED` SQE for `expert_id` into the registered
    /// buffer at `buf_index`, then wait for its completion. The
    /// portable `pread(2)` backend in [`crate::io_provider`] is the
    /// default; this is the high-throughput Linux-only path.
    pub async fn read_expert_fixed(
        &self,
        _expert_id: u32,
        _buf_index: u32,
    ) -> io::Result<usize> {
        // Intended call sequence (follow-up PR):
        //   1. ensure the per-expert fd is open (reuse `NvmeStorage`'s
        //      `fd_for` pattern).
        //   2. `let mut sq = ring.submission();`
        //      build a `READ_FIXED { fd, buf_index, offset: 0,
        //                            len: cfg.expert_size }` SQE,
        //      push it, drop sq.
        //   3. `ring.submit_and_wait(1)?;`
        //   4. drain `ring.completion()` and return the result code.
        unimplemented!(
            "io_uring backend stub: queue_depth={}, expert_size={}, base_path={}",
            self.cfg.queue_depth,
            self.cfg.expert_size,
            self.cfg.base_path.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registers_all_pool_slots() {
        let pool = BufferPool::new(4, 4096, 4096);
        let s = IoUringStorage::new(
            IoUringConfig {
                base_path: PathBuf::from("/tmp"),
                expert_size: 4096,
                block_align: 4096,
                queue_depth: 8,
            },
            &pool,
        )
        .unwrap();
        assert_eq!(s.registered_buffers(), 4);
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
            },
            &pool,
        );
        assert!(res.is_err());
    }
}
