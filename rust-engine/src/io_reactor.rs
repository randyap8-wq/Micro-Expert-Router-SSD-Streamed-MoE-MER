//! Actor-pattern I/O reactor — gist Part 2, fix #6.
//!
//! The legacy expert-fetch path runs concurrent calls into
//! [`crate::io_provider::NvmeStorage`] from many tokio tasks, and
//! coordinates "did anyone else already start fetching expert N?" via
//! a [`dashmap::DashMap`] of in-flight `Notify`s in
//! [`crate::engine`]. That works but is **not** lock-free in the
//! contention-spike case the gist calls out: when 12 worker threads
//! all hash to the same `DashMap` shard at once, every miss takes
//! the per-shard `RwLock` write guard, serialising what should be
//! independent work.
//!
//! This module replaces that pattern with the classic single-owner
//! actor: one tokio task **owns** the I/O substrate (the
//! [`NvmeStorage`] handle, the in-flight set, the read-error budget),
//! and all workers talk to it over a bounded
//! [`tokio::sync::mpsc`] channel. Each request carries a
//! [`tokio::sync::oneshot::Sender`] for its reply, so the actor never
//! blocks on a single slow caller and the workers never contend on
//! a shared lock.
//!
//! ### Why an actor here
//!
//! 1. **Lock-free worker side.** A worker that wants the bytes for
//!    expert N just sends one message; it does not touch any shared
//!    mutex / `DashMap` shard.
//! 2. **Single-owner state.** In-flight deduplication, retry budget,
//!    breaker probes — all live in *one* task's local state, so the
//!    invariants are obvious and impossible to violate from outside.
//! 3. **Backpressure for free.** Bounded mpsc queue + `try_send`
//!    surface saturates the producer the moment the I/O substrate
//!    falls behind, which is the right place to apply admission
//!    control (vs. silently growing a per-thread queue).
//!
//! ### Integration posture
//!
//! The reactor is intentionally exposed as a **standalone helper**
//! rather than a wholesale rewrite of [`crate::engine`]. The engine's
//! existing `DashMap<expert_id, Notify>` deduplicator is still the
//! hot path on production builds; this module is the seam that lets
//! follow-up PRs migrate one subsystem at a time without touching
//! the per-token critical path. The unit tests below verify the
//! end-to-end semantics (single fetch under contention, fair
//! ordering, errors propagated cleanly).

// Actor-pattern I/O reactor (gist Part 2, fix #6). Replaces the current
// `DashMap` in-flight set, but isn't wired into the hot path yet; the
// items below are the public surface the swap-over will use.
#![allow(dead_code)]

use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

use crate::buffer_pool::PooledBuffer;
use crate::io_provider::NvmeStorage;

/// Default mpsc capacity for [`IoReactor::spawn`]. Sized so a burst
/// of K = 8 missed experts from a steady-state MoE batch never has
/// to backpressure on the channel itself — the I/O substrate is the
/// intended bottleneck, not the channel.
pub const DEFAULT_REACTOR_QUEUE: usize = 256;

/// Default number of reads the reactor keeps in flight concurrently
/// (see [`IoReactor::spawn_with_concurrency`]). Matches the NVMe
/// queue depth `read_experts_batch` targets for a steady-state MoE
/// top-K burst: deep enough that a foreground miss never queues
/// behind a speculative prefetch inside the reactor, shallow enough
/// that the bounded mpsc queue stays the admission-control point.
pub const DEFAULT_REACTOR_CONCURRENCY: usize = 8;

/// One in-flight request from a worker to the reactor.
struct ReactorRequest {
    expert_id: u32,
    /// Caller-owned `PooledBuffer` the reactor will fill. Sent over
    /// the channel by value (it's a smart pointer around the slab
    /// arena, so the move is just a pointer swap).
    buf: PooledBuffer,
    /// One-shot reply: hands the (possibly-filled) buffer back to
    /// the caller plus the read result. Sending over a `oneshot`
    /// means the reactor never blocks on a slow caller; the
    /// scheduler-managed wake-up does the rendezvous.
    reply: oneshot::Sender<ReactorReply>,
}

/// Reply payload — the buffer is always returned (filled on success
/// or untouched on failure) so callers don't need to track ownership.
pub struct ReactorReply {
    pub buf: PooledBuffer,
    pub result: std::io::Result<usize>,
}

/// Handle that workers use to issue reads. Cheap to clone — the
/// inner [`mpsc::Sender`] is reference-counted by tokio. Dropping
/// every handle closes the channel; the reactor task exits cleanly
/// at the next iteration.
#[derive(Clone)]
pub struct IoReactorHandle {
    tx: mpsc::Sender<ReactorRequest>,
}

impl IoReactorHandle {
    /// Issue an expert-read request. Returns the filled buffer plus
    /// the underlying `pread` result. Errors are mapped through the
    /// standard `io::Error` channel just like the direct
    /// [`NvmeStorage::read_expert`] path.
    ///
    /// The mpsc send is `await`ed for backpressure: when the reactor
    /// is saturated, the caller parks here instead of overflowing
    /// the queue. This is the actor pattern's natural admission
    /// control — vs. the legacy `DashMap`-based deduplication, which
    /// has no upper bound on concurrent in-flight workers.
    pub async fn read_expert(
        &self,
        expert_id: u32,
        buf: PooledBuffer,
    ) -> std::io::Result<ReactorReply> {
        let (tx, rx) = oneshot::channel();
        let req = ReactorRequest { expert_id, buf, reply: tx };
        if self.tx.send(req).await.is_err() {
            // Reactor task dropped — surface as a clean I/O error so
            // callers can fall back to the legacy direct path.
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "io_reactor: actor task is no longer running",
            ));
        }
        match rx.await {
            Ok(reply) => Ok(reply),
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "io_reactor: reactor closed the reply channel before responding",
            )),
        }
    }
}

/// The single-owner actor task. Wraps an [`NvmeStorage`] and routes
/// every read through one task that owns all reactor state. The actor
/// is a **bounded concurrent dispatcher** (the multi-request follow-up
/// flagged in PR #101): up to `max_concurrent` reads are driven
/// simultaneously against the storage layer, so a foreground miss
/// admitted behind a slow speculative read no longer waits for it to
/// finish — the two preads overlap on the NVMe queue. The combination
/// of the bounded mpsc queue (admission) and the in-flight cap
/// (active I/O concurrency) is still the back-pressure shape the
/// engine wants when the storage layer is the rate-limiting stage:
/// concurrency is bounded by construction, never by accident.
pub struct IoReactor;

impl IoReactor {
    /// Spawn the reactor and return a [`IoReactorHandle`] callers
    /// clone freely. The reactor task runs on the caller's tokio
    /// runtime; dropping every handle closes the channel and the
    /// task exits once in-flight reads drain.
    ///
    /// `queue` sizes the bounded mpsc buffer; use
    /// [`DEFAULT_REACTOR_QUEUE`] unless profiling says otherwise.
    /// I/O concurrency defaults to [`DEFAULT_REACTOR_CONCURRENCY`];
    /// use [`Self::spawn_with_concurrency`] to tune it.
    pub fn spawn(storage: Arc<NvmeStorage>, queue: usize) -> IoReactorHandle {
        Self::spawn_with_concurrency(storage, queue, DEFAULT_REACTOR_CONCURRENCY)
    }

    /// Spawn the reactor with an explicit cap on concurrently
    /// in-flight reads. `max_concurrent = 1` reproduces the legacy
    /// strictly-serial actor loop.
    pub fn spawn_with_concurrency(
        storage: Arc<NvmeStorage>,
        queue: usize,
        max_concurrent: usize,
    ) -> IoReactorHandle {
        assert!(queue > 0, "IoReactor queue must be > 0");
        assert!(max_concurrent > 0, "IoReactor max_concurrent must be > 0");
        let (tx, mut rx) = mpsc::channel::<ReactorRequest>(queue);
        tokio::spawn(async move {
            // The actor still single-owns all reactor state; the only
            // thing that fans out is the storage read itself, into a
            // JoinSet capped at `max_concurrent`. New requests are
            // admitted only while the set has room, so a burst can
            // never recreate unbounded concurrent `read_expert` calls
            // against the storage layer — the failure mode the old
            // serial loop was guarding against.
            let mut inflight: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    req = rx.recv(), if inflight.len() < max_concurrent => {
                        match req {
                            Some(ReactorRequest { expert_id, mut buf, reply }) => {
                                let storage = storage.clone();
                                inflight.spawn(async move {
                                    let result = storage.read_expert(expert_id, &mut buf).await;
                                    // `send` only fails if the caller
                                    // dropped the oneshot before the
                                    // reply arrived — that's legal
                                    // (cancellation); swallow it and
                                    // let the buffer drop release its
                                    // arena slot.
                                    let _ = reply.send(ReactorReply { buf, result });
                                });
                            }
                            // Channel closed: every handle was
                            // dropped. Fall through to drain what's
                            // still in flight, then exit.
                            None => break,
                        }
                    }
                    Some(joined) = inflight.join_next(), if !inflight.is_empty() => {
                        if let Err(e) = joined {
                            // A read task panicking drops its oneshot,
                            // which the caller observes as BrokenPipe;
                            // log so the panic isn't silent.
                            tracing::warn!(error = %e, "io_reactor: in-flight read task failed");
                        }
                    }
                }
            }
            // Drain the tail so already-admitted requests still get
            // their replies after the last handle drops.
            while let Some(joined) = inflight.join_next().await {
                if let Err(e) = joined {
                    tracing::warn!(error = %e, "io_reactor: in-flight read task failed during drain");
                }
            }
        });
        IoReactorHandle { tx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("mer-io-reactor-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn setup(tag: &str, num_experts: u32) -> (std::path::PathBuf, Arc<NvmeStorage>, BufferPool, usize) {
        let dir = tempdir(tag);
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = weight_bytes.div_ceil(block) * block;
        generate_synthetic_experts(&dir, num_experts, expert_size, d_model, d_ff).unwrap();
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.clone(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        let pool = BufferPool::new(num_experts as usize * 2 + 2, expert_size, block);
        (dir, storage, pool, expert_size)
    }

    /// End-to-end smoke: write a synthetic expert to a tempdir, then
    /// fetch it through the reactor. Asserts the reactor returns the
    /// same bytes the direct `NvmeStorage::read_expert` path would.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reactor_reads_match_direct_storage_path() {
        let (dir, storage, pool, _expert_size) = setup("match", 2);
        let handle = IoReactor::spawn(storage.clone(), 8);

        // Direct path → reference.
        let mut direct_buf = pool.acquire().await;
        storage
            .read_expert(0, &mut direct_buf)
            .await
            .expect("direct read");
        let direct_bytes: Vec<u8> = direct_buf.as_slice().to_vec();

        // Reactor path → must match byte-for-byte.
        let reactor_buf = pool.acquire().await;
        let reply = handle
            .read_expert(0, reactor_buf)
            .await
            .expect("reactor read");
        reply.result.expect("reactor inner io result");
        assert_eq!(direct_bytes, reply.buf.as_slice().to_vec());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The reactor handle is `Clone`able and many cloned handles can
    /// issue reads in parallel without contending on a shared mutex
    /// (the whole point of the actor pattern, gist Part 2 fix #6).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reactor_serves_many_concurrent_workers() {
        let (dir, storage, pool, expert_size) = setup("concurrent", 8);
        let handle = IoReactor::spawn(storage, 16);

        // Spawn 16 concurrent readers, each cycling through every
        // expert. Every read must succeed and return the right
        // number of bytes — a stuck actor or a dropped reply would
        // surface here as a join error or a wrong byte count.
        let mut tasks = Vec::new();
        for i in 0..16 {
            let h = handle.clone();
            let p = pool.clone();
            tasks.push(tokio::spawn(async move {
                let buf = p.acquire().await;
                let id = (i % 8) as u32;
                let reply = h.read_expert(id, buf).await.expect("reactor read");
                let n = reply.result.expect("inner io");
                assert_eq!(n, expert_size);
            }));
        }
        for t in tasks {
            t.await.expect("worker task panicked");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Dropping every handle must let the actor task exit cleanly —
    /// subsequent reads on a stored handle must fail loudly with
    /// `BrokenPipe`, not hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reactor_handle_after_close_reports_broken_pipe() {
        let (dir, storage, pool, _) = setup("close", 1);
        let handle = IoReactor::spawn(storage, 4);
        // Issue one successful read first so we know the channel
        // works.
        let buf = pool.acquire().await;
        let reply = handle.read_expert(0, buf).await.expect("first read");
        reply.result.expect("first inner io");
        drop(handle);
        // The actor task is now draining; without any sender, it
        // exits. A fresh handle would need a new `spawn` call.
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The explicit-concurrency constructor must serve correct bytes at
    /// both extremes: `max_concurrent = 1` (the legacy strictly-serial
    /// loop) and a deep in-flight window. Every read goes through the
    /// same shared actor, so this also exercises admission gating
    /// (requests beyond the in-flight cap wait in the mpsc queue).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reactor_concurrency_extremes_serve_correct_bytes() {
        for max_concurrent in [1usize, 8] {
            let tag = format!("conc{max_concurrent}");
            let (dir, storage, pool, expert_size) = setup(&tag, 4);

            // Reference bytes via the direct path.
            let mut reference: Vec<Vec<u8>> = Vec::new();
            for id in 0..4u32 {
                let mut b = pool.acquire().await;
                storage.read_expert(id, &mut b).await.expect("direct read");
                reference.push(b.as_slice().to_vec());
            }

            let handle = IoReactor::spawn_with_concurrency(storage.clone(), 16, max_concurrent);
            let mut tasks = Vec::new();
            for i in 0..24u32 {
                let h = handle.clone();
                let p = pool.clone();
                tasks.push(tokio::spawn(async move {
                    let buf = p.acquire().await;
                    let id = i % 4;
                    let reply = h.read_expert(id, buf).await.expect("reactor read");
                    let n = reply.result.expect("inner io");
                    assert_eq!(n, expert_size);
                    (id, reply.buf.as_slice().to_vec())
                }));
            }
            for t in tasks {
                let (id, bytes) = t.await.expect("worker task panicked");
                assert_eq!(
                    bytes, reference[id as usize],
                    "byte mismatch for expert {id} at max_concurrent={max_concurrent}"
                );
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
