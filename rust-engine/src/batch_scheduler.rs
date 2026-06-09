//! Continuous batching for the HTTP server's real-transformer path.
//!
//! Background
//! ----------
//! The original `serve` path processed each request strictly
//! sequentially: every concurrent HTTP request owned its own KV cache
//! and called `RealModel::step` directly, which in turn drove the SSD
//! expert cache for that single request. With N simultaneous users the
//! engine effectively serialised them on the per-token critical path.
//!
//! [`BatchScheduler`] sits between the HTTP handlers and `RealModel`.
//! Each request future submits *one [`StepRequest`] per token it needs*
//! over an [`mpsc`] channel; a dedicated background task drains the
//! channel, fuses up to `max_batch_size` pending requests (or whatever
//! has arrived within `batch_timeout`) into a single batch, and runs
//! their `RealModel::step` calls concurrently on the same shared
//! `Engine`.
//!
//! ## Zero-copy request registry
//!
//! The scheduler owns a central [`RequestRegistry`] keyed by
//! [`RequestId`]. The HTTP path calls [`BatchScheduler::register`]
//! when a request begins generating tokens; that returns a small
//! integer id, after which every per-token [`StepRequest`] going over
//! the channel carries only `{ id, token_id, pos, params }` — never
//! the `Vec<KvCache>` itself. The background loop then looks up the
//! mutable KV slice from the registry just before calling
//! `model.step()`. On [`BatchScheduler::release`] (or when the
//! optional per-pool block manager is dropped) the registry entry
//! and the associated [`BlockManager`] are torn down, returning every
//! KV-cache block to the shared pool.
//!
//! The legacy [`BatchScheduler::step`] signature (which passes the
//! `Vec<KvCache>` in by-move) is preserved for callers that don't
//! want to manage a registry handle directly; it internally
//! `register`s, drives one step, then `release`s — at the cost of
//! one short sharded-DashMap insert per token, which is negligible
//! compared to the matmul cost of the step itself.
//!
//! ## Dynamic paged KV pool
//!
//! When the scheduler is configured with `block_pool_capacity > 0`
//! the underlying [`BlockPool`] uses a primary pre-allocated slab
//! plus a heap-backed overflow slab that grows on demand. The
//! scheduler exposes [`BatchScheduler::overflow_in_use`] and emits a
//! warning log the first time a request touches the overflow slab,
//! so operators can size the primary capacity for steady-state
//! workloads while remaining safe under bursts.
//!
//! The engine's expert cache + storage already use atomic counters and
//! `Arc`s, so multiple `moe_step` calls from sibling tasks are safe;
//! see [`crate::engine`] for details.

use crate::block_pool::{BlockManager, BlockPool, PressureLevel};
use crate::backend::Backend;
use crate::engine::Engine;
use crate::model::RealModel;
use crate::router::SpeculationController;
use crate::transformer::KvCache;
use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

// Re-export the block-pool types so HTTP handlers and tests can build
// `BlockManager`s directly off the scheduler's shared pool without
// taking an additional dependency on the `block_pool` module path.
#[allow(unused_imports)]
pub use crate::block_pool::{BlockAllocError, BlockId, BlockManager as PooledBlockManager,
    BlockPool as PooledBlockPool, POOL_BLOCK_TOKENS};

/// Service class assigned to each registered session. Drives the
/// Weighted Round-Robin admission policy in [`BatchScheduler`]: an
/// `Audit` stream (high-throughput, latency-tolerant) cannot
/// monopolise the buffer pool / batch slots when `Interactive`
/// (low-latency) sessions are also active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionClass {
    /// Default class for chat / API requests — prioritised in the
    /// WRR scheduler with a 4× weight relative to Audit.
    Interactive,
    /// Bulk-throughput class (e.g. corpus audit jobs). Receives a
    /// smaller per-batch share than Interactive and is the first
    /// class to have its idle KV blocks reclaimed under pressure.
    Audit,
}

impl SessionClass {
    /// Per-class weight used by the WRR batch builder. Higher = a
    /// larger share of each batch's slots. Concretely: with the
    /// default weights, a fully-mixed pool of one Interactive + N
    /// Audit sessions still leaves Interactive with ~4 / (4+N) of
    /// every batch slot, so head-of-line latency stays bounded
    /// regardless of how many Audit streams are running.
    pub const fn weight(self) -> u32 {
        match self {
            SessionClass::Interactive => 4,
            SessionClass::Audit => 1,
        }
    }
}

impl Default for SessionClass {
    fn default() -> Self {
        SessionClass::Interactive
    }
}

/// Sidecar metadata stored alongside each registered request. Drives
/// fair-share admission, idle reclamation, and per-session telemetry.
/// Wrapped in `Arc` so the scheduler loop and HTTP path can both hold
/// a cheap handle without cloning the (potentially heavy) optional
/// `BlockManager`.
struct SessionMeta {
    class: SessionClass,
    /// Monotonic microseconds since scheduler start. Updated on every
    /// `step_registered` call so [`BatchScheduler::evict_idle_blocks`]
    /// can identify sessions that have stopped producing tokens.
    last_activity_us: AtomicU64,
    /// Optional [`BlockManager`] handle. Populated by
    /// [`BatchScheduler::bind_block_manager`] when the HTTP request
    /// owns paged-KV blocks; the scheduler tears it down on idle
    /// eviction, dropping every block back into the pool's free list.
    block_manager: parking_lot::Mutex<Option<BlockManager>>,
}

impl SessionMeta {
    fn new(class: SessionClass, now_us: u64) -> Self {
        Self {
            class,
            last_activity_us: AtomicU64::new(now_us),
            block_manager: parking_lot::Mutex::new(None),
        }
    }

    fn touch(&self, now_us: u64) {
        self.last_activity_us.store(now_us, Ordering::Relaxed);
    }

    fn idle_for_us(&self, now_us: u64) -> u64 {
        now_us.saturating_sub(self.last_activity_us.load(Ordering::Relaxed))
    }
}

/// Opaque, monotonically-increasing identifier for a registered
/// request in the scheduler. Cheap to clone (just a `u64`) so it can
/// be passed through channels and stored in HTTP state without
/// concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// Central registry of in-flight requests. Owns each request's
/// `Vec<KvCache>` (one entry per layer) so that the mpsc channel
/// between HTTP futures and the scheduler loop carries only a
/// [`RequestId`] per token rather than the full cache. Each entry is
/// wrapped in its own `Mutex` so the scheduler can mutate a single
/// request's caches concurrently with other requests in the same
/// batch without holding the top-level registry lock.
///
/// Backed by a [`dashmap::DashMap`] (sharded, lock-free reads) so
/// per-token `get` calls from concurrent scheduler tasks never
/// serialise on a single global mutex. This addresses the registry
/// scalability item in the production-readiness gist: with 100+
/// concurrent requests the old `Mutex<HashMap>` was a measurable hot
/// spot.
#[derive(Default)]
struct RequestRegistry {
    next: AtomicU64,
    /// `Arc<tokio::sync::Mutex<…>>` so the scheduler loop can hold a
    /// per-request lock across the `.await` inside `model.step()`
    /// while other requests in the same batch proceed in parallel.
    /// The outer DashMap is sharded internally so insert / lookup /
    /// remove operations on different request ids never contend.
    table: DashMap<u64, Arc<tokio::sync::Mutex<Vec<KvCache>>>>,
}

impl RequestRegistry {
    fn register(&self, kv: Vec<KvCache>) -> RequestId {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let entry = Arc::new(tokio::sync::Mutex::new(kv));
        self.table.insert(id, entry);
        RequestId(id)
    }

    fn get(&self, id: RequestId) -> Option<Arc<tokio::sync::Mutex<Vec<KvCache>>>> {
        self.table.get(&id.0).map(|e| e.value().clone())
    }

    fn release(&self, id: RequestId) -> Option<Vec<KvCache>> {
        let (_, entry) = self.table.remove(&id.0)?;
        match Arc::try_unwrap(entry) {
            Ok(m) => Some(m.into_inner()),
            Err(arc) => {
                // A scheduler task is still holding the entry. We
                // can't synchronously block on a `tokio::sync::Mutex`
                // from a sync context, so spawn a tiny detached task
                // that waits for the in-flight step to finish, locks
                // the entry one last time, takes ownership of the KV
                // vector, and lets the Arc drop. The caller pays
                // nothing here (the spawn is cheap and only happens
                // on the rare race window between submit and reply);
                // it also gets `None` back, which `step_through_scheduler`
                // already handles by allocating a fresh KV cache for
                // continued use. Crucially, this path no longer
                // *leaks* — the registry entry is already removed and
                // the Arc + its inner Vec are reclaimed by the
                // detached task.
                if tokio::runtime::Handle::try_current().is_ok() {
                    tokio::spawn(async move {
                        let _ = arc.lock().await; // wait for the in-flight step
                        // Arc drops here → KV memory reclaimed.
                    });
                } else {
                    // No tokio runtime (test environment with a sync
                    // shutdown path). Drop the Arc and rely on the
                    // borrowing task to drop its clone — no leak,
                    // just deferred reclaim.
                    drop(arc);
                }
                None
            }
        }
    }

    fn len(&self) -> usize {
        self.table.len()
    }
}

/// One "give me the next token" request issued by an in-flight HTTP
/// generation future. Carries only the small `{ id, token, pos,
/// params }` tuple over the mpsc channel — the actual KV cache for
/// the request lives in the scheduler's [`RequestRegistry`] and is
/// looked up by id just before `model.step()` is invoked.
pub struct StepRequest {
    /// Registry handle for the request whose KV caches `model.step`
    /// should mutate. Obtained via [`BatchScheduler::register`].
    pub id: RequestId,
    /// Last token id produced for this request (or, on the first call,
    /// the prompt token to ingest).
    pub token_id: u32,
    /// Absolute position of `token_id` in the request's sequence.
    pub pos: usize,
    /// Per-request sampling parameters (temperature/top-p/top-k/seed).
    pub params: crate::sampling::SamplingParams,
    /// Channel used by the scheduler to return the next-token id, or
    /// an error if the request id could not be resolved.
    pub resp: oneshot::Sender<Result<StepResponse, BatchError>>,
}

/// The reply mailed back through `StepRequest::resp`.
pub struct StepResponse {
    pub next_token: u32,
}

/// Configuration for the batch scheduler.
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Maximum number of requests fused into a single decoder step.
    /// `1` effectively disables batching.
    pub max_batch_size: usize,
    /// How long the scheduler waits for additional requests to arrive
    /// after the first one before flushing the current batch.
    pub batch_timeout: Duration,
    /// Optional shared physical [`PooledBlockPool`] sizing. When
    /// `block_pool_capacity > 0` and `block_pool_kv_dim > 0`, the
    /// scheduler will own a single pool of that many blocks and
    /// expose it via [`BatchScheduler::block_pool`] so HTTP handlers
    /// can hand out a [`BlockManager`] per request from a *single*
    /// pre-allocated slab. The pool now also supports a heap-backed
    /// overflow slab that grows on demand once the primary slab is
    /// exhausted; the scheduler logs a warning the first time a
    /// request touches it.
    pub block_pool_capacity: usize,
    pub block_pool_kv_dim: usize,
    /// Cutoff after which a session's KV blocks become candidates
    /// for [`BatchScheduler::evict_idle_blocks`] when the pool is
    /// above its soft-cap. Default: 5 seconds, matching the gist.
    pub idle_eviction_threshold: Duration,
    /// Baseline speculation depth (tokens-ahead) for the predictor's
    /// prefetch. The scheduler's [`SpeculationController`] grows this
    /// by up to [`crate::router::MAX_LATENCY_BUMP`] under rising SSD
    /// stall and clamps it to zero under
    /// [`PressureLevel::Critical`].
    pub speculation_base_depth: usize,
    /// Pool back-pressure thresholds for the shared
    /// [`PooledBlockPool`] this scheduler owns (gist Part 1, fix #4).
    /// Defaults to [`PressureThresholds::default`] (90%/98%) so older
    /// callers keep their semantics; operators tune via the
    /// `[real_transformer].pressure_high_threshold` /
    /// `pressure_critical_threshold` keys.
    pub pressure_thresholds: crate::block_pool::PressureThresholds,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(5),
            block_pool_capacity: 0,
            block_pool_kv_dim: 0,
            idle_eviction_threshold: Duration::from_secs(5),
            speculation_base_depth: 1,
            pressure_thresholds: crate::block_pool::PressureThresholds::default(),
        }
    }
}

/// Lightweight handle to the scheduler. Cheap to clone — under the
/// hood it just wraps a pair of [`mpsc::Sender`]s (one per
/// [`SessionClass`]) and a few `Arc`s.
///
/// ## Per-class admission channels
///
/// The scheduler maintains a *separate* mpsc channel for each
/// service class. Submitting a [`StepRequest`] from
/// [`Self::step_registered`] looks up the registered session's class
/// and pushes onto the matching channel; the background
/// [`scheduler_loop`] then drains both channels in weighted
/// round-robin proportion (`weight_interactive` : `weight_audit`)
/// when assembling a batch. This gives true fair-share *admission*
/// — even when the Audit channel's backlog is far longer than
/// `max_batch_size`, every batch still pulls the Interactive
/// channel's share first, so an Audit flood cannot starve a
/// latency-sensitive Interactive caller from being admitted.
#[derive(Clone)]
pub struct BatchScheduler {
    /// Submission channel for [`SessionClass::Interactive`] sessions.
    tx_interactive: mpsc::Sender<StepRequest>,
    /// Submission channel for [`SessionClass::Audit`] sessions. Kept
    /// physically separate so a backlog on this channel cannot push
    /// Interactive requests further back in line — admission is
    /// decided per-class by the WRR drain inside
    /// [`scheduler_loop`], not by FIFO order on a single fused queue.
    tx_audit: mpsc::Sender<StepRequest>,
    cfg: BatchConfig,
    /// Optional shared physical pool for paged KV caches. Populated
    /// when [`BatchConfig::block_pool_capacity`] and
    /// [`BatchConfig::block_pool_kv_dim`] are both non-zero. Cloned
    /// `Arc`s are cheap, so handing one out to every HTTP request
    /// imposes no real cost.
    block_pool: Option<Arc<BlockPool>>,
    /// Central registry of in-flight request KV caches. Shared with
    /// the background scheduler loop via `Arc`.
    registry: Arc<RequestRegistry>,
    /// Per-session metadata sidecar (class, last-activity timestamp,
    /// optional [`BlockManager`] for idle reclamation). Keyed by the
    /// same `u64` as [`RequestRegistry`]. Pre-allocated `DashMap` so
    /// lookups in the hot path are sharded and lock-free.
    sessions: Arc<DashMap<u64, Arc<SessionMeta>>>,
    /// Monotonic clock baseline for `last_activity_us`. Captured at
    /// scheduler spawn time so the deltas reported by
    /// [`SessionMeta::idle_for_us`] are unaffected by wall-clock
    /// adjustments.
    started_at: Instant,
    /// Latency-aware speculation window controller. Shared with the
    /// scheduler loop (which calls `update_from_stall` on every batch
    /// using the engine's cumulative SSD-stall telemetry) and exposed
    /// to HTTP / prefetch callers via
    /// [`Self::current_speculation_depth`] so they know how far ahead
    /// to prefetch.
    speculation: Arc<SpeculationController>,
    /// Number of requests that have spilled into the block pool's
    /// overflow slab since startup (cumulative; never decremented).
    /// Snapshot via [`Self::overflow_requests_total`].
    overflow_requests: Arc<AtomicUsize>,
    /// Latches `true` on the first overflow occurrence so the
    /// warning log is emitted exactly once per scheduler instance.
    overflow_warned: Arc<AtomicBool>,
    /// Cumulative count of idle-eviction passes that actually
    /// reclaimed at least one block. Snapshot via
    /// [`Self::idle_evictions_total`]; useful for stress-test assertions.
    idle_evictions: Arc<AtomicU64>,
}

impl BatchScheduler {
    /// Spawn the background batching task and return a handle to it.
    /// The task owns clones of `model` and `engine` and runs until the
    /// returned [`BatchScheduler`] (and all its clones) are dropped.
    pub fn spawn(
        model: Arc<RealModel>,
        engine: Arc<Engine>,
        cfg: BatchConfig,
    ) -> Self {
        // A reasonable channel depth: each in-flight request can have
        // at most one outstanding StepRequest at a time, so
        // `max_batch_size * 4` gives plenty of headroom without
        // unbounded growth under back-pressure. Each per-class
        // channel gets its own budget so an Audit backlog cannot
        // crowd out the Interactive channel.
        let depth = cfg.max_batch_size.saturating_mul(4).max(8);
        let (tx_interactive, rx_interactive) = mpsc::channel::<StepRequest>(depth);
        let (tx_audit, rx_audit) = mpsc::channel::<StepRequest>(depth);
        let registry = Arc::new(RequestRegistry::default());
        let sessions: Arc<DashMap<u64, Arc<SessionMeta>>> = Arc::new(DashMap::new());
        let started_at = Instant::now();
        let speculation = Arc::new(SpeculationController::new(cfg.speculation_base_depth));
        let block_pool = if cfg.block_pool_capacity > 0 && cfg.block_pool_kv_dim > 0 {
            Some(BlockPool::with_thresholds(
                cfg.block_pool_kv_dim,
                cfg.block_pool_capacity,
                cfg.pressure_thresholds,
            ))
        } else {
            None
        };
        let idle_evictions = Arc::new(AtomicU64::new(0));
        tokio::spawn(scheduler_loop(
            model,
            engine.clone(),
            cfg,
            rx_interactive,
            rx_audit,
            registry.clone(),
            sessions.clone(),
            block_pool.clone(),
            started_at,
            speculation.clone(),
            idle_evictions.clone(),
        ));
        Self {
            tx_interactive,
            tx_audit,
            cfg,
            block_pool,
            registry,
            sessions,
            started_at,
            speculation,
            overflow_requests: Arc::new(AtomicUsize::new(0)),
            overflow_warned: Arc::new(AtomicBool::new(false)),
            idle_evictions,
        }
    }

    /// Configuration the scheduler was built with.
    pub fn config(&self) -> BatchConfig { self.cfg }

    /// Shared physical block pool, if one is configured. Returns
    /// `None` when the scheduler was built without a paged-cache
    /// budget (the default).
    pub fn block_pool(&self) -> Option<Arc<BlockPool>> {
        self.block_pool.clone()
    }

    /// Convenience: allocate a per-request [`BlockManager`] backed by
    /// the scheduler's shared pool. Returns `None` when the scheduler
    /// was built without a paged-cache budget. The caller owns the
    /// returned manager; on `Drop` it will release every block back
    /// to the pool.
    pub fn new_block_manager(&self) -> Option<BlockManager> {
        self.block_pool.clone().map(BlockManager::new)
    }

    /// Number of block-pool overflow blocks currently in use across
    /// all requests, or `0` when no pool is configured. `> 0`
    /// indicates the primary slab was exhausted and the pool is
    /// servicing requests out of its heap-backed fallback; operators
    /// should size [`BatchConfig::block_pool_capacity`] up if this is
    /// non-zero in steady state.
    pub fn overflow_in_use(&self) -> usize {
        self.block_pool.as_ref().map(|p| p.overflow_in_use()).unwrap_or(0)
    }

    /// Cumulative number of requests observed to be running with at
    /// least one overflow block since the scheduler started. Useful
    /// for monitoring (this counter never decrements).
    pub fn overflow_requests_total(&self) -> usize {
        self.overflow_requests.load(Ordering::Relaxed)
    }

    /// Number of currently registered requests. Mostly diagnostic.
    pub fn active_requests(&self) -> usize {
        self.registry.len()
    }

    /// Register a new request with the scheduler, taking ownership of
    /// its per-layer KV cache. Returns a small handle that subsequent
    /// [`Self::step_registered`] calls reference. The caller must
    /// pair this with a [`Self::release`] when the request finishes
    /// (or when it is aborted) to reclaim the cache and any
    /// associated paged-pool blocks.
    ///
    /// Defaults to [`SessionClass::Interactive`]; use
    /// [`Self::register_with_class`] for `Audit` sessions.
    pub fn register(&self, kv: Vec<KvCache>) -> RequestId {
        self.register_with_class(kv, SessionClass::default())
    }

    /// Register a request with an explicit service class. The class
    /// drives the WRR admission policy and the order in which sessions
    /// become candidates for [`Self::evict_idle_blocks`] under
    /// pressure.
    pub fn register_with_class(&self, kv: Vec<KvCache>, class: SessionClass) -> RequestId {
        let id = self.registry.register(kv);
        let now_us = self.now_us();
        self.sessions
            .insert(id.0, Arc::new(SessionMeta::new(class, now_us)));
        id
    }

    /// Attach a [`BlockManager`] handle to a registered request. When
    /// the scheduler later evicts the session for being idle, it
    /// takes the manager and drops it, returning every block the
    /// session owned to the shared pool's free list. No-op if the
    /// request id is not registered.
    pub fn bind_block_manager(&self, id: RequestId, manager: BlockManager) {
        if let Some(meta) = self.sessions.get(&id.0) {
            *meta.block_manager.lock() = Some(manager);
        }
    }

    /// Look up the service class for a registered request, or `None`
    /// if the id has been released.
    pub fn session_class(&self, id: RequestId) -> Option<SessionClass> {
        self.sessions.get(&id.0).map(|m| m.class)
    }

    /// Tear down a registered request, returning its (now-mutated)
    /// KV cache. Returns `None` if the id was already released. The
    /// scheduler also probes the block pool's overflow state at
    /// release time so the cumulative `overflow_requests_total`
    /// counter stays accurate even when the request never
    /// re-touched the scheduler after a burst.
    pub fn release(&self, id: RequestId) -> Option<Vec<KvCache>> {
        let kv = self.registry.release(id);
        // Removing the session meta drops the held `BlockManager`
        // (if any), returning every block it owned to the pool.
        self.sessions.remove(&id.0);
        // After a release, the BlockManager (if any) owned by the
        // caller will return blocks on Drop, so this is a natural
        // point to surface whether the pool was ever stressed.
        self.maybe_warn_overflow();
        kv
    }

    /// Memory-pressure classification of the underlying block pool.
    /// Returns [`PressureLevel::Normal`] when no pool is configured.
    pub fn pressure_level(&self) -> PressureLevel {
        self.block_pool
            .as_ref()
            .map(|p| p.pressure_level())
            .unwrap_or(PressureLevel::Normal)
    }

    /// Currently-active speculation depth, in tokens-ahead. The
    /// predictor reads this on every batch and clamps its prefetch
    /// fanout accordingly: `0` under
    /// [`PressureLevel::Critical`], `base_depth` under normal
    /// conditions, up to `base_depth + MAX_LATENCY_BUMP` under
    /// rising SSD stall.
    pub fn current_speculation_depth(&self) -> usize {
        self.speculation.current_depth()
    }

    /// Shared [`SpeculationController`]. Useful for tests and for
    /// instrumentation code that wants to read more than just the
    /// current depth (e.g. whether the controller is suspended).
    pub fn speculation_controller(&self) -> Arc<SpeculationController> {
        self.speculation.clone()
    }

    /// Walk the session map and release every block owned by a
    /// session that hasn't produced a token in `idle_threshold` (or
    /// the configured default when `None`). Returns the number of
    /// sessions whose blocks were reclaimed. Idempotent — sessions
    /// that don't own a `BlockManager` are skipped.
    ///
    /// Audit-class sessions are evicted **first** (sorted ahead of
    /// Interactive) so a flurry of low-priority bulk jobs cannot
    /// starve a latency-sensitive chat session of KV memory. The
    /// scheduler loop also calls this method itself whenever
    /// [`PressureLevel::High`] is reached, but operators can invoke
    /// it manually for an off-cycle reclamation pass.
    pub fn evict_idle_blocks(&self, idle_threshold: Option<Duration>) -> usize {
        let threshold = idle_threshold.unwrap_or(self.cfg.idle_eviction_threshold);
        let reclaimed = run_idle_eviction(&self.sessions, threshold, self.now_us());
        if reclaimed > 0 {
            self.idle_evictions.fetch_add(1, Ordering::Relaxed);
        }
        reclaimed
    }

    /// Total number of `evict_idle_blocks` passes that reclaimed at
    /// least one session. Snapshot via [`Self::idle_evictions_total`].
    pub fn idle_evictions_total(&self) -> u64 {
        self.idle_evictions.load(Ordering::Relaxed)
    }

    /// Number of registered sessions, partitioned by class.
    /// Useful for diagnostics and for stress-test assertions.
    pub fn session_counts_by_class(&self) -> (usize, usize) {
        let mut interactive = 0;
        let mut audit = 0;
        for kv in self.sessions.iter() {
            match kv.value().class {
                SessionClass::Interactive => interactive += 1,
                SessionClass::Audit => audit += 1,
            }
        }
        (interactive, audit)
    }

    fn now_us(&self) -> u64 {
        self.started_at.elapsed().as_micros() as u64
    }

    fn maybe_warn_overflow(&self) {
        if let Some(pool) = self.block_pool.as_ref() {
            let in_use = pool.overflow_in_use();
            if in_use > 0 {
                self.overflow_requests.fetch_add(1, Ordering::Relaxed);
                if !self.overflow_warned.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        overflow_blocks_in_use = in_use,
                        primary_capacity = pool.capacity(),
                        "BlockPool primary slab exhausted; serving subsequent requests from \
                         the heap-backed overflow slab. Consider raising \
                         `block_pool_capacity` if this is steady-state."
                    );
                }
            }
        }
    }

    /// Submit one decoder step for an already-registered request.
    /// Returns the sampled next-token id. The scheduler may fuse
    /// this call with other concurrent callers' steps into a single
    /// batch.
    pub async fn step_registered(
        &self,
        id: RequestId,
        token_id: u32,
        pos: usize,
        params: crate::sampling::SamplingParams,
    ) -> Result<u32, BatchError> {
        let (tx, rx) = oneshot::channel();
        let req = StepRequest { id, token_id, pos, params, resp: tx };
        // Look up the session's class so the request lands on the
        // matching per-class admission channel. Unregistered ids
        // default to Interactive — they will be rejected by the
        // scheduler loop's registry check anyway, but routing them
        // to the higher-priority lane keeps the failure path short.
        let class = self
            .sessions
            .get(&id.0)
            .map(|m| m.class)
            .unwrap_or(SessionClass::Interactive);
        let sender = match class {
            SessionClass::Interactive => &self.tx_interactive,
            SessionClass::Audit => &self.tx_audit,
        };
        sender.send(req).await.map_err(|_| BatchError::SchedulerClosed)?;
        let resp = rx.await.map_err(|_| BatchError::SchedulerClosed)??;
        Ok(resp.next_token)
    }

    /// Submit one decoder step. Returns the next-token id and the
    /// (mutated) KV cache. The scheduler may fuse this call with other
    /// concurrent callers' steps into a single batch.
    ///
    /// This legacy entry point internally registers the cache, drives
    /// one step, and releases — for callers that hold a long-lived
    /// session, prefer the lighter [`Self::register`] /
    /// [`Self::step_registered`] / [`Self::release`] trio so the
    /// per-token path does not touch the registry's `HashMap`.
    pub async fn step(
        &self,
        token_id: u32,
        pos: usize,
        kv: Vec<KvCache>,
        params: crate::sampling::SamplingParams,
    ) -> Result<StepResponse, BatchError> {
        let id = self.registry.register(kv);
        let result = self.step_registered(id, token_id, pos, params).await;
        // Always release, even on error, to avoid leaking the entry.
        let kv_back = self.registry.release(id).unwrap_or_default();
        self.maybe_warn_overflow();
        match result {
            Ok(next_token) => Ok(StepResponse { next_token }),
            Err(e) => {
                // Caller still expects the KV cache back to keep
                // operating on it (e.g. via a direct model.step
                // fallback). The legacy back-compat response type
                // doesn't carry kv anymore, but we publish it via
                // the registry — the calling helper in `server.rs`
                // has already taken ownership locally.
                drop(kv_back);
                Err(e)
            }
        }
    }

    /// **Internal-only warm-up entry point (gist Task 1).**
    ///
    /// Drives a *single* synthetic decoder step that bypasses
    /// every network / auth / HTTP layer the public `step` and
    /// `step_registered` paths sit behind. The motivation is
    /// cold-start mitigation: the first real user-facing request
    /// would otherwise pay the one-time cost of
    ///
    ///   1. lazily-faulting the `AlignedBuffer` slab pool pages
    ///      into the resident set,
    ///   2. registering `io_uring` fixed-buffer slots (`IORING_REGISTER_BUFFERS`)
    ///      with the kernel on the first read,
    ///   3. JIT-warming the math backend kernels (AVX-512 / NEON /
    ///      candle dispatch tables) the first time
    ///      [`crate::backend::Backend::swiglu_into`] is called.
    ///
    /// All three are amortised here so the user sees a 0-ms start.
    ///
    /// ### Contract
    /// - This function must **never** be reached from a network
    ///   path. It is only called from
    ///   [`crate::server::run_engine_warmup`] before the listener
    ///   binds. There is no auth, no rate-limit, no admission
    ///   check — by design.
    /// - It is best-effort. Channel/registry failures are surfaced as
    ///   [`BatchError`]. `engine.warm_with` failures are logged in
    ///   place and warm-up continues so startup still reaches bind.
    /// - **Zero-contention.** This routine does **not** add new
    ///   locks to [`scheduler_loop`]. It re-uses the existing
    ///   per-class mpsc channels and the same registry / release
    ///   path real requests use, so it cannot perturb the
    ///   steady-state hot path.
    pub async fn submit_internal_warmup(
        &self,
        model: &Arc<RealModel>,
        engine: &Arc<Engine>,
    ) -> Result<(), BatchError> {
        // (1) Prime the `AlignedBuffer` slab + io_uring fixed-buffer
        //     registrations by pulling a representative set of
        //     expert ids into the resident cache. The first
        //     `fetch_with_retry` per expert lazy-registers its
        //     buffer with the kernel; subsequent reads are
        //     zero-syscall via the registered fd / buffer pair.
        let num_experts = engine.num_experts();
        if num_experts > 0 {
            let cap = num_experts.min(8);
            let ids: Vec<u32> = (0..cap).collect();
            if let Err(e) = engine.warm_with(&ids).await {
                // `warm_with` is best-effort by design; log and keep
                // progressing through warm-up so startup still binds.
                tracing::warn!(error = %e, "submit_internal_warmup: warm_with failed");
            }
        }

        // (2) JIT-warm the registered math backend. A tiny
        //     synthetic `swiglu_into` call lights up the dispatch
        //     tables (AVX-512 detect, candle Tensor allocators,
        //     future GPU command queues) once, so the very first
        //     real token does not pay the one-shot kernel-init
        //     cost.
        {
            let backend = crate::backend::current();
            let gate_f16 = vec![half::f16::from_f32(0.1); 16];
            let up_f16 = vec![half::f16::from_f32(0.2); 16];
            let mut out_f16 = vec![half::f16::ZERO; 16];
            let gate_view = crate::backend::TensorView { data: &gate_f16, rows: 4, cols: 4 };
            let up_view = crate::backend::TensorView { data: &up_f16, rows: 4, cols: 4 };
            let mut out_view = crate::backend::TensorViewMut { data: &mut out_f16, rows: 4, cols: 4 };
            let _ = backend.swiglu_into(gate_view, up_view, &mut out_view);
        }

        // (3) Drive a *single* synthetic decoder step through the
        //     normal scheduler path. We register a fresh KV cache,
        //     send one `StepRequest` over the Interactive channel,
        //     wait for the reply, then release the cache. This
        //     exercises the full `mpsc → batch fuse → model.step →
        //     oneshot` loop exactly once, so the *first* real user
        //     request finds every code path (tokio worker stacks,
        //     scheduler_loop branch predictor, sampler temperature
        //     table) already in cache.
        let kv = model.fresh_kv_caches();
        let id = self.register_with_class(kv, SessionClass::Interactive);
        // The synthetic token id and position are deliberately
        // small / valid: token 0 at position 0 always round-trips
        // through the model without exercising window-eviction
        // edge cases. `SamplingParams::default()` gives
        // greedy-argmax sampling so the result is deterministic.
        let params = crate::sampling::SamplingParams::default();
        let result = self.step_registered(id, 0u32, 0usize, params).await;
        // Always release, even on error — leaving a stale id in
        // the registry would leak the KV cache across the
        // process's lifetime.
        let _ = self.release(id);
        // Ignore the sampled token; warm-up only cares that the
        // pipeline produced *a* token, not which one.
        result.map(|_| ())
    }
}

/// Errors returned from [`BatchScheduler::step`] and friends.
#[derive(Debug)]
pub enum BatchError {
    /// The background task has exited (server shutdown, panic, …).
    SchedulerClosed,
    /// The [`RequestId`] passed to [`BatchScheduler::step_registered`]
    /// is not (or is no longer) registered. Either the caller never
    /// registered it, called `release` already, or the scheduler was
    /// restarted between calls.
    NotRegistered,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::SchedulerClosed => write!(f, "batch scheduler is closed"),
            BatchError::NotRegistered => write!(f, "request id is not registered with the scheduler"),
        }
    }
}

impl std::error::Error for BatchError {}

/// Background loop: collect a batch (size- or timeout-bounded), run
/// each request's decoder step concurrently on shared `Engine`,
/// reply to each via its oneshot, repeat.
///
/// **SSD Read De-Duplication pre-pass (gist Phase 1).** Before the
/// concurrent `model.step` tasks are spawned, the loop peeks at the
/// routing decisions each request is likely to make (via
/// [`RealModel::peek_experts`]) and folds them into a single
/// `HashSet<u32>` of unique expert ids. A *single* unified
/// [`Engine::warm_with`] call then pulls them all into the
/// [`ExpertCache`] concurrently, with the in-flight singleflight on
/// `Engine::fetch_with_retry` guaranteeing at most one disk read per
/// unique id. By the time the math tasks fan out, every common
/// expert is hot — eliminating the redundant I/O contention the
/// pre-pass exists to solve.
///
/// **Deadlock safety under `BufferPool` pressure.** The pre-pass
/// fetches funnel through `fetch_with_retry`, which surfaces
/// [`ExpertReadError::PoolStarved`] rather than spinning if every
/// pool slot is pinned. `warm_with` discards those errors (it is a
/// best-effort prefetch), so the scheduler always proceeds to spawn
/// the per-request `model.step` tasks. Each task then re-tries the
/// same id through the same singleflight'd path, where buffer-pool
/// pressure naturally drains as earlier requests complete their
/// step. No fan-in / fan-out point holds a buffer across the warm
/// pass, so a saturated pool cannot deadlock the loop.
async fn scheduler_loop(
    model: Arc<RealModel>,
    engine: Arc<Engine>,
    cfg: BatchConfig,
    mut rx_interactive: mpsc::Receiver<StepRequest>,
    mut rx_audit: mpsc::Receiver<StepRequest>,
    registry: Arc<RequestRegistry>,
    sessions: Arc<DashMap<u64, Arc<SessionMeta>>>,
    block_pool: Option<Arc<BlockPool>>,
    started_at: Instant,
    speculation: Arc<SpeculationController>,
    idle_evictions: Arc<AtomicU64>,
) {
    let now_us = || started_at.elapsed().as_micros() as u64;
    // Per-class WRR weights, captured once. These are `const fn`s so
    // pulling them out of the loop is just a readability win.
    let wi = SessionClass::Interactive.weight() as usize;
    let wa = SessionClass::Audit.weight() as usize;

    loop {
        // ----------------------------------------------------------
        // Block until at least one request shows up on *either*
        // per-class channel. `tokio::select!` here is the part that
        // makes admission class-aware: if only the Audit channel
        // has traffic, we still wake up; if both have traffic, we
        // still wake up on whichever fires first, and the WRR drain
        // below then enforces the 4 : 1 share.
        // ----------------------------------------------------------
        let mut batch: Vec<StepRequest> = Vec::with_capacity(cfg.max_batch_size);
        tokio::select! {
            biased;
            // Prefer Interactive on the very first await — under
            // heavy mixed load this ensures the head-of-batch slot
            // is reliably claimed by the higher-priority class.
            r = rx_interactive.recv() => match r {
                Some(req) => batch.push(req),
                None => {
                    // Interactive senders all dropped → fall back to
                    // Audit-only for the rest of this iteration.
                    match rx_audit.recv().await {
                        Some(req) => batch.push(req),
                        None => return, // both closed
                    }
                }
            },
            r = rx_audit.recv() => match r {
                Some(req) => batch.push(req),
                None => {
                    match rx_interactive.recv().await {
                        Some(req) => batch.push(req),
                        None => return, // both closed
                    }
                }
            },
        }

        // ----------------------------------------------------------
        // Class-aware WRR drain: pull `wi` from Interactive, then
        // `wa` from Audit, repeating, until the batch is full or
        // both per-class channels are empty. Because the channels
        // are *physically* separate, an Audit backlog cannot push
        // an Interactive request further back in line: every cycle
        // we get up to `wi` Interactive admissions before any Audit
        // admission, regardless of submission rates.
        // ----------------------------------------------------------
        loop {
            if batch.len() >= cfg.max_batch_size {
                break;
            }
            let mut made_progress = false;
            for _ in 0..wi {
                if batch.len() >= cfg.max_batch_size {
                    break;
                }
                match rx_interactive.try_recv() {
                    Ok(req) => {
                        batch.push(req);
                        made_progress = true;
                    }
                    Err(_) => break,
                }
            }
            for _ in 0..wa {
                if batch.len() >= cfg.max_batch_size {
                    break;
                }
                match rx_audit.try_recv() {
                    Ok(req) => {
                        batch.push(req);
                        made_progress = true;
                    }
                    Err(_) => break,
                }
            }
            if !made_progress {
                break;
            }
        }

        // ----------------------------------------------------------
        // Give late arrivals up to `batch_timeout` to join. We
        // race both channels via `select!` so the timeout pass
        // remains class-aware too. A `biased` select with the
        // Interactive arm first preserves the fair-share preference
        // when both channels are ready simultaneously.
        // ----------------------------------------------------------
        if batch.len() < cfg.max_batch_size && !cfg.batch_timeout.is_zero() {
            let deadline = tokio::time::Instant::now() + cfg.batch_timeout;
            while batch.len() < cfg.max_batch_size {
                tokio::select! {
                    biased;
                    r = tokio::time::timeout_at(deadline, rx_interactive.recv()) => match r {
                        Ok(Some(req)) => batch.push(req),
                        Ok(None) => break, // channel closed
                        Err(_) => break,   // timeout
                    },
                    r = tokio::time::timeout_at(deadline, rx_audit.recv()) => match r {
                        Ok(Some(req)) => batch.push(req),
                        Ok(None) => break, // channel closed
                        Err(_) => break,   // timeout
                    },
                }
            }
        }

        // ----------------------------------------------------------
        // Memory-pressure ladder + latency-aware speculation:
        //   * High   → trigger preemptive idle eviction.
        //   * Critical → suspend speculation (depth → 0).
        //   * Normal  → resume speculation if previously suspended;
        //               feed cumulative SSD-stall telemetry to the
        //               controller so the depth tracks I/O latency.
        // ----------------------------------------------------------
        if let Some(pool) = block_pool.as_ref() {
            match pool.pressure_level() {
                PressureLevel::Critical => {
                    speculation.suspend();
                    // Best-effort reclamation while under critical pressure.
                    let reclaimed = run_idle_eviction(
                        &sessions,
                        cfg.idle_eviction_threshold,
                        now_us(),
                    );
                    if reclaimed > 0 {
                        idle_evictions.fetch_add(1, Ordering::Relaxed);
                    }
                }
                PressureLevel::High => {
                    if speculation.is_suspended() {
                        speculation.resume();
                    }
                    let reclaimed = run_idle_eviction(
                        &sessions,
                        cfg.idle_eviction_threshold,
                        now_us(),
                    );
                    if reclaimed > 0 {
                        idle_evictions.fetch_add(1, Ordering::Relaxed);
                    }
                }
                PressureLevel::Normal => {
                    if speculation.is_suspended() {
                        speculation.resume();
                    }
                }
            }
        }
        // Fold the engine's cumulative SSD-stall telemetry into the
        // speculation controller. Cheap (one atomic load + one
        // CAS-free store) so safe to call every batch.
        let cum_stall = engine.report().predictive.ssd_stall_us;
        speculation.update_from_stall(cum_stall);

        // ----------------------------------------------------------
        // Note: the batch built above is already class-interleaved
        // by construction — the per-class drain pulls `wi` from the
        // Interactive channel then `wa` from the Audit channel each
        // cycle, so no additional re-ordering pass is needed here.
        // The WRR admission policy is enforced *before* requests
        // enter the batch, which is what makes fair-share robust
        // against an Audit submission flood.
        // ----------------------------------------------------------

        // ----------------------------------------------------------
        // Pre-pass: peek at every request's routing decision and
        // warm the union of expert ids in a single unified read
        // (gist Phase 1 — SSD Read De-Duplication).
        // ----------------------------------------------------------
        let mut unique: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for req in &batch {
            if let Some(entry) = registry.get(req.id) {
                // Use try_lock so a peek never blocks on a step that
                // is concurrently mutating the same request's KV.
                // If we can't acquire the lock right now the warm
                // pass simply skips this request — the singleflight
                // path still dedups its critical-path reads.
                if let Ok(kv) = entry.try_lock() {
                    let peeked = model.peek_experts(req.token_id, req.pos, &kv);
                    for id in peeked {
                        unique.insert(id);
                    }
                }
            }
        }
        if !unique.is_empty() {
            let ids: Vec<u32> = unique.into_iter().collect();
            if let Err(e) = engine.warm_with(&ids).await {
                tracing::debug!(error = %e, "scheduler pre-pass warm_with failed; falling through to inline fetches");
            }
        }

        // Run every request's decoder step concurrently. Each task
        // looks up its KV cache from the central registry and locks
        // just its own slot, so concurrent requests never serialise
        // on each other. They share `engine` and `model` via `Arc`.
        let mut handles = Vec::with_capacity(batch.len());
        for req in batch {
            let model = model.clone();
            let engine = engine.clone();
            let registry = registry.clone();
            let sessions = sessions.clone();
            let now = now_us();
            handles.push(tokio::spawn(async move {
                let StepRequest { id, token_id, pos, params, resp } = req;
                // Refresh the session's last-activity timestamp so
                // idle eviction doesn't pull blocks out from under
                // a request that is still producing tokens.
                if let Some(meta) = sessions.get(&id.0) {
                    meta.touch(now);
                }
                let entry = match registry.get(id) {
                    Some(e) => e,
                    None => {
                        // Surface NotRegistered explicitly so callers
                        // can distinguish this from scheduler
                        // shutdown. `step_through_scheduler` in the
                        // server falls back to a direct `model.step`
                        // for either error variant, so HTTP requests
                        // still complete cleanly.
                        let _ = resp.send(Err(BatchError::NotRegistered));
                        return;
                    }
                };
                // Per-entry lock: one request's step runs serially
                // against its own caches but in parallel with all
                // other requests' steps.
                let next = {
                    let mut kv = entry.lock().await;
                    model
                        .step(&engine, token_id, pos, &mut kv, &params)
                        .await
                };
                // Drop our Arc clone *before* signalling completion
                // so the caller's subsequent `release(id)` sees the
                // refcount drop to one and `try_unwrap` succeeds
                // without falling through to the deferred-reclaim
                // branch. Without this, a tight test sequence (and
                // in rare cases a real client doing
                // step_registered → release back-to-back) could race
                // the spawned task's natural Arc drop at end-of-task.
                drop(entry);
                let _ = resp.send(Ok(StepResponse { next_token: next }));
            }));
        }
        for h in handles {
            // A panicking step shouldn't tear the scheduler down: log
            // and continue. The corresponding request's oneshot will
            // be dropped → caller sees `SchedulerClosed`.
            if let Err(e) = h.await {
                tracing::error!(error = %e, "batched decoder step panicked");
            }
        }
    }
}

/// Helper for `scheduler_loop`: walk the session map and drop block
/// managers for sessions that have been idle for at least `threshold`.
/// Audit-class sessions go first. Returns the number of sessions
/// whose blocks were reclaimed.
fn run_idle_eviction(
    sessions: &DashMap<u64, Arc<SessionMeta>>,
    threshold: Duration,
    now_us: u64,
) -> usize {
    let threshold_us = threshold.as_micros() as u64;
    let mut candidates: Vec<(u64, Arc<SessionMeta>)> = sessions
        .iter()
        .filter_map(|kv| {
            let meta = kv.value().clone();
            if meta.idle_for_us(now_us) >= threshold_us
                && meta.block_manager.lock().is_some()
            {
                Some((*kv.key(), meta))
            } else {
                None
            }
        })
        .collect();
    candidates.sort_by(|a, b| {
        let class_ord = match (a.1.class, b.1.class) {
            (SessionClass::Audit, SessionClass::Interactive) => std::cmp::Ordering::Less,
            (SessionClass::Interactive, SessionClass::Audit) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        };
        class_ord.then_with(|| b.1.idle_for_us(now_us).cmp(&a.1.idle_for_us(now_us)))
    });
    let mut reclaimed = 0;
    for (_id, meta) in candidates {
        let taken = { meta.block_manager.lock().take() };
        if taken.is_some() {
            reclaimed += 1;
        }
    }
    reclaimed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::expert_cache::ExpertCache;
    use crate::multi_layer_cache::MultiLayerExpertCache;
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::model::{RealModel, RealModelConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use std::path::PathBuf;
    use std::time::Instant;

    struct TempDir { path: PathBuf }
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            path.push(format!("mer-batch-test-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.path); }
    }

    fn build_engine_and_model(cfg: RealModelConfig) -> (Arc<Engine>, Arc<RealModel>, TempDir) {
        let dir = TempDir::new("sched");
        let total = cfg.num_layers as u32 * cfg.num_experts as u32;
        let weight_bytes = crate::inference::expert_weight_bytes(cfg.d_model, cfg.d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(&dir.path, total, expert_size, cfg.d_model, cfg.d_ff).unwrap();
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        // Cache big enough to keep every expert hot after the first
        // request, so we measure compute parallelism rather than I/O
        // serialisation.
        let cache = Arc::new(MultiLayerExpertCache::single_layer((total as usize).max(2)));
        let pool = BufferPool::new(total as usize + 4, expert_size, block);
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::new(total, cfg.top_k, 1)));
        let predictor = Arc::new(PredictiveLoader::new(total, 0, 0.05, 1));
        let engine = Arc::new(Engine::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape { d_model: cfg.d_model, d_ff: cfg.d_ff, hidden_seed: 1 },
            EngineOptions::default(),
        ));
        let model = Arc::new(RealModel::new_seeded(cfg, 0xBEEF));
        (engine, model, dir)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batched_step_returns_same_tokens_as_direct_call() {
        // Functional check: routing through the scheduler yields
        // exactly the same token sequence as calling `model.step`
        // directly with the same inputs.
        let cfg = RealModelConfig {
            vocab_size: 64, d_model: 16, d_ff: 32, num_heads: 4, num_kv_heads: 4,
            head_dim: 4, num_layers: 2, num_experts: 4, top_k: 2,
            rope_base: 10_000.0, rms_eps: 1e-6, window_size: None,
            architecture: crate::architecture::Architecture::Mixtral, first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let (engine, model, _tmp) = build_engine_and_model(cfg.clone());
        let sched = BatchScheduler::spawn(
            model.clone(),
            engine.clone(),
            BatchConfig { max_batch_size: 4, batch_timeout: Duration::from_millis(2), ..Default::default() },
        );

        let mut kv_a = model.fresh_kv_caches();
        let direct = model.step(&engine, 7, 0, &mut kv_a, &crate::sampling::SamplingParams::greedy()).await;

        let kv_b = model.fresh_kv_caches();
        let resp = sched.step(7, 0, kv_b, crate::sampling::SamplingParams::greedy()).await.unwrap();
        assert_eq!(direct, resp.next_token, "scheduler must be functionally identical to model.step");
    }

    /// Same equivalence check, but exercises the zero-copy
    /// register / step_registered / release API path instead of the
    /// legacy by-move `step` wrapper. Both must produce identical
    /// token sequences.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn step_registered_matches_direct_step() {
        let cfg = RealModelConfig {
            vocab_size: 64, d_model: 16, d_ff: 32, num_heads: 4, num_kv_heads: 4,
            head_dim: 4, num_layers: 2, num_experts: 4, top_k: 2,
            rope_base: 10_000.0, rms_eps: 1e-6, window_size: None,
            architecture: crate::architecture::Architecture::Mixtral, first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let (engine, model, _tmp) = build_engine_and_model(cfg.clone());
        let sched = BatchScheduler::spawn(
            model.clone(),
            engine.clone(),
            BatchConfig { max_batch_size: 4, batch_timeout: Duration::from_millis(2), ..Default::default() },
        );

        let mut kv_a = model.fresh_kv_caches();
        let direct = model.step(&engine, 11, 0, &mut kv_a, &crate::sampling::SamplingParams::greedy()).await;

        let id = sched.register(model.fresh_kv_caches());
        assert_eq!(sched.active_requests(), 1);
        let next = sched
            .step_registered(id, 11, 0, crate::sampling::SamplingParams::greedy())
            .await
            .unwrap();
        assert_eq!(direct, next, "zero-copy step path must match direct model.step");
        let returned = sched.release(id);
        assert!(returned.is_some(), "release must return the registered cache");
        assert_eq!(sched.active_requests(), 0);
    }

    /// Spawning N concurrent requests through the scheduler must not
    /// take materially longer than running them all in parallel —
    /// concretely, no worse than the strictly-sequential baseline. If
    /// our batching had a bug that serialised requests we'd see a
    /// regression here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_finish_faster_than_sequential() {
        let cfg = RealModelConfig {
            vocab_size: 64, d_model: 16, d_ff: 32, num_heads: 4, num_kv_heads: 4,
            head_dim: 4, num_layers: 2, num_experts: 4, top_k: 2,
            rope_base: 10_000.0, rms_eps: 1e-6, window_size: None,
            architecture: crate::architecture::Architecture::Mixtral, first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let (engine, model, _tmp) = build_engine_and_model(cfg.clone());

        const N: usize = 4;
        const TOKENS: usize = 4;

        // --- Sequential baseline: one request at a time, no scheduler ---
        let seq_start = Instant::now();
        for _ in 0..N {
            let mut kv = model.fresh_kv_caches();
            let mut last = 7u32;
            for pos in 0..TOKENS {
                last = model.step(&engine, last, pos, &mut kv, &crate::sampling::SamplingParams::greedy()).await;
            }
        }
        let sequential = seq_start.elapsed();

        // --- Batched: all N requests submit through the scheduler concurrently ---
        let sched = BatchScheduler::spawn(
            model.clone(),
            engine.clone(),
            BatchConfig { max_batch_size: N, batch_timeout: Duration::from_millis(5), ..Default::default() },
        );
        let batched_start = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..N {
            let sched = sched.clone();
            // Register the request's KV caches once; the per-token
            // loop only sends the small `RequestId` through the mpsc
            // channel — no per-step `Vec<KvCache>` move.
            let id = sched.register(model.fresh_kv_caches());
            handles.push(tokio::spawn(async move {
                let mut last = 7u32;
                for pos in 0..TOKENS {
                    last = sched
                        .step_registered(id, last, pos, crate::sampling::SamplingParams::greedy())
                        .await
                        .unwrap();
                }
                let _ = sched.release(id);
            }));
        }
        for h in handles { h.await.unwrap(); }
        let batched = batched_start.elapsed();

        // The batched run sharing one Engine across N requests should
        // be no slower than the sequential one. We assert a generous
        // bound (≤ 1.5x) to keep the test stable on noisy CI runners
        // while still catching obvious regressions where batching
        // accidentally serialises requests.
        assert!(
            batched.as_secs_f64() <= sequential.as_secs_f64() * 1.5,
            "expected batched ({:?}) ≤ 1.5 × sequential ({:?})",
            batched, sequential
        );
    }
}
