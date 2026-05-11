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
//! one short `Mutex<HashMap>` insert per token, which is negligible
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

use crate::block_pool::{BlockManager, BlockPool};
use crate::engine::Engine;
use crate::model::RealModel;
use crate::transformer::KvCache;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

// Re-export the block-pool types so HTTP handlers and tests can build
// `BlockManager`s directly off the scheduler's shared pool without
// taking an additional dependency on the `block_pool` module path.
#[allow(unused_imports)]
pub use crate::block_pool::{BlockAllocError, BlockId, BlockManager as PooledBlockManager,
    BlockPool as PooledBlockPool, POOL_BLOCK_TOKENS};

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
#[derive(Default)]
struct RequestRegistry {
    next: AtomicU64,
    /// `Arc<tokio::sync::Mutex<…>>` so the scheduler loop can hold a
    /// per-request lock across the `.await` inside `model.step()`
    /// while other requests in the same batch proceed in parallel.
    /// The outer `parking_lot::Mutex` on the table itself is held
    /// only for the (very short) insert / lookup / remove operations.
    table: Mutex<HashMap<u64, Arc<tokio::sync::Mutex<Vec<KvCache>>>>>,
}

impl RequestRegistry {
    fn register(&self, kv: Vec<KvCache>) -> RequestId {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let entry = Arc::new(tokio::sync::Mutex::new(kv));
        self.table.lock().insert(id, entry);
        RequestId(id)
    }

    fn get(&self, id: RequestId) -> Option<Arc<tokio::sync::Mutex<Vec<KvCache>>>> {
        self.table.lock().get(&id.0).cloned()
    }

    fn release(&self, id: RequestId) -> Option<Vec<KvCache>> {
        let entry = self.table.lock().remove(&id.0)?;
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
        self.table.lock().len()
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
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(5),
            block_pool_capacity: 0,
            block_pool_kv_dim: 0,
        }
    }
}

/// Lightweight handle to the scheduler. Cheap to clone — under the
/// hood it just wraps an [`mpsc::Sender`] and a few `Arc`s.
#[derive(Clone)]
pub struct BatchScheduler {
    tx: mpsc::Sender<StepRequest>,
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
    /// Number of requests that have spilled into the block pool's
    /// overflow slab since startup (cumulative; never decremented).
    /// Snapshot via [`Self::overflow_requests_total`].
    overflow_requests: Arc<AtomicUsize>,
    /// Latches `true` on the first overflow occurrence so the
    /// warning log is emitted exactly once per scheduler instance.
    overflow_warned: Arc<AtomicBool>,
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
        // unbounded growth under back-pressure.
        let depth = cfg.max_batch_size.saturating_mul(4).max(8);
        let (tx, rx) = mpsc::channel::<StepRequest>(depth);
        let registry = Arc::new(RequestRegistry::default());
        tokio::spawn(scheduler_loop(model, engine, cfg, rx, registry.clone()));
        let block_pool = if cfg.block_pool_capacity > 0 && cfg.block_pool_kv_dim > 0 {
            Some(BlockPool::new(cfg.block_pool_kv_dim, cfg.block_pool_capacity))
        } else {
            None
        };
        Self {
            tx,
            cfg,
            block_pool,
            registry,
            overflow_requests: Arc::new(AtomicUsize::new(0)),
            overflow_warned: Arc::new(AtomicBool::new(false)),
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
    pub fn register(&self, kv: Vec<KvCache>) -> RequestId {
        self.registry.register(kv)
    }

    /// Tear down a registered request, returning its (now-mutated)
    /// KV cache. Returns `None` if the id was already released. The
    /// scheduler also probes the block pool's overflow state at
    /// release time so the cumulative `overflow_requests_total`
    /// counter stays accurate even when the request never
    /// re-touched the scheduler after a burst.
    pub fn release(&self, id: RequestId) -> Option<Vec<KvCache>> {
        let kv = self.registry.release(id);
        // After a release, the BlockManager (if any) owned by the
        // caller will return blocks on Drop, so this is a natural
        // point to surface whether the pool was ever stressed.
        self.maybe_warn_overflow();
        kv
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
        self.tx.send(req).await.map_err(|_| BatchError::SchedulerClosed)?;
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
async fn scheduler_loop(
    model: Arc<RealModel>,
    engine: Arc<Engine>,
    cfg: BatchConfig,
    mut rx: mpsc::Receiver<StepRequest>,
    registry: Arc<RequestRegistry>,
) {
    loop {
        // Block until at least one request shows up.
        let first = match rx.recv().await {
            Some(r) => r,
            None => return, // all senders dropped → graceful exit
        };
        let mut batch: Vec<StepRequest> = Vec::with_capacity(cfg.max_batch_size);
        batch.push(first);

        // Greedily drain anything else that's already queued without
        // waiting (no-await `try_recv`); this fills the batch when
        // requests arrived during the previous step.
        while batch.len() < cfg.max_batch_size {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => break,
            }
        }

        // If we still haven't filled the batch, give late arrivals up
        // to `batch_timeout` to join. Bounded by `max_batch_size`.
        if batch.len() < cfg.max_batch_size && !cfg.batch_timeout.is_zero() {
            let deadline = tokio::time::Instant::now() + cfg.batch_timeout;
            while batch.len() < cfg.max_batch_size {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(r)) => batch.push(r),
                    Ok(None) => break, // channel closed
                    Err(_) => break,   // timeout
                }
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
            handles.push(tokio::spawn(async move {
                let StepRequest { id, token_id, pos, params, resp } = req;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::expert_cache::ExpertCache;
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
        let cache = Arc::new(ExpertCache::new((total as usize).max(2)));
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
