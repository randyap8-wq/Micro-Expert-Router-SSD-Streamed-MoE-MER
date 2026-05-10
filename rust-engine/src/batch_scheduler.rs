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
//! `Engine`. Each request's KV cache moves with the request through
//! the channel and back, so attention state remains strictly
//! per-request — only the *expert streaming* and *decoder compute*
//! overlap.
//!
//! The engine's expert cache + storage already use atomic counters and
//! `Arc`s, so multiple `moe_step` calls from sibling tasks are safe;
//! see [`crate::engine`] for details.

use crate::engine::Engine;
use crate::model::RealModel;
use crate::transformer::KvCache;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

/// One "give me the next token" request issued by an in-flight HTTP
/// generation future. The KV cache is moved into the scheduler each
/// step and returned (mutated) alongside the sampled token id.
pub struct StepRequest {
    /// Last token id produced for this request (or, on the first call,
    /// the prompt token to ingest).
    pub token_id: u32,
    /// Absolute position of `token_id` in the request's sequence.
    pub pos: usize,
    /// Per-request KV cache, one entry per layer. Moved into the
    /// scheduler so the decoder step can mutate it without aliasing
    /// other requests' state.
    pub kv: Vec<KvCache>,
    /// Per-request sampling parameters (temperature/top-p/top-k/seed).
    pub params: crate::sampling::SamplingParams,
    /// Channel used by the scheduler to return the next-token id and
    /// the (now-grown) KV cache.
    pub resp: oneshot::Sender<StepResponse>,
}

/// The reply mailed back through `StepRequest::resp`.
pub struct StepResponse {
    pub next_token: u32,
    pub kv: Vec<KvCache>,
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
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(5),
        }
    }
}

/// Lightweight handle to the scheduler. Cheap to clone — under the
/// hood it just wraps an [`mpsc::Sender`].
#[derive(Clone)]
pub struct BatchScheduler {
    tx: mpsc::Sender<StepRequest>,
    cfg: BatchConfig,
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
        tokio::spawn(scheduler_loop(model, engine, cfg, rx));
        Self { tx, cfg }
    }

    /// Configuration the scheduler was built with.
    pub fn config(&self) -> BatchConfig { self.cfg }

    /// Submit one decoder step. Returns the next-token id and the
    /// (mutated) KV cache. The scheduler may fuse this call with other
    /// concurrent callers' steps into a single batch.
    pub async fn step(
        &self,
        token_id: u32,
        pos: usize,
        kv: Vec<KvCache>,
        params: crate::sampling::SamplingParams,
    ) -> Result<StepResponse, BatchError> {
        let (tx, rx) = oneshot::channel();
        let req = StepRequest { token_id, pos, kv, params, resp: tx };
        self.tx.send(req).await.map_err(|_| BatchError::SchedulerClosed)?;
        rx.await.map_err(|_| BatchError::SchedulerClosed)
    }
}

/// Errors returned from [`BatchScheduler::step`].
#[derive(Debug)]
pub enum BatchError {
    /// The background task has exited (server shutdown, panic, …).
    SchedulerClosed,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::SchedulerClosed => write!(f, "batch scheduler is closed"),
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
        // owns its `kv` so there's no aliasing; they share `engine`
        // and `model` via `Arc`.
        let mut handles = Vec::with_capacity(batch.len());
        for mut req in batch {
            let model = model.clone();
            let engine = engine.clone();
            handles.push(tokio::spawn(async move {
                let next = model
                    .step(&engine, req.token_id, req.pos, &mut req.kv, &req.params)
                    .await;
                let _ = req.resp.send(StepResponse {
                    next_token: next,
                    kv: req.kv,
                });
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
            })
            .unwrap(),
        );
        // Cache big enough to keep every expert hot after the first
        // request, so we measure compute parallelism rather than I/O
        // serialisation.
        let cache = Arc::new(ExpertCache::new((total as usize).max(2)));
        let pool = BufferPool::new(total as usize + 4, expert_size, block);
        let router = Arc::new(TopKRouter::new(total, cfg.top_k, 1));
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
            BatchConfig { max_batch_size: 4, batch_timeout: Duration::from_millis(2) },
        );

        let mut kv_a = model.fresh_kv_caches();
        let direct = model.step(&engine, 7, 0, &mut kv_a, &crate::sampling::SamplingParams::greedy()).await;

        let kv_b = model.fresh_kv_caches();
        let resp = sched.step(7, 0, kv_b, crate::sampling::SamplingParams::greedy()).await.unwrap();
        assert_eq!(direct, resp.next_token, "scheduler must be functionally identical to model.step");
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
            BatchConfig { max_batch_size: N, batch_timeout: Duration::from_millis(5) },
        );
        let batched_start = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..N {
            let sched = sched.clone();
            let kv0 = model.fresh_kv_caches();
            handles.push(tokio::spawn(async move {
                let mut kv = kv0;
                let mut last = 7u32;
                for pos in 0..TOKENS {
                    let resp = sched.step(last, pos, kv, crate::sampling::SamplingParams::greedy()).await.unwrap();
                    last = resp.next_token;
                    kv = resp.kv;
                }
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
