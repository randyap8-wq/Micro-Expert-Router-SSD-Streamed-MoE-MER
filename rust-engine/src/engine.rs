//! Top-level engine that wires the router, cache, buffer pool, predictive
//! loader, storage, and inference placeholder together.
//!
//! Responsibilities of [`Engine::generate`]:
//!
//! 1. Ask the router which experts a given token needs.
//! 2. For each required expert, look up the cache:
//!    * **Hit** — clone the `Arc<ExpertResident>` and pass it to inference.
//!    * **Miss** — acquire a buffer from the pool, dispatch a (real)
//!      io_uring read, install the resident in the cache, then run inference.
//! 3. Run the placeholder inference function on the bytes.
//! 4. Update the predictive Markov model with the observed transition.
//! 5. Speculatively kick off prefetches for the most likely next experts.
//! 6. Record per-token latency and emit structured tracing events.

use crate::buffer_pool::BufferPool;
use crate::expert_cache::{ExpertCache, ExpertResident};
use crate::inference::{
    combine_outputs, run_inference, synth_hidden_state, HiddenState, InferenceOutput,
};
use crate::io_provider::NvmeStorage;
use crate::router::{PredictiveLoader, TopKRouter};
use hdrhistogram::Histogram;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

#[derive(Debug, Default, Clone, Copy)]
pub struct CycleStats {
    pub hits: u64,
    pub misses: u64,
    pub prefetch_hits: u64,
    pub bytes_read: u64,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    prefetch_completed: AtomicU64,
    prefetch_used: AtomicU64,
    bytes_read: AtomicU64,
}

/// Shape parameters of the SwiGLU expert FFN executed by the engine.
///
/// Each on-disk expert file is a flat blob of `f32` weights laid out as
/// `gate_proj || up_proj || down_proj` (see [`crate::inference`]).
#[derive(Clone, Copy, Debug)]
pub struct ModelShape {
    pub d_model: usize,
    pub d_ff: usize,
    /// Seed used to derive per-token hidden states. In a real model this
    /// would come from the previous transformer layer; here it lets us
    /// produce reproducible activations for the synthetic stream.
    pub hidden_seed: u64,
}

pub struct Engine {
    cache: Arc<ExpertCache>,
    pool: BufferPool,
    storage: Arc<NvmeStorage>,
    router: Arc<TopKRouter>,
    predictor: Arc<PredictiveLoader>,
    shape: ModelShape,
    counters: Arc<Counters>,
    /// Latency histogram of per-token cycle time, in microseconds.
    cycle_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Latency histogram of cache-miss I/O reads, in microseconds.
    io_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Latency histogram of per-token compute (FFN forward), in microseconds.
    compute_hist: parking_lot::Mutex<Histogram<u64>>,
    last_experts: parking_lot::Mutex<Vec<u32>>,
}

impl Engine {
    pub fn new(
        cache: Arc<ExpertCache>,
        pool: BufferPool,
        storage: Arc<NvmeStorage>,
        router: Arc<TopKRouter>,
        predictor: Arc<PredictiveLoader>,
        shape: ModelShape,
    ) -> Self {
        Self {
            cache,
            pool,
            storage,
            router,
            predictor,
            shape,
            counters: Arc::new(Counters::default()),
            cycle_hist: parking_lot::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3)
                    .expect("hdr histogram bounds (1us..60s, 3 sig figs) are valid"),
            ),
            io_hist: parking_lot::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3)
                    .expect("hdr histogram bounds (1us..60s, 3 sig figs) are valid"),
            ),
            compute_hist: parking_lot::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3)
                    .expect("hdr histogram bounds (1us..60s, 3 sig figs) are valid"),
            ),
            last_experts: parking_lot::Mutex::new(Vec::new()),
        }
    }

    pub fn shape(&self) -> ModelShape {
        self.shape
    }

    /// Process a single token: route, fetch missing experts, run inference,
    /// update predictor, and kick off prefetches. Returns one [`CycleStats`].
    pub async fn generate(self: &Arc<Self>, token_idx: u64) -> CycleStats {
        let cycle_start = Instant::now();
        let target = self.router.route(token_idx);
        let mut stats = CycleStats::default();

        // 1) Make sure every required expert is resident.
        let mut residents = Vec::with_capacity(target.len());
        for &id in &target {
            if let Some(r) = self.cache.get(id) {
                self.counters.hits.fetch_add(1, Ordering::Relaxed);
                stats.hits += 1;
                debug!(expert = id, "cache hit");
                residents.push(r);
            } else {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                stats.misses += 1;
                debug!(expert = id, "cache miss, fetching from NVMe");
                let r = self.fetch(id).await;
                stats.bytes_read += r.buffer.len() as u64;
                self.counters
                    .bytes_read
                    .fetch_add(r.buffer.len() as u64, Ordering::Relaxed);
                residents.push(r);
            }
        }

        // 2) Real expert FFN forward pass over weights streamed from SSD.
        //    `synth_hidden_state` mocks the residual-stream activation that
        //    would normally come from the previous transformer layer.
        let x: HiddenState = synth_hidden_state(token_idx, self.shape.d_model, self.shape.hidden_seed);
        let compute_start = Instant::now();
        let mut per_expert_y: Vec<HiddenState> = Vec::with_capacity(residents.len());
        let mut outputs: Vec<InferenceOutput> = Vec::with_capacity(residents.len());
        for r in &residents {
            match run_inference(token_idx, r, &x, self.shape.d_model, self.shape.d_ff) {
                Ok((out, y)) => {
                    outputs.push(out);
                    per_expert_y.push(y);
                }
                Err(e) => {
                    // The on-disk file is truncated / corrupt or violates
                    // an alignment invariant. Log and skip this expert
                    // rather than aborting the whole token cycle / run.
                    warn!(
                        token = token_idx,
                        expert = r.id,
                        error = %e,
                        "skipping expert: failed to reinterpret buffer as SwiGLU weights"
                    );
                }
            }
        }
        let combined = combine_outputs(&per_expert_y);
        let compute_us = compute_start.elapsed().as_micros() as u64;
        let _ = self.compute_hist.lock().record(compute_us.max(1));
        debug!(
            token = token_idx,
            d_model = self.shape.d_model,
            d_ff = self.shape.d_ff,
            ?outputs,
            combined_norm = combined.iter().map(|v| v * v).sum::<f32>().sqrt(),
            "FFN forward complete"
        );

        // 3) Update predictor with the observed transition.
        {
            let mut last = self.last_experts.lock();
            if !last.is_empty() {
                self.predictor.observe_step(&last, &target);
            }
            *last = target.clone();
        }

        // 4) Kick off speculative prefetches for the most-recent expert.
        if let Some(&seed) = target.last() {
            let preds = self.predictor.predict_next(seed);
            for (id, p) in preds {
                if !self.cache.contains(id) {
                    self.spawn_prefetch(id, p);
                }
            }
        }

        let cycle_us = cycle_start.elapsed().as_micros() as u64;
        let _ = self.cycle_hist.lock().record(cycle_us.max(1));

        stats
    }

    async fn fetch(self: &Arc<Self>, id: u32) -> Arc<ExpertResident> {
        let io_start = Instant::now();
        // Race-free acquire-with-eviction: in a loop, evict an LRU entry
        // if the cache is at capacity (which releases its `PooledBuffer` on
        // Arc drop), then try to acquire. If a concurrent prefetch task
        // grabbed the freed slot before we did, we evict another LRU and
        // retry. This guarantees forward progress on the required path.
        let mut buf;
        loop {
            if self.cache.len() >= self.cache.capacity() {
                if let Some(evicted) = self.cache.evict_lru() {
                    debug!(evicted = evicted.id, "evicted LRU to make room");
                    drop(evicted);
                }
            }
            if let Some(b) = self.pool.try_acquire() {
                buf = b;
                break;
            }
            // Pool is empty even though cache is below capacity — i.e. some
            // other task (prefetch or another fetch) is holding buffers.
            // Yield to the runtime briefly to let them make progress.
            tokio::task::yield_now().await;
        }
        match self.storage.read_expert(id, &mut buf).await {
            Ok(_) => {
                let io_us = io_start.elapsed().as_micros() as u64;
                let _ = self.io_hist.lock().record(io_us.max(1));
                let resident = Arc::new(ExpertResident { id, buffer: buf });
                if let Some(_evicted) = self.cache.insert(resident.clone()) {
                    debug!(expert = id, "inserted (with eviction)");
                } else {
                    debug!(expert = id, "inserted");
                }
                resident
            }
            Err(e) => {
                // The buffer is returned to the pool when `buf` is dropped.
                warn!(expert = id, error = %e, "expert read failed");
                // Surface the error by panicking — the engine cannot make
                // progress without the requested expert. A production build
                // would route around the failure or retry.
                panic!("failed to read expert {id}: {e}");
            }
        }
    }

    fn spawn_prefetch(self: &Arc<Self>, id: u32, p: f64) {
        let me = self.clone();
        tokio::spawn(async move {
            // Re-check (could have been loaded by another task in the meantime).
            if me.cache.contains(id) {
                return;
            }
            // Prefetches are *speculative*. They must never evict resident
            // experts (which could starve a real cache miss) and must never
            // block waiting for a buffer (same reason). The buffer pool is
            // sized with extra slots specifically for in-flight prefetches.
            let mut buf = match me.pool.try_acquire() {
                Some(b) => b,
                None => {
                    debug!(expert = id, "skipping prefetch: pool busy");
                    return;
                }
            };
            let started = Instant::now();
            match me.storage.read_expert(id, &mut buf).await {
                Ok(_) => {
                    me.counters.prefetch_completed.fetch_add(1, Ordering::Relaxed);
                    me.counters
                        .bytes_read
                        .fetch_add(buf.len() as u64, Ordering::Relaxed);
                    let resident = Arc::new(ExpertResident { id, buffer: buf });
                    me.cache.insert(resident);
                    debug!(
                        expert = id,
                        prob = p,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "prefetch complete"
                    );
                }
                Err(e) => warn!(expert = id, error = %e, "prefetch failed"),
            }
        });
    }

    /// Account for the fact that an expert was a hit *because* we prefetched it.
    pub fn note_prefetch_hit(&self) {
        self.counters.prefetch_used.fetch_add(1, Ordering::Relaxed);
    }

    /// Force-fetch a specific set of experts and load them into the cache.
    /// Mirrors the spec example "the router selects Expert ID 3 and 7".
    pub async fn warm_with(self: &Arc<Self>, ids: &[u32]) -> std::io::Result<()> {
        for &id in ids {
            if id >= self.router.num_experts() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("expert id {id} >= num_experts"),
                ));
            }
            if !self.cache.contains(id) {
                let _ = self.fetch(id).await;
            }
        }
        Ok(())
    }

    pub fn report(&self) -> EngineReport {
        let cycle = self.cycle_hist.lock();
        let io = self.io_hist.lock();
        let compute = self.compute_hist.lock();
        EngineReport {
            hits: self.counters.hits.load(Ordering::Relaxed),
            misses: self.counters.misses.load(Ordering::Relaxed),
            prefetch_completed: self.counters.prefetch_completed.load(Ordering::Relaxed),
            bytes_read: self.counters.bytes_read.load(Ordering::Relaxed),
            cycle_p50_us: cycle.value_at_quantile(0.50),
            cycle_p95_us: cycle.value_at_quantile(0.95),
            cycle_p99_us: cycle.value_at_quantile(0.99),
            cycle_max_us: cycle.max(),
            io_p50_us: io.value_at_quantile(0.50),
            io_p95_us: io.value_at_quantile(0.95),
            io_p99_us: io.value_at_quantile(0.99),
            io_count: io.len(),
            compute_p50_us: compute.value_at_quantile(0.50),
            compute_p95_us: compute.value_at_quantile(0.95),
            compute_p99_us: compute.value_at_quantile(0.99),
            cache_capacity: self.cache.capacity(),
            pool_capacity: self.pool.capacity(),
            num_experts: self.router.num_experts(),
            top_k: self.router.k(),
            d_model: self.shape.d_model,
            d_ff: self.shape.d_ff,
            predictor_observations: self.predictor.observations(),
        }
    }

    pub fn print_summary(&self) {
        let r = self.report();
        let total = r.hits + r.misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            r.hits as f64 / total as f64 * 100.0
        };
        info!("===================== run summary =====================");
        info!(
            "experts:       {} (top-{}), cache={} slots, pool={} slots",
            r.num_experts, r.top_k, r.cache_capacity, r.pool_capacity
        );
        info!(
            "ffn shape:     d_model={}  d_ff={}  bytes/expert={}",
            r.d_model,
            r.d_ff,
            crate::inference::expert_weight_bytes(r.d_model, r.d_ff)
        );
        info!(
            "lookups:       hits={}  misses={}  hit_rate={:.2}%",
            r.hits, r.misses, hit_rate
        );
        info!(
            "prefetches:    completed={}  predictor_observations={}",
            r.prefetch_completed, r.predictor_observations
        );
        info!(
            "i/o:           reads={}  bytes={:.2} MiB",
            r.io_count,
            r.bytes_read as f64 / (1024.0 * 1024.0)
        );
        info!(
            "i/o latency:   p50={}us  p95={}us  p99={}us",
            r.io_p50_us, r.io_p95_us, r.io_p99_us
        );
        info!(
            "compute:       p50={}us  p95={}us  p99={}us  (SwiGLU FFN per token)",
            r.compute_p50_us, r.compute_p95_us, r.compute_p99_us
        );
        info!(
            "cycle latency: p50={}us  p95={}us  p99={}us  max={}us",
            r.cycle_p50_us, r.cycle_p95_us, r.cycle_p99_us, r.cycle_max_us
        );
        info!("=======================================================");
    }
}

#[derive(Debug, Clone)]
pub struct EngineReport {
    pub hits: u64,
    pub misses: u64,
    pub prefetch_completed: u64,
    pub bytes_read: u64,
    pub cycle_p50_us: u64,
    pub cycle_p95_us: u64,
    pub cycle_p99_us: u64,
    pub cycle_max_us: u64,
    pub io_p50_us: u64,
    pub io_p95_us: u64,
    pub io_p99_us: u64,
    pub io_count: u64,
    pub compute_p50_us: u64,
    pub compute_p95_us: u64,
    pub compute_p99_us: u64,
    pub cache_capacity: usize,
    pub pool_capacity: usize,
    pub num_experts: u32,
    pub top_k: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub predictor_observations: u64,
}
