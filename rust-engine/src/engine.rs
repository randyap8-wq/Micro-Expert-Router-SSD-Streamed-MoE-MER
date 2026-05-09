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

/// Run-time options that affect how `Engine::generate` executes a token.
///
/// The defaults model a normal end-to-end run (router → I/O → SwiGLU
/// FFN); `io_only` flips off the FFN compute so the same instrumentation
/// can be used to measure pure I/O cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineOptions {
    /// When `true`, skip [`run_inference`] and instead XOR every byte of
    /// the resident buffer to force the read to fully materialise. This
    /// isolates the SSD-streaming cost from FFN compute and is what
    /// `--io-only` on the CLI maps to.
    pub io_only: bool,
}

pub struct Engine {
    cache: Arc<ExpertCache>,
    pool: BufferPool,
    storage: Arc<NvmeStorage>,
    router: Arc<TopKRouter>,
    predictor: Arc<PredictiveLoader>,
    shape: ModelShape,
    options: EngineOptions,
    counters: Arc<Counters>,
    /// Latency histogram of per-token cycle time, in microseconds.
    cycle_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Latency histogram of cache-miss I/O reads, in microseconds.
    io_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Latency histogram of per-token compute (FFN forward), in microseconds.
    compute_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Aggregate microseconds spent on I/O wait across all tokens (i.e.
    /// the sum of per-token critical-path miss latencies). Lets us
    /// report `avg_io_wait_us` and "% of token time on I/O" without
    /// re-deriving them from the histogram.
    total_io_wait_us: AtomicU64,
    /// Aggregate microseconds spent on per-token compute across all tokens.
    total_compute_us: AtomicU64,
    /// Aggregate microseconds spent on per-token cycle (compute + I/O wait
    /// + scheduling overhead) across all tokens.
    total_cycle_us: AtomicU64,
    /// Number of tokens processed (i.e. `Engine::generate` calls).
    tokens_processed: AtomicU64,
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
        Self::with_options(cache, pool, storage, router, predictor, shape, EngineOptions::default())
    }

    pub fn with_options(
        cache: Arc<ExpertCache>,
        pool: BufferPool,
        storage: Arc<NvmeStorage>,
        router: Arc<TopKRouter>,
        predictor: Arc<PredictiveLoader>,
        shape: ModelShape,
        options: EngineOptions,
    ) -> Self {
        Self {
            cache,
            pool,
            storage,
            router,
            predictor,
            shape,
            options,
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
            total_io_wait_us: AtomicU64::new(0),
            total_compute_us: AtomicU64::new(0),
            total_cycle_us: AtomicU64::new(0),
            tokens_processed: AtomicU64::new(0),
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
        //
        // Cache-miss reads are issued concurrently. Two routed experts
        // that both miss kick off two `pread(2)` calls in parallel via
        // `tokio::spawn`, so the NVMe queue actually sees the queue depth
        // the routing decision implies; sequentially `await`-ing each
        // fetch would serialise an opportunity the device can already
        // satisfy concurrently. Hits are resolved inline.
        let io_wait_start = Instant::now();
        let mut residents: Vec<Option<Arc<ExpertResident>>> = vec![None; target.len()];
        let mut miss_handles: Vec<(usize, tokio::task::JoinHandle<Arc<ExpertResident>>)> =
            Vec::new();
        for (i, &id) in target.iter().enumerate() {
            if let Some(r) = self.cache.get(id) {
                self.counters.hits.fetch_add(1, Ordering::Relaxed);
                stats.hits += 1;
                debug!(expert = id, "cache hit");
                residents[i] = Some(r);
            } else {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                stats.misses += 1;
                debug!(expert = id, "cache miss, fetching from NVMe");
                let me = self.clone();
                miss_handles.push((
                    i,
                    tokio::spawn(async move { me.fetch(id).await }),
                ));
            }
        }
        let had_misses = !miss_handles.is_empty();
        for (i, h) in miss_handles {
            // `fetch` panics on a fatal read error (the engine cannot
            // make progress without the requested expert); propagate by
            // unwrapping the `JoinError` so the panic surfaces exactly
            // as it did before this was made concurrent.
            let r = h.await.expect("expert fetch task panicked");
            stats.bytes_read += r.buffer.len() as u64;
            self.counters
                .bytes_read
                .fetch_add(r.buffer.len() as u64, Ordering::Relaxed);
            residents[i] = Some(r);
        }
        let io_wait_us = if had_misses {
            io_wait_start.elapsed().as_micros() as u64
        } else {
            0
        };
        let residents: Vec<Arc<ExpertResident>> = residents
            .into_iter()
            .map(|r| r.expect("internal invariant: every routed expert slot must be populated by either a hit or a completed miss fetch"))
            .collect();

        // 2) Either run the real SwiGLU FFN, or — under `--io-only` —
        //    just touch every byte of the resident buffer with a cheap
        //    XOR checksum so the kernel actually delivers the page data
        //    and we can isolate the SSD-streaming cost from FFN compute.
        let compute_start = Instant::now();
        let compute_us = if self.options.io_only {
            let mut digest: u64 = 0;
            let mut total_bytes: u64 = 0;
            for r in &residents {
                let bytes = r.data();
                total_bytes += bytes.len() as u64;
                // XOR every byte. The accumulator is 64 bits wide so we
                // also rotate per chunk; this prevents a smart compiler
                // from folding the loop and guarantees every read byte
                // is observed, the whole point of `--io-only`.
                let mut acc: u64 = 0;
                for chunk in bytes.chunks(8) {
                    // Final chunk may be < 8 bytes; the remaining slots
                    // in `buf` stay zero. XOR with zero is a no-op, so
                    // the digest is still deterministic and every
                    // actually-read byte still contributes.
                    let mut buf = [0u8; 8];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    acc ^= u64::from_le_bytes(buf);
                }
                // `% 63` (deliberately not 64): `rotate_left(0)` and
                // `rotate_left(64)` are both no-ops on `u64`. Using 63
                // keeps the rotation amount in `0..63` so adjacent
                // expert ids actually pick different rotations and
                // the per-expert contributions don't collapse.
                digest ^= acc.rotate_left((r.id % 63) as u32);
            }
            let us = compute_start.elapsed().as_micros() as u64;
            debug!(
                token = token_idx,
                bytes_touched = total_bytes,
                io_only_digest = digest,
                "io-only mode: skipped FFN, touched buffer bytes"
            );
            us
        } else {
            // Real expert FFN forward pass over weights streamed from SSD.
            // `synth_hidden_state` mocks the residual-stream activation that
            // would normally come from the previous transformer layer.
            let x: HiddenState =
                synth_hidden_state(token_idx, self.shape.d_model, self.shape.hidden_seed);
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
            let us = compute_start.elapsed().as_micros() as u64;
            debug!(
                token = token_idx,
                d_model = self.shape.d_model,
                d_ff = self.shape.d_ff,
                ?outputs,
                combined_norm = combined.iter().map(|v| v * v).sum::<f32>().sqrt(),
                "FFN forward complete"
            );
            us
        };
        let _ = self.compute_hist.lock().record(compute_us.max(1));
        self.total_compute_us.fetch_add(compute_us, Ordering::Relaxed);
        self.total_io_wait_us.fetch_add(io_wait_us, Ordering::Relaxed);

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
        self.total_cycle_us.fetch_add(cycle_us, Ordering::Relaxed);
        self.tokens_processed.fetch_add(1, Ordering::Relaxed);

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
        let tokens = self.tokens_processed.load(Ordering::Relaxed);
        let total_io_wait_us = self.total_io_wait_us.load(Ordering::Relaxed);
        let total_compute_us = self.total_compute_us.load(Ordering::Relaxed);
        let total_cycle_us = self.total_cycle_us.load(Ordering::Relaxed);
        let avg_io_wait_us = if tokens == 0 { 0.0 } else { total_io_wait_us as f64 / tokens as f64 };
        let avg_compute_us = if tokens == 0 { 0.0 } else { total_compute_us as f64 / tokens as f64 };
        let pct_time_io = if total_cycle_us == 0 {
            0.0
        } else {
            (total_io_wait_us as f64 / total_cycle_us as f64) * 100.0
        };
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
            tokens_processed: tokens,
            avg_io_wait_us,
            avg_compute_us,
            total_io_wait_us,
            total_cycle_us,
            pct_time_io,
            io_only: self.options.io_only,
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
            "compute:       p50={}us  p95={}us  p99={}us  ({})",
            r.compute_p50_us,
            r.compute_p95_us,
            r.compute_p99_us,
            if r.io_only { "io-only XOR digest, FFN skipped" } else { "SwiGLU FFN per token" }
        );
        info!(
            "cycle latency: p50={}us  p95={}us  p99={}us  max={}us",
            r.cycle_p50_us, r.cycle_p95_us, r.cycle_p99_us, r.cycle_max_us
        );
        info!(
            "per-token avg: io_wait={:.1}us  compute={:.1}us  (over {} tokens)",
            r.avg_io_wait_us, r.avg_compute_us, r.tokens_processed
        );
        info!(
            "I/O share:     {:.2}% of token cycle time spent waiting on SSD reads",
            r.pct_time_io
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
    /// Number of `Engine::generate` calls completed.
    pub tokens_processed: u64,
    /// Mean per-token critical-path I/O wait, in microseconds. Tokens that
    /// were entirely served from cache contribute 0 to this average.
    pub avg_io_wait_us: f64,
    /// Mean per-token compute (FFN forward, or XOR-digest under
    /// `--io-only`), in microseconds.
    pub avg_compute_us: f64,
    /// Sum of per-token critical-path I/O wait (microseconds).
    pub total_io_wait_us: u64,
    /// Sum of per-token cycle time (microseconds).
    pub total_cycle_us: u64,
    /// `total_io_wait_us / total_cycle_us * 100` — the headline "what
    /// fraction of token time was the engine waiting on SSD?" number
    /// the gist asks the run summary to print.
    pub pct_time_io: f64,
    /// Whether this run was executed in `--io-only` mode (FFN skipped).
    pub io_only: bool,
}

#[cfg(test)]
mod tests {
    //! Integration test for the full `Engine::generate` loop.
    //!
    //! Wires the real `NvmeStorage` (with `O_DIRECT` disabled — required on
    //! tmpfs/CI), real `BufferPool`, real `ExpertCache`, real `TopKRouter`
    //! and `PredictiveLoader` against on-disk synthetic experts written by
    //! `generate_synthetic_experts`, and runs many tokens through
    //! `Engine::generate`. This is the "no integration tests for the full
    //! Engine::generate loop" gap closed.
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::expert_cache::ExpertCache;
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Self-cleaning unique temp directory for test fixtures.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            // Combine pid + monotonic counter + nanos for uniqueness across
            // parallel test runs without pulling in a tempfile dependency.
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "micro-expert-router-{label}-{}-{n}-{ts}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn build_engine(
        data_dir: &std::path::Path,
        num_experts: u32,
        d_model: usize,
        d_ff: usize,
        cache_slots: usize,
        top_k: usize,
        predict_fanout: usize,
        seed: u64,
    ) -> Arc<Engine> {
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        // Round expert_size up to a multiple of block_align (an O_DIRECT
        // invariant the storage layer asserts even when --no-direct is set).
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;

        generate_synthetic_experts(data_dir, num_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");

        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: data_dir.to_path_buf(),
                expert_size,
                block_align,
                // tmpfs / overlayfs (typical for CI) doesn't support O_DIRECT.
                use_direct_io: false,
            })
            .expect("storage init"),
        );
        storage
            .warmup_fds(0..num_experts)
            .expect("pre-open expert fds");

        let pool_slots = cache_slots + predict_fanout.max(1);
        let pool = BufferPool::new(pool_slots, expert_size, block_align);
        let cache = Arc::new(ExpertCache::new(cache_slots));
        let router = Arc::new(TopKRouter::new(num_experts, top_k, seed));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, predict_fanout, 0.05, seed));

        Arc::new(Engine::new(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape { d_model, d_ff, hidden_seed: seed },
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_loop_routes_fetches_and_runs_inference() {
        let dir = TempDir::new("gen-integration");
        let num_experts: u32 = 16;
        let top_k = 2;
        let d_model = 32;
        let d_ff = 64;
        let cache_slots = 8;
        let predict_fanout = 2;
        let tokens: u64 = 64;

        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0xC0FFEE,
        );

        let mut total_hits = 0u64;
        let mut total_misses = 0u64;
        let mut total_bytes = 0u64;
        for t in 0..tokens {
            let s = engine.generate(t).await;
            total_hits += s.hits;
            total_misses += s.misses;
            total_bytes += s.bytes_read;
        }

        // Every token routes to exactly `top_k` experts, so the cumulative
        // hit + miss count must be exactly `tokens * top_k`.
        assert_eq!(
            total_hits + total_misses,
            tokens * top_k as u64,
            "every routed expert must produce exactly one cache lookup"
        );

        // The first token always misses (cold cache); after that the
        // cache + prefetcher should eventually start serving experts
        // from RAM rather than disk.
        assert!(total_hits > 0, "expected at least some cache hits across {tokens} tokens");
        assert!(total_misses > 0, "expected at least some cache misses across {tokens} tokens");
        assert!(total_bytes > 0, "expected the engine to read bytes from the SSD");

        // The aggregate report mirrors the per-cycle totals on the
        // critical path. `r.bytes_read` may exceed `total_bytes` because
        // background prefetch tasks also contribute to the counter
        // without being part of any single token's stats.
        let r = engine.report();
        assert_eq!(r.hits, total_hits);
        assert_eq!(r.misses, total_misses);
        assert!(
            r.bytes_read >= total_bytes,
            "report bytes_read ({}) must include at least the critical-path bytes ({total_bytes})",
            r.bytes_read
        );
        assert!(r.io_count >= total_misses, "io histogram must record every miss");
        // Latency histograms must have observed at least one sample of each
        // category (compute always, I/O at least once because a cold start
        // forces a miss).
        assert!(r.cycle_p50_us > 0);
        assert!(r.compute_p50_us > 0);
        assert!(r.io_p50_us > 0);

        // Predictor learned something (transitions other than the very first
        // were observed).
        assert!(
            r.predictor_observations > 0,
            "predictor should have logged at least one transition"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn warm_with_preloads_experts_into_cache() {
        // Mirrors the spec's "router selects Expert ID 3 and 7" warm-up.
        let dir = TempDir::new("gen-warm");
        let num_experts: u32 = 8;
        let engine = build_engine(&dir.path, num_experts, 16, 32, 4, 2, 1, 0xBEEF);

        engine.warm_with(&[3, 7]).await.expect("warm fetch");

        // `warm_with` reads through `fetch`, which doesn't bump the
        // hit/miss/bytes counters (those track router-driven `generate`
        // traffic only). The observable side-effect is that both warmed
        // experts are now resident in the cache.
        let r = engine.report();
        assert_eq!(r.hits, 0);
        assert_eq!(r.misses, 0);
        assert!(engine.cache.contains(3));
        assert!(engine.cache.contains(7));

        // Subsequent generate calls now have warmed slots to hit.
        let _ = engine.generate(0).await;
        // After at least one token, the per-token cycle histogram must
        // have recorded a sample.
        let r = engine.report();
        assert!(r.cycle_p50_us > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cache_cap_bounds_residency_under_load() {
        // The engine must never let more than `cache_slots` experts be
        // resident at once, even under heavy churn. Pick num_experts >>
        // cache_slots to force eviction on most tokens.
        let dir = TempDir::new("gen-evict");
        let num_experts: u32 = 32;
        let cache_slots = 4;
        let engine = build_engine(&dir.path, num_experts, 16, 32, cache_slots, 2, 2, 7);

        for t in 0..50 {
            let _ = engine.generate(t).await;
            // Residency must NEVER exceed the configured cache capacity,
            // even mid-stream — this is the actual invariant the test
            // name promises. Asserting after every token catches a class
            // of regressions where the cache temporarily holds N+1
            // entries in between an insert and an eviction.
            assert!(
                engine.cache.resident_ids().len() <= cache_slots,
                "cache residency {} exceeded capacity {} at token {t}",
                engine.cache.resident_ids().len(),
                cache_slots
            );
            assert!(
                engine.cache.len() <= cache_slots,
                "cache.len() {} exceeded capacity {} at token {t}",
                engine.cache.len(),
                cache_slots
            );
        }
        let r = engine.report();
        assert_eq!(r.cache_capacity, cache_slots);
        assert!(
            engine.cache.resident_ids().len() <= cache_slots,
            "post-stream residency {} exceeded capacity {}",
            engine.cache.resident_ids().len(),
            cache_slots
        );
        // Misses dominate when cache_slots is small relative to working set.
        assert!(r.misses > r.hits / 2, "expected eviction churn to produce many misses");
    }
}
