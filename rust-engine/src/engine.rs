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
    combine_outputs, run_inference, run_inference_f16, run_inference_int8, run_inference_q4k, synth_hidden_state,
    uniform_scores, HiddenState,
    InferenceOutput, WeightDtype,
};
use crate::io_provider::NvmeStorage;
use crate::metrics::Metrics;
use crate::router::{LocalityMonitor, NeuralSpeculator, PredictiveLoader, TopKRouter};
use hdrhistogram::Histogram;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
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

/// Snapshot of the engine's predictive-architecture telemetry: the
/// running accuracy of the [`NeuralSpeculator`] (M arm), the running
/// hit rate of the [`LocalityMonitor`] (L arm), and the cumulative
/// SSD-stall time on the inference critical path. Returned by
/// [`Engine::predictive_telemetry`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PredictiveTelemetry {
    pub speculator_hits: u64,
    pub speculator_misses: u64,
    /// `speculator_hits / (speculator_hits + speculator_misses)`, or
    /// `0.0` when neither has fired.
    pub speculator_accuracy: f64,
    pub locality_hits: u64,
    pub locality_misses: u64,
    /// `locality_hits / (locality_hits + locality_misses)`, or `0.0`
    /// when neither has fired.
    pub locality_hit_rate: f64,
    /// Cumulative SSD critical-path stall, in microseconds.
    pub ssd_stall_us: u64,
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
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    /// When `true`, skip [`run_inference`] and instead XOR every byte of
    /// the resident buffer to force the read to fully materialise. This
    /// isolates the SSD-streaming cost from FFN compute and is what
    /// `--io-only` on the CLI maps to.
    pub io_only: bool,
    /// On-disk weight dtype. Selects which of the `run_inference*`
    /// variants is dispatched per cache hit.
    pub dtype: WeightDtype,
    /// Fraction of `d_model` columns to load when the partial-load path
    /// is enabled (`(0.1..1.0)`). `1.0` disables partial loading and
    /// the engine reads the full expert as before.
    pub partial_load_fraction: f64,
    /// After an expert has been observed this many times in routing
    /// targets, pin it permanently in the LRU cache. `0` disables
    /// frequency-based pinning entirely.
    pub pin_after_observations: u64,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            io_only: false,
            dtype: WeightDtype::F32,
            partial_load_fraction: 1.0,
            pin_after_observations: 0,
        }
    }
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
    /// Set of experts active two tokens ago — fed to the predictor's
    /// 2nd-order rows so prefetch decisions condition on the
    /// `(prev_prev, prev)` pair when one is available.
    last_last_experts: parking_lot::Mutex<Vec<u32>>,
    /// Per-expert routing-observation counts used by frequency-based
    /// pinning. Once an expert's count crosses
    /// `options.pin_after_observations`, the engine asks the cache to
    /// pin it.
    route_observations: RwLock<HashMap<u32, u64>>,
    /// Optional alias map: when present, any routed/predicted expert id
    /// is remapped to its canonical id before the cache is consulted.
    /// Used for **expert deduplication** — pairs of experts that the
    /// offline analyser flagged as numerically near-identical share a
    /// single resident copy. `None` means no aliasing.
    alias_map: Option<Arc<HashMap<u32, u32>>>,
    /// Number of times an alias redirect actually changed an expert id
    /// during routing/prefetch (for diagnostics).
    alias_redirects: AtomicU64,
    /// Locality monitor — sliding-window heat map over recently-routed
    /// experts. When configured, the engine reconciles its hot set
    /// against the expert cache after every token: ids in the hot set
    /// are pinned (cannot be LRU-evicted) and ids that just dropped
    /// out are unpinned. Forms the **L** arm of the speculative I/O
    /// union `E = S ∪ L ∪ M`.
    locality: Option<Arc<LocalityMonitor>>,
    /// Set of expert ids the locality monitor pinned on the previous
    /// reconciliation. Diff'd against the current hot set so we only
    /// `pin`/`unpin` ids that actually changed status.
    locality_pinned: parking_lot::Mutex<HashSet<u32>>,
    /// Heat threshold for [`Self::locality`]. Mirrors
    /// [`LocalityMonitor::DEFAULT_THRESHOLD_PCT`] when not overridden.
    locality_threshold_pct: f32,
    /// Neural speculator — a tiny 2-layer MLP that predicts the gate's
    /// top-K from the hidden state. Forms the **M** arm of the union
    /// `E = S ∪ L ∪ M` and is trained online against the actual gate
    /// decision. Wrapped in an `Arc` for cheap cloning into spawned
    /// prefetch tasks; internal weights are guarded by an `RwLock`
    /// owned by the speculator itself.
    speculator: Option<Arc<NeuralSpeculator>>,
    /// Number of speculator predictions pulled per token (top-K size
    /// for the M arm). Defaults to the router's `top_k`.
    speculator_topk: usize,
    /// Optional Prometheus metrics sink. When present, the locality
    /// hit / miss counters and speculator hit / miss counters are
    /// updated alongside the per-Engine atomics.
    metrics: Option<Metrics>,
    /// Cumulative microseconds spent on the SSD critical-path stall —
    /// the wall-clock window during which the engine was blocked
    /// waiting for cache-miss reads to land. Distinct from
    /// `total_io_wait_us` only in that it's exported as its own
    /// Prometheus histogram (`mer_ssd_stall_seconds`).
    total_ssd_stall_us: AtomicU64,
    /// Cumulative speculator hit count (predictions that intersected
    /// the gate's actual top-K).
    spec_hits: AtomicU64,
    /// Cumulative speculator miss count.
    spec_misses: AtomicU64,
    /// Cumulative locality-hit count (target experts that were already
    /// in the locality monitor's hot set at routing time).
    locality_hits: AtomicU64,
    /// Cumulative locality-miss count.
    locality_misses: AtomicU64,
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
        let speculator_topk_default = router.k();
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
            last_last_experts: parking_lot::Mutex::new(Vec::new()),
            route_observations: RwLock::new(HashMap::new()),
            alias_map: None,
            alias_redirects: AtomicU64::new(0),
            locality: None,
            locality_pinned: parking_lot::Mutex::new(HashSet::new()),
            locality_threshold_pct: LocalityMonitor::DEFAULT_THRESHOLD_PCT,
            speculator: None,
            speculator_topk: speculator_topk_default,
            metrics: None,
            total_ssd_stall_us: AtomicU64::new(0),
            spec_hits: AtomicU64::new(0),
            spec_misses: AtomicU64::new(0),
            locality_hits: AtomicU64::new(0),
            locality_misses: AtomicU64::new(0),
        }
    }

    /// Install an alias map. Calls to [`Self::generate`] / prefetch will
    /// remap ids through it before consulting the cache, so multiple
    /// near-identical experts share a single resident copy.
    pub fn with_alias_map(mut self, map: HashMap<u32, u32>) -> Self {
        // Keep only entries that actually move ids. Self-aliases are noise.
        let cleaned: HashMap<u32, u32> = map.into_iter().filter(|(k, v)| k != v).collect();
        self.alias_map = if cleaned.is_empty() {
            None
        } else {
            Some(Arc::new(cleaned))
        };
        self
    }

    /// Install a sliding-window [`LocalityMonitor`]. The engine will
    /// observe every routed expert and, after each `generate` /
    /// `moe_step`, reconcile the monitor's hot set with the cache's pin
    /// state — newly hot ids are pinned, ids that fell below the heat
    /// threshold are unpinned.
    pub fn with_locality_monitor(mut self, monitor: Arc<LocalityMonitor>, threshold_pct: f32) -> Self {
        self.locality = Some(monitor);
        // Clamp into a sane range; values outside `[0,1]` make no
        // semantic sense for a "fraction of the window" threshold.
        self.locality_threshold_pct = threshold_pct.clamp(0.0, 1.0);
        self
    }

    /// Install a [`NeuralSpeculator`]. When set, the engine will (a)
    /// query the speculator for its top-K prediction at every routed
    /// hidden state, (b) compare it against the actual gate decision
    /// to update speculator-accuracy telemetry, (c) feed that decision
    /// back into a single online SGD step, and (d) union the
    /// speculator's prediction with the predictor's Markov chain hint
    /// when issuing speculative prefetches.
    pub fn with_speculator(mut self, spec: Arc<NeuralSpeculator>, top_k: usize) -> Self {
        self.speculator = Some(spec);
        self.speculator_topk = top_k.max(1);
        self
    }

    /// Wire a Prometheus metrics sink. The engine will mirror its
    /// telemetry counters (locality / speculator hits & misses, SSD
    /// stall) into the metrics registry alongside its own atomics.
    pub fn with_metrics(mut self, m: Metrics) -> Self {
        self.metrics = Some(m);
        self
    }

    /// Resolve an id through the alias map (if any), bumping the
    /// redirect counter on a hit. Pure function on `&self`; safe to
    /// call from any context.
    fn resolve_alias(&self, id: u32) -> u32 {
        if let Some(m) = &self.alias_map {
            if let Some(&canon) = m.get(&id) {
                if canon != id {
                    self.alias_redirects.fetch_add(1, Ordering::Relaxed);
                    return canon;
                }
            }
        }
        id
    }

    pub fn shape(&self) -> ModelShape {
        self.shape
    }

    /// Process a single token: route, fetch missing experts, run inference,
    /// update predictor, and kick off prefetches. Returns one [`CycleStats`].
    pub async fn generate(self: &Arc<Self>, token_idx: u64) -> CycleStats {
        let cycle_start = Instant::now();
        let raw_target = self.router.route(token_idx);
        // Resolve aliases up front so the cache + predictor only ever
        // see canonical expert ids. This is what makes deduplicated
        // experts share one resident copy.
        let target: Vec<u32> = raw_target.iter().map(|&id| self.resolve_alias(id)).collect();
        let mut stats = CycleStats::default();

        // Locality monitor: observe the chosen experts and reconcile
        // pin state. When no monitor is configured this is a no-op
        // and we fall back to the legacy frequency-based pinning
        // below. (The two are intentionally orthogonal — frequency
        // pinning is monotonic and global; locality pinning is
        // sliding-window and topical.)
        self.locality_observe_and_reconcile(&target);

        // Frequency-based pinning: bump observation counts and ask the
        // cache to pin any id that crossed the threshold this token.
        if self.options.pin_after_observations > 0 {
            let mut obs = self.route_observations.write();
            let threshold = self.options.pin_after_observations;
            for &id in &target {
                let entry = obs.entry(id).or_insert(0);
                *entry += 1;
                if *entry == threshold {
                    debug!(expert = id, count = *entry, "pinning hot expert");
                    self.cache.pin(id);
                }
            }
        }

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
        // The *SSD stall* is the slice of the critical path we were
        // actually blocked on reads. With concurrent miss fetches it's
        // bounded by `io_wait_us`; we report them as the same value
        // here, since a mock-storage benchmark has no separate "in
        // flight, but not blocking" component. The Prometheus sink
        // exports it as its own histogram so future overlapped-fetch
        // refactors can decouple the two without breaking dashboards.
        if io_wait_us > 0 {
            self.total_ssd_stall_us.fetch_add(io_wait_us, Ordering::Relaxed);
            if let Some(m) = &self.metrics {
                m.record_ssd_stall(io_wait_us as f64 / 1_000_000.0);
            }
        }
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
                let res = match self.options.dtype {
                    WeightDtype::F32 => run_inference(token_idx, r, &x, self.shape.d_model, self.shape.d_ff),
                    WeightDtype::F16 => run_inference_f16(token_idx, r, &x, self.shape.d_model, self.shape.d_ff),
                    WeightDtype::Int8 => run_inference_int8(token_idx, r, &x, self.shape.d_model, self.shape.d_ff),
                    WeightDtype::Q4K => run_inference_q4k(token_idx, r, &x, self.shape.d_model, self.shape.d_ff),
                };
                match res {
                    Ok((out, y)) => {
                        outputs.push(out);
                        per_expert_y.push(y);
                    }
                    Err(e) => {
                        warn!(
                            token = token_idx,
                            expert = r.id,
                            error = %e,
                            "skipping expert: failed to reinterpret buffer as SwiGLU weights"
                        );
                    }
                }
            }
            // Synthetic / benchmark path has no real gating network, so
            // weight every routed expert uniformly (`1/k`) — that matches
            // the legacy averaging behaviour bit-for-bit while flowing
            // through the new softmax-gated combiner signature.
            let scores = uniform_scores(per_expert_y.len());
            let combined = combine_outputs(&per_expert_y, &scores);
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
        //    Use the 2nd-order helper when we have a `prev_prev` set
        //    (anything from token_idx >= 2), so the predictor learns
        //    `(prev_prev -> prev -> next)` triples in addition to the
        //    `(prev -> next)` baseline.
        {
            let mut last = self.last_experts.lock();
            let mut last_last = self.last_last_experts.lock();
            if !last.is_empty() {
                self.predictor.observe_step2(&last_last, &last, &target);
            }
            *last_last = last.clone();
            *last = target.clone();
        }

        // 4) Kick off speculative prefetches for the most-recent expert,
        //    using the 2nd-order predictor when a prev_prev is available
        //    (which gives sharper distributions than 1st-order alone and
        //    therefore wastes less prefetch bandwidth). When a neural
        //    speculator is configured, also union its top-K (the **M**
        //    arm) and the locality monitor's hot set (the **L** arm)
        //    into the prefetch set — see [`Engine::union_prefetch`].
        if let Some(&seed) = target.last() {
            let last_last = self.last_last_experts.lock();
            let s_markov = match last_last.last() {
                Some(&pp) => self.predictor.predict_next2(pp, seed),
                None => self.predictor.predict_next(seed),
            };
            drop(last_last);
            // Speculator: predict + train on the synthetic hidden state
            // (when the speculator's d_model matches; otherwise this is
            // a no-op — see `speculator_predict_and_train`).
            let x_for_spec: HiddenState =
                synth_hidden_state(token_idx, self.shape.d_model, self.shape.hidden_seed);
            let m_speculator = self.speculator_predict_and_train(&x_for_spec, &target);
            self.union_prefetch(&s_markov, &m_speculator, &HashSet::new());
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

    // -----------------------------------------------------------------
    // Locality / speculator integration helpers.
    //
    // These are called from `generate` and `moe_step` after the gating
    // decision (`target`) is known. They are no-ops when neither
    // monitor is configured, which preserves the legacy code path
    // bit-for-bit.
    // -----------------------------------------------------------------

    /// Observe the chosen expert ids in the locality monitor and
    /// reconcile pinning with the expert cache: ids that just entered
    /// the hot set are pinned (LRU-eviction-protected), ids that just
    /// dropped out are unpinned.
    ///
    /// Also records per-token locality hit/miss telemetry: a chosen
    /// expert is a "locality hit" if it was *already* in the hot set
    /// at the time of routing (i.e. before this token's observation
    /// pushed it in or out). Returns the size of the hot set, useful
    /// for tests.
    fn locality_observe_and_reconcile(&self, target: &[u32]) -> usize {
        let Some(monitor) = self.locality.as_ref() else {
            return 0;
        };
        // Snapshot pre-observation hit/miss against the *current* hot set.
        let mut hits: u64 = 0;
        let mut misses: u64 = 0;
        for &id in target {
            if monitor.is_hot(id, self.locality_threshold_pct) {
                hits += 1;
            } else {
                misses += 1;
            }
        }
        if hits > 0 {
            self.locality_hits.fetch_add(hits, Ordering::Relaxed);
        }
        if misses > 0 {
            self.locality_misses.fetch_add(misses, Ordering::Relaxed);
        }
        if let Some(m) = &self.metrics {
            m.record_locality(hits, misses);
        }

        // Update the monitor's window with this token's activations.
        monitor.observe(target);

        // Reconcile pin set against the post-observation hot set.
        let new_hot: HashSet<u32> = monitor
            .hot_set(self.locality_threshold_pct)
            .into_iter()
            .collect();
        let mut prev = self.locality_pinned.lock();
        for &id in new_hot.iter() {
            if !prev.contains(&id) {
                self.cache.pin(id);
            }
        }
        for &id in prev.iter() {
            if !new_hot.contains(&id) {
                self.cache.unpin(id);
            }
        }
        let len = new_hot.len();
        *prev = new_hot;
        len
    }

    /// Run the speculator forward over `x`, compare its top-K to the
    /// gate's actual `target`, record accuracy telemetry, and take one
    /// online SGD step against the actual decision. Returns the
    /// speculator's prediction so the caller can union it into the
    /// prefetch set.
    fn speculator_predict_and_train(&self, x: &[f32], target: &[u32]) -> Vec<u32> {
        let Some(spec) = self.speculator.as_ref() else {
            return Vec::new();
        };
        if x.len() != spec.d_model() {
            // Hidden state shape mismatch — nothing useful we can
            // predict against, so silently disable for this token.
            // This makes the speculator graceful in the synthetic
            // benchmark where d_model can disagree with the real model.
            return Vec::new();
        }
        let preds = spec.predict_topk(x, self.speculator_topk);
        let target_set: HashSet<u32> = target.iter().copied().collect();
        let mut hits: u64 = 0;
        for &p in &preds {
            if target_set.contains(&p) {
                hits += 1;
            }
        }
        let misses = preds.len() as u64 - hits;
        if hits > 0 {
            self.spec_hits.fetch_add(hits, Ordering::Relaxed);
        }
        if misses > 0 {
            self.spec_misses.fetch_add(misses, Ordering::Relaxed);
        }
        if let Some(m) = &self.metrics {
            m.record_speculator(hits, misses);
        }
        // Online SGD step against the *actual* gate decision.
        let _loss = spec.train_step(x, target, NeuralSpeculator::DEFAULT_LR);
        preds
    }

    /// Prefetch every id in the union `S ∪ L ∪ M` that isn't already
    /// resident — the **speculative I/O union-fetch** described in the
    /// design spec. `s_markov` is the predictor's Markov-chain top-K
    /// (already prob-ranked), `m_speculator` is the neural speculator's
    /// top-K, `target_seed` is used to dedupe against ids that the
    /// caller already kicked off via the regular cache-miss path.
    fn union_prefetch(
        self: &Arc<Self>,
        s_markov: &[(u32, f64)],
        m_speculator: &[u32],
        already_in_flight: &HashSet<u32>,
    ) {
        // Preserve the predictor's per-id probability when it has one;
        // ids that come only from the locality / speculator arms
        // borrow the speculator's "best guess" probability of 0.5
        // (high enough to clear most prefetch budget thresholds, low
        // enough to be visibly different in the prefetch logs from
        // a real Markov-chain prediction).
        let mut seen: HashSet<u32> = already_in_flight.clone();
        for &(id, p) in s_markov {
            let canon = self.resolve_alias(id);
            if seen.insert(canon) && !self.cache.contains(canon) {
                self.spawn_prefetch(canon, p);
            }
        }
        if let Some(monitor) = self.locality.as_ref() {
            for id in monitor.hot_set(self.locality_threshold_pct) {
                let canon = self.resolve_alias(id);
                if seen.insert(canon) && !self.cache.contains(canon) {
                    self.spawn_prefetch(canon, 0.5);
                }
            }
        }
        for &id in m_speculator {
            let canon = self.resolve_alias(id);
            if seen.insert(canon) && !self.cache.contains(canon) {
                self.spawn_prefetch(canon, 0.5);
            }
        }
    }

    /// Snapshot of the engine's predictive-architecture telemetry. The
    /// returned ratios are in `[0, 1]`; both fall back to `0.0` when no
    /// observations have been recorded yet (the safer default for a
    /// freshly-warmed engine).
    pub fn predictive_telemetry(&self) -> PredictiveTelemetry {
        let s_hits = self.spec_hits.load(Ordering::Relaxed);
        let s_misses = self.spec_misses.load(Ordering::Relaxed);
        let l_hits = self.locality_hits.load(Ordering::Relaxed);
        let l_misses = self.locality_misses.load(Ordering::Relaxed);
        let s_total = s_hits + s_misses;
        let l_total = l_hits + l_misses;
        PredictiveTelemetry {
            speculator_hits: s_hits,
            speculator_misses: s_misses,
            speculator_accuracy: if s_total == 0 {
                0.0
            } else {
                s_hits as f64 / s_total as f64
            },
            locality_hits: l_hits,
            locality_misses: l_misses,
            locality_hit_rate: if l_total == 0 {
                0.0
            } else {
                l_hits as f64 / l_total as f64
            },
            ssd_stall_us: self.total_ssd_stall_us.load(Ordering::Relaxed),
        }
    }

    /// **Real-transformer MoE step.** Given a hidden state `x` and the
    /// expert ids the gating network selected for it, ensure every chosen
    /// expert is resident in the SSD-streaming cache (concurrent
    /// `pread(2)` for the misses, exactly as `generate` does), run each
    /// expert's SwiGLU FFN over `x`, and return the per-expert output
    /// vectors aligned with the input `experts` slice.
    ///
    /// This is the bridge from the dense `TransformerLayer` code (which
    /// produces a routing decision) to the MoE compute (which the
    /// SSD-streaming substrate makes interesting). The caller — typically
    /// `crate::model::RealModel::step` — then folds the returned vectors
    /// back into the residual stream via `TransformerLayer::moe_combine`.
    ///
    /// The same hits / misses / bytes / latency counters that
    /// `Engine::generate` updates are bumped here too, so
    /// `engine.print_summary()` shows the same shape regardless of
    /// whether the engine is driving the benchmark Markov path or a real
    /// transformer.
    ///
    /// `token_idx` is used only as a digest seed for `InferenceOutput`;
    /// it has no effect on the activation produced.
    pub async fn moe_step(
        self: &Arc<Self>,
        token_idx: u64,
        x: &HiddenState,
        experts: &[u32],
    ) -> Vec<HiddenState> {
        let cycle_start = Instant::now();
        // Resolve aliases up front so the cache + predictor only ever
        // see canonical expert ids (mirrors `generate`).
        let target: Vec<u32> = experts.iter().map(|&id| self.resolve_alias(id)).collect();

        // Locality monitor: observe and reconcile pinning. Same
        // semantics as in `generate`.
        self.locality_observe_and_reconcile(&target);

        // Speculator: predict against the *real* hidden state (this is
        // the path where d_model matches by construction) and train
        // online against the gate's actual top-K decision.
        let m_speculator = self.speculator_predict_and_train(x, &target);

        // Frequency-based pinning: same logic as `generate`.
        if self.options.pin_after_observations > 0 {
            let mut obs = self.route_observations.write();
            let threshold = self.options.pin_after_observations;
            for &id in &target {
                let entry = obs.entry(id).or_insert(0);
                *entry += 1;
                if *entry == threshold {
                    debug!(expert = id, count = *entry, "pinning hot expert");
                    self.cache.pin(id);
                }
            }
        }

        // Concurrent miss fetches; hits resolved inline.
        let io_wait_start = Instant::now();
        let mut residents: Vec<Option<Arc<ExpertResident>>> = vec![None; target.len()];
        let mut miss_handles: Vec<(usize, tokio::task::JoinHandle<Arc<ExpertResident>>)> =
            Vec::new();
        for (i, &id) in target.iter().enumerate() {
            if let Some(r) = self.cache.get(id) {
                self.counters.hits.fetch_add(1, Ordering::Relaxed);
                residents[i] = Some(r);
            } else {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                let me = self.clone();
                miss_handles.push((i, tokio::spawn(async move { me.fetch(id).await })));
            }
        }
        let had_misses = !miss_handles.is_empty();
        for (i, h) in miss_handles {
            let r = h.await.expect("expert fetch task panicked");
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
            .map(|r| r.expect("internal invariant: every routed expert slot must be populated"))
            .collect();

        // Run the SwiGLU FFN per expert against the hidden state.
        let compute_start = Instant::now();
        let mut per_expert_y: Vec<HiddenState> = Vec::with_capacity(residents.len());
        for r in &residents {
            let res = match self.options.dtype {
                WeightDtype::F32 => run_inference(token_idx, r, x, self.shape.d_model, self.shape.d_ff),
                WeightDtype::F16 => run_inference_f16(token_idx, r, x, self.shape.d_model, self.shape.d_ff),
                WeightDtype::Int8 => run_inference_int8(token_idx, r, x, self.shape.d_model, self.shape.d_ff),
                WeightDtype::Q4K => run_inference_q4k(token_idx, r, x, self.shape.d_model, self.shape.d_ff),
            };
            match res {
                Ok((_out, y)) => per_expert_y.push(y),
                Err(e) => {
                    warn!(
                        token = token_idx,
                        expert = r.id,
                        error = %e,
                        "skipping expert: failed to reinterpret buffer as SwiGLU weights"
                    );
                    // Push a zero vector so the caller's weights[] alignment
                    // stays valid; combining with weight `w_i * 0 = 0` is
                    // the same as if this expert were never picked.
                    per_expert_y.push(vec![0.0f32; self.shape.d_model]);
                }
            }
        }
        let compute_us = compute_start.elapsed().as_micros() as u64;
        let _ = self.compute_hist.lock().record(compute_us.max(1));
        self.total_compute_us.fetch_add(compute_us, Ordering::Relaxed);
        self.total_io_wait_us.fetch_add(io_wait_us, Ordering::Relaxed);
        if io_wait_us > 0 {
            self.total_ssd_stall_us.fetch_add(io_wait_us, Ordering::Relaxed);
            if let Some(m) = &self.metrics {
                m.record_ssd_stall(io_wait_us as f64 / 1_000_000.0);
            }
        }

        // Speculative I/O union-fetch (S ∪ L ∪ M). Fire the predictor's
        // 2nd-order Markov-chain hint and union it with the locality
        // hot set and the speculator's top-K so all three arms compete
        // for cache slots together. We fold this into the predictor's
        // observation history first so the next token's S has a chance
        // to learn from this token's transition.
        if let Some(&seed) = target.last() {
            // Update predictor history (mirrors `generate`).
            {
                let mut last = self.last_experts.lock();
                let mut last_last = self.last_last_experts.lock();
                if !last.is_empty() {
                    self.predictor.observe_step2(&last_last, &last, &target);
                }
                *last_last = last.clone();
                *last = target.clone();
            }
            let last_last = self.last_last_experts.lock();
            let s_markov = match last_last.last() {
                Some(&pp) => self.predictor.predict_next2(pp, seed),
                None => self.predictor.predict_next(seed),
            };
            drop(last_last);
            self.union_prefetch(&s_markov, &m_speculator, &HashSet::new());
        }

        let cycle_us = cycle_start.elapsed().as_micros() as u64;
        let _ = self.cycle_hist.lock().record(cycle_us.max(1));
        self.total_cycle_us.fetch_add(cycle_us, Ordering::Relaxed);
        self.tokens_processed.fetch_add(1, Ordering::Relaxed);

        per_expert_y
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
            pinned_count: self.cache.pinned_count(),
            alias_redirects: self.alias_redirects.load(Ordering::Relaxed),
            dtype: self.options.dtype,
            partial_load_fraction: self.options.partial_load_fraction,
            predictive: self.predictive_telemetry(),
            locality_enabled: self.locality.is_some(),
            speculator_enabled: self.speculator.is_some(),
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
            "ffn shape:     d_model={}  d_ff={}  bytes/expert={} (dtype={})",
            r.d_model,
            r.d_ff,
            crate::inference::expert_weight_bytes_for(r.d_model, r.d_ff, r.dtype),
            r.dtype.as_str()
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
        info!(
            "energy knobs:  dtype={}  partial_load_fraction={:.2}  pinned={}  alias_redirects={}",
            r.dtype.as_str(),
            r.partial_load_fraction,
            r.pinned_count,
            r.alias_redirects
        );
        // Only emit the predictive line when either L or M is wired in;
        // the legacy benchmark path (everything off) keeps its existing
        // summary shape so older diff-on-output tests stay valid.
        if r.locality_enabled || r.speculator_enabled {
            info!(
                "predictive:    locality={} (hit_rate={:.2}%)  speculator={} (accuracy={:.2}%)  ssd_stall={:.1}ms",
                if r.locality_enabled { "on" } else { "off" },
                r.predictive.locality_hit_rate * 100.0,
                if r.speculator_enabled { "on" } else { "off" },
                r.predictive.speculator_accuracy * 100.0,
                r.predictive.ssd_stall_us as f64 / 1000.0,
            );
        }
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
    /// Number of experts currently pinned in the LRU cache (Change 5:
    /// frequency-based pinning).
    pub pinned_count: usize,
    /// Number of times an alias map redirected an expert id to a
    /// canonical id (Change 6: expert deduplication). Each redirect is
    /// one cache lookup that targeted a deduplicated copy.
    pub alias_redirects: u64,
    /// On-disk weight dtype used by this engine instance (Change 1).
    pub dtype: WeightDtype,
    /// Partial-load fraction used by this engine instance (Change 3).
    pub partial_load_fraction: f64,
    /// Snapshot of the predictive-architecture telemetry: locality
    /// hit-rate, speculator accuracy, and cumulative SSD critical-path
    /// stall. Populated regardless of whether the L/M arms are wired
    /// in (the counters stay at zero when disabled, which still
    /// produces the correct `0.0` ratios).
    pub predictive: PredictiveTelemetry,
    /// Whether the [`LocalityMonitor`] (the **L** arm of the
    /// predictive `S ∪ L ∪ M` union-fetch) was configured on this run.
    pub locality_enabled: bool,
    /// Whether the [`NeuralSpeculator`] (the **M** arm of the
    /// predictive `S ∪ L ∪ M` union-fetch) was configured on this run.
    pub speculator_enabled: bool,
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

    // ----------- Locality / Speculator / Union-Fetch tests ----------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_with_locality_monitor_pins_hot_experts() {
        // Build an engine with a tight hot threshold so any expert
        // routed twice in the recent window enters the hot set, and
        // verify that those experts get pinned in the cache.
        let dir = TempDir::new("locality-pin");
        let num_experts: u32 = 8;
        let top_k = 2;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 6;
        let predict_fanout = 1;

        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0x10CA117F,
        );
        // Re-wrap with a locality monitor. We drop the previous Arc
        // and rebuild via the same helpers; the cleanest way is to
        // unwrap and rebuild — the helper returns `Arc<Engine>` so
        // we mutate via a fresh constructor instead.
        let engine = {
            // SAFETY: tests own the only Arc reference at this point.
            let cache = engine.cache.clone();
            let pool = engine.pool.clone();
            let storage = engine.storage.clone();
            let router = engine.router.clone();
            let predictor = engine.predictor.clone();
            let shape = engine.shape;
            let monitor = Arc::new(LocalityMonitor::new(num_experts, /*window=*/ 16));
            // Threshold of 0.05 ⇒ any id observed at least once in
            // the 16-slot window is "hot" — easy to trip.
            Arc::new(
                Engine::new(cache, pool, storage, router, predictor, shape)
                    .with_locality_monitor(monitor.clone(), 0.05),
            )
        };
        // Drive a few tokens; the synthetic router routes deterministically,
        // so after several tokens the locality monitor will see repeated
        // ids and start pinning them.
        for t in 0..32u64 {
            let _ = engine.generate(t).await;
        }
        let pinned = engine.cache.pinned_count();
        assert!(
            pinned > 0,
            "locality monitor should have pinned at least one hot expert; got {pinned}"
        );
        // Telemetry must show non-zero locality observations.
        let tele = engine.predictive_telemetry();
        assert!(
            tele.locality_hits + tele.locality_misses > 0,
            "expected locality counters to fire; got {:?}",
            tele
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_with_speculator_records_accuracy_telemetry() {
        let dir = TempDir::new("spec-accuracy");
        let num_experts: u32 = 8;
        let top_k = 2;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 6;
        let predict_fanout = 1;

        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0x5EEEEDED,
        );
        let engine = {
            let cache = engine.cache.clone();
            let pool = engine.pool.clone();
            let storage = engine.storage.clone();
            let router = engine.router.clone();
            let predictor = engine.predictor.clone();
            let shape = engine.shape;
            let spec = Arc::new(NeuralSpeculator::new(d_model, 32, num_experts, 0xABCD));
            Arc::new(
                Engine::new(cache, pool, storage, router, predictor, shape)
                    .with_speculator(spec, top_k),
            )
        };
        for t in 0..50u64 {
            let _ = engine.generate(t).await;
        }
        let tele = engine.predictive_telemetry();
        assert!(
            tele.speculator_hits + tele.speculator_misses > 0,
            "speculator counters should be non-zero after 50 tokens; got {:?}",
            tele
        );
        assert!(tele.speculator_accuracy >= 0.0 && tele.speculator_accuracy <= 1.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_predictive_telemetry_reports_ssd_stall() {
        let dir = TempDir::new("ssd-stall");
        let num_experts: u32 = 8;
        let top_k = 2;
        let d_model = 16;
        let d_ff = 32;
        // Tiny cache so we must take SSD misses.
        let cache_slots = 2;
        let predict_fanout = 1;
        let engine = build_engine(
            &dir.path, num_experts, d_model, d_ff, cache_slots, top_k, predict_fanout, 0xDEADBEEF,
        );
        for t in 0..16u64 {
            let _ = engine.generate(t).await;
        }
        let tele = engine.predictive_telemetry();
        // With a 2-slot cache and 8 experts at top-k=2, we expect to
        // pay for at least *some* SSD stall.
        assert!(
            tele.ssd_stall_us > 0,
            "expected non-zero ssd stall; got {tele:?}"
        );
    }
}
