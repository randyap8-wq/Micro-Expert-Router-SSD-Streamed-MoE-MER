//! Prometheus metrics export (gist Phase 9).
//!
//! Exposes:
//! * `mer_requests_total{endpoint}` — completed HTTP requests by endpoint
//! * `mer_request_latency_seconds{endpoint}` — request latency histogram
//! * `mer_tokens_generated_total` — total tokens generated server-wide
//! * `mer_cache_hits_total`, `mer_cache_misses_total` — expert cache stats
//! * `mer_io_wait_seconds` — histogram of per-token critical-path I/O wait
//! * `mer_nonfinite_softmax_fallbacks` — attention-softmax non-finite fallbacks
//!   (gauge, mirrored from a process-wide atomic)
//!
//! All counters are mirrored from the engine's existing atomic counters
//! (so the Prometheus snapshot is always consistent with `engine.report()`).

use prometheus::{
    register_counter_vec_with_registry, register_counter_with_registry,
    register_histogram_vec_with_registry, register_histogram_with_registry,
    register_int_gauge_with_registry, Counter, CounterVec, Encoder, Histogram, HistogramVec,
    IntGauge, Registry, TextEncoder,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Owns a `prometheus::Registry` and the metric handles. Cheap to clone
/// (`Arc` inside).
#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    registry: Registry,
    pub requests_total: CounterVec,
    pub request_latency_seconds: HistogramVec,
    pub tokens_generated_total: Counter,
    pub cache_hits_total: Counter,
    pub cache_misses_total: Counter,
    pub io_wait_seconds: Histogram,
    /// Cumulative wall time by real-transformer request stage.
    pub stage_seconds_total: CounterVec,
    /// Number of locally-aggregated timing events by request stage.
    pub stage_events_total: CounterVec,
    /// Per-request total wall time observed for each request stage.
    pub stage_request_seconds: HistogramVec,
    /// Predicted expert IDs contained in the gate's actual top-K
    /// (prediction precision@K numerator).
    pub speculator_hits_total: Counter,
    /// Predicted expert IDs not contained in the gate's actual top-K
    /// (prediction precision@K denominator component).
    pub speculator_misses_total: Counter,
    /// Tokens for which the speculator's **top-1** prediction matched
    /// the actual top-1 routed expert. The "Omniscient Predictive
    /// Architecture" design spec calls this counter
    /// `mer_speculator_accuracy_total` and uses it as the primary
    /// quality signal for the predictive controller.
    pub speculator_accuracy_total: Counter,
    /// Valid top-1 speculator evaluations, whether or not the
    /// prediction matched. Denominator for top-1 accuracy.
    pub speculator_evaluations_total: Counter,
    /// Activations whose chosen expert was already in the locality
    /// monitor's hot set at the time of routing.
    pub locality_hits_total: Counter,
    /// Activations whose chosen expert was *not* in the hot set.
    pub locality_misses_total: Counter,
    /// Per-token cumulative SSD stall time (the slice of the critical
    /// path that was actually waiting on the storage device, as
    /// distinct from the total I/O time which can overlap compute).
    pub ssd_stall_seconds: Histogram,
    /// VRAM tier probe hits — lookups for which the requested expert
    /// was already present in
    /// [`GpuExpertCache`](crate::expert_cache::GpuExpertCache)
    /// at probe time (Phase 1).
    pub gpu_cache_hits_total: Counter,
    /// VRAM tier probe misses — lookups for which the requested expert
    /// was not present in VRAM at probe time and therefore required
    /// lower-tier resolution/promotion logic (Phase 1).
    pub gpu_cache_misses_total: Counter,
    /// **Gauge** of currently-resident VRAM bytes across the Anchor +
    /// LRU regions. Phase 1.
    pub vram_used_bytes: IntGauge,
    /// **Gauge** of the total VRAM byte budget for the
    /// [`GpuExpertCache`](crate::expert_cache::GpuExpertCache)
    /// (Anchor + LRU capacity). Set once when the GPU cache is
    /// installed so dashboards can compute utilisation as
    /// `mer_vram_used_bytes / mer_vram_capacity_bytes` without
    /// relying on the `/v1/admin/health/experts` admin endpoint.
    /// Stays at `0` when the GPU cache is disabled. Phase 1.
    pub vram_capacity_bytes: IntGauge,
    /// Total RAM → VRAM promotions performed since startup. Each
    /// promotion is the result of an `ExpertResident` crossing
    /// `gpu_cache.promote_after_hits` and being copied into the
    /// Anchor Core (or the LRU Edge as a fallback). Phase 1.
    pub promotions_total: Counter,
    /// Speculative prefetches dropped because no pool buffer could be
    /// acquired (shadow half starved even after recycling, or legacy
    /// primary pool busy). Makes shadow-pool starvation visible in
    /// `/metrics` instead of a `debug!` only.
    pub prefetch_dropped_pool_starved_total: Counter,
    /// Tokens for which the neural speculator (M arm) was disabled by
    /// a hidden-state / `d_model` mismatch.
    pub speculator_disabled_total: Counter,
    /// Expert activations that fell back from the GPU fast path to the
    /// CPU path because the VRAM dispatch errored.
    pub gpu_cpu_fallbacks_total: Counter,
    /// **Gauge** mirroring the process-wide cumulative count of
    /// attention-softmax non-finite fallbacks
    /// ([`crate::transformer::nonfinite_softmax_fallbacks`]). Refreshed from
    /// the global atomic at scrape time. Distinct from router filtering of
    /// non-finite gate scores. A nonzero (or increasing) value indicates the
    /// attention path substituted a uniform distribution for a
    /// `NaN`/`±inf`/fully-masked row.
    pub nonfinite_softmax_fallbacks: IntGauge,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        // Build a private registry so multiple `Metrics` instances (in
        // tests, especially) don't collide on the global registry.
        let registry = Registry::new();
        let requests_total = register_counter_vec_with_registry!(
            "mer_requests_total",
            "Total completed HTTP requests by endpoint.",
            &["endpoint"],
            registry
        )
        .expect("metric registration: mer_requests_total");
        let request_latency_seconds = register_histogram_vec_with_registry!(
            "mer_request_latency_seconds",
            "HTTP request latency in seconds, by endpoint.",
            &["endpoint"],
            // Buckets aimed at LLM-server token latencies: 1 ms .. 30 s.
            vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry
        )
        .expect("metric registration: mer_request_latency_seconds");
        let tokens_generated_total = register_counter_with_registry!(
            "mer_tokens_generated_total",
            "Total tokens generated by the server.",
            registry
        )
        .expect("metric registration: mer_tokens_generated_total");
        let cache_hits_total = register_counter_with_registry!(
            "mer_cache_hits_total",
            "Total expert cache hits across all routed activations.",
            registry
        )
        .expect("metric registration: mer_cache_hits_total");
        let cache_misses_total = register_counter_with_registry!(
            "mer_cache_misses_total",
            "Total expert cache misses (= NVMe reads issued).",
            registry
        )
        .expect("metric registration: mer_cache_misses_total");
        let io_wait_seconds = register_histogram_with_registry!(
            "mer_io_wait_seconds",
            "Per-token critical-path SSD I/O wait time in seconds.",
            // 100us .. 1s, log-spaced.
            vec![
                0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                1.0
            ],
            registry
        )
        .expect("metric registration: mer_io_wait_seconds");
        let stage_seconds_total = register_counter_vec_with_registry!(
            "mer_stage_seconds_total",
            "Cumulative wall time spent in real-transformer request stages.",
            &["stage"],
            registry
        )
        .expect("metric registration: mer_stage_seconds_total");
        let stage_events_total = register_counter_vec_with_registry!(
            "mer_stage_events_total",
            "Cumulative locally-aggregated timing event count by real-transformer request stage.",
            &["stage"],
            registry
        )
        .expect("metric registration: mer_stage_events_total");
        let stage_request_seconds = register_histogram_vec_with_registry!(
            "mer_stage_request_seconds",
            "Per-request total wall time observed for a real-transformer request stage.",
            &["stage"],
            vec![
                0.000001, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001,
                0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
            ],
            registry
        )
        .expect("metric registration: mer_stage_request_seconds");
        let speculator_hits_total = register_counter_with_registry!(
            "mer_speculator_hits_total",
            "Neural speculator predicted expert IDs contained in the gate's actual top-K; numerator of prediction precision@K.",
            registry
        )
        .expect("metric registration: mer_speculator_hits_total");
        let speculator_misses_total = register_counter_with_registry!(
            "mer_speculator_misses_total",
            "Neural speculator predicted expert IDs not contained in the gate's actual top-K; with hits, forms the prediction precision@K denominator.",
            registry
        )
        .expect("metric registration: mer_speculator_misses_total");
        let speculator_accuracy_total = register_counter_with_registry!(
            "mer_speculator_accuracy_total",
            "Tokens for which the neural speculator's top-1 prediction matched the gate's actual top-1 routed expert.",
            registry
        )
        .expect("metric registration: mer_speculator_accuracy_total");
        let speculator_evaluations_total = register_counter_with_registry!(
            "mer_speculator_evaluations_total",
            "Valid neural speculator top-1 evaluations, whether or not the prediction matched.",
            registry
        )
        .expect("metric registration: mer_speculator_evaluations_total");
        let locality_hits_total = register_counter_with_registry!(
            "mer_locality_hits_total",
            "Routed activations whose chosen expert was in the locality monitor's hot set.",
            registry
        )
        .expect("metric registration: mer_locality_hits_total");
        let locality_misses_total = register_counter_with_registry!(
            "mer_locality_misses_total",
            "Routed activations whose chosen expert was NOT in the locality monitor's hot set.",
            registry
        )
        .expect("metric registration: mer_locality_misses_total");
        let ssd_stall_seconds = register_histogram_with_registry!(
            "mer_ssd_stall_seconds",
            "Per-token cumulative SSD stall time on the inference critical path.",
            // 10us .. 1s log-spaced; SSD stall is typically much
            // smaller than the wall-clock io wait when prefetch lands.
            vec![
                0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
                0.1, 0.25, 0.5, 1.0
            ],
            registry
        )
        .expect("metric registration: mer_ssd_stall_seconds");
        let gpu_cache_hits_total = register_counter_with_registry!(
            "mer_gpu_cache_hits_total",
            "VRAM-tier expert cache hits (lookups served out of GpuExpertCache).",
            registry
        )
        .expect("metric registration: mer_gpu_cache_hits_total");
        let gpu_cache_misses_total = register_counter_with_registry!(
            "mer_gpu_cache_misses_total",
            "VRAM-tier expert cache misses (lookups that fell through to RAM/NVMe).",
            registry
        )
        .expect("metric registration: mer_gpu_cache_misses_total");
        let vram_used_bytes = register_int_gauge_with_registry!(
            "mer_vram_used_bytes",
            "Currently-resident VRAM bytes across the Anchor Core + LRU Edge regions.",
            registry
        )
        .expect("metric registration: mer_vram_used_bytes");
        let vram_capacity_bytes = register_int_gauge_with_registry!(
            "mer_vram_capacity_bytes",
            "Total VRAM byte budget configured for the GpuExpertCache (Anchor + LRU). 0 when the GPU cache is disabled.",
            registry
        )
        .expect("metric registration: mer_vram_capacity_bytes");
        let promotions_total = register_counter_with_registry!(
            "mer_promotions_total",
            "Total RAM-to-VRAM promotions performed since startup.",
            registry
        )
        .expect("metric registration: mer_promotions_total");
        let prefetch_dropped_pool_starved_total = register_counter_with_registry!(
            "mer_prefetch_dropped_pool_starved_total",
            "Speculative prefetches dropped because no pool buffer (shadow or legacy primary) could be acquired.",
            registry
        )
        .expect("metric registration: mer_prefetch_dropped_pool_starved_total");
        let speculator_disabled_total = register_counter_with_registry!(
            "mer_speculator_disabled_total",
            "Tokens for which the neural speculator was disabled by a hidden-state/d_model mismatch.",
            registry
        )
        .expect("metric registration: mer_speculator_disabled_total");
        let gpu_cpu_fallbacks_total = register_counter_with_registry!(
            "mer_gpu_cpu_fallbacks_total",
            "Expert activations that fell back from the GPU fast path to the CPU path.",
            registry
        )
        .expect("metric registration: mer_gpu_cpu_fallbacks_total");
        let nonfinite_softmax_fallbacks = register_int_gauge_with_registry!(
            "mer_nonfinite_softmax_fallbacks",
            "Cumulative attention-softmax non-finite (NaN/inf/fully-masked) fallbacks to a uniform distribution. Distinct from router non-finite score filtering.",
            registry
        )
        .expect("metric registration: mer_nonfinite_softmax_fallbacks");
        Self {
            inner: Arc::new(MetricsInner {
                registry,
                requests_total,
                request_latency_seconds,
                tokens_generated_total,
                cache_hits_total,
                cache_misses_total,
                io_wait_seconds,
                stage_seconds_total,
                stage_events_total,
                stage_request_seconds,
                speculator_hits_total,
                speculator_misses_total,
                speculator_accuracy_total,
                speculator_evaluations_total,
                locality_hits_total,
                locality_misses_total,
                ssd_stall_seconds,
                gpu_cache_hits_total,
                gpu_cache_misses_total,
                vram_used_bytes,
                vram_capacity_bytes,
                promotions_total,
                prefetch_dropped_pool_starved_total,
                speculator_disabled_total,
                gpu_cpu_fallbacks_total,
                nonfinite_softmax_fallbacks,
            }),
        }
    }

    pub fn record_request(&self, endpoint: &str, latency_seconds: f64) {
        self.inner
            .requests_total
            .with_label_values(&[endpoint])
            .inc();
        self.inner
            .request_latency_seconds
            .with_label_values(&[endpoint])
            .observe(latency_seconds);
    }

    pub fn record_tokens(&self, n: u64) {
        self.inner.tokens_generated_total.inc_by(n as f64);
    }

    pub fn record_cache(&self, hits: u64, misses: u64) {
        if hits > 0 {
            self.inner.cache_hits_total.inc_by(hits as f64);
        }
        if misses > 0 {
            self.inner.cache_misses_total.inc_by(misses as f64);
        }
    }

    pub fn record_io_wait(&self, seconds: f64) {
        self.inner.io_wait_seconds.observe(seconds);
    }

    /// Publish one request-local stage-timing snapshot. The hot path
    /// aggregates locally; this method emits one counter update and one
    /// histogram observation per stage at request completion/drop.
    pub fn record_stage_timings(
        &self,
        snapshot: &BTreeMap<String, crate::stage_timing::StageTimingSnapshot>,
    ) {
        for (stage, timing) in snapshot {
            if timing.total_seconds > 0.0 {
                self.inner
                    .stage_seconds_total
                    .with_label_values(&[stage.as_str()])
                    .inc_by(timing.total_seconds);
                self.inner
                    .stage_request_seconds
                    .with_label_values(&[stage.as_str()])
                    .observe(timing.total_seconds);
            }
            if timing.count > 0 {
                self.inner
                    .stage_events_total
                    .with_label_values(&[stage.as_str()])
                    .inc_by(timing.count as f64);
            }
        }
    }

    /// Record speculator prediction precision@K components: `hits`
    /// predicted expert IDs contained in the gate's actual top-K and
    /// `misses` predicted expert IDs not contained in that top-K.
    /// Either may be zero; `hits / (hits + misses)` is precision@K.
    pub fn record_speculator(&self, hits: u64, misses: u64) {
        if hits > 0 {
            self.inner.speculator_hits_total.inc_by(hits as f64);
        }
        if misses > 0 {
            self.inner.speculator_misses_total.inc_by(misses as f64);
        }
    }

    /// Record one token's worth of **top-1** speculator accuracy.
    /// Pass `1` if the speculator's highest-logit expert matched the
    /// gate's actual top-1 routed expert for this token, `0` otherwise.
    /// Increments `mer_speculator_evaluations_total` once per call and
    /// `mer_speculator_accuracy_total` only on a match.
    pub fn record_speculator_top1(&self, top1_match: u64) {
        self.inner.speculator_evaluations_total.inc();
        if top1_match == 1 {
            self.inner.speculator_accuracy_total.inc();
        }
    }

    /// Record locality monitor effectiveness for one token's set of
    /// chosen activations.
    pub fn record_locality(&self, hits: u64, misses: u64) {
        if hits > 0 {
            self.inner.locality_hits_total.inc_by(hits as f64);
        }
        if misses > 0 {
            self.inner.locality_misses_total.inc_by(misses as f64);
        }
    }

    /// Record the SSD stall time portion of the per-token critical path.
    pub fn record_ssd_stall(&self, seconds: f64) {
        self.inner.ssd_stall_seconds.observe(seconds);
    }

    /// Record one VRAM (GPU) tier cache lookup outcome. Mirrors
    /// `record_cache` for the new top-tier in the 3-tier hierarchy.
    pub fn record_gpu_cache(&self, hits: u64, misses: u64) {
        if hits > 0 {
            self.inner.gpu_cache_hits_total.inc_by(hits as f64);
        }
        if misses > 0 {
            self.inner.gpu_cache_misses_total.inc_by(misses as f64);
        }
    }

    /// Set the currently-resident VRAM bytes gauge. Called by the
    /// `GpuExpertCache` whenever the resident set changes (insert /
    /// promote / evict).
    pub fn set_vram_used_bytes(&self, bytes: u64) {
        self.inner.vram_used_bytes.set(bytes as i64);
    }

    /// Set the total VRAM byte budget gauge. Called once when the
    /// `GpuExpertCache` is installed (constant for the lifetime of
    /// the process); stays at `0` when the GPU cache is disabled.
    pub fn set_vram_capacity_bytes(&self, bytes: u64) {
        self.inner.vram_capacity_bytes.set(bytes as i64);
    }

    /// Record `n` RAM → VRAM promotions.
    pub fn record_promotions(&self, n: u64) {
        if n > 0 {
            self.inner.promotions_total.inc_by(n as f64);
        }
    }

    /// Record `n` speculative prefetches dropped to pool starvation.
    pub fn record_prefetch_dropped_pool_starved(&self, n: u64) {
        if n > 0 {
            self.inner
                .prefetch_dropped_pool_starved_total
                .inc_by(n as f64);
        }
    }

    /// Record `n` tokens with the speculator disabled by d_model mismatch.
    pub fn record_speculator_disabled(&self, n: u64) {
        if n > 0 {
            self.inner.speculator_disabled_total.inc_by(n as f64);
        }
    }

    /// Record `n` GPU → CPU expert dispatch fallbacks.
    pub fn record_gpu_cpu_fallback(&self, n: u64) {
        if n > 0 {
            self.inner.gpu_cpu_fallbacks_total.inc_by(n as f64);
        }
    }

    /// Render the registry to a Prometheus text-format payload (the body
    /// of `GET /metrics`).
    pub fn render(&self) -> Result<Vec<u8>, prometheus::Error> {
        // Refresh gauges mirrored from process-wide atomics at scrape time.
        self.inner
            .nonfinite_softmax_fallbacks
            .set(crate::transformer::nonfinite_softmax_fallbacks() as i64);
        let metric_families = self.inner.registry.gather();
        let mut buf = Vec::new();
        TextEncoder::new().encode(&metric_families, &mut buf)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_includes_all_metric_names() {
        let m = Metrics::new();
        m.record_request("/v1/completions", 0.05);
        m.record_tokens(10);
        m.record_cache(3, 1);
        m.record_io_wait(0.002);
        let mut stage_timings = BTreeMap::new();
        stage_timings.insert(
            crate::stage_timing::EMBEDDING.to_string(),
            crate::stage_timing::StageTimingSnapshot {
                count: 2,
                total_seconds: 0.003,
                mean_seconds: 0.0015,
                max_seconds: 0.002,
            },
        );
        m.record_stage_timings(&stage_timings);
        m.record_speculator(7, 3);
        m.record_speculator_top1(1);
        m.record_speculator_top1(0);
        m.record_locality(5, 2);
        m.record_ssd_stall(0.0005);
        m.record_gpu_cache(2, 1);
        m.set_vram_used_bytes(1_048_576);
        m.set_vram_capacity_bytes(8_388_608);
        m.record_promotions(3);
        m.record_prefetch_dropped_pool_starved(1);
        m.record_speculator_disabled(1);
        m.record_gpu_cpu_fallback(1);
        let body = String::from_utf8(m.render().unwrap()).unwrap();
        for name in [
            "mer_requests_total",
            "mer_request_latency_seconds",
            "mer_tokens_generated_total",
            "mer_cache_hits_total",
            "mer_cache_misses_total",
            "mer_io_wait_seconds",
            "mer_stage_seconds_total",
            "mer_stage_events_total",
            "mer_stage_request_seconds",
            "mer_speculator_hits_total",
            "mer_speculator_misses_total",
            "mer_speculator_accuracy_total",
            "mer_speculator_evaluations_total",
            "mer_locality_hits_total",
            "mer_locality_misses_total",
            "mer_ssd_stall_seconds",
            "mer_gpu_cache_hits_total",
            "mer_gpu_cache_misses_total",
            "mer_vram_used_bytes",
            "mer_vram_capacity_bytes",
            "mer_promotions_total",
            "mer_prefetch_dropped_pool_starved_total",
            "mer_speculator_disabled_total",
            "mer_gpu_cpu_fallbacks_total",
            "mer_nonfinite_softmax_fallbacks",
        ] {
            assert!(
                body.contains(name),
                "metric {name} missing from /metrics body:\n{body}"
            );
        }
        assert_metric_value(&body, "mer_speculator_accuracy_total", 1.0);
        assert_metric_value(&body, "mer_speculator_evaluations_total", 2.0);
        assert_metric_value(
            &body,
            r#"mer_stage_seconds_total{stage="embedding"}"#,
            0.003,
        );
        assert_metric_value(&body, r#"mer_stage_events_total{stage="embedding"}"#, 2.0);
        // The label we recorded must show up in the rendered text.
        assert!(body.contains("/v1/completions"));
    }

    #[test]
    fn cache_counters_are_idempotent_under_zero() {
        let m = Metrics::new();
        m.record_cache(0, 0);
        let body = String::from_utf8(m.render().unwrap()).unwrap();
        // Counters should still be exported (with value 0).
        assert!(body.contains("mer_cache_hits_total"));
    }

    fn assert_metric_value(body: &str, name: &str, expected: f64) {
        let line = body
            .lines()
            .find(|line| line.starts_with(name))
            .unwrap_or_else(|| panic!("metric {name} missing from /metrics body:\n{body}"));
        let value: f64 = line
            .split_whitespace()
            .nth(1)
            .unwrap_or_else(|| panic!("metric {name} line missing value: {line}"))
            .parse()
            .unwrap_or_else(|_| panic!("metric {name} line has non-numeric value: {line}"));
        assert!(
            (value - expected).abs() < f64::EPSILON,
            "metric {name} expected {expected}, got {value} in line {line}"
        );
    }
}
