//! Prompt 2 Phase 3A foreground demand-miss observability.
//!
//! The resident path never touches this module. A [`LayerFetchTracker`] is
//! allocated only after a routed layer observes its first initial cache miss;
//! all counters below are consequently miss-only. Benchmark code derives the
//! zero-miss bucket from the already-qualified Phase 1 routed-layer counter.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const LATENCY_BUCKET_UPPER_US: [u64; 16] = [
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    u64::MAX,
];

#[inline]
fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[inline]
fn elapsed_ns(started: Instant) -> u64 {
    duration_ns(started.elapsed())
}

#[inline]
fn ns_seconds(ns: u64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

#[inline]
fn atomic_max(cell: &AtomicU64, value: u64) {
    let _ = cell.fetch_max(value, Ordering::Relaxed);
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LatencyBucketSnapshot {
    /// Inclusive upper bound. `None` is the final overflow bucket.
    pub upper_bound_microseconds: Option<u64>,
    pub count: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct WorstLayerFetch {
    pub token_index: u64,
    pub layer: u32,
    pub final_straggler_expert_id: u32,
    pub final_straggler_routed_slot: usize,
    pub missing_experts: usize,
    pub physical_reads: u64,
    pub critical_path_seconds: f64,
    pub final_straggler_storage_service_seconds: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DemandFetchSnapshot {
    /// Raw miss-only buckets. Bucket zero intentionally remains zero here;
    /// the benchmark report derives it without adding work to resident hits.
    pub missing_experts_per_layer_nonzero: Vec<u64>,
    pub layers_with_multiple_simultaneous_misses: u64,
    pub layers_with_one_physical_read: u64,
    pub layers_with_no_foreground_physical_read: u64,
    pub layers_with_serial_physical_reads: u64,
    pub layers_with_overlapping_physical_reads: u64,
    pub layers_beginning_compute_before_all_misses_available: u64,
    pub foreground_physical_read_operations: u64,
    pub peak_foreground_physical_reads_in_flight: u64,
    /// Integral of active foreground reads over wall time. Dividing by
    /// `foreground_physical_read_active_seconds` yields time-weighted QD.
    pub foreground_physical_read_concurrency_integral_seconds: f64,
    /// Union wall time during which at least one foreground physical read was active.
    pub foreground_physical_read_active_seconds: f64,
    pub average_foreground_physical_read_concurrency: f64,
    pub physical_read_issue_to_completion_seconds: f64,
    pub physical_read_issue_to_completion_mean_seconds: f64,
    pub physical_read_issue_to_completion_max_seconds: f64,
    pub physical_read_issue_to_completion_histogram: Vec<LatencyBucketSnapshot>,
    pub primary_buffer_acquisition_wait_seconds: f64,
    pub primary_buffer_acquisition_wait_mean_seconds: f64,
    pub primary_buffer_acquisition_wait_max_seconds: f64,
    /// Always zero in the audited production path: it has no foreground
    /// admission semaphore. Kept explicit so a future policy cannot hide here.
    pub foreground_admission_wait_seconds: f64,
    pub foreground_admission_wait_mean_seconds: f64,
    pub singleflight_wait_seconds: f64,
    pub completion_to_consumption_delay_seconds: f64,
    pub completion_to_consumption_delay_mean_seconds: f64,
    pub completion_to_consumption_delay_max_seconds: f64,
    pub first_miss_to_first_read_issue_seconds: f64,
    pub first_miss_to_last_read_issue_seconds: f64,
    pub first_to_last_read_issue_spread_seconds: f64,
    pub first_miss_to_first_required_expert_available_seconds: f64,
    pub first_miss_to_final_required_expert_available_seconds: f64,
    pub first_to_last_required_expert_completion_spread_seconds: f64,
    pub first_miss_to_expert_compute_begin_seconds: f64,
    pub first_miss_to_layer_completion_seconds: f64,
    pub layer_expert_fetch_critical_path_seconds: f64,
    pub layer_expert_fetch_critical_path_mean_seconds: f64,
    pub layer_expert_fetch_critical_path_max_seconds: f64,
    pub demand_reads_issued_while_speculative_reads_active: u64,
    /// Device-level delay cannot be inferred merely from overlap, so Phase 3A
    /// reports this as unobservable instead of manufacturing a causal count.
    pub demand_critical_reads_delayed_by_speculative_activity: Option<u64>,
    pub final_straggler_routed_slot_histogram: Vec<u64>,
    pub worst_layer_fetch: Option<WorstLayerFetch>,
}

#[derive(Default)]
struct ConcurrencyState {
    active: u64,
    peak: u64,
    last_change: Option<Instant>,
    active_ns: u64,
    integral_ns: u64,
}

impl ConcurrencyState {
    fn account_until(&mut self, now: Instant) {
        if let Some(last) = self.last_change {
            let dt = duration_ns(now.saturating_duration_since(last));
            if self.active > 0 {
                self.active_ns = self.active_ns.saturating_add(dt);
                self.integral_ns = self
                    .integral_ns
                    .saturating_add(dt.saturating_mul(self.active));
            }
        }
        self.last_change = Some(now);
    }

    fn begin(&mut self, now: Instant) {
        self.account_until(now);
        self.active = self.active.saturating_add(1);
        self.peak = self.peak.max(self.active);
    }

    fn end(&mut self, now: Instant) {
        self.account_until(now);
        self.active = self.active.saturating_sub(1);
        if self.active == 0 {
            self.last_change = None;
        }
    }

    fn snapshot(&self, now: Instant) -> (u64, u64, u64) {
        let mut active_ns = self.active_ns;
        let mut integral_ns = self.integral_ns;
        if self.active > 0 {
            if let Some(last) = self.last_change {
                let dt = duration_ns(now.saturating_duration_since(last));
                active_ns = active_ns.saturating_add(dt);
                integral_ns = integral_ns.saturating_add(dt.saturating_mul(self.active));
            }
        }
        (self.peak, active_ns, integral_ns)
    }

    fn reset_if_idle(&mut self) -> bool {
        if self.active != 0 {
            return false;
        }
        *self = Self::default();
        true
    }
}

#[derive(Default)]
struct WorstState {
    critical_path_ns: u64,
    sample: Option<WorstLayerFetch>,
}

/// Engine-scoped miss-only counters. The benchmark resets this at prompt and
/// decode boundaries, which makes peak concurrency and the bounded worst-layer
/// sample phase-local instead of warmup-contaminated.
pub struct DemandFetchTelemetry {
    missing_layers: Box<[AtomicU64]>,
    layers_multiple: AtomicU64,
    layers_one_read: AtomicU64,
    layers_no_read: AtomicU64,
    layers_serial: AtomicU64,
    layers_overlap: AtomicU64,
    layers_compute_early: AtomicU64,
    reads: AtomicU64,
    service_ns: AtomicU64,
    service_max_ns: AtomicU64,
    latency_buckets: [AtomicU64; LATENCY_BUCKET_UPPER_US.len()],
    buffer_wait_count: AtomicU64,
    buffer_wait_ns: AtomicU64,
    buffer_wait_max_ns: AtomicU64,
    admission_wait_count: AtomicU64,
    admission_wait_ns: AtomicU64,
    singleflight_wait_ns: AtomicU64,
    completion_consumption_count: AtomicU64,
    completion_consumption_ns: AtomicU64,
    completion_consumption_max_ns: AtomicU64,
    discovery_first_issue_ns: AtomicU64,
    discovery_last_issue_ns: AtomicU64,
    issue_spread_ns: AtomicU64,
    discovery_first_available_ns: AtomicU64,
    discovery_last_available_ns: AtomicU64,
    availability_spread_ns: AtomicU64,
    discovery_compute_ns: AtomicU64,
    discovery_layer_complete_ns: AtomicU64,
    layer_critical_ns: AtomicU64,
    layer_critical_max_ns: AtomicU64,
    reads_while_speculative: AtomicU64,
    speculative_active: AtomicU64,
    straggler_slots: Box<[AtomicU64]>,
    concurrency: parking_lot::Mutex<ConcurrencyState>,
    worst: parking_lot::Mutex<WorstState>,
}

impl DemandFetchTelemetry {
    pub fn new(top_k: usize) -> Self {
        let bucket_count = top_k.saturating_add(1).max(1);
        Self {
            missing_layers: (0..bucket_count)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            layers_multiple: AtomicU64::new(0),
            layers_one_read: AtomicU64::new(0),
            layers_no_read: AtomicU64::new(0),
            layers_serial: AtomicU64::new(0),
            layers_overlap: AtomicU64::new(0),
            layers_compute_early: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            service_ns: AtomicU64::new(0),
            service_max_ns: AtomicU64::new(0),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            buffer_wait_count: AtomicU64::new(0),
            buffer_wait_ns: AtomicU64::new(0),
            buffer_wait_max_ns: AtomicU64::new(0),
            admission_wait_count: AtomicU64::new(0),
            admission_wait_ns: AtomicU64::new(0),
            singleflight_wait_ns: AtomicU64::new(0),
            completion_consumption_count: AtomicU64::new(0),
            completion_consumption_ns: AtomicU64::new(0),
            completion_consumption_max_ns: AtomicU64::new(0),
            discovery_first_issue_ns: AtomicU64::new(0),
            discovery_last_issue_ns: AtomicU64::new(0),
            issue_spread_ns: AtomicU64::new(0),
            discovery_first_available_ns: AtomicU64::new(0),
            discovery_last_available_ns: AtomicU64::new(0),
            availability_spread_ns: AtomicU64::new(0),
            discovery_compute_ns: AtomicU64::new(0),
            discovery_layer_complete_ns: AtomicU64::new(0),
            layer_critical_ns: AtomicU64::new(0),
            layer_critical_max_ns: AtomicU64::new(0),
            reads_while_speculative: AtomicU64::new(0),
            speculative_active: AtomicU64::new(0),
            straggler_slots: (0..bucket_count)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            concurrency: parking_lot::Mutex::new(ConcurrencyState::default()),
            worst: parking_lot::Mutex::new(WorstState::default()),
        }
    }

    pub fn begin_speculative_read(self: &Arc<Self>) -> SpeculativeReadGuard {
        self.speculative_active.fetch_add(1, Ordering::Relaxed);
        SpeculativeReadGuard {
            telemetry: self.clone(),
        }
    }

    fn begin_physical_read(&self, now: Instant) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.speculative_active.load(Ordering::Relaxed) > 0 {
            self.reads_while_speculative.fetch_add(1, Ordering::Relaxed);
        }
        self.concurrency.lock().begin(now);
    }

    fn end_physical_read(&self, now: Instant, service_ns: u64) {
        self.concurrency.lock().end(now);
        self.service_ns.fetch_add(service_ns, Ordering::Relaxed);
        atomic_max(&self.service_max_ns, service_ns);
        let service_us = service_ns / 1_000;
        let idx = LATENCY_BUCKET_UPPER_US
            .iter()
            .position(|&upper| service_us <= upper)
            .unwrap_or(LATENCY_BUCKET_UPPER_US.len() - 1);
        self.latency_buckets[idx].fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset(&self) -> bool {
        if !self.concurrency.lock().reset_if_idle() {
            return false;
        }
        for counter in self
            .missing_layers
            .iter()
            .chain(self.straggler_slots.iter())
        {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in &self.latency_buckets {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in [
            &self.layers_multiple,
            &self.layers_one_read,
            &self.layers_no_read,
            &self.layers_serial,
            &self.layers_overlap,
            &self.layers_compute_early,
            &self.reads,
            &self.service_ns,
            &self.service_max_ns,
            &self.buffer_wait_count,
            &self.buffer_wait_ns,
            &self.buffer_wait_max_ns,
            &self.admission_wait_count,
            &self.admission_wait_ns,
            &self.singleflight_wait_ns,
            &self.completion_consumption_count,
            &self.completion_consumption_ns,
            &self.completion_consumption_max_ns,
            &self.discovery_first_issue_ns,
            &self.discovery_last_issue_ns,
            &self.issue_spread_ns,
            &self.discovery_first_available_ns,
            &self.discovery_last_available_ns,
            &self.availability_spread_ns,
            &self.discovery_compute_ns,
            &self.discovery_layer_complete_ns,
            &self.layer_critical_ns,
            &self.layer_critical_max_ns,
            &self.reads_while_speculative,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        *self.worst.lock() = WorstState::default();
        true
    }

    pub fn snapshot(&self) -> DemandFetchSnapshot {
        let reads = self.reads.load(Ordering::Relaxed);
        let service_ns = self.service_ns.load(Ordering::Relaxed);
        let buffer_wait_count = self.buffer_wait_count.load(Ordering::Relaxed);
        let buffer_wait_ns = self.buffer_wait_ns.load(Ordering::Relaxed);
        let admission_wait_count = self.admission_wait_count.load(Ordering::Relaxed);
        let admission_wait_ns = self.admission_wait_ns.load(Ordering::Relaxed);
        let consumption_count = self.completion_consumption_count.load(Ordering::Relaxed);
        let consumption_ns = self.completion_consumption_ns.load(Ordering::Relaxed);
        let miss_layers: u64 = self
            .missing_layers
            .iter()
            .skip(1)
            .map(|v| v.load(Ordering::Relaxed))
            .sum();
        let (peak, active_ns, integral_ns) = self.concurrency.lock().snapshot(Instant::now());
        DemandFetchSnapshot {
            missing_experts_per_layer_nonzero: self
                .missing_layers
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect(),
            layers_with_multiple_simultaneous_misses: self.layers_multiple.load(Ordering::Relaxed),
            layers_with_one_physical_read: self.layers_one_read.load(Ordering::Relaxed),
            layers_with_no_foreground_physical_read: self.layers_no_read.load(Ordering::Relaxed),
            layers_with_serial_physical_reads: self.layers_serial.load(Ordering::Relaxed),
            layers_with_overlapping_physical_reads: self.layers_overlap.load(Ordering::Relaxed),
            layers_beginning_compute_before_all_misses_available: self
                .layers_compute_early
                .load(Ordering::Relaxed),
            foreground_physical_read_operations: reads,
            peak_foreground_physical_reads_in_flight: peak,
            foreground_physical_read_concurrency_integral_seconds: ns_seconds(integral_ns),
            foreground_physical_read_active_seconds: ns_seconds(active_ns),
            average_foreground_physical_read_concurrency: if active_ns == 0 {
                0.0
            } else {
                integral_ns as f64 / active_ns as f64
            },
            physical_read_issue_to_completion_seconds: ns_seconds(service_ns),
            physical_read_issue_to_completion_mean_seconds: if reads == 0 {
                0.0
            } else {
                ns_seconds(service_ns) / reads as f64
            },
            physical_read_issue_to_completion_max_seconds: ns_seconds(
                self.service_max_ns.load(Ordering::Relaxed),
            ),
            physical_read_issue_to_completion_histogram: self
                .latency_buckets
                .iter()
                .enumerate()
                .map(|(idx, count)| LatencyBucketSnapshot {
                    upper_bound_microseconds: (LATENCY_BUCKET_UPPER_US[idx] != u64::MAX)
                        .then_some(LATENCY_BUCKET_UPPER_US[idx]),
                    count: count.load(Ordering::Relaxed),
                })
                .collect(),
            primary_buffer_acquisition_wait_seconds: ns_seconds(buffer_wait_ns),
            primary_buffer_acquisition_wait_mean_seconds: if buffer_wait_count == 0 {
                0.0
            } else {
                ns_seconds(buffer_wait_ns) / buffer_wait_count as f64
            },
            primary_buffer_acquisition_wait_max_seconds: ns_seconds(
                self.buffer_wait_max_ns.load(Ordering::Relaxed),
            ),
            foreground_admission_wait_seconds: ns_seconds(admission_wait_ns),
            foreground_admission_wait_mean_seconds: if admission_wait_count == 0 {
                0.0
            } else {
                ns_seconds(admission_wait_ns) / admission_wait_count as f64
            },
            singleflight_wait_seconds: ns_seconds(
                self.singleflight_wait_ns.load(Ordering::Relaxed),
            ),
            completion_to_consumption_delay_seconds: ns_seconds(consumption_ns),
            completion_to_consumption_delay_mean_seconds: if consumption_count == 0 {
                0.0
            } else {
                ns_seconds(consumption_ns) / consumption_count as f64
            },
            completion_to_consumption_delay_max_seconds: ns_seconds(
                self.completion_consumption_max_ns.load(Ordering::Relaxed),
            ),
            first_miss_to_first_read_issue_seconds: ns_seconds(
                self.discovery_first_issue_ns.load(Ordering::Relaxed),
            ),
            first_miss_to_last_read_issue_seconds: ns_seconds(
                self.discovery_last_issue_ns.load(Ordering::Relaxed),
            ),
            first_to_last_read_issue_spread_seconds: ns_seconds(
                self.issue_spread_ns.load(Ordering::Relaxed),
            ),
            first_miss_to_first_required_expert_available_seconds: ns_seconds(
                self.discovery_first_available_ns.load(Ordering::Relaxed),
            ),
            first_miss_to_final_required_expert_available_seconds: ns_seconds(
                self.discovery_last_available_ns.load(Ordering::Relaxed),
            ),
            first_to_last_required_expert_completion_spread_seconds: ns_seconds(
                self.availability_spread_ns.load(Ordering::Relaxed),
            ),
            first_miss_to_expert_compute_begin_seconds: ns_seconds(
                self.discovery_compute_ns.load(Ordering::Relaxed),
            ),
            first_miss_to_layer_completion_seconds: ns_seconds(
                self.discovery_layer_complete_ns.load(Ordering::Relaxed),
            ),
            layer_expert_fetch_critical_path_seconds: ns_seconds(
                self.layer_critical_ns.load(Ordering::Relaxed),
            ),
            layer_expert_fetch_critical_path_mean_seconds: if miss_layers == 0 {
                0.0
            } else {
                ns_seconds(self.layer_critical_ns.load(Ordering::Relaxed)) / miss_layers as f64
            },
            layer_expert_fetch_critical_path_max_seconds: ns_seconds(
                self.layer_critical_max_ns.load(Ordering::Relaxed),
            ),
            demand_reads_issued_while_speculative_reads_active: self
                .reads_while_speculative
                .load(Ordering::Relaxed),
            demand_critical_reads_delayed_by_speculative_activity: None,
            final_straggler_routed_slot_histogram: self
                .straggler_slots
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect(),
            worst_layer_fetch: self.worst.lock().sample.clone(),
        }
    }
}

pub struct SpeculativeReadGuard {
    telemetry: Arc<DemandFetchTelemetry>,
}

impl Drop for SpeculativeReadGuard {
    fn drop(&mut self) {
        self.telemetry
            .speculative_active
            .fetch_sub(1, Ordering::Relaxed);
    }
}

struct SlotTrace {
    expert_id: u32,
    missed: AtomicU64,
    buffer_wait_ns: AtomicU64,
    singleflight_wait_ns: AtomicU64,
    first_issue_ns: AtomicU64,
    last_issue_ns: AtomicU64,
    last_complete_ns: AtomicU64,
    current_issue_ns: AtomicU64,
    service_ns: AtomicU64,
    physical_reads: AtomicU64,
    available_ns: AtomicU64,
    consumed_ns: AtomicU64,
}

impl SlotTrace {
    fn new(expert_id: u32) -> Self {
        Self {
            expert_id,
            missed: AtomicU64::new(0),
            buffer_wait_ns: AtomicU64::new(0),
            singleflight_wait_ns: AtomicU64::new(0),
            first_issue_ns: AtomicU64::new(0),
            last_issue_ns: AtomicU64::new(0),
            last_complete_ns: AtomicU64::new(0),
            current_issue_ns: AtomicU64::new(0),
            service_ns: AtomicU64::new(0),
            physical_reads: AtomicU64::new(0),
            available_ns: AtomicU64::new(0),
            consumed_ns: AtomicU64::new(0),
        }
    }
}

/// Bounded per-layer trace. It contains exactly one fixed slot per routed
/// expert (top-k) and is dropped after the layer completes.
pub struct LayerFetchTracker {
    token_index: u64,
    layer: u32,
    started: Instant,
    slots: Vec<SlotTrace>,
    missing_count: AtomicU64,
    active_physical_reads: AtomicU64,
    physical_overlap_observed: AtomicU64,
    compute_begin_ns: AtomicU64,
}

impl LayerFetchTracker {
    pub fn new(token_index: u64, layer: u32, experts: &[u32], started: Instant) -> Arc<Self> {
        Arc::new(Self {
            token_index,
            layer,
            started,
            slots: experts.iter().copied().map(SlotTrace::new).collect(),
            missing_count: AtomicU64::new(0),
            active_physical_reads: AtomicU64::new(0),
            physical_overlap_observed: AtomicU64::new(0),
            compute_begin_ns: AtomicU64::new(0),
        })
    }

    pub fn mark_miss(&self, slot: usize) {
        if self.slots[slot].missed.swap(1, Ordering::Relaxed) == 0 {
            self.missing_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_buffer_wait(&self, slot: usize, duration: Duration) {
        self.slots[slot]
            .buffer_wait_ns
            .fetch_add(duration_ns(duration), Ordering::Relaxed);
    }

    pub fn record_singleflight_wait(&self, slot: usize, duration: Duration) {
        self.slots[slot]
            .singleflight_wait_ns
            .fetch_add(duration_ns(duration), Ordering::Relaxed);
    }

    pub fn physical_read_observer(
        self: &Arc<Self>,
        telemetry: Arc<DemandFetchTelemetry>,
        slot: usize,
    ) -> DemandPhysicalReadObserver {
        DemandPhysicalReadObserver {
            layer: self.clone(),
            telemetry,
            slot,
        }
    }

    pub fn record_available(&self, slot: usize) {
        self.slots[slot].available_ns.store(
            elapsed_ns(self.started).saturating_add(1),
            Ordering::Release,
        );
    }

    pub fn record_consumed(&self, slot: usize) {
        self.slots[slot].consumed_ns.store(
            elapsed_ns(self.started).saturating_add(1),
            Ordering::Release,
        );
    }

    pub fn record_compute_begin(&self) {
        self.compute_begin_ns.store(
            elapsed_ns(self.started).saturating_add(1),
            Ordering::Release,
        );
    }

    pub fn finish_fetch(&self, telemetry: &DemandFetchTelemetry) {
        let missing = self.missing_count.load(Ordering::Relaxed) as usize;
        if missing == 0 {
            return;
        }
        let bucket = missing.min(telemetry.missing_layers.len().saturating_sub(1));
        telemetry.missing_layers[bucket].fetch_add(1, Ordering::Relaxed);
        if missing >= 2 {
            telemetry.layers_multiple.fetch_add(1, Ordering::Relaxed);
        }

        let mut first_issue = u64::MAX;
        let mut last_issue = 0u64;
        let mut first_available = u64::MAX;
        let mut last_available = 0u64;
        let mut final_slot = 0usize;
        let mut completion_delay_total = 0u64;
        let mut completion_delay_count = 0u64;
        let mut completion_delay_max = 0u64;
        let mut buffer_wait_total = 0u64;
        let mut buffer_wait_count = 0u64;
        let mut buffer_wait_max = 0u64;
        let mut singleflight_wait_total = 0u64;
        let mut physical_reads = 0u64;

        for (slot, trace) in self.slots.iter().enumerate() {
            if trace.missed.load(Ordering::Relaxed) == 0 {
                continue;
            }
            let issue = trace.first_issue_ns.load(Ordering::Acquire);
            if issue > 0 {
                first_issue = first_issue.min(issue - 1);
            }
            let issue = trace.last_issue_ns.load(Ordering::Acquire);
            if issue > 0 {
                last_issue = last_issue.max(issue - 1);
            }
            physical_reads =
                physical_reads.saturating_add(trace.physical_reads.load(Ordering::Relaxed));
            let available = trace.available_ns.load(Ordering::Acquire);
            if available > 0 {
                let available = available - 1;
                if available < first_available {
                    first_available = available;
                }
                if available >= last_available {
                    last_available = available;
                    final_slot = slot;
                }
                let consumed = trace.consumed_ns.load(Ordering::Acquire);
                if consumed > 0 {
                    let delay = (consumed - 1).saturating_sub(available);
                    completion_delay_total = completion_delay_total.saturating_add(delay);
                    completion_delay_count = completion_delay_count.saturating_add(1);
                    completion_delay_max = completion_delay_max.max(delay);
                }
            }
            let wait = trace.buffer_wait_ns.load(Ordering::Relaxed);
            if trace.physical_reads.load(Ordering::Relaxed) > 0 {
                buffer_wait_count = buffer_wait_count.saturating_add(1);
                buffer_wait_total = buffer_wait_total.saturating_add(wait);
                buffer_wait_max = buffer_wait_max.max(wait);
            }
            singleflight_wait_total = singleflight_wait_total
                .saturating_add(trace.singleflight_wait_ns.load(Ordering::Relaxed));
        }

        match physical_reads {
            0 => {
                telemetry.layers_no_read.fetch_add(1, Ordering::Relaxed);
            }
            1 => {
                telemetry.layers_one_read.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                if self.physical_overlap_observed.load(Ordering::Relaxed) != 0 {
                    telemetry.layers_overlap.fetch_add(1, Ordering::Relaxed);
                } else {
                    telemetry.layers_serial.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        telemetry
            .buffer_wait_count
            .fetch_add(buffer_wait_count, Ordering::Relaxed);
        telemetry
            .buffer_wait_ns
            .fetch_add(buffer_wait_total, Ordering::Relaxed);
        atomic_max(&telemetry.buffer_wait_max_ns, buffer_wait_max);
        telemetry
            .singleflight_wait_ns
            .fetch_add(singleflight_wait_total, Ordering::Relaxed);
        telemetry
            .completion_consumption_count
            .fetch_add(completion_delay_count, Ordering::Relaxed);
        telemetry
            .completion_consumption_ns
            .fetch_add(completion_delay_total, Ordering::Relaxed);
        atomic_max(
            &telemetry.completion_consumption_max_ns,
            completion_delay_max,
        );

        if first_issue != u64::MAX {
            telemetry
                .discovery_first_issue_ns
                .fetch_add(first_issue, Ordering::Relaxed);
            telemetry
                .discovery_last_issue_ns
                .fetch_add(last_issue, Ordering::Relaxed);
            telemetry
                .issue_spread_ns
                .fetch_add(last_issue.saturating_sub(first_issue), Ordering::Relaxed);
        }
        if first_available != u64::MAX {
            telemetry
                .discovery_first_available_ns
                .fetch_add(first_available, Ordering::Relaxed);
            telemetry
                .discovery_last_available_ns
                .fetch_add(last_available, Ordering::Relaxed);
            telemetry.availability_spread_ns.fetch_add(
                last_available.saturating_sub(first_available),
                Ordering::Relaxed,
            );
            telemetry
                .layer_critical_ns
                .fetch_add(last_available, Ordering::Relaxed);
            atomic_max(&telemetry.layer_critical_max_ns, last_available);
            if final_slot < telemetry.straggler_slots.len() {
                telemetry.straggler_slots[final_slot].fetch_add(1, Ordering::Relaxed);
            }
            let compute = self.compute_begin_ns.load(Ordering::Acquire);
            if compute > 0 {
                let compute = compute - 1;
                telemetry
                    .discovery_compute_ns
                    .fetch_add(compute, Ordering::Relaxed);
                if compute < last_available {
                    telemetry
                        .layers_compute_early
                        .fetch_add(1, Ordering::Relaxed);
                }
            }

            let final_trace = &self.slots[final_slot];
            let mut worst = telemetry.worst.lock();
            if last_available >= worst.critical_path_ns {
                worst.critical_path_ns = last_available;
                worst.sample = Some(WorstLayerFetch {
                    token_index: self.token_index,
                    layer: self.layer,
                    final_straggler_expert_id: final_trace.expert_id,
                    final_straggler_routed_slot: final_slot,
                    missing_experts: missing,
                    physical_reads,
                    critical_path_seconds: ns_seconds(last_available),
                    final_straggler_storage_service_seconds: ns_seconds(
                        final_trace.service_ns.load(Ordering::Relaxed),
                    ),
                });
            }
        }
    }

    pub fn finish_layer(&self, telemetry: &DemandFetchTelemetry) {
        telemetry
            .discovery_layer_complete_ns
            .fetch_add(elapsed_ns(self.started), Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_read_interval_ns(
        &self,
        telemetry: &DemandFetchTelemetry,
        slot: usize,
        issue_ns: u64,
        complete_ns: u64,
    ) {
        let trace = &self.slots[slot];
        if self.slots.iter().any(|prior| {
            let prior_issue = prior.first_issue_ns.load(Ordering::Relaxed);
            let prior_complete = prior.last_complete_ns.load(Ordering::Relaxed);
            prior_issue > 0
                && prior_complete > 0
                && issue_ns < prior_complete - 1
                && prior_issue - 1 < complete_ns
        }) {
            self.physical_overlap_observed.store(1, Ordering::Relaxed);
        }
        trace
            .first_issue_ns
            .compare_exchange(0, issue_ns + 1, Ordering::Relaxed, Ordering::Relaxed)
            .ok();
        trace.last_issue_ns.store(issue_ns + 1, Ordering::Relaxed);
        trace
            .last_complete_ns
            .store(complete_ns + 1, Ordering::Relaxed);
        trace
            .service_ns
            .fetch_add(complete_ns.saturating_sub(issue_ns), Ordering::Relaxed);
        trace.physical_reads.fetch_add(1, Ordering::Relaxed);
        telemetry.reads.fetch_add(1, Ordering::Relaxed);
        telemetry
            .service_ns
            .fetch_add(complete_ns.saturating_sub(issue_ns), Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_available_ns(&self, slot: usize, value: u64) {
        self.slots[slot]
            .available_ns
            .store(value + 1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_consumed_ns(&self, slot: usize, value: u64) {
        self.slots[slot]
            .consumed_ns
            .store(value + 1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_compute_ns(&self, value: u64) {
        self.compute_begin_ns.store(value + 1, Ordering::Relaxed);
    }
}

/// Observer installed only around the foreground `NvmeStorage` call. Its
/// issue callback runs after fd lookup and immediately before `block_in_place`
/// enters the positional-read service; completion runs immediately after the
/// storage call returns. No tracker lock is held across I/O.
pub struct DemandPhysicalReadObserver {
    layer: Arc<LayerFetchTracker>,
    telemetry: Arc<DemandFetchTelemetry>,
    slot: usize,
}

impl crate::io_provider::PhysicalReadObserver for DemandPhysicalReadObserver {
    fn issued(&self) {
        let now = Instant::now();
        let offset = duration_ns(now.saturating_duration_since(self.layer.started));
        let trace = &self.layer.slots[self.slot];
        trace
            .first_issue_ns
            .compare_exchange(0, offset + 1, Ordering::Relaxed, Ordering::Relaxed)
            .ok();
        trace.last_issue_ns.store(offset + 1, Ordering::Release);
        trace.current_issue_ns.store(offset + 1, Ordering::Release);
        trace.physical_reads.fetch_add(1, Ordering::Relaxed);
        if self
            .layer
            .active_physical_reads
            .fetch_add(1, Ordering::AcqRel)
            > 0
        {
            self.layer
                .physical_overlap_observed
                .store(1, Ordering::Relaxed);
        }
        self.telemetry.begin_physical_read(now);
    }

    fn completed(&self) {
        self.layer
            .active_physical_reads
            .fetch_sub(1, Ordering::AcqRel);
        let now = Instant::now();
        let offset = duration_ns(now.saturating_duration_since(self.layer.started));
        let trace = &self.layer.slots[self.slot];
        let issue = trace.current_issue_ns.swap(0, Ordering::AcqRel);
        let service_ns = if issue == 0 {
            0
        } else {
            offset.saturating_sub(issue - 1)
        };
        trace
            .last_complete_ns
            .store(offset.saturating_add(1), Ordering::Release);
        trace.service_ns.fetch_add(service_ns, Ordering::Relaxed);
        self.telemetry.end_physical_read(now, service_ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_layer(
        top_k: usize,
        misses: &[(usize, u32)],
    ) -> (Arc<DemandFetchTelemetry>, Arc<LayerFetchTracker>) {
        let telemetry = Arc::new(DemandFetchTelemetry::new(top_k));
        let experts: Vec<u32> = (0..top_k as u32).collect();
        let layer = LayerFetchTracker::new(7, 3, &experts, Instant::now());
        for &(slot, _) in misses {
            layer.mark_miss(slot);
        }
        (telemetry, layer)
    }

    #[test]
    fn zero_miss_epoch_remains_inactive_and_resets() {
        let telemetry = DemandFetchTelemetry::new(8);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.foreground_physical_read_operations, 0);
        assert!(snapshot
            .missing_experts_per_layer_nonzero
            .iter()
            .all(|&count| count == 0));
        assert!(telemetry.reset());
    }

    #[test]
    fn one_and_multiple_missing_experts_use_exact_buckets() {
        let (one_t, one) = synthetic_layer(8, &[(2, 2)]);
        one.record_available_ns(2, 20);
        one.record_consumed_ns(2, 21);
        one.record_compute_ns(22);
        one.finish_fetch(&one_t);
        assert_eq!(one_t.snapshot().missing_experts_per_layer_nonzero[1], 1);

        let (many_t, many) = synthetic_layer(8, &[(0, 0), (3, 3), (7, 7)]);
        for (slot, at) in [(0, 20), (3, 30), (7, 40)] {
            many.record_available_ns(slot, at);
            many.record_consumed_ns(slot, 45);
        }
        many.record_compute_ns(46);
        many.finish_fetch(&many_t);
        let snapshot = many_t.snapshot();
        assert_eq!(snapshot.missing_experts_per_layer_nonzero[3], 1);
        assert_eq!(snapshot.layers_with_multiple_simultaneous_misses, 1);
    }

    #[test]
    fn overlapping_and_serial_reads_are_distinguished() {
        let (overlap_t, overlap) = synthetic_layer(2, &[(0, 0), (1, 1)]);
        overlap.record_read_interval_ns(&overlap_t, 0, 10, 30);
        overlap.record_read_interval_ns(&overlap_t, 1, 20, 40);
        for (slot, at) in [(0, 31), (1, 41)] {
            overlap.record_available_ns(slot, at);
            overlap.record_consumed_ns(slot, 42);
        }
        overlap.record_compute_ns(43);
        overlap.finish_fetch(&overlap_t);
        assert_eq!(
            overlap_t.snapshot().layers_with_overlapping_physical_reads,
            1
        );

        let (serial_t, serial) = synthetic_layer(2, &[(0, 0), (1, 1)]);
        serial.record_read_interval_ns(&serial_t, 0, 10, 20);
        serial.record_read_interval_ns(&serial_t, 1, 20, 30);
        for (slot, at) in [(0, 21), (1, 31)] {
            serial.record_available_ns(slot, at);
            serial.record_consumed_ns(slot, 32);
        }
        serial.record_compute_ns(33);
        serial.finish_fetch(&serial_t);
        assert_eq!(serial_t.snapshot().layers_with_serial_physical_reads, 1);
    }

    #[test]
    fn time_weighted_concurrency_and_peak_are_exact() {
        let base = Instant::now();
        let at = |ns| base + Duration::from_nanos(ns);
        let mut state = ConcurrencyState::default();
        state.begin(at(0));
        state.begin(at(10));
        state.end(at(30));
        state.end(at(50));
        let (peak, active_ns, integral_ns) = state.snapshot(at(60));
        assert_eq!(peak, 2);
        assert_eq!(active_ns, 50);
        assert_eq!(integral_ns, 70); // 1*10 + 2*20 + 1*20
        assert!((integral_ns as f64 / active_ns as f64 - 1.4).abs() < f64::EPSILON);
    }

    #[test]
    fn buffer_and_admission_waits_are_not_storage_latency() {
        let (telemetry, layer) = synthetic_layer(1, &[(0, 0)]);
        layer.slots[0].buffer_wait_ns.store(11, Ordering::Relaxed);
        layer.record_read_interval_ns(&telemetry, 0, 20, 50);
        layer.record_available_ns(0, 51);
        layer.record_consumed_ns(0, 52);
        layer.record_compute_ns(53);
        layer.finish_fetch(&telemetry);
        let snapshot = telemetry.snapshot();
        assert_eq!(
            snapshot.primary_buffer_acquisition_wait_seconds,
            ns_seconds(11)
        );
        assert_eq!(
            snapshot.physical_read_issue_to_completion_seconds,
            ns_seconds(30)
        );
        assert_eq!(snapshot.foreground_admission_wait_seconds, 0.0);
    }

    #[test]
    fn slow_read_is_final_straggler_and_unrelated_ids_overlap() {
        let (telemetry, layer) = synthetic_layer(3, &[(0, 0), (2, 2)]);
        layer.record_read_interval_ns(&telemetry, 0, 10, 20);
        layer.record_read_interval_ns(&telemetry, 2, 11, 50);
        layer.record_available_ns(0, 21);
        layer.record_available_ns(2, 51);
        layer.record_consumed_ns(0, 22);
        layer.record_consumed_ns(2, 52);
        layer.record_compute_ns(53);
        layer.finish_fetch(&telemetry);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.layers_with_overlapping_physical_reads, 1);
        let worst = snapshot.worst_layer_fetch.expect("worst layer");
        assert_eq!(worst.final_straggler_expert_id, 2);
        assert_eq!(worst.final_straggler_routed_slot, 2);
        assert_eq!(
            snapshot.layers_beginning_compute_before_all_misses_available,
            0
        );
    }

    #[test]
    fn reset_clears_bounded_counters_between_runs() {
        let (telemetry, layer) = synthetic_layer(1, &[(0, 0)]);
        layer.record_read_interval_ns(&telemetry, 0, 1, 2);
        layer.record_available_ns(0, 3);
        layer.record_consumed_ns(0, 4);
        layer.record_compute_ns(5);
        layer.finish_fetch(&telemetry);
        assert_eq!(telemetry.snapshot().missing_experts_per_layer_nonzero[1], 1);
        assert!(telemetry.reset());
        let after = telemetry.snapshot();
        assert_eq!(after.missing_experts_per_layer_nonzero[1], 0);
        assert_eq!(after.foreground_physical_read_operations, 0);
        assert!(after.worst_layer_fetch.is_none());
    }
}
