//! Adaptive, self-regulating prefetch admission controller.
//!
//! ## Why this exists
//!
//! The legacy speculative-I/O path admits every predicted expert that
//! clears a *static* probability threshold and can grab a semaphore
//! permit. On a structureless or weakly-correlated workload the
//! predictors degrade to ~chance accuracy, yet the engine keeps issuing
//! the same volume of speculative reads. Those reads are not free: on a
//! bandwidth-bound SSD they **queue ahead of foreground cache misses**,
//! inflating the latency of the reads that actually block token
//! generation. Field data from a Mixtral-8x7B run made this concrete —
//! with ~9k speculative reads at ~0.8 % precision the foreground miss
//! p50 rose from ~120 ms (no speculation) to ~405 ms (3.4x), while hit
//! rate barely moved.
//!
//! The [`PrefetchGovernor`] closes the loop: it continuously measures
//!
//! * **precision** — the fraction of completed prefetches that were
//!   actually consumed before eviction (an EWMA), and
//! * **contention** — how many foreground (blocking) reads are in
//!   flight right now,
//!
//! and admits a speculative read only when its *expected value* beats
//! the *expected contention cost*. When the predictors are paying off
//! and the disk is idle it admits liberally; when precision collapses or
//! real misses are queued it throttles toward zero, handing the scarce
//! I/O bandwidth back to the foreground path.
//!
//! The controller is **opt-in** (`EngineOptions::prefetch_governor`,
//! default `false`) so existing deployments and benchmarks are
//! bit-for-bit unchanged until they enable it.

use serde::Serialize;
use std::sync::{
    atomic::{AtomicI64, AtomicU64, Ordering},
    Mutex,
};

/// Maximum number of exact decision samples retained for percentile
/// calculation. Counts, means, extrema, and decision-boundary values are
/// accumulated over every decision in the window; only percentile samples
/// are bounded to the most recent entries.
pub const GOVERNOR_SCORE_SAMPLE_CAPACITY: usize = 512;

/// Pack/unpack an `f64` into the bits of an `AtomicU64` so the EWMA can
/// be read on the hot admission path with a single relaxed load.
#[inline]
fn load_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Relaxed))
}
#[inline]
fn store_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug)]
struct GovernorDecision {
    candidate_probability: f64,
    effective_precision: f64,
    candidate_score: f64,
    base_threshold: f64,
    foreground_inflight: u64,
    contention_weight: f64,
    effective_threshold: f64,
    admitted: bool,
    score_minus_threshold_margin: f64,
    score_to_threshold_ratio: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct GovernorDecisionSample {
    candidate_probability: f64,
    effective_precision: f64,
    candidate_score: f64,
    effective_threshold: f64,
    score_to_threshold_ratio: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunningStats {
    count: u64,
    minimum: Option<f64>,
    maximum: Option<f64>,
    mean: f64,
}

impl RunningStats {
    fn record(&mut self, value: f64) {
        debug_assert!(value.is_finite());
        self.count = self.count.saturating_add(1);
        self.minimum = Some(self.minimum.map_or(value, |current| current.min(value)));
        self.maximum = Some(self.maximum.map_or(value, |current| current.max(value)));
        self.mean += (value - self.mean) / self.count as f64;
    }
}

/// Finite distribution summary. Every value is `null` when `count == 0`.
/// Percentiles use nearest-rank selection over the bounded decision sample.
#[derive(Clone, Debug, Serialize)]
pub struct GovernorDistributionSummary {
    pub count: u64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

impl GovernorDistributionSummary {
    fn empty() -> Self {
        Self {
            count: 0,
            minimum: None,
            maximum: None,
            mean: None,
            p50: None,
            p90: None,
            p95: None,
            p99: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GovernorDecisionBoundary {
    pub maximum_rejected_candidate_score: Option<f64>,
    pub maximum_rejected_score_to_threshold_ratio: Option<f64>,
    pub minimum_admitted_candidate_score: Option<f64>,
    pub minimum_admitted_score_to_threshold_ratio: Option<f64>,
    pub closest_rejected_score_minus_threshold_margin: Option<f64>,
    pub closest_admitted_score_minus_threshold_margin: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GovernorForegroundDecisionCounts {
    pub foreground_inflight_zero: u64,
    pub foreground_inflight_zero_admitted: u64,
    pub foreground_inflight_zero_rejected: u64,
    pub foreground_inflight_positive: u64,
    pub foreground_inflight_positive_admitted: u64,
    pub foreground_inflight_positive_rejected: u64,
}

/// Bounded Phase 4D-B governor score diagnostics for one reset window.
///
/// Exact counters and running statistics cover every decision. Percentiles are
/// nearest-rank values over a deterministic ring containing the most recent
/// [`GOVERNOR_SCORE_SAMPLE_CAPACITY`] finite decisions. The private sample is
/// retained only so measured-run snapshots can be merged without serializing
/// per-decision events.
#[derive(Clone, Debug, Serialize)]
pub struct GovernorDecisionDiagnosticsSnapshot {
    pub semantics: &'static str,
    pub enabled: bool,
    pub sample_capacity: usize,
    pub sampled_decisions: usize,
    pub total_decisions: u64,
    pub admitted: u64,
    pub rejected: u64,
    pub invalid_numeric_decisions: u64,
    pub ratio_undefined_decisions: u64,
    pub base_threshold: f64,
    pub contention_weight: f64,
    pub candidate_probability: GovernorDistributionSummary,
    pub effective_precision: GovernorDistributionSummary,
    pub candidate_score: GovernorDistributionSummary,
    pub effective_threshold: GovernorDistributionSummary,
    pub score_to_threshold_ratio: GovernorDistributionSummary,
    pub decision_boundary: GovernorDecisionBoundary,
    pub foreground_inflight_decisions: GovernorForegroundDecisionCounts,
    #[serde(skip)]
    samples: Vec<GovernorDecisionSample>,
}

#[derive(Debug, Default)]
struct GovernorDecisionDiagnostics {
    total_decisions: u64,
    admitted: u64,
    rejected: u64,
    invalid_numeric_decisions: u64,
    ratio_undefined_decisions: u64,
    candidate_probability: RunningStats,
    effective_precision: RunningStats,
    candidate_score: RunningStats,
    effective_threshold: RunningStats,
    score_to_threshold_ratio: RunningStats,
    decision_boundary: GovernorDecisionBoundary,
    foreground_inflight_decisions: GovernorForegroundDecisionCounts,
    samples: Vec<GovernorDecisionSample>,
    next_sample: usize,
}

impl GovernorDecisionDiagnostics {
    fn record(&mut self, decision: GovernorDecision) {
        self.total_decisions = self.total_decisions.saturating_add(1);
        if decision.admitted {
            self.admitted = self.admitted.saturating_add(1);
        } else {
            self.rejected = self.rejected.saturating_add(1);
        }

        let foreground = &mut self.foreground_inflight_decisions;
        if decision.foreground_inflight == 0 {
            foreground.foreground_inflight_zero =
                foreground.foreground_inflight_zero.saturating_add(1);
            if decision.admitted {
                foreground.foreground_inflight_zero_admitted = foreground
                    .foreground_inflight_zero_admitted
                    .saturating_add(1);
            } else {
                foreground.foreground_inflight_zero_rejected = foreground
                    .foreground_inflight_zero_rejected
                    .saturating_add(1);
            }
        } else {
            foreground.foreground_inflight_positive =
                foreground.foreground_inflight_positive.saturating_add(1);
            if decision.admitted {
                foreground.foreground_inflight_positive_admitted = foreground
                    .foreground_inflight_positive_admitted
                    .saturating_add(1);
            } else {
                foreground.foreground_inflight_positive_rejected = foreground
                    .foreground_inflight_positive_rejected
                    .saturating_add(1);
            }
        }

        let required_values = [
            decision.candidate_probability,
            decision.effective_precision,
            decision.candidate_score,
            decision.base_threshold,
            decision.contention_weight,
            decision.effective_threshold,
            decision.score_minus_threshold_margin,
        ];
        if required_values.iter().any(|value| !value.is_finite()) {
            self.invalid_numeric_decisions = self.invalid_numeric_decisions.saturating_add(1);
            return;
        }

        self.candidate_probability
            .record(decision.candidate_probability);
        self.effective_precision
            .record(decision.effective_precision);
        self.candidate_score.record(decision.candidate_score);
        self.effective_threshold
            .record(decision.effective_threshold);
        if let Some(ratio) = decision.score_to_threshold_ratio {
            if ratio.is_finite() {
                self.score_to_threshold_ratio.record(ratio);
            } else {
                self.ratio_undefined_decisions = self.ratio_undefined_decisions.saturating_add(1);
            }
        } else {
            self.ratio_undefined_decisions = self.ratio_undefined_decisions.saturating_add(1);
        }

        let boundary = &mut self.decision_boundary;
        if decision.admitted {
            boundary.minimum_admitted_candidate_score = Some(
                boundary
                    .minimum_admitted_candidate_score
                    .map_or(decision.candidate_score, |current| {
                        current.min(decision.candidate_score)
                    }),
            );
            boundary.closest_admitted_score_minus_threshold_margin = Some(
                boundary
                    .closest_admitted_score_minus_threshold_margin
                    .map_or(decision.score_minus_threshold_margin, |current| {
                        current.min(decision.score_minus_threshold_margin)
                    }),
            );
            if let Some(ratio) = decision.score_to_threshold_ratio.filter(|v| v.is_finite()) {
                boundary.minimum_admitted_score_to_threshold_ratio = Some(
                    boundary
                        .minimum_admitted_score_to_threshold_ratio
                        .map_or(ratio, |current| current.min(ratio)),
                );
            }
        } else {
            boundary.maximum_rejected_candidate_score = Some(
                boundary
                    .maximum_rejected_candidate_score
                    .map_or(decision.candidate_score, |current| {
                        current.max(decision.candidate_score)
                    }),
            );
            boundary.closest_rejected_score_minus_threshold_margin = Some(
                boundary
                    .closest_rejected_score_minus_threshold_margin
                    .map_or(decision.score_minus_threshold_margin, |current| {
                        current.max(decision.score_minus_threshold_margin)
                    }),
            );
            if let Some(ratio) = decision.score_to_threshold_ratio.filter(|v| v.is_finite()) {
                boundary.maximum_rejected_score_to_threshold_ratio = Some(
                    boundary
                        .maximum_rejected_score_to_threshold_ratio
                        .map_or(ratio, |current| current.max(ratio)),
                );
            }
        }

        let sample = GovernorDecisionSample {
            candidate_probability: decision.candidate_probability,
            effective_precision: decision.effective_precision,
            candidate_score: decision.candidate_score,
            effective_threshold: decision.effective_threshold,
            score_to_threshold_ratio: decision
                .score_to_threshold_ratio
                .filter(|value| value.is_finite()),
        };
        if self.samples.len() < GOVERNOR_SCORE_SAMPLE_CAPACITY {
            self.samples.push(sample);
        } else {
            self.samples[self.next_sample] = sample;
            self.next_sample = (self.next_sample + 1) % GOVERNOR_SCORE_SAMPLE_CAPACITY;
        }
    }

    fn chronological_samples(&self) -> Vec<GovernorDecisionSample> {
        if self.samples.len() < GOVERNOR_SCORE_SAMPLE_CAPACITY || self.next_sample == 0 {
            return self.samples.clone();
        }
        let mut samples = Vec::with_capacity(GOVERNOR_SCORE_SAMPLE_CAPACITY);
        samples.extend_from_slice(&self.samples[self.next_sample..]);
        samples.extend_from_slice(&self.samples[..self.next_sample]);
        samples
    }
}

fn nearest_rank(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let rank = (quantile * values.len() as f64).ceil().max(1.0) as usize;
    values.get(rank.saturating_sub(1)).copied()
}

fn distribution_summary(
    stats: RunningStats,
    samples: &[GovernorDecisionSample],
    select: fn(&GovernorDecisionSample) -> Option<f64>,
) -> GovernorDistributionSummary {
    if stats.count == 0 {
        return GovernorDistributionSummary::empty();
    }
    let sampled: Vec<f64> = samples.iter().filter_map(select).collect();
    let percentile = |quantile| {
        let mut values = sampled.clone();
        nearest_rank(&mut values, quantile)
    };
    GovernorDistributionSummary {
        count: stats.count,
        minimum: stats.minimum,
        maximum: stats.maximum,
        mean: Some(stats.mean),
        p50: percentile(0.50),
        p90: percentile(0.90),
        p95: percentile(0.95),
        p99: percentile(0.99),
    }
}

fn merge_distribution(
    summaries: impl Iterator<Item = GovernorDistributionSummary>,
    samples: &[GovernorDecisionSample],
    select: fn(&GovernorDecisionSample) -> Option<f64>,
) -> GovernorDistributionSummary {
    let mut merged = RunningStats::default();
    for summary in summaries {
        if summary.count == 0 {
            continue;
        }
        let prior_count = merged.count;
        let next_count = prior_count.saturating_add(summary.count);
        let summary_mean = summary
            .mean
            .expect("non-empty governor distribution has a mean");
        merged.mean = if prior_count == 0 {
            summary_mean
        } else {
            merged.mean + (summary_mean - merged.mean) * (summary.count as f64 / next_count as f64)
        };
        merged.count = next_count;
        if let Some(minimum) = summary.minimum {
            merged.minimum = Some(
                merged
                    .minimum
                    .map_or(minimum, |current| current.min(minimum)),
            );
        }
        if let Some(maximum) = summary.maximum {
            merged.maximum = Some(
                merged
                    .maximum
                    .map_or(maximum, |current| current.max(maximum)),
            );
        }
    }
    distribution_summary(merged, samples, select)
}

fn push_bounded_sample(
    samples: &mut Vec<GovernorDecisionSample>,
    next_sample: &mut usize,
    sample: GovernorDecisionSample,
) {
    if samples.len() < GOVERNOR_SCORE_SAMPLE_CAPACITY {
        samples.push(sample);
    } else {
        samples[*next_sample] = sample;
        *next_sample = (*next_sample + 1) % GOVERNOR_SCORE_SAMPLE_CAPACITY;
    }
}

fn optional_max(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(|current, value| current.max(value))
}

fn optional_min(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(|current, value| current.min(value))
}

impl GovernorDecisionDiagnosticsSnapshot {
    fn from_state(
        enabled: bool,
        base_threshold: f64,
        contention_weight: f64,
        state: &GovernorDecisionDiagnostics,
    ) -> Self {
        let samples = state.chronological_samples();
        Self {
            semantics: "exact all-decision counts, running means, extrema, boundaries, and foreground splits; nearest-rank percentiles over a deterministic ring of the most recent 512 finite decisions since reset; ratio is undefined and excluded when the effective threshold is zero or division is nonfinite",
            enabled,
            sample_capacity: GOVERNOR_SCORE_SAMPLE_CAPACITY,
            sampled_decisions: samples.len(),
            total_decisions: state.total_decisions,
            admitted: state.admitted,
            rejected: state.rejected,
            invalid_numeric_decisions: state.invalid_numeric_decisions,
            ratio_undefined_decisions: state.ratio_undefined_decisions,
            base_threshold,
            contention_weight,
            candidate_probability: distribution_summary(
                state.candidate_probability,
                &samples,
                |sample| Some(sample.candidate_probability),
            ),
            effective_precision: distribution_summary(
                state.effective_precision,
                &samples,
                |sample| Some(sample.effective_precision),
            ),
            candidate_score: distribution_summary(
                state.candidate_score,
                &samples,
                |sample| Some(sample.candidate_score),
            ),
            effective_threshold: distribution_summary(
                state.effective_threshold,
                &samples,
                |sample| Some(sample.effective_threshold),
            ),
            score_to_threshold_ratio: distribution_summary(
                state.score_to_threshold_ratio,
                &samples,
                |sample| sample.score_to_threshold_ratio,
            ),
            decision_boundary: state.decision_boundary.clone(),
            foreground_inflight_decisions: state.foreground_inflight_decisions.clone(),
            samples,
        }
    }

    /// Merge measured-run diagnostics. Exact summaries are merged by count;
    /// percentiles use the same bounded ring semantics over the retained run
    /// samples, in run order.
    pub fn merge(snapshots: &[Self]) -> Self {
        let default = GovernorConfig::default();
        let enabled = snapshots.iter().any(|snapshot| snapshot.enabled);
        let base_threshold = snapshots
            .first()
            .map(|snapshot| snapshot.base_threshold)
            .unwrap_or(default.base_threshold);
        let contention_weight = snapshots
            .first()
            .map(|snapshot| snapshot.contention_weight)
            .unwrap_or(default.contention_weight);
        let mut samples = Vec::with_capacity(GOVERNOR_SCORE_SAMPLE_CAPACITY);
        let mut next_sample = 0;
        for snapshot in snapshots {
            for &sample in &snapshot.samples {
                push_bounded_sample(&mut samples, &mut next_sample, sample);
            }
        }
        if samples.len() == GOVERNOR_SCORE_SAMPLE_CAPACITY && next_sample > 0 {
            samples.rotate_left(next_sample);
        }
        let total_decisions = snapshots
            .iter()
            .map(|snapshot| snapshot.total_decisions)
            .sum();
        let admitted = snapshots.iter().map(|snapshot| snapshot.admitted).sum();
        let rejected = snapshots.iter().map(|snapshot| snapshot.rejected).sum();
        let invalid_numeric_decisions = snapshots
            .iter()
            .map(|snapshot| snapshot.invalid_numeric_decisions)
            .sum();
        let ratio_undefined_decisions = snapshots
            .iter()
            .map(|snapshot| snapshot.ratio_undefined_decisions)
            .sum();
        let foreground_inflight_decisions = GovernorForegroundDecisionCounts {
            foreground_inflight_zero: snapshots
                .iter()
                .map(|snapshot| {
                    snapshot
                        .foreground_inflight_decisions
                        .foreground_inflight_zero
                })
                .sum(),
            foreground_inflight_zero_admitted: snapshots
                .iter()
                .map(|snapshot| {
                    snapshot
                        .foreground_inflight_decisions
                        .foreground_inflight_zero_admitted
                })
                .sum(),
            foreground_inflight_zero_rejected: snapshots
                .iter()
                .map(|snapshot| {
                    snapshot
                        .foreground_inflight_decisions
                        .foreground_inflight_zero_rejected
                })
                .sum(),
            foreground_inflight_positive: snapshots
                .iter()
                .map(|snapshot| {
                    snapshot
                        .foreground_inflight_decisions
                        .foreground_inflight_positive
                })
                .sum(),
            foreground_inflight_positive_admitted: snapshots
                .iter()
                .map(|snapshot| {
                    snapshot
                        .foreground_inflight_decisions
                        .foreground_inflight_positive_admitted
                })
                .sum(),
            foreground_inflight_positive_rejected: snapshots
                .iter()
                .map(|snapshot| {
                    snapshot
                        .foreground_inflight_decisions
                        .foreground_inflight_positive_rejected
                })
                .sum(),
        };
        Self {
            semantics: "exact merged measured-run counts, weighted running means, extrema, boundaries, and foreground splits; nearest-rank percentiles over a deterministic ring of the most recent 512 retained finite decision samples in run order; ratio excludes zero-threshold or nonfinite divisions",
            enabled,
            sample_capacity: GOVERNOR_SCORE_SAMPLE_CAPACITY,
            sampled_decisions: samples.len(),
            total_decisions,
            admitted,
            rejected,
            invalid_numeric_decisions,
            ratio_undefined_decisions,
            base_threshold,
            contention_weight,
            candidate_probability: merge_distribution(
                snapshots
                    .iter()
                    .map(|snapshot| snapshot.candidate_probability.clone()),
                &samples,
                |sample| Some(sample.candidate_probability),
            ),
            effective_precision: merge_distribution(
                snapshots
                    .iter()
                    .map(|snapshot| snapshot.effective_precision.clone()),
                &samples,
                |sample| Some(sample.effective_precision),
            ),
            candidate_score: merge_distribution(
                snapshots
                    .iter()
                    .map(|snapshot| snapshot.candidate_score.clone()),
                &samples,
                |sample| Some(sample.candidate_score),
            ),
            effective_threshold: merge_distribution(
                snapshots
                    .iter()
                    .map(|snapshot| snapshot.effective_threshold.clone()),
                &samples,
                |sample| Some(sample.effective_threshold),
            ),
            score_to_threshold_ratio: merge_distribution(
                snapshots
                    .iter()
                    .map(|snapshot| snapshot.score_to_threshold_ratio.clone()),
                &samples,
                |sample| sample.score_to_threshold_ratio,
            ),
            decision_boundary: GovernorDecisionBoundary {
                maximum_rejected_candidate_score: optional_max(
                    snapshots.iter().map(|snapshot| {
                        snapshot
                            .decision_boundary
                            .maximum_rejected_candidate_score
                    }),
                ),
                maximum_rejected_score_to_threshold_ratio: optional_max(
                    snapshots.iter().map(|snapshot| {
                        snapshot
                            .decision_boundary
                            .maximum_rejected_score_to_threshold_ratio
                    }),
                ),
                minimum_admitted_candidate_score: optional_min(
                    snapshots.iter().map(|snapshot| {
                        snapshot
                            .decision_boundary
                            .minimum_admitted_candidate_score
                    }),
                ),
                minimum_admitted_score_to_threshold_ratio: optional_min(
                    snapshots.iter().map(|snapshot| {
                        snapshot
                            .decision_boundary
                            .minimum_admitted_score_to_threshold_ratio
                    }),
                ),
                closest_rejected_score_minus_threshold_margin: optional_max(
                    snapshots.iter().map(|snapshot| {
                        snapshot
                            .decision_boundary
                            .closest_rejected_score_minus_threshold_margin
                    }),
                ),
                closest_admitted_score_minus_threshold_margin: optional_min(
                    snapshots.iter().map(|snapshot| {
                        snapshot
                            .decision_boundary
                            .closest_admitted_score_minus_threshold_margin
                    }),
                ),
            },
            foreground_inflight_decisions,
            samples,
        }
    }
}

/// Tunables for [`PrefetchGovernor`]. All values have safe, conservative
/// defaults; the engine fills these from [`EngineOptions`].
#[derive(Clone, Copy, Debug)]
pub struct GovernorConfig {
    /// EWMA smoothing factor for the measured precision signal, in
    /// `(0, 1]`. Higher reacts faster to distribution shift; lower is
    /// steadier. `0.2` blends roughly the last ~5 measurement windows.
    pub precision_alpha: f64,
    /// Floor applied to the precision EWMA when scoring admissions, so a
    /// transient run of wasted prefetches can't latch the controller at
    /// exactly zero and starve it of the future hits that would let it
    /// recover. Also the value the EWMA is seeded with.
    pub precision_floor: f64,
    /// Per-outstanding-foreground-read multiplier on the admission
    /// threshold. With `contention_weight = 1.0`, one in-flight
    /// foreground miss doubles the bar a speculative read must clear,
    /// two triple it, and so on.
    pub contention_weight: f64,
    /// Base admission threshold the (probability x precision) product is
    /// compared against when the disk is otherwise idle.
    pub base_threshold: f64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            precision_alpha: 0.2,
            precision_floor: 0.05,
            contention_weight: 1.0,
            base_threshold: 0.02,
        }
    }
}

/// Adaptive prefetch admission controller. The decision inputs and direct
/// counters remain atomic; enabled Phase 4D-B diagnostics add one bounded
/// telemetry lock after the decision has been computed.
#[derive(Debug)]
pub struct PrefetchGovernor {
    enabled: bool,
    precision_alpha: f64,
    precision_floor: f64,
    contention_weight: f64,
    base_threshold: f64,

    /// EWMA of recent prefetch precision (consumed / completed), in
    /// `[0, 1]`. Read on the admission hot path.
    precision_ewma: AtomicU64,

    /// Gauge of foreground (blocking) reads currently in flight. A
    /// speculative read admitted while this is non-zero is directly
    /// competing with a token-blocking miss for device bandwidth.
    foreground_inflight: AtomicI64,

    /// Rolling within-window counters folded into the EWMA by
    /// [`Self::refresh`]. `completed` counts prefetch reads that landed;
    /// `used` counts those that were consumed by a subsequent hit before
    /// eviction.
    window_completed: AtomicU64,
    window_used: AtomicU64,

    /// Telemetry: prefetches the governor declined to admit.
    throttled: AtomicU64,
    /// Telemetry: prefetches the governor admitted.
    admitted: AtomicU64,

    /// Bounded score/threshold diagnostics. This lock never participates in
    /// the admission calculation; poisoning is recovered so telemetry cannot
    /// change a decision or fail the request path.
    decision_diagnostics: Mutex<GovernorDecisionDiagnostics>,
}

/// RAII token for foreground-read accounting. Dropping the guard always
/// balances the corresponding [`PrefetchGovernor::begin_foreground`] call.
pub struct ForegroundGuard<'a> {
    governor: &'a PrefetchGovernor,
}

impl<'a> ForegroundGuard<'a> {
    fn new(governor: &'a PrefetchGovernor) -> Self {
        governor.begin_foreground();
        Self { governor }
    }
}

impl Drop for ForegroundGuard<'_> {
    fn drop(&mut self) {
        self.governor.end_foreground();
    }
}

impl PrefetchGovernor {
    /// Construct a governor. When `enabled` is `false` the controller is
    /// a transparent pass-through: [`Self::admit`] always returns `true`
    /// and the accounting hooks are no-ops, so the legacy unbounded
    /// behaviour is preserved exactly.
    pub fn new(enabled: bool, cfg: GovernorConfig) -> Self {
        let floor = cfg.precision_floor.clamp(0.0, 1.0);
        let g = Self {
            enabled,
            precision_alpha: cfg.precision_alpha.clamp(1e-3, 1.0),
            precision_floor: floor,
            contention_weight: cfg.contention_weight.max(0.0),
            base_threshold: cfg.base_threshold.max(0.0),
            precision_ewma: AtomicU64::new(0),
            foreground_inflight: AtomicI64::new(0),
            window_completed: AtomicU64::new(0),
            window_used: AtomicU64::new(0),
            throttled: AtomicU64::new(0),
            admitted: AtomicU64::new(0),
            decision_diagnostics: Mutex::new(GovernorDecisionDiagnostics::default()),
        };
        // Seed the EWMA optimistically at `max(floor, 0.5)` so a freshly
        // started engine gives speculation a fair chance to prove itself
        // before the measured signal takes over.
        store_f64(&g.precision_ewma, floor.max(0.5));
        g
    }

    /// A disabled pass-through governor (no gating, no accounting).
    pub fn disabled() -> Self {
        Self::new(false, GovernorConfig::default())
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Decide whether a speculative read for an expert predicted with
    /// probability/score `prob` should be admitted *right now*.
    ///
    /// Returns `true` unconditionally when the governor is disabled.
    /// Otherwise admits iff
    ///
    /// ```text
    ///   prob * max(precision_ewma, floor)
    ///       >= base_threshold * (1 + contention_weight * foreground_inflight)
    /// ```
    ///
    /// i.e. the expected value of the speculation (its probability scaled
    /// by how often speculation has recently paid off) must clear a bar
    /// that rises with the number of foreground misses currently
    /// competing for the device.
    #[inline]
    pub fn admit(&self, prob: f64) -> bool {
        if !self.enabled {
            return true;
        }
        let decision = self.evaluate(prob);
        let mut diagnostics = self
            .decision_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if decision.admitted {
            self.admitted.fetch_add(1, Ordering::Relaxed);
        } else {
            self.throttled.fetch_add(1, Ordering::Relaxed);
        }
        diagnostics.record(decision);
        decision.admitted
    }

    #[inline]
    fn evaluate(&self, prob: f64) -> GovernorDecision {
        let precision = load_f64(&self.precision_ewma).max(self.precision_floor);
        let foreground_inflight = self.foreground_inflight.load(Ordering::Relaxed).max(0) as u64;
        let candidate_probability = prob.max(0.0);
        let candidate_score = candidate_probability * precision;
        let effective_threshold =
            self.base_threshold * (1.0 + self.contention_weight * foreground_inflight as f64);
        let admitted = candidate_score >= effective_threshold;
        let score_minus_threshold_margin = candidate_score - effective_threshold;
        let score_to_threshold_ratio = if effective_threshold > 0.0 {
            let ratio = candidate_score / effective_threshold;
            ratio.is_finite().then_some(ratio)
        } else {
            None
        };
        GovernorDecision {
            candidate_probability,
            effective_precision: precision,
            candidate_score,
            base_threshold: self.base_threshold,
            foreground_inflight,
            contention_weight: self.contention_weight,
            effective_threshold,
            admitted,
            score_minus_threshold_margin,
            score_to_threshold_ratio,
        }
    }

    /// RAII-free gauge bump: a foreground (blocking) read has started.
    /// Pair with [`Self::end_foreground`]. No-op when disabled.
    #[inline]
    pub fn begin_foreground(&self) {
        if self.enabled {
            self.foreground_inflight.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A foreground read has finished. No-op when disabled.
    #[inline]
    pub fn end_foreground(&self) {
        if self.enabled {
            self.foreground_inflight.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Begin foreground-read accounting and return a guard that ends it
    /// on drop. Preserves the disabled-governor no-op semantics of the
    /// explicit begin/end methods.
    #[inline]
    pub fn foreground_guard(&self) -> ForegroundGuard<'_> {
        ForegroundGuard::new(self)
    }

    /// Record that a speculative read landed (became resident).
    #[inline]
    pub fn record_completed(&self) {
        if self.enabled {
            self.window_completed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record that a previously-prefetched expert was consumed by a hit
    /// before it was evicted — i.e. the speculation paid off.
    #[inline]
    pub fn record_used(&self) {
        if self.enabled {
            self.window_used.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Fold the current measurement window into the precision EWMA and
    /// reset the window. Cheap enough to call once per token. When no
    /// prefetches completed in the window the EWMA is left untouched
    /// (no signal → no update), which keeps the controller stable during
    /// cache-resident bursts.
    pub fn refresh(&self) {
        if !self.enabled {
            return;
        }
        let completed = self.window_completed.swap(0, Ordering::Relaxed);
        if completed == 0 {
            // No completions ⇒ don't drag the EWMA toward 0 for an idle
            // window; just clear any stray `used` credits.
            self.window_used.store(0, Ordering::Relaxed);
            return;
        }
        let used = self.window_used.swap(0, Ordering::Relaxed).min(completed);
        let sample = used as f64 / completed as f64;
        let prev = load_f64(&self.precision_ewma);
        let next = prev + self.precision_alpha * (sample - prev);
        store_f64(&self.precision_ewma, next.clamp(0.0, 1.0));
    }

    /// Current precision EWMA (for telemetry / the run summary).
    pub fn precision(&self) -> f64 {
        load_f64(&self.precision_ewma)
    }

    /// Current foreground-read gauge (for telemetry).
    pub fn foreground_inflight(&self) -> i64 {
        self.foreground_inflight.load(Ordering::Relaxed)
    }

    /// `(admitted, throttled)` admission decisions so far.
    pub fn decisions(&self) -> (u64, u64) {
        (
            self.admitted.load(Ordering::Relaxed),
            self.throttled.load(Ordering::Relaxed),
        )
    }

    /// Snapshot bounded governor score diagnostics since the most recent
    /// reset. Disabled governors return zero decisions and null distributions.
    pub fn decision_diagnostics(&self) -> GovernorDecisionDiagnosticsSnapshot {
        let diagnostics = self
            .decision_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        GovernorDecisionDiagnosticsSnapshot::from_state(
            self.enabled,
            self.base_threshold,
            self.contention_weight,
            &diagnostics,
        )
    }

    /// Atomically observe direct counters and their matching bounded
    /// diagnostic window. Admission updates both while holding the same lock,
    /// so the two views cannot straddle a decision.
    pub fn decisions_and_diagnostics(&self) -> ((u64, u64), GovernorDecisionDiagnosticsSnapshot) {
        let diagnostics = self
            .decision_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let decisions = (
            self.admitted.load(Ordering::Relaxed),
            self.throttled.load(Ordering::Relaxed),
        );
        let snapshot = GovernorDecisionDiagnosticsSnapshot::from_state(
            self.enabled,
            self.base_threshold,
            self.contention_weight,
            &diagnostics,
        );
        (decisions, snapshot)
    }

    /// Start a fresh bounded diagnostic window without changing governor
    /// precision, direct counters, foreground accounting, or admission policy.
    /// The returned direct-counter boundary is sampled under the same lock as
    /// the reset so subsequent deltas reconcile exactly with the new window.
    pub fn reset_decision_diagnostics(&self) -> (u64, u64) {
        let mut diagnostics = self
            .decision_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *diagnostics = GovernorDecisionDiagnostics::default();
        (
            self.admitted.load(Ordering::Relaxed),
            self.throttled.load(Ordering::Relaxed),
        )
    }
}

impl Default for PrefetchGovernor {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn disabled_governor_admits_everything() {
        let g = PrefetchGovernor::disabled();
        assert!(g.admit(0.0));
        assert!(g.admit(1.0));
        // Accounting hooks are inert.
        g.begin_foreground();
        g.record_completed();
        g.refresh();
        assert!(g.admit(0.0));
    }

    #[test]
    fn high_precision_idle_disk_admits() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        // Seeded optimistic; a confident prediction on an idle disk is
        // admitted.
        assert!(g.admit(0.9));
    }

    #[test]
    fn collapsed_precision_throttles_even_when_idle() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        // Many windows of completed-but-never-used reads drive the
        // precision EWMA down to its floor.
        for _ in 0..40 {
            for _ in 0..10 {
                g.record_completed();
            }
            g.refresh();
        }
        // At the floor (0.05) a moderate 0.3-probability prediction's
        // expected value (0.015) no longer clears even the idle bar
        // (base_threshold 0.02), so it is declined.
        assert!(!g.admit(0.3));
        // A near-certain prediction (0.9 * 0.05 = 0.045) still gets
        // through — the governor throttles junk, not signal.
        assert!(g.admit(0.9));
    }

    #[test]
    fn foreground_contention_raises_the_bar() {
        let cfg = GovernorConfig {
            // Isolate the contention term from the precision term.
            precision_floor: 1.0,
            base_threshold: 0.2,
            contention_weight: 1.0,
            ..GovernorConfig::default()
        };
        let g = PrefetchGovernor::new(true, cfg);
        store_f64(&g.precision_ewma, 1.0);
        // Idle disk: prob 0.3 clears the 0.2 bar.
        assert!(g.admit(0.3));
        // Two foreground misses in flight ⇒ bar = 0.2 * 3 = 0.6.
        g.begin_foreground();
        g.begin_foreground();
        assert!(!g.admit(0.3));
        // A near-certain prediction still gets through.
        assert!(g.admit(0.9));
        g.end_foreground();
        g.end_foreground();
        assert!(g.admit(0.3));
    }

    #[test]
    fn foreground_guard_releases_on_drop() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        let before = g.foreground_inflight();
        {
            let _guard = g.foreground_guard();
            assert_eq!(g.foreground_inflight(), before + 1);
        }
        assert_eq!(g.foreground_inflight(), before);

        let disabled = PrefetchGovernor::disabled();
        {
            let _guard = disabled.foreground_guard();
            assert_eq!(disabled.foreground_inflight(), 0);
        }
        assert_eq!(disabled.foreground_inflight(), 0);
    }

    #[test]
    fn precision_recovers_when_prefetches_get_used() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        for _ in 0..100 {
            g.record_completed();
        }
        g.refresh(); // precision drops (0 used / 100 completed)
        let low = g.precision();
        // Now a window where every completion is consumed.
        for _ in 0..100 {
            g.record_completed();
            g.record_used();
        }
        g.refresh();
        assert!(g.precision() > low);
    }

    #[test]
    fn used_is_clamped_to_completed() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        g.record_completed();
        g.record_used();
        g.record_used();
        g.record_used();
        // Should not panic or exceed 1.0.
        g.refresh();
        assert!(g.precision() <= 1.0);
    }

    #[test]
    fn direct_decision_counters_reconcile() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        assert!(g.admit(1.0));
        assert!(!g.admit(0.0));
        assert!(g.admit(1.0));
        let (admitted, rejected) = g.decisions();
        let total = admitted + rejected;
        assert_eq!(admitted, 2);
        assert_eq!(rejected, 1);
        assert_eq!(total, 3);
    }

    #[test]
    fn governor_score_diagnostics_use_exact_admission_values() {
        let cfg = GovernorConfig {
            precision_floor: 0.4,
            contention_weight: 0.5,
            base_threshold: 0.2,
            ..GovernorConfig::default()
        };
        let g = PrefetchGovernor::new(true, cfg);
        store_f64(&g.precision_ewma, 0.4);
        g.begin_foreground();
        g.begin_foreground();

        let decision = g.evaluate(0.75);
        assert_close(decision.candidate_probability, 0.75);
        assert_close(decision.effective_precision, 0.4);
        assert_close(decision.candidate_score, 0.3);
        assert_close(decision.base_threshold, 0.2);
        assert_eq!(decision.foreground_inflight, 2);
        assert_close(decision.contention_weight, 0.5);
        assert_close(decision.effective_threshold, 0.4);
        assert!(!decision.admitted);
        assert_close(decision.score_minus_threshold_margin, -0.1);
        assert_close(decision.score_to_threshold_ratio.unwrap(), 0.75);

        assert!(!g.admit(0.75));
        let diagnostics = g.decision_diagnostics();
        assert_eq!(diagnostics.total_decisions, 1);
        assert_eq!(diagnostics.rejected, 1);
        assert_close(diagnostics.candidate_score.mean.unwrap(), 0.3);
        assert_close(diagnostics.effective_threshold.mean.unwrap(), 0.4);
        assert_close(diagnostics.score_to_threshold_ratio.mean.unwrap(), 0.75);
        assert_close(
            diagnostics
                .decision_boundary
                .closest_rejected_score_minus_threshold_margin
                .unwrap(),
            -0.1,
        );
        assert_eq!(
            diagnostics
                .foreground_inflight_decisions
                .foreground_inflight_positive_rejected,
            1
        );
    }

    #[test]
    fn governor_score_diagnostics_cover_admission_and_foreground_groups() {
        let cfg = GovernorConfig {
            precision_floor: 1.0,
            base_threshold: 0.2,
            contention_weight: 1.0,
            ..GovernorConfig::default()
        };
        let g = PrefetchGovernor::new(true, cfg);
        store_f64(&g.precision_ewma, 1.0);
        assert!(g.admit(0.3));
        g.begin_foreground();
        assert!(!g.admit(0.3));
        let diagnostics = g.decision_diagnostics();
        assert_eq!(diagnostics.total_decisions, 2);
        assert_eq!(diagnostics.admitted, 1);
        assert_eq!(diagnostics.rejected, 1);
        assert_eq!(diagnostics.candidate_probability.count, 2);
        assert_close(diagnostics.effective_threshold.minimum.unwrap(), 0.2);
        assert_close(diagnostics.effective_threshold.maximum.unwrap(), 0.4);
        assert_eq!(
            diagnostics
                .foreground_inflight_decisions
                .foreground_inflight_zero_admitted,
            1
        );
        assert_eq!(
            diagnostics
                .foreground_inflight_decisions
                .foreground_inflight_positive_rejected,
            1
        );
        assert_close(
            diagnostics
                .decision_boundary
                .minimum_admitted_candidate_score
                .unwrap(),
            0.3,
        );
        assert_close(
            diagnostics
                .decision_boundary
                .maximum_rejected_score_to_threshold_ratio
                .unwrap(),
            0.75,
        );
    }

    #[test]
    fn governor_score_empty_and_zero_threshold_semantics_are_explicit() {
        let disabled = PrefetchGovernor::disabled().decision_diagnostics();
        assert!(!disabled.enabled);
        assert_eq!(disabled.total_decisions, 0);
        assert!(disabled.candidate_score.mean.is_none());
        assert!(disabled.score_to_threshold_ratio.p99.is_none());
        assert!(disabled
            .decision_boundary
            .minimum_admitted_candidate_score
            .is_none());

        let g = PrefetchGovernor::new(
            true,
            GovernorConfig {
                base_threshold: 0.0,
                ..GovernorConfig::default()
            },
        );
        assert!(g.admit(0.0));
        let diagnostics = g.decision_diagnostics();
        assert_eq!(diagnostics.total_decisions, 1);
        assert_eq!(diagnostics.ratio_undefined_decisions, 1);
        assert_eq!(diagnostics.score_to_threshold_ratio.count, 0);
        assert!(diagnostics.score_to_threshold_ratio.mean.is_none());
    }

    #[test]
    fn governor_score_percentiles_are_nearest_rank_and_memory_is_bounded() {
        let g = PrefetchGovernor::new(
            true,
            GovernorConfig {
                precision_floor: 1.0,
                base_threshold: 0.1,
                ..GovernorConfig::default()
            },
        );
        store_f64(&g.precision_ewma, 1.0);
        for value in 1..=10 {
            g.admit(value as f64 / 10.0);
        }
        let percentiles = g.decision_diagnostics();
        assert_close(percentiles.candidate_probability.p50.unwrap(), 0.5);
        assert_close(percentiles.candidate_probability.p90.unwrap(), 0.9);
        assert_close(percentiles.candidate_probability.p95.unwrap(), 1.0);
        assert_close(percentiles.candidate_probability.p99.unwrap(), 1.0);

        g.reset_decision_diagnostics();
        for index in 0..(GOVERNOR_SCORE_SAMPLE_CAPACITY + 73) {
            g.admit((index % 101) as f64 / 100.0);
        }
        let bounded = g.decision_diagnostics();
        assert_eq!(
            bounded.total_decisions,
            (GOVERNOR_SCORE_SAMPLE_CAPACITY + 73) as u64
        );
        assert_eq!(bounded.sample_capacity, GOVERNOR_SCORE_SAMPLE_CAPACITY);
        assert_eq!(bounded.sampled_decisions, GOVERNOR_SCORE_SAMPLE_CAPACITY);
        assert_eq!(bounded.samples.len(), GOVERNOR_SCORE_SAMPLE_CAPACITY);
    }

    #[test]
    fn governor_score_outputs_are_finite_and_counters_reconcile() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        for probability in [0.0, 0.01, 0.1, 0.5, 1.0] {
            g.admit(probability);
        }
        let diagnostics = g.decision_diagnostics();
        let (admitted, rejected) = g.decisions();
        assert_eq!(diagnostics.total_decisions, admitted + rejected);
        assert_eq!(diagnostics.admitted, admitted);
        assert_eq!(diagnostics.rejected, rejected);
        assert_eq!(diagnostics.invalid_numeric_decisions, 0);
        for distribution in [
            &diagnostics.candidate_probability,
            &diagnostics.effective_precision,
            &diagnostics.candidate_score,
            &diagnostics.effective_threshold,
            &diagnostics.score_to_threshold_ratio,
        ] {
            for value in [
                distribution.minimum,
                distribution.maximum,
                distribution.mean,
                distribution.p50,
                distribution.p90,
                distribution.p95,
                distribution.p99,
            ]
            .into_iter()
            .flatten()
            {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn governor_score_reset_returns_exact_direct_counter_boundary() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        assert!(g.admit(1.0));
        let boundary = g.reset_decision_diagnostics();
        assert_eq!(boundary, (1, 0));

        assert!(!g.admit(0.0));
        let (direct, diagnostics) = g.decisions_and_diagnostics();
        assert_eq!(direct, (1, 1));
        assert_eq!(
            diagnostics.total_decisions,
            direct.0.saturating_sub(boundary.0) + direct.1.saturating_sub(boundary.1)
        );
        assert_eq!(diagnostics.admitted, 0);
        assert_eq!(diagnostics.rejected, 1);
    }

    #[test]
    fn governor_score_measured_run_snapshots_merge_with_exact_counts() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        assert!(g.admit(1.0));
        assert!(!g.admit(0.0));
        let first = g.decision_diagnostics();
        g.reset_decision_diagnostics();
        assert!(g.admit(0.5));
        let second = g.decision_diagnostics();
        let merged = GovernorDecisionDiagnosticsSnapshot::merge(&[first, second]);
        assert_eq!(merged.total_decisions, 3);
        assert_eq!(merged.admitted, 2);
        assert_eq!(merged.rejected, 1);
        assert_eq!(merged.sampled_decisions, 3);
        assert_eq!(merged.candidate_score.count, 3);
        assert_eq!(
            merged
                .foreground_inflight_decisions
                .foreground_inflight_zero,
            3
        );
        assert!(merged.candidate_score.mean.unwrap().is_finite());
    }

    #[test]
    fn governor_score_decision_matches_pre_diagnostic_formula() {
        let cfg = GovernorConfig {
            precision_floor: 0.25,
            base_threshold: 0.05,
            contention_weight: 0.75,
            ..GovernorConfig::default()
        };
        let g = PrefetchGovernor::new(true, cfg);
        store_f64(&g.precision_ewma, 0.3);
        for inflight in 0..=3 {
            while g.foreground_inflight() < inflight {
                g.begin_foreground();
            }
            for probability in [-0.5_f64, 0.0, 0.1, 0.5, 1.0] {
                let precision = load_f64(&g.precision_ewma).max(g.precision_floor);
                let foreground = g.foreground_inflight().max(0) as f64;
                let legacy = probability.max(0.0) * precision
                    >= g.base_threshold * (1.0 + g.contention_weight * foreground);
                assert_eq!(g.evaluate(probability).admitted, legacy);
            }
        }
    }
}
