use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const EMBEDDING: &str = "embedding";
pub const RMS_NORM: &str = "rms_norm";
pub const Q_PROJECTION: &str = "q_projection";
pub const K_PROJECTION: &str = "k_projection";
pub const V_PROJECTION: &str = "v_projection";
pub const ROPE: &str = "rope";
pub const ATTENTION_SCORE_VALUE: &str = "attention_score_value";
pub const O_PROJECTION: &str = "o_projection";
pub const ROUTER_GATE: &str = "router_gate";
pub const EXPERT_CACHE_LOOKUP: &str = "expert_cache_lookup";
pub const FOREGROUND_EXPERT_IO_WAIT: &str = "foreground_expert_io_wait";
/// Exclusive wall time spent awaiting foreground expert-fetch join handles.
/// Unlike `FOREGROUND_EXPERT_IO_WAIT`, this starts after cache lookup/task
/// submission and is safe to add to the critical-path attribution.
pub const FOREGROUND_EXPERT_IO_AWAIT: &str = "foreground_expert_io_await";
#[allow(dead_code)]
pub const EXPERT_PREPARATION: &str = "expert_preparation";
pub const Q8_PREPARATION: &str = "q8_preparation";
pub const Q8_GATE_UP_KERNEL: &str = "q8_gate_up_kernel";
pub const Q8_DOWN_KERNEL: &str = "q8_down_kernel";
pub const EXPERT_COMPUTE: &str = "expert_compute";
pub const MOE_WEIGHTED_COMBINATION: &str = "moe_weighted_combination";
pub const FINAL_RMS_NORM: &str = "final_rms_norm";
pub const LM_HEAD: &str = "lm_head";
pub const SAMPLING: &str = "sampling";
pub const SCHEDULER_OVERHEAD: &str = "scheduler_overhead";
pub const TOTAL_PROMPT: &str = "total_prompt";
pub const TOTAL_DECODE: &str = "total_decode";

#[derive(Clone, Debug, Serialize)]
pub struct StageTimingSnapshot {
    pub count: u64,
    pub total_seconds: f64,
    pub mean_seconds: f64,
    pub max_seconds: f64,
}

/// Non-overlapping wall-clock categories for one benchmark phase. These are
/// derived only from leaf/enclosing-wall timers that are sequential on the
/// request path. Nested Q8 kernel diagnostics are intentionally excluded.
#[derive(Clone, Debug, Serialize)]
pub struct CriticalPathCategories {
    pub embedding_input_preparation_seconds: f64,
    pub normalization_seconds: f64,
    pub qkv_projection_seconds: f64,
    pub rope_attention_seconds: f64,
    pub attention_output_projection_seconds: f64,
    pub routing_seconds: f64,
    pub expert_cache_coordination_seconds: f64,
    pub foreground_expert_io_wait_seconds: f64,
    pub expert_compute_seconds: f64,
    pub expert_weighted_combination_seconds: f64,
    pub lm_head_evaluation_seconds: f64,
    pub sampling_seconds: f64,
    pub scheduler_runtime_overhead_seconds: f64,
}

impl CriticalPathCategories {
    pub fn attributed_seconds(&self) -> f64 {
        self.embedding_input_preparation_seconds
            + self.normalization_seconds
            + self.qkv_projection_seconds
            + self.rope_attention_seconds
            + self.attention_output_projection_seconds
            + self.routing_seconds
            + self.expert_cache_coordination_seconds
            + self.foreground_expert_io_wait_seconds
            + self.expert_compute_seconds
            + self.expert_weighted_combination_seconds
            + self.lm_head_evaluation_seconds
            + self.sampling_seconds
            + self.scheduler_runtime_overhead_seconds
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CriticalPathReport {
    pub wall_seconds: f64,
    pub categories: CriticalPathCategories,
    pub attributed_seconds: f64,
    /// Signed residual (`wall - attributed`). It is deliberately not clamped:
    /// a negative value exposes overlap instead of hiding it.
    pub unattributed_residual_seconds: f64,
    pub coverage_ratio: f64,
    pub non_overlap_invariant_passed: bool,
    pub coverage_95_percent_passed: bool,
    pub qualification_passed: bool,
}

impl CriticalPathReport {
    pub fn from_snapshot(
        wall_seconds: f64,
        stages: &BTreeMap<String, StageTimingSnapshot>,
    ) -> Self {
        let stage = |name: &str| {
            stages
                .get(name)
                .map(|timing| timing.total_seconds)
                .unwrap_or(0.0)
        };
        let categories = CriticalPathCategories {
            embedding_input_preparation_seconds: stage(EMBEDDING),
            normalization_seconds: stage(RMS_NORM) + stage(FINAL_RMS_NORM),
            qkv_projection_seconds: stage(Q_PROJECTION) + stage(K_PROJECTION) + stage(V_PROJECTION),
            rope_attention_seconds: stage(ROPE) + stage(ATTENTION_SCORE_VALUE),
            attention_output_projection_seconds: stage(O_PROJECTION),
            routing_seconds: stage(ROUTER_GATE),
            expert_cache_coordination_seconds: stage(EXPERT_CACHE_LOOKUP),
            foreground_expert_io_wait_seconds: stage(FOREGROUND_EXPERT_IO_AWAIT),
            expert_compute_seconds: stage(EXPERT_COMPUTE),
            expert_weighted_combination_seconds: stage(MOE_WEIGHTED_COMBINATION),
            lm_head_evaluation_seconds: stage(LM_HEAD),
            sampling_seconds: stage(SAMPLING),
            scheduler_runtime_overhead_seconds: stage(SCHEDULER_OVERHEAD),
        };
        let attributed_seconds = categories.attributed_seconds();
        let unattributed_residual_seconds = wall_seconds - attributed_seconds;
        let tolerance = (wall_seconds.abs() * 1e-6).max(1e-9);
        let non_overlap_invariant_passed = unattributed_residual_seconds >= -tolerance;
        let coverage_ratio = if wall_seconds > 0.0 {
            attributed_seconds / wall_seconds
        } else if attributed_seconds == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
        let coverage_95_percent_passed = coverage_ratio >= 0.95;
        let qualification_passed = non_overlap_invariant_passed && coverage_95_percent_passed;
        Self {
            wall_seconds,
            categories,
            attributed_seconds,
            unattributed_residual_seconds,
            coverage_ratio,
            non_overlap_invariant_passed,
            coverage_95_percent_passed,
            qualification_passed,
        }
    }
}

pub fn merge_snapshots(
    snapshots: &[&BTreeMap<String, StageTimingSnapshot>],
) -> BTreeMap<String, StageTimingSnapshot> {
    let mut merged: BTreeMap<String, StageTimingSnapshot> = BTreeMap::new();
    for snapshot in snapshots {
        for (name, timing) in snapshot.iter() {
            let entry = merged.entry(name.clone()).or_insert(StageTimingSnapshot {
                count: 0,
                total_seconds: 0.0,
                mean_seconds: 0.0,
                max_seconds: 0.0,
            });
            entry.count += timing.count;
            entry.total_seconds += timing.total_seconds;
            entry.max_seconds = entry.max_seconds.max(timing.max_seconds);
        }
    }
    for timing in merged.values_mut() {
        timing.mean_seconds = if timing.count == 0 {
            0.0
        } else {
            timing.total_seconds / timing.count as f64
        };
    }
    merged
}

#[derive(Clone, Copy, Debug, Default)]
struct StageTimingAccumulator {
    count: u64,
    total: Duration,
    max: Duration,
}

#[derive(Debug, Default)]
pub struct StageTimings {
    inner: Mutex<BTreeMap<&'static str, StageTimingAccumulator>>,
}

impl StageTimings {
    pub fn record(&self, stage: &'static str, duration: Duration) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        let acc = inner.entry(stage).or_default();
        acc.count += 1;
        acc.total += duration;
        acc.max = acc.max.max(duration);
    }

    pub fn time<T>(&self, stage: &'static str, f: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let out = f();
        self.record(stage, started.elapsed());
        out
    }

    pub fn snapshot(&self) -> BTreeMap<String, StageTimingSnapshot> {
        let inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner
            .iter()
            .map(|(&stage, acc)| {
                let total_seconds = acc.total.as_secs_f64();
                let mean_seconds = if acc.count == 0 {
                    0.0
                } else {
                    total_seconds / acc.count as f64
                };
                (
                    stage.to_string(),
                    StageTimingSnapshot {
                        count: acc.count,
                        total_seconds,
                        mean_seconds,
                        max_seconds: acc.max.as_secs_f64(),
                    },
                )
            })
            .collect()
    }
}

pub fn record_optional(timings: Option<&StageTimings>, stage: &'static str, duration: Duration) {
    if let Some(timings) = timings {
        timings.record(stage, duration);
    }
}

pub fn time_optional<T>(
    timings: Option<&StageTimings>,
    stage: &'static str,
    f: impl FnOnce() -> T,
) -> T {
    match timings {
        Some(timings) => timings.time(stage, f),
        None => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_timings_accumulate_count_total_mean_and_max() {
        let timings = StageTimings::default();
        timings.record(EMBEDDING, Duration::from_millis(2));
        timings.record(EMBEDDING, Duration::from_millis(4));

        let snapshot = timings.snapshot();
        let embedding = snapshot.get(EMBEDDING).expect("embedding timing");
        assert_eq!(embedding.count, 2);
        assert!((embedding.total_seconds - 0.006).abs() < f64::EPSILON);
        assert!((embedding.mean_seconds - 0.003).abs() < f64::EPSILON);
        assert!((embedding.max_seconds - 0.004).abs() < f64::EPSILON);
    }

    #[test]
    fn time_optional_skips_when_absent_and_records_when_present() {
        let value = time_optional::<u32>(None, LM_HEAD, || 7);
        assert_eq!(value, 7);

        let timings = StageTimings::default();
        let value = time_optional(Some(&timings), LM_HEAD, || 11);
        assert_eq!(value, 11);
        assert_eq!(timings.snapshot().get(LM_HEAD).unwrap().count, 1);
    }

    fn snapshot(entries: &[(&str, f64)]) -> BTreeMap<String, StageTimingSnapshot> {
        entries
            .iter()
            .map(|(name, seconds)| {
                (
                    (*name).to_string(),
                    StageTimingSnapshot {
                        count: 1,
                        total_seconds: *seconds,
                        mean_seconds: *seconds,
                        max_seconds: *seconds,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn critical_path_reports_residual_and_coverage_without_clamping() {
        let stages = snapshot(&[(EMBEDDING, 0.2), (EXPERT_COMPUTE, 0.75)]);
        let report = CriticalPathReport::from_snapshot(1.0, &stages);
        assert!((report.attributed_seconds - 0.95).abs() < 1e-12);
        assert!((report.unattributed_residual_seconds - 0.05).abs() < 1e-12);
        assert!((report.coverage_ratio - 0.95).abs() < 1e-12);
        assert!(report.non_overlap_invariant_passed);
        assert!(report.coverage_95_percent_passed);
        assert!(report.qualification_passed);
    }

    #[test]
    fn critical_path_exposes_overlap_as_negative_residual() {
        let stages = snapshot(&[(EMBEDDING, 0.6), (EXPERT_COMPUTE, 0.5)]);
        let report = CriticalPathReport::from_snapshot(1.0, &stages);
        assert!((report.unattributed_residual_seconds + 0.1).abs() < 1e-12);
        assert!(!report.non_overlap_invariant_passed);
        assert!(!report.qualification_passed);
    }

    #[test]
    fn critical_path_ignores_nested_q8_diagnostics() {
        let stages = snapshot(&[
            (EXPERT_COMPUTE, 0.8),
            (Q8_GATE_UP_KERNEL, 0.7),
            (Q8_DOWN_KERNEL, 0.6),
        ]);
        let report = CriticalPathReport::from_snapshot(1.0, &stages);
        assert!((report.attributed_seconds - 0.8).abs() < 1e-12);
        assert!(report.non_overlap_invariant_passed);
    }

    #[test]
    fn critical_path_serializes_signed_residual_and_qualification() {
        let report = CriticalPathReport::from_snapshot(
            1.0,
            &snapshot(&[(EMBEDDING, 0.25), (EXPERT_COMPUTE, 0.70)]),
        );
        let value = serde_json::to_value(report).unwrap();
        assert!((value["coverage_ratio"].as_f64().unwrap() - 0.95).abs() < 1e-12);
        assert!((value["unattributed_residual_seconds"].as_f64().unwrap() - 0.05).abs() < 1e-12);
        assert_eq!(value["qualification_passed"], true);
        assert!(value["categories"]["expert_compute_seconds"].is_number());
    }

    #[test]
    fn merge_snapshots_recomputes_mean_and_max() {
        let a = snapshot(&[(LM_HEAD, 0.2)]);
        let b = snapshot(&[(LM_HEAD, 0.4)]);
        let merged = merge_snapshots(&[&a, &b]);
        let lm = merged.get(LM_HEAD).unwrap();
        assert_eq!(lm.count, 2);
        assert!((lm.total_seconds - 0.6).abs() < 1e-12);
        assert!((lm.mean_seconds - 0.3).abs() < 1e-12);
        assert!((lm.max_seconds - 0.4).abs() < 1e-12);
    }
}
