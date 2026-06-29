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
pub const EXPERT_COMPUTE: &str = "expert_compute";
pub const MOE_WEIGHTED_COMBINATION: &str = "moe_weighted_combination";
pub const FINAL_RMS_NORM: &str = "final_rms_norm";
pub const LM_HEAD: &str = "lm_head";
pub const SAMPLING: &str = "sampling";
pub const TOTAL_PROMPT: &str = "total_prompt";
pub const TOTAL_DECODE: &str = "total_decode";

#[derive(Clone, Debug, Serialize)]
pub struct StageTimingSnapshot {
    pub count: u64,
    pub total_seconds: f64,
    pub mean_seconds: f64,
    pub max_seconds: f64,
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
}
