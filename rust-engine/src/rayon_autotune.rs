use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::numa::{format_cpulist, EffectiveCpuAffinity};

pub const AUTOTUNE_PROBE_ENV: &str = "MER_RAYON_AUTOTUNE_PROBE";
pub const DEFAULT_AUTOTUNE_TOKENS: u64 = 2_000;
pub const DEFAULT_AUTOTUNE_COARSE_TOKENS: u64 = 512;
pub const DEFAULT_AUTOTUNE_REPEATS: usize = 2;
pub const DEFAULT_AUTOTUNE_TOP_CANDIDATES: usize = 3;
pub const DEFAULT_SLOW_P95_THRESHOLD_MS: f64 = 80.0;
pub const DEFAULT_SLOW_P99_THRESHOLD_MS: f64 = 120.0;

const LOW_VARIANCE_CV: f64 = 0.15;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuAutotuneKey {
    pub machine_fingerprint: String,
    pub model_fingerprint: String,
    pub backend_fingerprint: String,
}

impl CpuAutotuneKey {
    pub fn cache_key(&self) -> String {
        format!(
            "{}|{}|{}",
            self.machine_fingerprint, self.model_fingerprint, self.backend_fingerprint
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuAutotuneMachine {
    pub os: String,
    pub arch: String,
    pub cpu_vendor: String,
    pub cpu_model: String,
    pub cpu_features: Vec<&'static str>,
    pub available_logical_cores: usize,
    pub effective_cpu_mask: Option<Vec<usize>>,
}

impl CpuAutotuneMachine {
    pub fn from_effective_affinity(affinity: &EffectiveCpuAffinity) -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_vendor: crate::kernels::cpu_features().vendor.clone(),
            cpu_model: crate::kernels::cpu_features().model.clone(),
            cpu_features: detected_cpu_features(),
            available_logical_cores: affinity.logical_cores,
            effective_cpu_mask: affinity.cpus.clone(),
        }
    }

    pub fn fingerprint(&self) -> String {
        let mask = self
            .effective_cpu_mask
            .as_deref()
            .map(format_cpulist)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unavailable".to_string());
        let features = if self.cpu_features.is_empty() {
            "none".to_string()
        } else {
            self.cpu_features.join("+")
        };
        format!(
            "version=2;os={};arch={};cpu_vendor={};cpu_model={};features={};logical={};affinity={}",
            self.os,
            self.arch,
            self.cpu_vendor,
            self.cpu_model,
            features,
            self.available_logical_cores,
            mask
        )
    }
}

pub fn machine_fingerprint_from_affinity(affinity: &EffectiveCpuAffinity) -> String {
    CpuAutotuneMachine::from_effective_affinity(affinity).fingerprint()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detected_cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if std::is_x86_feature_detected!("avx2") {
        features.push("avx2");
    }
    if std::is_x86_feature_detected!("avx512f") {
        features.push("avx512f");
    }
    if std::is_x86_feature_detected!("fma") {
        features.push("fma");
    }
    #[cfg(feature = "nightly-amx")]
    if std::is_x86_feature_detected!("amx-tile") {
        features.push("amx-tile");
    }
    features
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detected_cpu_features() -> Vec<&'static str> {
    Vec::new()
}

pub fn default_thread_candidates(effective_logical_cores: usize) -> Vec<usize> {
    let n = effective_logical_cores.max(1);
    let mut candidates = BTreeSet::new();
    candidates.insert(1);
    candidates.insert(n);
    candidates.insert(crate::parallel::default_compute_threads(n));

    for divisor in [2usize, 3, 4] {
        let lower = (n / divisor).max(1);
        let upper = n.div_ceil(divisor).max(1);
        candidates.insert(lower);
        candidates.insert(upper);
    }

    for delta in [1usize, 2, 4, 8] {
        if n > delta {
            candidates.insert(n - delta);
        }
    }

    let default = crate::parallel::default_compute_threads(n);
    for delta in [1usize, 2] {
        if default > delta {
            candidates.insert(default - delta);
        }
        let plus = default.saturating_add(delta);
        if plus <= n {
            candidates.insert(plus);
        }
    }

    candidates.into_iter().filter(|&c| c <= n).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RayonAutotuneProbeResult {
    pub threads: usize,
    pub valid: bool,
    pub p50_ms: f64,
    pub p95_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<f64>,
    pub sustained_tps: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RayonAutotuneProbeStage {
    Coarse,
    Fine,
}

impl RayonAutotuneProbeStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coarse => "coarse",
            Self::Fine => "fine",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RayonAutotuneProbeObservation {
    pub threads: usize,
    pub stage: RayonAutotuneProbeStage,
    /// One-based repeat index within this stage.
    pub repeat: usize,
    pub tokens: u64,
    pub valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sustained_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl RayonAutotuneProbeObservation {
    pub fn from_probe_result(
        stage: RayonAutotuneProbeStage,
        repeat: usize,
        tokens: u64,
        result: RayonAutotuneProbeResult,
    ) -> Self {
        Self {
            threads: result.threads,
            stage,
            repeat,
            tokens,
            valid: result.valid,
            p50_ms: Some(result.p50_ms),
            p95_ms: Some(result.p95_ms),
            p99_ms: result.p99_ms,
            sustained_tps: Some(result.sustained_tps),
            reason: (!result.valid).then(|| "probe reported invalid".to_string()),
        }
    }

    pub fn invalid(
        threads: usize,
        stage: RayonAutotuneProbeStage,
        repeat: usize,
        tokens: u64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            threads,
            stage,
            repeat,
            tokens,
            valid: false,
            p50_ms: None,
            p95_ms: None,
            p99_ms: None,
            sustained_tps: None,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RayonAutotuneConfidence {
    High,
    Medium,
    Low,
}

impl Default for RayonAutotuneConfidence {
    fn default() -> Self {
        Self::Low
    }
}

impl RayonAutotuneConfidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RayonAutotuneCandidateSummary {
    pub threads: usize,
    pub requested_repeats: usize,
    pub successful_repeats: usize,
    pub all_repeats_successful: bool,
    pub p95_below_slow_threshold: bool,
    pub p99_below_slow_threshold: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_p50_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_p95_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_p99_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_sustained_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50_cv: Option<f64>,
    pub confidence: RayonAutotuneConfidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RayonAutotuneSelection {
    pub selected: RayonAutotuneCandidateSummary,
    pub confidence: RayonAutotuneConfidence,
    pub reason: String,
}

pub fn select_best_probe(
    probes: &[RayonAutotuneProbeResult],
    slow_p95_threshold_ms: f64,
) -> Option<RayonAutotuneProbeResult> {
    let mut valid: Vec<RayonAutotuneProbeResult> =
        probes.iter().copied().filter(|p| p.valid).collect();
    if valid.is_empty() {
        return None;
    }
    let under_threshold: Vec<RayonAutotuneProbeResult> = valid
        .iter()
        .copied()
        .filter(|p| p.p95_ms <= slow_p95_threshold_ms)
        .collect();
    if !under_threshold.is_empty() {
        valid = under_threshold;
    }
    valid.sort_by(|a, b| {
        a.p50_ms
            .total_cmp(&b.p50_ms)
            .then_with(|| b.sustained_tps.total_cmp(&a.sustained_tps))
            .then_with(|| a.threads.cmp(&b.threads))
    });
    valid.into_iter().next()
}

pub fn summarize_candidate_results(
    candidates: &[usize],
    requested_repeats: usize,
    probes: &[RayonAutotuneProbeObservation],
    slow_p95_threshold_ms: f64,
    slow_p99_threshold_ms: f64,
) -> Vec<RayonAutotuneCandidateSummary> {
    candidates
        .iter()
        .copied()
        .map(|threads| {
            summarize_one_candidate(
                threads,
                requested_repeats,
                probes,
                slow_p95_threshold_ms,
                slow_p99_threshold_ms,
            )
        })
        .collect()
}

pub fn ranked_candidate_summaries(
    candidates: &[RayonAutotuneCandidateSummary],
) -> Vec<RayonAutotuneCandidateSummary> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(compare_candidate_summaries);
    ranked
}

pub fn fine_thread_candidates(
    effective_logical_cores: usize,
    top_coarse_candidate_count: usize,
    previous_high_confidence_profile_threads: Option<usize>,
    coarse_summaries: &[RayonAutotuneCandidateSummary],
) -> Vec<usize> {
    let logical = effective_logical_cores.max(1);
    let mut selected = Vec::new();
    push_valid_candidate(
        &mut selected,
        crate::parallel::default_compute_threads(logical),
        logical,
    );
    push_valid_candidate(&mut selected, logical, logical);
    if logical > 1 {
        push_valid_candidate(&mut selected, logical - 1, logical);
    }
    if let Some(threads) = previous_high_confidence_profile_threads {
        push_valid_candidate(&mut selected, threads, logical);
    }

    for candidate in ranked_candidate_summaries(coarse_summaries)
        .into_iter()
        .filter(|c| c.successful_repeats > 0)
        .take(top_coarse_candidate_count)
    {
        push_valid_candidate(&mut selected, candidate.threads, logical);
    }

    selected
}

fn push_valid_candidate(out: &mut Vec<usize>, threads: usize, logical: usize) {
    if (1..=logical).contains(&threads) && !out.contains(&threads) {
        out.push(threads);
    }
}

pub fn select_best_candidate(
    candidates: &[RayonAutotuneCandidateSummary],
) -> Option<RayonAutotuneSelection> {
    let ranked = ranked_candidate_summaries(candidates);
    let selected = ranked
        .into_iter()
        .find(|c| c.successful_repeats > 0 && c.median_p50_ms.is_some())?;
    let confidence = selected.confidence;
    let reason = selection_reason(&selected);
    Some(RayonAutotuneSelection {
        selected,
        confidence,
        reason,
    })
}

fn summarize_one_candidate(
    threads: usize,
    requested_repeats: usize,
    probes: &[RayonAutotuneProbeObservation],
    slow_p95_threshold_ms: f64,
    slow_p99_threshold_ms: f64,
) -> RayonAutotuneCandidateSummary {
    let mut p50s = Vec::new();
    let mut p95s = Vec::new();
    let mut p99s = Vec::new();
    let mut tps = Vec::new();
    for probe in probes.iter().filter(|p| p.threads == threads && p.valid) {
        if let Some(v) = finite(probe.p50_ms) {
            p50s.push(v);
        }
        if let Some(v) = finite(probe.p95_ms) {
            p95s.push(v);
        }
        if let Some(v) = finite(probe.p99_ms) {
            p99s.push(v);
        }
        if let Some(v) = finite(probe.sustained_tps) {
            tps.push(v);
        }
    }

    let successful_repeats = probes
        .iter()
        .filter(|p| p.threads == threads && p.valid)
        .count();
    let all_repeats_successful = requested_repeats > 0 && successful_repeats == requested_repeats;
    let median_p50_ms = median(&mut p50s);
    let worst_p95_ms = max_finite(&p95s);
    let worst_p99_ms = max_finite(&p99s);
    let median_sustained_tps = median(&mut tps);
    let p50_cv = coefficient_of_variation(&p50s);
    let p95_below_slow_threshold = worst_p95_ms
        .map(|v| v <= slow_p95_threshold_ms)
        .unwrap_or(false);
    let p99_below_slow_threshold = worst_p99_ms
        .map(|v| v <= slow_p99_threshold_ms)
        .unwrap_or(true);

    let rejection_reason = if successful_repeats == 0 {
        Some("no successful probes".to_string())
    } else if !all_repeats_successful {
        Some(format!(
            "only {successful_repeats}/{requested_repeats} probes succeeded"
        ))
    } else if !p95_below_slow_threshold {
        Some(format!(
            "worst p95 {:.1}ms exceeds {:.1}ms slow threshold",
            worst_p95_ms.unwrap_or(f64::INFINITY),
            slow_p95_threshold_ms
        ))
    } else if !p99_below_slow_threshold {
        Some(format!(
            "worst p99 {:.1}ms exceeds {:.1}ms slow threshold",
            worst_p99_ms.unwrap_or(f64::INFINITY),
            slow_p99_threshold_ms
        ))
    } else if requested_repeats < 2 {
        Some("only one repeat; stability not established".to_string())
    } else {
        None
    };

    let high_variance =
        requested_repeats >= 2 && p50_cv.map(|cv| cv > LOW_VARIANCE_CV).unwrap_or(false);
    let confidence = if rejection_reason.is_some() {
        RayonAutotuneConfidence::Low
    } else if high_variance {
        RayonAutotuneConfidence::Medium
    } else {
        RayonAutotuneConfidence::High
    };

    RayonAutotuneCandidateSummary {
        threads,
        requested_repeats,
        successful_repeats,
        all_repeats_successful,
        p95_below_slow_threshold,
        p99_below_slow_threshold,
        median_p50_ms,
        worst_p95_ms,
        worst_p99_ms,
        median_sustained_tps,
        p50_cv,
        confidence,
        rejection_reason,
    }
}

fn compare_candidate_summaries(
    a: &RayonAutotuneCandidateSummary,
    b: &RayonAutotuneCandidateSummary,
) -> std::cmp::Ordering {
    b.all_repeats_successful
        .cmp(&a.all_repeats_successful)
        .then_with(|| b.p95_below_slow_threshold.cmp(&a.p95_below_slow_threshold))
        .then_with(|| b.p99_below_slow_threshold.cmp(&a.p99_below_slow_threshold))
        .then_with(|| confidence_rank(b.confidence).cmp(&confidence_rank(a.confidence)))
        .then_with(|| cmp_optional_f64_asc(a.median_p50_ms, b.median_p50_ms))
        .then_with(|| cmp_optional_f64_desc(a.median_sustained_tps, b.median_sustained_tps))
        .then_with(|| cmp_optional_f64_asc(a.p50_cv, b.p50_cv))
        .then_with(|| a.threads.cmp(&b.threads))
}

fn confidence_rank(confidence: RayonAutotuneConfidence) -> u8 {
    match confidence {
        RayonAutotuneConfidence::High => 2,
        RayonAutotuneConfidence::Medium => 1,
        RayonAutotuneConfidence::Low => 0,
    }
}

fn cmp_optional_f64_asc(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn cmp_optional_f64_desc(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    cmp_optional_f64_asc(b, a)
}

fn selection_reason(selected: &RayonAutotuneCandidateSummary) -> String {
    let p50 = selected
        .median_p50_ms
        .map(format_ms)
        .unwrap_or_else(|| "n/a".to_string());
    let p95 = selected
        .worst_p95_ms
        .map(format_ms)
        .unwrap_or_else(|| "n/a".to_string());
    let p99 = selected
        .worst_p99_ms
        .map(format_ms)
        .unwrap_or_else(|| "n/a".to_string());
    let tps = selected
        .median_sustained_tps
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "n/a".to_string());
    let base = format!(
        "selected {} threads: {}/{} fine probes succeeded; worst p95={p95}; worst p99={p99}; median p50={p50}; median TPS={tps}; confidence={}",
        selected.threads,
        selected.successful_repeats,
        selected.requested_repeats,
        selected.confidence.as_str()
    );
    match selected.rejection_reason.as_deref() {
        Some(reason) => format!("{base}; {reason}"),
        None => base,
    }
}

fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|v| v.is_finite())
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

fn max_finite(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.total_cmp(b))
}

fn coefficient_of_variation(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    if values.len() == 1 {
        return Some(0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return Some(0.0);
    }
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt() / mean.abs())
}

fn format_ms(v: f64) -> String {
    format!("{v:.1}ms")
}

pub fn format_autotune_table(
    probes: &[RayonAutotuneProbeObservation],
    candidates: &[RayonAutotuneCandidateSummary],
) -> String {
    let mut reasons = BTreeMap::new();
    for candidate in candidates {
        if let Some(reason) = candidate.rejection_reason.as_deref() {
            reasons.insert(candidate.threads, reason);
        }
    }

    let mut lines = Vec::with_capacity(probes.len() + 1);
    lines.push(format!(
        "{:<7} {:>7} {:>6} {:>9} {:>9} {:>9} {:>9} {:>7} {}",
        "stage", "threads", "repeat", "p50", "p95", "p99", "TPS", "valid", "reason"
    ));
    for probe in probes {
        let reason = if probe.valid {
            reasons
                .get(&probe.threads)
                .copied()
                .unwrap_or("")
                .to_string()
        } else {
            probe.reason.as_deref().unwrap_or("invalid").to_string()
        };
        lines.push(format!(
            "{:<7} {:>7} {:>6} {:>9} {:>9} {:>9} {:>9} {:>7} {}",
            probe.stage.as_str(),
            probe.threads,
            probe.repeat,
            format_optional_ms(probe.p50_ms),
            format_optional_ms(probe.p95_ms),
            format_optional_ms(probe.p99_ms),
            probe
                .sustained_tps
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "-".to_string()),
            if probe.valid { "yes" } else { "no" },
            reason
        ));
    }
    lines.join("\n")
}

fn format_optional_ms(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "-".to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgressWatchdogConfig {
    pub timeout: Option<Duration>,
}

impl ProgressWatchdogConfig {
    pub const fn disabled() -> Self {
        Self { timeout: None }
    }

    pub fn enabled(&self) -> bool {
        self.timeout.is_some()
    }
}

pub fn normalize_progress_timeout_secs(
    cli: Option<u64>,
    config: Option<u64>,
) -> ProgressWatchdogConfig {
    match cli.or(config) {
        Some(0) | None => ProgressWatchdogConfig::disabled(),
        Some(secs) => ProgressWatchdogConfig {
            timeout: Some(Duration::from_secs(secs)),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayonAutotuneProfile {
    pub threads: usize,
    #[serde(default)]
    pub effective_cpu_mask: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_cpu_mask_display: Option<String>,
    #[serde(default)]
    pub logical_cores: usize,
    #[serde(default)]
    pub repeats: usize,
    #[serde(default)]
    pub p50_ms: f64,
    #[serde(default)]
    pub p95_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<f64>,
    #[serde(default)]
    pub sustained_tps: f64,
    #[serde(default)]
    pub median_p50_ms: f64,
    #[serde(default)]
    pub worst_p95_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_p99_ms: Option<f64>,
    #[serde(default)]
    pub median_sustained_tps: f64,
    #[serde(default)]
    pub confidence: RayonAutotuneConfidence,
    #[serde(default)]
    pub selection_reason: String,
    #[serde(default)]
    pub candidate_results: Vec<RayonAutotuneCandidateSummary>,
    #[serde(default)]
    pub probe_results: Vec<RayonAutotuneProbeObservation>,
}

impl RayonAutotuneProfile {
    pub fn reusable_by_default(&self) -> bool {
        self.confidence != RayonAutotuneConfidence::Low
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RayonAutotuneProfileStore {
    profiles: BTreeMap<String, RayonAutotuneProfile>,
}

pub fn default_profile_path(data_dir: &Path) -> PathBuf {
    data_dir.join(".mer-rayon-autotune.json")
}

pub fn load_profile(path: &Path, key: &CpuAutotuneKey) -> Option<RayonAutotuneProfile> {
    let body = std::fs::read_to_string(path).ok()?;
    let store: RayonAutotuneProfileStore = serde_json::from_str(&body).ok()?;
    store.profiles.get(&key.cache_key()).cloned()
}

pub fn save_profile(
    path: &Path,
    key: &CpuAutotuneKey,
    profile: RayonAutotuneProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut store = match std::fs::read_to_string(path) {
        Ok(body) => serde_json::from_str::<RayonAutotuneProfileStore>(&body).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "failed to parse existing Rayon autotune profile store {}: {e}; refusing to overwrite",
                    path.display()
                ),
            )
        })?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => RayonAutotuneProfileStore::default(),
        Err(e) => return Err(e.into()),
    };
    store.profiles.insert(key.cache_key(), profile);
    let body = serde_json::to_vec_pretty(&store)?;
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("profile path {} has no file name", path.display()),
        )
    })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        nonce
    ));
    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut tmp = std::fs::File::create(&tmp_path)?;
        tmp.write_all(&body)?;
        tmp.sync_all()?;
        drop(tmp);
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    write_result
}

pub fn percentile_ms(sorted_us: &[u64], q: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted_us.len() - 1) as f64 * q).round() as usize;
    sorted_us[idx] as f64 / 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa::EffectiveCpuAffinity;

    fn test_machine() -> CpuAutotuneMachine {
        CpuAutotuneMachine {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            cpu_vendor: "Vendor A".to_string(),
            cpu_model: "Example CPU  1 @ 2.90GHz".to_string(),
            cpu_features: vec!["avx2", "fma"],
            available_logical_cores: 32,
            effective_cpu_mask: Some((0..32).collect()),
        }
    }

    #[test]
    fn identical_machine_data_has_identical_fingerprint() {
        assert_eq!(test_machine().fingerprint(), test_machine().fingerprint());
    }

    #[test]
    fn fingerprint_differs_when_only_cpu_vendor_changes() {
        let a = test_machine();
        let mut b = a.clone();
        b.cpu_vendor = "Vendor B".to_string();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_differs_when_only_cpu_model_changes() {
        let a = test_machine();
        let mut b = a.clone();
        b.cpu_model = "Example CPU  2 @ 2.90GHz".to_string();
        assert_ne!(a.fingerprint(), b.fingerprint());

        b.cpu_model = "Example CPU 1 @ 2.90GHz".to_string();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_is_version_2_with_exact_cpu_identity() {
        assert_eq!(
            test_machine().fingerprint(),
            "version=2;os=linux;arch=x86_64;cpu_vendor=Vendor A;cpu_model=Example CPU  1 @ 2.90GHz;features=avx2+fma;logical=32;affinity=0-31"
        );
    }

    #[test]
    fn fingerprint_preserves_literal_unknown_cpu_identity() {
        let mut machine = test_machine();
        machine.cpu_vendor = "unknown".to_string();
        machine.cpu_model = "unknown".to_string();
        assert!(machine
            .fingerprint()
            .contains(";cpu_vendor=unknown;cpu_model=unknown;"));
    }

    #[test]
    fn machine_uses_cached_cpu_identity() {
        let affinity = EffectiveCpuAffinity {
            cpus: Some((0..32).collect()),
            logical_cores: 32,
            display: "0-31".to_string(),
        };
        let machine = CpuAutotuneMachine::from_effective_affinity(&affinity);
        let cached = crate::kernels::cpu_features();
        assert_eq!(machine.cpu_vendor, cached.vendor);
        assert_eq!(machine.cpu_model, cached.model);
    }

    #[test]
    fn fingerprint_separates_masks_with_identical_logical_core_counts() {
        let a = test_machine();
        let mut b = a.clone();
        b.effective_cpu_mask = Some((32..64).collect());
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    const LEGACY_PROFILE_STORE_JSON: &str = r#"{
        "profiles": {
            "os=linux;arch=x86_64;features=avx2+fma;logical=32;affinity=0-31|model|backend": {
                "threads": 30,
                "p50_ms": 58.0,
                "p95_ms": 61.0,
                "sustained_tps": 17.0,
                "confidence": "high"
            }
        }
    }"#;

    fn legacy_profile_key() -> CpuAutotuneKey {
        CpuAutotuneKey {
            machine_fingerprint: "os=linux;arch=x86_64;features=avx2+fma;logical=32;affinity=0-31"
                .to_string(),
            model_fingerprint: "model".to_string(),
            backend_fingerprint: "backend".to_string(),
        }
    }

    #[test]
    fn legacy_profile_store_json_remains_readable() {
        let store: RayonAutotuneProfileStore =
            serde_json::from_str(LEGACY_PROFILE_STORE_JSON).unwrap();
        assert_eq!(store.profiles.len(), 1);
        let profile = store
            .profiles
            .get(&legacy_profile_key().cache_key())
            .unwrap();
        assert_eq!(profile.threads, 30);
        assert_eq!(profile.confidence, RayonAutotuneConfidence::High);
        assert!(profile.reusable_by_default());
    }

    #[test]
    fn legacy_profile_misses_v2_key_and_can_coexist_with_v2_profile() {
        let dir = std::env::temp_dir().join(format!(
            "mer-rayon-autotune-v2-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = default_profile_path(&dir);
        std::fs::write(&path, LEGACY_PROFILE_STORE_JSON).unwrap();
        let legacy_key = legacy_profile_key();
        let v2_key = CpuAutotuneKey {
            machine_fingerprint: test_machine().fingerprint(),
            ..legacy_key.clone()
        };

        let legacy_profile = load_profile(&path, &legacy_key).unwrap();
        assert!(legacy_profile.reusable_by_default());
        assert_ne!(legacy_key.cache_key(), v2_key.cache_key());
        assert!(load_profile(&path, &v2_key).is_none());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            LEGACY_PROFILE_STORE_JSON
        );

        let mut v2_profile = legacy_profile.clone();
        v2_profile.threads = 28;
        save_profile(&path, &v2_key, v2_profile).unwrap();
        assert_eq!(load_profile(&path, &v2_key).unwrap().threads, 28);
        let saved_legacy = load_profile(&path, &legacy_key).unwrap();
        assert_eq!(
            serde_json::to_value(saved_legacy).unwrap(),
            serde_json::to_value(legacy_profile).unwrap()
        );
        let store: RayonAutotuneProfileStore =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(store.profiles.len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fingerprint_differs_for_different_effective_masks() {
        let a = EffectiveCpuAffinity {
            cpus: Some((0..25).collect()),
            logical_cores: 25,
            display: "0-24".to_string(),
        };
        let b = EffectiveCpuAffinity {
            cpus: Some((0..16).collect()),
            logical_cores: 16,
            display: "0-15".to_string(),
        };
        assert_ne!(
            machine_fingerprint_from_affinity(&a),
            machine_fingerprint_from_affinity(&b)
        );
    }

    #[test]
    fn fingerprint_separates_unknown_affinity_from_known_mask() {
        let unknown = EffectiveCpuAffinity {
            cpus: None,
            logical_cores: 25,
            display: "unavailable".to_string(),
        };
        let masked = EffectiveCpuAffinity {
            cpus: Some((0..25).collect()),
            logical_cores: 25,
            display: "0-24".to_string(),
        };
        assert_ne!(
            machine_fingerprint_from_affinity(&unknown),
            machine_fingerprint_from_affinity(&masked)
        );
    }

    #[test]
    fn candidate_generation_uses_effective_logical_cores() {
        let candidates = default_thread_candidates(25);
        assert!(candidates.contains(&25));
        assert!(candidates.contains(&crate::parallel::default_compute_threads(25)));
        assert!(candidates.iter().all(|&c| (1..=25).contains(&c)));
    }

    #[test]
    fn p95_aware_selection_prefers_non_slow_candidates_when_possible() {
        let probes = [
            RayonAutotuneProbeResult {
                threads: 8,
                valid: true,
                p50_ms: 8.0,
                p95_ms: 200.0,
                p99_ms: Some(220.0),
                sustained_tps: 125.0,
            },
            RayonAutotuneProbeResult {
                threads: 12,
                valid: true,
                p50_ms: 9.0,
                p95_ms: 20.0,
                p99_ms: Some(25.0),
                sustained_tps: 111.0,
            },
        ];
        assert_eq!(select_best_probe(&probes, 100.0).unwrap().threads, 12);
    }

    fn probe(
        threads: usize,
        repeat: usize,
        p50_ms: f64,
        p95_ms: f64,
        p99_ms: f64,
        sustained_tps: f64,
    ) -> RayonAutotuneProbeObservation {
        RayonAutotuneProbeObservation {
            threads,
            stage: RayonAutotuneProbeStage::Fine,
            repeat,
            tokens: 2_000,
            valid: true,
            p50_ms: Some(p50_ms),
            p95_ms: Some(p95_ms),
            p99_ms: Some(p99_ms),
            sustained_tps: Some(sustained_tps),
            reason: None,
        }
    }

    #[test]
    fn repeated_candidate_aggregation_tracks_median_and_worst_case() {
        let probes = [
            probe(25, 1, 58.0, 61.0, 70.0, 17.0),
            probe(25, 2, 62.0, 75.0, 82.0, 16.0),
        ];
        let summaries = summarize_candidate_results(&[25], 2, &probes, 80.0, 120.0);
        let s = &summaries[0];
        assert_eq!(s.successful_repeats, 2);
        assert!(s.all_repeats_successful);
        assert_eq!(s.median_p50_ms, Some(60.0));
        assert_eq!(s.worst_p95_ms, Some(75.0));
        assert_eq!(s.worst_p99_ms, Some(82.0));
        assert_eq!(s.median_sustained_tps, Some(16.5));
        assert_eq!(s.confidence, RayonAutotuneConfidence::High);
    }

    #[test]
    fn stable_candidate_beats_lower_p50_with_high_worst_p95() {
        let probes = [
            probe(8, 1, 50.0, 120.0, 140.0, 20.0),
            probe(8, 2, 51.0, 130.0, 150.0, 19.0),
            probe(12, 1, 58.0, 68.0, 90.0, 17.0),
            probe(12, 2, 59.0, 70.0, 94.0, 16.0),
        ];
        let summaries = summarize_candidate_results(&[8, 12], 2, &probes, 80.0, 120.0);
        let selected = select_best_candidate(&summaries).unwrap();
        assert_eq!(selected.selected.threads, 12);
        assert_eq!(selected.confidence, RayonAutotuneConfidence::High);
    }

    #[test]
    fn high_variance_candidate_loses_to_high_confidence_candidate() {
        let probes = [
            probe(8, 1, 45.0, 72.0, 88.0, 22.0),
            probe(8, 2, 80.0, 74.0, 90.0, 21.0),
            probe(12, 1, 64.0, 70.0, 82.0, 16.0),
            probe(12, 2, 65.0, 71.0, 83.0, 15.0),
        ];
        let summaries = summarize_candidate_results(&[8, 12], 2, &probes, 80.0, 120.0);
        let selected = select_best_candidate(&summaries).unwrap();
        assert_eq!(selected.selected.threads, 12);
        assert_eq!(selected.confidence, RayonAutotuneConfidence::High);
    }

    #[test]
    fn fine_candidates_include_anchors_regardless_of_coarse_rank() {
        let probes = [
            probe(1, 1, 10.0, 12.0, 14.0, 100.0),
            probe(2, 1, 11.0, 13.0, 15.0, 90.0),
            probe(5, 1, 80.0, 82.0, 84.0, 12.0),
            probe(7, 1, 90.0, 92.0, 94.0, 11.0),
            probe(8, 1, 95.0, 97.0, 99.0, 10.0),
        ];
        let summaries = summarize_candidate_results(&[1, 2, 5, 7, 8], 1, &probes, 100.0, 120.0);
        let fine = fine_thread_candidates(8, 2, Some(5), &summaries);
        assert!(fine.contains(&crate::parallel::default_compute_threads(8)));
        assert!(fine.contains(&8));
        assert!(fine.contains(&7));
        assert!(fine.contains(&5));
        assert!(fine.contains(&1));
        assert!(fine.contains(&2));
    }

    #[test]
    fn profile_confidence_metadata_serializes_and_deserializes() {
        let probes = vec![probe(25, 1, 58.0, 61.0, 70.0, 17.0)];
        let candidates = summarize_candidate_results(&[25], 1, &probes, 80.0, 120.0);
        let profile = RayonAutotuneProfile {
            threads: 25,
            effective_cpu_mask: Some((0..25).collect()),
            effective_cpu_mask_display: Some("0-24".to_string()),
            logical_cores: 25,
            repeats: 1,
            p50_ms: 58.0,
            p95_ms: 61.0,
            p99_ms: Some(70.0),
            sustained_tps: 17.0,
            median_p50_ms: 58.0,
            worst_p95_ms: 61.0,
            worst_p99_ms: Some(70.0),
            median_sustained_tps: 17.0,
            confidence: RayonAutotuneConfidence::Low,
            selection_reason: "only one repeat; stability not established".to_string(),
            candidate_results: candidates,
            probe_results: probes,
        };
        let json = serde_json::to_string(&profile).unwrap();
        let back: RayonAutotuneProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(back.threads, 25);
        assert_eq!(back.confidence, RayonAutotuneConfidence::Low);
        assert_eq!(back.effective_cpu_mask_display.as_deref(), Some("0-24"));
        assert_eq!(back.logical_cores, 25);
        assert_eq!(back.repeats, 1);
        assert_eq!(back.worst_p99_ms, Some(70.0));
        assert!(back.selection_reason.contains("only one repeat"));
        assert_eq!(back.candidate_results.len(), 1);
        assert_eq!(back.probe_results.len(), 1);
    }

    #[test]
    fn legacy_profile_without_confidence_defaults_to_low() {
        let legacy = r#"{
            "threads": 25,
            "p50_ms": 58.0,
            "p95_ms": 61.0,
            "sustained_tps": 17.0
        }"#;
        let profile: RayonAutotuneProfile = serde_json::from_str(legacy).unwrap();
        assert_eq!(profile.confidence, RayonAutotuneConfidence::Low);
        assert!(!profile.reusable_by_default());
    }

    #[test]
    fn save_profile_refuses_to_clobber_unparseable_store() {
        let dir = std::env::temp_dir().join(format!(
            "mer-rayon-autotune-save-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("profile.json");
        std::fs::write(&path, "{not-json").unwrap();
        let key = CpuAutotuneKey {
            machine_fingerprint: "machine".to_string(),
            model_fingerprint: "model".to_string(),
            backend_fingerprint: "backend".to_string(),
        };
        let probes = vec![probe(25, 1, 58.0, 61.0, 70.0, 17.0)];
        let candidates = summarize_candidate_results(&[25], 1, &probes, 80.0, 120.0);
        let profile = RayonAutotuneProfile {
            threads: 25,
            effective_cpu_mask: Some((0..25).collect()),
            effective_cpu_mask_display: Some("0-24".to_string()),
            logical_cores: 25,
            repeats: 1,
            p50_ms: 58.0,
            p95_ms: 61.0,
            p99_ms: Some(70.0),
            sustained_tps: 17.0,
            median_p50_ms: 58.0,
            worst_p95_ms: 61.0,
            worst_p99_ms: Some(70.0),
            median_sustained_tps: 17.0,
            confidence: RayonAutotuneConfidence::Low,
            selection_reason: "test".to_string(),
            candidate_results: candidates,
            probe_results: probes,
        };

        assert!(save_profile(&path, &key, profile).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{not-json");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn selection_reason_is_recorded() {
        let probes = [
            probe(25, 1, 58.0, 61.0, 70.0, 17.0),
            probe(25, 2, 59.0, 62.0, 71.0, 16.0),
        ];
        let summaries = summarize_candidate_results(&[25], 2, &probes, 80.0, 120.0);
        let selected = select_best_candidate(&summaries).unwrap();
        assert!(selected.reason.contains("selected 25 threads"));
        assert!(selected.reason.contains("confidence=high"));
    }

    #[cfg(all(
        not(feature = "nightly-amx"),
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn stable_feature_list_does_not_probe_unstable_amx_detection() {
        assert!(!detected_cpu_features()
            .iter()
            .any(|feature| feature.starts_with("amx")));
    }

    #[test]
    fn progress_timeout_zero_disables_watchdog() {
        assert!(!normalize_progress_timeout_secs(Some(0), Some(300)).enabled());
        assert!(!normalize_progress_timeout_secs(None, Some(0)).enabled());
    }

    #[test]
    fn positive_progress_timeout_enables_watchdog_config() {
        let cfg = normalize_progress_timeout_secs(Some(300), None);
        assert!(cfg.enabled());
        assert_eq!(cfg.timeout, Some(Duration::from_secs(300)));
    }
}
