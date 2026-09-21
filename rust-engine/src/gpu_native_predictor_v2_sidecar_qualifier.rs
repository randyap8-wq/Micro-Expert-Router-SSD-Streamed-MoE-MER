//! P1K qualification-only control/treatment driver for the frozen P1J mechanism.
//! No tuning or serving activation. Existing production stepping owns all work.

use crate::gpu_native_real_benchmark as evidence;
use crate::predictor_v2::{
    p1e, ObservationConfig, ReconciliationSnapshot, RequestPhase, TerminalReason,
};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "mer.predictor-v2-p1k-sidecar-runtime-qualifier.v1";
const MAX_POSITIONS: usize = 4096;

#[derive(Clone, Debug, clap::Args, Serialize)]
pub(crate) struct CommandArgs {
    #[arg(long)]
    pub(crate) config: PathBuf,
    #[arg(long)]
    request_json: PathBuf,
    #[arg(long)]
    expected_adapter_name: String,
    #[arg(long)]
    report_out: PathBuf,
}

// Reject unsupported request semantics instead of silently discarding them.
// Sampling must explicitly select greedy; neutral optional fields are accepted.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestInput {
    prompt: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: usize,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    n: Option<usize>,
    stream: Option<bool>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    repetition_penalty: Option<f64>,
    model: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

fn parse_request(bytes: &[u8]) -> Result<(String, usize)> {
    let input: RequestInput = serde_json::from_slice(bytes)?;
    if input.max_tokens == 0
        || !(input.temperature == Some(0.0) || input.top_k == Some(1))
        || input.temperature.is_some_and(|v| v != 0.0)
        || input.top_k.is_some_and(|v| v != 1)
        || input.top_p.is_some_and(|v| v != 1.0)
        || input.n.is_some_and(|v| v != 1)
        || input.stream == Some(true)
        || input.frequency_penalty.is_some_and(|v| v != 0.0)
        || input.presence_penalty.is_some_and(|v| v != 0.0)
        || input.repetition_penalty.is_some_and(|v| v != 1.0)
    {
        return Err("requires exact positive max_tokens and explicit unmodified greedy sampling (temperature=0 or top_k=1)".into());
    }
    // A model name is descriptive only; strict artifact identity is validated
    // against the resolved checkpoint below, as in the existing CLI helpers.
    let _ = input.model;
    let prompt = match (input.prompt, input.messages) {
        (Some(prompt), None) => prompt,
        (None, Some(messages)) if !messages.is_empty() => {
            if messages.iter().any(|m| {
                !matches!(m.role.as_str(), "system" | "user" | "assistant")
                    || m.content.trim().is_empty()
            }) {
                return Err(
                    "messages require explicit text content and system/user/assistant roles".into(),
                );
            }
            let values = messages
                .iter()
                .map(serde_json::to_value)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            crate::flatten_bench_messages(&values)
        }
        _ => {
            return Err("requires exactly one nonempty string prompt or text messages array".into())
        }
    };
    if prompt.trim().is_empty() {
        return Err("prompt must contain non-whitespace text".into());
    }
    Ok((prompt, input.max_tokens))
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| "observation count overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| "observation count overflow".into())
}
fn increment(count: &mut usize) -> Result<()> {
    *count = add(*count, 1)?;
    Ok(())
}

fn observation_config(prompt_length: usize, output_tokens: usize) -> Result<ObservationConfig> {
    let decode = output_tokens.checked_sub(1).ok_or("zero output tokens")?;
    let positions = add(prompt_length, decode)?;
    if prompt_length == 0 || !(3..=MAX_POSITIONS).contains(&positions) {
        return Err("requires 3..=4096 completed positions including a nonempty prompt".into());
    }
    // P0 retains 48 route records per completed position, and at most 64 new
    // cells per adjacent top-8 transition. Each (layer pair, position-kind pair)
    // has only 128*128 possible cells. Count prompt and decode buckets separately;
    // the single prompt->decode transition is a distinct 64-cell bucket.
    // This is the smallest common capacity implied by these worst-case bounds,
    // with no padding. It also dominates P1E's same-layer table and one report
    // record per completed position (including the final censored nomination).
    // Recovery does not retrain P0; P1E keeps its frozen 512 events/record ceiling.
    let cells = |transitions: usize| -> Result<usize> { Ok(mul(transitions, 64)?.min(128 * 128)) };
    let within = mul(47, add(cells(prompt_length)?, cells(decode)?)?)?;
    let prompt_edges = prompt_length.checked_sub(1).ok_or("empty prompt")?;
    let decode_edges = if decode == 0 {
        0
    } else {
        decode.checked_sub(1).ok_or("decode edge overflow")?
    };
    let across = add(cells(prompt_edges)?, cells(decode_edges)?)?;
    let transitions = add(add(within, across)?, if decode > 0 { 64 } else { 0 })?;
    Ok(ObservationConfig {
        phase: RequestPhase::Measured,
        phase_run_index: 0,
        prompt_length,
        capacity_per_collection: transitions.max(mul(positions, 48)?),
    })
}

fn ensure_output_absent(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(format!(
            "refusing to overwrite qualification evidence {}",
            path.display()
        )
        .into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn write_report(path: &Path, report: &impl Serialize) -> Result<()> {
    ensure_output_absent(path)?;
    let bytes = serde_json::to_vec_pretty(report)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("report path has no file name")?
        .to_string_lossy();
    let temp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| -> Result<()> {
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
        // Atomic no-clobber publication in the same directory. Unlike rename,
        // hard_link cannot replace evidence created after the preflight check.
        std::fs::hard_link(&temp, path)?;
        std::fs::remove_file(&temp)?;
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

// The normalized causal trace is separate from raw mechanical evidence:
// recovery events, physical LRU/install snapshots and runtime counters remain
// lossless in raw_p1e_report/runtime_counters. P1J may change those mechanics.
// Within predictor/route truth only arm-local identities and time are removed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NormalizedObservation {
    candidate: p1e::Candidate,
    current_at_f: Option<bool>,
    source_at_f: p1e::HostSource,
    freeze_evidence_present: bool,
    freeze_incomplete: Option<p1e::Error>,
    deadline: Option<NormalizedDeadline>,
    outcome: p1e::Outcome,
    opportunities: p1e::Opportunities,
    incomplete: Option<p1e::Error>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NormalizedDeadline {
    request: crate::predictor_v2::RequestIdentity,
    position: crate::predictor_v2::PositionIdentity,
    current: Option<bool>,
    evidence_present: bool,
    incomplete: Option<p1e::Error>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NormalizedP1e {
    request: crate::predictor_v2::RequestIdentity,
    observations: Vec<NormalizedObservation>,
    opportunities: Vec<(u64, p1e::Opportunities)>,
    no_emissions: Vec<p1e::NoEmission>,
    incomplete: Option<p1e::Error>,
    partitions: p1e::Partitions,
    readiness_measured: bool,
}
fn normalize_request(
    mut id: crate::predictor_v2::RequestIdentity,
) -> crate::predictor_v2::RequestIdentity {
    id.runtime_namespace = 0;
    id.request_sequence = 0;
    id
}
fn normalize(raw: &p1e::Report) -> NormalizedP1e {
    NormalizedP1e {
        request: normalize_request(raw.request),
        observations: raw
            .observations
            .iter()
            .map(|r| {
                let mut candidate = r.freeze.candidate;
                candidate.request = normalize_request(candidate.request);
                candidate.namespace.runtime = 0;
                candidate.namespace.context = 0;
                candidate.namespace.arena = 0;
                // Candidate generation is a deterministic request-local signal
                // counter, not a runtime incarnation: retain it and both cutoffs.
                let mut source = r.freeze.source;
                // Logical cache generations are runtime-local. Preserve existence.
                source.logical_generation = source.logical_generation.map(|_| 0);
                NormalizedObservation {
                    candidate,
                    current_at_f: r.freeze.current,
                    source_at_f: source,
                    freeze_evidence_present: r.freeze.physical.is_some(),
                    freeze_incomplete: r.freeze.incomplete,
                    deadline: r.deadline.map(|d| NormalizedDeadline {
                        request: normalize_request(d.request),
                        position: d.position,
                        current: d.current,
                        evidence_present: d.physical.is_some(),
                        incomplete: d.incomplete,
                    }),
                    outcome: r.outcome,
                    opportunities: r.opportunities(),
                    incomplete: r.incomplete,
                }
            })
            .collect(),
        opportunities: raw.opportunities.clone(),
        no_emissions: raw.no_emissions.clone(),
        incomplete: raw.incomplete,
        partitions: raw.partitions,
        readiness_measured: raw.readiness_measured,
    }
}

#[derive(Debug, Serialize)]
struct FrozenContract {
    source_layer: usize,
    target_layer: usize,
    position_delta: u64,
    candidate_count: usize,
    signal_revision: u64,
    request_local: bool,
    host_backed_source_only: bool,
    sidecar_bank: u32,
    sidecar_slot: u32,
    sidecar_bytes: u64,
    ordinary_capacity: usize,
    nvme_fallback: bool,
    ram_only_fallback: bool,
    score_threshold: Option<u64>,
    runner_up: bool,
    serving_activation: bool,
}
impl Default for FrozenContract {
    fn default() -> Self {
        Self {
            source_layer: 47,
            target_layer: 47,
            position_delta: 1,
            candidate_count: 1,
            signal_revision: p1e::REVISION,
            request_local: true,
            host_backed_source_only: true,
            sidecar_bank: 1,
            sidecar_slot: 0,
            sidecar_bytes: 2_654_212,
            ordinary_capacity: 8,
            nvme_fallback: false,
            ram_only_fallback: false,
            score_threshold: None,
            runner_up: false,
            serving_activation: false,
        }
    }
}
// Serialize constants rather than mutable report fields: this phase cannot
// accidentally acquire performance authority through a qualification result.
#[derive(Debug)]
struct PerformanceBoundary;
impl Serialize for PerformanceBoundary {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("PerformanceBoundary", 3)?;
        s.serialize_field("performance_comparison_authorized", &false)?;
        s.serialize_field("resource_footprint_matched", &false)?;
        s.serialize_field("performance_verdict", "NOT_AUTHORIZED")?;
        s.end()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum Arm {
    Control,
    Treatment,
}

#[derive(Serialize)]
struct ArmProvenance {
    provenance: evidence::BenchmarkProvenance,
    config_path: String,
    config_sha256: String,
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    production_configuration: evidence::ProductionConfiguration,
}
struct PreparedArm {
    spec: crate::ResolvedRealCliSpec,
    provenance: ArmProvenance,
}
#[derive(Serialize)]
struct ArmReport {
    arm: Arm,
    provenance: Option<ArmProvenance>,
    runtime_build_attempted: bool,
    runtime_constructed: bool,
    runtime_resolved_config_sha256: Option<String>,
    model_load: Option<crate::greedy_parity::ModelLoadEvidence>,
    adapter: Option<crate::backend::GpuDeviceIdentity>,
    runtime_contract: Option<evidence::RuntimeContractEvidence>,
    generated_token_ids: Vec<u32>,
    generated_token_ids_sha256: String,
    completed_positions: usize,
    observation_capacity_per_collection: usize,
    raw_p1e_report: Option<p1e::Report>,
    normalized_p1e: Option<NormalizedP1e>,
    predictor_v2_snapshot: Option<ReconciliationSnapshot>,
    runtime_counters: Option<evidence::RequestSnapshots>,
    runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
    errors: Vec<String>,
    ordinary_invariants_pass: bool,
}
impl ArmReport {
    fn new(arm: Arm) -> Self {
        Self {
            arm,
            provenance: None,
            runtime_build_attempted: false,
            runtime_constructed: false,
            runtime_resolved_config_sha256: None,
            model_load: None,
            adapter: None,
            runtime_contract: None,
            generated_token_ids: Vec::new(),
            generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&[]),
            completed_positions: 0,
            observation_capacity_per_collection: 0,
            raw_p1e_report: None,
            normalized_p1e: None,
            predictor_v2_snapshot: None,
            runtime_counters: None,
            runtime_shutdown: None,
            errors: Vec::new(),
            ordinary_invariants_pass: false,
        }
    }
}
#[derive(Debug, Serialize)]
struct Parity {
    output_exact_match: bool,
    normalized_p1e_exact_match: bool,
    mismatch_details: Vec<String>,
}
#[derive(Debug, Serialize)]
struct Mechanism {
    control_movement_zero: bool,
    treatment_movement_accounting_pass: bool,
    treatment_direct_matching_demand_credits: Option<u64>,
    qualification_failure_reasons: Vec<String>,
}
#[derive(Serialize)]
struct Report {
    schema: &'static str,
    mode: &'static str,
    expected_adapter_name: String,
    request: evidence::RequestEvidence,
    planned_positions: usize,
    frozen_p1j_contract: FrozenContract,
    control: ArmReport,
    treatment: ArmReport,
    parity: Parity,
    mechanism: Mechanism,
    #[serde(flatten)]
    performance: PerformanceBoundary,
    qualification_pass: bool,
}

fn control_gate(p: &ReconciliationSnapshot) -> Vec<String> {
    let mut errors = Vec::new();
    if p.incomplete.is_some() {
        errors.push(format!("control accounting incomplete: {:?}", p.incomplete));
    }
    // No P0 movement lifecycle stage is permitted in the control arm.
    let mut zero = p.clone();
    zero.incomplete = None;
    if zero != ReconciliationSnapshot::default() {
        errors.push(format!(
            "control P0 movement lifecycle must be entirely zero: {p:?}"
        ));
    }
    errors
}
fn treatment_gate(p: &ReconciliationSnapshot) -> Vec<String> {
    let mut errors = Vec::new();
    if p.incomplete.is_some() {
        errors.push(format!(
            "treatment accounting incomplete: {:?}",
            p.incomplete
        ));
    }
    for (valid, reason) in [
        (p.emitted > 0, "emitted must be positive"),
        (
            p.terminal_predictions == p.emitted,
            "terminal_predictions must equal emitted",
        ),
        (p.live_predictions == 0, "live_predictions must be zero"),
        (p.accepted == p.emitted, "accepted must equal emitted"),
        (
            p.source_completed == p.emitted,
            "source_completed must equal emitted",
        ),
        (p.source_failed == 0, "source_failed must be zero"),
        (p.source_cancelled == 0, "source_cancelled must be zero"),
        (p.source_live == 0, "source_live must be zero"),
        (p.reservations_live == 0, "reservations_live must be zero"),
        (p.available_installs == 0, "available_installs must be zero"),
        (
            p.direct_matching_demand_credits > 0,
            "direct_matching_demand_credits must be positive",
        ),
    ] {
        if !valid {
            errors.push(format!("treatment {reason}"));
        }
    }
    for (reason, count) in &p.terminal_categories {
        if *count > 0
            && matches!(
                reason,
                TerminalReason::InstallFailed | TerminalReason::SourceFailed
            )
        {
            errors.push(format!("treatment terminal {reason:?} count={count}"));
        }
    }
    errors
}
fn compare(control: &ArmReport, treatment: &ArmReport) -> Parity {
    let mut mismatch_details = Vec::new();
    let output_exact_match = control.generated_token_ids == treatment.generated_token_ids
        && control.generated_token_ids_sha256 == treatment.generated_token_ids_sha256
        && control.generated_token_ids_sha256
            == crate::greedy_parity::token_ids_sha256(&control.generated_token_ids)
        && treatment.generated_token_ids_sha256
            == crate::greedy_parity::token_ids_sha256(&treatment.generated_token_ids);
    if !output_exact_match {
        let first = control
            .generated_token_ids
            .iter()
            .zip(&treatment.generated_token_ids)
            .position(|(a, b)| a != b)
            .unwrap_or(
                control
                    .generated_token_ids
                    .len()
                    .min(treatment.generated_token_ids.len()),
            );
        mismatch_details.push(format!("generated output IDs/hash mismatch; first unequal/missing index={first}, lengths={}/{}", control.generated_token_ids.len(), treatment.generated_token_ids.len()));
    }
    let normalized_p1e_exact_match = match (&control.normalized_p1e, &treatment.normalized_p1e) {
        (Some(c), Some(t)) => {
            if c != t {
                if c.observations != t.observations {
                    let first = c
                        .observations
                        .iter()
                        .zip(&t.observations)
                        .position(|(a, b)| a != b)
                        .unwrap_or(c.observations.len().min(t.observations.len()));
                    mismatch_details.push(format!("normalized prediction mismatch at index {first}: control={:?}; treatment={:?}", c.observations.get(first), t.observations.get(first)));
                }
                if c.no_emissions != t.no_emissions {
                    mismatch_details.push("normalized no-emission sequence/reason mismatch".into());
                }
                if c.request != t.request
                    || c.partitions != t.partitions
                    || c.incomplete != t.incomplete
                    || c.readiness_measured != t.readiness_measured
                    || c.opportunities != t.opportunities
                {
                    mismatch_details.push(
                        "normalized request/partition/opportunity/completeness mismatch".into(),
                    );
                }
            }
            c == t
        }
        _ => {
            mismatch_details.push("normalized P1E evidence unavailable".into());
            false
        }
    };
    Parity {
        output_exact_match,
        normalized_p1e_exact_match,
        mismatch_details,
    }
}
fn finish_report(report: &mut Report) {
    report.parity = compare(&report.control, &report.treatment);
    let control_errors = report
        .control
        .predictor_v2_snapshot
        .as_ref()
        .map(control_gate)
        .unwrap_or_else(|| vec!["control P0 reconciliation unavailable".into()]);
    let treatment_errors = report
        .treatment
        .predictor_v2_snapshot
        .as_ref()
        .map(treatment_gate)
        .unwrap_or_else(|| vec!["treatment P0 reconciliation unavailable".into()]);
    let mut errors = control_errors.clone();
    errors.extend(treatment_errors.iter().cloned());
    errors.extend(report.parity.mismatch_details.iter().cloned());
    for arm in [&report.control, &report.treatment] {
        if !arm.ordinary_invariants_pass {
            errors.push(format!("{:?} ordinary invariants failed", arm.arm));
        }
        errors.extend(arm.errors.iter().map(|e| format!("{:?}: {e}", arm.arm)));
    }
    report.mechanism = Mechanism {
        control_movement_zero: control_errors.is_empty(),
        treatment_movement_accounting_pass: treatment_errors.is_empty(),
        treatment_direct_matching_demand_credits: report
            .treatment
            .predictor_v2_snapshot
            .as_ref()
            .map(|s| s.direct_matching_demand_credits),
        qualification_failure_reasons: errors,
    };
    report.qualification_pass = report.mechanism.qualification_failure_reasons.is_empty()
        && report.parity.output_exact_match
        && report.parity.normalized_p1e_exact_match;
}

// Both arms must exclude every optional legacy predictive mode. Accept the
// user's configuration only when it already meets this contract; never rewrite it.
fn validate_no_other_predictive_modes(cfg: &crate::config::Config) -> Result<()> {
    let p = &cfg.predictive;
    if cfg.storage.predict_fanout != 0
        || p.locality_enabled
        || p.speculator_enabled
        || p.affinity_enabled
        || p.prefetch_governor
        || p.cost_aware_eviction
        || p.pregate_enabled
        || p.static_residency_fraction != 0.0
        || p.static_residency_warmup_tokens != 0
        || p.static_residency_profile.is_some()
    {
        return Err(
            "P1K requires predict_fanout=0 and all optional predictive modes disabled".into(),
        );
    }
    Ok(())
}

// Revalidate each arm from source bytes and retain its independent provenance.
// The second arm also has to match the first arm's resolved input identity.
fn prepare_arm(args: &CommandArgs) -> Result<PreparedArm> {
    let build = crate::qualification::BuildProvenance::embedded();
    evidence::validate_preflight_provenance(&build)?;
    let bytes = std::fs::read(&args.config)?;
    let config_sha256 = crate::greedy_parity::sha256_hex(&bytes);
    let cfg: crate::config::Config = toml::from_str(std::str::from_utf8(&bytes)?)?;
    cfg.validate()?;
    evidence::validate_source_config(&cfg)?;
    validate_no_other_predictive_modes(&cfg)?;
    let (artifacts, errors) = crate::qualification_artifacts(&args.config, &cfg);
    evidence::validate_artifacts(&artifacts, &errors)?;
    if artifacts.config.as_ref().map(|v| v.sha256.as_str()) != Some(config_sha256.as_str()) {
        return Err("config changed between parsed snapshot and artifact hashing".into());
    }
    let metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|e| format!("expert metadata unavailable: {e}"))?;
    evidence::validate_expert_metadata(&metadata)?;
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err("strict Qwen3-MoE Q4_0 model identity required".into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let provenance = ArmProvenance {
        provenance: evidence::BenchmarkProvenance {
            build,
            executable_canonical_path: std::fs::canonicalize(executable)?.display().to_string(),
            executable_sha256,
            resolved_config_sha256: crate::resolved_real_cli_spec_sha256(&spec)?,
            artifacts,
            expert_metadata: metadata.clone(),
        },
        config_path: std::fs::canonicalize(&args.config)?.display().to_string(),
        config_sha256,
        model_identity,
        production_configuration: evidence::ProductionConfiguration::from_config(
            &spec.cfg, &metadata,
        ),
    };
    Ok(PreparedArm { spec, provenance })
}
fn validate_runtime(
    runtime: &crate::BenchRealRuntime,
    expected_adapter: &str,
    report: &mut ArmReport,
) -> Result<()> {
    let hash = crate::resolved_real_runtime_identity_sha256(
        &runtime.cfg,
        runtime.model.config.architecture,
        runtime.model.config.first_k_dense_replace,
        &runtime.model.config.advanced,
    )?;
    report.runtime_resolved_config_sha256 = Some(hash.clone());
    if report
        .provenance
        .as_ref()
        .map(|p| p.provenance.resolved_config_sha256.as_str())
        != Some(hash.as_str())
    {
        return Err("runtime configuration identity drift".into());
    }
    let model_load = crate::greedy_parity_model_load(runtime);
    let input = evidence::RuntimeContractInput {
        real_transformer_enabled: runtime.cfg.real_transformer.enabled,
        real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
        compute_offload: runtime.cfg.real_transformer.compute_offload,
        legacy_execution_plan: runtime.engine.execution_context().plan().into(),
        token_loop_geometry: runtime
            .gpu_native_token_loop
            .as_ref()
            .map(|t| t.model_geometry()),
        authoritative_device: runtime.engine.gpu_device_identity(),
        model_load: model_load.clone(),
        routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
    };
    report.model_load = Some(model_load);
    report.adapter = input.authoritative_device.clone();
    report.runtime_contract =
        Some(evidence::validate_runtime_contract(&input, expected_adapter)?.0);
    let t = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing token loop")?;
    if t.snapshot() != Default::default()
        || t.recovery_snapshot() != Default::default()
        || runtime.engine.routed_expert_execution_snapshot() != Default::default()
    {
        return Err("isolated runtime has nonzero initial counters".into());
    }
    Ok(())
}

async fn execute_control(
    runtime: &crate::BenchRealRuntime,
    prompt: &[u32],
    output_tokens: usize,
    report: &mut ArmReport,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing token loop")?;
    let config = observation_config(prompt.len(), output_tokens)?;
    report.observation_capacity_per_collection = config.capacity_per_collection;
    let snapshots = evidence::RequestSnapshotStart::capture(runtime)?;
    let mut request = token_loop.create_request_state()?;
    let enable = request
        .enable_predictor_v2_p1e_observation(token_loop, config)
        .map_err(|e| format!("P1E enablement failed: {e:?}"));
    // Even an enable failure follows the normal finalization/evidence path.
    let execution = match enable {
        Ok(()) => step_request(runtime, &mut request, prompt, output_tokens, report).await,
        Err(e) => Err(e.into()),
    };
    finish_request(&mut request, report, execution.is_err());
    finish_snapshots(snapshots, runtime, report);
    execution
}
async fn execute_treatment(
    runtime: &crate::BenchRealRuntime,
    prompt: &[u32],
    output_tokens: usize,
    report: &mut ArmReport,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing token loop")?;
    let config = observation_config(prompt.len(), output_tokens)?;
    report.observation_capacity_per_collection = config.capacity_per_collection;
    let snapshots = evidence::RequestSnapshotStart::capture(runtime)?;
    let mut request = token_loop.create_request_state()?;
    let execution: Result<()> = async {
        request
            .enable_predictor_v2_p1e_observation(token_loop, config)
            .map_err(|e| format!("P1E enablement failed: {e:?}"))?;
        request.enable_predictor_v2_p1j_sidecar(token_loop)?;
        step_request(runtime, &mut request, prompt, output_tokens, report).await
    }
    .await;
    finish_request(&mut request, report, execution.is_err());
    finish_snapshots(snapshots, runtime, report);
    execution
}
async fn step_request(
    runtime: &crate::BenchRealRuntime,
    request: &mut crate::gpu_native_token_loop::GpuNativeRequestState,
    prompt: &[u32],
    output_tokens: usize,
    report: &mut ArmReport,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing token loop")?;
    let planned = add(
        prompt.len(),
        output_tokens.checked_sub(1).ok_or("zero output tokens")?,
    )?;
    if planned > token_loop.max_seq_len() {
        return Err("request exceeds runtime context limit".into());
    }
    let execution: Result<()> = async {
        for position in 0..planned {
            let (token, sample) = if position < prompt.len() {
                (
                    prompt[position],
                    position.checked_add(1) == Some(prompt.len()),
                )
            } else {
                (
                    *report
                        .generated_token_ids
                        .last()
                        .ok_or("missing previous generated token")?,
                    true,
                )
            };
            let sampled = token_loop
                .step_token(&runtime.engine, request, token, position, sample)
                .await?;
            increment(&mut report.completed_positions)?;
            match (sample, sampled) {
                (true, Some(id)) => report.generated_token_ids.push(id),
                (false, None) => {}
                _ => return Err("ordinary step returned an unexpected sampling result".into()),
            }
        }
        Ok(())
    }
    .await;
    execution
}
fn finish_snapshots(
    snapshots: evidence::RequestSnapshotStart,
    runtime: &crate::BenchRealRuntime,
    report: &mut ArmReport,
) {
    match snapshots.finish(runtime) {
        Ok(s) => report.runtime_counters = Some(s),
        Err(e) => report.errors.push(e.to_string()),
    }
}
fn finish_request(
    request: &mut crate::gpu_native_token_loop::GpuNativeRequestState,
    report: &mut ArmReport,
    cancelled: bool,
) {
    request.finish_predictor_v2_observation(cancelled);
    report.raw_p1e_report = request.predictor_v2_p1e_report();
    report.normalized_p1e = report.raw_p1e_report.as_ref().map(normalize);
    report.generated_token_ids_sha256 =
        crate::greedy_parity::token_ids_sha256(&report.generated_token_ids);
    match request.predictor_v2_snapshot() {
        Some(Ok(s)) => report.predictor_v2_snapshot = Some(s),
        Some(Err(e)) => report
            .errors
            .push(format!("P0 reconciliation failed: {e:?}")),
        None => report.errors.push("P0 reconciliation unavailable".into()),
    }
}
fn check_arm(report: &mut ArmReport, prompt_tokens: usize, output_tokens: usize, positions: usize) {
    if report.completed_positions != positions {
        report
            .errors
            .push("completed positions differ from planned positions".into());
    }
    if report.generated_token_ids.len() != output_tokens {
        report
            .errors
            .push("generated output count differs from requested output count".into());
    }
    match &report.raw_p1e_report {
        None => report.errors.push("P1E report unavailable".into()),
        Some(raw) => {
            if raw.incomplete.is_some() {
                report
                    .errors
                    .push(format!("global P1E incomplete: {:?}", raw.incomplete));
            }
            if raw.observations.len().checked_add(raw.no_emissions.len()) != Some(positions) {
                report
                    .errors
                    .push("P1E observations/no-emissions do not cover completed positions".into());
            }
            let mut partitions = p1e::Partitions {
                emitted: raw.observations.len(),
                ..Default::default()
            };
            for (i, r) in raw.observations.iter().enumerate() {
                let c = r.freeze.candidate;
                if c.request != raw.request
                    || c.namespace.runtime != raw.request.runtime_namespace
                    || c.source_layer != p1e::LAYER
                    || c.target_layer != p1e::LAYER
                    || c.signal_revision != p1e::REVISION
                    || c.position_distance != 1
                    || c.source_position.absolute_position.checked_add(1)
                        != Some(c.target_position.absolute_position)
                    || r.deadline
                        .is_some_and(|d| d.request != c.request || d.position != c.target_position)
                {
                    report
                        .errors
                        .push(format!("P1E record {i} identity/causal contract mismatch"));
                }
                match r.outcome {
                    p1e::Outcome::Pending => partitions.pending += 1,
                    p1e::Outcome::Censored => partitions.censored += 1,
                    p1e::Outcome::Resolved { prediction_hit } => {
                        partitions.resolved += 1;
                        if prediction_hit {
                            partitions.prediction_hits += 1;
                        } else {
                            partitions.route_misses += 1;
                        }
                    }
                }
                match r.freeze.current {
                    Some(true) => {
                        partitions.valid_f += 1;
                        partitions.current_at_f += 1;
                    }
                    Some(false) => {
                        partitions.valid_f += 1;
                        partitions.absent_at_f += 1;
                    }
                    None => {}
                }
                partitions.useful += usize::from(r.opportunities().useful == Some(true));
                partitions.target_confirmed_useful +=
                    usize::from(r.opportunities().target_confirmed_useful == Some(true));
                if r.incomplete.is_some()
                    || r.freeze.incomplete.is_some()
                    || r.deadline.is_some_and(|d| d.incomplete.is_some())
                {
                    report.errors.push(format!("P1E record {i} incomplete"));
                }
                if r.outcome == p1e::Outcome::Pending {
                    report
                        .errors
                        .push(format!("P1E record {i} remains pending"));
                }
            }
            if partitions != raw.partitions {
                report
                    .errors
                    .push("P1E partitions do not reconcile with raw records".into());
            }
            if raw
                .no_emissions
                .iter()
                .any(|r| r.reason == p1e::NoEmissionReason::Incomplete)
            {
                report
                    .errors
                    .push("incomplete P1E no-emission record".into());
            }
            if raw.partitions.pending != 0 {
                report.errors.push("P1E pending partition nonzero".into());
            }
            if raw.opportunities
                != raw
                    .observations
                    .iter()
                    .map(|r| (r.freeze.candidate.sequence, r.opportunities()))
                    .collect::<Vec<_>>()
            {
                report
                    .errors
                    .push("P1E opportunities disagree with raw observations".into());
            }
        }
    }
    if let Some(c) = &report.runtime_counters {
        if let Err(e) = evidence::validate_request_postconditions(
            prompt_tokens,
            output_tokens,
            report.generated_token_ids.len(),
            c.token_loop_delta,
            c.recovery_delta,
            c.routed_execution_delta,
        ) {
            report.errors.push(e.to_string());
        }
        let t = c.token_loop_delta;
        if t.queue_submissions != t.token_attempts
            || t.boundary_maps != t.token_attempts
            || t.boundary_readbacks != t.token_attempts
        {
            report
                .errors
                .push("ordinary attempt/submission/map/readback accounting mismatch".into());
        }
    } else {
        report
            .errors
            .push("ordinary runtime snapshots unavailable".into());
    }
    if !shutdown_complete(report) {
        report
            .errors
            .push("normal isolated shutdown evidence missing or incomplete".into());
    }
    report.ordinary_invariants_pass = report.errors.is_empty();
}
fn shutdown_complete(report: &ArmReport) -> bool {
    report
        .runtime_shutdown
        .is_some_and(|s| s.controlled_shutdown_requested && s.all_runtime_resources_released)
}
async fn shutdown(runtime: crate::BenchRealRuntime, report: &mut ArmReport) {
    match runtime.shutdown_isolated().await {
        Ok(s) => report.runtime_shutdown = Some(s),
        Err(e) => report.errors.push(e.to_string()),
    }
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<()> {
    ensure_output_absent(&args.report_out)?;
    if args.expected_adapter_name.trim().is_empty() {
        return Err("exact expected adapter name required".into());
    }
    let (prompt, output_tokens) = parse_request(&std::fs::read(&args.request_json)?)?;
    let control = prepare_arm(&args)?;
    let mode = crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark;
    let tokenizer = crate::load_real_cli_tokenizer(&control.spec.cfg, mode)?;
    // Exactly one tokenization, immutable prompt IDs passed to both arms.
    let prompt_ids = tokenizer.encode(&prompt)?;
    observation_config(prompt_ids.len(), output_tokens)?;
    let positions = add(
        prompt_ids.len(),
        output_tokens.checked_sub(1).ok_or("zero output tokens")?,
    )?;
    if positions > control.spec.cfg.real_transformer.gpu_native_max_seq_len {
        return Err("request exceeds configured context limit".into());
    }
    let expected_provenance = serde_json::to_value(&control.provenance)?;
    let mut report = Report {
        schema: SCHEMA,
        mode: "isolated-sidecar-mechanism-parity-qualification",
        expected_adapter_name: args.expected_adapter_name.clone(),
        request: evidence::RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(prompt.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: output_tokens,
            greedy: true,
        },
        planned_positions: positions,
        frozen_p1j_contract: FrozenContract::default(),
        control: ArmReport::new(Arm::Control),
        treatment: ArmReport::new(Arm::Treatment),
        parity: Parity {
            output_exact_match: false,
            normalized_p1e_exact_match: false,
            mismatch_details: Vec::new(),
        },
        mechanism: Mechanism {
            control_movement_zero: false,
            treatment_movement_accounting_pass: false,
            treatment_direct_matching_demand_credits: None,
            qualification_failure_reasons: Vec::new(),
        },
        performance: PerformanceBoundary,
        qualification_pass: false,
    };
    report.control.provenance = Some(control.provenance);
    report.control.runtime_build_attempted = true;
    match crate::build_isolated_greedy_runtime(&control.spec, mode, tokenizer.clone()).await {
        Err(e) => report.control.errors.push(e.to_string()),
        Ok(runtime) => {
            report.control.runtime_constructed = true;
            let execution: Result<()> = async {
                validate_runtime(&runtime, &args.expected_adapter_name, &mut report.control)?;
                execute_control(&runtime, &prompt_ids, output_tokens, &mut report.control).await
            }
            .await;
            if let Err(e) = execution {
                report.control.errors.push(e.to_string());
            }
            shutdown(runtime, &mut report.control).await;
        }
    }
    check_arm(
        &mut report.control,
        prompt_ids.len(),
        output_tokens,
        positions,
    );
    // A failed shutdown must never overlap a second mutable runtime family.
    // Other control failures do not discard obtainable treatment evidence.
    if shutdown_complete(&report.control) {
        let treatment: Result<()> = async {
            let treatment = prepare_arm(&args)?;
            let observed_provenance = serde_json::to_value(&treatment.provenance)?;
            report.treatment.provenance = Some(treatment.provenance);
            if observed_provenance != expected_provenance {
                return Err("arm input provenance changed".into());
            }
            report.treatment.runtime_build_attempted = true;
            match crate::build_isolated_greedy_runtime(&treatment.spec, mode, tokenizer).await {
                Err(e) => report.treatment.errors.push(e.to_string()),
                Ok(runtime) => {
                    report.treatment.runtime_constructed = true;
                    let execution: Result<()> = async {
                        validate_runtime(
                            &runtime,
                            &args.expected_adapter_name,
                            &mut report.treatment,
                        )?;
                        execute_treatment(
                            &runtime,
                            &prompt_ids,
                            output_tokens,
                            &mut report.treatment,
                        )
                        .await
                    }
                    .await;
                    if let Err(e) = execution {
                        report.treatment.errors.push(e.to_string());
                    }
                    shutdown(runtime, &mut report.treatment).await;
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = treatment {
            report.treatment.errors.push(e.to_string());
        }
    } else {
        report
            .treatment
            .errors
            .push("treatment not constructed: control shutdown not proven complete".into());
    }
    check_arm(
        &mut report.treatment,
        prompt_ids.len(),
        output_tokens,
        positions,
    );
    finish_report(&mut report);
    write_report(&args.report_out, &report)?;
    if !report.qualification_pass {
        return Err(report
            .mechanism
            .qualification_failure_reasons
            .join("; ")
            .into());
    }
    Ok(())
}

#[cfg(test)]
mod p1k_tests {
    use super::*;
    use crate::predictor_v2::{ModelMetadata, PositionIdentity, RequestIdentity};
    use serde_json::json;
    fn identity() -> RequestIdentity {
        RequestIdentity {
            runtime_namespace: 1,
            request_sequence: 1,
            phase: RequestPhase::Measured,
            phase_run_index: 0,
        }
    }
    fn namespace() -> p1e::Namespace {
        p1e::Namespace {
            runtime: 1,
            context: 2,
            arena: 3,
            layer: 47,
            capacity: 8,
        }
    }
    fn model() -> ModelMetadata {
        ModelMetadata {
            num_layers: 48,
            num_experts: 128,
            top_k: 8,
        }
    }
    fn physical(ids: [u32; 8]) -> p1e::PhysicalEvidence {
        p1e::PhysicalEvidence {
            snapshot: p1e::PhysicalSnapshot {
                namespace: namespace(),
                residents: std::array::from_fn(|i| {
                    Some(p1e::Resident {
                        expert: ids[i],
                        generation: 1,
                        bank: 0,
                        slot: i as u32,
                        epoch: 1,
                    })
                }),
            },
            event_cutoff: 3,
            committed_installs: 2,
            physical_victims: 1,
        }
    }
    fn record(sequence: u64, hit: bool, f: Option<bool>, d: Option<bool>) -> p1e::Observation {
        let source = PositionIdentity::from_prompt_length(sequence as usize + 2, 4).unwrap();
        let target = PositionIdentity::from_prompt_length(sequence as usize + 3, 4).unwrap();
        let candidate = p1e::Candidate {
            request: identity(),
            model: model(),
            namespace: namespace(),
            source_position: source,
            target_position: target,
            source_layer: 47,
            target_layer: 47,
            position_distance: 1,
            nominal_layer_lead: 0,
            source_set: [0, 1, 2, 3, 4, 5, 6, 7],
            expert: 8,
            score: 17,
            signal_revision: p1e::REVISION,
            generation: sequence + 3,
            sequence,
            committed_position_cutoff: sequence + 3,
            table_update_cutoff: sequence + 2,
        };
        // Deserialize the complete public report format to retain its private
        // chronology bits, without exposing or changing any P1E implementation.
        serde_json::from_value(json!({
            "freeze": p1e::Freeze { candidate, timestamp_ns: 100,
                physical: Some(physical([0,1,2,3,4,5,6,7])), current: f,
                source: p1e::HostSource { logical_generation: Some(9), logical_materialized: false,
                    ram_resident: false, permanence: p1e::Permanence::Unknown }, incomplete: None },
            "deadline": p1e::Deadline { request: identity(), position: target,
                timestamp_ns: 100 + (sequence + 1) * 10,
                physical: Some(physical([0,1,2,3,4,5,6,7])), current: d,
                host_lead_ns: Some((sequence + 1) * 10), incomplete: None },
            "outcome": p1e::Outcome::Resolved { prediction_hit: hit },
            "recovery": [p1e::RecoveryEvent { attempt: 1, attempted_start: 47, attempted_end: 48,
                first_failure_layer: Some(47), final_status: 1, layer47_demand_service_completed: true }],
            "completion_physical": physical([8,9,10,11,12,13,14,15]),
            "incomplete": null, "initial_attempt_seen": true, "deadline_eligible": false
        })).unwrap()
    }
    fn raw(records: Vec<p1e::Observation>) -> p1e::Report {
        let mut p = p1e::Partitions {
            emitted: records.len(),
            ..Default::default()
        };
        for r in &records {
            match r.outcome {
                p1e::Outcome::Pending => p.pending += 1,
                p1e::Outcome::Censored => p.censored += 1,
                p1e::Outcome::Resolved { prediction_hit } => {
                    p.resolved += 1;
                    if prediction_hit {
                        p.prediction_hits += 1;
                    } else {
                        p.route_misses += 1;
                    }
                }
            }
            match r.freeze.current {
                Some(true) => {
                    p.valid_f += 1;
                    p.current_at_f += 1;
                }
                Some(false) => {
                    p.valid_f += 1;
                    p.absent_at_f += 1;
                }
                None => {}
            }
            let o = r.opportunities();
            p.useful += usize::from(o.useful == Some(true));
            p.target_confirmed_useful += usize::from(o.target_confirmed_useful == Some(true));
        }
        p1e::Report {
            request: identity(),
            opportunities: records
                .iter()
                .map(|r| (r.freeze.candidate.sequence, r.opportunities()))
                .collect(),
            observations: records,
            no_emissions: Vec::new(),
            incomplete: None,
            partitions: p,
            readiness_measured: false,
        }
    }
    fn successful_p0() -> ReconciliationSnapshot {
        ReconciliationSnapshot {
            emitted: 2,
            terminal_predictions: 2,
            accepted: 2,
            source_leaders: 2,
            source_completed: 2,
            reservations: 2,
            reservations_committed: 1,
            reservations_aborted: 1,
            install_owners: 1,
            direct_matching_demand_credits: 1,
            terminal_categories: vec![
                (TerminalReason::ConsumedByMatchingRoute, 1),
                (TerminalReason::RequestEnded, 1),
            ],
            ..Default::default()
        }
    }
    fn arm(arm: Arm) -> ArmReport {
        let mut r = ArmReport::new(arm);
        let raw = raw(vec![record(0, true, Some(false), Some(false))]);
        r.normalized_p1e = Some(normalize(&raw));
        r.raw_p1e_report = Some(raw);
        r.generated_token_ids = vec![11, 22, 33];
        r.generated_token_ids_sha256 =
            crate::greedy_parity::token_ids_sha256(&r.generated_token_ids);
        r.predictor_v2_snapshot = Some(if arm == Arm::Control {
            Default::default()
        } else {
            successful_p0()
        });
        r.ordinary_invariants_pass = true;
        r
    }
    fn report() -> Report {
        Report {
            schema: SCHEMA,
            mode: "isolated-sidecar-mechanism-parity-qualification",
            expected_adapter_name: "fixture".into(),
            request: evidence::RequestEvidence {
                prompt_sha256: "prompt".into(),
                prompt_token_ids_sha256: "tokens".into(),
                prompt_token_count: 3,
                requested_output_tokens: 3,
                greedy: true,
            },
            planned_positions: 5,
            frozen_p1j_contract: Default::default(),
            control: arm(Arm::Control),
            treatment: arm(Arm::Treatment),
            parity: Parity {
                output_exact_match: false,
                normalized_p1e_exact_match: false,
                mismatch_details: Vec::new(),
            },
            mechanism: Mechanism {
                control_movement_zero: false,
                treatment_movement_accounting_pass: false,
                treatment_direct_matching_demand_credits: None,
                qualification_failure_reasons: Vec::new(),
            },
            performance: PerformanceBoundary,
            qualification_pass: false,
        }
    }
    #[test]
    fn strict_request_accepts_only_p1f_neutral_greedy_contract() {
        for input in [
            json!({"prompt":" x ","max_tokens":3,"temperature":0}),
            json!({"prompt":"x","max_tokens":3,"top_k":1,"top_p":1,"n":1,"stream":false,"frequency_penalty":0,"presence_penalty":0,"repetition_penalty":1,"model":"descriptive"}),
            json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}],"max_tokens":3,"top_k":1}),
        ] {
            assert_eq!(
                parse_request(&serde_json::to_vec(&input).unwrap())
                    .unwrap()
                    .1,
                3
            );
        }
    }
    #[test]
    fn strict_request_rejects_ambiguous_empty_unknown_and_non_neutral_inputs() {
        let valid = json!({"prompt":"x","max_tokens":3,"temperature":0});
        for (key, value) in [
            ("max_tokens", json!(0)),
            ("max_tokens", json!(-1)),
            ("max_tokens", json!(1.5)),
            ("prompt", json!(" \n")),
            ("prompt", json!(["x"])),
            ("messages", json!([])),
            ("temperature", json!(0.1)),
            ("temperature", json!(null)),
            ("top_k", json!(2)),
            ("top_p", json!(0.9)),
            ("n", json!(2)),
            ("stream", json!(true)),
            ("frequency_penalty", json!(1)),
            ("presence_penalty", json!(-1)),
            ("repetition_penalty", json!(0)),
            ("seed", json!(1)),
            ("stop", json!(["x"])),
        ] {
            let mut input = valid.clone();
            input[key] = value;
            assert!(
                parse_request(&serde_json::to_vec(&input).unwrap()).is_err(),
                "{input}"
            );
        }
        for messages in [
            json!([]),
            json!([{"role":"tool","content":"x"}]),
            json!([{"role":"user","content":" "}]),
            json!([{"role":"user","content":[]}]),
            json!([{"role":"user","content":"x","name":"n"}]),
        ] {
            assert!(parse_request(
                &serde_json::to_vec(&json!({"messages":messages,"max_tokens":3,"top_k":1}))
                    .unwrap()
            )
            .is_err());
        }
    }
    #[test]
    fn observation_capacity_exact_arithmetic() {
        assert_eq!(
            observation_config(3, 1).unwrap().capacity_per_collection,
            47 * 3 * 64 + 2 * 64
        );
        assert_eq!(
            observation_config(2, 2).unwrap().capacity_per_collection,
            47 * 3 * 64 + 64 + 64
        );
        assert_eq!(
            observation_config(1, 3).unwrap().capacity_per_collection,
            47 * 3 * 64 + 64 + 64
        );
        assert_eq!(
            observation_config(256, 257)
                .unwrap()
                .capacity_per_collection,
            47 * 2 * 16384 + 2 * 255 * 64 + 64
        );
        assert_eq!(
            observation_config(4096, 1).unwrap().capacity_per_collection,
            48 * 16384
        );
    }
    #[test]
    fn observation_ceiling_4096_and_checked_overflow() {
        for (p, o) in [(4096, 1), (1, 4096), (2048, 2049), (3, 1)] {
            assert!(observation_config(p, o).is_ok());
        }
        for (p, o) in [
            (4096, 2),
            (1, 4097),
            (0, 3),
            (3, 0),
            (1, 1),
            (1, 2),
            (usize::MAX, 2),
            (2, usize::MAX),
        ] {
            assert!(observation_config(p, o).is_err());
        }
        assert!(add(usize::MAX, 1).is_err());
        assert!(mul(usize::MAX, 2).is_err());
    }
    #[test]
    fn normalized_namespaces_times_and_logical_generations_compare_equal() {
        let a = raw(vec![record(0, true, Some(false), Some(false))]);
        let mut b = a.clone();
        b.request.runtime_namespace = 100;
        b.request.request_sequence = 9;
        let r = &mut b.observations[0];
        r.freeze.candidate.request = b.request;
        r.freeze.candidate.namespace.runtime = 100;
        r.freeze.candidate.namespace.context = 50;
        r.freeze.candidate.namespace.arena = 88;
        r.freeze.timestamp_ns = 5000;
        r.freeze.source.logical_generation = Some(333);
        let d = r.deadline.as_mut().unwrap();
        d.request = b.request;
        d.timestamp_ns = 9000;
        d.host_lead_ns = Some(4000);
        assert_eq!(normalize(&a), normalize(&b));
        assert_ne!(a, b);
        assert_eq!(normalize(&a).observations[0].candidate.generation, 3);
        assert_eq!(
            normalize(&a).observations[0].candidate.table_update_cutoff,
            2
        );
    }
    fn mismatch(change: impl FnOnce(&mut p1e::Observation)) {
        let mut r = report();
        change(&mut r.treatment.raw_p1e_report.as_mut().unwrap().observations[0]);
        r.treatment.normalized_p1e = r.treatment.raw_p1e_report.as_ref().map(normalize);
        finish_report(&mut r);
        assert!(!r.parity.normalized_p1e_exact_match);
        assert!(!r.qualification_pass);
        assert!(!r.parity.mismatch_details.is_empty());
    }
    #[test]
    fn expert_mismatch_fails() {
        mismatch(|r| r.freeze.candidate.expert += 1);
    }
    #[test]
    fn score_mismatch_fails() {
        mismatch(|r| r.freeze.candidate.score += 1);
    }
    #[test]
    fn source_set_mismatch_fails() {
        mismatch(|r| r.freeze.candidate.source_set[0] = 100);
    }
    #[test]
    fn outcome_mismatch_fails() {
        mismatch(|r| r.outcome = p1e::Outcome::Censored);
    }
    #[test]
    fn prediction_hit_mismatch_fails() {
        mismatch(|r| {
            r.outcome = p1e::Outcome::Resolved {
                prediction_hit: false,
            }
        });
    }
    #[test]
    fn freeze_currentness_mismatch_fails() {
        mismatch(|r| r.freeze.current = Some(true));
        mismatch(|r| r.freeze.current = None);
    }
    #[test]
    fn deadline_currentness_mismatch_fails() {
        mismatch(|r| r.deadline.as_mut().unwrap().current = Some(true));
        mismatch(|r| r.deadline.as_mut().unwrap().current = None);
        mismatch(|r| r.deadline = None);
    }
    #[test]
    fn host_source_mismatch_fails() {
        mismatch(|r| r.freeze.source.logical_materialized = true);
        mismatch(|r| r.freeze.source.ram_resident = true);
        mismatch(|r| r.freeze.source.logical_generation = None);
    }
    #[test]
    fn all_candidate_causal_fields_survive_normalization() {
        mismatch(|r| r.freeze.candidate.sequence += 1);
        mismatch(|r| r.freeze.candidate.source_position.absolute_position += 1);
        mismatch(|r| {
            r.freeze.candidate.target_position.position_kind =
                crate::predictor_v2::PositionKind::Decode
        });
        mismatch(|r| r.freeze.candidate.source_layer -= 1);
        mismatch(|r| r.freeze.candidate.target_layer -= 1);
        mismatch(|r| r.freeze.candidate.committed_position_cutoff += 1);
        mismatch(|r| r.freeze.candidate.table_update_cutoff += 1);
        mismatch(|r| r.freeze.candidate.signal_revision += 1);
        mismatch(|r| r.freeze.candidate.generation += 1);
        mismatch(|r| r.freeze.candidate.request.phase = RequestPhase::Warmup);
        mismatch(|r| r.freeze.candidate.namespace.capacity += 1);
        mismatch(|r| r.incomplete = Some(p1e::Error::Chronology));
        mismatch(|r| r.freeze.incomplete = Some(p1e::Error::PhysicalEvidence));
        mismatch(|r| r.freeze.physical = None);
    }
    #[test]
    fn no_emission_sequence_reason_and_position_mismatch_fails() {
        let mut r = report();
        let n = p1e::NoEmission {
            source: PositionIdentity::from_prompt_length(0, 3).unwrap(),
            target: PositionIdentity::from_prompt_length(1, 3).unwrap(),
            reason: p1e::NoEmissionReason::NoPositiveHistory,
        };
        r.control
            .normalized_p1e
            .as_mut()
            .unwrap()
            .no_emissions
            .push(n);
        r.treatment
            .normalized_p1e
            .as_mut()
            .unwrap()
            .no_emissions
            .push(n);
        assert!(compare(&r.control, &r.treatment).normalized_p1e_exact_match);
        r.treatment.normalized_p1e.as_mut().unwrap().no_emissions[0].reason =
            p1e::NoEmissionReason::Incomplete;
        finish_report(&mut r);
        assert!(!r.qualification_pass);
        r.treatment.normalized_p1e.as_mut().unwrap().no_emissions[0] = n;
        r.treatment.normalized_p1e.as_mut().unwrap().no_emissions[0]
            .source
            .absolute_position = 2;
        assert!(!compare(&r.control, &r.treatment).normalized_p1e_exact_match);
    }
    #[test]
    fn recovery_mechanics_remain_raw_without_becoming_causal_truth() {
        let a = raw(vec![record(0, true, Some(false), Some(false))]);
        let mut b = a.clone();
        b.observations[0].recovery.clear();
        b.observations[0]
            .completion_physical
            .as_mut()
            .unwrap()
            .committed_installs += 1;
        assert_eq!(normalize(&a), normalize(&b));
        assert_ne!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
    }
    #[test]
    fn exact_token_ids_and_hash_mismatch_fails_qualification() {
        let mut r = report();
        finish_report(&mut r);
        assert!(r.qualification_pass);
        r.treatment.generated_token_ids[1] += 1;
        r.treatment.generated_token_ids_sha256 =
            crate::greedy_parity::token_ids_sha256(&r.treatment.generated_token_ids);
        finish_report(&mut r);
        assert!(!r.qualification_pass);
        assert!(!r.parity.output_exact_match);
        let mut r = report();
        r.treatment.generated_token_ids_sha256 = "invalid".into();
        finish_report(&mut r);
        assert!(!r.qualification_pass);
    }
    #[test]
    fn control_emitted_nonzero_fails() {
        let mut p = ReconciliationSnapshot::default();
        p.emitted = 1;
        assert!(!control_gate(&p).is_empty());
    }
    #[test]
    fn control_direct_credit_nonzero_fails() {
        let mut p = ReconciliationSnapshot::default();
        p.direct_matching_demand_credits = 1;
        assert!(!control_gate(&p).is_empty());
    }
    #[test]
    fn control_all_movement_stages_must_be_zero() {
        let mut p = ReconciliationSnapshot::default();
        p.source_completed = 1;
        assert!(!control_gate(&p).is_empty());
        p = Default::default();
        p.incomplete = Some(crate::predictor_v2::AccountingError::Overflow);
        assert!(!control_gate(&p).is_empty());
    }
    #[test]
    fn treatment_emitted_zero_fails() {
        let mut p = successful_p0();
        p.emitted = 0;
        assert!(!treatment_gate(&p).is_empty());
    }
    #[test]
    fn treatment_direct_credit_zero_fails() {
        let mut p = successful_p0();
        p.direct_matching_demand_credits = 0;
        assert!(!treatment_gate(&p).is_empty());
    }
    #[test]
    fn treatment_live_prediction_fails() {
        let mut p = successful_p0();
        p.live_predictions = 1;
        assert!(!treatment_gate(&p).is_empty());
    }
    #[test]
    fn treatment_source_install_failure_fails() {
        for reason in [TerminalReason::SourceFailed, TerminalReason::InstallFailed] {
            let mut p = successful_p0();
            p.terminal_categories.push((reason, 1));
            assert!(!treatment_gate(&p).is_empty());
        }
        let mut p = successful_p0();
        p.source_failed = 1;
        assert!(!treatment_gate(&p).is_empty());
    }
    #[test]
    fn treatment_all_required_reconciliation_fields_fail_closed() {
        let mutations: &[fn(&mut ReconciliationSnapshot)] = &[
            |p| p.incomplete = Some(crate::predictor_v2::AccountingError::Overflow),
            |p| p.terminal_predictions = 1,
            |p| p.accepted = 1,
            |p| p.source_completed = 1,
            |p| p.source_cancelled = 1,
            |p| p.source_live = 1,
            |p| p.reservations_live = 1,
            |p| p.available_installs = 1,
        ];
        for mutate in mutations {
            let mut p = successful_p0();
            mutate(&mut p);
            assert!(!treatment_gate(&p).is_empty());
        }
    }
    #[test]
    fn unused_request_ended_cancelled_evicted_superseded_are_allowed() {
        let mut p = successful_p0();
        for reason in [
            TerminalReason::RequestEnded,
            TerminalReason::Cancelled,
            TerminalReason::EvictedUnused,
            TerminalReason::Superseded,
        ] {
            p.emitted += 1;
            p.accepted += 1;
            p.source_completed += 1;
            p.terminal_predictions += 1;
            p.terminal_categories.push((reason, 1));
        }
        assert!(treatment_gate(&p).is_empty());
    }
    #[test]
    fn performance_boundary_is_permanently_unauthorized() {
        let mut r = report();
        for failed in [false, true] {
            if failed {
                r.treatment.errors.push("fixture failure".into());
            }
            finish_report(&mut r);
            let v = serde_json::to_value(&r).unwrap();
            assert_eq!(v["performance_comparison_authorized"], false);
            assert_eq!(v["resource_footprint_matched"], false);
            assert_eq!(v["performance_verdict"], "NOT_AUTHORIZED");
        }
    }
    #[test]
    fn evidence_publication_is_no_clobber_and_retains_failed_report() {
        let dir = TestDirectory::new();
        let path = dir.path().join("evidence.json");
        let mut r = report();
        r.treatment.errors.push("fixture runtime failure".into());
        finish_report(&mut r);
        write_report(&path, &r).unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(
            !serde_json::from_slice::<serde_json::Value>(&before).unwrap()["qualification_pass"]
                .as_bool()
                .unwrap()
        );
        assert!(write_report(&path, &json!({"replacement":true})).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn evidence_publication_refuses_temp_collision_and_dangling_symlink() {
        let dir = TestDirectory::new();
        let path = dir.path().join("evidence.json");
        let temp = dir
            .path()
            .join(format!(".evidence.json.{}.tmp", std::process::id()));
        std::fs::write(&temp, b"older evidence").unwrap();
        assert!(write_report(&path, &json!({})).is_err());
        assert!(!path.exists());
        assert_eq!(std::fs::read(&temp).unwrap(), b"older evidence");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path().join("absent"), &path).unwrap();
            assert!(ensure_output_absent(&path).is_err());
            assert!(write_report(&path, &json!({})).is_err());
        }
    }
    #[test]
    fn publication_race_cannot_replace_new_evidence() {
        struct Competitor<'a>(&'a Path);
        impl Serialize for Competitor<'_> {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                std::fs::write(self.0, b"concurrent evidence").unwrap();
                serializer.serialize_bool(false)
            }
        }
        let dir = TestDirectory::new();
        let path = dir.path().join("race.json");
        assert!(write_report(&path, &Competitor(&path)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"concurrent evidence");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn ordinary_incomplete_pending_and_coverage_fail_closed() {
        let mut a = arm(Arm::Control);
        a.completed_positions = 1;
        a.raw_p1e_report.as_mut().unwrap().observations[0].outcome = p1e::Outcome::Pending;
        a.raw_p1e_report.as_mut().unwrap().observations[0]
            .freeze
            .incomplete = Some(p1e::Error::PhysicalEvidence);
        a.raw_p1e_report.as_mut().unwrap().incomplete = Some(p1e::Error::Incomplete);
        check_arm(&mut a, 3, 3, 5);
        assert!(!a.ordinary_invariants_pass);
        for text in [
            "completed positions",
            "global P1E incomplete",
            "remains pending",
            "record 0 incomplete",
            "partitions do not reconcile",
            "snapshots unavailable",
            "shutdown evidence missing",
        ] {
            assert!(
                a.errors.iter().any(|e| e.contains(text)),
                "missing {text}: {:?}",
                a.errors
            );
        }
    }
    #[test]
    fn shutdown_gate_requires_both_controlled_and_released() {
        let mut a = arm(Arm::Control);
        assert!(!shutdown_complete(&a));
        for controlled in [false, true] {
            for released in [false, true] {
                a.runtime_shutdown = Some(crate::greedy_parity::BackgroundShutdownEvidence {
                    controlled_shutdown_requested: controlled,
                    all_runtime_resources_released: released,
                    poll_iterations: 1,
                });
                assert_eq!(shutdown_complete(&a), controlled && released);
            }
        }
    }
    #[test]
    fn source_config_rejects_other_predictive_modes_without_mutating_config() {
        let mut cfg: crate::config::Config =
            toml::from_str(include_str!("../../config.toml")).unwrap();
        cfg.storage.predict_fanout = 0;
        validate_no_other_predictive_modes(&cfg).unwrap();
        let changes: &[fn(&mut crate::config::Config)] = &[
            |c| c.storage.predict_fanout = 1,
            |c| c.predictive.locality_enabled = true,
            |c| c.predictive.speculator_enabled = true,
            |c| c.predictive.affinity_enabled = true,
            |c| c.predictive.prefetch_governor = true,
            |c| c.predictive.cost_aware_eviction = true,
            |c| c.predictive.pregate_enabled = true,
            |c| c.predictive.static_residency_fraction = 0.1,
            |c| c.predictive.static_residency_warmup_tokens = 1,
            |c| c.predictive.static_residency_profile = Some("fixture".into()),
        ];
        for change in changes {
            let mut bad = cfg.clone();
            change(&mut bad);
            assert!(validate_no_other_predictive_modes(&bad).is_err());
        }
        validate_no_other_predictive_modes(&cfg).unwrap();
    }
    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "mer-p1k-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn exact_four_required_cli_arguments_without_application_execution() {
        use clap::Parser;
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: CommandArgs,
        }
        let args = [
            "fixture",
            "--config",
            "config.toml",
            "--request-json",
            "request.json",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "new.json",
        ];
        assert!(Cli::try_parse_from(args).is_ok());
        for i in [1, 3, 5, 7] {
            let mut missing = args.to_vec();
            missing.drain(i..i + 2);
            assert!(Cli::try_parse_from(missing).is_err());
        }
        for flag in [
            "--layer",
            "--score-threshold",
            "--fanout",
            "--sidecar-size",
            "--serve",
            "--performance",
        ] {
            let mut extra = args.to_vec();
            extra.push(flag);
            extra.push("1");
            assert!(Cli::try_parse_from(extra).is_err());
        }
    }
    fn production() -> &'static str {
        include_str!("gpu_native_predictor_v2_sidecar_qualifier.rs")
            .split("#[cfg(test)]\nmod p1k_tests")
            .next()
            .unwrap()
    }
    fn section<'a>(s: &'a str, start: &str, end: &str) -> &'a str {
        s.split(start).nth(1).unwrap().split(end).next().unwrap()
    }
    #[test]
    fn control_source_has_no_sidecar_enable_or_allocation() {
        let s = section(
            production(),
            "async fn execute_control(",
            "async fn execute_treatment(",
        );
        assert!(s.contains("enable_predictor_v2_p1e_observation("));
        assert!(!s.contains("p1j"));
        assert!(!s.contains("sidecar"));
        assert!(!s.contains("create_buffer"));
    }
    #[test]
    fn treatment_source_enables_p1e_then_p1j_before_ordinary_steps() {
        let p = production();
        assert_eq!(p.matches(".enable_predictor_v2_p1j_sidecar(").count(), 1);
        let s = section(p, "async fn execute_treatment(", "async fn step_request(");
        assert!(
            s.find("enable_predictor_v2_p1e_observation(").unwrap()
                < s.find("enable_predictor_v2_p1j_sidecar(").unwrap()
        );
        assert!(
            s.find("enable_predictor_v2_p1j_sidecar(").unwrap() < s.find("step_request(").unwrap()
        );
    }
    #[test]
    fn exactly_two_isolated_builds_and_control_shutdown_precedes_treatment() {
        let p = production();
        let s = p.split("pub(crate) async fn run_command(").nth(1).unwrap();
        assert_eq!(
            p.matches("crate::build_isolated_greedy_runtime(").count(),
            2
        );
        assert!(s.contains("RealCliRuntimeMode::IsolatedGpuNativeBenchmark"));
        let first = s.find("crate::build_isolated_greedy_runtime(").unwrap();
        let shut = s
            .find("shutdown(runtime, &mut report.control).await")
            .unwrap();
        let gate = s.find("if shutdown_complete(&report.control)").unwrap();
        let second = s.rfind("crate::build_isolated_greedy_runtime(").unwrap();
        assert!(first < shut && shut < gate && gate < second);
        assert_eq!(p.matches("tokenizer.encode(").count(), 1);
        assert_eq!(p.matches("token_loop.create_request_state()").count(), 2);
        let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(compact.contains("execute_control(&runtime,&prompt_ids,output_tokens"));
        assert!(compact.contains("execute_treatment(&runtime,&prompt_ids,output_tokens"));
        assert!(!s.contains("loop {"));
    }
    #[test]
    fn only_ordinary_step_token_and_no_runtime_in_tests() {
        let p = production();
        assert_eq!(p.matches(".step_token(").count(), 1);
        for forbidden in [
            "step_token_oracle",
            "step_token_diagnostic",
            "enable_q4_route_parallel_qualification",
            "queue.submit",
            "device.poll",
            "warmup_runs",
            "RunTiming",
            "decode_tps",
        ] {
            assert!(!p.contains(forbidden), "{forbidden}");
        }
    }
    #[test]
    fn no_serving_config_environment_or_direct_main_activation() {
        for s in [
            include_str!("server.rs"),
            include_str!("gpu_native_predictor_v2_observation.rs"),
            include_str!("main.rs"),
        ] {
            assert!(!s.contains("enable_predictor_v2_p1j_sidecar"));
            assert!(!s.contains("enable_p1j_sidecar"));
        }
        for token in [
            "std::env",
            "env!(",
            "option_env!",
            "create_buffer",
            "enable_p1j_sidecar(",
        ] {
            assert!(!production().contains(token));
        }
        let main = include_str!("main.rs");
        assert_eq!(
            main.matches("gpu_native_predictor_v2_sidecar_qualifier::run_command")
                .count(),
            1
        );
        let command = section(
            production(),
            "pub(crate) struct CommandArgs {",
            "// Reject unsupported",
        );
        assert_eq!(command.matches("#[arg(long)]").count(), 4);
    }
    #[test]
    fn source_failure_publication_happens_before_nonzero_return() {
        let s = production()
            .split("pub(crate) async fn run_command(")
            .nth(1)
            .unwrap();
        assert!(s.find("ensure_output_absent(").unwrap() < s.find("prepare_arm(").unwrap());
        assert!(
            s.find("write_report(&args.report_out, &report)?").unwrap()
                < s.find("if !report.qualification_pass").unwrap()
        );
        let write = section(production(), "fn write_report(", "// The normalized");
        for (a, b) in [
            ("file.flush()", "file.sync_all()"),
            ("file.sync_all()", "std::fs::hard_link("),
            ("std::fs::hard_link(", "std::fs::remove_file("),
            (
                "std::fs::remove_file(",
                "std::fs::File::open(parent)?.sync_all()",
            ),
        ] {
            assert!(write.find(a).unwrap() < write.find(b).unwrap());
        }
    }
}
