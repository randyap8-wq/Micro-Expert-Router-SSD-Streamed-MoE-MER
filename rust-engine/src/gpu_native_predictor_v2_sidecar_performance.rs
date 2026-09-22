//! P1Q matched-resource performance harness. Runtime requires separate authority.
//! Independent P1O v3 certification; fixed six-pair methodology from issue #199.

use crate::gpu_native_real_benchmark as evidence;
use crate::gpu_native_token_loop::{P1jLaunchSnapshot, P1jRequestMode, P1qInitialSnapshot};
use crate::predictor_v2::{
    p1e, ObservationConfig, ReconciliationSnapshot, RequestPhase, TerminalReason,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "mer.predictor-v2-p1q-matched-resource-performance.v1";
const MAX_POSITIONS: usize = 4096;
const PAIRS: usize = 6;
const OUTPUT_TOKENS: usize = 128;
const PLANNED_POSITIONS: usize = 143;

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
    #[arg(long, value_enum)]
    experiment_mode: ExperimentMode,
    #[arg(long)]
    noise_calibration_report: Option<PathBuf>,
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
    if input.max_tokens != OUTPUT_TOKENS
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
        return Err("requires max_tokens=128 and explicit unmodified greedy sampling (temperature=0 or top_k=1)".into());
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

// The normalized semantic trace retains the entire candidate except for the
// five runtime-local identity fields. Mechanical evidence is compared separately;
// timestamps, recovery events and physical snapshots remain lossless in raw P1E.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SemanticObservation {
    candidate: p1e::Candidate,
    freeze_incomplete: Option<p1e::Error>,
    deadline: Option<SemanticDeadline>,
    outcome: p1e::Outcome,
    incomplete: Option<p1e::Error>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SemanticDeadline {
    request: crate::predictor_v2::RequestIdentity,
    position: crate::predictor_v2::PositionIdentity,
    incomplete: Option<p1e::Error>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SemanticRoute {
    request: crate::predictor_v2::RequestIdentity,
    observations: Vec<SemanticObservation>,
    prediction_hits_by_sequence: Vec<(u64, Option<bool>)>,
    no_emissions: Vec<p1e::NoEmission>,
    incomplete: Option<p1e::Error>,
    partitions: RoutePartitions,
    readiness_measured: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RoutePartitions {
    emitted: usize,
    resolved: usize,
    pending: usize,
    censored: usize,
    prediction_hits: usize,
    route_misses: usize,
}
fn normalize_request(
    mut id: crate::predictor_v2::RequestIdentity,
) -> crate::predictor_v2::RequestIdentity {
    id.runtime_namespace = 0;
    id.request_sequence = 0;
    id
}
fn semantic_route(raw: &p1e::Report) -> SemanticRoute {
    SemanticRoute {
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
                SemanticObservation {
                    candidate,
                    freeze_incomplete: r.freeze.incomplete,
                    deadline: r.deadline.map(|d| SemanticDeadline {
                        request: normalize_request(d.request),
                        position: d.position,
                        incomplete: d.incomplete,
                    }),
                    outcome: r.outcome,
                    incomplete: r.incomplete,
                }
            })
            .collect(),
        prediction_hits_by_sequence: raw
            .opportunities
            .iter()
            .map(|(sequence, o)| (*sequence, o.prediction_hit))
            .collect(),
        no_emissions: raw.no_emissions.clone(),
        incomplete: raw.incomplete,
        partitions: RoutePartitions {
            emitted: raw.partitions.emitted,
            resolved: raw.partitions.resolved,
            pending: raw.partitions.pending,
            censored: raw.partitions.censored,
            prediction_hits: raw.partitions.prediction_hits,
            route_misses: raw.partitions.route_misses,
        },
        readiness_measured: raw.readiness_measured,
    }
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct EvidenceStructure {
    sequence: u64,
    f_physical_evidence_present: bool,
    deadline_present: bool,
    // None distinguishes no deadline from a deadline with no physical evidence.
    d_physical_evidence_present: Option<bool>,
}
fn evidence_structure(raw: &p1e::Report) -> Vec<EvidenceStructure> {
    raw.observations
        .iter()
        .map(|r| EvidenceStructure {
            sequence: r.freeze.candidate.sequence,
            f_physical_evidence_present: r.freeze.physical.is_some(),
            deadline_present: r.deadline.is_some(),
            d_physical_evidence_present: r.deadline.map(|d| d.physical.is_some()),
        })
        .collect()
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct PhysicalOpportunities {
    physical_miss_at_f: Option<bool>,
    physical_miss_at_d: Option<bool>,
    useful: Option<bool>,
    target_confirmed_useful: Option<bool>,
    already_resident: Option<bool>,
    redundant_route_hit: Option<bool>,
}
impl From<p1e::Opportunities> for PhysicalOpportunities {
    fn from(o: p1e::Opportunities) -> Self {
        Self {
            physical_miss_at_f: o.physical_miss_at_f,
            physical_miss_at_d: o.physical_miss_at_d,
            useful: o.useful,
            target_confirmed_useful: o.target_confirmed_useful,
            already_resident: o.already_resident,
            redundant_route_hit: o.redundant_route_hit,
        }
    }
}
#[derive(Debug, Eq, PartialEq, Serialize)]
struct MechanicalObservation {
    sequence: u64,
    f_current: Option<bool>,
    f_source_logical_generation_present: bool,
    f_source_logical_materialized: bool,
    f_source_ram_resident: bool,
    f_source_permanence: p1e::Permanence,
    d_current: Option<bool>,
    #[serde(flatten)]
    opportunities: PhysicalOpportunities,
}
#[derive(Debug, Eq, PartialEq, Serialize)]
struct PhysicalPartitions {
    valid_f: usize,
    current_at_f: usize,
    absent_at_f: usize,
    useful: usize,
    target_confirmed_useful: usize,
}
#[derive(Debug, Eq, PartialEq, Serialize)]
struct MechanicalState {
    observations: Vec<MechanicalObservation>,
    // Keep the reported rows independently, including order and duplicate or
    // missing sequences. Do not replace reported evidence with recomputed rows.
    reported_opportunities: Vec<(u64, PhysicalOpportunities)>,
    physical_partitions: PhysicalPartitions,
}
fn mechanical_state(raw: &p1e::Report) -> MechanicalState {
    MechanicalState {
        observations: raw
            .observations
            .iter()
            .map(|r| MechanicalObservation {
                sequence: r.freeze.candidate.sequence,
                f_current: r.freeze.current,
                f_source_logical_generation_present: r.freeze.source.logical_generation.is_some(),
                f_source_logical_materialized: r.freeze.source.logical_materialized,
                f_source_ram_resident: r.freeze.source.ram_resident,
                f_source_permanence: r.freeze.source.permanence,
                d_current: r.deadline.and_then(|d| d.current),
                opportunities: r.opportunities().into(),
            })
            .collect(),
        reported_opportunities: raw
            .opportunities
            .iter()
            .map(|(s, o)| (*s, (*o).into()))
            .collect(),
        physical_partitions: PhysicalPartitions {
            valid_f: raw.partitions.valid_f,
            current_at_f: raw.partitions.current_at_f,
            absent_at_f: raw.partitions.absent_at_f,
            useful: raw.partitions.useful,
            target_confirmed_useful: raw.partitions.target_confirmed_useful,
        },
    }
}

#[derive(Debug, Serialize)]
struct ExactParity<T> {
    exact_match: bool,
    control: Option<T>,
    treatment: Option<T>,
}
impl<T> Default for ExactParity<T> {
    fn default() -> Self {
        Self {
            exact_match: false,
            control: None,
            treatment: None,
        }
    }
}
fn exact_parity<T: Eq>(control: Option<T>, treatment: Option<T>) -> ExactParity<T> {
    // Even two missing reports cannot certify parity.
    let exact_match = matches!((&control, &treatment), (Some(c), Some(t)) if c == t);
    ExactParity {
        exact_match,
        control,
        treatment,
    }
}

#[derive(Debug, Default, Serialize)]
struct MechanicalDiff {
    divergent_observation_count: usize,
    divergent_sequences: Vec<u64>,
    field_divergence_counts: std::collections::BTreeMap<&'static str, usize>,
    divergent_reported_opportunity_count: usize,
    divergent_reported_opportunity_sequences: Vec<u64>,
    reported_opportunity_field_divergence_counts: std::collections::BTreeMap<&'static str, usize>,
}
#[derive(Debug, Default, Serialize)]
struct MechanicalStateComparison {
    control: Option<MechanicalState>,
    treatment: Option<MechanicalState>,
    // Unavailable evidence is not reported as zero divergence.
    diff: Option<MechanicalDiff>,
}

fn compare_mechanical(control: &ArmReport, treatment: &ArmReport) -> MechanicalStateComparison {
    let control = control.raw_p1e_report.as_ref().map(mechanical_state);
    let treatment = treatment.raw_p1e_report.as_ref().map(mechanical_state);
    let diff = match (&control, &treatment) {
        (Some(c), Some(t)) => {
            let mut diff = MechanicalDiff::default();
            // Pair ordered observations, without a lossy map keyed by sequence.
            // Sequence/length disagreement independently fails semantic parity.
            macro_rules! fields {
                ($counts:expr, $c:ident, $t:ident, $($name:literal => $($field:ident).+),+ $(,)?) => {{
                    let mut divergent = false;
                    $(let unequal = $c.map(|r| &r.$($field).+) != $t.map(|r| &r.$($field).+);
                    *$counts.entry($name).or_default() += usize::from(unequal);
                    divergent |= unequal;)+
                    divergent
                }};
            }
            for i in 0..c.observations.len().max(t.observations.len()) {
                let c = c.observations.get(i);
                let t = t.observations.get(i);
                if fields!(diff.field_divergence_counts, c, t,
                    "sequence" => sequence,
                    "f_current" => f_current,
                    "f_source_logical_generation_present" => f_source_logical_generation_present,
                    "f_source_logical_materialized" => f_source_logical_materialized,
                    "f_source_ram_resident" => f_source_ram_resident,
                    "f_source_permanence" => f_source_permanence,
                    "d_current" => d_current,
                    "physical_miss_at_f" => opportunities.physical_miss_at_f,
                    "physical_miss_at_d" => opportunities.physical_miss_at_d,
                    "useful" => opportunities.useful,
                    "target_confirmed_useful" => opportunities.target_confirmed_useful,
                    "already_resident" => opportunities.already_resident,
                    "redundant_route_hit" => opportunities.redundant_route_hit,
                ) {
                    diff.divergent_observation_count += 1;
                    diff.divergent_sequences
                        .extend(c.into_iter().chain(t).map(|r| r.sequence));
                }
            }
            for i in 0..c
                .reported_opportunities
                .len()
                .max(t.reported_opportunities.len())
            {
                let cr = c.reported_opportunities.get(i);
                let tr = t.reported_opportunities.get(i);
                let c = cr.map(|r| &r.1);
                let t = tr.map(|r| &r.1);
                let fields_differ = fields!(diff.reported_opportunity_field_divergence_counts, c, t,
                    "physical_miss_at_f" => physical_miss_at_f,
                    "physical_miss_at_d" => physical_miss_at_d,
                    "useful" => useful,
                    "target_confirmed_useful" => target_confirmed_useful,
                    "already_resident" => already_resident,
                    "redundant_route_hit" => redundant_route_hit,
                );
                if fields_differ || cr.map(|r| r.0) != tr.map(|r| r.0) {
                    diff.divergent_reported_opportunity_count += 1;
                    diff.divergent_reported_opportunity_sequences
                        .extend(cr.into_iter().chain(tr).map(|r| r.0));
                }
            }
            diff.divergent_sequences.sort_unstable();
            diff.divergent_sequences.dedup();
            diff.divergent_reported_opportunity_sequences
                .sort_unstable();
            diff.divergent_reported_opportunity_sequences.dedup();
            Some(diff)
        }
        _ => None,
    };
    MechanicalStateComparison {
        control,
        treatment,
        diff,
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
    predictor_v2_snapshot: Option<ReconciliationSnapshot>,
    p1j_launch_before: Option<P1jLaunchSnapshot>,
    p1j_launch_after: Option<P1jLaunchSnapshot>,
    p1j_launch_delta: Option<P1jLaunchSnapshot>,
    runtime_counters: Option<evidence::RequestSnapshots>,
    runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
    errors: Vec<String>,
    ordinary_invariants_pass: bool,
    mode: P1jRequestMode,
    resources_before_opt_in: Option<crate::gpu_native_residency::P1qResourceSnapshot>,
    initial: Option<P1qInitialSnapshot>,
    resource_identity: Option<Value>,
    sidecar_freshly_initialized: bool,
    initial_state_pass: bool,
    request_wall_ns: Option<u64>,
    generated_tps: Option<f64>,
    planned_position_tps: Option<f64>,
    production_install_after: Option<Value>,
    production_install_delta: Option<Value>,
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
            predictor_v2_snapshot: None,
            p1j_launch_before: None,
            p1j_launch_after: None,
            p1j_launch_delta: None,
            runtime_counters: None,
            runtime_shutdown: None,
            errors: Vec::new(),
            ordinary_invariants_pass: false,
            mode: P1jRequestMode::InertResourceOnly,
            resources_before_opt_in: None,
            initial: None,
            resource_identity: None,
            sidecar_freshly_initialized: false,
            initial_state_pass: false,
            request_wall_ns: None,
            generated_tps: None,
            planned_position_tps: None,
            production_install_after: None,
            production_install_delta: None,
        }
    }
}
#[derive(Debug, Default, Serialize)]
struct Parity {
    output_exact_match: bool,
    semantic_route_parity: ExactParity<SemanticRoute>,
    evidence_structure_parity: ExactParity<Vec<EvidenceStructure>>,
    mismatch_details: Vec<String>,
}
#[derive(Debug, Serialize)]
struct Mechanism {
    control_movement_zero: bool,
    treatment_movement_accounting_pass: bool,
    treatment_direct_matching_demand_credits: Option<u64>,
    qualification_failure_reasons: Vec<String>,
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
    let semantic_route_parity = exact_parity(
        control.raw_p1e_report.as_ref().map(semantic_route),
        treatment.raw_p1e_report.as_ref().map(semantic_route),
    );
    if !semantic_route_parity.exact_match {
        mismatch_details.push(
            "semantic route evidence unequal or unavailable; see both semantic_route_parity arms"
                .into(),
        );
    }
    let evidence_structure_parity = exact_parity(
        control.raw_p1e_report.as_ref().map(evidence_structure),
        treatment.raw_p1e_report.as_ref().map(evidence_structure),
    );
    if !evidence_structure_parity.exact_match {
        mismatch_details.push("F/deadline/D evidence structure unequal or unavailable; see both evidence_structure_parity arms".into());
    }
    Parity {
        output_exact_match,
        semantic_route_parity,
        evidence_structure_parity,
        mismatch_details,
    }
}
// v2 requires auditable before/after/delta diagnostics. These checks can only
// add failures; the original parity/P0/mechanism gates remain authoritative.
fn launch_diagnostics_gate(report: &ArmReport) -> Vec<String> {
    let mut errors = Vec::new();
    match (
        report.p1j_launch_before,
        report.p1j_launch_after,
        report.p1j_launch_delta,
    ) {
        (Some(before), Some(after), Some(delta)) => {
            if after.checked_delta(before) != Some(delta)
                || !before.reconciled()
                || !after.reconciled()
                || !delta.reconciled()
            {
                errors.push(format!(
                    "{:?} P1J launch diagnostics incomplete or unreconciled",
                    report.arm
                ));
            }
            if report.mode == P1jRequestMode::InertResourceOnly
                && [before, after, delta]
                    .iter()
                    .any(|s| *s != P1jLaunchSnapshot::default())
            {
                errors.push("control P1J launch diagnostics must be entirely zero".into());
            }
        }
        _ => errors.push(format!(
            "{:?} P1J launch diagnostics unavailable",
            report.arm
        )),
    }
    errors
}

fn finish_launch_diagnostics(report: &mut ArmReport, after: P1jLaunchSnapshot) {
    report.p1j_launch_after = Some(after);
    report.p1j_launch_delta = report
        .p1j_launch_before
        .and_then(|before| after.checked_delta(before));
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
            "P1Q requires predict_fanout=0 and all optional predictive modes disabled".into(),
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

fn finish_snapshots(
    snapshots: evidence::RequestSnapshotStart,
    runtime: &crate::BenchRealRuntime,
    report: &mut ArmReport,
) {
    if let Some(token_loop) = runtime.gpu_native_token_loop.as_ref() {
        finish_launch_diagnostics(report, token_loop.p1j_launch_snapshot());
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ExperimentMode {
    AaNoiseCalibration,
    AbMovement,
}
impl ExperimentMode {
    fn request_mode(self, arm: Arm) -> P1jRequestMode {
        if self == Self::AbMovement && arm == Arm::Treatment {
            P1jRequestMode::Active
        } else {
            P1jRequestMode::InertResourceOnly
        }
    }
    fn labels(self) -> [&'static str; 2] {
        match self {
            Self::AaNoiseCalibration => ["A", "B"],
            Self::AbMovement => ["C", "T"],
        }
    }
}
fn execution_order(pair: usize) -> [Arm; 2] {
    if pair % 2 == 0 {
        [Arm::Control, Arm::Treatment]
    } else {
        [Arm::Treatment, Arm::Control]
    }
}
fn validate_args(args: &CommandArgs) -> Result<()> {
    if args.expected_adapter_name != "NVIDIA L4" {
        return Err("P1Q requires expected-adapter-name NVIDIA L4".into());
    }
    if (args.experiment_mode == ExperimentMode::AbMovement)
        != args.noise_calibration_report.is_some()
    {
        return Err("noise-calibration-report is required only for ab-movement".into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct NoiseStatistics {
    median_delta_pct: f64,
    mad_delta_pct: f64,
    max_abs_delta_pct: f64,
    positive: usize,
    negative: usize,
    tied: usize,
    min_delta_pct: f64,
    max_delta_pct: f64,
}
fn median_six(mut values: [f64; PAIRS]) -> Result<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err("non-finite pair metric".into());
    }
    values.sort_by(f64::total_cmp);
    Ok(values[2] / 2.0 + values[3] / 2.0)
}
fn noise_statistics(values: [f64; PAIRS]) -> Result<NoiseStatistics> {
    let median = median_six(values)?;
    Ok(NoiseStatistics {
        median_delta_pct: median,
        mad_delta_pct: median_six(values.map(|v| (v - median).abs()))?,
        max_abs_delta_pct: values.iter().map(|v| v.abs()).fold(0.0, f64::max),
        positive: values.iter().filter(|v| **v > 0.0).count(),
        negative: values.iter().filter(|v| **v < 0.0).count(),
        tied: values.iter().filter(|v| **v == 0.0).count(),
        min_delta_pct: values.iter().copied().fold(f64::INFINITY, f64::min),
        max_delta_pct: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}
fn noise_stable(s: &NoiseStatistics) -> bool {
    s.median_delta_pct.abs() <= 1.0 && s.mad_delta_pct <= 1.0 && s.max_abs_delta_pct <= 3.0
}
fn noise_floor(s: &NoiseStatistics) -> f64 {
    2.0_f64.max(3.0 * s.mad_delta_pct)
}
fn ab_verdict(s: &NoiseStatistics, floor: f64) -> &'static str {
    if s.median_delta_pct > floor && s.positive >= 5 {
        "WIN"
    } else if s.median_delta_pct < -floor && s.negative >= 5 {
        "LOSS"
    } else {
        "INCONCLUSIVE"
    }
}
fn throughput(wall_ns: u64) -> Result<(f64, f64)> {
    if wall_ns == 0 {
        return Err("zero request wall time".into());
    }
    Ok((128e9 / wall_ns as f64, 143e9 / wall_ns as f64))
}
fn paired_delta(a_ns: u64, b_ns: u64) -> Result<f64> {
    let (a, _) = throughput(a_ns)?;
    let (b, _) = throughput(b_ns)?;
    Ok(100.0 * (b / a - 1.0))
}

// Full embedded source bytes and clean build/executable/artifact provenance
// bind calibration to the very same implementation, independently of path names.
fn source_identity() -> Value {
    json!({
        "residency": crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_residency.rs")),
        "token_loop": crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_token_loop.rs")),
        "performance": crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_predictor_v2_sidecar_performance.rs")),
        "main": crate::greedy_parity::sha256_hex(include_bytes!("main.rs")),
        "observation": crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_predictor_v2_observation.rs")),
        "q4_route": crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_q4_route_parallel.rs")),
        "p1q0_qualifier": crate::greedy_parity::sha256_hex(include_bytes!("gpu_native_predictor_v2_sidecar_qualifier.rs")),
    })
}

// Only three runtime-local namespace identities and the declared mode label
// are omitted. Geometry, counters, model/config/request/adapter values remain.
fn structural_initial(initial: &P1qInitialSnapshot) -> Result<Value> {
    let mut value = serde_json::to_value(initial)?;
    value
        .as_object_mut()
        .ok_or("invalid initial snapshot")?
        .remove("mode");
    for key in ["runtime", "context", "arena"] {
        let id = value
            .pointer_mut(&format!("/resources/namespace/{key}"))
            .ok_or("missing sidecar namespace")?;
        *id = json!(0);
    }
    Ok(value)
}

// Fresh runtimes have distinct legacy context incarnations. Normalize only
// that ID for resource/calibration equality; retain the raw ArmReport evidence.
fn structural_runtime_contract(raw: &evidence::RuntimeContractEvidence) -> Result<Value> {
    let mut value = serde_json::to_value(raw)?;
    *value
        .pointer_mut("/legacy_execution_plan/context_id")
        .ok_or("missing legacy execution context ID")? = json!("runtime-local");
    Ok(value)
}

fn initial_state_valid(
    before: &crate::gpu_native_residency::P1qResourceSnapshot,
    s: &P1qInitialSnapshot,
) -> bool {
    let r = &s.resources;
    !before.sidecar_initialized
        && !before.sidecar_allocated
        && before.namespace.is_none()
        && r.sidecar_initialized
        && r.sidecar_allocated
        && r.namespace.is_some()
        && r.sidecar_stride_bytes == 2_654_212
        && r.sidecar_bank == 1
        && r.sidecar_slot == 0
        && r.ordinary_total_expert_budget_bytes == 2 * 1024 * 1024 * 1024
        && r.ordinary_layer_capacities.len() == 48
        && r.ordinary_layer_capacities.get(47) == Some(&8)
        && r.ordinary_layer_resident_counts == vec![0; 48]
        && before.ordinary_layer_resident_counts == r.ordinary_layer_resident_counts
        && before.ordinary_layer_capacities == r.ordinary_layer_capacities
        && before.ordinary_total_expert_budget_bytes == r.ordinary_total_expert_budget_bytes
        && before.ordinary_arena_allocation_bytes == r.ordinary_arena_allocation_bytes
        && r.p1e_shadow_present
        && !before.p1e_shadow_present
        && before.activity_counters == [0; 14]
        && r.activity_counters == [0; 14]
        && s.token_loop == Default::default()
        && s.recovery == Default::default()
        && s.production_install == Default::default()
        && s.launch == Default::default()
        && s.p0 == Default::default()
        && s.committed_position == 0
        && !s.pending_sidecar
        && s.observation_capacity
            == observation_config(16, OUTPUT_TOKENS)
                .map(|c| c.capacity_per_collection)
                .unwrap_or(0)
}
fn counter_delta(before: &Value, after: &Value) -> Result<Value> {
    let b = before.as_object().ok_or("counter object unavailable")?;
    let a = after.as_object().ok_or("counter object unavailable")?;
    if b.len() != a.len() {
        return Err("counter schema changed".into());
    }
    let mut delta = serde_json::Map::new();
    for (name, value) in b {
        let count = a
            .get(name)
            .and_then(Value::as_u64)
            .ok_or("counter missing")?
            .checked_sub(value.as_u64().ok_or("invalid counter")?)
            .ok_or("counter regressed")?;
        delta.insert(name.clone(), json!(count));
    }
    Ok(Value::Object(delta))
}

async fn execute_arm(
    runtime: &crate::BenchRealRuntime,
    prompt: &[u32],
    report: &mut ArmReport,
    expected_resources: &mut Option<Value>,
    calibration: Option<&Calibration>,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing token loop")?;
    let config = observation_config(prompt.len(), OUTPUT_TOKENS)?;
    report.observation_capacity_per_collection = config.capacity_per_collection;
    report.p1j_launch_before = Some(token_loop.p1j_launch_snapshot());
    let snapshots = evidence::RequestSnapshotStart::capture(runtime)?;
    let mut request = token_loop.create_request_state()?;
    let execution: Result<()> = async {
        report.resources_before_opt_in = Some(token_loop.p1q_resource_snapshot()?);
        request.enable_predictor_v2_p1e_observation(token_loop, config)
            .map_err(|e| format!("P1E enablement failed: {e:?}"))?;
        match report.mode {
            P1jRequestMode::InertResourceOnly => request.enable_predictor_v2_p1j_sidecar_inert(token_loop)?,
            P1jRequestMode::Active => request.enable_predictor_v2_p1j_sidecar(token_loop)?,
        }
        let initial = request.p1q_initial_snapshot(token_loop)?;
        report.initial_state_pass = initial_state_valid(report.resources_before_opt_in.as_ref().ok_or("missing allocation pre-state")?, &initial);
        report.sidecar_freshly_initialized = report.initial_state_pass;
        report.resource_identity = Some(json!({
            "initial": structural_initial(&initial)?,
            "runtime_config_identity": report.runtime_resolved_config_sha256,
            "model_identity": report.provenance.as_ref().ok_or("missing provenance")?.model_identity,
            "production_configuration": report.provenance.as_ref().ok_or("missing provenance")?.production_configuration,
            "adapter": report.adapter,
            "model_load": report.model_load,
            "runtime_contract": structural_runtime_contract(report.runtime_contract.as_ref().ok_or("missing runtime contract")?)?,
            "prompt_token_count": prompt.len(),
            "prompt_token_ids_sha256": crate::greedy_parity::token_ids_sha256(prompt),
            "output_token_count": OUTPUT_TOKENS,
            "planned_positions": PLANNED_POSITIONS,
            "max_seq_len": token_loop.max_seq_len(),
        }));
        report.initial = Some(initial);
        if !report.initial_state_pass { return Err("P1Q initial resource/counter state invalid".into()); }
        let identity = report.resource_identity.as_ref().ok_or("missing resource identity")?;
        if expected_resources.as_ref().is_some_and(|p| p != identity)
            || calibration.map(|c| serialized_identity_matches(&c.resource_identity, identity)).transpose()?.is_some_and(|matched| !matched) {
            return Err("pre-request resource/config/model/adapter binding mismatch".into());
        }
        if expected_resources.is_none() { *expected_resources = Some(identity.clone()); }
        step_request(runtime, &mut request, prompt, report).await
    }.await;
    // Timing has already stopped. Finalize even failed arms and retain all evidence.
    finish_request(&mut request, report, execution.is_err());
    finish_snapshots(snapshots, runtime, report);
    let after = serde_json::to_value(token_loop.p1q_install_snapshot())?;
    if let Some(initial) = &report.initial {
        report.production_install_delta = Some(counter_delta(
            &serde_json::to_value(initial.production_install)?,
            &after,
        )?);
    }
    report.production_install_after = Some(after);
    execution
}

async fn step_request(
    runtime: &crate::BenchRealRuntime,
    request: &mut crate::gpu_native_token_loop::GpuNativeRequestState,
    prompt: &[u32],
    report: &mut ArmReport,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing token loop")?;
    if prompt.len() != 16 || PLANNED_POSITIONS > token_loop.max_seq_len() {
        return Err("frozen P1Q requires 16 prompt tokens and 143 planned positions".into());
    }
    report.generated_token_ids.reserve_exact(OUTPUT_TOKENS);
    let mut started = None;
    let mut stopped = None;
    let execution: Result<()> = async {
        for position in 0..PLANNED_POSITIONS {
            let (token, sample) = if position < prompt.len() {
                (prompt[position], position + 1 == prompt.len())
            } else {
                (
                    *report
                        .generated_token_ids
                        .last()
                        .ok_or("missing previous generated token")?,
                    true,
                )
            };
            if position == 0 {
                started = Some(Instant::now());
            }
            let sampled = token_loop
                .step_token(&runtime.engine, request, token, position, sample)
                .await?;
            if position + 1 == PLANNED_POSITIONS {
                stopped = Some(Instant::now());
            }
            increment(&mut report.completed_positions)?;
            match (sample, sampled) {
                (true, Some(id)) => report.generated_token_ids.push(id),
                (false, None) => {}
                _ => return Err("ordinary step returned unexpected sampling result".into()),
            }
        }
        Ok(())
    }
    .await;
    if execution.is_ok() {
        let ns: u64 = stopped
            .ok_or("missing request stop")?
            .duration_since(started.ok_or("missing request start")?)
            .as_nanos()
            .try_into()?;
        let (generated, planned) = throughput(ns)?;
        report.request_wall_ns = Some(ns);
        report.generated_tps = Some(generated);
        report.planned_position_tps = Some(planned);
    }
    execution
}

#[derive(Serialize)]
struct PairReport {
    pair_index: usize,
    execution_order: [&'static str; 2],
    control: ArmReport,
    treatment: ArmReport,
    parity: Parity,
    mechanical_state_comparison: MechanicalStateComparison,
    mechanism: Mechanism,
    resource_footprint_matched: bool,
    resource_mismatch_details: Vec<String>,
    correctness_pass: bool,
    pair_valid: bool,
    paired_generated_tps_delta_pct: Option<f64>,
}
fn finish_pair(
    index: usize,
    mode: ExperimentMode,
    control: ArmReport,
    treatment: ArmReport,
) -> PairReport {
    let labels = mode.labels();
    let order = execution_order(index).map(|a| labels[usize::from(a == Arm::Treatment)]);
    let parity = compare(&control, &treatment);
    let mechanical_state_comparison = compare_mechanical(&control, &treatment);
    let gate = |arm: &ArmReport| {
        arm.predictor_v2_snapshot
            .as_ref()
            .map(|p| {
                if arm.mode == P1jRequestMode::Active {
                    treatment_gate(p)
                } else {
                    control_gate(p)
                }
            })
            .unwrap_or_else(|| vec!["P0 reconciliation unavailable".into()])
    };
    let c_errors = gate(&control);
    let t_errors = gate(&treatment);
    let mut errors = c_errors.clone();
    errors.extend(t_errors.iter().cloned());
    errors.extend(parity.mismatch_details.iter().cloned());
    for arm in [&control, &treatment] {
        errors.extend(launch_diagnostics_gate(arm));
        errors.extend(arm.errors.iter().cloned());
        if arm.mode != mode.request_mode(arm.arm)
            || !arm.ordinary_invariants_pass
            || !shutdown_complete(arm)
            || !arm.initial_state_pass
            || !arm.sidecar_freshly_initialized
            || arm.generated_token_ids.len() != OUTPUT_TOKENS
            || arm.completed_positions != PLANNED_POSITIONS
            || arm.request_wall_ns.is_none_or(|v| v == 0)
        {
            errors.push(format!(
                "{:?} mode/initial/ordinary/timing/shutdown gate failed",
                arm.arm
            ));
        }
    }
    let mut mismatch = Vec::new();
    match (&control.resource_identity, &treatment.resource_identity) {
        (Some(c), Some(t)) if c == t => {},
        (c, t) => mismatch.push(format!("pre-request structural resources differ or unavailable: control={c:?}; treatment={t:?}")),
    }
    if !control.initial_state_pass || !treatment.initial_state_pass {
        mismatch.push("initial state is not certified fresh".into());
    }
    let correctness_pass = errors.is_empty();
    let matched = mismatch.is_empty();
    let delta = control
        .request_wall_ns
        .zip(treatment.request_wall_ns)
        .and_then(|(c, t)| paired_delta(c, t).ok());
    let mechanism = Mechanism {
        control_movement_zero: c_errors.is_empty(),
        treatment_movement_accounting_pass: t_errors.is_empty(),
        treatment_direct_matching_demand_credits: treatment
            .predictor_v2_snapshot
            .as_ref()
            .map(|p| p.direct_matching_demand_credits),
        qualification_failure_reasons: errors,
    };
    PairReport {
        pair_index: index + 1,
        execution_order: order,
        control,
        treatment,
        parity,
        mechanical_state_comparison,
        mechanism,
        resource_footprint_matched: matched,
        resource_mismatch_details: mismatch,
        correctness_pass,
        pair_valid: matched && correctness_pass && delta.is_some(),
        paired_generated_tps_delta_pct: delta,
    }
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    experiment_mode: ExperimentMode,
    expected_adapter_name: String,
    request_sha256: String,
    source_identity: Value,
    input_provenance: Option<Value>,
    resource_identity: Option<Value>,
    frozen_p1j_contract: FrozenContract,
    planned_positions: usize,
    pairs: Vec<PairReport>,
    noise_calibration_report_sha256: Option<String>,
    frozen_noise_floor_pct: Option<f64>,
    statistics: Option<NoiseStatistics>,
    median_control_tps: Option<f64>,
    median_treatment_tps: Option<f64>,
    resource_footprint_matched: bool,
    resource_mismatch_details: Vec<String>,
    qualification_pass: bool,
    performance_comparison_authorized: bool,
    performance_verdict: &'static str,
    errors: Vec<String>,
}
fn finish_experiment(report: &mut Report) -> Result<()> {
    report.resource_footprint_matched =
        report.pairs.len() == PAIRS && report.pairs.iter().all(|p| p.resource_footprint_matched);
    report.resource_mismatch_details = report
        .pairs
        .iter()
        .flat_map(|p| {
            p.resource_mismatch_details
                .iter()
                .map(move |d| format!("pair {}: {d}", p.pair_index))
        })
        .collect();
    report.qualification_pass = report.errors.is_empty()
        && report.pairs.len() == PAIRS
        && report.pairs.iter().all(|p| p.pair_valid);
    report.performance_comparison_authorized = false;
    report.performance_verdict = "NOT_AUTHORIZED";
    if !report.qualification_pass || !report.resource_footprint_matched {
        return Ok(());
    }
    let values: [f64; PAIRS] = report
        .pairs
        .iter()
        .map(|p| {
            p.paired_generated_tps_delta_pct
                .ok_or("missing pair metric")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "requires six pairs")?;
    let stats = noise_statistics(values)?;
    let median_arm = |control: bool| -> Result<f64> {
        let values = report
            .pairs
            .iter()
            .map(|p| {
                if control {
                    p.control.generated_tps
                } else {
                    p.treatment.generated_tps
                }
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("missing TPS")?;
        median_six(values.try_into().map_err(|_| "requires six pairs")?)
    };
    report.median_control_tps = Some(median_arm(true)?);
    report.median_treatment_tps = Some(median_arm(false)?);
    match report.experiment_mode {
        ExperimentMode::AaNoiseCalibration => {
            if noise_stable(&stats) {
                report.frozen_noise_floor_pct = Some(noise_floor(&stats));
                report.performance_comparison_authorized = true;
                report.performance_verdict = "NOISE_STABLE";
            } else {
                report.frozen_noise_floor_pct = None;
                report.performance_verdict = "NOISE_UNSTABLE";
            }
        }
        ExperimentMode::AbMovement => {
            if let Some(floor) = report
                .frozen_noise_floor_pct
                .filter(|f| f.is_finite() && *f >= 2.0)
            {
                if report.noise_calibration_report_sha256.is_some() {
                    report.performance_comparison_authorized = true;
                    report.performance_verdict = ab_verdict(&stats, floor);
                }
            }
        }
    }
    report.statistics = Some(stats);
    Ok(())
}

struct Calibration {
    bytes_sha256: String,
    floor: f64,
    input_provenance: Value,
    resource_identity: Value,
}
fn required<'a>(v: &'a Value, pointer: &str) -> Result<&'a Value> {
    v.pointer(pointer)
        .filter(|v| !v.is_null())
        .ok_or_else(|| format!("calibration missing {pointer}").into())
}
fn require_true(v: &Value, pointer: &str) -> Result<()> {
    if required(v, pointer)? != &json!(true) {
        return Err(format!("calibration failed {pointer}").into());
    }
    Ok(())
}
// serde_json's default (non-float_roundtrip) reader can shift the last bit.
// Compare derived metadata through the same serialization/readback path. The
// statistics used for gates are still recomputed from integer nanoseconds.
fn transport_value(value: &impl Serialize) -> Result<Value> {
    Ok(serde_json::from_slice(&serde_json::to_vec(value)?)?)
}

fn serialized_identity_matches(expected: &Value, current: &impl Serialize) -> Result<bool> {
    Ok(expected == &transport_value(current)?)
}

// Recover the frozen threshold directly from its original top-level numeric
// lexeme, with Rust's correctly rounded parser. No tolerance, decimal rounding,
// inferred threshold, alternate field, or Cargo feature change is involved.
// The complete JSON grammar has already been validated by serde_json.
fn calibration_floor_from_bytes(bytes: &[u8]) -> Result<f64> {
    let mut depth = 0usize;
    let mut i = 0usize;
    let mut found = None;
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.checked_sub(1).ok_or("invalid JSON depth")?,
            b'"' => {
                let start = i;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        break;
                    }
                    i += 1;
                }
                if depth == 1
                    && serde_json::from_slice::<String>(&bytes[start..=i])?
                        == "frozen_noise_floor_pct"
                {
                    let mut value_start = i + 1;
                    while bytes.get(value_start).is_some_and(u8::is_ascii_whitespace) {
                        value_start += 1;
                    }
                    if bytes.get(value_start) == Some(&b':') {
                        value_start += 1;
                        let mut end = value_start;
                        while bytes.get(end).is_some_and(|b| !matches!(b, b',' | b'}')) {
                            end += 1;
                        }
                        let value = std::str::from_utf8(&bytes[value_start..end])?
                            .trim()
                            .parse::<f64>()?;
                        if found.replace(value).is_some() {
                            return Err("duplicate calibration threshold".into());
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    found
        .filter(|f| f.is_finite() && *f >= 2.0)
        .ok_or_else(|| "invalid frozen calibration threshold".into())
}

fn validate_calibration(bytes: &[u8], request_sha: &str, sources: &Value) -> Result<Calibration> {
    let value: Value = serde_json::from_slice(bytes)?;
    for (key, expected) in [
        ("schema", json!(SCHEMA)),
        ("experiment_mode", json!("aa-noise-calibration")),
        ("performance_verdict", json!("NOISE_STABLE")),
        ("request_sha256", json!(request_sha)),
        ("source_identity", sources.clone()),
        ("expected_adapter_name", json!("NVIDIA L4")),
        ("planned_positions", json!(PLANNED_POSITIONS)),
    ] {
        if required(&value, &format!("/{key}"))? != &expected {
            return Err(format!("calibration {key} mismatch").into());
        }
    }
    for flag in [
        "/qualification_pass",
        "/resource_footprint_matched",
        "/performance_comparison_authorized",
    ] {
        require_true(&value, flag)?;
    }
    if required(&value, "/errors")? != &json!([])
        || required(&value, "/resource_mismatch_details")? != &json!([])
    {
        return Err("calibration contains errors".into());
    }
    // The independently serialized arm is authoritative. The aggregate JSON
    // summary may promote f32 configuration values while constructing a Value.
    let provenance = required(&value, "/pairs/0/control/provenance")?.clone();
    let resources = required(&value, "/resource_identity")?.clone();
    let pairs = required(&value, "/pairs")?
        .as_array()
        .ok_or("invalid calibration pairs")?;
    if pairs.len() != PAIRS {
        return Err("calibration requires exactly six pairs".into());
    }
    let mut deltas = [0.0; PAIRS];
    for (index, pair) in pairs.iter().enumerate() {
        let expected_order = if index % 2 == 0 {
            json!(["A", "B"])
        } else {
            json!(["B", "A"])
        };
        if required(pair, "/pair_index")? != &json!(index + 1)
            || required(pair, "/execution_order")? != &expected_order
        {
            return Err("calibration schedule mismatch".into());
        }
        for flag in [
            "/pair_valid",
            "/correctness_pass",
            "/resource_footprint_matched",
            "/parity/output_exact_match",
            "/parity/semantic_route_parity/exact_match",
            "/parity/evidence_structure_parity/exact_match",
            "/mechanism/control_movement_zero",
            "/mechanism/treatment_movement_accounting_pass",
        ] {
            require_true(pair, flag)?;
        }
        let mut walls = [0; 2];
        let mut raw = Vec::new();
        let mut outputs = Vec::new();
        for (i, name) in ["control", "treatment"].iter().enumerate() {
            let arm = required(pair, &format!("/{name}"))?;
            if required(arm, "/mode")? != &json!("InertResourceOnly")
                || required(arm, "/provenance")? != &provenance
                || required(arm, "/resource_identity")? != &resources
                || required(arm, "/predictor_v2_snapshot")?
                    != &serde_json::to_value(ReconciliationSnapshot::default())?
                || required(arm, "/errors")? != &json!([])
                || required(arm, "/completed_positions")? != &json!(PLANNED_POSITIONS)
            {
                return Err("calibration arm provenance/resource/movement/state mismatch".into());
            }
            for flag in [
                "/initial_state_pass",
                "/sidecar_freshly_initialized",
                "/ordinary_invariants_pass",
                "/runtime_shutdown/controlled_shutdown_requested",
                "/runtime_shutdown/all_runtime_resources_released",
            ] {
                require_true(arm, flag)?;
            }
            for name in ["p1j_launch_before", "p1j_launch_after", "p1j_launch_delta"] {
                if required(arm, &format!("/{name}"))?
                    != &serde_json::to_value(P1jLaunchSnapshot::default())?
                {
                    return Err("calibration inert launch activity".into());
                }
            }
            let ids: Vec<u32> =
                serde_json::from_value(required(arm, "/generated_token_ids")?.clone())?;
            if ids.len() != OUTPUT_TOKENS
                || required(arm, "/generated_token_ids_sha256")?
                    != &json!(crate::greedy_parity::token_ids_sha256(&ids))
            {
                return Err("calibration output hash/count invalid".into());
            }
            outputs.push(ids);
            raw.push(serde_json::from_value::<p1e::Report>(
                required(arm, "/raw_p1e_report")?.clone(),
            )?);
            walls[i] = required(arm, "/request_wall_ns")?
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or("invalid calibration wall time")?;
            let (generated, planned) = throughput(walls[i])?;
            if required(arm, "/generated_tps")? != &transport_value(&generated)?
                || required(arm, "/planned_position_tps")? != &transport_value(&planned)?
            {
                return Err("calibration throughput mismatch".into());
            }
        }
        if outputs[0] != outputs[1]
            || semantic_route(&raw[0]) != semantic_route(&raw[1])
            || evidence_structure(&raw[0]) != evidence_structure(&raw[1])
        {
            return Err("calibration raw parity failed".into());
        }
        deltas[index] = paired_delta(walls[0], walls[1])?;
        if required(pair, "/paired_generated_tps_delta_pct")? != &transport_value(&deltas[index])? {
            return Err("calibration pair metric mismatch".into());
        }
    }
    let stats = noise_statistics(deltas)?;
    let floor = calibration_floor_from_bytes(bytes)?;
    if floor != noise_floor(&stats)
        || !noise_stable(&stats)
        || required(&value, "/statistics")? != &transport_value(&stats)?
        || required(&value, "/frozen_noise_floor_pct")? != &transport_value(&floor)?
    {
        return Err("calibration noise statistics/floor invalid".into());
    }
    Ok(Calibration {
        bytes_sha256: crate::greedy_parity::sha256_hex(bytes),
        floor,
        input_provenance: provenance,
        resource_identity: resources,
    })
}

async fn run_arm(
    args: &CommandArgs,
    bytes: &[u8],
    arm: Arm,
    expected_provenance: &mut Option<Value>,
    expected_resources: &mut Option<Value>,
    calibration: Option<&Calibration>,
) -> ArmReport {
    let mut report = ArmReport::new(arm);
    report.mode = args.experiment_mode.request_mode(arm);
    let preparation: Result<_> = (|| {
        let (prompt, _) = parse_request(bytes)?;
        let prepared = prepare_arm(args)?;
        let identity = serde_json::to_value(&prepared.provenance)?;
        let calibration_matches = calibration
            .map(|c| serialized_identity_matches(&c.input_provenance, &prepared.provenance))
            .transpose()?
            .unwrap_or(true);
        report.provenance = Some(prepared.provenance);
        if expected_provenance.as_ref().is_some_and(|p| p != &identity) || !calibration_matches {
            return Err("input/config/model/executable provenance drift".into());
        }
        *expected_provenance = Some(identity);
        let mode = crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark;
        let tokenizer = crate::load_real_cli_tokenizer(&prepared.spec.cfg, mode)?;
        Ok((prepared.spec, tokenizer, prompt))
    })();
    match preparation {
        Err(e) => report.errors.push(e.to_string()),
        Ok((spec, tokenizer, prompt)) => {
            report.runtime_build_attempted = true;
            match crate::build_isolated_greedy_runtime(
                &spec,
                crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
                tokenizer.clone(),
            )
            .await
            {
                Err(e) => report.errors.push(e.to_string()),
                Ok(runtime) => {
                    report.runtime_constructed = true;
                    let execution: Result<()> = async {
                        validate_runtime(&runtime, &args.expected_adapter_name, &mut report)?;
                        // Exactly one deterministic tokenization per independently built arm.
                        let prompt_ids = tokenizer.encode(&prompt)?;
                        if prompt_ids.len() != 16 {
                            return Err("frozen P1Q requires exactly 16 prompt tokens".into());
                        }
                        execute_arm(
                            &runtime,
                            &prompt_ids,
                            &mut report,
                            expected_resources,
                            calibration,
                        )
                        .await
                    }
                    .await;
                    if let Err(e) = execution {
                        report.errors.push(e.to_string());
                    }
                    // Always shut down before the next arm, including setup/request failure.
                    shutdown(runtime, &mut report).await;
                }
            }
        }
    }
    if let Some(identity) = &report.resource_identity {
        if expected_resources.as_ref().is_some_and(|p| p != identity) {
            report.errors.push(
                "resource/config/model/adapter calibration or experiment identity drift".into(),
            );
        }
        if expected_resources.is_none() {
            *expected_resources = Some(identity.clone());
        }
    }
    check_arm(&mut report, 16, OUTPUT_TOKENS, PLANNED_POSITIONS);
    report
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<()> {
    validate_args(&args)?;
    ensure_output_absent(&args.report_out)?;
    let request_bytes = std::fs::read(&args.request_json)?;
    parse_request(&request_bytes)?;
    let request_sha = crate::greedy_parity::sha256_hex(&request_bytes);
    let sources = source_identity();
    // Read once: validation, SHA and all later binding use the exact same bytes.
    let calibration = args
        .noise_calibration_report
        .as_ref()
        .map(|path| -> Result<Calibration> {
            let bytes = std::fs::read(path)?;
            validate_calibration(&bytes, &request_sha, &sources)
        })
        .transpose()?;
    let mut report = Report {
        schema: SCHEMA,
        experiment_mode: args.experiment_mode,
        expected_adapter_name: args.expected_adapter_name.clone(),
        request_sha256: request_sha,
        source_identity: sources,
        input_provenance: None,
        resource_identity: None,
        frozen_p1j_contract: FrozenContract::default(),
        planned_positions: PLANNED_POSITIONS,
        pairs: Vec::with_capacity(PAIRS),
        noise_calibration_report_sha256: calibration.as_ref().map(|c| c.bytes_sha256.clone()),
        frozen_noise_floor_pct: calibration.as_ref().map(|c| c.floor),
        statistics: None,
        median_control_tps: None,
        median_treatment_tps: None,
        resource_footprint_matched: false,
        resource_mismatch_details: Vec::new(),
        qualification_pass: false,
        performance_comparison_authorized: false,
        performance_verdict: "NOT_AUTHORIZED",
        errors: Vec::new(),
    };
    for index in 0..PAIRS {
        let order = execution_order(index);
        let first = run_arm(
            &args,
            &request_bytes,
            order[0],
            &mut report.input_provenance,
            &mut report.resource_identity,
            calibration.as_ref(),
        )
        .await;
        let second = if shutdown_complete(&first) {
            run_arm(
                &args,
                &request_bytes,
                order[1],
                &mut report.input_provenance,
                &mut report.resource_identity,
                calibration.as_ref(),
            )
            .await
        } else {
            let mut skipped = ArmReport::new(order[1]);
            skipped.mode = args.experiment_mode.request_mode(order[1]);
            skipped
                .errors
                .push("not constructed: previous arm shutdown unproven".into());
            skipped
        };
        let (control, treatment) = if order[0] == Arm::Control {
            (first, second)
        } else {
            (second, first)
        };
        let pair = finish_pair(index, args.experiment_mode, control, treatment);
        let valid = pair.pair_valid;
        report.pairs.push(pair);
        // Retain the failed pair, then stop. Never continue and select survivors.
        if !valid {
            break;
        }
    }
    finish_experiment(&mut report)?;
    write_report(&args.report_out, &report)?;
    if !report.qualification_pass {
        return Err("P1Q certification failed; complete evidence retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod p1q_tests {
    use super::*;
    use crate::predictor_v2::{ModelMetadata, PositionIdentity, RequestIdentity};
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
    fn resources(allocated: bool) -> crate::gpu_native_residency::P1qResourceSnapshot {
        crate::gpu_native_residency::P1qResourceSnapshot {
            sidecar_initialized: allocated,
            sidecar_allocated: allocated,
            sidecar_stride_bytes: 2_654_212,
            sidecar_bank: 1,
            sidecar_slot: 0,
            namespace: allocated.then_some(namespace()),
            ordinary_total_expert_budget_bytes: 2 * 1024 * 1024 * 1024,
            ordinary_arena_allocation_bytes: 48 * 8 * 2_654_212,
            ordinary_layer_capacities: vec![8; 48],
            ordinary_layer_resident_counts: vec![0; 48],
            p1e_shadow_present: allocated,
            activity_counters: [0; 14],
        }
    }
    fn initial() -> P1qInitialSnapshot {
        P1qInitialSnapshot {
            mode: P1jRequestMode::InertResourceOnly,
            resources: resources(true),
            token_loop: Default::default(),
            recovery: Default::default(),
            production_install: Default::default(),
            launch: Default::default(),
            p0: Default::default(),
            committed_position: 0,
            pending_sidecar: false,
            observation_capacity: observation_config(16, 128).unwrap().capacity_per_collection,
        }
    }
    fn arm(mode: ExperimentMode, which: Arm) -> ArmReport {
        let mut r = ArmReport::new(which);
        r.mode = mode.request_mode(which);
        r.raw_p1e_report = Some(raw(vec![record(0, true, Some(false), Some(false))]));
        r.generated_token_ids = vec![11; OUTPUT_TOKENS];
        r.generated_token_ids_sha256 =
            crate::greedy_parity::token_ids_sha256(&r.generated_token_ids);
        r.predictor_v2_snapshot = Some(if r.mode == P1jRequestMode::Active {
            successful_p0()
        } else {
            Default::default()
        });
        r.p1j_launch_before = Some(Default::default());
        let after = if r.mode == P1jRequestMode::Active {
            P1jLaunchSnapshot {
                launch_considered: 2,
                source_first_attempt_clean: 1,
                source_checkpoint_recovered_clean: 1,
                writer_spawned: 2,
                ..Default::default()
            }
        } else {
            Default::default()
        };
        finish_launch_diagnostics(&mut r, after);
        r.runtime_shutdown = Some(crate::greedy_parity::BackgroundShutdownEvidence {
            controlled_shutdown_requested: true,
            all_runtime_resources_released: true,
            poll_iterations: 1,
        });
        r.completed_positions = PLANNED_POSITIONS;
        r.ordinary_invariants_pass = true;
        r.initial_state_pass = true;
        r.sidecar_freshly_initialized = true;
        r.resources_before_opt_in = Some(resources(false));
        r.initial = Some(initial());
        r.initial.as_mut().unwrap().mode = r.mode;
        r.resource_identity = Some(
            json!({"initial":structural_initial(r.initial.as_ref().unwrap()).unwrap(),"adapter":"NVIDIA L4","model":"fixture","config":"fixture"}),
        );
        r.request_wall_ns = Some(1_000_000_000);
        r.generated_tps = Some(128.0);
        r.planned_position_tps = Some(143.0);
        r
    }
    fn pair(mode: ExperimentMode, index: usize) -> PairReport {
        finish_pair(
            index,
            mode,
            arm(mode, Arm::Control),
            arm(mode, Arm::Treatment),
        )
    }
    fn experiment(mode: ExperimentMode) -> Report {
        Report {
            schema: SCHEMA,
            experiment_mode: mode,
            expected_adapter_name: "NVIDIA L4".into(),
            request_sha256: "request".into(),
            source_identity: json!({"fixture":"source"}),
            input_provenance: Some(json!({"fixture":"provenance"})),
            resource_identity: arm(mode, Arm::Control).resource_identity,
            frozen_p1j_contract: Default::default(),
            planned_positions: PLANNED_POSITIONS,
            pairs: (0..PAIRS).map(|i| pair(mode, i)).collect(),
            noise_calibration_report_sha256: None,
            frozen_noise_floor_pct: None,
            statistics: None,
            median_control_tps: None,
            median_treatment_tps: None,
            resource_footprint_matched: false,
            resource_mismatch_details: vec![],
            qualification_pass: false,
            performance_comparison_authorized: false,
            performance_verdict: "NOT_AUTHORIZED",
            errors: vec![],
        }
    }
    fn calibration_value() -> Value {
        let mut r = experiment(ExperimentMode::AaNoiseCalibration);
        finish_experiment(&mut r).unwrap();
        let mut v = serde_json::to_value(r).unwrap();
        let provenance = v["input_provenance"].clone();
        for pair in v["pairs"].as_array_mut().unwrap() {
            for name in ["control", "treatment"] {
                pair[name]["provenance"] = provenance.clone();
            }
        }
        v
    }
    fn calibration(v: &Value) -> Result<Calibration> {
        validate_calibration(
            &serde_json::to_vec(v).unwrap(),
            "request",
            &json!({"fixture":"source"}),
        )
    }
    fn args(mode: ExperimentMode) -> CommandArgs {
        CommandArgs {
            config: "config".into(),
            request_json: "request".into(),
            expected_adapter_name: "NVIDIA L4".into(),
            report_out: "report".into(),
            experiment_mode: mode,
            noise_calibration_report: None,
        }
    }

    mod p1q1 {
        use super::*;

        fn runtime_contract(context_id: &str) -> evidence::RuntimeContractEvidence {
            evidence::RuntimeContractEvidence {
                real_transformer_enabled: true,
                real_transformer_gpu_native: true,
                compute_offload: "gpu".into(),
                ordinary_step_token_only: true,
                legacy_execution_plan: crate::qualification::ExecutionPlanEvidence {
                    context_id: context_id.into(),
                    requested: "gpu".into(),
                    resolved: "gpu".into(),
                    embeddings: "cpu".into(),
                    lm_head: "cpu".into(),
                    dense_projections: "cpu".into(),
                    attention: "gpu".into(),
                    kv: "gpu".into(),
                    router: "cpu".into(),
                    routed_experts: "cpu".into(),
                    routed_expert_dtype: "q4_0".into(),
                    fallback_occurred: false,
                    reason: None,
                },
                token_loop_geometry: crate::gpu_native_token_loop::GpuNativeModelGeometry {
                    num_layers: 48,
                    d_model: 2048,
                    d_ff: 768,
                    num_experts: 128,
                    top_k: 8,
                    num_heads: 64,
                    num_kv_heads: 8,
                    head_dim: 128,
                    rope_dim: 128,
                    vocab_size: 151936,
                    max_seq_len: 4096,
                    rms_eps: 1e-6,
                    rope_base: 10000.0,
                },
                strict_fail_closed_routed_experts: true,
            }
        }

        type ContractDrift = (&'static str, fn(&mut evidence::RuntimeContractEvidence));
        fn contract_drifts() -> Vec<ContractDrift> {
            vec![
                ("real_transformer_enabled", |r| {
                    r.real_transformer_enabled = false
                }),
                ("real_transformer_gpu_native", |r| {
                    r.real_transformer_gpu_native = false
                }),
                ("compute_offload", |r| r.compute_offload = "cpu".into()),
                ("ordinary_step_token_only", |r| {
                    r.ordinary_step_token_only = false
                }),
                ("strict_fail_closed_routed_experts", |r| {
                    r.strict_fail_closed_routed_experts = false
                }),
                ("requested", |r| {
                    r.legacy_execution_plan.requested = "auto".into()
                }),
                ("resolved", |r| {
                    r.legacy_execution_plan.resolved = "cpu".into()
                }),
                ("embeddings", |r| {
                    r.legacy_execution_plan.embeddings = "gpu".into()
                }),
                ("lm_head", |r| {
                    r.legacy_execution_plan.lm_head = "gpu".into()
                }),
                ("dense_projections", |r| {
                    r.legacy_execution_plan.dense_projections = "gpu".into()
                }),
                ("attention", |r| {
                    r.legacy_execution_plan.attention = "cpu".into()
                }),
                ("kv", |r| r.legacy_execution_plan.kv = "cpu".into()),
                ("router", |r| r.legacy_execution_plan.router = "gpu".into()),
                ("routed_experts", |r| {
                    r.legacy_execution_plan.routed_experts = "gpu".into()
                }),
                ("routed_expert_dtype", |r| {
                    r.legacy_execution_plan.routed_expert_dtype = "f32".into()
                }),
                ("fallback_occurred", |r| {
                    r.legacy_execution_plan.fallback_occurred = true
                }),
                ("reason", |r| {
                    r.legacy_execution_plan.reason = Some("fallback".into())
                }),
                ("num_layers", |r| r.token_loop_geometry.num_layers += 1),
                ("d_model", |r| r.token_loop_geometry.d_model += 1),
                ("d_ff", |r| r.token_loop_geometry.d_ff += 1),
                ("num_experts", |r| r.token_loop_geometry.num_experts += 1),
                ("top_k", |r| r.token_loop_geometry.top_k += 1),
                ("num_heads", |r| r.token_loop_geometry.num_heads += 1),
                ("num_kv_heads", |r| r.token_loop_geometry.num_kv_heads += 1),
                ("head_dim", |r| r.token_loop_geometry.head_dim += 1),
                ("rope_dim", |r| r.token_loop_geometry.rope_dim += 1),
                ("vocab_size", |r| r.token_loop_geometry.vocab_size += 1),
                ("max_seq_len", |r| r.token_loop_geometry.max_seq_len += 1),
                ("rms_eps", |r| r.token_loop_geometry.rms_eps *= 2.0),
                ("rope_base", |r| r.token_loop_geometry.rope_base += 1.0),
            ]
        }

        // Report-only fixtures: no runtime, model, device, or request execution.
        fn fresh_arm(mode: ExperimentMode, which: Arm, context_id: &str) -> ArmReport {
            let mut r = arm(mode, which);
            r.runtime_contract = Some(runtime_contract(context_id));
            r.resource_identity = Some(json!({
                "initial": structural_initial(r.initial.as_ref().unwrap()).unwrap(),
                "runtime_config_identity": "config",
                "model_identity": {"sha256": "model"},
                "production_configuration": {"strict_weights": true},
                "adapter": {
                    "name": "NVIDIA L4", "vendor_id": 0x10de, "device_id": 0x27b8,
                    "device_type": "DiscreteGpu", "wgpu_backend": "vulkan",
                    "driver": "fixture", "driver_info": "fixture",
                    "compute_plane": "wgpu-vulkan", "software_adapter": false
                },
                "model_load": {"strict": true, "loaded_tensors": 10},
                "runtime_contract": structural_runtime_contract(r.runtime_contract.as_ref().unwrap()).unwrap(),
                "prompt_token_count": 16,
                "prompt_token_ids_sha256": "prompt",
                "output_token_count": OUTPUT_TOKENS,
                "planned_positions": PLANNED_POSITIONS,
                "max_seq_len": 4096,
            }));
            r
        }

        fn fresh_experiment() -> Report {
            let mode = ExperimentMode::AaNoiseCalibration;
            let mut r = experiment(mode);
            r.pairs = (0..PAIRS)
                .map(|i| {
                    finish_pair(
                        i,
                        mode,
                        fresh_arm(mode, Arm::Control, &(2 + i * 2).to_string()),
                        fresh_arm(mode, Arm::Treatment, &(3 + i * 2).to_string()),
                    )
                })
                .collect();
            r.resource_identity = r.pairs[0].control.resource_identity.clone();
            finish_experiment(&mut r).unwrap();
            r
        }

        fn fresh_calibration_value() -> Value {
            let mut v = serde_json::to_value(fresh_experiment()).unwrap();
            let provenance = v["input_provenance"].clone();
            for pair in v["pairs"].as_array_mut().unwrap() {
                for name in ["control", "treatment"] {
                    pair[name]["provenance"] = provenance.clone();
                }
            }
            v
        }

        #[test]
        fn context_ids_normalize_equal_and_raw_arm_evidence_is_lossless() {
            let a = runtime_contract("2");
            let b = runtime_contract("3");
            let before_a = serde_json::to_value(&a).unwrap();
            let before_b = serde_json::to_value(&b).unwrap();
            assert_ne!(before_a, before_b);
            assert_eq!(
                structural_runtime_contract(&a).unwrap(),
                structural_runtime_contract(&b).unwrap()
            );
            let mut expected = before_a.clone();
            expected["legacy_execution_plan"]["context_id"] = json!("runtime-local");
            assert_eq!(structural_runtime_contract(&a).unwrap(), expected);
            assert_eq!(serde_json::to_value(&a).unwrap(), before_a);
            assert_eq!(serde_json::to_value(&b).unwrap(), before_b);
            for (id, raw) in [("2", &a), ("3", &b)] {
                let report = fresh_arm(ExperimentMode::AaNoiseCalibration, Arm::Control, id);
                let serialized = transport_value(&report).unwrap();
                assert_eq!(
                    serialized["runtime_contract"],
                    transport_value(&raw).unwrap()
                );
                assert_eq!(
                    serialized["runtime_contract"]["legacy_execution_plan"]["context_id"],
                    id
                );
            }
        }

        #[test]
        fn every_substantive_runtime_contract_field_remains_exact() {
            let expected = structural_runtime_contract(&runtime_contract("2")).unwrap();
            for (field, drift) in contract_drifts() {
                let mut current = runtime_contract("3");
                drift(&mut current);
                assert_ne!(
                    expected,
                    structural_runtime_contract(&current).unwrap(),
                    "{field}"
                );
            }
        }

        #[test]
        fn namespace_normalization_is_only_three_ids_and_declared_mode() {
            let a = initial();
            let mut expected = serde_json::to_value(&a).unwrap();
            expected.as_object_mut().unwrap().remove("mode");
            for key in ["runtime", "context", "arena"] {
                expected["resources"]["namespace"][key] = json!(0);
            }
            assert_eq!(structural_initial(&a).unwrap(), expected);
            let mut b = a.clone();
            b.mode = P1jRequestMode::Active;
            let ns = b.resources.namespace.as_mut().unwrap();
            ns.runtime += 11;
            ns.context += 13;
            ns.arena += 17;
            assert_eq!(structural_initial(&b).unwrap(), expected);
            for drift in [
                (|s: &mut P1qInitialSnapshot| s.resources.namespace.as_mut().unwrap().layer += 1)
                    as fn(&mut P1qInitialSnapshot),
                |s| s.resources.namespace.as_mut().unwrap().capacity += 1,
            ] {
                let mut changed = b.clone();
                drift(&mut changed);
                assert_ne!(structural_initial(&changed).unwrap(), expected);
            }
            b.resources.namespace = None;
            assert!(structural_initial(&b).is_err());
        }

        #[test]
        fn namespace_layer_and_capacity_remain_exact() {
            let expected = structural_initial(&initial()).unwrap();
            let changes: &[fn(&mut p1e::Namespace)] = &[
                |namespace| namespace.layer += 1,
                |namespace| namespace.capacity += 1,
            ];
            for change in changes {
                let mut current = initial();
                change(current.resources.namespace.as_mut().unwrap());
                assert_ne!(structural_initial(&current).unwrap(), expected);
            }
        }

        #[test]
        fn fresh_context_aa_fixture_satisfies_resource_equality() {
            let r = fresh_experiment();
            assert!(r.qualification_pass);
            assert!(r.resource_footprint_matched);
            assert!(r.performance_comparison_authorized);
            assert_eq!(r.performance_verdict, "NOISE_STABLE");
            for pair in &r.pairs {
                assert!(pair.pair_valid);
                assert_eq!(
                    pair.control.resource_identity,
                    pair.treatment.resource_identity
                );
                assert_ne!(
                    pair.control
                        .runtime_contract
                        .as_ref()
                        .unwrap()
                        .legacy_execution_plan
                        .context_id,
                    pair.treatment
                        .runtime_contract
                        .as_ref()
                        .unwrap()
                        .legacy_execution_plan
                        .context_id,
                );
            }
        }

        #[test]
        fn fresh_context_ab_calibration_binding_accepts_only_incarnation_drift() {
            let v = fresh_calibration_value();
            let calibrated = calibration(&v).unwrap();
            for which in [Arm::Control, Arm::Treatment] {
                let current = fresh_arm(ExperimentMode::AbMovement, which, "99");
                assert!(serialized_identity_matches(
                    &calibrated.resource_identity,
                    current.resource_identity.as_ref().unwrap()
                )
                .unwrap());
            }
            for (field, drift) in contract_drifts() {
                let mut current = fresh_arm(ExperimentMode::AbMovement, Arm::Treatment, "99");
                let raw = current.runtime_contract.as_mut().unwrap();
                drift(raw);
                current.resource_identity.as_mut().unwrap()["runtime_contract"] =
                    structural_runtime_contract(raw).unwrap();
                let identity = current.resource_identity.as_ref().unwrap();
                assert!(
                    !serialized_identity_matches(&calibrated.resource_identity, identity).unwrap(),
                    "{field}"
                );
                let mut changed = v.clone();
                changed["pairs"][0]["treatment"]["resource_identity"] =
                    transport_value(identity).unwrap();
                assert!(calibration(&changed).is_err(), "{field}");
            }
        }

        // Mutate each leaf independently so newly added identity fields also
        // remain covered by the exact pair and calibration equality gates.
        fn leaf_drifts(value: &Value, path: &str, out: &mut Vec<(String, Value)>) {
            match value {
                Value::Object(fields) => {
                    for (key, value) in fields {
                        let key = key.replace('~', "~0").replace('/', "~1");
                        leaf_drifts(value, &format!("{path}/{key}"), out);
                    }
                }
                Value::Array(values) => {
                    for (i, value) in values.iter().enumerate() {
                        leaf_drifts(value, &format!("{path}/{i}"), out);
                    }
                }
                Value::Bool(value) => out.push((path.into(), json!(!value))),
                Value::Number(value) => {
                    out.push((path.into(), json!(value.as_f64().unwrap() + 1.0)))
                }
                Value::String(value) => out.push((path.into(), json!(format!("{value}-drift")))),
                Value::Null => out.push((path.into(), json!("drift"))),
            }
        }

        #[test]
        fn adapter_config_model_geometry_and_state_drift_blocks_all_verdicts() {
            let v = fresh_calibration_value();
            let calibrated = calibration(&v).unwrap();
            let mut changes = Vec::new();
            leaf_drifts(&calibrated.resource_identity, "", &mut changes);
            for (path, value) in changes {
                let mut r = fresh_experiment();
                let c = fresh_arm(r.experiment_mode, Arm::Control, "2");
                let mut t = fresh_arm(r.experiment_mode, Arm::Treatment, "3");
                let identity = t.resource_identity.as_mut().unwrap();
                *identity.pointer_mut(&path).unwrap() = value;
                assert!(
                    !serialized_identity_matches(&calibrated.resource_identity, identity).unwrap(),
                    "{path}"
                );
                let mut changed = v.clone();
                changed["pairs"][0]["treatment"]["resource_identity"] =
                    transport_value(identity).unwrap();
                assert!(calibration(&changed).is_err(), "{path}");
                r.pairs[0] = finish_pair(0, r.experiment_mode, c, t);
                assert!(!r.pairs[0].pair_valid, "{path}");
                finish_experiment(&mut r).unwrap();
                assert!(!r.resource_footprint_matched, "{path}");
                assert!(!r.performance_comparison_authorized, "{path}");
                assert_eq!(r.performance_verdict, "NOT_AUTHORIZED", "{path}");
            }
        }
    }

    #[test]
    fn p1q_mode_classification() {
        assert_eq!(
            ExperimentMode::AaNoiseCalibration.request_mode(Arm::Control),
            P1jRequestMode::InertResourceOnly
        );
        assert_eq!(
            ExperimentMode::AaNoiseCalibration.request_mode(Arm::Treatment),
            P1jRequestMode::InertResourceOnly
        );
        assert_eq!(
            ExperimentMode::AbMovement.request_mode(Arm::Control),
            P1jRequestMode::InertResourceOnly
        );
        assert_eq!(
            ExperimentMode::AbMovement.request_mode(Arm::Treatment),
            P1jRequestMode::Active
        );
    }
    #[test]
    fn p1q_fresh_allocation_snapshot() {
        assert!(initial_state_valid(&resources(false), &initial()));
        assert!(!initial_state_valid(&resources(true), &initial()));
    }
    #[test]
    fn p1q_initial_geometry_hard() {
        let changes: &[fn(&mut P1qInitialSnapshot)] = &[
            |s| s.resources.sidecar_stride_bytes += 1,
            |s| s.resources.sidecar_bank = 0,
            |s| s.resources.sidecar_slot = 1,
            |s| s.resources.ordinary_layer_capacities[47] = 9,
            |s| s.resources.ordinary_total_expert_budget_bytes += 1,
            |s| s.resources.sidecar_allocated = false,
            |s| s.resources.p1e_shadow_present = false,
            |s| s.resources.ordinary_layer_resident_counts[0] = 1,
            |s| s.resources.activity_counters[0] = 1,
        ];
        for change in changes {
            let mut s = initial();
            change(&mut s);
            assert!(!initial_state_valid(&resources(false), &s));
        }
    }
    #[test]
    fn p1q_initial_counters_and_pending_hard() {
        let changes: &[fn(&mut P1qInitialSnapshot)] = &[
            |s| s.token_loop.token_attempts = 1,
            |s| s.recovery.resume_attempts = 1,
            |s| s.production_install.reservation_attempts = 1,
            |s| s.launch.launch_considered = 1,
            |s| s.p0.emitted = 1,
            |s| s.committed_position = 1,
            |s| s.pending_sidecar = true,
            |s| s.observation_capacity += 1,
        ];
        for change in changes {
            let mut s = initial();
            change(&mut s);
            assert!(!initial_state_valid(&resources(false), &s));
        }
    }
    #[test]
    fn p1q_structural_equality_normalizes_only_runtime_ids_and_mode() {
        let a = initial();
        let mut b = initial();
        b.mode = P1jRequestMode::Active;
        let ns = b.resources.namespace.as_mut().unwrap();
        ns.runtime += 9;
        ns.context += 7;
        ns.arena += 5;
        assert_eq!(
            structural_initial(&a).unwrap(),
            structural_initial(&b).unwrap()
        );
        b.resources.namespace.as_mut().unwrap().layer += 1;
        assert_ne!(
            structural_initial(&a).unwrap(),
            structural_initial(&b).unwrap()
        );
    }
    #[test]
    fn p1q_resource_mismatch_blocks_all_verdicts() {
        let mut r = experiment(ExperimentMode::AaNoiseCalibration);
        let c = arm(r.experiment_mode, Arm::Control);
        let mut t = arm(r.experiment_mode, Arm::Treatment);
        t.resource_identity.as_mut().unwrap()["adapter"] = json!("different");
        r.pairs[2] = finish_pair(2, r.experiment_mode, c, t);
        finish_experiment(&mut r).unwrap();
        assert!(!r.resource_footprint_matched);
        assert!(!r.performance_comparison_authorized);
        assert_eq!(r.performance_verdict, "NOT_AUTHORIZED");
        assert!(!r.resource_mismatch_details.is_empty());
    }
    #[test]
    fn p1q_strict_request_128_and_neutral_greedy() {
        for v in [
            json!({"prompt":"x","max_tokens":128,"temperature":0}),
            json!({"messages":[{"role":"user","content":"x"}],"max_tokens":128,"top_k":1}),
        ] {
            assert!(parse_request(&serde_json::to_vec(&v).unwrap()).is_ok());
        }
        for v in [
            json!({"prompt":"x","max_tokens":127,"temperature":0}),
            json!({"prompt":"x","max_tokens":128}),
            json!({"prompt":"x","max_tokens":128,"temperature":0,"top_p":0.9}),
            json!({"prompt":"x","messages":[],"max_tokens":128,"temperature":0}),
            json!({"prompt":"x","max_tokens":128,"temperature":0,"seed":1}),
            json!({"prompt":"x","max_tokens":128,"temperature":0,"stream":true}),
        ] {
            assert!(parse_request(&serde_json::to_vec(&v).unwrap()).is_err());
        }
    }
    #[test]
    fn p1q_cli_exact_and_calibration_required_only_ab() {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            args: CommandArgs,
        }
        let argv = [
            "p1q",
            "--config",
            "c",
            "--request-json",
            "r",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--experiment-mode",
            "aa-noise-calibration",
            "--report-out",
            "o",
        ];
        assert!(Cli::try_parse_from(argv).is_ok());
        for extra in ["--prompt", "--pairs", "--threshold", "--noise-floor"] {
            let mut a = argv.to_vec();
            a.extend([extra, "1"]);
            assert!(Cli::try_parse_from(a).is_err());
        }
        assert!(validate_args(&args(ExperimentMode::AaNoiseCalibration)).is_ok());
        let mut a = args(ExperimentMode::AbMovement);
        assert!(validate_args(&a).is_err());
        a.noise_calibration_report = Some("aa.json".into());
        assert!(validate_args(&a).is_ok());
        a.experiment_mode = ExperimentMode::AaNoiseCalibration;
        assert!(validate_args(&a).is_err());
    }
    #[test]
    fn p1q_schema_and_balanced_fixed_schedule() {
        assert_eq!(
            SCHEMA,
            "mer.predictor-v2-p1q-matched-resource-performance.v1"
        );
        assert_eq!(PAIRS, 6);
        for i in 0..6 {
            assert_eq!(
                execution_order(i),
                if i % 2 == 0 {
                    [Arm::Control, Arm::Treatment]
                } else {
                    [Arm::Treatment, Arm::Control]
                }
            );
        }
    }
    #[test]
    fn p1q_integer_timing_throughput_and_order_independent_delta() {
        assert_eq!(throughput(1_000_000_000).unwrap(), (128.0, 143.0));
        assert!(throughput(0).is_err());
        assert_eq!(paired_delta(2_000_000_000, 1_000_000_000).unwrap(), 100.0);
        for i in 0..6 {
            let mut c = arm(ExperimentMode::AaNoiseCalibration, Arm::Control);
            let t = arm(ExperimentMode::AaNoiseCalibration, Arm::Treatment);
            c.request_wall_ns = Some(2_000_000_000);
            assert_eq!(
                finish_pair(i, ExperimentMode::AaNoiseCalibration, c, t)
                    .paired_generated_tps_delta_pct,
                Some(100.0)
            );
        }
    }
    #[test]
    fn p1q_exact_median_mad_and_counts() {
        let s = noise_statistics([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]).unwrap();
        assert_eq!(s.median_delta_pct, 0.5);
        assert_eq!(s.mad_delta_pct, 1.5);
        assert_eq!(s.max_abs_delta_pct, 4.0);
        assert_eq!((s.positive, s.negative, s.tied), (3, 2, 1));
        assert_eq!((s.min_delta_pct, s.max_delta_pct), (-2.0, 4.0));
        assert!(median_six([f64::NAN; 6]).is_err());
    }
    #[test]
    fn p1q_noise_boundaries_inclusive() {
        let s = noise_statistics([-1.0, 0.0, 1.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            (s.median_delta_pct, s.mad_delta_pct, s.max_abs_delta_pct),
            (1.0, 1.0, 3.0)
        );
        assert!(noise_stable(&s));
        assert_eq!(noise_floor(&s), 3.0);
    }
    #[test]
    fn p1q_each_just_over_noise_boundary_fails() {
        let s = noise_statistics([-1.0, 0.0, 1.0, 1.0, 2.0, 3.0]).unwrap();
        for i in 0..3 {
            let mut s = s.clone();
            match i {
                0 => s.median_delta_pct = 1.0000001,
                1 => s.mad_delta_pct = 1.0000001,
                _ => s.max_abs_delta_pct = 3.0000001,
            };
            assert!(!noise_stable(&s));
        }
    }
    #[test]
    fn p1q_noise_floor_and_unstable_report() {
        assert_eq!(noise_floor(&noise_statistics([0.0; 6]).unwrap()), 2.0);
        let mut r = experiment(ExperimentMode::AaNoiseCalibration);
        for p in &mut r.pairs {
            p.paired_generated_tps_delta_pct = Some(4.0);
        }
        finish_experiment(&mut r).unwrap();
        assert_eq!(r.performance_verdict, "NOISE_UNSTABLE");
        assert!(!r.performance_comparison_authorized);
        assert!(r.frozen_noise_floor_pct.is_none());
    }
    #[test]
    fn p1q_aa_both_arms_movement_and_launch_zero() {
        for which in [Arm::Control, Arm::Treatment] {
            let c = arm(ExperimentMode::AaNoiseCalibration, Arm::Control);
            let t = arm(ExperimentMode::AaNoiseCalibration, Arm::Treatment);
            let (mut c, mut t) = (c, t);
            let a = if which == Arm::Control {
                &mut c
            } else {
                &mut t
            };
            a.predictor_v2_snapshot.as_mut().unwrap().emitted = 1;
            assert!(!finish_pair(0, ExperimentMode::AaNoiseCalibration, c, t).pair_valid);
        }
        let mut t = arm(ExperimentMode::AaNoiseCalibration, Arm::Treatment);
        finish_launch_diagnostics(
            &mut t,
            P1jLaunchSnapshot {
                launch_considered: 1,
                source_first_attempt_clean: 1,
                no_pending_freeze: 1,
                ..Default::default()
            },
        );
        assert!(!launch_diagnostics_gate(&t).is_empty());
    }
    #[test]
    fn p1q_ab_control_zero_and_treatment_positive() {
        assert!(pair(ExperimentMode::AbMovement, 0).pair_valid);
        let mut c = arm(ExperimentMode::AbMovement, Arm::Control);
        c.predictor_v2_snapshot
            .as_mut()
            .unwrap()
            .direct_matching_demand_credits = 1;
        assert!(
            !finish_pair(
                0,
                ExperimentMode::AbMovement,
                c,
                arm(ExperimentMode::AbMovement, Arm::Treatment)
            )
            .pair_valid
        );
        for i in 0..2 {
            let mut t = arm(ExperimentMode::AbMovement, Arm::Treatment);
            let p = t.predictor_v2_snapshot.as_mut().unwrap();
            if i == 0 {
                *p = Default::default();
            } else {
                p.direct_matching_demand_credits = 0;
            }
            assert!(
                !finish_pair(
                    0,
                    ExperimentMode::AbMovement,
                    arm(ExperimentMode::AbMovement, Arm::Control),
                    t
                )
                .pair_valid
            );
        }
    }
    #[test]
    fn p1q_treatment_accounting_failure_gates() {
        let changes: &[fn(&mut ReconciliationSnapshot)] = &[
            |p| p.terminal_predictions = 0,
            |p| p.accepted = 0,
            |p| p.source_completed = 0,
            |p| p.source_failed = 1,
            |p| p.source_cancelled = 1,
            |p| p.source_live = 1,
            |p| p.reservations_live = 1,
            |p| p.available_installs = 1,
            |p| p.live_predictions = 1,
        ];
        for change in changes {
            let mut p = successful_p0();
            change(&mut p);
            assert!(!treatment_gate(&p).is_empty());
        }
    }
    #[test]
    fn p1q_output_and_semantic_fields_hard() {
        let c = arm(ExperimentMode::AbMovement, Arm::Control);
        let mut t = arm(ExperimentMode::AbMovement, Arm::Treatment);
        t.generated_token_ids[0] += 1;
        assert!(!compare(&c, &t).output_exact_match);
        let changes: &[fn(&mut p1e::Candidate)] = &[
            |c| c.expert += 1,
            |c| c.source_set[0] += 1,
            |c| c.score += 1,
            |c| c.sequence += 1,
            |c| c.generation += 1,
            |c| c.committed_position_cutoff += 1,
            |c| c.table_update_cutoff += 1,
            |c| c.source_position.absolute_position += 1,
            |c| c.target_position.absolute_position += 1,
            |c| c.source_layer += 1,
            |c| c.target_layer += 1,
            |c| c.position_distance += 1,
            |c| c.nominal_layer_lead += 1,
            |c| c.signal_revision += 1,
            |c| c.model.num_layers += 1,
        ];
        for change in changes {
            let mut t = arm(ExperimentMode::AbMovement, Arm::Treatment);
            change(
                &mut t.raw_p1e_report.as_mut().unwrap().observations[0]
                    .freeze
                    .candidate,
            );
            assert!(!compare(&c, &t).semantic_route_parity.exact_match);
        }
    }
    #[test]
    fn p1q_evidence_presence_hard() {
        for i in 0..3 {
            let c = arm(ExperimentMode::AbMovement, Arm::Control);
            let mut t = arm(ExperimentMode::AbMovement, Arm::Treatment);
            let r = &mut t.raw_p1e_report.as_mut().unwrap().observations[0];
            match i {
                0 => r.freeze.physical = None,
                1 => r.deadline = None,
                _ => r.deadline.as_mut().unwrap().physical = None,
            };
            assert!(!finish_pair(0, ExperimentMode::AbMovement, c, t).pair_valid);
        }
    }
    #[test]
    fn p1q_mechanical_divergence_descriptive_and_raw_lossless() {
        let c = arm(ExperimentMode::AbMovement, Arm::Control);
        let mut t = arm(ExperimentMode::AbMovement, Arm::Treatment);
        let raw = t.raw_p1e_report.as_mut().unwrap();
        raw.observations[0].freeze.current = Some(true);
        raw.observations[0].freeze.source.logical_materialized = true;
        raw.partitions.current_at_f += 1;
        let before = serde_json::to_value(&t.raw_p1e_report).unwrap();
        let p = finish_pair(0, ExperimentMode::AbMovement, c, t);
        assert!(p.pair_valid);
        assert_eq!(
            p.mechanical_state_comparison
                .diff
                .unwrap()
                .divergent_observation_count,
            1
        );
        assert_eq!(
            serde_json::to_value(&p.treatment.raw_p1e_report).unwrap(),
            before
        );
    }
    #[test]
    fn p1q_invalid_or_missing_pair_prevents_verdict() {
        for i in 0..6 {
            let mut r = experiment(ExperimentMode::AaNoiseCalibration);
            r.pairs[i].pair_valid = false;
            finish_experiment(&mut r).unwrap();
            assert_eq!(r.performance_verdict, "NOT_AUTHORIZED");
            assert_eq!(r.pairs.len(), 6);
        }
        let mut r = experiment(ExperimentMode::AaNoiseCalibration);
        r.pairs.pop();
        finish_experiment(&mut r).unwrap();
        assert_eq!(r.performance_verdict, "NOT_AUTHORIZED");
    }
    #[test]
    fn p1q_calibration_roundtrip_and_snapshot_hash() {
        let v = calibration_value();
        let c = calibration(&v).unwrap();
        assert_eq!(c.floor, 2.0);
        assert_eq!(
            c.bytes_sha256,
            crate::greedy_parity::sha256_hex(&serde_json::to_vec(&v).unwrap())
        );
    }
    #[test]
    fn p1q_wrong_calibration_schema_mode_verdict_rejected() {
        for (key, value) in [
            ("schema", json!("v0")),
            ("experiment_mode", json!("ab-movement")),
            ("performance_verdict", json!("WIN")),
            ("request_sha256", json!("different")),
            ("source_identity", json!({})),
            ("frozen_noise_floor_pct", json!(0.0)),
        ] {
            let mut v = calibration_value();
            v[key] = value;
            assert!(calibration(&v).is_err(), "{key}");
        }
    }
    #[test]
    fn p1q_calibration_provenance_config_request_model_adapter_drift_rejected() {
        for pointer in [
            "/pairs/0/control/provenance",
            "/pairs/0/control/resource_identity/config",
            "/pairs/0/control/resource_identity/model",
            "/pairs/0/control/resource_identity/adapter",
            "/pairs/0/control/request_wall_ns",
            "/pairs/0/control/predictor_v2_snapshot/emitted",
            "/pairs/0/control/p1j_launch_after/writer_spawned",
        ] {
            let mut v = calibration_value();
            *v.pointer_mut(pointer).unwrap() = json!(99);
            assert!(calibration(&v).is_err(), "{pointer}");
        }
    }
    #[test]
    fn p1q_calibration_requires_all_six_gates_and_balancing() {
        for pointer in [
            "/qualification_pass",
            "/resource_footprint_matched",
            "/pairs/4/pair_valid",
            "/pairs/3/correctness_pass",
            "/pairs/1/control/initial_state_pass",
            "/pairs/5/treatment/runtime_shutdown/all_runtime_resources_released",
        ] {
            let mut v = calibration_value();
            *v.pointer_mut(pointer).unwrap() = json!(false);
            assert!(calibration(&v).is_err(), "{pointer}");
        }
        let mut v = calibration_value();
        v["pairs"].as_array_mut().unwrap().pop();
        assert!(calibration(&v).is_err());
        let mut v = calibration_value();
        v["pairs"][1]["execution_order"] = json!(["A", "B"]);
        assert!(calibration(&v).is_err());
    }
    #[test]
    fn p1q_win_loss_repeatability_and_strict_threshold() {
        assert_eq!(
            ab_verdict(
                &noise_statistics([3.0, 3.0, 3.0, 3.0, 3.0, -1.0]).unwrap(),
                2.0
            ),
            "WIN"
        );
        assert_eq!(
            ab_verdict(
                &noise_statistics([-3.0, -3.0, -3.0, -3.0, -3.0, 1.0]).unwrap(),
                2.0
            ),
            "LOSS"
        );
        for values in [
            [2.0; 6],
            [-2.0; 6],
            [3.0, 3.0, 3.0, 3.0, -1.0, -1.0],
            [-3.0, -3.0, -3.0, -3.0, 1.0, 1.0],
            [0.0; 6],
        ] {
            assert_eq!(
                ab_verdict(&noise_statistics(values).unwrap(), 2.0),
                "INCONCLUSIVE"
            );
        }
    }
    #[test]
    fn p1q_ab_never_authorized_without_validated_calibration() {
        let mut r = experiment(ExperimentMode::AbMovement);
        finish_experiment(&mut r).unwrap();
        assert_eq!(r.performance_verdict, "NOT_AUTHORIZED");
        r.frozen_noise_floor_pct = Some(2.0);
        finish_experiment(&mut r).unwrap();
        assert_eq!(r.performance_verdict, "NOT_AUTHORIZED");
        r.noise_calibration_report_sha256 = Some("bound".into());
        finish_experiment(&mut r).unwrap();
        assert_eq!(r.performance_verdict, "INCONCLUSIVE");
    }
    #[test]
    fn p1q_counter_delta_no_wrapping() {
        assert_eq!(
            counter_delta(&json!({"x":1}), &json!({"x":3})).unwrap(),
            json!({"x":2})
        );
        assert!(counter_delta(&json!({"x":3}), &json!({"x":1})).is_err());
    }

    fn production() -> &'static str {
        include_str!("gpu_native_predictor_v2_sidecar_performance.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap()
    }
    fn part<'a>(s: &'a str, start: &str, end: &str) -> &'a str {
        s.split(start).nth(1).unwrap().split(end).next().unwrap()
    }
    fn base(path: &str) -> String {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let o = std::process::Command::new("git")
            .args([
                "show",
                &format!("1849d8ff0d2114100bd89d1d682d4e902ec6825e:rust-engine/src/{path}"),
            ])
            .current_dir(root)
            .output()
            .unwrap();
        assert!(o.status.success());
        String::from_utf8(o.stdout).unwrap()
    }
    #[test]
    fn p1q_inert_returns_before_launch_prepare_spawn_or_pending() {
        let s = include_str!("gpu_native_token_loop.rs");
        let launch = part(
            s,
            "    fn launch_p1j_at_freeze(",
            "    fn publish_p1j_at_deadline(",
        );
        let guard = launch
            .find("movement.mode == P1jRequestMode::InertResourceOnly")
            .unwrap();
        let ret = launch[guard..].find("return;").unwrap() + guard;
        for action in [
            "p1m_launch(",
            ".try_prepare(",
            "*pending = Some(",
            "writer.spawn()",
        ] {
            assert!(ret < launch.find(action).unwrap());
        }
    }
    #[test]
    fn p1q_enable_uses_same_seam_and_active_preconditions() {
        let s = include_str!("gpu_native_token_loop.rs");
        let active = part(
            s,
            "    pub(crate) fn enable_predictor_v2_p1j_sidecar(",
            "    /// Performance-only",
        );
        assert!(active.contains("self.enable_p1j_mode(token_loop, P1jRequestMode::Active)"));
        let inert = part(
            s,
            "    pub(crate) fn enable_predictor_v2_p1j_sidecar_inert(",
            "    fn enable_p1j_mode(",
        );
        assert!(
            inert.contains("self.enable_p1j_mode(token_loop, P1jRequestMode::InertResourceOnly)")
        );
        let common = part(
            s,
            "    fn enable_p1j_mode(",
            "    /// Pre-request evidence only",
        );
        for t in [
            "self.committed_position != 0",
            "self.p1j.is_some()",
            "token_loop.q4_qualification.get().is_some()",
            "token_loop.snapshot().token_attempts != 0",
            "o.p1j_ready()",
            ".enable_p1j_sidecar(namespace.runtime)?",
            "pending: None",
        ] {
            assert!(common.contains(t), "{t}");
        }
        assert_eq!(common.matches(".enable_p1j_sidecar(").count(), 1);
    }
    #[test]
    fn p1q_frozen_active_publish_finish_deadline_and_execution_bytes() {
        let old = base("gpu_native_token_loop.rs");
        let new = include_str!("gpu_native_token_loop.rs");
        for (a, b) in [
            (
                "    fn publish_p1j_at_deadline(",
                "    pub(crate) fn enable_q4",
            ),
            ("    fn execute_token_segment_unified(", "    fn "),
        ] {
            if a.contains("execute_token_segment") {
                continue;
            } // Whole active region through finish and D below.
            let _ = b;
            let start = old.find(a).unwrap();
            let end = old[start..]
                .find("    pub(crate) fn")
                .map(|i| start + i)
                .unwrap();
            let segment = &old[start..end];
            assert!(new.contains(segment));
        }
        // Every old function containing the execution/D selection stays byte-exact.
        for name in [
            "observe_p1e_deadline",
            "execute_token_segment_unified",
            "finish_p1j_target",
        ] {
            let needle = format!("fn {name}(");
            let start = old.find(&needle).unwrap();
            let open = old[start..].find('{').unwrap() + start;
            let mut depth = 1;
            let mut end = open + 1;
            for (i, ch) in old[open + 1..].char_indices() {
                if ch == '{' {
                    depth += 1;
                }
                if ch == '}' {
                    depth -= 1;
                }
                if depth == 0 {
                    end = open + 2 + i;
                    break;
                }
            }
            assert!(new.contains(&old[start..end]), "{name}");
        }
        let d = part(new, "fn observe_p1e_deadline(", "fn ");
        assert!(d.contains("request.p1j.is_some()"));
        assert!(new.contains(".and_then(|s| s.pending.as_ref())"));
    }
    #[test]
    fn p1q_snapshot_nonblocking_read_only_no_gpu() {
        let s = include_str!("gpu_native_residency.rs");
        let snap = part(
            s,
            "pub(crate) fn p1q_resource_snapshot(",
            "    fn p1j_current(",
        );
        assert!(snap.contains(".try_lock()"));
        for bad in [
            ".lock()",
            ".await",
            ".get_or_init(",
            ".create_p1j_sidecar(",
            "device.poll",
            "queue.submit",
            ".fetch_add(",
            "&mut",
        ] {
            assert!(!snap.contains(bad), "{bad}");
        }
    }
    #[test]
    fn p1q_fresh_runtime_no_warmup_and_failed_pair_retained() {
        let s = production();
        assert_eq!(
            s.matches("crate::build_isolated_greedy_runtime(").count(),
            1
        );
        let arm = part(s, "async fn run_arm(", "pub(crate) async fn run_command(");
        assert!(arm.contains("prepare_arm(args)?"));
        assert_eq!(arm.matches("tokenizer.encode(").count(), 1);
        let run = part(s, "pub(crate) async fn run_command(", "\0");
        assert!(run.contains("for index in 0..PAIRS"));
        assert!(run.contains("if shutdown_complete(&first)"));
        assert!(run.find("report.pairs.push(pair)").unwrap() < run.find("if !valid").unwrap());
        assert!(!run.contains(".filter("));
        assert!(!s.contains("RequestPhase::Warmup"));
    }
    #[test]
    fn p1q_timer_excludes_setup_snapshots_finalization_shutdown() {
        let s = production();
        let timed = part(
            s,
            "async fn step_request(",
            "#[derive(Serialize)]\nstruct PairReport",
        );
        assert_eq!(timed.matches("Instant::now()").count(), 2);
        assert!(
            timed.find("started = Some(Instant::now())").unwrap()
                < timed.find(".step_token(").unwrap()
        );
        assert!(
            timed.find(".step_token(").unwrap()
                < timed.find("stopped = Some(Instant::now())").unwrap()
        );
        for bad in [
            "prepare_arm(",
            "tokenizer",
            "enable_predictor",
            "finish_request(",
            "shutdown(",
            "write_report(",
        ] {
            assert!(!timed.contains(bad));
        }
        let exec = part(s, "async fn execute_arm(", "async fn step_request(");
        assert!(exec.find("p1q_initial_snapshot").unwrap() < exec.find("step_request(").unwrap());
        assert!(exec.find("step_request(").unwrap() < exec.find("finish_request(").unwrap());
    }
    #[test]
    fn p1q_no_serving_config_environment_activation() {
        for s in [include_str!("server.rs"), include_str!("config.rs")] {
            assert!(!s.contains("sidecar_performance"));
            assert!(!s.contains("P1jRequestMode"));
        }
        let m = include_str!("main.rs");
        assert_eq!(
            m.matches("gpu_native_predictor_v2_sidecar_performance::run_command(args)")
                .count(),
            1
        );
        assert!(!m.contains("enable_predictor_v2_p1j"));
        assert!(!production().contains("std::env::"));
    }
    #[test]
    fn p1q_qualifier_frozen_and_witness_literals_exact_only() {
        assert_eq!(
            crate::greedy_parity::sha256_hex(include_bytes!(
                "gpu_native_predictor_v2_sidecar_qualifier.rs"
            )),
            "858e9c854647f5658a393d260a91c3844b1c57198ed87fedea216032448f15d3"
        );
        for (file, source, oldhash, new) in [
            (
                "gpu_native_predictor_v2_observation.rs",
                include_bytes!("gpu_native_token_loop.rs").as_slice(),
                "b6e1769508d8ed219d411c77318d28fa08752895ca03f947c55b7422106a8a0e",
                include_str!("gpu_native_predictor_v2_observation.rs"),
            ),
            (
                "gpu_native_q4_route_parallel.rs",
                include_bytes!("gpu_native_residency.rs").as_slice(),
                "4d1554bad7a97e72c2df423e2696d5b76fe8ace8280e3d5642c8213962faa50e",
                include_str!("gpu_native_q4_route_parallel.rs"),
            ),
        ] {
            let original = base(file);
            assert_eq!(original.matches(oldhash).count(), 1);
            let expected = original.replacen(oldhash, &crate::greedy_parity::sha256_hex(source), 1);
            assert_eq!(new, expected, "{file}");
        }
    }
    #[test]
    fn p1q_six_path_scope_and_protected_files() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let o = std::process::Command::new("git")
            .args([
                "diff",
                "--name-only",
                "1849d8ff0d2114100bd89d1d682d4e902ec6825e",
                "--",
            ])
            .current_dir(root)
            .output()
            .unwrap();
        assert!(o.status.success());
        let allowed = [
            "gpu_native_residency.rs",
            "gpu_native_token_loop.rs",
            "gpu_native_predictor_v2_sidecar_performance.rs",
            "main.rs",
            "gpu_native_predictor_v2_observation.rs",
            "gpu_native_q4_route_parallel.rs",
        ];
        let names = String::from_utf8(o.stdout).unwrap();
        for name in names.lines() {
            assert!(
                allowed
                    .iter()
                    .any(|a| name == format!("rust-engine/src/{a}")),
                "{name}"
            );
        }
        assert!(names.lines().count() <= 6);
    }
    #[test]
    fn p1q_calibration_nontrivial_float_roundtrip_and_exact_floor_lexeme() {
        let mut r = experiment(ExperimentMode::AaNoiseCalibration);
        for (i, pair) in r.pairs.iter_mut().enumerate() {
            let c = 1_000_000_000 + (i as u64 + 1) * 7321;
            let t = 1_000_000_000 + (i as u64 + 1) * 4123;
            for (arm, ns) in [(&mut pair.control, c), (&mut pair.treatment, t)] {
                arm.request_wall_ns = Some(ns);
                let (g, p) = throughput(ns).unwrap();
                arm.generated_tps = Some(g);
                arm.planned_position_tps = Some(p);
            }
            pair.paired_generated_tps_delta_pct = Some(paired_delta(c, t).unwrap());
        }
        finish_experiment(&mut r).unwrap();
        let mut value = serde_json::to_value(r).unwrap();
        let provenance = value["input_provenance"].clone();
        for pair in value["pairs"].as_array_mut().unwrap() {
            for name in ["control", "treatment"] {
                pair[name]["provenance"] = provenance.clone();
            }
        }
        assert!(calibration(&value).is_ok());
        let threshold = 2.718281828459045;
        let bytes = serde_json::to_vec(
            &json!({"nested":{"frozen_noise_floor_pct":99},"frozen_noise_floor_pct":threshold}),
        )
        .unwrap();
        assert_eq!(calibration_floor_from_bytes(&bytes).unwrap(), threshold);
        assert!(calibration_floor_from_bytes(
            br#"{"frozen_noise_floor_pct":2,"frozen_noise_floor_pct":3}"#
        )
        .is_err());
    }
    #[test]
    fn p1q_calibration_typed_f32_and_f64_identity_roundtrip() {
        #[derive(Serialize)]
        struct Identity {
            anchor: f32,
            configuration: f64,
        }
        let current = Identity {
            anchor: 0.6,
            configuration: 127.99906291886037,
        };
        let from_report: Value =
            serde_json::from_slice(&serde_json::to_vec(&current).unwrap()).unwrap();
        assert_ne!(serde_json::to_value(&current).unwrap(), from_report);
        assert!(serialized_identity_matches(&from_report, &current).unwrap());
        let changed = Identity {
            anchor: 0.5,
            configuration: current.configuration,
        };
        assert!(!serialized_identity_matches(&from_report, &changed).unwrap());
        let resource = json!({"configuration": current.configuration});
        assert!(
            serialized_identity_matches(&transport_value(&resource).unwrap(), &resource).unwrap()
        );
    }

    #[test]
    fn p1q_independent_p1o_contract_preserves_frozen_function_bytes() {
        let old = base("gpu_native_predictor_v2_sidecar_qualifier.rs");
        let new = production();
        for name in [
            "control_gate",
            "treatment_gate",
            "semantic_route",
            "evidence_structure",
            "mechanical_state",
            "compare_mechanical",
            "compare",
            "check_arm",
            "validate_runtime",
        ] {
            let start = old.find(&format!("fn {name}(")).unwrap();
            let open = start + old[start..].find('{').unwrap();
            let mut depth = 1;
            let mut end = open + 1;
            for (i, ch) in old[open + 1..].char_indices() {
                if ch == '{' {
                    depth += 1;
                }
                if ch == '}' {
                    depth -= 1;
                }
                if depth == 0 {
                    end = open + i + 2;
                    break;
                }
            }
            assert!(new.contains(&old[start..end]), "{name}");
        }
    }
}
