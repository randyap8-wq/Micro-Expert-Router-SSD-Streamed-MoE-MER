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
const CHILD_PROTOCOL: &str = "mer.predictor-v2-p1q3-arm.v1";
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
    #[arg(long, hide = true, requires_all = ["p1q3_child_pair_index", "p1q3_child_arm"])]
    p1q3_child_protocol: Option<String>,
    #[arg(long, hide = true, requires_all = ["p1q3_child_protocol", "p1q3_child_arm"], value_parser = clap::value_parser!(u8).range(1..=6))]
    p1q3_child_pair_index: Option<u8>,
    #[arg(long, hide = true, value_enum, requires_all = ["p1q3_child_protocol", "p1q3_child_pair_index"])]
    p1q3_child_arm: Option<Arm>,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
enum Arm {
    Control,
    Treatment,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmProvenance {
    #[serde(deserialize_with = "Deserialize::deserialize")]
    provenance: evidence::BenchmarkProvenance,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    config_path: String,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    config_sha256: String,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    production_configuration: evidence::ProductionConfiguration,
}

// Typed transport implementations live only in this qualification module.
// The constructor lists EVERY original field (no defaults or struct update),
// so additions to the original structs must also be handled here to compile.
// Require even nullable fields to be present; null is distinct from omission.
macro_rules! arm_transport_struct {
    ($ty:path { $($field:ident: $field_ty:ty,)+ }) => {
        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Fields {
                    $(#[serde(deserialize_with = "Deserialize::deserialize")]
                    $field: $field_ty,)+
                }
                let v = Fields::deserialize(d)?;
                Ok(Self { $($field: v.$field,)+ })
            }
        }
    };
}
macro_rules! arm_transport_enum {
    ($ty:path { $($variant:ident,)+ }) => {
        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
                #[derive(Deserialize)]
                enum Variant { $($variant,)+ }
                Ok(match Variant::deserialize(d)? { $(Variant::$variant => Self::$variant,)+ })
            }
        }
    };
}
arm_transport_struct!(crate::qualification::BuildProvenance {
    git_sha: Option<String>,
    dirty: Option<bool>,
    package_version: String,
});
arm_transport_struct!(crate::qualification::ArtifactDigest {
    configured_path: String,
    canonical_path: String,
    byte_length: u64,
    sha256: String,
});
arm_transport_struct!(crate::qualification::QualificationArtifacts {
    config: Option<crate::qualification::ArtifactDigest>,
    tokenizer: Option<crate::qualification::ArtifactDigest>,
    expert_metadata: Option<crate::qualification::ArtifactDigest>,
    packed_manifest: Option<crate::qualification::ArtifactDigest>,
    weights_config: Option<crate::qualification::ArtifactDigest>,
    dense_weights_directory: Option<String>,
    expert_data_directory: String,
    packed_expert_blob: Option<String>,
    large_artifacts_recursively_hashed: bool,
});
arm_transport_struct!(crate::qualification::ExpertMetadataEvidence {
    dtype: Option<String>,
    q4_0_layout: Option<String>,
    conversion_mode: Option<String>,
    source: Option<String>,
    explicitly_synthetic: bool,
});
arm_transport_struct!(crate::greedy_parity::ModelIdentityEvidence {
    architecture: String,
    num_layers: usize,
    num_experts_per_layer: u32,
    total_experts: u64,
    top_k: usize,
    d_model: usize,
    d_ff: usize,
    routed_expert_dtype: String,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::BenchmarkProvenance {
    build: crate::qualification::BuildProvenance,
    executable_canonical_path: String,
    executable_sha256: String,
    resolved_config_sha256: String,
    artifacts: crate::qualification::QualificationArtifacts,
    expert_metadata: crate::qualification::ExpertMetadataEvidence,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::CacheResidencyConfiguration {
    ram_cache_slots: usize,
    block_align: usize,
    direct_io: bool,
    pipeline_depth: u32,
    partial_load_fraction: f64,
    pin_after_observations: u64,
    packed_blob: Option<String>,
    packed_manifest: Option<String>,
    gpu_cache_enabled: bool,
    gpu_vram_capacity_mb: usize,
    gpu_vram_anchor_ratio: f32,
    gpu_promote_after_hits: u64,
    gpu_cache_dtype: String,
    gpu_native_max_seq_len: usize,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::PredictorPrefetchConfiguration {
    predict_fanout: usize,
    predict_min_prob: f64,
    max_concurrent_prefetches: usize,
    max_fetch_yields: usize,
    locality_enabled: bool,
    locality_window: usize,
    locality_threshold_pct: f32,
    speculator_enabled: bool,
    speculator_hidden_dim: usize,
    speculator_top_k: usize,
    affinity_enabled: bool,
    affinity_neighbors_k: usize,
    affinity_decay_epoch: u64,
    prefetch_governor: bool,
    prefetch_precision_floor: f64,
    prefetch_contention_weight: f64,
    cost_aware_eviction: bool,
    pregate_enabled: bool,
    static_residency_fraction: f64,
    static_residency_warmup_tokens: u64,
    static_residency_profile: Option<String>,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::ProductionConfiguration {
    q4_dtype: String,
    q4_layout: Option<String>,
    cache_residency: crate::gpu_native_real_benchmark::CacheResidencyConfiguration,
    predictor_prefetch: crate::gpu_native_real_benchmark::PredictorPrefetchConfiguration,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::RuntimeContractEvidence {
    real_transformer_enabled: bool,
    real_transformer_gpu_native: bool,
    compute_offload: String,
    ordinary_step_token_only: bool,
    legacy_execution_plan: crate::qualification::ExecutionPlanEvidence,
    token_loop_geometry: crate::gpu_native_token_loop::GpuNativeModelGeometry,
    strict_fail_closed_routed_experts: bool,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::CounterRatios {
    attempts_per_completed_position: f64,
    misses_per_completed_position: f64,
    replays_per_completed_position: f64,
    submissions_per_completed_position: f64,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::RecoveryRatios {
    resume_attempts_per_completed_position: f64,
    recovery_segments_per_completed_position: f64,
    checkpoint_captures_per_completed_position: f64,
    checkpoint_restores_per_completed_position: f64,
    full_token_replays_per_completed_position: f64,
    layers_encoded_per_completed_position: f64,
    attention_layers_reexecuted_per_completed_position: f64,
    expert_layers_reexecuted_per_completed_position: f64,
    invalid_tail_layers_encoded_per_completed_position: f64,
    residency_service_us_per_completed_position: f64,
    boundary_wait_us_per_completed_position: f64,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::EngineStorageSnapshot {
    ram_hits: u64,
    ram_misses: u64,
    nvme_read_operations: u64,
    nvme_bytes_read: u64,
    prefetch_completed: u64,
    predictor_observations: u64,
    ssd_stall_us: u64,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::GpuNativeResidencyDelta {
    vram_hits: u64,
    vram_misses: u64,
    physical_current_hits: u64,
    physical_source_acquisitions: u64,
    logical_admissions_for_physical_misses: u64,
    ram_to_vram_installs: u64,
    physical_evictions: u64,
    physical_reinstalls: u64,
    stale_generation_rejections: u64,
    demand_requests: u64,
    speculative_requests: u64,
    speculative_vram_hits: u64,
    speculative_ram_to_vram_installs: u64,
    speculative_dropped_capacity_or_pressure: u64,
});
arm_transport_struct!(crate::gpu_native_real_benchmark::RequestSnapshots {
    token_loop_before: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    token_loop_after: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    token_loop_delta: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    token_loop_ratios: crate::gpu_native_real_benchmark::CounterRatios,
    recovery_before: crate::gpu_native_token_loop::GpuNativeRecoverySnapshot,
    recovery_after: crate::gpu_native_token_loop::GpuNativeRecoverySnapshot,
    recovery_delta: crate::gpu_native_token_loop::GpuNativeRecoverySnapshot,
    recovery_ratios: crate::gpu_native_real_benchmark::RecoveryRatios,
    routed_execution_before: crate::engine::RoutedExpertExecutionSnapshot,
    routed_execution_after: crate::engine::RoutedExpertExecutionSnapshot,
    routed_execution_delta: crate::engine::RoutedExpertExecutionSnapshot,
    runtime_cache_before: crate::greedy_parity::RuntimeCacheSnapshot,
    runtime_cache_after: crate::greedy_parity::RuntimeCacheSnapshot,
    engine_storage_before: crate::gpu_native_real_benchmark::EngineStorageSnapshot,
    engine_storage_after: crate::gpu_native_real_benchmark::EngineStorageSnapshot,
    engine_storage_delta: crate::gpu_native_real_benchmark::EngineStorageSnapshot,
    gpu_expert_io_before: crate::backend::GpuExpertIoSnapshot,
    gpu_expert_io_after: crate::backend::GpuExpertIoSnapshot,
    gpu_expert_io_delta: crate::backend::GpuExpertIoSnapshot,
    gpu_expert_memory_before: crate::backend::GpuExpertMemorySnapshot,
    gpu_expert_memory_after: crate::backend::GpuExpertMemorySnapshot,
    gpu_native_residency_before: crate::gpu_native_residency::GpuNativeTieredResidencySnapshot,
    gpu_native_residency_after: crate::gpu_native_residency::GpuNativeTieredResidencySnapshot,
    gpu_native_residency_delta: crate::gpu_native_real_benchmark::GpuNativeResidencyDelta,
});
arm_transport_struct!(crate::gpu_native_residency::P1qResourceSnapshot {
    sidecar_initialized: bool,
    sidecar_allocated: bool,
    sidecar_stride_bytes: u64,
    sidecar_bank: u32,
    sidecar_slot: u32,
    namespace: Option<p1e::Namespace>,
    ordinary_total_expert_budget_bytes: u64,
    ordinary_arena_allocation_bytes: u64,
    ordinary_layer_capacities: Vec<usize>,
    ordinary_layer_resident_counts: Vec<usize>,
    p1e_shadow_present: bool,
    activity_counters: [u64; 14],
});
arm_transport_struct!(crate::gpu_native_token_loop::P1qInitialSnapshot {
    mode: crate::gpu_native_token_loop::P1jRequestMode,
    resources: crate::gpu_native_residency::P1qResourceSnapshot,
    token_loop: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    recovery: crate::gpu_native_token_loop::GpuNativeRecoverySnapshot,
    production_install: crate::backend::gpu_native::GpuNativeProductionPhysicalInstallSnapshot,
    launch: crate::gpu_native_token_loop::P1jLaunchSnapshot,
    p0: crate::predictor_v2::ReconciliationSnapshot,
    committed_position: usize,
    pending_sidecar: bool,
    observation_capacity: usize,
});
arm_transport_struct!(crate::predictor_v2::ReconciliationSnapshot {
    incomplete: Option<crate::predictor_v2::AccountingError>,
    emitted: u64,
    terminal_predictions: u64,
    live_predictions: u64,
    admission_pending: u64,
    accepted: u64,
    rejected: u64,
    skipped: u64,
    terminal_categories: Vec<(TerminalReason, u64)>,
    source_leaders: u64,
    source_followers: u64,
    source_completed: u64,
    source_failed: u64,
    source_cancelled: u64,
    source_live: u64,
    reservations: u64,
    reservations_committed: u64,
    reservations_aborted: u64,
    reservations_live: u64,
    install_owners: u64,
    install_followers: u64,
    available_installs: u64,
    direct_matching_demand_credits: u64,
});
arm_transport_struct!(P1jLaunchSnapshot {
    launch_considered: u64,
    source_first_attempt_clean: u64,
    source_checkpoint_recovered_clean: u64,
    source_not_eligible: u64,
    p1j_not_ready: u64,
    no_pending_freeze: u64,
    pending_sidecar_existing: u64,
    retirement_busy: u64,
    candidate_identity_invalid: u64,
    freeze_incomplete: u64,
    candidate_already_current_at_f: u64,
    physical_evidence_missing_or_invalid: u64,
    not_logical_materialized: u64,
    missing_logical_generation: u64,
    host_lease_busy: u64,
    host_lease_missing: u64,
    host_lease_stale: u64,
    host_lease_wrong_payload_kind: u64,
    host_lease_wrong_dtype: u64,
    host_lease_wrong_length: u64,
    sidecar_lock_busy: u64,
    sidecar_occupied: u64,
    sidecar_identity_rejected: u64,
    sidecar_epoch_exhausted: u64,
    sidecar_writer_sequence_exhausted: u64,
    p0_acquire_failed: u64,
    writer_spawned: u64,
    incomplete: bool,
});
arm_transport_enum!(crate::gpu_native_token_loop::P1jRequestMode {
    InertResourceOnly,
    Active,
});
arm_transport_enum!(crate::predictor_v2::AccountingError {
    InvalidIdentity,
    InvalidTransition,
    ReusedIdentity,
    Overflow,
    Capacity,
    Incomplete,
    Closed,
});

struct PreparedArm {
    spec: crate::ResolvedRealCliSpec,
    provenance: ArmProvenance,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmReport {
    #[serde(deserialize_with = "Deserialize::deserialize")]
    arm: Arm,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    provenance: Option<ArmProvenance>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    runtime_build_attempted: bool,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    runtime_constructed: bool,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    runtime_resolved_config_sha256: Option<String>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    model_load: Option<crate::greedy_parity::ModelLoadEvidence>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    adapter: Option<crate::backend::GpuDeviceIdentity>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    runtime_contract: Option<evidence::RuntimeContractEvidence>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    generated_token_ids: Vec<u32>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    generated_token_ids_sha256: String,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    completed_positions: usize,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    observation_capacity_per_collection: usize,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    raw_p1e_report: Option<p1e::Report>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    predictor_v2_snapshot: Option<ReconciliationSnapshot>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    p1j_launch_before: Option<P1jLaunchSnapshot>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    p1j_launch_after: Option<P1jLaunchSnapshot>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    p1j_launch_delta: Option<P1jLaunchSnapshot>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    runtime_counters: Option<evidence::RequestSnapshots>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    errors: Vec<String>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    ordinary_invariants_pass: bool,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    mode: P1jRequestMode,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    resources_before_opt_in: Option<crate::gpu_native_residency::P1qResourceSnapshot>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    initial: Option<P1qInitialSnapshot>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    resource_identity: Option<Value>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    sidecar_freshly_initialized: bool,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    initial_state_pass: bool,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    request_wall_ns: Option<u64>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    generated_tps: Option<f64>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    planned_position_tps: Option<f64>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    production_install_after: Option<Value>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
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
    child_request(args)?;
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
    child_processes: Vec<ChildProcessEvidence>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChildRequest {
    pair_index: usize,
    arm: Arm,
}

fn child_request(args: &CommandArgs) -> Result<Option<ChildRequest>> {
    match (
        args.p1q3_child_protocol.as_deref(),
        args.p1q3_child_pair_index,
        args.p1q3_child_arm,
    ) {
        (None, None, None) => Ok(None),
        (Some(CHILD_PROTOCOL), Some(pair), Some(arm)) if (1..=PAIRS).contains(&(pair as usize)) => {
            Ok(Some(ChildRequest { pair_index: pair as usize, arm }))
        }
        _ => Err("invalid or incomplete P1Q3 child protocol; requires exact protocol, pair 1..=6 and control/treatment".into()),
    }
}

impl Arm {
    fn child_arg(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Treatment => "treatment",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildArmArtifact {
    schema: String,
    protocol: String,
    pair_index: usize,
    requested_arm: Arm,
    child_pid: u32,
    executable_sha256: String,
    source_identity: Value,
    request_sha256: String,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    noise_calibration_report_sha256: Option<String>,
    arm_report: ArmReport,
}

// Transport evidence is descriptive only. Artifacts, including malformed or
// failed ones, are deterministically preserved alongside the parent report.
#[derive(Serialize)]
struct ChildProcessEvidence {
    protocol: &'static str,
    pair_index: usize,
    requested_arm: Arm,
    report_path: PathBuf,
    child_pid: Option<u32>,
    exit_code: Option<i32>,
    normal_exit: bool,
    report_sha256: Option<String>,
    transport_error: Option<String>,
}

struct ChildOutcome {
    report: Result<ArmReport>,
    evidence: ChildProcessEvidence,
}

// This synchronous boundary cannot return a successful arm before its child
// has exited. CPU doubles implement the same boundary without spawning MER.
trait ArmProcess {
    fn launch_and_wait(&mut self, request: ChildRequest) -> ChildOutcome;
}

struct ProcessLauncher<'a> {
    args: &'a CommandArgs,
    executable: PathBuf,
    executable_sha256: String,
    sources: Value,
    request_sha256: String,
    calibration_sha256: Option<String>,
}

fn child_report_path(parent: &Path, request: ChildRequest) -> PathBuf {
    let mut path = parent.as_os_str().to_os_string();
    path.push(format!(
        ".p1q3-pair-{}-{}.json",
        request.pair_index,
        request.arm.child_arg()
    ));
    PathBuf::from(path)
}

fn child_command(
    executable: &Path,
    args: &CommandArgs,
    request: ChildRequest,
    report_path: &Path,
) -> std::process::Command {
    let mut command = std::process::Command::new(executable);
    command
        .arg("qualify-gpu-native-predictor-v2-sidecar-performance")
        .arg("--config")
        .arg(&args.config)
        .arg("--request-json")
        .arg(&args.request_json)
        .arg("--expected-adapter-name")
        .arg(&args.expected_adapter_name)
        .arg("--experiment-mode")
        .arg(match args.experiment_mode {
            ExperimentMode::AaNoiseCalibration => "aa-noise-calibration",
            ExperimentMode::AbMovement => "ab-movement",
        });
    if let Some(path) = &args.noise_calibration_report {
        command.arg("--noise-calibration-report").arg(path);
    }
    command
        .arg("--p1q3-child-protocol")
        .arg(CHILD_PROTOCOL)
        .arg("--p1q3-child-pair-index")
        .arg(request.pair_index.to_string())
        .arg("--p1q3-child-arm")
        .arg(request.arm.child_arg())
        .arg("--report-out")
        .arg(report_path);
    // No environment, affinity, thread-count, backend or resource-limit edits.
    command
}

// The pinned serde_json reader does not enable float_roundtrip. A second JSON
// boundary must not change TPS or identity bits before the frozen calibration
// reader sees them. Validate the entire JSON grammar with serde_json, then feed
// the SAME bytes to the typed Deserialize visitor, parsing numeric lexemes with
// Rust's correctly rounded FromStr. This is syntax-only transport: no ArmReport
// fields, identities, counters or statistical semantics are interpreted here.
fn read_child_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice::<serde::de::IgnoredAny>(bytes)?;
    let mut json = ChildJson {
        remaining: std::str::from_utf8(bytes)?,
    };
    let result = T::deserialize(&mut json)?;
    if !json.remaining.trim().is_empty() {
        return Err("trailing child JSON".into());
    }
    Ok(result)
}

struct ChildJson<'de> {
    remaining: &'de str,
}
type JsonError = serde::de::value::Error;
impl ChildJson<'_> {
    fn error(message: impl std::fmt::Display) -> JsonError {
        serde::de::Error::custom(message)
    }
    fn peek(&mut self) -> Option<u8> {
        self.remaining = self.remaining.trim_start();
        self.remaining.as_bytes().first().copied()
    }
    fn take(&mut self, byte: u8) -> std::result::Result<(), JsonError> {
        if self.peek() != Some(byte) {
            return Err(Self::error("unexpected child JSON token"));
        }
        self.remaining = &self.remaining[1..];
        Ok(())
    }
    fn string(&mut self) -> std::result::Result<String, JsonError> {
        if self.peek() != Some(b'"') {
            return Err(Self::error("expected child JSON string"));
        }
        let bytes = self.remaining.as_bytes();
        let mut end = 1;
        while end < bytes.len() {
            match bytes[end] {
                b'\\' => end += 2,
                b'"' => {
                    let value =
                        serde_json::from_str(&self.remaining[..=end]).map_err(Self::error)?;
                    self.remaining = &self.remaining[end + 1..];
                    return Ok(value);
                }
                _ => end += 1,
            }
        }
        Err(Self::error("unterminated child JSON string"))
    }
    fn number(&mut self) -> std::result::Result<&str, JsonError> {
        self.peek();
        let end = self
            .remaining
            .bytes()
            .take_while(|b| matches!(b, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
            .count();
        if end == 0 {
            return Err(Self::error("expected child JSON number"));
        }
        let (number, rest) = self.remaining.split_at(end);
        self.remaining = rest;
        Ok(number)
    }
}

struct ChildJsonAccess<'a, 'de> {
    json: &'a mut ChildJson<'de>,
    first: bool,
}
impl ChildJsonAccess<'_, '_> {
    fn next(&mut self, end: u8) -> std::result::Result<bool, JsonError> {
        if self.json.peek() == Some(end) {
            return Ok(false);
        }
        if !self.first {
            self.json.take(b',')?;
        }
        self.first = false;
        Ok(true)
    }
}
impl<'de> serde::de::MapAccess<'de> for ChildJsonAccess<'_, 'de> {
    type Error = JsonError;
    fn next_key_seed<K: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> std::result::Result<Option<K::Value>, JsonError> {
        if !self.next(b'}')? {
            return Ok(None);
        }
        let key = seed.deserialize(serde::de::value::StringDeserializer::<JsonError>::new(
            self.json.string()?,
        ))?;
        self.json.take(b':')?;
        Ok(Some(key))
    }
    fn next_value_seed<V: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> std::result::Result<V::Value, JsonError> {
        seed.deserialize(&mut *self.json)
    }
}
impl<'de> serde::de::SeqAccess<'de> for ChildJsonAccess<'_, 'de> {
    type Error = JsonError;
    fn next_element_seed<T: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> std::result::Result<Option<T::Value>, JsonError> {
        if !self.next(b']')? {
            return Ok(None);
        }
        seed.deserialize(&mut *self.json).map(Some)
    }
}
impl<'de, 'a> serde::de::EnumAccess<'de> for &'a mut ChildJson<'de> {
    type Error = JsonError;
    type Variant = Self;
    fn variant_seed<V: serde::de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> std::result::Result<(V::Value, Self), JsonError> {
        let variant = seed.deserialize(serde::de::value::StringDeserializer::<JsonError>::new(
            self.string()?,
        ))?;
        self.take(b':')?;
        Ok((variant, self))
    }
}
impl<'de> serde::de::VariantAccess<'de> for &mut ChildJson<'de> {
    type Error = JsonError;
    fn unit_variant(self) -> std::result::Result<(), JsonError> {
        Deserialize::deserialize(self)
    }
    fn newtype_variant_seed<T: serde::de::DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> std::result::Result<T::Value, JsonError> {
        seed.deserialize(self)
    }
    fn tuple_variant<V: serde::de::Visitor<'de>>(
        self,
        _: usize,
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        serde::Deserializer::deserialize_seq(self, visitor)
    }
    fn struct_variant<V: serde::de::Visitor<'de>>(
        self,
        _: &'static [&'static str],
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        serde::Deserializer::deserialize_map(self, visitor)
    }
}
impl<'de> serde::Deserializer<'de> for &mut ChildJson<'de> {
    type Error = JsonError;
    fn deserialize_any<V: serde::de::Visitor<'de>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        match self.peek() {
            Some(b'{') => {
                self.take(b'{')?;
                let value = visitor.visit_map(ChildJsonAccess {
                    json: &mut *self,
                    first: true,
                })?;
                self.take(b'}')?;
                Ok(value)
            }
            Some(b'[') => {
                self.take(b'[')?;
                let value = visitor.visit_seq(ChildJsonAccess {
                    json: &mut *self,
                    first: true,
                })?;
                self.take(b']')?;
                Ok(value)
            }
            Some(b'"') => visitor.visit_string(self.string()?),
            Some(b'n') => {
                self.remaining = &self.remaining[4..];
                visitor.visit_unit()
            }
            Some(b't') => {
                self.remaining = &self.remaining[4..];
                visitor.visit_bool(true)
            }
            Some(b'f') => {
                self.remaining = &self.remaining[5..];
                visitor.visit_bool(false)
            }
            Some(b'-' | b'0'..=b'9') => {
                let number = self.number()?;
                if number.contains(['.', 'e', 'E']) || number == "-0" {
                    let value: f64 = number.parse().map_err(ChildJson::error)?;
                    if !value.is_finite() {
                        return Err(ChildJson::error("non-finite child JSON number"));
                    }
                    visitor.visit_f64(value)
                } else if number.starts_with('-') {
                    visitor.visit_i64(number.parse().map_err(ChildJson::error)?)
                } else {
                    visitor.visit_u64(number.parse().map_err(ChildJson::error)?)
                }
            }
            _ => Err(ChildJson::error("invalid child JSON token")),
        }
    }
    fn deserialize_f32<V: serde::de::Visitor<'de>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        let value: f32 = self.number()?.parse().map_err(ChildJson::error)?;
        if !value.is_finite() {
            return Err(ChildJson::error("non-finite child f32"));
        }
        visitor.visit_f32(value)
    }
    fn deserialize_f64<V: serde::de::Visitor<'de>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        let value: f64 = self.number()?.parse().map_err(ChildJson::error)?;
        if !value.is_finite() {
            return Err(ChildJson::error("non-finite child f64"));
        }
        visitor.visit_f64(value)
    }
    fn deserialize_option<V: serde::de::Visitor<'de>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        if self.peek() == Some(b'n') {
            self.remaining = &self.remaining[4..];
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }
    fn deserialize_newtype_struct<V: serde::de::Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        visitor.visit_newtype_struct(self)
    }
    fn deserialize_enum<V: serde::de::Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> std::result::Result<V::Value, JsonError> {
        if self.peek() == Some(b'"') {
            visitor.visit_enum(serde::de::value::StringDeserializer::<JsonError>::new(
                self.string()?,
            ))
        } else {
            self.take(b'{')?;
            let value = visitor.visit_enum(&mut *self)?;
            self.take(b'}')?;
            Ok(value)
        }
    }
    serde::forward_to_deserialize_any! { bool i8 i16 i32 i64 u8 u16 u32 u64 char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier ignored_any }
}

fn validate_child_artifact(
    bytes: &[u8],
    request: ChildRequest,
    process: &ChildProcessEvidence,
    launcher: &ProcessLauncher<'_>,
) -> Result<ArmReport> {
    if !process.normal_exit || process.exit_code != Some(0) {
        return Err(format!(
            "child did not exit normally and successfully: exit_code={:?}",
            process.exit_code
        )
        .into());
    }
    // Validate and decode the complete typed document without numeric drift.
    let artifact: ChildArmArtifact = read_child_json(bytes)?;
    if artifact.schema != CHILD_PROTOCOL
        || artifact.protocol != CHILD_PROTOCOL
        || artifact.pair_index != request.pair_index
        || artifact.requested_arm != request.arm
        || Some(artifact.child_pid) != process.child_pid
        || artifact.child_pid == 0
        || artifact.executable_sha256 != launcher.executable_sha256
        || artifact.source_identity != launcher.sources
        || artifact.request_sha256 != launcher.request_sha256
        || artifact.noise_calibration_report_sha256 != launcher.calibration_sha256
        || artifact.arm_report.arm != request.arm
        || artifact.arm_report.mode != launcher.args.experiment_mode.request_mode(request.arm)
    {
        return Err("P1Q3 child schema/protocol/pair/arm/PID/executable/source/request/calibration/mode identity mismatch".into());
    }
    Ok(artifact.arm_report)
}

impl ArmProcess for ProcessLauncher<'_> {
    fn launch_and_wait(&mut self, request: ChildRequest) -> ChildOutcome {
        let report_path = child_report_path(&self.args.report_out, request);
        let mut evidence = ChildProcessEvidence {
            protocol: CHILD_PROTOCOL,
            pair_index: request.pair_index,
            requested_arm: request.arm,
            report_path,
            child_pid: None,
            exit_code: None,
            normal_exit: false,
            report_sha256: None,
            transport_error: None,
        };
        let report = (|| -> Result<ArmReport> {
            ensure_output_absent(&evidence.report_path)?;
            let mut child =
                child_command(&self.executable, self.args, request, &evidence.report_path)
                    .spawn()?;
            evidence.child_pid = Some(child.id());
            let status = child.wait()?;
            evidence.exit_code = status.code();
            evidence.normal_exit = status.success();
            // One immutable snapshot supplies both the retained hash and parser.
            let bytes = std::fs::read(&evidence.report_path)?;
            evidence.report_sha256 = Some(crate::greedy_parity::sha256_hex(&bytes));
            validate_child_artifact(&bytes, request, &evidence, self)
        })();
        evidence.transport_error = report.as_ref().err().map(ToString::to_string);
        ChildOutcome { report, evidence }
    }
}

fn reconcile_child_identity(
    arm: &mut ArmReport,
    provenance: &mut Option<Value>,
    resources: &mut Option<Value>,
    calibration: Option<&Calibration>,
) {
    let result = (|| -> Result<()> {
        let input = arm
            .provenance
            .as_ref()
            .ok_or("child input provenance unavailable")?;
        let current = serde_json::to_value(input)?;
        let identity = arm
            .resource_identity
            .as_ref()
            .ok_or("child resource identity unavailable")?;
        if provenance.as_ref().is_some_and(|first| first != &current)
            || resources.as_ref().is_some_and(|first| first != identity)
        {
            return Err(
                "cross-process input provenance or structural resource identity drift".into(),
            );
        }
        if let Some(c) = calibration {
            if !serialized_identity_matches(&c.input_provenance, input)?
                || !serialized_identity_matches(&c.resource_identity, identity)?
            {
                return Err("cross-process calibration provenance/resource identity drift".into());
            }
        }
        if provenance.is_none() {
            *provenance = Some(current);
        }
        if resources.is_none() {
            *resources = Some(identity.clone());
        }
        Ok(())
    })();
    if let Err(error) = result {
        arm.errors.push(error.to_string());
        arm.ordinary_invariants_pass = false;
    }
}

fn accept_child(
    report: &mut Report,
    calibration: Option<&Calibration>,
    process: &mut impl ArmProcess,
    request: ChildRequest,
) -> ArmReport {
    let outcome = process.launch_and_wait(request);
    report.child_processes.push(outcome.evidence);
    match outcome.report {
        Ok(mut arm) => {
            reconcile_child_identity(
                &mut arm,
                &mut report.input_provenance,
                &mut report.resource_identity,
                calibration,
            );
            arm
        }
        Err(error) => {
            // No reconstruction of missing/invalid child evidence or retry.
            let mut arm = ArmReport::new(request.arm);
            arm.mode = report.experiment_mode.request_mode(request.arm);
            arm.errors
                .push(format!("P1Q3 child transport failed: {error}"));
            arm
        }
    }
}

fn run_pairs(
    report: &mut Report,
    calibration: Option<&Calibration>,
    process: &mut impl ArmProcess,
) {
    for index in 0..PAIRS {
        let order = execution_order(index);
        let first = accept_child(
            report,
            calibration,
            process,
            ChildRequest {
                pair_index: index + 1,
                arm: order[0],
            },
        );
        let second = if shutdown_complete(&first) {
            accept_child(
                report,
                calibration,
                process,
                ChildRequest {
                    pair_index: index + 1,
                    arm: order[1],
                },
            )
        } else {
            let mut skipped = ArmReport::new(order[1]);
            skipped.mode = report.experiment_mode.request_mode(order[1]);
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
        let pair = finish_pair(index, report.experiment_mode, control, treatment);
        let valid = pair.pair_valid;
        report.pairs.push(pair);
        // Retain the failed pair, then stop. Never continue and select survivors.
        if !valid {
            break;
        }
    }
}

async fn run_child(
    args: &CommandArgs,
    request: ChildRequest,
    request_bytes: &[u8],
    launcher: &ProcessLauncher<'_>,
    calibration: Option<&Calibration>,
) -> Result<()> {
    let arm_report = run_arm(
        args,
        request_bytes,
        request.arm,
        &mut None,
        &mut None,
        calibration,
    )
    .await;
    let artifact = ChildArmArtifact {
        schema: CHILD_PROTOCOL.into(),
        protocol: CHILD_PROTOCOL.into(),
        pair_index: request.pair_index,
        requested_arm: request.arm,
        child_pid: std::process::id(),
        executable_sha256: launcher.executable_sha256.clone(),
        source_identity: launcher.sources.clone(),
        request_sha256: launcher.request_sha256.clone(),
        noise_calibration_report_sha256: launcher.calibration_sha256.clone(),
        arm_report,
    };
    // Qualification errors belong to the complete arm. A successful exit means
    // only that its transport artifact was published after isolated shutdown.
    write_report(&args.report_out, &artifact)
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
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let mut launcher = ProcessLauncher {
        args: &args,
        executable,
        executable_sha256,
        sources: sources.clone(),
        request_sha256: request_sha.clone(),
        calibration_sha256: calibration.as_ref().map(|c| c.bytes_sha256.clone()),
    };
    if let Some(request) = child_request(&args)? {
        return run_child(
            &args,
            request,
            &request_bytes,
            &launcher,
            calibration.as_ref(),
        )
        .await;
    }
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
        child_processes: Vec::with_capacity(PAIRS * 2),
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
    run_pairs(&mut report, calibration.as_ref(), &mut launcher);
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
            child_processes: vec![],
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
            p1q3_child_protocol: None,
            p1q3_child_pair_index: None,
            p1q3_child_arm: None,
        }
    }

    mod p1q3 {
        use super::*;
        use clap::{CommandFactory, Parser};

        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            args: CommandArgs,
        }

        fn argv() -> Vec<&'static str> {
            vec![
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
            ]
        }

        fn provenance() -> ArmProvenance {
            ArmProvenance {
                provenance: evidence::BenchmarkProvenance {
                    build: crate::qualification::BuildProvenance {
                        git_sha: Some("frozen-source".into()),
                        dirty: Some(false),
                        package_version: "test".into(),
                    },
                    executable_canonical_path: "/fixture/mer".into(),
                    executable_sha256: "executable".into(),
                    resolved_config_sha256: "resolved".into(),
                    artifacts: crate::qualification::QualificationArtifacts {
                        config: Some(crate::qualification::ArtifactDigest {
                            configured_path: "config".into(),
                            canonical_path: "/fixture/config".into(),
                            byte_length: 123,
                            sha256: "config-hash".into(),
                        }),
                        ..Default::default()
                    },
                    expert_metadata: crate::qualification::ExpertMetadataEvidence {
                        dtype: Some("q4_0".into()),
                        q4_0_layout: Some("ggml".into()),
                        conversion_mode: None,
                        source: Some("fixture".into()),
                        explicitly_synthetic: false,
                    },
                },
                config_path: "/fixture/config".into(),
                config_sha256: "config-hash".into(),
                model_identity: crate::greedy_parity::ModelIdentityEvidence {
                    architecture: "qwen3_moe".into(),
                    num_layers: 48,
                    num_experts_per_layer: 128,
                    total_experts: 6144,
                    top_k: 8,
                    d_model: 2048,
                    d_ff: 768,
                    routed_expert_dtype: "q4_0".into(),
                },
                production_configuration: evidence::ProductionConfiguration {
                    cache_residency: evidence::CacheResidencyConfiguration {
                        gpu_vram_anchor_ratio: 0.6,
                        partial_load_fraction: 0.75,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            }
        }

        fn transported_arm(mode: ExperimentMode, which: Arm) -> ArmReport {
            let mut r = arm(mode, which);
            r.provenance = Some(provenance());
            r.runtime_build_attempted = true;
            r.runtime_constructed = true;
            r.runtime_resolved_config_sha256 = Some("resolved".into());
            r.model_load = Some(crate::greedy_parity::ModelLoadEvidence {
                strict: true,
                loader: "fixture".into(),
                loaded_tensors: 1,
                required_tensors: 1,
                optional_probed: 0,
                optional_loaded: 0,
                seeded_fallback_remained: false,
            });
            r.adapter = Some(crate::backend::GpuDeviceIdentity {
                name: "NVIDIA L4".into(),
                wgpu_backend: "vulkan".into(),
                device_type: "DiscreteGpu".into(),
                vendor_id: 4318,
                device_id: 10168,
                driver: "driver".into(),
                driver_info: "fixture".into(),
                compute_plane: "gpu".into(),
                software_adapter: false,
            });
            r.runtime_contract = Some(super::p1q1::runtime_contract("raw-context"));
            r.runtime_counters = Some(evidence::RequestSnapshots {
                token_loop_before: Default::default(),
                token_loop_after: Default::default(),
                token_loop_delta: Default::default(),
                token_loop_ratios: Default::default(),
                recovery_before: Default::default(),
                recovery_after: Default::default(),
                recovery_delta: Default::default(),
                recovery_ratios: Default::default(),
                routed_execution_before: Default::default(),
                routed_execution_after: Default::default(),
                routed_execution_delta: Default::default(),
                runtime_cache_before: Default::default(),
                runtime_cache_after: Default::default(),
                engine_storage_before: Default::default(),
                engine_storage_after: Default::default(),
                engine_storage_delta: Default::default(),
                gpu_expert_io_before: Default::default(),
                gpu_expert_io_after: Default::default(),
                gpu_expert_io_delta: Default::default(),
                gpu_expert_memory_before: Default::default(),
                gpu_expert_memory_after: Default::default(),
                gpu_native_residency_before: Default::default(),
                gpu_native_residency_after: Default::default(),
                gpu_native_residency_delta: Default::default(),
            });
            r.production_install_after = Some(json!({"sentinel":17}));
            r.production_install_delta = Some(json!({"sentinel":13}));
            r
        }

        fn launcher(args: &CommandArgs) -> ProcessLauncher<'_> {
            ProcessLauncher {
                args,
                executable: "/fixture/mer".into(),
                executable_sha256: "executable".into(),
                sources: json!({"fixture":"source"}),
                request_sha256: "request".into(),
                calibration_sha256: None,
            }
        }

        fn envelope(request: ChildRequest, mode: ExperimentMode) -> ChildArmArtifact {
            ChildArmArtifact {
                schema: CHILD_PROTOCOL.into(),
                protocol: CHILD_PROTOCOL.into(),
                pair_index: request.pair_index,
                requested_arm: request.arm,
                child_pid: 101,
                executable_sha256: "executable".into(),
                source_identity: json!({"fixture":"source"}),
                request_sha256: "request".into(),
                noise_calibration_report_sha256: None,
                arm_report: transported_arm(mode, request.arm),
            }
        }

        fn process_evidence(request: ChildRequest) -> ChildProcessEvidence {
            ChildProcessEvidence {
                protocol: CHILD_PROTOCOL,
                pair_index: request.pair_index,
                requested_arm: request.arm,
                report_path: child_report_path(Path::new("report"), request),
                child_pid: Some(101),
                exit_code: Some(0),
                normal_exit: true,
                report_sha256: None,
                transport_error: None,
            }
        }

        #[test]
        fn public_cli_and_hidden_protocol_fail_closed() {
            let normal = Cli::try_parse_from(argv()).unwrap();
            assert!(validate_args(&normal.args).is_ok());
            assert_eq!(child_request(&normal.args).unwrap(), None);
            let help = Cli::command().render_long_help().to_string();
            assert!(!help.contains("p1q3-child"));
            let mut ab = argv();
            ab[8] = "ab-movement";
            ab.extend(["--noise-calibration-report", "aa.json"]);
            assert!(validate_args(&Cli::try_parse_from(ab).unwrap().args).is_ok());
            let fields = [
                ["--p1q3-child-protocol", CHILD_PROTOCOL],
                ["--p1q3-child-pair-index", "1"],
                ["--p1q3-child-arm", "control"],
            ];
            for mask in 1..7 {
                let mut values = argv();
                for (i, pair) in fields.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        values.extend(pair);
                    }
                }
                assert!(Cli::try_parse_from(values).is_err(), "mask {mask}");
                let mut raw = args(ExperimentMode::AaNoiseCalibration);
                if mask & 1 != 0 {
                    raw.p1q3_child_protocol = Some(CHILD_PROTOCOL.into());
                }
                if mask & 2 != 0 {
                    raw.p1q3_child_pair_index = Some(1);
                }
                if mask & 4 != 0 {
                    raw.p1q3_child_arm = Some(Arm::Control);
                }
                assert!(validate_args(&raw).is_err());
            }
            for pair in 1..=6 {
                for arm in ["control", "treatment"] {
                    let index = pair.to_string();
                    let mut values = argv();
                    values.extend([
                        "--p1q3-child-protocol",
                        CHILD_PROTOCOL,
                        "--p1q3-child-pair-index",
                        &index,
                        "--p1q3-child-arm",
                        arm,
                    ]);
                    let parsed = Cli::try_parse_from(values).unwrap().args;
                    assert_eq!(child_request(&parsed).unwrap().unwrap().pair_index, pair);
                    assert!(validate_args(&parsed).is_ok());
                }
            }
        }

        #[test]
        fn invalid_pair_arm_protocol_rejected() {
            for (pair, arm, protocol) in [
                ("0", "control", CHILD_PROTOCOL),
                ("7", "control", CHILD_PROTOCOL),
                ("-1", "control", CHILD_PROTOCOL),
                ("256", "control", CHILD_PROTOCOL),
                ("1", "A", CHILD_PROTOCOL),
                ("1", "Control", CHILD_PROTOCOL),
                ("1", "other", CHILD_PROTOCOL),
                ("1", "control", "other"),
            ] {
                let mut values = argv();
                values.extend([
                    "--p1q3-child-protocol",
                    protocol,
                    "--p1q3-child-pair-index",
                    pair,
                    "--p1q3-child-arm",
                    arm,
                ]);
                assert!(Cli::try_parse_from(values)
                    .map(|c| validate_args(&c.args).is_err())
                    .unwrap_or(true));
            }
        }

        #[test]
        fn complete_typed_arm_roundtrip_including_failure_and_nullable_fields() {
            for mode in [
                ExperimentMode::AaNoiseCalibration,
                ExperimentMode::AbMovement,
            ] {
                let mut original = transported_arm(mode, Arm::Treatment);
                original.errors.push("retained failure".into());
                original.predictor_v2_snapshot.as_mut().unwrap().incomplete =
                    Some(crate::predictor_v2::AccountingError::Overflow);
                let bytes = serde_json::to_vec(&original).unwrap();
                let parsed: ArmReport = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(serde_json::to_vec(&parsed).unwrap(), bytes);
                assert_eq!(parsed.request_wall_ns, original.request_wall_ns);
                let value = serde_json::to_value(&original).unwrap();
                for key in value.as_object().unwrap().keys() {
                    let mut incomplete = value.clone();
                    incomplete.as_object_mut().unwrap().remove(key);
                    assert!(
                        serde_json::from_value::<ArmReport>(incomplete).is_err(),
                        "missing {key}"
                    );
                }
            }
        }

        #[test]
        fn typed_transport_preserves_nontrivial_tps_and_calibration_identity() {
            let mut original = transported_arm(ExperimentMode::AaNoiseCalibration, Arm::Control);
            original.request_wall_ns = Some(1_000_007_321);
            let (g, p) = throughput(original.request_wall_ns.unwrap()).unwrap();
            original.generated_tps = Some(g);
            original.planned_position_tps = Some(p);
            original
                .provenance
                .as_mut()
                .unwrap()
                .production_configuration
                .cache_residency
                .partial_load_fraction = 0.12799906291886037;
            original.resource_identity.as_mut().unwrap()["nontrivial_float"] =
                json!(0.12799906291886037);
            let bytes = serde_json::to_vec(&original).unwrap();
            let parsed: ArmReport = read_child_json(&bytes).unwrap();
            assert_eq!(parsed.generated_tps, original.generated_tps);
            assert_eq!(parsed.planned_position_tps, original.planned_position_tps);
            assert_eq!(serde_json::to_vec(&parsed).unwrap(), bytes);
        }

        #[test]
        fn lossless_json_numbers_containers_enums_and_invalid_documents() {
            #[derive(Debug, PartialEq, Serialize, Deserialize)]
            enum Shape {
                Unit,
                Newtype(u64),
                Tuple(f64, String),
                Struct { fixed: [u32; 2] },
            }
            for value in [
                Shape::Unit,
                Shape::Newtype(u64::MAX),
                Shape::Tuple(127.99906291886037, "escaped \" \\ 日本語".into()),
                Shape::Struct { fixed: [7, 9] },
            ] {
                let bytes = serde_json::to_vec(&value).unwrap();
                assert_eq!(read_child_json::<Shape>(&bytes).unwrap(), value);
            }
            for value in [
                f64::MIN_POSITIVE,
                f64::MAX,
                0.0,
                -0.0,
                0.12799906291886037,
                1e-300,
                2.718281828459045,
            ] {
                let bytes = serde_json::to_vec(&value).unwrap();
                assert_eq!(
                    read_child_json::<f64>(&bytes).unwrap().to_bits(),
                    value.to_bits()
                );
            }
            for value in [f32::MIN_POSITIVE, f32::MAX, 0.0, -0.0, 0.6] {
                let bytes = serde_json::to_vec(&value).unwrap();
                assert_eq!(
                    read_child_json::<f32>(&bytes).unwrap().to_bits(),
                    value.to_bits()
                );
            }
            assert_eq!(read_child_json::<[u8; 2]>(b"[1,2]").unwrap(), [1, 2]);
            assert!(read_child_json::<[u8; 2]>(b"[1,2,3]").is_err());
            for bad in [
                b"[1,]".as_slice(),
                b"{\"x\":1,}".as_slice(),
                b"NaN".as_slice(),
                b"01".as_slice(),
                b"1 2".as_slice(),
                b"[".as_slice(),
            ] {
                assert!(read_child_json::<Value>(bad).is_err());
            }
            assert!(read_child_json::<f64>(b"1e400").is_err());
            assert!(read_child_json::<f32>(b"1e100").is_err());
            assert!(read_child_json::<Shape>(br#"{"Newtype":1,"Newtype":2}"#).is_err());
        }

        #[test]
        fn transported_aa_calibration_keeps_original_integer_ns_rules() {
            let (mut report, _) = schedule(ExperimentMode::AaNoiseCalibration, None);
            for (index, pair) in report.pairs.iter_mut().enumerate() {
                for (arm, multiplier) in [(&mut pair.control, 7321), (&mut pair.treatment, 4123)] {
                    let ns = 1_000_000_000 + (index as u64 + 1) * multiplier;
                    arm.request_wall_ns = Some(ns);
                    let (g, p) = throughput(ns).unwrap();
                    arm.generated_tps = Some(g);
                    arm.planned_position_tps = Some(p);
                    *arm = read_child_json(&serde_json::to_vec(&arm).unwrap()).unwrap();
                }
                pair.paired_generated_tps_delta_pct = Some(
                    paired_delta(
                        pair.control.request_wall_ns.unwrap(),
                        pair.treatment.request_wall_ns.unwrap(),
                    )
                    .unwrap(),
                );
            }
            finish_experiment(&mut report).unwrap();
            let bytes = serde_json::to_vec(&report).unwrap();
            let calibration =
                validate_calibration(&bytes, "request", &json!({"fixture":"source"})).unwrap();
            let mut arm = transported_arm(ExperimentMode::AbMovement, Arm::Treatment);
            reconcile_child_identity(&mut arm, &mut None, &mut None, Some(&calibration));
            assert!(arm.errors.is_empty(), "{:?}", arm.errors);
            assert_eq!(calibration.floor, 2.0);
        }

        #[test]
        fn exact_artifact_identity_complete_parse_and_normal_exit() {
            let args = args(ExperimentMode::AaNoiseCalibration);
            let launcher = launcher(&args);
            let request = ChildRequest {
                pair_index: 1,
                arm: Arm::Control,
            };
            let artifact = envelope(request, args.experiment_mode);
            let bytes = serde_json::to_vec(&artifact).unwrap();
            let process = process_evidence(request);
            assert!(validate_child_artifact(&bytes, request, &process, &launcher).is_ok());
            for (path, value) in [
                ("/schema", json!("wrong")),
                ("/protocol", json!("wrong")),
                ("/pair_index", json!(2)),
                ("/requested_arm", json!("Treatment")),
                ("/child_pid", json!(102)),
                ("/executable_sha256", json!("wrong")),
                ("/source_identity/fixture", json!("wrong")),
                ("/request_sha256", json!("wrong")),
                ("/noise_calibration_report_sha256", json!("wrong")),
                ("/arm_report/arm", json!("Treatment")),
                ("/arm_report/mode", json!("Active")),
            ] {
                let mut v = serde_json::to_value(&artifact).unwrap();
                *v.pointer_mut(path).unwrap() = value;
                assert!(
                    validate_child_artifact(
                        &serde_json::to_vec(&v).unwrap(),
                        request,
                        &process,
                        &launcher
                    )
                    .is_err(),
                    "{path}"
                );
            }
            for extra in [b"{}".as_slice(), b"garbage".as_slice()] {
                let mut trailing = bytes.clone();
                trailing.extend(extra);
                assert!(validate_child_artifact(&trailing, request, &process, &launcher).is_err());
            }
            assert!(validate_child_artifact(
                &bytes[..bytes.len() - 1],
                request,
                &process,
                &launcher
            )
            .is_err());
            for (normal, code, pid) in [
                (false, None, Some(101)),
                (false, Some(1), Some(101)),
                (true, Some(1), Some(101)),
                (true, Some(0), None),
            ] {
                let mut p = process_evidence(request);
                p.normal_exit = normal;
                p.exit_code = code;
                p.child_pid = pid;
                assert!(validate_child_artifact(&bytes, request, &p, &launcher).is_err());
            }
        }

        #[test]
        fn exact_command_paths_calibration_and_unchanged_environment() {
            for mode in [
                ExperimentMode::AaNoiseCalibration,
                ExperimentMode::AbMovement,
            ] {
                let mut args = args(mode);
                args.config = "some config.toml".into();
                args.request_json = "request with spaces.json".into();
                if mode == ExperimentMode::AbMovement {
                    args.noise_calibration_report = Some("frozen aa.json".into());
                }
                let request = ChildRequest {
                    pair_index: 6,
                    arm: Arm::Treatment,
                };
                let output = child_report_path(&args.report_out, request);
                let command =
                    child_command(Path::new("/exact/executable"), &args, request, &output);
                assert_eq!(command.get_program(), "/exact/executable");
                assert_eq!(command.get_envs().count(), 0);
                assert!(command.get_current_dir().is_none());
                let mut argv = vec![std::ffi::OsString::from("p1q")];
                let full: Vec<_> = command.get_args().collect();
                assert_eq!(
                    full[0],
                    "qualify-gpu-native-predictor-v2-sidecar-performance"
                );
                argv.extend(full[1..].iter().map(|s| s.to_os_string()));
                let child = Cli::try_parse_from(argv).unwrap().args;
                assert_eq!(child.config, args.config);
                assert_eq!(child.request_json, args.request_json);
                assert_eq!(
                    child.noise_calibration_report,
                    args.noise_calibration_report
                );
                assert_eq!(child.experiment_mode, mode);
                assert_eq!(child.expected_adapter_name, "NVIDIA L4");
                assert_eq!(child.report_out, output);
                assert_eq!(child_request(&child).unwrap(), Some(request));
            }
            let paths: std::collections::BTreeSet<_> = (1..=PAIRS)
                .flat_map(|pair_index| {
                    [Arm::Control, Arm::Treatment].map(move |arm| {
                        child_report_path(Path::new("report"), ChildRequest { pair_index, arm })
                    })
                })
                .collect();
            assert_eq!(paths.len(), 12);
        }

        struct Double {
            mode: ExperimentMode,
            events: Vec<(&'static str, ChildRequest)>,
            fault: Option<(usize, &'static str)>,
        }
        impl ArmProcess for Double {
            fn launch_and_wait(&mut self, request: ChildRequest) -> ChildOutcome {
                assert!(self.events.last().is_none_or(|(e, _)| *e == "exit"));
                let n = self.events.len() / 2 + 1;
                self.events.push(("spawn", request));
                let mut envelope = envelope(request, self.mode);
                let fault = self.fault.filter(|(index, _)| *index == n).map(|(_, s)| s);
                match fault {
                    Some("shutdown") => {
                        envelope
                            .arm_report
                            .runtime_shutdown
                            .as_mut()
                            .unwrap()
                            .all_runtime_resources_released = false
                    }
                    Some("invalid") => envelope
                        .arm_report
                        .generated_token_ids
                        .pop()
                        .map(|_| ())
                        .unwrap(),
                    Some("provenance") => {
                        envelope
                            .arm_report
                            .provenance
                            .as_mut()
                            .unwrap()
                            .config_sha256 = "drift".into()
                    }
                    Some("resources") => {
                        envelope.arm_report.resource_identity.as_mut().unwrap()["config"] =
                            json!("drift")
                    }
                    Some("adapter") => {
                        let arm = &mut envelope.arm_report;
                        let raw = arm.runtime_contract.as_ref().unwrap();
                        arm.adapter.as_mut().unwrap().name = "NVIDIA L4/PCIe/SSE2".into();
                        arm.adapter.as_mut().unwrap().wgpu_backend = "gl".into();
                        let input = evidence::RuntimeContractInput {
                            real_transformer_enabled: true,
                            real_transformer_gpu_native: true,
                            compute_offload: crate::backend::ComputeOffload::Gpu,
                            legacy_execution_plan: raw.legacy_execution_plan.clone(),
                            token_loop_geometry: Some(raw.token_loop_geometry),
                            authoritative_device: arm.adapter.clone(),
                            model_load: arm.model_load.clone().unwrap(),
                            routed_failure_policy:
                                crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed,
                        };
                        let error =
                            evidence::validate_runtime_contract(&input, "NVIDIA L4").unwrap_err();
                        arm.errors.push(error.to_string());
                        arm.ordinary_invariants_pass = false;
                    }
                    _ => {}
                }
                self.events.push(("exit", request));
                let args = args(self.mode);
                let launcher = launcher(&args);
                let mut evidence = process_evidence(request);
                let bytes = serde_json::to_vec(&envelope).unwrap();
                evidence.report_sha256 = Some(crate::greedy_parity::sha256_hex(&bytes));
                let report = if fault == Some("transport") {
                    Err("exact simulated transport failure".into())
                } else {
                    validate_child_artifact(&bytes, request, &evidence, &launcher)
                };
                evidence.transport_error = report.as_ref().err().map(ToString::to_string);
                ChildOutcome { report, evidence }
            }
        }
        fn schedule(
            mode: ExperimentMode,
            fault: Option<(usize, &'static str)>,
        ) -> (Report, Double) {
            let mut report = experiment(mode);
            report.pairs.clear();
            report.input_provenance = None;
            report.resource_identity = None;
            let mut process = Double {
                mode,
                events: vec![],
                fault,
            };
            run_pairs(&mut report, None, &mut process);
            finish_experiment(&mut report).unwrap();
            (report, process)
        }

        #[test]
        fn twelve_children_alternate_and_exit_before_next_launch() {
            for mode in [
                ExperimentMode::AaNoiseCalibration,
                ExperimentMode::AbMovement,
            ] {
                let (report, double) = schedule(mode, None);
                assert_eq!(report.pairs.len(), 6);
                assert_eq!(report.child_processes.len(), 12);
                assert!(report.qualification_pass);
                assert_eq!(double.events.len(), 24);
                let expected = [
                    Arm::Control,
                    Arm::Treatment,
                    Arm::Treatment,
                    Arm::Control,
                    Arm::Control,
                    Arm::Treatment,
                    Arm::Treatment,
                    Arm::Control,
                    Arm::Control,
                    Arm::Treatment,
                    Arm::Treatment,
                    Arm::Control,
                ];
                for (index, events) in double.events.chunks_exact(2).enumerate() {
                    let request = ChildRequest {
                        pair_index: index / 2 + 1,
                        arm: expected[index],
                    };
                    assert_eq!(events, [("spawn", request), ("exit", request)]);
                }
                for pair in &report.pairs {
                    assert_eq!(pair.control.request_wall_ns, Some(1_000_000_000));
                    assert_eq!(pair.treatment.request_wall_ns, Some(1_000_000_000));
                }
            }
        }

        #[test]
        fn transport_failure_retained_and_no_future_children() {
            for failure in [1, 2, 3, 12] {
                let (report, double) = schedule(
                    ExperimentMode::AaNoiseCalibration,
                    Some((failure, "transport")),
                );
                assert_eq!(report.child_processes.len(), failure);
                assert_eq!(double.events.len(), failure * 2);
                assert_eq!(report.pairs.len(), (failure + 1) / 2);
                assert!(!report.qualification_pass);
                assert_eq!(
                    report
                        .child_processes
                        .last()
                        .unwrap()
                        .transport_error
                        .as_deref(),
                    Some("exact simulated transport failure")
                );
                assert_eq!(report.performance_verdict, "NOT_AUTHORIZED");
            }
        }

        #[test]
        fn invalid_pair_retained_without_later_pairs() {
            let (report, process) =
                schedule(ExperimentMode::AaNoiseCalibration, Some((2, "invalid")));
            assert_eq!(report.pairs.len(), 1);
            assert_eq!(process.events.len(), 4);
            assert!(!report.pairs[0].pair_valid);
        }

        #[test]
        fn unproven_first_shutdown_skips_second_even_after_normal_process_exit() {
            let (report, process) =
                schedule(ExperimentMode::AaNoiseCalibration, Some((1, "shutdown")));
            assert_eq!(process.events.len(), 2);
            assert!(report.child_processes[0].normal_exit);
            assert!(!report.qualification_pass);
            assert_eq!(report.pairs.len(), 1);
            assert_eq!(
                report.pairs[0].treatment.errors,
                vec!["not constructed: previous arm shutdown unproven"]
            );
        }

        #[test]
        fn cross_pair_provenance_and_resources_fail() {
            for fault in ["provenance", "resources"] {
                let (report, process) =
                    schedule(ExperimentMode::AaNoiseCalibration, Some((3, fault)));
                assert_eq!(report.pairs.len(), 2);
                assert_eq!(process.events.len(), 8);
                assert!(report.pairs[0].pair_valid);
                assert!(!report.pairs[1].pair_valid);
                assert!(report.pairs[1]
                    .treatment
                    .errors
                    .iter()
                    .any(|e| e.contains("cross-process")));
                assert!(!report.performance_comparison_authorized);
            }
        }

        #[test]
        fn wrong_adapter_is_retained_without_retry() {
            let (report, process) =
                schedule(ExperimentMode::AaNoiseCalibration, Some((1, "adapter")));
            assert_eq!(report.pairs.len(), 1);
            assert_eq!(process.events.len(), 4);
            let failed = &report.pairs[0].control;
            assert_eq!(failed.adapter.as_ref().unwrap().name, "NVIDIA L4/PCIe/SSE2");
            assert!(!failed.errors.is_empty());
            assert!(!report.qualification_pass);
        }

        #[test]
        fn calibration_identity_matches_and_rejects_each_drift() {
            let original = transported_arm(ExperimentMode::AbMovement, Arm::Treatment);
            let calibration = Calibration {
                bytes_sha256: "calibration".into(),
                floor: 2.0,
                input_provenance: transport_value(original.provenance.as_ref().unwrap()).unwrap(),
                resource_identity: transport_value(original.resource_identity.as_ref().unwrap())
                    .unwrap(),
            };
            for fault in [None, Some("provenance"), Some("resources")] {
                let mut arm: ArmReport =
                    serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
                match fault {
                    Some("provenance") => {
                        arm.provenance.as_mut().unwrap().config_sha256 = "drift".into()
                    }
                    Some("resources") => {
                        arm.resource_identity.as_mut().unwrap()["config"] = json!("drift")
                    }
                    _ => {}
                }
                reconcile_child_identity(&mut arm, &mut None, &mut None, Some(&calibration));
                assert_eq!(arm.errors.is_empty(), fault.is_none());
            }
        }

        fn base_function(name: &str) -> String {
            let output=std::process::Command::new("git").args(["show","d858dc338dcf7a90acd7cedc2cb41235721bb67a:rust-engine/src/gpu_native_predictor_v2_sidecar_performance.rs"]).current_dir(env!("CARGO_MANIFEST_DIR")).output().unwrap();
            assert!(output.status.success());
            let source = String::from_utf8(output.stdout).unwrap();
            let start = source.find(&format!("fn {name}(")).unwrap();
            let open = start + source[start..].find('{').unwrap();
            let mut depth = 1;
            for (i, c) in source[open + 1..].char_indices() {
                if c == '{' {
                    depth += 1;
                }
                if c == '}' {
                    depth -= 1;
                }
                if depth == 0 {
                    return source[start..open + i + 2].into();
                }
            }
            panic!("unclosed {name}")
        }

        #[test]
        fn frozen_timing_runtime_normalization_and_statistics_bytes() {
            let source = production();
            for name in [
                "run_arm",
                "execute_arm",
                "step_request",
                "shutdown",
                "shutdown_complete",
                "structural_initial",
                "structural_runtime_contract",
                "noise_statistics",
                "noise_stable",
                "noise_floor",
                "ab_verdict",
                "throughput",
                "paired_delta",
                "finish_pair",
                "finish_experiment",
                "validate_calibration",
                "calibration_floor_from_bytes",
                "serialized_identity_matches",
            ] {
                assert!(source.contains(&base_function(name)), "{name} changed");
            }
            assert_eq!((PAIRS, OUTPUT_TOKENS, PLANNED_POSITIONS), (6, 128, 143));
        }

        #[test]
        fn parent_call_graph_contains_no_runtime_and_child_has_one_arm() {
            let source = production();
            let parent = part(source, "pub(crate) async fn run_command(", "\0");
            assert!(parent.find("return run_child(").unwrap() < parent.find("run_pairs(").unwrap());
            let schedule = part(source, "fn run_pairs(", "async fn run_child(");
            let acceptance = part(source, "fn accept_child(", "fn run_pairs(");
            let launch = part(
                source,
                "impl ArmProcess for ProcessLauncher",
                "fn reconcile_child_identity(",
            );
            for block in [parent, schedule, acceptance, launch] {
                for forbidden in [
                    "run_arm(",
                    "prepare_arm(",
                    "build_isolated_greedy_runtime(",
                    "build_p1q_isolated_runtime(",
                    "load_real_cli_tokenizer(",
                    "execute_arm(",
                    "step_request(",
                    "step_token(",
                    "GpuBackend",
                    "wgpu::",
                    "Instant::now()",
                ] {
                    assert!(!block.contains(forbidden), "{forbidden}");
                }
                assert!(!block.contains("request_wall_ns ="));
            }
            let child = part(
                source,
                "async fn run_child(",
                "pub(crate) async fn run_command(",
            );
            assert_eq!(child.matches("run_arm(").count(), 1);
            assert!(child.find("run_arm(").unwrap() < child.find("write_report(").unwrap());
            assert!(!child.contains("for "));
            assert!(!child.contains("loop {"));
            assert_eq!(launch.matches(".spawn()?").count(), 1);
            assert_eq!(launch.matches(".wait()?").count(), 1);
            assert!(launch.find(".spawn()?").unwrap() < launch.find(".wait()?").unwrap());
            assert!(launch.find(".wait()?").unwrap() < launch.find("std::fs::read(").unwrap());
            assert!(
                launch.find("sha256_hex(&bytes)").unwrap()
                    < launch.find("validate_child_artifact(&bytes").unwrap()
            );
            assert!(!launch.contains("remove_file"));
            for bad in [
                "/proc/self/fd",
                "RLIMIT_NOFILE",
                "fd-lifetime-diagnostic",
                "WGPU_BACKEND",
                "setrlimit",
                "set_var(",
            ] {
                assert!(!source.contains(bad));
            }
        }

        #[test]
        fn child_artifact_no_clobber_and_preserved() {
            let dir =
                std::env::temp_dir().join(format!("mer-p1q3-transport-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let request = ChildRequest {
                pair_index: 1,
                arm: Arm::Control,
            };
            let output = child_report_path(&dir.join("report.json"), request);
            let artifact = envelope(request, ExperimentMode::AaNoiseCalibration);
            write_report(&output, &artifact).unwrap();
            let first = std::fs::read(&output).unwrap();
            assert!(write_report(&output, &artifact).is_err());
            let args = args(ExperimentMode::AaNoiseCalibration);
            assert!(validate_child_artifact(
                &first,
                request,
                &process_evidence(request),
                &launcher(&args)
            )
            .is_ok());
            assert_eq!(std::fs::read(&output).unwrap(), first);
            std::fs::remove_dir_all(&dir).unwrap();
        }

        #[test]
        fn exact_one_file_scope_against_p1q1() {
            let output = std::process::Command::new("git")
                .args([
                    "diff",
                    "--name-only",
                    "d858dc338dcf7a90acd7cedc2cb41235721bb67a",
                    "a6b9cbcd6e6b576f90a1d875c9351c7276c950e2",
                    "--",
                ])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .unwrap();
            assert!(output.status.success());
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                "rust-engine/src/gpu_native_predictor_v2_sidecar_performance.rs\n"
            );
        }
    }

    mod p1q1 {
        use super::*;

        pub(super) fn runtime_contract(context_id: &str) -> evidence::RuntimeContractEvidence {
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
        let s = crate::gpu_native_predictor_v2_critical_path_attribution::historical_source(
            "gpu_native_token_loop.rs",
        );
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
        let new = crate::gpu_native_predictor_v2_critical_path_attribution::historical_source(
            "gpu_native_token_loop.rs",
        );
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
        let run = part(s, "fn run_pairs(", "async fn run_child(");
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
                crate::gpu_native_predictor_v2_critical_path_attribution::historical_source(
                    "gpu_native_token_loop.rs",
                )
                .as_bytes(),
                "b6e1769508d8ed219d411c77318d28fa08752895ca03f947c55b7422106a8a0e",
                crate::gpu_native_predictor_v2_critical_path_attribution::historical_source(
                    "gpu_native_predictor_v2_observation.rs",
                ),
            ),
            (
                "gpu_native_q4_route_parallel.rs",
                crate::gpu_native_predictor_v2_critical_path_attribution::historical_source(
                    "gpu_native_residency.rs",
                )
                .as_bytes(),
                "4d1554bad7a97e72c2df423e2696d5b76fe8ace8280e3d5642c8213962faa50e",
                crate::gpu_native_predictor_v2_critical_path_attribution::historical_source(
                    "gpu_native_q4_route_parallel.rs",
                ),
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
                "a6b9cbcd6e6b576f90a1d875c9351c7276c950e2",
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
