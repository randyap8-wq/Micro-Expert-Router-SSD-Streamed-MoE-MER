//! P1F qualification driver: one ordinary runtime, one request, no warmup.
//! All execution and residency remain in the production token loop. The final
//! unexecuted prediction is censored by the existing request-finalization hook.

use crate::gpu_native_real_benchmark as benchmark;
use crate::predictor_v2::{p1e, ObservationConfig, RequestPhase};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "mer.predictor-v2-p1e-runtime-observation.v1";
const MODE: &str = "qualification-only-runtime-observation";
// P1E's existing combined observations/no-emissions ceiling. Never increase it.
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

#[derive(Debug, Serialize)]
struct FrozenContract {
    target_layer: usize,
    source_layer: usize,
    position_delta: u64,
    candidate_count: usize,
    signal_revision: u64,
    request_local_learning: bool,
    movement_enabled: bool,
    quarantine_allocated: bool,
    speculative_source_io: bool,
    serving_activation: bool,
    quarantine_readiness: &'static str,
}
impl FrozenContract {
    fn frozen() -> Self {
        Self {
            target_layer: p1e::LAYER,
            source_layer: p1e::LAYER,
            position_delta: 1,
            candidate_count: 1,
            signal_revision: p1e::REVISION,
            request_local_learning: true,
            movement_enabled: false,
            quarantine_allocated: false,
            speculative_source_io: false,
            serving_activation: false,
            quarantine_readiness: "NOT_MEASURED",
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct Counts {
    emitted: usize,
    resolved: usize,
    censored: usize,
    pending: usize,
    censored_or_pending: usize,
    prediction_hits: usize,
    route_misses: usize,
    valid_f: usize,
    current_at_f: usize,
    absent_at_f: usize,
    unknown_at_f: usize,
    valid_d: usize,
    current_at_d: usize,
    absent_at_d: usize,
    unknown_at_d: usize,
    useful_prediction_opportunities: usize,
    target_confirmed_useful_opportunities: usize,
    already_resident_predictions: usize,
    redundant_route_hits: usize,
    logical_materialized_at_f: usize,
    ram_resident_at_f: usize,
    host_source_absent: usize,
    host_source_unknown: usize,
    host_source_absent_or_unknown: usize,
    incomplete_records: usize,
    // One count per affected record/reason, even when F and D share a reason.
    incomplete_records_by_reason: BTreeMap<String, usize>,
    incomplete_no_emissions: usize,
}

#[derive(Debug, Serialize)]
struct Rates {
    prediction_hit_over_resolved: Option<f64>,
    absent_at_f_over_valid_f: Option<f64>,
    useful_over_emitted: Option<f64>,
    useful_over_hits: Option<f64>,
    target_confirmed_useful_over_emitted: Option<f64>,
    logical_materialized_at_f_over_valid_f: Option<f64>,
    ram_resident_at_f_over_valid_f: Option<f64>,
}
fn rate(n: usize, d: usize) -> Option<f64> {
    (d != 0).then(|| n as f64 / d as f64)
}

#[derive(Debug, Serialize)]
struct HostLeadNs {
    count: usize,
    minimum: Option<u64>,
    maximum: Option<u64>,
    mean: Option<f64>,
    p50: Option<u64>,
    p95: Option<u64>,
    percentile_semantics: &'static str,
}
fn host_lead_ns(mut values: Vec<u64>) -> Result<HostLeadNs> {
    values.sort_unstable();
    let sum = values
        .iter()
        .try_fold(0u128, |s, &v| s.checked_add(u128::from(v)))
        .ok_or("host lead sum overflow")?;
    let percentile = |percent: usize| -> Result<Option<u64>> {
        if values.is_empty() {
            return Ok(None);
        }
        let rank = add(mul(values.len(), percent)?, 99)? / 100;
        Ok(Some(
            values[rank.checked_sub(1).ok_or("invalid percentile rank")?],
        ))
    };
    Ok(HostLeadNs {
        count: values.len(),
        minimum: values.first().copied(),
        maximum: values.last().copied(),
        mean: (!values.is_empty()).then(|| sum as f64 / values.len() as f64),
        p50: percentile(50)?,
        p95: percentile(95)?,
        percentile_semantics: "nearest rank: sorted[ceil(n * percentile / 100) - 1]; p50 is the lower median for even n",
    })
}

#[derive(Debug, Serialize)]
struct Aggregate {
    counts: Counts,
    rates: Rates,
    host_lead_ns: HostLeadNs,
    report_incomplete_reason: Option<p1e::Error>,
}

fn aggregate(raw: &p1e::Report) -> Result<Aggregate> {
    let mut c = Counts::default();
    let mut leads = Vec::new();
    let mut sequences = BTreeSet::new();
    let mut opportunities = Vec::new();
    for r in &raw.observations {
        let candidate = r.freeze.candidate;
        if candidate.request != raw.request || !sequences.insert(candidate.sequence) {
            return Err("raw report request/sequence identity mismatch".into());
        }
        increment(&mut c.emitted)?;
        match r.outcome {
            p1e::Outcome::Resolved { prediction_hit } => {
                increment(&mut c.resolved)?;
                increment(if prediction_hit {
                    &mut c.prediction_hits
                } else {
                    &mut c.route_misses
                })?;
            }
            p1e::Outcome::Censored => increment(&mut c.censored)?,
            p1e::Outcome::Pending => increment(&mut c.pending)?,
        }
        match r.freeze.current {
            Some(current) => {
                increment(&mut c.valid_f)?;
                increment(if current {
                    &mut c.current_at_f
                } else {
                    &mut c.absent_at_f
                })?;
                if r.freeze.source.logical_materialized {
                    increment(&mut c.logical_materialized_at_f)?;
                }
                if r.freeze.source.ram_resident {
                    increment(&mut c.ram_resident_at_f)?;
                }
                if !r.freeze.source.logical_materialized && !r.freeze.source.ram_resident {
                    increment(&mut c.host_source_absent)?;
                }
            }
            None => {
                increment(&mut c.unknown_at_f)?;
                increment(&mut c.host_source_unknown)?;
            }
        }
        match r.deadline.and_then(|d| d.current) {
            Some(current) => {
                increment(&mut c.valid_d)?;
                increment(if current {
                    &mut c.current_at_d
                } else {
                    &mut c.absent_at_d
                })?;
            }
            None => increment(&mut c.unknown_at_d)?,
        }
        let o = r.opportunities();
        opportunities.push((candidate.sequence, o));
        for (flag, count) in [
            (o.useful, &mut c.useful_prediction_opportunities),
            (
                o.target_confirmed_useful,
                &mut c.target_confirmed_useful_opportunities,
            ),
            (o.already_resident, &mut c.already_resident_predictions),
            (o.redundant_route_hit, &mut c.redundant_route_hits),
        ] {
            if flag == Some(true) {
                increment(count)?;
            }
        }
        let reasons: BTreeSet<_> = [
            r.incomplete,
            r.freeze.incomplete,
            r.deadline.and_then(|d| d.incomplete),
        ]
        .into_iter()
        .flatten()
        .map(|e| format!("{e:?}"))
        .collect();
        if !reasons.is_empty() {
            increment(&mut c.incomplete_records)?;
        }
        for reason in reasons {
            increment(c.incomplete_records_by_reason.entry(reason).or_default())?;
        }
        if matches!(r.outcome, p1e::Outcome::Resolved { .. })
            && r.incomplete.is_none()
            && r.freeze.incomplete.is_none()
            && r.freeze.current.is_some()
        {
            if let Some(d) = r.deadline.filter(|d| {
                d.incomplete.is_none()
                    && d.current.is_some()
                    && d.request == candidate.request
                    && d.position == candidate.target_position
            }) {
                if let Some(lead) = d
                    .timestamp_ns
                    .checked_sub(r.freeze.timestamp_ns)
                    .filter(|v| *v > 0)
                {
                    if d.host_lead_ns == Some(lead) {
                        leads.push(lead);
                    }
                }
            }
        }
    }
    for r in &raw.no_emissions {
        if r.reason == p1e::NoEmissionReason::Incomplete {
            increment(&mut c.incomplete_no_emissions)?;
        }
    }
    c.censored_or_pending = add(c.censored, c.pending)?;
    c.host_source_absent_or_unknown = add(c.host_source_absent, c.host_source_unknown)?;
    let partitions = p1e::Partitions {
        emitted: c.emitted,
        resolved: c.resolved,
        pending: c.pending,
        censored: c.censored,
        prediction_hits: c.prediction_hits,
        route_misses: c.route_misses,
        valid_f: c.valid_f,
        current_at_f: c.current_at_f,
        absent_at_f: c.absent_at_f,
        useful: c.useful_prediction_opportunities,
        target_confirmed_useful: c.target_confirmed_useful_opportunities,
    };
    if partitions != raw.partitions
        || opportunities != raw.opportunities
        || c.emitted != add(c.resolved, c.censored_or_pending)?
        || c.resolved != add(c.prediction_hits, c.route_misses)?
        || c.valid_f != add(c.current_at_f, c.absent_at_f)?
        || c.valid_d != add(c.current_at_d, c.absent_at_d)?
        || c.emitted != add(c.valid_f, c.unknown_at_f)?
        || c.emitted != add(c.valid_d, c.unknown_at_d)?
        || c.useful_prediction_opportunities > c.prediction_hits
        || c.useful_prediction_opportunities > c.absent_at_f
        || c.target_confirmed_useful_opportunities > c.useful_prediction_opportunities
        || raw.readiness_measured
    {
        return Err("P1E raw/aggregate reconciliation failed".into());
    }
    let rates = Rates {
        prediction_hit_over_resolved: rate(c.prediction_hits, c.resolved),
        absent_at_f_over_valid_f: rate(c.absent_at_f, c.valid_f),
        useful_over_emitted: rate(c.useful_prediction_opportunities, c.emitted),
        useful_over_hits: rate(c.useful_prediction_opportunities, c.prediction_hits),
        target_confirmed_useful_over_emitted: rate(
            c.target_confirmed_useful_opportunities,
            c.emitted,
        ),
        logical_materialized_at_f_over_valid_f: rate(c.logical_materialized_at_f, c.valid_f),
        ram_resident_at_f_over_valid_f: rate(c.ram_resident_at_f, c.valid_f),
    };
    Ok(Aggregate {
        counts: c,
        rates,
        host_lead_ns: host_lead_ns(leads)?,
        report_incomplete_reason: raw.incomplete,
    })
}

#[derive(Serialize)]
struct ObservationConfiguration {
    phase: RequestPhase,
    phase_run_index: u64,
    prompt_length: usize,
    capacity_per_collection: usize,
}
impl From<ObservationConfig> for ObservationConfiguration {
    fn from(c: ObservationConfig) -> Self {
        Self {
            phase: c.phase,
            phase_run_index: c.phase_run_index,
            prompt_length: c.prompt_length,
            capacity_per_collection: c.capacity_per_collection,
        }
    }
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    mode: &'static str,
    provenance: benchmark::BenchmarkProvenance,
    config_path: String,
    config_sha256: String,
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    model_load: Option<crate::greedy_parity::ModelLoadEvidence>,
    adapter: Option<crate::backend::GpuDeviceIdentity>,
    runtime_contract: Option<benchmark::RuntimeContractEvidence>,
    production_configuration: benchmark::ProductionConfiguration,
    request: benchmark::RequestEvidence,
    generated_token_ids: Vec<u32>,
    generated_token_ids_sha256: String,
    completed_position_count: usize,
    observation_configuration: ObservationConfiguration,
    frozen_contract: FrozenContract,
    raw_p1e_report: Option<p1e::Report>,
    aggregate: Option<Aggregate>,
    runtime_counters: Option<benchmark::RequestSnapshots>,
    // Structural fact: P1E implements no movement operation. Counter snapshots
    // describe ordinary work and do not establish a no-overhead performance claim.
    predictor_v2_movement_operations: u64,
    runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
    errors: Vec<String>,
    observation_complete: bool,
    has_nonzero_predictions: Option<bool>,
    has_nonzero_useful_opportunities: Option<bool>,
    has_nonzero_target_confirmed_useful_opportunities: Option<bool>,
}

fn ensure_output_absent(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(format!(
            "refusing to overwrite observation evidence {}",
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

async fn execute_request(
    runtime: &crate::BenchRealRuntime,
    prompt: &[u32],
    report: &mut Report,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing authoritative token loop")?;
    let planned = add(
        prompt.len(),
        report
            .request
            .requested_output_tokens
            .checked_sub(1)
            .ok_or("zero output tokens")?,
    )?;
    if planned > token_loop.max_seq_len() {
        return Err("request exceeds runtime context limit".into());
    }
    let mut request = token_loop.create_request_state()?;
    request
        .enable_predictor_v2_p1e_observation(
            token_loop,
            observation_config(prompt.len(), report.request.requested_output_tokens)?,
        )
        .map_err(|e| format!("P1E enablement failed: {e:?}"))?;
    let snapshots = benchmark::RequestSnapshotStart::capture(runtime)?;
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
                .step_token(&runtime.engine, &mut request, token, position, sample)
                .await?;
            report.completed_position_count = add(report.completed_position_count, 1)?;
            match (sample, sampled) {
                (true, Some(id)) => report.generated_token_ids.push(id),
                (false, None) => {}
                _ => return Err("ordinary step returned an unexpected sampling result".into()),
            }
        }
        Ok(())
    }
    .await;
    request.finish_predictor_v2_observation(execution.is_err());
    report.raw_p1e_report = request.predictor_v2_p1e_report();
    report.generated_token_ids_sha256 =
        crate::greedy_parity::token_ids_sha256(&report.generated_token_ids);
    match snapshots.finish(runtime) {
        Ok(counters) => report.runtime_counters = Some(counters),
        Err(e) => report.errors.push(e.to_string()),
    }
    // Preserve the ordinary execution error before checking accounting evidence.
    execution?;
    let counters = report
        .runtime_counters
        .as_ref()
        .ok_or("runtime counters unavailable")?;
    benchmark::validate_request_postconditions(
        prompt.len(),
        report.request.requested_output_tokens,
        report.generated_token_ids.len(),
        counters.token_loop_delta,
        counters.recovery_delta,
        counters.routed_execution_delta,
    )?;
    let c = counters.token_loop_delta;
    if c.queue_submissions != c.token_attempts
        || c.boundary_maps != c.token_attempts
        || c.boundary_readbacks != c.token_attempts
    {
        return Err("ordinary attempt/submission/boundary accounting mismatch".into());
    }
    Ok(())
}

fn finalize_observation(report: &mut Report, positions: usize) {
    report.aggregate = None;
    report.observation_complete = false;
    report.has_nonzero_predictions = None;
    report.has_nonzero_useful_opportunities = None;
    report.has_nonzero_target_confirmed_useful_opportunities = None;
    if let Some(raw) = &report.raw_p1e_report {
        match aggregate(raw) {
            Ok(a) => {
                report.has_nonzero_predictions = Some(a.counts.emitted > 0);
                report.has_nonzero_useful_opportunities =
                    Some(a.counts.useful_prediction_opportunities > 0);
                report.has_nonzero_target_confirmed_useful_opportunities =
                    Some(a.counts.target_confirmed_useful_opportunities > 0);
                report.observation_complete = report.errors.is_empty()
                    && raw.incomplete.is_none()
                    && a.counts.incomplete_records == 0
                    && a.counts.incomplete_no_emissions == 0
                    && a.counts.pending == 0
                    && report.completed_position_count == positions
                    && a.counts.emitted.checked_add(raw.no_emissions.len()) == Some(positions)
                    && report.generated_token_ids.len() == report.request.requested_output_tokens
                    && report.runtime_shutdown.is_some_and(|s| {
                        s.controlled_shutdown_requested && s.all_runtime_resources_released
                    });
                report.aggregate = Some(a);
            }
            Err(e) => report.errors.push(format!("aggregation failed: {e}")),
        }
    } else {
        report.errors.push("P1E report unavailable".into());
    }
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<()> {
    ensure_output_absent(&args.report_out)?;
    if args.expected_adapter_name.trim().is_empty() {
        return Err("exact expected adapter name is required".into());
    }
    let (prompt, output_tokens) = parse_request(&std::fs::read(&args.request_json)?)?;
    let build = crate::qualification::BuildProvenance::embedded();
    benchmark::validate_preflight_provenance(&build)?;
    let config_bytes = std::fs::read(&args.config)?;
    let config_sha256 = crate::greedy_parity::sha256_hex(&config_bytes);
    let cfg: crate::config::Config = toml::from_str(std::str::from_utf8(&config_bytes)?)?;
    cfg.validate()?;
    benchmark::validate_source_config(&cfg)?;
    let (artifacts, errors) = crate::qualification_artifacts(&args.config, &cfg);
    benchmark::validate_artifacts(&artifacts, &errors)?;
    if artifacts.config.as_ref().map(|v| v.sha256.as_str()) != Some(config_sha256.as_str()) {
        return Err("config changed between parsed snapshot and artifact hashing".into());
    }
    let metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|e| format!("expert metadata unavailable: {e}"))?;
    benchmark::validate_expert_metadata(&metadata)?;
    let mode = crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark;
    let spec = crate::resolve_real_cli_spec_from_config(cfg, mode)?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err("strict Qwen3-MoE Q4_0 model identity required".into());
    }
    let resolved_hash = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(&spec.cfg, mode)?;
    let prompt_ids = tokenizer.encode(&prompt)?;
    let observation = observation_config(prompt_ids.len(), output_tokens)?;
    let positions = add(
        prompt_ids.len(),
        output_tokens.checked_sub(1).ok_or("zero output tokens")?,
    )?;
    if positions > spec.cfg.real_transformer.gpu_native_max_seq_len {
        return Err("request exceeds configured context limit".into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let configuration = benchmark::ProductionConfiguration::from_config(&spec.cfg, &metadata);
    let mut report = Report {
        schema: SCHEMA,
        mode: MODE,
        provenance: benchmark::BenchmarkProvenance {
            build,
            executable_canonical_path: std::fs::canonicalize(executable)?.display().to_string(),
            executable_sha256,
            resolved_config_sha256: resolved_hash.clone(),
            artifacts,
            expert_metadata: metadata,
        },
        config_path: std::fs::canonicalize(&args.config)?.display().to_string(),
        config_sha256,
        model_identity,
        model_load: None,
        adapter: None,
        runtime_contract: None,
        production_configuration: configuration,
        request: benchmark::RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(prompt.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: output_tokens,
            greedy: true,
        },
        generated_token_ids: Vec::new(),
        generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&[]),
        completed_position_count: 0,
        observation_configuration: observation.into(),
        frozen_contract: FrozenContract::frozen(),
        raw_p1e_report: None,
        aggregate: None,
        runtime_counters: None,
        predictor_v2_movement_operations: 0,
        runtime_shutdown: None,
        errors: Vec::new(),
        observation_complete: false,
        has_nonzero_predictions: None,
        has_nonzero_useful_opportunities: None,
        has_nonzero_target_confirmed_useful_opportunities: None,
    };
    // Exactly one production bootstrap. All paths after successful construction
    // reach normal isolated shutdown, including validation and stepping failures.
    match crate::build_isolated_greedy_runtime(&spec, mode, tokenizer).await {
        Err(e) => report.errors.push(e.to_string()),
        Ok(runtime) => {
            let execution: Result<()> = async {
                let observed_hash = crate::resolved_real_runtime_identity_sha256(
                    &runtime.cfg,
                    runtime.model.config.architecture,
                    runtime.model.config.first_k_dense_replace,
                    &runtime.model.config.advanced,
                )?;
                if observed_hash != resolved_hash {
                    return Err("runtime configuration identity drift".into());
                }
                let model_load = crate::greedy_parity_model_load(&runtime);
                let input = benchmark::RuntimeContractInput {
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
                let (contract, _) =
                    benchmark::validate_runtime_contract(&input, &args.expected_adapter_name)?;
                report.runtime_contract = Some(contract);
                let t = runtime
                    .gpu_native_token_loop
                    .as_ref()
                    .ok_or("missing token loop")?;
                if t.snapshot() != Default::default()
                    || t.recovery_snapshot() != Default::default()
                    || runtime.engine.routed_expert_execution_snapshot() != Default::default()
                {
                    return Err("isolated runtime has nonzero initial execution counters".into());
                }
                execute_request(&runtime, &prompt_ids, &mut report).await
            }
            .await;
            if let Err(e) = execution {
                report.errors.push(e.to_string());
            }
            match runtime.shutdown_isolated().await {
                Ok(evidence) => {
                    if !evidence.controlled_shutdown_requested
                        || !evidence.all_runtime_resources_released
                    {
                        report
                            .errors
                            .push("isolated runtime shutdown incomplete".into());
                    }
                    report.runtime_shutdown = Some(evidence);
                }
                Err(e) => report.errors.push(e.to_string()),
            }
        }
    }
    finalize_observation(&mut report, positions);
    write_report(&args.report_out, &report)?;
    if !report.errors.is_empty() {
        return Err(report.errors.join("; ").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor_v2::{ModelMetadata, PositionIdentity, RequestIdentity, RequestObserver};
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
    fn mixed() -> p1e::Report {
        let mut r = vec![
            record(0, true, Some(false), Some(false)),
            record(1, true, Some(true), Some(true)),
            record(2, false, Some(false), Some(true)),
            record(3, true, Some(false), None),
            record(4, true, None, None),
            record(5, true, Some(false), Some(true)),
        ];
        r[0].freeze.source.logical_materialized = true;
        r[1].freeze.source.ram_resident = true;
        r[3].outcome = p1e::Outcome::Censored;
        r[3].deadline = None;
        r[4].outcome = p1e::Outcome::Pending;
        r[4].deadline = None;
        r[4].freeze.incomplete = Some(p1e::Error::PhysicalEvidence);
        r[5].freeze.source.logical_materialized = true;
        r[5].freeze.source.ram_resident = true;
        raw(r)
    }
    fn report_fixture(raw: p1e::Report) -> Report {
        Report {
            schema: SCHEMA,
            mode: MODE,
            provenance: benchmark::BenchmarkProvenance {
                build: crate::qualification::BuildProvenance {
                    git_sha: Some("1".repeat(40)),
                    dirty: Some(false),
                    package_version: "test".into(),
                },
                executable_canonical_path: "/test/binary".into(),
                executable_sha256: "2".repeat(64),
                resolved_config_sha256: "3".repeat(64),
                artifacts: Default::default(),
                expert_metadata: crate::qualification::ExpertMetadataEvidence {
                    dtype: Some("q4_0".into()),
                    q4_0_layout: None,
                    conversion_mode: None,
                    source: None,
                    explicitly_synthetic: false,
                },
            },
            config_path: "/test/config".into(),
            config_sha256: "4".repeat(64),
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
            model_load: None,
            adapter: None,
            runtime_contract: None,
            production_configuration: Default::default(),
            request: benchmark::RequestEvidence {
                prompt_sha256: "5".repeat(64),
                prompt_token_ids_sha256: "6".repeat(64),
                prompt_token_count: 4,
                requested_output_tokens: 128,
                greedy: true,
            },
            generated_token_ids: vec![7, 9],
            generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&[7, 9]),
            completed_position_count: 5,
            observation_configuration: observation_config(4, 128).unwrap().into(),
            frozen_contract: FrozenContract::frozen(),
            aggregate: Some(aggregate(&raw).unwrap()),
            raw_p1e_report: Some(raw),
            runtime_counters: None,
            predictor_v2_movement_operations: 0,
            runtime_shutdown: None,
            errors: Vec::new(),
            observation_complete: false,
            has_nonzero_predictions: Some(true),
            has_nonzero_useful_opportunities: Some(true),
            has_nonzero_target_confirmed_useful_opportunities: Some(true),
        }
    }

    #[test]
    fn schema_serializes_typed_report_and_complete_raw_evidence() {
        let raw = mixed();
        let expected = serde_json::to_value(&raw).unwrap();
        let value = serde_json::to_value(report_fixture(raw.clone())).unwrap();
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["mode"], MODE);
        assert_eq!(value["raw_p1e_report"], expected);
        let roundtrip: p1e::Report =
            serde_json::from_value(value["raw_p1e_report"].clone()).unwrap();
        assert_eq!(roundtrip, raw);
        let candidate = &value["raw_p1e_report"]["observations"][0]["freeze"]["candidate"];
        assert_eq!(candidate["score"], 17);
        assert_eq!(candidate["source_set"], json!([0, 1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(candidate["committed_position_cutoff"], 3);
        assert_eq!(value["predictor_v2_movement_operations"], 0);
    }
    #[test]
    fn emitted_resolved_hit_miss_reconcile_and_censoring_is_not_a_miss() {
        let a = aggregate(&mixed()).unwrap();
        assert_eq!(
            (
                a.counts.emitted,
                a.counts.resolved,
                a.counts.prediction_hits,
                a.counts.route_misses
            ),
            (6, 4, 3, 1)
        );
        assert_eq!(
            (
                a.counts.censored,
                a.counts.pending,
                a.counts.censored_or_pending
            ),
            (1, 1, 2)
        );
        assert_eq!(a.rates.prediction_hit_over_resolved, Some(0.75));
    }
    #[test]
    fn f_and_d_partitions_preserve_unknown() {
        let c = aggregate(&mixed()).unwrap().counts;
        assert_eq!(
            (c.valid_f, c.current_at_f, c.absent_at_f, c.unknown_at_f),
            (5, 1, 4, 1)
        );
        assert_eq!(
            (c.valid_d, c.current_at_d, c.absent_at_d, c.unknown_at_d),
            (4, 3, 1, 2)
        );
        assert_eq!(c.already_resident_predictions, 1);
        assert_eq!(c.redundant_route_hits, 1);
    }
    #[test]
    fn opportunity_bounds_and_denominators() {
        let a = aggregate(&mixed()).unwrap();
        let c = &a.counts;
        assert_eq!(c.useful_prediction_opportunities, 2);
        assert_eq!(c.target_confirmed_useful_opportunities, 1);
        assert!(c.useful_prediction_opportunities <= c.prediction_hits);
        assert!(c.useful_prediction_opportunities <= c.absent_at_f);
        assert!(c.target_confirmed_useful_opportunities <= c.useful_prediction_opportunities);
        assert_eq!(a.rates.useful_over_emitted, Some(2.0 / 6.0));
        assert_eq!(a.rates.useful_over_hits, Some(2.0 / 3.0));
        assert_eq!(
            a.rates.target_confirmed_useful_over_emitted,
            Some(1.0 / 6.0)
        );
        assert_eq!(a.rates.absent_at_f_over_valid_f, Some(4.0 / 5.0));
    }
    #[test]
    fn zero_denominators_and_empty_lead_are_unavailable() {
        let a = aggregate(&raw(Vec::new())).unwrap();
        let rates = serde_json::to_value(a.rates).unwrap();
        assert!(rates
            .as_object()
            .unwrap()
            .values()
            .all(serde_json::Value::is_null));
        let lead = serde_json::to_value(a.host_lead_ns).unwrap();
        for key in ["minimum", "maximum", "mean", "p50", "p95"] {
            assert!(lead[key].is_null());
        }
        assert_eq!(lead["count"], 0);
    }
    #[test]
    fn host_lead_min_max_mean_nearest_rank_p50_p95() {
        let a = aggregate(&mixed()).unwrap().host_lead_ns;
        assert_eq!(a.count, 4);
        assert_eq!(
            (a.minimum, a.maximum, a.p50, a.p95),
            (Some(10), Some(60), Some(20), Some(60))
        );
        assert_eq!(a.mean, Some(30.0));
        let lead = host_lead_ns((1..=20).rev().collect()).unwrap();
        assert_eq!((lead.p50, lead.p95), (Some(10), Some(19)));
        assert_eq!(lead.mean, Some(10.5));
        let lead = host_lead_ns(vec![u64::MAX, u64::MAX]).unwrap();
        assert_eq!(lead.maximum, Some(u64::MAX));
        assert!(lead.mean.unwrap().is_finite());
    }
    #[test]
    fn lead_excludes_invalid_chronology_identity_and_censoring() {
        for scenario in 0..10 {
            let mut r = record(0, true, Some(false), Some(false));
            match scenario {
                0 => r.deadline.as_mut().unwrap().timestamp_ns = 100,
                1 => r.deadline.as_mut().unwrap().timestamp_ns = 99,
                2 => r.deadline.as_mut().unwrap().host_lead_ns = Some(999),
                3 => r.deadline.as_mut().unwrap().request.request_sequence = 9,
                4 => r.deadline.as_mut().unwrap().incomplete = Some(p1e::Error::Chronology),
                5 => r.freeze.incomplete = Some(p1e::Error::ShadowDisagreement),
                6 => r.outcome = p1e::Outcome::Censored,
                7 => r.incomplete = Some(p1e::Error::MissingDeadline),
                8 => r.freeze.current = None,
                _ => r.deadline.as_mut().unwrap().current = None,
            }
            assert_eq!(
                aggregate(&raw(vec![r])).unwrap().host_lead_ns.count,
                0,
                "scenario {scenario}"
            );
        }
    }
    #[test]
    fn source_classes_include_overlapping_logical_and_ram_evidence() {
        let a = aggregate(&mixed()).unwrap();
        assert_eq!(
            (
                a.counts.logical_materialized_at_f,
                a.counts.ram_resident_at_f
            ),
            (2, 2)
        );
        assert_eq!(
            (
                a.counts.host_source_absent,
                a.counts.host_source_unknown,
                a.counts.host_source_absent_or_unknown
            ),
            (2, 1, 3)
        );
        assert_eq!(a.rates.logical_materialized_at_f_over_valid_f, Some(0.4));
        assert_eq!(a.rates.ram_resident_at_f_over_valid_f, Some(0.4));
    }
    #[test]
    fn incomplete_reasons_count_records_once_and_retain_global_reason() {
        let mut r = record(0, true, None, None);
        r.incomplete = Some(p1e::Error::Chronology);
        r.freeze.incomplete = Some(p1e::Error::Chronology);
        r.deadline.as_mut().unwrap().incomplete = Some(p1e::Error::PhysicalEvidence);
        let mut raw = raw(vec![r]);
        raw.incomplete = Some(p1e::Error::Capacity);
        raw.no_emissions.push(p1e::NoEmission {
            source: r_position(0),
            target: r_position(1),
            reason: p1e::NoEmissionReason::Incomplete,
        });
        let a = aggregate(&raw).unwrap();
        assert_eq!(a.counts.incomplete_records, 1);
        assert_eq!(
            a.counts.incomplete_records_by_reason,
            BTreeMap::from([("Chronology".into(), 1), ("PhysicalEvidence".into(), 1)])
        );
        assert_eq!(a.counts.incomplete_no_emissions, 1);
        assert_eq!(a.report_incomplete_reason, Some(p1e::Error::Capacity));
    }
    fn r_position(p: usize) -> PositionIdentity {
        PositionIdentity::from_prompt_length(p, 4).unwrap()
    }
    #[test]
    fn aggregate_failure_never_produces_partial_statistics() {
        let mut raw = mixed();
        raw.partitions.resolved = usize::MAX;
        assert!(aggregate(&raw).is_err());
        let mut raw = mixed();
        raw.opportunities[0].1.useful = Some(false);
        assert!(aggregate(&raw).is_err());
        let mut raw = mixed();
        raw.observations[1].freeze.candidate.sequence = 0;
        assert!(aggregate(&raw).is_err());
        assert!(add(usize::MAX, 1).is_err());
        assert!(mul(usize::MAX, 2).is_err());
    }
    #[test]
    fn frozen_contract_has_no_movement_quarantine_source_io_or_serving() {
        let c = serde_json::to_value(FrozenContract::frozen()).unwrap();
        for key in [
            "movement_enabled",
            "quarantine_allocated",
            "speculative_source_io",
            "serving_activation",
        ] {
            assert_eq!(c[key], false);
        }
        assert_eq!(c["target_layer"], 47);
        assert_eq!(c["source_layer"], 47);
        assert_eq!(c["position_delta"], 1);
        assert_eq!(c["candidate_count"], 1);
        assert_eq!(c["signal_revision"], 1);
        assert_eq!(c["request_local_learning"], true);
        assert_eq!(c["quarantine_readiness"], "NOT_MEASURED");
    }
    #[test]
    fn exact_prompt_length_phase_and_run_index() {
        for prompt in [1, 21, 128, 3969] {
            let c = observation_config(prompt, 128).unwrap();
            assert_eq!(c.prompt_length, prompt);
            assert_eq!(c.phase, RequestPhase::Measured);
            assert_eq!(c.phase_run_index, 0);
            assert!(c.capacity_per_collection >= (prompt + 127) * 48);
        }
        assert_eq!(
            observation_config(21, 128).unwrap().capacity_per_collection,
            (148 * 48 - 1) * 64
        );
        assert!(observation_config(0, 128).is_err());
        assert!(observation_config(1, 2).is_err());
        assert!(observation_config(3970, 128).is_err());
        assert!(observation_config(usize::MAX, 128).is_err());
    }
    #[test]
    fn same_request_prompt_and_128_outputs_fit_and_fresh_request_has_no_history() {
        let config = observation_config(21, 128).unwrap();
        let mut observer =
            RequestObserver::new(identity(), model(), config.capacity_per_collection).unwrap();
        observer.enable_temporal(namespace()).unwrap();
        for p in 0..148 {
            let ids: [u32; 8] = std::array::from_fn(|i| ((p % 16) * 8 + i) as u32);
            let t = observer.temporal_mut().unwrap();
            if let Some(candidate) = t.pending_candidate(p) {
                assert!(t.begin_attempt(p, true));
                t.deadline(
                    candidate.request,
                    p,
                    p as u64 * 100,
                    Ok(physical(candidate.source_set)),
                );
            }
            let pos = PositionIdentity::from_prompt_length(p, 21).unwrap();
            let next = PositionIdentity::from_prompt_length(p + 1, 21).unwrap();
            observer.observe_completed_position(identity(), pos, model(), &vec![ids.to_vec(); 48]);
            if let Some(candidate) = observer.prepare_temporal(pos, next) {
                observer.temporal_mut().unwrap().freeze(
                    candidate,
                    p as u64 * 100 + 1,
                    Ok(physical(ids)),
                    p1e::HostSource {
                        logical_generation: None,
                        logical_materialized: false,
                        ram_resident: false,
                        permanence: p1e::Permanence::Unknown,
                    },
                );
            }
        }
        assert_eq!(observer.completed_positions(), 148);
        observer.finish(false);
        let raw = observer.temporal().unwrap().report();
        assert_eq!(raw.incomplete, None);
        assert_eq!(raw.observations.len() + raw.no_emissions.len(), 148);
        assert!(!raw.observations.is_empty());
        assert_eq!(raw.partitions.pending, 0);
        aggregate(&raw).unwrap();
        let mut fresh = RequestObserver::new(
            RequestIdentity {
                request_sequence: 2,
                ..identity()
            },
            model(),
            config.capacity_per_collection,
        )
        .unwrap();
        fresh.enable_temporal(namespace()).unwrap();
        assert!(fresh.temporal().unwrap().report().observations.is_empty());
        assert_eq!(fresh.completed_positions(), 0);
    }
    #[test]
    fn request_conversion_preserves_exact_prompt_and_count() {
        assert_eq!(
            parse_request(br#"{"prompt":"  hello\n", "max_tokens":128,"temperature":0}"#).unwrap(),
            ("  hello\n".into(), 128)
        );
        assert_eq!(parse_request(br#"{"messages":[{"role":"system","content":"Be exact."},{"role":"user","content":"Add."}],"max_tokens":128,"top_k":1}"#).unwrap(), ("system: Be exact.\nuser: Add.\n".into(),128));
    }
    #[test]
    fn request_rejects_non_greedy_or_ignored_semantics() {
        for patch in [
            json!({"temperature":0.1}),
            json!({"top_p":0.5}),
            json!({"top_k":8}),
            json!({"n":2}),
            json!({"stream":true}),
            json!({"stop":"x"}),
            json!({"max_tokens":0}),
            json!({"presence_penalty":1}),
            json!({"logit_bias":{"1":1}}),
        ] {
            let mut v = json!({"prompt":"hello","max_tokens":128,"temperature":0});
            v.as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            assert!(
                parse_request(&serde_json::to_vec(&v).unwrap()).is_err(),
                "{v}"
            );
        }
        for input in [
            r#"{"prompt":"hello","max_tokens":128}"#,
            r#"{"prompt":"hello","temperature":0}"#,
            r#"{"prompt":" ","max_tokens":128,"temperature":0}"#,
            r#"{"prompt":"hello","messages":[],"max_tokens":128,"temperature":0}"#,
            r#"{"messages":[{"role":"user","content":[]}],"max_tokens":128,"temperature":0}"#,
        ] {
            assert!(parse_request(input.as_bytes()).is_err(), "{input}");
        }
    }

    fn runtime_input() -> benchmark::RuntimeContractInput {
        use crate::qualification::ExecutionPlanEvidence;
        benchmark::RuntimeContractInput {
            real_transformer_enabled: true,
            real_transformer_gpu_native: true,
            compute_offload: crate::backend::ComputeOffload::Gpu,
            legacy_execution_plan: ExecutionPlanEvidence {
                context_id: "test".into(),
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
            token_loop_geometry: Some(crate::gpu_native_token_loop::GpuNativeModelGeometry {
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
            }),
            authoritative_device: Some(crate::backend::GpuDeviceIdentity {
                name: "NVIDIA L4".into(),
                vendor_id: 0x10de,
                device_id: 0x27b8,
                device_type: "DiscreteGpu".into(),
                wgpu_backend: "vulkan".into(),
                driver: "test".into(),
                driver_info: "test".into(),
                compute_plane: "wgpu-vulkan".into(),
                software_adapter: false,
            }),
            model_load: crate::greedy_parity::ModelLoadEvidence {
                strict: true,
                loader: "safetensors".into(),
                loaded_tensors: 10,
                required_tensors: 10,
                optional_probed: 0,
                optional_loaded: 0,
                seeded_fallback_remained: false,
            },
            routed_failure_policy: crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed,
        }
    }
    #[test]
    fn runtime_contract_rejects_wrong_adapter() {
        assert!(benchmark::validate_runtime_contract(&runtime_input(), "NVIDIA L4").is_ok());
        assert_eq!(
            benchmark::validate_runtime_contract(&runtime_input(), "different")
                .unwrap_err()
                .code,
            "wrong-adapter"
        );
    }
    #[test]
    fn runtime_contract_rejects_each_wrong_geometry_axis() {
        for axis in 0..5 {
            let mut input = runtime_input();
            let g = input.token_loop_geometry.as_mut().unwrap();
            match axis {
                0 => g.num_layers = 47,
                1 => g.num_experts = 127,
                2 => g.top_k = 7,
                3 => g.d_model = 2047,
                _ => g.d_ff = 767,
            }
            assert_eq!(
                benchmark::validate_runtime_contract(&input, "NVIDIA L4")
                    .unwrap_err()
                    .code,
                "wrong-model-geometry"
            );
        }
    }
    #[test]
    fn runtime_contract_rejects_fallback_and_non_strict_loading() {
        for case in 0..5 {
            let mut input = runtime_input();
            match case {
                0 => input.model_load.seeded_fallback_remained = true,
                1 => input.model_load.strict = false,
                2 => input.model_load.loaded_tensors = 9,
                3 => input.legacy_execution_plan.fallback_occurred = true,
                _ => {
                    input.routed_failure_policy =
                        crate::engine::RoutedExpertGpuFailurePolicy::ServingCpuFallback
                }
            }
            assert!(benchmark::validate_runtime_contract(&input, "NVIDIA L4").is_err());
        }
    }
    fn temporary_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mer-p1f-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn output_refuses_overwrite_and_publishes_complete_json() {
        let dir = temporary_dir();
        let path = dir.join("report.json");
        let value = json!({"schema":SCHEMA});
        write_report(&path, &value).unwrap();
        let original = std::fs::read(&path).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&original).unwrap(),
            value
        );
        assert!(write_report(&path, &json!({"different":true})).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn output_rejects_dangling_symlink_and_failed_serialization_leaves_no_file() {
        let dir = temporary_dir();
        let path = dir.join("report.json");
        std::os::unix::fs::symlink(dir.join("absent"), &path).unwrap();
        assert!(write_report(&path, &json!({})).is_err());
        std::fs::remove_file(&path).unwrap();
        struct Bad;
        impl Serialize for Bad {
            fn serialize<S: serde::Serializer>(
                &self,
                _: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("test"))
            }
        }
        assert!(write_report(&path, &Bad).is_err());
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn cli_has_only_four_required_qualification_arguments() {
        use clap::Parser;
        let argv = [
            "mer",
            "observe-gpu-native-predictor-v2-p1e",
            "--config",
            "c",
            "--request-json",
            "r",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "o",
        ];
        let cli = crate::Cli::try_parse_from(argv).unwrap();
        let crate::Cmd::ObserveGpuNativePredictorV2P1e(args) = cli.cmd else {
            panic!("wrong command")
        };
        let json = serde_json::to_value(args).unwrap();
        assert_eq!(json.as_object().unwrap().len(), 4);
        for index in [2, 4, 6, 8] {
            let args: Vec<_> = argv
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != index && *i != index + 1)
                .map(|(_, v)| *v)
                .collect();
            assert!(crate::Cli::try_parse_from(args).is_err());
        }
        for flag in [
            "--target-layer",
            "--capacity-per-collection",
            "--prefetch",
            "--quarantine",
            "--signal-revision",
        ] {
            assert!(crate::Cli::try_parse_from(argv.into_iter().chain([flag, "1"])).is_err());
        }
    }
    #[test]
    fn source_proves_one_local_request_ordinary_step_and_no_movement() {
        let source = include_str!("gpu_native_predictor_v2_observation.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap();
        assert_eq!(source.matches(".create_request_state()").count(), 1);
        assert_eq!(
            source
                .matches(".enable_predictor_v2_p1e_observation(")
                .count(),
            1
        );
        assert_eq!(source.matches(".step_token(").count(), 1);
        assert_eq!(source.matches("build_isolated_greedy_runtime(").count(), 1);
        assert_eq!(source.matches("tokenizer.encode(").count(), 1);
        assert!(
            source
                .find(".enable_predictor_v2_p1e_observation(")
                .unwrap()
                < source.find(".step_token(").unwrap()
        );
        for forbidden in [
            "step_token_diagnostic",
            "step_token_oracle",
            "step_token_router",
            "step_token_semantic",
            "step_token_full",
            "ensure_speculative",
            "spawn_prefetch",
            "fetch_with_retry",
            "O_DIRECT",
            "reserve_physical",
            "physical_install(",
            "quarantine_buffer",
            "select_victim",
            "create_buffer",
            "queue.submit",
            "map_async",
            "device.poll",
            "std::env::var",
            "\nstatic ",
            "AtomicBool",
        ] {
            assert!(!source.contains(forbidden), "{forbidden}");
        }
    }
    #[test]
    fn frozen_signal_token_loop_and_config_bytes_unchanged() {
        for (source, digest) in [
            (
                include_str!("predictor_v2.rs"),
                "2d7202e76059e23593e98e8fc07b90d34b2dc26d255f9c48ad0a15b8022518a9",
            ),
            (
                include_str!("gpu_native_token_loop.rs"),
                "e88c54265b794015eb4f1ee910ccfc379efd824545d865a90c475ddd0e7a2376",
            ),
            (
                include_str!("config.rs"),
                "f57f8131c2f37976a5019cd16d9e8fdac83f75379bab324235e27f4a27395428",
            ),
        ] {
            assert_eq!(crate::greedy_parity::sha256_hex(source.as_bytes()), digest);
        }
        assert!(!include_str!("server.rs").contains("enable_predictor_v2_p1e_observation"));
        let main = include_str!("main.rs");
        assert!(!main.contains("enable_predictor_v2_p1e_observation"));
        assert_eq!(
            main.matches("gpu_native_predictor_v2_observation::run_command")
                .count(),
            1
        );
    }

    #[test]
    fn source_contract_rejects_all_fail_open_policies_without_loading_a_model() {
        let mut cfg: crate::config::Config =
            toml::from_str(include_str!("../../config.toml")).unwrap();
        cfg.real_transformer.enabled = true;
        cfg.real_transformer.gpu_native = true;
        cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
        cfg.real_transformer.weights_dir = Some(PathBuf::from("/not-loaded"));
        cfg.real_transformer.strict_weights = true;
        cfg.real_transformer.allow_seeded_fallback = false;
        cfg.real_transformer.allow_degraded_experts = false;
        cfg.real_transformer.allow_nonfinite_attention_fallback = false;
        cfg.real_transformer.allow_truncated_expert_payloads = false;
        cfg.distributed.enabled = false;
        cfg.gpu_cache.enabled = true;
        cfg.gpu_cache.vram_capacity_mb = 2048;
        cfg.model.dtype = crate::inference::WeightDtype::Q4_0;
        benchmark::validate_source_config(&cfg).unwrap();
        for policy in 0..6 {
            let mut bad = cfg.clone();
            match policy {
                0 => bad.real_transformer.allow_seeded_fallback = true,
                1 => bad.real_transformer.allow_degraded_experts = true,
                2 => bad.real_transformer.allow_nonfinite_attention_fallback = true,
                3 => bad.real_transformer.allow_truncated_expert_payloads = true,
                4 => bad.real_transformer.strict_weights = false,
                _ => bad.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu,
            }
            assert!(benchmark::validate_source_config(&bad).is_err());
        }
    }

    #[test]
    fn aggregation_error_artifact_retains_raw_and_has_no_partial_metrics_or_flags() {
        let mut report = report_fixture(mixed());
        report.raw_p1e_report.as_mut().unwrap().partitions.emitted = usize::MAX;
        let expected = serde_json::to_value(&report.raw_p1e_report).unwrap();
        finalize_observation(&mut report, 5);
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["raw_p1e_report"], expected);
        assert!(value["aggregate"].is_null());
        assert!(value["has_nonzero_predictions"].is_null());
        assert!(value["has_nonzero_useful_opportunities"].is_null());
        assert_eq!(value["observation_complete"], false);
        assert!(value["errors"][0]
            .as_str()
            .unwrap()
            .starts_with("aggregation failed:"));
    }

    #[test]
    fn normal_end_censoring_is_complete_but_incomplete_accounting_is_descriptive_only() {
        let mut raw = raw(vec![
            record(0, true, Some(false), Some(false)),
            record(1, true, Some(false), None),
        ]);
        raw.observations[1].outcome = p1e::Outcome::Censored;
        raw.observations[1].deadline = None;
        raw.partitions.resolved = 1;
        raw.partitions.prediction_hits = 1;
        raw.partitions.useful = 1;
        raw.partitions.censored = 1;
        raw.opportunities = raw
            .observations
            .iter()
            .map(|r| (r.freeze.candidate.sequence, r.opportunities()))
            .collect();
        raw.no_emissions = (0..3)
            .map(|p| p1e::NoEmission {
                source: r_position(p),
                target: r_position(p + 1),
                reason: p1e::NoEmissionReason::NoPositiveHistory,
            })
            .collect();
        let mut report = report_fixture(raw);
        report.request.requested_output_tokens = 2;
        report.runtime_shutdown = Some(crate::greedy_parity::BackgroundShutdownEvidence {
            controlled_shutdown_requested: true,
            all_runtime_resources_released: true,
            poll_iterations: 1,
        });
        finalize_observation(&mut report, 5);
        assert!(report.observation_complete);
        assert_eq!(report.has_nonzero_predictions, Some(true));
        assert_eq!(
            report.has_nonzero_target_confirmed_useful_opportunities,
            Some(true)
        );
        report.raw_p1e_report.as_mut().unwrap().incomplete = Some(p1e::Error::Capacity);
        finalize_observation(&mut report, 5);
        assert!(!report.observation_complete);
        assert!(report.errors.is_empty());
        assert!(report.aggregate.is_some());
    }
}
