//! HMA-1G: baseline-only lifecycle/order diagnosis. No low-level source logic.
//! The worker reuses HMA-1F-B's observer; the auditor is entirely offline.
#[path = "gpu_native_fresh_process_lifecycle.rs"]
pub(crate) mod fresh_process_lifecycle;

use super::*;

pub(crate) const SCHEMA: &str = "mer.gpu-native-baseline-runtime-lifecycle.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-baseline-runtime-lifecycle";
const BEGIN: &str = "HMA1G_QUALIFIER_BEGIN order=baseline-0,baseline-1,baseline-2,baseline-3";
const END: &str = "HMA1G_QUALIFIER_COMPLETE payload_sha256=";
const ORDER: [&str; 4] = ["baseline-0", "baseline-1", "baseline-2", "baseline-3"];
const PRIMARY: &str = "Measured K>1; right minus left, denominator left. Adjacent B0->B1, B1->B2, B2->B3 and endpoint B0->B3. Three measured requests; widths 2..8 sign evidence; width 1 descriptive only. Exact integer thresholds.";
const CLEANUP: &str = "HMA-1F-B mapped-baseline only; no registration/unregister/SQEs; exact pre=active=after VmPin and unchanged VmLck; ring drop before historical mapped-view teardown; isolated runtime shutdown/drop before next arm.";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureIdentity {
    stage: String,
    code: String,
    detail: String,
}
impl From<&BenchmarkFailure> for FailureIdentity {
    fn from(f: &BenchmarkFailure) -> Self {
        Self {
            stage: f.stage.clone(),
            code: f.code.clone(),
            detail: f.detail.clone(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case", deny_unknown_fields)]
enum Shutdown {
    NotAttempted,
    Succeeded,
    Failed { failure: FailureIdentity },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StartupEvidence {
    primary: FailureIdentity,
    benchmark: Value,
    construction_completed: bool,
    qualification_enable_completed: bool,
    runtime_validation_completed: bool,
    shutdown: Shutdown,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TranscriptLink {
    source: String,
    begin_marker: String,
    end_marker: String,
}
impl TranscriptLink {
    fn arm(position: usize) -> Self {
        Self {
            source: "bound-worker-transcript".into(),
            begin_marker: format!("HMA1G_ARM_BEGIN index={position} name={}", ORDER[position]),
            end_marker: format!("HMA1G_ARM_END index={position} name={}", ORDER[position]),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailedArm {
    position: Option<usize>,
    name: Option<String>,
    primary: FailureIdentity,
    startup: Option<StartupEvidence>,
    incomplete_arm: Option<Value>,
    shutdown: Option<Shutdown>,
    before: Option<ProcessMemory>,
    after: Option<ProcessMemory>,
    records: Vec<Record>,
    /// Logs are explicitly transcript evidence, never manufactured report.hardware.
    transcript: Option<TranscriptLink>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    schema: String,
    mode: String,
    primary: String,
    cleanup_contract: String,
    frozen_workload: Value,
    arms: Vec<ArmData>,
    complete: bool,
    failure: Option<FailedArm>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    payload_json_utf8: String,
    payload_sha256: String,
    transcript_sha256: String,
    transcript_bytes: usize,
    worker_exit_success: bool,
}
fn empty_payload() -> Result<Payload> {
    Ok(Payload {
        schema: SCHEMA.into(),
        mode: MODE.into(),
        primary: PRIMARY.into(),
        cleanup_contract: CLEANUP.into(),
        frozen_workload: serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
        arms: Vec::new(),
        complete: false,
        failure: None,
    })
}
fn production(run: PhysicalInstallArmRun) -> Result<Value> {
    Ok(serde_json::to_value(UploadArmReport {
        run: ConcurrencyArmReport {
            common: run.common,
            warmup_mechanism: run.warmup_concurrency,
            mechanism: run.concurrency,
        },
        warmup_upload: run.warmup_upload,
        upload: run.upload,
    })?)
}
enum Attempt {
    Complete(ArmData),
    Failed(FailedArm),
}
async fn execute_arm(prepared: &Prepared, args: &CommandArgs, position: usize) -> Result<Attempt> {
    let before = ProcessMemory::capture()?;
    let observer = Arc::new(Observer::new(Mode::MappedBaseline));
    let outcome = run_physical_install_arm_observed_outcome(
        prepared,
        args,
        PhysicalInstallQualificationRun::SourceToUpload(Arm::Treatment),
        Some(MODE),
        None,
        Some(observer.clone()),
    )
    .await;
    // The shared runner has already shut down and consumed/dropped its runtime.
    // Preserve the outcome even when the post-run process snapshot fails.
    let after = ProcessMemory::capture().ok();
    let records = observer.records();
    match outcome {
        ObservedArmOutcome::StartupFailed(evidence) => {
            let startup: StartupEvidence =
                serde_json::from_value(serde_json::to_value(&evidence)?)?;
            Ok(Attempt::Failed(FailedArm {
                position: Some(position),
                name: Some(ORDER[position].into()),
                primary: startup.primary.clone(),
                shutdown: Some(startup.shutdown.clone()),
                startup: Some(startup),
                incomplete_arm: None,
                before: Some(before),
                after,
                records,
                transcript: Some(TranscriptLink::arm(position)),
            }))
        }
        ObservedArmOutcome::Run {
            run,
            primary_failure,
            shutdown,
        } => {
            let failure = primary_failure
                .as_ref()
                .or(run.common.failure.as_ref())
                .map(FailureIdentity::from);
            let complete = run.common.complete;
            let production = production(run)?;
            if complete && after.is_some() {
                Ok(Attempt::Complete(ArmData {
                    position,
                    mode: Mode::MappedBaseline,
                    before,
                    after: after.unwrap(),
                    production,
                    records,
                }))
            } else {
                let primary = failure.unwrap_or_else(|| FailureIdentity {
                    stage: "evidence".into(),
                    code: "post-runtime-status-unavailable".into(),
                    detail: "post-runtime process snapshot unavailable".into(),
                });
                // Retain the actual run even if process evidence is unavailable.
                // This is never promoted into the completed-arm vector.
                let incomplete = json!({"position":position,"mode":Mode::MappedBaseline,
                    "before":before,"after":after,"production":production,"records":records});
                Ok(Attempt::Failed(FailedArm {
                    position: Some(position),
                    name: Some(ORDER[position].into()),
                    primary,
                    startup: None,
                    before: Some(before),
                    after,
                    records,
                    incomplete_arm: Some(incomplete),
                    shutdown: Some(serde_json::from_value(serde_json::to_value(shutdown)?)?),
                    transcript: Some(TranscriptLink::arm(position)),
                }))
            }
        }
    }
}
pub(crate) async fn run_worker(args: CommandArgs) -> Result<()> {
    crate::gpu_native_mapped_pin::require_platform()?;
    println!("{BEGIN}");
    let mut payload = empty_payload()?;
    let mut position = None;
    let result = async {
        require(
            args.expected_adapter_name == "NVIDIA L4",
            "HMA-1G requires NVIDIA L4",
        )?;
        let prepared = prepare(&args)?;
        for index in 0..ORDER.len() {
            position = Some(index);
            let link = TranscriptLink::arm(index);
            println!("{}", link.begin_marker);
            let attempt = execute_arm(&prepared, &args, index).await;
            println!("{}", link.end_marker);
            match attempt? {
                Attempt::Complete(arm) => {
                    payload.arms.push(arm);
                    validate_completed(&payload.arms)?;
                }
                Attempt::Failed(failure) => {
                    let error = format!("{}: {}", failure.primary.code, failure.primary.detail);
                    payload.failure = Some(failure);
                    return Err(error.into());
                }
            }
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    }
    .await;
    payload.complete = result.is_ok();
    if let Err(error) = &result {
        if payload.failure.is_none() {
            let primary = error
                .downcast_ref::<BenchmarkFailure>()
                .map(FailureIdentity::from)
                .unwrap_or_else(|| FailureIdentity {
                    stage: "evidence".into(),
                    code: "worker-failed".into(),
                    detail: error.to_string(),
                });
            payload.failure = Some(FailedArm {
                position,
                name: position.map(|i| ORDER[i].into()),
                primary,
                startup: None,
                incomplete_arm: None,
                shutdown: None,
                before: None,
                after: None,
                records: Vec::new(),
                transcript: position.map(TranscriptLink::arm),
            });
        }
    }
    let bytes = serde_json::to_vec(&payload)?;
    write_new(&args.report_out, &bytes)?;
    println!("{END}{}", sha(&bytes));
    std::io::stdout().flush()?;
    result
}
fn canonical_output(path: &Path) -> Result<PathBuf> {
    require(path.file_name().is_some(), "output needs a filename")?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => return Err("HMA-1G outputs must be fresh (including symlinks)".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(e) => return Err(e.into()),
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(std::fs::canonicalize(parent)?.join(path.file_name().unwrap()))
}
pub(crate) fn launch(
    config: &Path,
    report_out: &Path,
    transcript_out: &Path,
    raw_args: &[std::ffi::OsString],
) -> Result<()> {
    crate::gpu_native_mapped_pin::require_platform()?;
    let mut draft_name = report_out.as_os_str().to_owned();
    draft_name.push(".worker-payload.json");
    let draft = PathBuf::from(draft_name);
    let paths = [
        canonical_output(report_out)?,
        canonical_output(transcript_out)?,
        canonical_output(&draft)?,
    ];
    require(
        paths.iter().collect::<BTreeSet<_>>().len() == 3,
        "artifact paths must be distinct",
    )?;
    let args = worker_arguments(config, &draft, raw_args)?;
    let transcript = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(transcript_out)?;
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(args)
        .stdout(transcript.try_clone()?)
        .stderr(transcript)
        .status()?;
    let transcript_bytes = std::fs::read(transcript_out)?;
    let payload_bytes = std::fs::read(&draft)?;
    let envelope = Envelope {
        schema: SCHEMA.into(),
        payload_sha256: sha(&payload_bytes),
        payload_json_utf8: String::from_utf8(payload_bytes)?,
        transcript_sha256: sha(&transcript_bytes),
        transcript_bytes: transcript_bytes.len(),
        worker_exit_success: status.success(),
    };
    let raw = serde_json::to_vec_pretty(&envelope)?;
    write_new(report_out, &raw)?;
    let audit = audit_bytes(&raw, &transcript_bytes);
    eprintln!(
        "HMA1G_RESULT={} raw_report_sha256={} transcript_sha256={}",
        audit.classification,
        sha(&raw),
        envelope.transcript_sha256
    );
    require(
        audit.authoritative,
        "HMA-1G NON_AUTHORITATIVE; see bound report/transcript",
    )
}

fn worker_arguments(
    config: &Path,
    draft: &Path,
    raw_args: &[std::ffi::OsString],
) -> Result<Vec<std::ffi::OsString>> {
    // Preserve the user's process settings but force info-level transcript authority.
    let command_index = raw_args
        .iter()
        .position(|s| s == MODE)
        .ok_or("missing qualifier command")?;
    require(
        !raw_args
            .iter()
            .any(|a| a.to_string_lossy().starts_with("--autotune")),
        "HMA-1G forbids autotune/probe workloads",
    )?;
    let mut args = Vec::new();
    let mut supplied = raw_args.iter().enumerate().skip(1);
    while let Some((index, arg)) = supplied.next() {
        if index == command_index {
            continue;
        }
        // Remove public command paths and logging wherever Clap accepted them.
        // Preserve process settings both before and after the subcommand.
        let replaced = ["--log", "--config", "--report-out", "--transcript-out"];
        if replaced.iter().any(|flag| arg == *flag) {
            let _ = supplied.next();
            continue;
        }
        if replaced
            .iter()
            .any(|flag| arg.to_string_lossy().starts_with(&format!("{flag}=")))
        {
            continue;
        }
        args.push(arg.clone());
    }
    args.extend([
        "--log".into(),
        "info".into(),
        "hma1g-worker-internal".into(),
        "--config".into(),
        config.as_os_str().to_owned(),
        "--report-out".into(),
        draft.as_os_str().to_owned(),
    ]);
    Ok(args)
}

/// Require the actual BenchmarkReport wire shape, including explicit nulls.
fn validate_benchmark_header(b: &Value) -> Result<()> {
    use crate::gpu_native_real_benchmark as benchmark;
    let fields = [
        "schema",
        "mode",
        "optimization",
        "immediate_comparison_commit",
        "pr1b_a_experiment_commit",
        "pr1b_a_experiment_report_sha256",
        "original_baseline_commit",
        "benchmark_complete",
        "failure",
        "qualification_pass",
        "correctness_qualification_pending",
        "provenance",
        "hardware",
        "model_identity",
        "model_load",
        "request",
        "cache_reset",
        "warmup_runs",
        "warmup_runs_completed",
        "measured_runs",
        "runtime_constructions",
        "runtime_shutdowns",
        "per_run_results",
        "aggregate",
        "runtime_contract",
        "production_configuration",
        "production_semantics",
    ];
    require(
        b.as_object().is_some_and(|o| {
            o.keys().map(String::as_str).collect::<BTreeSet<_>>() == fields.into_iter().collect()
        }),
        "malformed/incomplete benchmark report",
    )?;
    for (field, expected) in [
        ("schema", benchmark::SCHEMA),
        ("mode", benchmark::MODE),
        ("optimization", benchmark::OPTIMIZATION),
        (
            "immediate_comparison_commit",
            benchmark::IMMEDIATE_COMPARISON_COMMIT,
        ),
        (
            "pr1b_a_experiment_commit",
            benchmark::PR1B_A_EXPERIMENT_COMMIT,
        ),
        (
            "pr1b_a_experiment_report_sha256",
            benchmark::PR1B_A_EXPERIMENT_REPORT_SHA256,
        ),
        (
            "original_baseline_commit",
            benchmark::ORIGINAL_BASELINE_COMMIT,
        ),
    ] {
        require(b[field] == expected, "benchmark identity drift")?;
    }
    require(
        b["qualification_pass"] == false && b["correctness_qualification_pending"] == true,
        "benchmark qualification semantics drift",
    )
}

fn validate_completed(arms: &[ArmData]) -> Result<()> {
    require(
        !arms.is_empty() && arms.len() <= 4,
        "one to four completed arms required",
    )?;
    let mut ids = BTreeSet::new();
    for (i, a) in arms.iter().enumerate() {
        require(
            a.position == i && a.mode == Mode::MappedBaseline,
            "baseline-only arm order mismatch",
        )?;
        validate_benchmark_header(&a.production["benchmark"])?;
        validate_arm(a)?;
        require(
            ids.insert(context_id(&a.production["benchmark"])?),
            "runtime context reused across arms",
        )?;
        if i == 0 {
            continue;
        }
        let c = &arms[0].production;
        let t = &a.production;
        contracts_equal(&c["benchmark"], &t["benchmark"])?;
        require(
            c["warmup_ram_cache_state_sha256"] == t["warmup_ram_cache_state_sha256"],
            "warmup cache/reset drift",
        )?;
        for (key, measured) in [("warmup_results", false), ("per_run_results", true)] {
            let cr = if measured {
                &c["benchmark"][key]
            } else {
                &c[key]
            };
            let tr = if measured {
                &t["benchmark"][key]
            } else {
                &t[key]
            };
            for (x, y) in cr.as_array().unwrap().iter().zip(tr.as_array().unwrap()) {
                for field in [
                    "run_index",
                    "generated_tokens",
                    "generated_token_ids_sha256",
                    "generated_text_sha256",
                ] {
                    require(x[field] == y[field], "token/text work drift")?;
                }
                if measured {
                    require(
                        x["generated_token_ids"] == y["generated_token_ids"],
                        "generated token stream drift",
                    )?;
                }
            }
        }
        for prefix in ["warmup_", ""] {
            let mk = if prefix.is_empty() {
                "mechanism"
            } else {
                "warmup_mechanism"
            };
            let cs: Snapshot = decode(&c[mk])?;
            let ts: Snapshot = decode(&t[mk])?;
            let cu: UploadSnapshot = decode(&c[format!("{prefix}upload")])?;
            let tu: UploadSnapshot = decode(&t[format!("{prefix}upload")])?;
            let cp: GpuNativeProductionPhysicalInstallSnapshot =
                decode(&c[format!("{prefix}production_physical_install")])?;
            let tp = decode(&t[format!("{prefix}production_physical_install")])?;
            let cds: ProductionDemandSourceSnapshot = decode(&c[format!("{prefix}production")])?;
            let tds = decode(&t[format!("{prefix}production")])?;
            let cw: ArmWorkEvidence = decode(&c[format!("{prefix}work")])?;
            let tw: ArmWorkEvidence = decode(&t[format!("{prefix}work")])?;
            let g = pair_mechanism_gate(&cs, &ts, &cp, &tp, &cu, &tu, &cds, &tds);
            require(
                g.source_and_route_streams_exact
                    && g.physical_install_reservation_victim_and_publication_exact
                    && g.source_scheduler_exact
                    && work_pair_exact(&cw, &tw)
                    && cw.gpu_expert_io.expert_weight_upload_bytes
                        == tw.gpu_expert_io.expert_weight_upload_bytes
                    && cw.gpu_native_residency.vram_hits == tw.gpu_native_residency.vram_hits
                    && cw.gpu_native_residency.vram_misses == tw.gpu_native_residency.vram_misses
                    && cs.primary_pool_capacity == ts.primary_pool_capacity
                    && cu.logical_admission_ids_sha256 == tu.logical_admission_ids_sha256
                    && cu.logical_generation_ids_sha256 == tu.logical_generation_ids_sha256
                    && cu.metrics.logical_admissions == tu.metrics.logical_admissions
                    && cu.metrics.logical_generation_observations
                        == tu.metrics.logical_generation_observations
                    && cu.metrics.logical_materialization_operations
                        == tu.metrics.logical_materialization_operations
                    && cu.metrics.logical_materialization_bytes
                        == tu.metrics.logical_materialization_bytes
                    && cu.metrics.shared_payload_reuse == tu.metrics.shared_payload_reuse
                    && cu.source_upload_fd_proof == tu.source_upload_fd_proof,
                "production route/source/cache/work drift",
            )?;
        }
        require(
            arms[0].records.len() == a.records.len(),
            "source stream length drift",
        )?;
        for (x, y) in arms[0].records.iter().zip(&a.records) {
            require(
                (x.measured, x.request_index, x.source_index, &x.ids, x.bytes)
                    == (y.measured, y.request_index, y.source_index, &y.ids, y.bytes),
                "source IDs/order/width/request drift",
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
struct Delta {
    left: u64,
    right: u64,
    signed_delta: i128,
    /// Display only. All authority uses integer cross products below.
    percent: Option<f64>,
}
impl Delta {
    fn new(left: u64, right: u64) -> Self {
        let signed_delta = i128::from(right) - i128::from(left);
        Self {
            left,
            right,
            signed_delta,
            percent: (left != 0).then(|| signed_delta as f64 * 100.0 / left as f64),
        }
    }
    fn reaches(&self, percent: i128) -> bool {
        self.left != 0 && self.signed_delta.abs() * 100 >= i128::from(self.left) * percent
    }
    fn stable(&self) -> bool {
        self.left != 0 && self.signed_delta.abs() * 100 < i128::from(self.left) * 3
    }
}
#[derive(Clone, Debug, Serialize)]
struct Comparison {
    read_critical_span_ns: Delta,
    batch_max_read_wall_ns: Delta,
}
impl Comparison {
    fn new(left: Endpoints, right: Endpoints) -> Self {
        Self {
            read_critical_span_ns: Delta::new(left.critical, right.critical),
            batch_max_read_wall_ns: Delta::new(left.max_wall, right.max_wall),
        }
    }
    fn direction(&self) -> i128 {
        let direction = self.read_critical_span_ns.signed_delta.signum();
        if direction == self.batch_max_read_wall_ns.signed_delta.signum() {
            direction
        } else {
            0
        }
    }
    fn stable(&self) -> bool {
        self.read_critical_span_ns.stable() && self.batch_max_read_wall_ns.stable()
    }
}
#[derive(Debug, Serialize)]
struct RequestComparison {
    request_index: usize,
    comparison: Comparison,
}
#[derive(Debug, Serialize)]
struct WidthComparison {
    width: usize,
    comparison: Comparison,
}
#[derive(Debug, Serialize)]
struct Pair {
    left_arm: usize,
    right_arm: usize,
    pooled_k_gt_one: Comparison,
    measured_requests: Vec<RequestComparison>,
    widths: Vec<WidthComparison>,
    width_one_descriptive: Comparison,
}
impl Pair {
    fn drift(&self, threshold: i128) -> bool {
        let pooled = &self.pooled_k_gt_one;
        let direction = pooled.direction();
        direction != 0
            && pooled.read_critical_span_ns.reaches(threshold)
            && pooled.batch_max_read_wall_ns.reaches(threshold)
            && self
                .measured_requests
                .iter()
                .filter(|r| r.comparison.direction() == direction)
                .count()
                >= 2
            && self
                .widths
                .iter()
                .filter(|w| w.comparison.read_critical_span_ns.signed_delta.signum() == direction)
                .count()
                >= 5
    }
}
#[derive(Debug, Serialize)]
struct Analysis {
    adjacent: Vec<Pair>,
    endpoint_b0_b3: Comparison,
}
fn compare(
    arms: &[ArmData],
    left: usize,
    right: usize,
    request: Option<usize>,
    width: Option<usize>,
) -> Result<Comparison> {
    Ok(Comparison::new(
        totals(&arms[left].records, request, width)?,
        totals(&arms[right].records, request, width)?,
    ))
}
fn analyze(arms: &[ArmData]) -> Result<Analysis> {
    require(
        arms.len() == 4,
        "four complete arms required for performance analysis",
    )?;
    let mut adjacent = Vec::new();
    for left in 0..3 {
        let pooled = compare(arms, left, left + 1, None, None)?;
        require(
            pooled.read_critical_span_ns.left > 0 && pooled.batch_max_read_wall_ns.left > 0,
            "empty K>1 primary",
        )?;
        adjacent.push(Pair {
            left_arm: left,
            right_arm: left + 1,
            pooled_k_gt_one: pooled,
            measured_requests: (0..3)
                .map(|request_index| {
                    Ok(RequestComparison {
                        request_index,
                        comparison: compare(arms, left, left + 1, Some(request_index), None)?,
                    })
                })
                .collect::<Result<_>>()?,
            widths: (2..=WIDTH)
                .map(|width| {
                    Ok(WidthComparison {
                        width,
                        comparison: compare(arms, left, left + 1, None, Some(width))?,
                    })
                })
                .collect::<Result<_>>()?,
            width_one_descriptive: compare(arms, left, left + 1, None, Some(1))?,
        });
    }
    Ok(Analysis {
        adjacent,
        endpoint_b0_b3: compare(arms, 0, 3, None, None)?,
    })
}
fn classify(a: &Analysis) -> &'static str {
    if a.adjacent.iter().any(|p| p.drift(10)) {
        "RUNTIME_LARGE_DRIFT"
    } else if a.adjacent.iter().any(|p| p.drift(3)) {
        "RUNTIME_MATERIAL_DRIFT"
    } else if a.adjacent.iter().all(|p| p.pooled_k_gt_one.stable()) && a.endpoint_b0_b3.stable() {
        "RUNTIME_STABLE"
    } else {
        "AMBIGUOUS"
    }
}
#[derive(Debug, Serialize)]
struct TranscriptDiagnostics {
    source: &'static str,
    begin_byte: usize,
    end_byte: usize,
    sha256: String,
    exact_utf8: String,
}
fn marker_offset(text: &str, marker: &str) -> Result<usize> {
    let mut offset = 0;
    let mut found = Vec::new();
    for line in text.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == marker {
            found.push(offset);
        }
        offset += line.len();
    }
    require(found.len() == 1, "missing/duplicate transcript marker")?;
    Ok(found[0])
}
fn validate_transcript(
    e: &Envelope,
    p: &Payload,
    bytes: &[u8],
) -> Result<Vec<TranscriptDiagnostics>> {
    require(
        sha(bytes) == e.transcript_sha256 && bytes.len() == e.transcript_bytes,
        "transcript identity mismatch",
    )?;
    require(
        pattern_counts(bytes).values().all(|&count| count == 0),
        "retry/breaker transcript contamination",
    )?;
    let text = std::str::from_utf8(bytes)?;
    let begin = marker_offset(text, BEGIN)?;
    let end = marker_offset(text, &format!("{END}{}", e.payload_sha256))?;
    require(
        begin < end
            && text.matches("HMA1G_QUALIFIER_BEGIN").count() == 1
            && text.matches("HMA1G_QUALIFIER_COMPLETE").count() == 1,
        "qualifier marker order/count",
    )?;
    let attempted = if p.complete {
        4
    } else {
        match p.failure.as_ref().and_then(|f| f.position) {
            Some(index) => index.checked_add(1).ok_or("attempted arm index overflow")?,
            None => 0,
        }
    };
    require(
        attempted <= 4
            && text.matches("HMA1G_ARM_BEGIN").count() == attempted
            && text.matches("HMA1G_ARM_END").count() == attempted,
        "attempted arm marker count",
    )?;
    let mut previous = begin;
    let mut diagnostics = Vec::new();
    for index in 0..attempted {
        let link = TranscriptLink::arm(index);
        let left = marker_offset(text, &link.begin_marker)?;
        let right = marker_offset(text, &link.end_marker)? + link.end_marker.len();
        require(
            previous < left && left < right && right < end,
            "arm transcript chronology",
        )?;
        let slice = &bytes[left..right];
        diagnostics.push(TranscriptDiagnostics {
            source: "bound-worker-transcript",
            begin_byte: left,
            end_byte: right,
            sha256: sha(slice),
            exact_utf8: std::str::from_utf8(slice)?.into(),
        });
        previous = right;
    }
    Ok(diagnostics)
}
fn validate_lifecycle(p: &Payload, diagnostics: &[TranscriptDiagnostics]) -> Result<()> {
    require(
        !p.complete && !p.arms.is_empty() && p.arms.len() < 4,
        "lifecycle failure needs authoritative B0 and a later missing arm",
    )?;
    let f = p.failure.as_ref().ok_or("missing failure evidence")?;
    let index = p.arms.len();
    require(
        f.position == Some(index)
            && f.name.as_deref() == Some(ORDER[index])
            && f.transcript == Some(TranscriptLink::arm(index)),
        "failing arm identity/linkage",
    )?;
    require(
        f.incomplete_arm.is_none() && f.records.is_empty(),
        "startup failure must precede source workload",
    )?;
    f.before
        .as_ref()
        .ok_or("failed arm pre status missing")?
        .cleaned(f.after.as_ref().ok_or("failed arm after status missing")?)?;
    let s = f.startup.as_ref().ok_or("unrecognized incomplete arm")?;
    require(
        serde_json::to_value(&f.primary)? == serde_json::to_value(&s.primary)?
            && f.primary.stage == "startup"
            && !f.primary.detail.trim().is_empty(),
        "exact original startup failure required",
    )?;
    require(
        serde_json::to_value(&f.shutdown)? == serde_json::to_value(Some(&s.shutdown))?
            && !s.runtime_validation_completed,
        "startup/shutdown stage mismatch",
    )?;
    let b = &s.benchmark;
    validate_benchmark_header(b)?;
    let previous = &p.arms[0].production["benchmark"];
    for field in [
        "request",
        "provenance",
        "model_identity",
        "production_configuration",
        "production_semantics",
        "cache_reset",
        "warmup_runs",
        "measured_runs",
    ] {
        require(
            !b[field].is_null() && b[field] == previous[field],
            "partial startup provenance/workload drift",
        )?;
    }
    require(
        b["benchmark_complete"] == false
            && b["qualification_pass"] == false
            && b["failure"].is_null()
            && b["warmup_runs_completed"] == 0
            && b["per_run_results"] == json!([])
            && b["aggregate"].is_null(),
        "startup partial report already measured/completed/malformed",
    )?;
    let constructions = b["runtime_constructions"]
        .as_array()
        .ok_or("missing construction evidence")?;
    let shutdowns = b["runtime_shutdowns"]
        .as_array()
        .ok_or("missing shutdown evidence")?;
    let trace = &diagnostics
        .get(index)
        .ok_or("failed arm transcript missing")?
        .exact_utf8;
    if !s.construction_completed {
        require(
            !s.qualification_enable_completed
                && matches!(s.shutdown, Shutdown::NotAttempted)
                && constructions.is_empty()
                && shutdowns.is_empty()
                && b["hardware"].is_null()
                && b["runtime_contract"].is_null()
                && b["model_load"].is_null(),
            "construction failure stage/report mismatch",
        )?;
        // construct_runtime also wraps model/config failures. Recognize only
        // existing device/adapter initialization errors, never that broad code alone.
        let known = [
            (
                "adapter found but request_device failed:",
                "wgpu request_device failed",
            ),
            (
                "no adapters exposed by wgpu;",
                "wgpu request_adapter(HighPerformance) returned no adapter",
            ),
            (
                "only software adapters found by wgpu",
                "wgpu adapter visible",
            ),
            (
                "required feature or limit unsupported by visible wgpu adapters:",
                "wgpu adapter rejected: required feature or limit unsupported",
            ),
        ];
        require(
            f.primary.code == "runtime-construction-failed"
                && known.iter().any(|(detail, event)| {
                    f.primary.detail.contains(detail) && trace.contains(event)
                }),
            "unrecognized construction failure (model/config failures are not lifecycle authority)",
        )?;
    } else {
        require(
            s.qualification_enable_completed
                && matches!(s.shutdown, Shutdown::Succeeded)
                && constructions.len() == 1
                && shutdowns.len() == 1,
            "failed runtime must complete enable and successful shutdown",
        )?;
        require(
            constructions[0]["phase"] == "treatment"
                && constructions[0]["run_index"].is_null()
                && constructions[0]["seconds"]
                    .as_f64()
                    .is_some_and(|s| s.is_finite() && s >= 0.0),
            "partial construction evidence malformed",
        )?;
        require(
            shutdowns[0]["phase"] == "treatment"
                && shutdowns[0]["run_index"].is_null()
                && shutdowns[0]["evidence"]["controlled_shutdown_requested"] == true
                && shutdowns[0]["evidence"]["all_runtime_resources_released"] == true,
            "failed runtime cleanup invalid",
        )?;
        require(
            [
                "wrong-adapter",
                "wrong-gpu-backend",
                "wrong-l4-hardware",
                "software-adapter",
                "missing-authoritative-adapter",
            ]
            .contains(&f.primary.code.as_str())
                && trace.contains("wgpu adapter visible")
                && trace.contains("selected wgpu compute plane"),
            "unrecognized adapter initialization failure",
        )?;
        // validate_and_record_runtime rejects before assigning these fields.
        require(
            b["hardware"].is_null() && b["runtime_contract"].is_null() && b["model_load"].is_null(),
            "unexpected manufactured partial hardware/contract",
        )?;
    }
    Ok(())
}
#[derive(Debug, Serialize)]
struct Audit {
    schema: &'static str,
    raw_report_sha256: String,
    transcript_sha256: String,
    authoritative: bool,
    classification: &'static str,
    completed_arm_count: usize,
    lifecycle_failure: Option<FailedArm>,
    transcript_diagnostics: Vec<TranscriptDiagnostics>,
    analysis: Option<Analysis>,
    failure: Option<String>,
}
fn audit_bytes(raw: &[u8], transcript: &[u8]) -> Audit {
    let mut audit = Audit {
        schema: "mer.gpu-native-baseline-runtime-lifecycle.audit.v1",
        raw_report_sha256: sha(raw),
        transcript_sha256: sha(transcript),
        authoritative: false,
        classification: "NON_AUTHORITATIVE",
        completed_arm_count: 0,
        lifecycle_failure: None,
        transcript_diagnostics: Vec::new(),
        analysis: None,
        failure: None,
    };
    let result = (|| -> Result<()> {
        let e: Envelope = serde_json::from_slice(raw)?;
        require(
            e.schema == SCHEMA && e.payload_sha256 == sha(e.payload_json_utf8.as_bytes()),
            "schema/payload identity mismatch",
        )?;
        let p: Payload = serde_json::from_str(&e.payload_json_utf8)?;
        require(
            serde_json::to_value(&p)? == serde_json::from_str::<Value>(&e.payload_json_utf8)?,
            "missing/extra typed payload authority fields",
        )?;
        require(
            p.schema == SCHEMA
                && p.mode == MODE
                && p.primary == PRIMARY
                && p.cleanup_contract == CLEANUP
                && p.frozen_workload == serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
            "frozen schema/mode/workload contract",
        )?;
        let diagnostics = validate_transcript(&e, &p, transcript)?;
        audit.completed_arm_count = p.arms.len();
        validate_completed(&p.arms)?; // All work/provenance/mechanism gates precede timing.
        if p.complete {
            require(
                p.arms.len() == 4 && p.failure.is_none() && e.worker_exit_success,
                "four-arm completion/exit mismatch",
            )?;
            let analysis = analyze(&p.arms)?;
            audit.classification = classify(&analysis);
            audit.analysis = Some(analysis);
        } else {
            require(
                !e.worker_exit_success,
                "incomplete worker must exit nonzero",
            )?;
            validate_lifecycle(&p, &diagnostics)?;
            audit.classification = "RUNTIME_LIFECYCLE_FAILURE";
            audit.lifecycle_failure = p.failure;
        }
        audit.transcript_diagnostics = diagnostics;
        audit.authoritative = true;
        Ok(())
    })();
    if let Err(error) = result {
        audit.failure = Some(error.to_string());
    }
    audit
}
/// Read each immutable input once. No preparation, model, storage or runtime call.
pub(crate) fn audit_command(raw: &Path, transcript: &Path, report_out: &Path) -> Result<()> {
    require(
        raw != transcript && raw != report_out && transcript != report_out,
        "artifact paths must differ",
    )?;
    canonical_output(report_out)?;
    let bytes = std::fs::read(raw)?;
    let transcript = std::fs::read(transcript)?;
    let audit = audit_bytes(&bytes, &transcript);
    write_new(report_out, &serde_json::to_vec_pretty(&audit)?)?;
    require(
        audit.authoritative,
        "HMA-1G NON_AUTHORITATIVE; see audit report",
    )
}

#[cfg(test)]
#[path = "gpu_native_baseline_runtime_lifecycle_tests.rs"]
mod tests;
