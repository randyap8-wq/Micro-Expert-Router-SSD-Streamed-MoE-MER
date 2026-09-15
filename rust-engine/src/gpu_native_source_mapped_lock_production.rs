//! Frozen HMA-1F-A: four fresh mapped production arms and an offline auditor.
//! Reuses production-v2 preparation, request execution, runtime gate and teardown.
use super::*;
use crate::gpu_native_mapped_lock::{Mode, Observer, ProcessMemory, Record, ORDER, WIDTH};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

pub(crate) const MODE: &str = "qualify-gpu-native-source-mapped-lock-production";
pub(crate) const SCHEMA: &str = "mer.gpu-native-source-mapped-lock-production.v1";
pub(crate) const BEGIN: &str =
    "HMA1F_QUALIFIER_BEGIN order=mapped-baseline,mapped-locked,mapped-locked,mapped-baseline";
pub(crate) const END: &str = "HMA1F_QUALIFIER_COMPLETE payload_sha256=";
const PRIMARY: &str = "Measured K>1 only. Locked minus baseline; percentage denominator is the corresponding baseline integer total. Width 1 and batch_sum_read_wall_ns are descriptive only.";
const CLEANUP: &str = "Exact per-process VmLck bytes and RLIMIT_MEMLOCK equality before isolated runtime construction and after controlled shutdown. No global Mlocked sampling or tolerance.";
const PATTERNS: [&str; 4] = [
    "transient I/O error; retrying",
    "expert fetch recovered after retry",
    "expert fetch failed; will retry",
    "circuit breaker",
];
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(why.into())
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| crate::gpu_native_real_benchmark::is_hex(s, 64))
}
fn decode<T: DeserializeOwned + Serialize>(v: &Value) -> Result<T> {
    let t: T = serde_json::from_value(v.clone())?;
    require(
        serde_json::to_value(&t)? == *v,
        "missing/extra typed authority fields",
    )?;
    Ok(t)
}
pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmData {
    position: usize,
    mode: Mode,
    before: ProcessMemory,
    after: ProcessMemory,
    production: Value,
    records: Vec<Record>,
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
    failure: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    /// Exact worker snapshot; preserve numeric spelling and whitespace when binding bytes.
    payload_json_utf8: String,
    /// Hash of the exact payload bytes, also recorded by the worker's end marker.
    payload_sha256: String,
    transcript_sha256: String,
    transcript_bytes: usize,
}
/// The internal worker alone executes exactly four arms. The public launcher
/// binds its complete stdout+stderr transcript after process exit.
pub(crate) async fn run_worker(args: CommandArgs) -> Result<()> {
    println!("{BEGIN}");
    let mut payload = Payload {
        schema: SCHEMA.into(),
        mode: MODE.into(),
        primary: PRIMARY.into(),
        cleanup_contract: CLEANUP.into(),
        frozen_workload: serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
        arms: Vec::new(),
        complete: false,
        failure: None,
    };
    let result = async {
        require(
            args.expected_adapter_name == "NVIDIA L4",
            "HMA-1F requires NVIDIA L4",
        )?;
        let prepared = prepare(&args)?;
        for (position, mode) in ORDER.into_iter().enumerate() {
            let before = ProcessMemory::capture()?;
            before.capacity((CAPACITY * FULL) as u64)?;
            let observer = Arc::new(Observer::new(mode));
            let run = run_physical_install_arm_observed(
                &prepared,
                &args,
                PhysicalInstallQualificationRun::SourceToUpload(Arm::Treatment),
                Some(MODE),
                Some(observer.clone()),
            )
            .await?;
            let after = ProcessMemory::capture()?;
            let failure = run.common.failure.as_ref().map(ToString::to_string);
            let production = serde_json::to_value(UploadArmReport {
                run: ConcurrencyArmReport {
                    common: run.common,
                    warmup_mechanism: run.warmup_concurrency,
                    mechanism: run.concurrency,
                },
                warmup_upload: run.warmup_upload,
                upload: run.upload,
            })?;
            payload.arms.push(ArmData {
                position,
                mode,
                before,
                after,
                production,
                records: observer.records(),
            });
            if let Some(failure) = failure {
                return Err(failure.into());
            }
            payload
                .arms
                .last()
                .unwrap()
                .before
                .cleaned(&payload.arms.last().unwrap().after)?;
            // Independently validate each completed runtime before constructing the next.
            validate_arm(payload.arms.last().unwrap())?;
        }
        validate_work(&payload.arms)?;
        Ok::<(), Box<dyn std::error::Error>>(())
    }
    .await;
    payload.complete = result.is_ok();
    payload.failure = result.as_ref().err().map(ToString::to_string);
    let bytes = serde_json::to_vec(&payload)?;
    write_new(&args.report_out, &bytes)?;
    println!("{END}{}", sha(&bytes));
    std::io::stdout().flush()?;
    result
}
/// Parent performs no model/GPU/storage work. Both child streams share one file
/// description; the wait binds complete bytes, including startup and teardown.
pub(crate) fn launch(
    config: &Path,
    report_out: &Path,
    transcript_out: &Path,
    raw_args: &[std::ffi::OsString],
) -> Result<()> {
    require(
        report_out != transcript_out,
        "raw report and transcript must differ",
    )?;
    let draft = report_out.with_extension("payload.json");
    require(
        !report_out.exists() && !draft.exists() && !transcript_out.exists(),
        "HMA-1F outputs must be fresh",
    )?;
    let transcript = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(transcript_out)?;
    let args = worker_arguments(config, &draft, raw_args)?;
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(args)
        .stdout(transcript.try_clone()?)
        .stderr(transcript)
        .status()?;
    // Read once, hash those same bytes, deserialize only that snapshot.
    let transcript_bytes = std::fs::read(transcript_out)?;
    let draft_bytes = std::fs::read(&draft)?;
    let payload_json_utf8 = String::from_utf8(draft_bytes.clone())?;
    let envelope = Envelope {
        schema: SCHEMA.into(),
        payload_json_utf8,
        payload_sha256: sha(&draft_bytes),
        transcript_sha256: sha(&transcript_bytes),
        transcript_bytes: transcript_bytes.len(),
    };
    let bytes = serde_json::to_vec_pretty(&envelope)?;
    write_new(report_out, &bytes)?;
    let audit = audit_bytes(&bytes, &transcript_bytes);
    eprintln!(
        "HMA1F_RESULT={} raw_report_sha256={} transcript_sha256={}",
        audit.classification,
        sha(&bytes),
        envelope.transcript_sha256
    );
    require(
        status.success() && audit.authoritative,
        "HMA-1F run non-authoritative; see raw report/transcript",
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
        "HMA-1F forbids autotune/probe workloads",
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
        "hma1f-worker-internal".into(),
        "--config".into(),
        config.as_os_str().to_owned(),
        "--report-out".into(),
        draft.as_os_str().to_owned(),
    ]);
    Ok(args)
}
fn context_id(b: &Value) -> Result<u64> {
    let s = b["runtime_contract"]["legacy_execution_plan"]["context_id"]
        .as_str()
        .ok_or("context ID missing/not string")?;
    let id = s.parse::<u64>()?;
    require(
        id > 0 && id.to_string() == s,
        "context ID must be a canonical positive u64",
    )?;
    Ok(id)
}
fn contracts_equal(a: &Value, b: &Value) -> Result<()> {
    context_id(a)?;
    context_id(b)?;
    for field in [
        "request",
        "production_configuration",
        "production_semantics",
        "model_identity",
        "model_load",
        "hardware",
        "provenance",
    ] {
        require(
            a[field].is_object() && a[field] == b[field],
            "arm semantic authority drift",
        )?;
    }
    let mut ac = a["runtime_contract"].clone();
    let mut bc = b["runtime_contract"].clone();
    // Sole explicit instance-identity exception. No recursive normalization.
    ac["legacy_execution_plan"]
        .as_object_mut()
        .ok_or("context object")?
        .remove("context_id");
    bc["legacy_execution_plan"]
        .as_object_mut()
        .ok_or("context object")?
        .remove("context_id");
    require(ac == bc, "second runtime-contract difference")
}
fn mapped_ring_exact(u: &UploadSnapshot) -> bool {
    let m = &u.metrics;
    u.ring_capacity == CAPACITY
        && u.active_leases == 0
        && u.pending_leases == 0
        && m.high_water > 0
        && m.high_water <= CAPACITY as u64
        && m.acquisition_attempts == m.leases_created
        && m.leases_created == m.direct_source_reads
        && m.leases_consumed == m.leases_created
        && m.leases_released == m.leases_created
        && m.leases_consumed == m.fused_installs
        && m.map_attempts == m.map_completions
        && m.map_completions == m.leases_created
        && m.unmaps == m.map_completions
        && m.remap_attempts == m.remap_completions
        && m.remap_completions <= m.map_completions
        && m.copy_submissions > 0
        && m.copy_submissions == m.copy_command_buffers
        && m.copy_submissions == m.fused_install_sets
        && m.copied_experts == m.fused_installs
        && m.copied_bytes == m.fused_gpu_copy_bytes
}
fn validate_arm(a: &ArmData) -> Result<()> {
    a.before.capacity((CAPACITY * FULL) as u64)?;
    a.before.cleaned(&a.after)?;
    let p = &a.production;
    let b = &p["benchmark"];
    require(
        p["arm"] == "treatment"
            && p["complete"] == true
            && p["failure"].is_null()
            && p["isolated_runtime"] == true,
        "incomplete/nonmapped/nonisolated arm",
    )?;
    require(
        b["benchmark_complete"] == true && b.get("failure") == Some(&Value::Null),
        "incomplete benchmark",
    )?;
    require(
        b["cache_reset"] == "keep"
            && b["warmup_runs"] == 1
            && b["warmup_runs_completed"] == 1
            && b["measured_runs"] == 3,
        "frozen request/reset schedule",
    )?;
    require(
        b["request"]["greedy"] == true
            && b["request"]["requested_output_tokens"] == 128
            && b["request"]["prompt_sha256"] == sha(FROZEN_PROMPT.as_bytes())
            && hash(&b["request"]["prompt_token_ids_sha256"])
            && b["request"]["prompt_token_count"]
                .as_u64()
                .is_some_and(|n| n > 0),
        "frozen prompt/request authority",
    )?;
    require(
        hash(&b["provenance"]["executable_sha256"])
            && hash(&b["provenance"]["resolved_config_sha256"])
            && b["provenance"]["build"]["dirty"] == false
            && b["provenance"]["build"]["git_sha"]
                .as_str()
                .is_some_and(|s| crate::gpu_native_real_benchmark::is_hex(s, 40))
            && b["provenance"]["artifacts"]["config"]["sha256"] == FROZEN_CONFIG_SHA256,
        "build/config provenance",
    )?;
    require(b["production_semantics"] == serde_json::to_value(crate::gpu_native_real_benchmark::ProductionSemantics::physical_source_of_truth_pr1bb())?, "production semantics drift")?;
    require(
        b["model_identity"]
            == json!({"architecture":"qwen3_moe","num_layers":48,"num_experts_per_layer":128,"total_experts":6144,"top_k":8,"d_model":2048,"d_ff":768,"routed_expert_dtype":"q4_0"}),
        "model identity authority",
    )?;
    let config: ProductionConfiguration = decode(&b["production_configuration"])?;
    let c = &config.cache_residency;
    let ppc = &config.predictor_prefetch;
    require(
        config.q4_dtype == "q4_0"
            && config.q4_layout.as_deref() == Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
            && c.block_align == 4096
            && c.direct_io
            && c.gpu_cache_enabled
            && c.gpu_vram_capacity_mb > 0
            && c.pin_after_observations == 0
            && c.packed_blob.is_none()
            && c.packed_manifest.is_none()
            && ppc.predict_fanout == 0
            && !ppc.locality_enabled
            && !ppc.speculator_enabled
            && !ppc.affinity_enabled
            && !ppc.pregate_enabled
            && !ppc.cost_aware_eviction
            && ppc.static_residency_fraction == 0.0
            && ppc.static_residency_profile.is_none(),
        "production configuration authority",
    )?;
    crate::gpu_native_real_benchmark::audit_recorded_runtime(b)?;
    context_id(b)?;
    let hardware = &b["hardware"];
    require(
        hardware["name"] == "NVIDIA L4"
            && hardware["wgpu_backend"] == "vulkan"
            && hardware["device_type"] == "DiscreteGpu"
            && hardware["vendor_id"] == 0x10de
            && hardware["software_adapter"] == false,
        "hardware authority",
    )?;
    let shutdown = b["runtime_shutdowns"]
        .as_array()
        .ok_or("shutdown evidence")?;
    let startup = b["runtime_constructions"]
        .as_array()
        .ok_or("construction evidence")?;
    require(
        startup.len() == 1
            && startup[0]["phase"] == "treatment"
            && startup[0]["run_index"].is_null()
            && shutdown.len() == 1
            && shutdown[0]["phase"] == "treatment"
            && shutdown[0]["run_index"].is_null()
            && shutdown[0]["evidence"]["controlled_shutdown_requested"] == true
            && shutdown[0]["evidence"]["all_runtime_resources_released"] == true,
        "isolated runtime lifecycle",
    )?;
    require(
        hash(&p["warmup_ram_cache_state_sha256"]),
        "warmup cache identity",
    )?;
    for (key, count) in [("warmup_results", 1), ("measured", 3)] {
        let runs = if key == "measured" {
            &b["per_run_results"]
        } else {
            &p[key]
        };
        let runs = runs.as_array().ok_or("missing generated runs")?;
        require(runs.len() == count, "generated run count")?;
        for (i, r) in runs.iter().enumerate() {
            require(
                r["run_index"] == i
                    && r["generated_tokens"] == 128
                    && hash(&r["generated_token_ids_sha256"])
                    && hash(&r["generated_text_sha256"]),
                "generated run identity",
            )?;
            if key == "measured" {
                let ids: Vec<u32> = serde_json::from_value(r["generated_token_ids"].clone())?;
                require(
                    ids.len() == 128
                        && r["generated_token_ids_sha256"]
                            == crate::greedy_parity::token_ids_sha256(&ids),
                    "token stream hash",
                )?;
            }
        }
    }
    validate_request_counters(b, p)?;
    require(
        a.records.windows(2).all(|w| {
            (w[0].measured, w[0].request_index, w[0].source_index)
                < (w[1].measured, w[1].request_index, w[1].source_index)
        }),
        "warmup/measured source chronology",
    )?;
    for measured in [false, true] {
        let prefix = if measured { "" } else { "warmup_" };
        let mk = if measured {
            "mechanism"
        } else {
            "warmup_mechanism"
        };
        let s: Snapshot = decode(&p[mk])?;
        let u: UploadSnapshot = decode(&p[format!("{prefix}upload")])?;
        let pi: GpuNativeProductionPhysicalInstallSnapshot =
            decode(&p[format!("{prefix}production_physical_install")])?;
        let source: ProductionDemandSourceSnapshot = decode(&p[format!("{prefix}production")])?;
        let w: ArmWorkEvidence = decode(&p[format!("{prefix}work")])?;
        require(
            p[format!("{prefix}source")] == serde_json::to_value(concurrency_common_snapshot(&s))?,
            "source snapshot reconstruction",
        )?;
        // Reuse the exact mapped-side production-v2 equations and failure gates.
        let gate = pair_mechanism_gate(&s, &s, &pi, &pi, &u, &u, &source, &source);
        require(
            u.arm == Arm::Treatment
                && u.production_owned
                && mapped_ring_exact(&u)
                && gate.treatment_fused_and_fallback_bytes_exact
                && gate.treatment_every_nvme_read_fused
                && install_exact(&s, &pi, &u)
                && upload_errors_zero(&u)
                && zero_fill_production::failures_zero(&s)
                && zero_fill_production::timing_accounting_exact(&s)
                && u.metrics.logical_materialization_operations
                    == u.metrics.shared_payload_constructions
                && u.metrics
                    .logical_materialization_operations
                    .checked_mul(PAYLOAD as u64)
                    == Some(u.metrics.logical_materialization_bytes),
            "mapped production mechanism authority",
        )?;
        require(
            work_errors_zero(&w)
                && arm_all_speculative_work_zero(&w)
                && production_safety_zero(&source)
                && source.ordinary_production_path_exercised
                && source.production_sequential_fallback_batch_read_error == 0
                && source.production_batch_attempts == source.production_batch_successes
                && w.gpu_native_residency.ram_to_vram_installs == s.physical_install_completions
                && w.gpu_native_residency
                    .logical_admissions_for_physical_misses
                    == u.metrics.logical_admissions
                && w.engine_storage.nvme_bytes_read == s.source_nvme_bytes
                && s.source_nvme_reads
                    .checked_sub(source.production_batch_experts)
                    .and_then(|n| n.checked_add(source.production_batch_successes))
                    == Some(w.engine_storage.nvme_read_operations),
            "production work/recovery/source authority",
        )?;
        let proof = u
            .source_upload_fd_proof
            .as_ref()
            .ok_or("fd proof missing")?;
        require(
            proof.source_upload_fd_proof_requests == u.metrics.direct_source_reads
                && proof
                    .source_upload_fd_proof_hits
                    .checked_add(proof.source_upload_fd_proof_misses)
                    == Some(proof.source_upload_fd_proof_requests)
                && proof.source_upload_fd_proof_failures == 0
                && (measured || proof.source_upload_fd_proof_misses > 0),
            "fd proof authority",
        )?;
        validate_records(a, measured, &s, &u, &source)?;
    }
    Ok(())
}
/// Re-run the production benchmark's per-request postconditions and reconcile
/// retained before/after counters, rather than trusting cached completion flags.
fn validate_request_counters(b: &Value, arm: &Value) -> Result<()> {
    use crate::gpu_native_real_benchmark::{
        recovery_delta, routed_delta, token_loop_delta, validate_request_postconditions,
    };
    let prompt = b["request"]["prompt_token_count"]
        .as_u64()
        .ok_or("prompt count")? as usize;
    let mut totals = BTreeMap::<String, BTreeMap<String, u64>>::new();
    for r in b["per_run_results"].as_array().ok_or("per-run counters")? {
        require(
            r["prompt_tokens"] == prompt && r["requested_output_tokens"] == 128,
            "per-run request drift",
        )?;
        let c = &r["counters"];
        let token = token_loop_delta(
            decode(&c["token_loop_before"])?,
            decode(&c["token_loop_after"])?,
        )?;
        let recovery = recovery_delta(
            decode(&c["recovery_before"])?,
            decode(&c["recovery_after"])?,
        )?;
        let routed = routed_delta(
            decode(&c["routed_execution_before"])?,
            decode(&c["routed_execution_after"])?,
        )?;
        let storage_after: EngineStorageSnapshot = decode(&c["engine_storage_after"])?;
        let storage = storage_after.checked_delta(decode(&c["engine_storage_before"])?)?;
        validate_request_postconditions(prompt, 128, 128, token, recovery, routed)?;
        for (field, work, value) in [
            (
                "token_loop_delta",
                "token_loop",
                serde_json::to_value(token)?,
            ),
            (
                "recovery_delta",
                "recovery",
                serde_json::to_value(recovery)?,
            ),
            (
                "routed_execution_delta",
                "routed_execution",
                serde_json::to_value(routed)?,
            ),
            (
                "engine_storage_delta",
                "engine_storage",
                serde_json::to_value(storage)?,
            ),
        ] {
            require(c[field] == value, "per-run counter reconstruction mismatch")?;
            for (key, n) in value.as_object().ok_or("counter object")? {
                let sum = totals
                    .entry(work.into())
                    .or_default()
                    .entry(key.clone())
                    .or_default();
                *sum = sum
                    .checked_add(n.as_u64().ok_or("counter integer")?)
                    .ok_or("counter sum overflow")?;
            }
        }
    }
    for (work, total) in totals {
        require(
            arm["work"][&work] == serde_json::to_value(total)?,
            "request counters do not reconcile with measured work",
        )?;
    }
    let warm: ArmWorkEvidence = decode(&arm["warmup_work"])?;
    validate_request_postconditions(
        prompt,
        128,
        128,
        warm.token_loop,
        warm.recovery,
        warm.routed_execution,
    )?;
    Ok(())
}

fn validate_records(
    a: &ArmData,
    measured: bool,
    s: &Snapshot,
    u: &UploadSnapshot,
    source: &ProductionDemandSourceSnapshot,
) -> Result<()> {
    let records: Vec<_> = a
        .records
        .iter()
        .filter(|r| r.measured == measured)
        .collect();
    require(!records.is_empty(), "missing source stream")?;
    let mut ids_hash = Sha256::new();
    let mut bytes = 0u64;
    let mut reads = 0u64;
    let mut batch_experts = 0u64;
    let mut batches = 0u64;
    let mut positions = BTreeMap::<usize, usize>::new();
    let mut last = 0usize;
    for r in &records {
        let width = r.ids.len();
        require(
            (1..=WIDTH).contains(&width)
                && r.request_index < if measured { 3 } else { 1 }
                && r.request_index >= last
                && r.source_index == *positions.entry(r.request_index).or_default()
                && r.failure.is_none()
                && r.bytes == width * FULL
                && r.ids.iter().all(|&id| id < 48 * 128)
                && r.ids.iter().collect::<BTreeSet<_>>().len() == width,
            "source record identity/order/bytes/failure",
        )?;
        last = r.request_index;
        *positions.get_mut(&r.request_index).unwrap() += 1;
        r.timing
            .as_ref()
            .ok_or("missing timing")?
            .validate(width, r.caller_ns)?;
        r.locks.validate(a.mode, width)?;
        for id in &r.ids {
            ids_hash.update(id.to_le_bytes());
        }
        reads = reads
            .checked_add(width as u64)
            .ok_or("read count overflow")?;
        bytes = bytes
            .checked_add(r.bytes as u64)
            .ok_or("source bytes overflow")?;
        if width > 1 {
            batches += 1;
            batch_experts += width as u64;
        }
    }
    require(
        positions.len() == if measured { 3 } else { 1 },
        "request source stream missing",
    )?;
    require(
        format!("{:x}", ids_hash.finalize()) == u.ordered_nvme_ids_sha256
            && reads == s.source_nvme_reads
            && bytes == s.source_nvme_bytes
            && reads == u.metrics.direct_source_reads
            && bytes == u.metrics.direct_source_bytes
            && batches == source.production_batch_successes
            && batch_experts == source.production_batch_experts,
        "source stream accounting/hash/width reconciliation",
    )
}
fn validate_work(arms: &[ArmData]) -> Result<()> {
    require(arms.len() == 4, "exactly four arms required")?;
    let mut ids = BTreeSet::new();
    for (i, a) in arms.iter().enumerate() {
        require(
            a.position == i && a.mode == ORDER[i],
            "four-arm order mismatch",
        )?;
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct Endpoints {
    critical: u64,
    max_wall: u64,
}
impl Endpoints {
    fn add(&mut self, other: Self) -> Result<()> {
        self.critical = self
            .critical
            .checked_add(other.critical)
            .ok_or("critical total overflow")?;
        self.max_wall = self
            .max_wall
            .checked_add(other.max_wall)
            .ok_or("max-wall total overflow")?;
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct Comparison {
    baseline: Endpoints,
    locked: Endpoints,
    dlock_crit: i128,
    dlock_max: i128,
}
impl Comparison {
    fn new(baseline: Endpoints, locked: Endpoints) -> Self {
        Self {
            baseline,
            locked,
            dlock_crit: i128::from(locked.critical) - i128::from(baseline.critical),
            dlock_max: i128::from(locked.max_wall) - i128::from(baseline.max_wall),
        }
    }
    fn negative(&self) -> bool {
        self.dlock_crit < 0 && self.dlock_max < 0
    }
    fn positive(&self) -> bool {
        self.dlock_crit > 0 && self.dlock_max > 0
    }
    fn at_most(&self, p: i128) -> bool {
        self.dlock_crit * 100 <= i128::from(self.baseline.critical) * p
            && self.dlock_max * 100 <= i128::from(self.baseline.max_wall) * p
    }
    fn at_least(&self, p: i128) -> bool {
        self.dlock_crit * 100 >= i128::from(self.baseline.critical) * p
            && self.dlock_max * 100 >= i128::from(self.baseline.max_wall) * p
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Stratum {
    block: Option<usize>,
    request: Option<usize>,
    width: Option<usize>,
    comparison: Comparison,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Analysis {
    pooled: Comparison,
    blocks: Vec<Stratum>,
    request_pairs: Vec<Stratum>,
    widths: Vec<Stratum>,
    width_one: Comparison,
}
fn totals(records: &[Record], request: Option<usize>, width: Option<usize>) -> Result<Endpoints> {
    let mut total = Endpoints::default();
    for r in records.iter().filter(|r| {
        r.measured
            && request.is_none_or(|i| r.request_index == i)
            && width.map_or(r.ids.len() > 1, |w| r.ids.len() == w)
    }) {
        let t = r.timing.as_ref().ok_or("missing timing")?;
        total.add(Endpoints {
            critical: t.read_critical_span_ns,
            max_wall: t.batch_max_read_wall_ns,
        })?;
    }
    Ok(total)
}
fn compare_block(
    arms: &[ArmData],
    block: usize,
    request: Option<usize>,
    width: Option<usize>,
) -> Result<Comparison> {
    let (b, l) = if block == 0 { (0, 1) } else { (3, 2) };
    Ok(Comparison::new(
        totals(&arms[b].records, request, width)?,
        totals(&arms[l].records, request, width)?,
    ))
}
fn pool(comparisons: impl Iterator<Item = Comparison>) -> Result<Comparison> {
    let mut b = Endpoints::default();
    let mut l = Endpoints::default();
    for c in comparisons {
        b.add(c.baseline)?;
        l.add(c.locked)?;
    }
    Ok(Comparison::new(b, l))
}
fn analyze(arms: &[ArmData]) -> Result<Analysis> {
    require(arms.len() == 4, "four arms for endpoints")?;
    let mut blocks = Vec::new();
    let mut request_pairs = Vec::new();
    for block in 0..2 {
        blocks.push(Stratum {
            block: Some(block),
            request: None,
            width: None,
            comparison: compare_block(arms, block, None, None)?,
        });
        for request in 0..3 {
            request_pairs.push(Stratum {
                block: Some(block),
                request: Some(request),
                width: None,
                comparison: compare_block(arms, block, Some(request), None)?,
            });
        }
    }
    let mut widths = Vec::new();
    for width in 2..=WIDTH {
        widths.push(Stratum {
            block: None,
            request: None,
            width: Some(width),
            comparison: pool(
                [
                    compare_block(arms, 0, None, Some(width))?,
                    compare_block(arms, 1, None, Some(width))?,
                ]
                .into_iter(),
            )?,
        });
    }
    let width_one = pool(
        [
            compare_block(arms, 0, None, Some(1))?,
            compare_block(arms, 1, None, Some(1))?,
        ]
        .into_iter(),
    )?;
    let pooled = pool(blocks.iter().map(|b| b.comparison))?;
    require(
        pooled.baseline.critical > 0 && pooled.baseline.max_wall > 0,
        "empty K>1 primary",
    )?;
    Ok(Analysis {
        pooled,
        blocks,
        request_pairs,
        widths,
        width_one,
    })
}
fn classify(a: &Analysis) -> &'static str {
    let p = &a.pooled;
    let neg = a.blocks.len() == 2
        && a.blocks.iter().all(|s| s.comparison.negative())
        && a.request_pairs.len() == 6
        && a.request_pairs
            .iter()
            .filter(|s| s.comparison.negative())
            .count()
            >= 4
        && a.widths.len() == 7
        && a.widths
            .iter()
            .filter(|s| s.comparison.dlock_crit < 0)
            .count()
            >= 5;
    let pos = a.blocks.len() == 2
        && a.blocks.iter().all(|s| s.comparison.positive())
        && a.request_pairs.len() == 6
        && a.request_pairs
            .iter()
            .filter(|s| s.comparison.positive())
            .count()
            >= 4
        && a.widths.len() == 7
        && a.widths
            .iter()
            .filter(|s| s.comparison.dlock_crit > 0)
            .count()
            >= 5;
    if p.at_most(-3) && neg {
        "RESIDENCY_LOCK_MITIGATES_STRAGGLER"
    } else if p.at_most(-1) && !p.at_most(-3) && neg {
        "RESIDENCY_LOCK_DIRECTIONAL_MITIGATION"
    } else if p.dlock_crit * 100 > -i128::from(p.baseline.critical)
        && p.dlock_crit * 100 < i128::from(p.baseline.critical)
        && p.dlock_max * 100 > -i128::from(p.baseline.max_wall)
        && p.dlock_max * 100 < i128::from(p.baseline.max_wall)
    {
        "RESIDENCY_LOCK_NO_MATERIAL_EFFECT"
    } else if p.at_least(3) && pos {
        "RESIDENCY_LOCK_WORSENS"
    } else {
        "AMBIGUOUS_LOCK_INTERACTION"
    }
}
fn pattern_counts(transcript: &[u8]) -> BTreeMap<String, usize> {
    PATTERNS
        .into_iter()
        .map(|p| {
            (
                p.to_string(),
                transcript
                    .windows(p.len())
                    .filter(|w| *w == p.as_bytes())
                    .count(),
            )
        })
        .collect()
}
fn validate_transcript(e: &Envelope, transcript: &[u8]) -> Result<()> {
    require(
        sha(transcript) == e.transcript_sha256 && transcript.len() == e.transcript_bytes,
        "transcript identity mismatch",
    )?;
    let t = std::str::from_utf8(transcript)?;
    require(
        t.lines().filter(|l| *l == BEGIN).count() == 1
            && t.matches("HMA1F_QUALIFIER_BEGIN").count() == 1
            && t.matches("HMA1F_QUALIFIER_COMPLETE").count() == 1
            && t.trim_end()
                .ends_with(&format!("{END}{}", e.payload_sha256)),
        "incomplete/duplicate transcript markers",
    )?;
    require(
        pattern_counts(transcript).values().all(|&n| n == 0),
        "retry/breaker transcript contamination",
    )
}
#[derive(Debug, Serialize)]
struct Audit {
    schema: &'static str,
    raw_report_sha256: String,
    transcript_sha256: String,
    transcript_pattern_counts: BTreeMap<String, usize>,
    authoritative: bool,
    classification: &'static str,
    analysis: Option<Analysis>,
    failure: Option<String>,
}
fn audit_bytes(raw: &[u8], transcript: &[u8]) -> Audit {
    let mut audit = Audit {
        schema: "mer.gpu-native-source-mapped-lock-production.audit.v1",
        raw_report_sha256: sha(raw),
        transcript_sha256: sha(transcript),
        transcript_pattern_counts: pattern_counts(transcript),
        authoritative: false,
        classification: "NON_AUTHORITATIVE",
        analysis: None,
        failure: None,
    };
    let result = (|| -> Result<Analysis> {
        let e: Envelope = serde_json::from_slice(raw)?;
        require(
            e.payload_sha256 == sha(e.payload_json_utf8.as_bytes()),
            "payload identity mismatch",
        )?;
        let payload: Payload = serde_json::from_slice(e.payload_json_utf8.as_bytes())?;
        require(
            e.schema == SCHEMA
                && payload.schema == SCHEMA
                && payload.mode == MODE
                && payload.complete
                && payload.failure.is_none(),
            "incomplete/schema/mode failure",
        )?;
        validate_transcript(&e, transcript)?; // Must pass before classifier is reachable.
        require(
            payload.primary == PRIMARY
                && payload.cleanup_contract == CLEANUP
                && payload.frozen_workload
                    == serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
            "frozen experiment contract",
        )?;
        validate_work(&payload.arms)?;
        analyze(&payload.arms)
    })();
    match result {
        Ok(analysis) => {
            audit.classification = classify(&analysis);
            audit.authoritative = true;
            audit.analysis = Some(analysis);
        }
        Err(e) => audit.failure = Some(e.to_string()),
    }
    audit
}
/// CPU-only, read-once immutable snapshots; dispatch precedes all runtime setup.
pub(crate) fn audit_command(raw: &Path, transcript: &Path, report_out: &Path) -> Result<()> {
    require(
        raw != report_out && transcript != report_out && raw != transcript,
        "artifact paths must differ",
    )?;
    let bytes = std::fs::read(raw)?;
    let transcript = std::fs::read(transcript)?;
    let audit = audit_bytes(&bytes, &transcript);
    write_new(report_out, &serde_json::to_vec_pretty(&audit)?)?;
    require(
        audit.authoritative,
        "HMA-1F NON_AUTHORITATIVE; see audit report",
    )
}

#[cfg(test)]
mod hma1f_tests {
    use super::*;
    use crate::gpu_native_mapped_lock::{LockEvidence, Range, Read, Timing};
    fn record(mode: Mode, measured: bool, request_index: usize, width: usize, wall: u64) -> Record {
        let ranges = (1..=width)
            .map(|i| Range {
                pointer: i * (FULL + 4096),
                length: FULL,
            })
            .collect();
        let mut locks = LockEvidence::new(mode, ranges);
        if mode == Mode::MappedLocked {
            locks.lock_attempts = width as u64;
            locks.lock_successes = width as u64;
            locks.unlock_attempts = width as u64;
            locks.unlock_successes = width as u64;
            locks.mlock_wall_ns = 100;
            locks.munlock_wall_ns = 100;
        }
        Record {
            measured,
            request_index,
            source_index: 0,
            ids: (0..width as u32).collect(),
            bytes: width * FULL,
            caller_ns: wall + 20,
            timing: Some(
                Timing::from_reads(
                    (0..width)
                        .map(|_| Read {
                            wrapper_start_ns: 10,
                            wrapper_end_ns: 10 + wall,
                            wrapper_wall_ns: wall,
                            success: true,
                        })
                        .collect(),
                )
                .unwrap(),
            ),
            locks,
            failure: None,
        }
    }
    fn scale(mut v: Value, n: u64) -> Value {
        for (k, v) in v.as_object_mut().unwrap() {
            if ![
                "ring_capacity",
                "high_water",
                "install_set_width_min",
                "install_set_width_max",
                "rayon_num_threads",
                "max_in_flight_physical_staging",
                "primary_pool_capacity",
                "shadow_pool_capacity",
            ]
            .contains(&k.as_str())
            {
                if let Some(value) = v.as_u64() {
                    *v = json!(value * n);
                }
            }
        }
        v
    }
    fn fixture() -> Payload {
        let mut arms = Vec::new();
        for (position, mode) in ORDER.into_iter().enumerate() {
            let mut records = vec![record(mode, false, 0, 2, 100)];
            records.extend((0..3).map(|r| record(mode, true, r, 2, 100)));
            let mut b = crate::gpu_native_real_benchmark::hma1f_test_runtime();
            b["runtime_contract"]["legacy_execution_plan"]["context_id"] =
                json!((position + 1).to_string());
            b["benchmark_complete"] = json!(true);
            b["failure"] = Value::Null;
            b["cache_reset"] = json!("keep");
            b["warmup_runs"] = json!(1);
            b["warmup_runs_completed"] = json!(1);
            b["measured_runs"] = json!(3);
            b["request"] = json!({"prompt_sha256":sha(FROZEN_PROMPT.as_bytes()),"prompt_token_ids_sha256":"a".repeat(64),"prompt_token_count":1,"requested_output_tokens":128,"greedy":true});
            b["provenance"] = json!({"executable_sha256":"a".repeat(64),"resolved_config_sha256":"b".repeat(64),"build":{"dirty":false,"git_sha":"c".repeat(40)},"artifacts":{"config":{"sha256":FROZEN_CONFIG_SHA256}}});
            b["production_semantics"]=serde_json::to_value(crate::gpu_native_real_benchmark::ProductionSemantics::physical_source_of_truth_pr1bb()).unwrap();
            b["model_identity"] = json!({"architecture":"qwen3_moe","num_layers":48,"num_experts_per_layer":128,"total_experts":6144,"top_k":8,"d_model":2048,"d_ff":768,"routed_expert_dtype":"q4_0"});
            let mut cfg = ProductionConfiguration::default();
            cfg.q4_dtype = "q4_0".into();
            cfg.q4_layout = Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1.into());
            cfg.cache_residency.block_align = 4096;
            cfg.cache_residency.direct_io = true;
            cfg.cache_residency.gpu_cache_enabled = true;
            cfg.cache_residency.gpu_vram_capacity_mb = 8192;
            b["production_configuration"] = serde_json::to_value(cfg).unwrap();
            b["runtime_constructions"] =
                json!([{"phase":"treatment","run_index":null,"seconds":1.0}]);
            b["runtime_shutdowns"] = json!([{"phase":"treatment","run_index":null,"evidence":{"controlled_shutdown_requested":true,"all_runtime_resources_released":true}}]);
            let ids = vec![17u32; 128];
            let token_hash = crate::greedy_parity::token_ids_sha256(&ids);
            b["per_run_results"]=json!((0..3).map(|i|json!({"run_index":i,"generated_tokens":128,"generated_token_ids":ids,"generated_token_ids_sha256":token_hash,"generated_text_sha256":"d".repeat(64)})).collect::<Vec<_>>());
            for (i, r) in b["per_run_results"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                r["prompt_tokens"] = json!(1);
                r["requested_output_tokens"] = json!(128);
                let before = GpuNativeTokenLoopSnapshot {
                    tokens_completed: 128 * (i as u64 + 1),
                    ..Default::default()
                };
                let after = GpuNativeTokenLoopSnapshot {
                    tokens_completed: before.tokens_completed + 128,
                    ..Default::default()
                };
                let recovery = GpuNativeRecoverySnapshot::default();
                let routed = RoutedExpertExecutionSnapshot::default();
                let sb = EngineStorageSnapshot {
                    nvme_read_operations: i as u64 + 1,
                    nvme_bytes_read: 2 * FULL as u64 * (i as u64 + 1),
                    ..Default::default()
                };
                let sa = EngineStorageSnapshot {
                    nvme_read_operations: sb.nvme_read_operations + 1,
                    nvme_bytes_read: sb.nvme_bytes_read + 2 * FULL as u64,
                    ..Default::default()
                };
                r["counters"] = json!({"token_loop_before":before,"token_loop_after":after,"token_loop_delta":crate::gpu_native_real_benchmark::token_loop_delta(before,after).unwrap(),"recovery_before":recovery,"recovery_after":recovery,"recovery_delta":recovery,"routed_execution_before":routed,"routed_execution_after":routed,"routed_execution_delta":routed,"engine_storage_before":sb,"engine_storage_after":sa,"engine_storage_delta":sa.checked_delta(sb).unwrap()});
            }
            let mut p = json!({"arm":"treatment","complete":true,"failure":null,"isolated_runtime":true,"benchmark":b,"warmup_ram_cache_state_sha256":"e".repeat(64),"warmup_results":[{"run_index":0,"generated_tokens":128,"generated_token_ids_sha256":token_hash,"generated_text_sha256":"d".repeat(64)}]});
            for measured in [false, true] {
                let prefix = if measured { "" } else { "warmup_" };
                let mk = if measured {
                    "mechanism"
                } else {
                    "warmup_mechanism"
                };
                let factor = if measured { 3 } else { 1 };
                let (s, pi, u) = super::super::tests::fixture(Arm::Treatment);
                let s: Snapshot =
                    serde_json::from_value(scale(serde_json::to_value(s).unwrap(), factor))
                        .unwrap();
                let pi: GpuNativeProductionPhysicalInstallSnapshot =
                    serde_json::from_value(scale(serde_json::to_value(pi).unwrap(), factor))
                        .unwrap();
                let mut u: UploadSnapshot =
                    serde_json::from_value(scale(serde_json::to_value(u).unwrap(), factor))
                        .unwrap();
                let mut h = Sha256::new();
                for r in records.iter().filter(|r| r.measured == measured) {
                    for id in &r.ids {
                        h.update(id.to_le_bytes());
                    }
                }
                u.ordered_nvme_ids_sha256 = format!("{:x}", h.finalize());
                u.source_upload_fd_proof = Some(crate::io_provider::SourceUploadFdProofSnapshot {
                    source_upload_fd_proof_requests: 2 * factor,
                    source_upload_fd_proof_hits: if measured { 2 * factor } else { 0 },
                    source_upload_fd_proof_misses: if measured { 0 } else { 2 * factor },
                    source_upload_fd_proof_failures: 0,
                });
                let mut source = super::super::tests::source_fixture();
                source.production_source_sets = factor;
                source.production_batch_eligible_sets = factor;
                source.production_batch_attempts = factor;
                source.production_batch_successes = factor;
                source.production_batch_experts = 2 * factor;
                source.production_batch_width_min = 2;
                source.production_batch_width_max = 2;
                source.production_batch_width_mean = 2.0;
                let mut w = ArmWorkEvidence {
                    token_loop: Default::default(),
                    recovery: Default::default(),
                    routed_execution: Default::default(),
                    engine_storage: Default::default(),
                    gpu_expert_io: Default::default(),
                    gpu_expert_memory_before: Default::default(),
                    gpu_expert_memory_after: Default::default(),
                    gpu_native_residency: Default::default(),
                };
                w.gpu_native_residency.ram_to_vram_installs = s.physical_install_completions;
                w.gpu_native_residency
                    .logical_admissions_for_physical_misses = u.metrics.logical_admissions;
                w.engine_storage.nvme_bytes_read = s.source_nvme_bytes;
                w.engine_storage.nvme_read_operations = factor;
                w.token_loop.tokens_completed = 128 * factor;
                p[mk] = serde_json::to_value(&s).unwrap();
                p[format!("{prefix}source")] =
                    serde_json::to_value(concurrency_common_snapshot(&s)).unwrap();
                p[format!("{prefix}upload")] = serde_json::to_value(u).unwrap();
                p[format!("{prefix}production")] = serde_json::to_value(source).unwrap();
                p[format!("{prefix}production_physical_install")] =
                    serde_json::to_value(pi).unwrap();
                p[format!("{prefix}work")] = serde_json::to_value(w).unwrap();
            }
            let before = ProcessMemory {
                rlimit_memlock_soft: u64::MAX,
                rlimit_memlock_hard: u64::MAX,
                vmlck_bytes: 0,
            };
            arms.push(ArmData {
                position,
                mode,
                before: before.clone(),
                after: before,
                production: p,
                records,
            });
        }
        Payload {
            schema: SCHEMA.into(),
            mode: MODE.into(),
            primary: PRIMARY.into(),
            cleanup_contract: CLEANUP.into(),
            frozen_workload: serde_json::to_value(frozen_workload("NVIDIA L4".into())).unwrap(),
            arms,
            complete: true,
            failure: None,
        }
    }
    fn bound(p: Payload, contamination: &str) -> (Vec<u8>, Vec<u8>) {
        let ph = sha(&serde_json::to_vec(&p).unwrap());
        let t = format!("{BEGIN}\n{contamination}\n{END}{ph}\n").into_bytes();
        let e = Envelope {
            schema: SCHEMA.into(),
            payload_json_utf8: serde_json::to_string(&p).unwrap(),
            payload_sha256: ph,
            transcript_sha256: sha(&t),
            transcript_bytes: t.len(),
        };
        (serde_json::to_vec(&e).unwrap(), t)
    }
    fn audit(p: Payload) -> Audit {
        let (r, t) = bound(p, "");
        audit_bytes(&r, &t)
    }
    #[test]
    fn hma1f_complete_offline_reconstruction_accepts_four_distinct_contexts() {
        let a = audit(fixture());
        assert!(a.authoritative, "{:?}", a.failure);
        assert_eq!(a.classification, "RESIDENCY_LOCK_NO_MATERIAL_EFFECT");
        assert_eq!(a.analysis.unwrap().request_pairs.len(), 6);
    }
    #[test]
    fn hma1f_wrong_order_rejects() {
        let mut p = fixture();
        p.arms.swap(0, 1);
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_no_fifth_arm() {
        let mut p = fixture();
        p.arms.push(p.arms[0].clone());
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_runtime_context_reuse_rejects() {
        let mut p = fixture();
        p.arms[1].production["benchmark"]["runtime_contract"]["legacy_execution_plan"]
            ["context_id"] = json!("1");
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_invalid_context_ids_reject() {
        for value in [
            json!("0"),
            json!("01"),
            json!("-1"),
            json!(1),
            Value::Null,
            json!("18446744073709551616"),
        ] {
            let mut p = fixture();
            p.arms[1].production["benchmark"]["runtime_contract"]["legacy_execution_plan"]
                ["context_id"] = value;
            assert!(!audit(p).authoritative);
        }
    }
    #[test]
    fn hma1f_second_runtime_difference_rejects() {
        let mut p = fixture();
        p.arms[1].production["benchmark"]["runtime_contract"]["extra"] = json!(true);
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_each_semantic_contract_drift_rejects() {
        for field in [
            "request",
            "production_configuration",
            "production_semantics",
            "model_identity",
            "model_load",
            "hardware",
            "provenance",
        ] {
            let mut p = fixture();
            p.arms[1].production["benchmark"][field]["extra"] = json!(true);
            assert!(!audit(p).authoritative, "{field}");
        }
    }
    #[test]
    fn hma1f_token_text_route_source_width_drift_rejects() {
        for mutation in 0..5 {
            let mut p = fixture();
            match mutation {
                0 => {
                    p.arms[1].production["benchmark"]["per_run_results"][0]
                        ["generated_text_sha256"] = json!("a".repeat(64))
                }
                1 => {
                    p.arms[1].production["mechanism"]["selected_route_ids_sha256"] =
                        json!("a".repeat(64))
                }
                2 => p.arms[1].records[1].ids.reverse(),
                3 => {
                    p.arms[1].records[1].ids.push(2);
                }
                _ => {
                    p.arms[1].production["benchmark"]["per_run_results"][0]["generated_token_ids"]
                        [0] = json!(18)
                }
            }
            assert!(!audit(p).authoritative);
        }
    }
    #[test]
    fn hma1f_request_pair_position_drift_rejects() {
        let mut p = fixture();
        p.arms[2].records[2].request_index = 0;
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_lock_and_unlock_evidence_corruption_rejects() {
        for field in [
            "lock_successes",
            "unlock_successes",
            "pointer_mod_4096",
            "lengths",
            "errno_histogram",
        ] {
            let p = fixture();
            let mut r = serde_json::to_value(&p.arms[1].records[0]).unwrap();
            r["locks"][field] = Value::Null;
            assert!(serde_json::from_value::<Record>(r).is_err());
        }
        let mut p = fixture();
        p.arms[1].records[0].locks.unlock_failures = 1;
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_baseline_syscall_evidence_rejects() {
        let mut p = fixture();
        p.arms[0].records[0].locks.lock_attempts = 1;
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_vmlck_leak_rejects() {
        let mut p = fixture();
        p.arms[1].after.vmlck_bytes = 4096;
        assert!(!audit(p).authoritative);
    }
    #[test]
    fn hma1f_recovery_and_source_failure_rejects() {
        for (object, key) in [
            ("work", "recovery"),
            ("work", "token_loop"),
            ("upload", "source_failures"),
            (
                "production",
                "production_sequential_fallback_batch_read_error",
            ),
        ] {
            let mut p = fixture();
            if object == "work" {
                p.arms[1].production[object][key][if key == "recovery" {
                    "full_token_replay_attempts"
                } else {
                    "replay_attempts"
                }] = json!(1);
            } else {
                p.arms[1].production[object][key] = json!(1);
            }
            assert!(!audit(p).authoritative);
        }
    }
    #[test]
    fn hma1f_fd_proof_failure_rejects() {
        let mut p = fixture();
        p.arms[2].production["upload"]["source_upload_fd_proof"]
            ["source_upload_fd_proof_failures"] = json!(1);
        assert!(!audit(p).authoritative);
    }
    fn contaminated(pattern: &str) {
        let (r, t) = bound(fixture(), pattern);
        let a = audit_bytes(&r, &t);
        assert!(!a.authoritative);
        assert_eq!(a.classification, "NON_AUTHORITATIVE");
        assert!(a.analysis.is_none());
    }
    #[test]
    fn hma1f_transient_retry_pattern_rejects() {
        contaminated(PATTERNS[0]);
    }
    #[test]
    fn hma1f_recovered_retry_pattern_rejects() {
        contaminated(PATTERNS[1]);
    }
    #[test]
    fn hma1f_fetch_retry_pattern_rejects() {
        contaminated(PATTERNS[2]);
    }
    #[test]
    fn hma1f_breaker_pattern_rejects() {
        contaminated(PATTERNS[3]);
    }
    #[test]
    fn hma1f_all_patterns_reject() {
        contaminated(&PATTERNS.join("\n"));
    }
    #[test]
    fn hma1f_transcript_hash_truncation_duplicate_markers_reject() {
        let (r, t) = bound(fixture(), "");
        for bad in [
            t[..t.len() - 20].to_vec(),
            [t.as_slice(), b"extra"].concat(),
            [BEGIN.as_bytes(), t.as_slice()].concat(),
        ] {
            assert!(!audit_bytes(&r, &bad).authoritative);
        }
        let mut e: Envelope = serde_json::from_slice(&r).unwrap();
        let duplicate = [BEGIN.as_bytes(), b"\n", t.as_slice()].concat();
        e.transcript_sha256 = sha(&duplicate);
        e.transcript_bytes = duplicate.len();
        assert!(!audit_bytes(&serde_json::to_vec(&e).unwrap(), &duplicate).authoritative);
    }
    #[test]
    fn hma1f_raw_payload_hash_mismatch_rejects() {
        let (r, t) = bound(fixture(), "");
        let mut e: Envelope = serde_json::from_slice(&r).unwrap();
        e.payload_json_utf8.push(' ');
        assert!(!audit_bytes(&serde_json::to_vec(&e).unwrap(), &t).authoritative);
    }
    #[test]
    fn hma1f_exact_payload_snapshot_preserves_json_spelling() {
        let (r, t) = bound(fixture(), "");
        let mut e: Envelope = serde_json::from_slice(&r).unwrap();
        let old_hash = e.payload_sha256.clone();
        e.payload_json_utf8 = format!(" \n{}\n", e.payload_json_utf8);
        e.payload_sha256 = sha(e.payload_json_utf8.as_bytes());
        let t = String::from_utf8(t)
            .unwrap()
            .replace(&old_hash, &e.payload_sha256)
            .into_bytes();
        e.transcript_sha256 = sha(&t);
        e.transcript_bytes = t.len();
        let a = audit_bytes(&serde_json::to_vec(&e).unwrap(), &t);
        assert!(a.authoritative, "{:?}", a.failure);
    }
    #[test]
    fn hma1f_cached_analysis_cannot_override_audit() {
        let (r, t) = bound(fixture(), "");
        let mut v: Value = serde_json::from_slice(&r).unwrap();
        v["analysis"] = json!({"authoritative":true});
        assert!(!audit_bytes(&serde_json::to_vec(&v).unwrap(), &t).authoritative);
    }
    fn analysis(delta: i128) -> Analysis {
        let baseline = Endpoints {
            critical: 10000,
            max_wall: 10000,
        };
        let c = Comparison::new(
            baseline,
            Endpoints {
                critical: (10000 + delta) as u64,
                max_wall: (10000 + delta) as u64,
            },
        );
        Analysis {
            pooled: c,
            blocks: (0..2)
                .map(|b| Stratum {
                    block: Some(b),
                    request: None,
                    width: None,
                    comparison: c,
                })
                .collect(),
            request_pairs: (0..6)
                .map(|r| Stratum {
                    block: Some(r / 3),
                    request: Some(r % 3),
                    width: None,
                    comparison: c,
                })
                .collect(),
            widths: (2..=8)
                .map(|w| Stratum {
                    block: None,
                    request: None,
                    width: Some(w),
                    comparison: c,
                })
                .collect(),
            width_one: c,
        }
    }
    #[test]
    fn hma1f_classifier_exact_integer_thresholds() {
        for (d, label) in [
            (-301, "RESIDENCY_LOCK_MITIGATES_STRAGGLER"),
            (-300, "RESIDENCY_LOCK_MITIGATES_STRAGGLER"),
            (-299, "RESIDENCY_LOCK_DIRECTIONAL_MITIGATION"),
            (-100, "RESIDENCY_LOCK_DIRECTIONAL_MITIGATION"),
            (-99, "RESIDENCY_LOCK_NO_MATERIAL_EFFECT"),
            (0, "RESIDENCY_LOCK_NO_MATERIAL_EFFECT"),
            (99, "RESIDENCY_LOCK_NO_MATERIAL_EFFECT"),
            (100, "AMBIGUOUS_LOCK_INTERACTION"),
            (299, "AMBIGUOUS_LOCK_INTERACTION"),
            (300, "RESIDENCY_LOCK_WORSENS"),
            (301, "RESIDENCY_LOCK_WORSENS"),
        ] {
            assert_eq!(classify(&analysis(d)), label, "{d}");
        }
    }
    #[test]
    fn hma1f_classifier_requires_both_endpoints_and_all_sign_strata() {
        let mut a = analysis(-400);
        a.blocks[1].comparison = analysis(1).pooled;
        assert_eq!(classify(&a), "AMBIGUOUS_LOCK_INTERACTION");
        let mut a = analysis(-400);
        for r in &mut a.request_pairs[..3] {
            r.comparison = analysis(1).pooled;
        }
        assert_eq!(classify(&a), "AMBIGUOUS_LOCK_INTERACTION");
        let mut a = analysis(-400);
        for r in &mut a.widths[..3] {
            r.comparison = analysis(1).pooled;
        }
        assert_eq!(classify(&a), "AMBIGUOUS_LOCK_INTERACTION");
        let mut a = analysis(-400);
        a.pooled.dlock_max = -50;
        assert_eq!(classify(&a), "AMBIGUOUS_LOCK_INTERACTION");
    }
    #[test]
    fn hma1f_width_one_never_enters_primary() {
        let mut p = fixture();
        let a = analyze(&p.arms).unwrap();
        for arm in &mut p.arms {
            arm.records.push(record(arm.mode, true, 0, 1, u64::MAX / 4));
        }
        let b = analyze(&p.arms).unwrap();
        assert_eq!(a.pooled, b.pooled);
        assert!(b.width_one.baseline.critical > 0);
        assert_eq!(b.request_pairs.len(), 6);
    }
    #[test]
    fn hma1f_block_b_is_locked_minus_baseline() {
        let mut p = fixture();
        for r in p.arms[2].records.iter_mut().filter(|r| r.measured) {
            let mut t = r.timing.take().unwrap();
            for read in &mut t.reads {
                read.wrapper_end_ns -= 10;
                read.wrapper_wall_ns -= 10;
            }
            r.timing = Some(Timing::from_reads(t.reads).unwrap());
        }
        let a = analyze(&p.arms).unwrap();
        assert_eq!(a.blocks[1].comparison.dlock_crit, -30);
        assert_eq!(a.request_pairs[3].comparison.dlock_crit, -10);
    }
    #[test]
    fn hma1f_launcher_preserves_process_arguments_on_both_sides_of_command() {
        use clap::Parser;
        for global_before in [false, true] {
            let settings = [
                "--rayon-threads=3",
                "--progress-timeout-secs",
                "29",
                "--log=warn",
            ];
            let mut raw = vec!["mer"];
            if global_before {
                raw.extend(settings);
            }
            raw.extend([
                MODE,
                "--config",
                FROZEN_CONFIG_PATH,
                "--report-out=raw.json",
                "--transcript-out",
                "run.txt",
            ]);
            if !global_before {
                raw.extend(settings);
            }
            let raw: Vec<std::ffi::OsString> = raw.into_iter().map(Into::into).collect();
            let args = worker_arguments(
                Path::new(FROZEN_CONFIG_PATH),
                Path::new("raw.payload.json"),
                &raw,
            )
            .unwrap();
            let cli = crate::Cli::try_parse_from(
                std::iter::once(std::ffi::OsString::from("mer")).chain(args),
            )
            .unwrap();
            assert_eq!(cli.rayon_threads, Some(3));
            assert_eq!(cli.progress_timeout_secs, Some(29));
            assert_eq!(cli.log, "info");
            assert!(
                matches!(cli.cmd, crate::Cmd::Hma1fWorkerInternal { config, report_out } if config == Path::new(FROZEN_CONFIG_PATH) && report_out == Path::new("raw.payload.json"))
            );
        }
    }
    #[test]
    fn hma1f_auditor_and_probe_dispatch_before_runtime_setup() {
        let s = include_str!("main.rs");
        let main = s.split("fn main() ->").nth(1).unwrap();
        for marker in [
            "mapped_lock::audit_command",
            "gpu_native_mapped_lock::probe_command",
            "mapped_lock::launch",
        ] {
            assert!(main.find(marker).unwrap() < main.find("init_logging(").unwrap());
        }
    }
}
