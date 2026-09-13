//! Qualification-only real-inference A/B of single-read source/upload fusion.
//! Reuses physical-install workload, isolated runtime, observation and teardown.
use super::*;
use crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot as Snapshot;

pub(crate) const SCHEMA: &str = "mer.gpu-native-source-to-upload-copy-elision-production.v2";
pub(crate) const MODE: &str = "qualify-gpu-native-source-to-upload-copy-elision-production";
const LOGICAL_EXPERT_BYTES: u64 = 2_654_208;
const SLOT_STRIDE_BYTES: u64 = 2_654_212;
const EPOCH_BYTES: u64 = 4;
use crate::gpu_native_source_upload::{Arm, Snapshot as UploadSnapshot, CAPACITY, FULL, PAYLOAD};

pub(crate) const fn qualification_arms() -> (Arm, Arm) {
    (Arm::Control, Arm::Treatment)
}

/// Additive HMA-1A diagnostic under v2; existing timing, workload and copy
/// counters retain their meanings. Counters span the same idle-boundary resets.
#[derive(Clone, Debug, Serialize)]
struct SourceUploadFdProofGate {
    diagnostic_version: &'static str,
    warmup_accounting_exact: bool,
    measured_accounting_exact: bool,
    first_use_misses_observed: bool,
    measured_hits_observed: bool,
    passed: bool,
}

fn source_upload_fd_proof_interval_exact(c: &UploadSnapshot, t: &UploadSnapshot) -> bool {
    let (Some(c), Some(p)) = (&c.source_upload_fd_proof, &t.source_upload_fd_proof) else {
        return false;
    };
    *c == crate::io_provider::SourceUploadFdProofSnapshot::default()
        && p.source_upload_fd_proof_requests > 0
        && p.source_upload_fd_proof_hits
            .checked_add(p.source_upload_fd_proof_misses)
            == Some(p.source_upload_fd_proof_requests)
        && p.source_upload_fd_proof_requests == t.metrics.direct_source_reads
        && p.source_upload_fd_proof_failures == 0
}

fn source_upload_fd_proof_gate(
    cw: &UploadSnapshot,
    tw: &UploadSnapshot,
    cm: &UploadSnapshot,
    tm: &UploadSnapshot,
) -> SourceUploadFdProofGate {
    let warmup_accounting_exact = source_upload_fd_proof_interval_exact(cw, tw);
    let measured_accounting_exact = source_upload_fd_proof_interval_exact(cm, tm);
    let first_use_misses_observed = tw
        .source_upload_fd_proof
        .as_ref()
        .is_some_and(|p| p.source_upload_fd_proof_misses > 0);
    // fd capacity depends on RLIMIT_NOFILE; caches are kept after warmup.
    // Full churn can give zero hits, and complete warmup coverage zero measured
    // misses. Neither is an accounting failure. First use must miss in warmup.
    let measured_hits_observed = tm
        .source_upload_fd_proof
        .as_ref()
        .is_some_and(|p| p.source_upload_fd_proof_hits > 0);
    SourceUploadFdProofGate {
        diagnostic_version: "hma1a.fd-proof.v1",
        warmup_accounting_exact,
        measured_accounting_exact,
        first_use_misses_observed,
        measured_hits_observed,
        passed: warmup_accounting_exact && measured_accounting_exact && first_use_misses_observed,
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct PairMechanismGate {
    source_and_route_streams_exact: bool,
    logical_materialization_and_generations_exact: bool,
    physical_install_reservation_victim_and_publication_exact: bool,
    source_scheduler_exact: bool,
    control_copy_accounting_exact: bool,
    treatment_fused_and_fallback_bytes_exact: bool,
    treatment_every_nvme_read_fused: bool,
    ring_and_submission_accounting_exact: bool,
    production_upload_ownership_exact: bool,
    failures_and_accounting_errors_zero: bool,
    production_install_reconciliation: bool,
    passed: bool,
}

fn install_exact(
    s: &Snapshot,
    p: &GpuNativeProductionPhysicalInstallSnapshot,
    u: &UploadSnapshot,
) -> bool {
    let n = s.physical_install_completions;
    n > 0
        && production_concurrency_exercised(p)
        && s.physical_install_attempts == n
        && s.physical_install_experts == n
        && s.reservation_attempts == n
        && s.reservation_successes == n
        && s.physical_stage_attempts == n
        && s.physical_stage_completions == n
        && s.direct_staging_writes
            .checked_add(u.metrics.fused_installs)
            == Some(n)
        && s.ordered_commit_attempts == n
        && s.ordered_commit_completions == n
        && s.mapping_publications == n
        && s.physical_install_sets == p.physical_install_sets
        && n == p.physical_install_experts
        && n == p.physical_install_attempts
        && n == p.physical_stage_completions
        && n == p.ordered_commit_completions
        && s.parallel_eligible_sets == p.parallel_eligible_sets
        && s.parallel_staging_sets == p.parallel_staging_sets
        && s.parallel_staging_experts == p.parallel_staging_experts
        && s.singleton_staging_sets == p.singleton_staging_sets
        && s.max_in_flight_physical_staging >= 2
        && s.parallel_staging_sets > 0
        && s.parallel_staging_sets == s.parallel_eligible_sets
        && s.parallel_staging_experts == s.parallel_eligible_experts
        && s.parallel_staging_experts
            .checked_add(s.singleton_staging_sets)
            == Some(n)
        && s.parallel_staging_sets
            .checked_add(s.singleton_staging_sets)
            == Some(s.physical_install_sets)
        && n.checked_mul(EPOCH_BYTES) == Some(s.physical_slot_epoch_write_bytes)
        && n.checked_mul(SLOT_STRIDE_BYTES) == Some(s.physical_slot_bytes_staged)
        && s.physical_bytes_staged == s.physical_slot_bytes_staged
        && s.physical_slot_zero_fill_bytes == 0
        && s.full_slot_vec_materializations == 0
}
fn upload_errors_zero(u: &UploadSnapshot) -> bool {
    let m = &u.metrics;
    m.source_failures == 0
        && m.source_fallback_reads == 0
        && m.alignment_failures == 0
        && m.mapped_direct_io_rejections == 0
        && m.remap_failures == 0
        && m.copy_failures == 0
        && m.accounting_errors == 0
        && m.leases_dropped_unconsumed == 0
        && m.non_shared_logical_fallbacks == 0
        && m.non_shared_logical_rejections == 0
        && u.active_leases == 0
        && u.pending_leases == 0
}
fn ring_exact(c: &UploadSnapshot, t: &UploadSnapshot) -> bool {
    let (c, ring_capacity, active, m) = (&c.metrics, t.ring_capacity, t.active_leases, &t.metrics);
    [
        c.acquisition_attempts,
        c.acquisition_waits,
        c.acquisition_wait_us,
        c.high_water,
        c.map_attempts,
        c.map_completions,
        c.map_wait_us,
        c.unmaps,
        c.remap_attempts,
        c.remap_completions,
        c.remap_wait_us,
        c.leases_consumed,
        c.leases_released,
        c.odirect_observations,
        c.direct_payload_bytes,
        c.copied_experts,
        c.copied_bytes,
        c.fused_install_sets,
        c.fallback_installs,
        c.fallback_payload_copy_bytes,
    ]
    .iter()
    .all(|v| *v == 0)
        && c.copy_submissions == 0
        && c.copy_command_buffers == 0
        && c.leases_created == 0
        && c.direct_source_reads == 0
        && c.direct_source_bytes == 0
        && c.fused_installs == 0
        && c.fused_gpu_copy_bytes == 0
        && ring_capacity == CAPACITY
        && active == 0
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
fn pair_mechanism_gate(
    c: &Snapshot,
    t: &Snapshot,
    cp: &GpuNativeProductionPhysicalInstallSnapshot,
    tp: &GpuNativeProductionPhysicalInstallSnapshot,
    cu: &UploadSnapshot,
    tu: &UploadSnapshot,
    cs: &ProductionDemandSourceSnapshot,
    ts: &ProductionDemandSourceSnapshot,
) -> PairMechanismGate {
    let (cm, tm) = (&cu.metrics, &tu.metrics);
    let total = t
        .physical_install_completions
        .checked_mul(LOGICAL_EXPERT_BYTES);
    let source_and_route_streams_exact = c.selected_route_ids_sha256 == t.selected_route_ids_sha256
        && c.physical_missing_ids_sha256 == t.physical_missing_ids_sha256
        && c.physical_missing_experts == t.physical_missing_experts
        && c.demand_source_request_ids_sha256 == t.demand_source_request_ids_sha256
        && c.demand_source_requests == t.demand_source_requests
        && c.source_ram_hits == t.source_ram_hits
        && c.source_ram_misses == t.source_ram_misses
        && c.source_nvme_reads == t.source_nvme_reads
        && c.source_nvme_bytes == t.source_nvme_bytes
        && cu.ordered_nvme_ids_sha256 == tu.ordered_nvme_ids_sha256
        && c.ram_cache_inserts == t.ram_cache_inserts
        && c.ram_cache_evictions == t.ram_cache_evictions
        && c.demand_ram_insert_ids_sha256 == t.demand_ram_insert_ids_sha256
        && c.demand_ram_eviction_ids_sha256 == t.demand_ram_eviction_ids_sha256;
    let logical_materialization_and_generations_exact = cm.logical_admissions
        == tm.logical_admissions
        && cm.logical_generation_observations == tm.logical_generation_observations
        && cu.logical_admission_ids_sha256 == tu.logical_admission_ids_sha256
        && cu.logical_generation_ids_sha256 == tu.logical_generation_ids_sha256
        && cm.logical_materialization_operations == tm.logical_materialization_operations
        && cm.logical_materialization_bytes == tm.logical_materialization_bytes
        && cm
            .logical_materialization_operations
            .checked_mul(LOGICAL_EXPERT_BYTES)
            == Some(cm.logical_materialization_bytes)
        && tm.logical_materialization_operations == tm.shared_payload_constructions
        && cm.shared_payload_constructions == 0;
    let physical_install_reservation_victim_and_publication_exact = c.physical_install_completions
        == t.physical_install_completions
        && c.physical_install_sets == t.physical_install_sets
        && c.reservation_identity_sha256 == t.reservation_identity_sha256
        && c.physical_victim_ids_sha256 == t.physical_victim_ids_sha256
        && c.physical_residency_identity_sha256 == t.physical_residency_identity_sha256
        && c.mapping_publications == t.mapping_publications
        && c.mapping_unpublications == t.mapping_unpublications
        && c.physical_slot_bytes_staged == t.physical_slot_bytes_staged
        && c.ordered_install_set_behavior_sha256 == t.ordered_install_set_behavior_sha256
        && c.install_set_width_min == t.install_set_width_min
        && c.install_set_width_max == t.install_set_width_max
        && c.install_set_width_mean == t.install_set_width_mean
        && c.parallel_eligible_sets == t.parallel_eligible_sets
        && c.parallel_eligible_experts == t.parallel_eligible_experts
        && c.parallel_staging_sets == t.parallel_staging_sets
        && c.parallel_staging_experts == t.parallel_staging_experts
        && c.singleton_staging_sets == t.singleton_staging_sets
        && c.rayon_num_threads >= 2
        && c.rayon_num_threads == t.rayon_num_threads
        && c.caller_was_already_rayon_worker == t.caller_was_already_rayon_worker;
    let source_scheduler_exact = match (serde_json::to_value(cs), serde_json::to_value(ts)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    let control_copy_accounting_exact = cu.arm == Arm::Control
        && cu.ring_capacity == 0
        && c.physical_install_completions
            .checked_mul(LOGICAL_EXPERT_BYTES)
            == Some(cm.physical_cpu_payload_copy_bytes)
        && cm.physical_cpu_payload_copy_bytes == c.physical_slot_payload_copy_bytes
        && c.direct_staging_writes == c.physical_install_completions
        && cm.total_payload_bytes_staged == cm.physical_cpu_payload_copy_bytes;
    let treatment_fused_and_fallback_bytes_exact = tu.arm == Arm::Treatment
        && tm.fused_installs.checked_add(tm.fallback_installs)
            == Some(t.physical_install_completions)
        && tm.fused_installs.checked_mul(LOGICAL_EXPERT_BYTES) == Some(tm.fused_gpu_copy_bytes)
        && tm.fallback_installs.checked_mul(LOGICAL_EXPERT_BYTES)
            == Some(tm.fallback_payload_copy_bytes)
        && tm
            .fused_gpu_copy_bytes
            .checked_add(tm.fallback_payload_copy_bytes)
            == total
        && tm.physical_cpu_payload_copy_bytes == tm.fallback_payload_copy_bytes
        && tm.physical_cpu_payload_copy_bytes == t.physical_slot_payload_copy_bytes
        && tm.fallback_installs == t.direct_staging_writes
        && Some(tm.total_payload_bytes_staged) == total;
    let treatment_every_nvme_read_fused = tm.direct_source_reads > 0
        && tm.direct_source_reads == t.source_nvme_reads
        && tm.direct_source_bytes == t.source_nvme_bytes
        && tm.direct_source_reads.checked_mul(FULL as u64) == Some(tm.direct_source_bytes)
        && tm.direct_source_reads.checked_mul(PAYLOAD as u64) == Some(tm.direct_payload_bytes)
        && tm.odirect_observations == tm.direct_source_reads;
    let ring_and_submission_accounting_exact = ring_exact(cu, tu);
    let production_upload_ownership_exact = !cu.production_owned && tu.production_owned;
    let failures_and_accounting_errors_zero = upload_errors_zero(cu)
        && upload_errors_zero(tu)
        && zero_fill_production::failures_zero(c)
        && zero_fill_production::failures_zero(t)
        && zero_fill_production::timing_accounting_exact(c)
        && zero_fill_production::timing_accounting_exact(t);
    let production_install_reconciliation = install_exact(c, cp, cu) && install_exact(t, tp, tu);
    let passed = source_and_route_streams_exact
        && logical_materialization_and_generations_exact
        && physical_install_reservation_victim_and_publication_exact
        && source_scheduler_exact
        && control_copy_accounting_exact
        && treatment_fused_and_fallback_bytes_exact
        && treatment_every_nvme_read_fused
        && ring_and_submission_accounting_exact
        && production_upload_ownership_exact
        && failures_and_accounting_errors_zero
        && production_install_reconciliation;
    PairMechanismGate {
        source_and_route_streams_exact,
        logical_materialization_and_generations_exact,
        physical_install_reservation_victim_and_publication_exact,
        source_scheduler_exact,
        control_copy_accounting_exact,
        treatment_fused_and_fallback_bytes_exact,
        treatment_every_nvme_read_fused,
        ring_and_submission_accounting_exact,
        production_upload_ownership_exact,
        failures_and_accounting_errors_zero,
        production_install_reconciliation,
        passed,
    }
}

#[derive(Clone, Debug, Serialize)]
struct UploadArmReport {
    #[serde(flatten)]
    run: ConcurrencyArmReport,
    warmup_upload: Option<UploadSnapshot>,
    upload: Option<UploadSnapshot>,
}
impl std::ops::Deref for UploadArmReport {
    type Target = ConcurrencyArmReport;
    fn deref(&self) -> &Self::Target {
        &self.run
    }
}

#[derive(Clone, Debug, Serialize)]
struct Gates {
    source_upload_fd_proof: SourceUploadFdProofGate,
    behavioral: BehavioralGate,
    work_equivalence: WorkEquivalenceGate,
    warmup_mechanism: PairMechanismGate,
    measured_mechanism: PairMechanismGate,
    warmup_and_measured_work_exact: bool,
    token_ids_text_and_routes_exact: bool,
    stale_generation_and_install_errors_zero: bool,
    physical_installs_reconcile_with_residency: bool,
    source_bytes_and_logical_admissions_reconcile: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Performance {
    #[serde(flatten)]
    physical_install: ConcurrencyPerformanceComparison,
    mean_time_to_first_token_seconds: MetricComparison,
    fused_install_coverage_percent: f64,
    fused_payload_coverage_percent: f64,
    control_cpu_physical_payload_copy_bytes: u64,
    control_cpu_physical_payload_copy_us: u64,
    treatment_mechanism: crate::gpu_native_source_upload::Metrics,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    #[serde(skip)]
    order_straggler:
        Option<Vec<crate::gpu_native_source_order_straggler_production::StoreSnapshot>>,
    schema: &'static str,
    mode: &'static str,
    control: Option<UploadArmReport>,
    treatment: Option<UploadArmReport>,
    control_path: &'static str,
    treatment_path: &'static str,
    both_arms_same_reservation_and_commit: bool,
    source_scheduler_changed: bool,
    staging_byte_count_changed: bool,
    queue_ordering_changed: bool,
    payload_offset_bytes: u64,
    logical_expert_bytes: u64,
    slot_stride_bytes: u64,
    tail_padding_bytes: u64,
    no_zero_requires_complete_coverage_before_first_write: bool,
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    timing_definitions: ConcurrencyTimingDefinitions,
    logical_materialization_timing_definition: &'static str,
    reconciliation: Option<ProductionReconciliation>,
    gates: Option<Gates>,
    performance: Option<Performance>,
    benchmark_complete: bool,
    qualification_pass: bool,
    performance_result: &'static str,
    failure: Option<BenchmarkFailure>,
}

fn work_pair_exact(c: &ArmWorkEvidence, t: &ArmWorkEvidence) -> bool {
    c.gpu_native_residency
        .logical_admissions_for_physical_misses
        == t.gpu_native_residency
            .logical_admissions_for_physical_misses
        && c.gpu_native_residency.ram_to_vram_installs
            == t.gpu_native_residency.ram_to_vram_installs
        && c.gpu_native_residency.physical_evictions == t.gpu_native_residency.physical_evictions
        && c.gpu_native_residency.physical_reinstalls == t.gpu_native_residency.physical_reinstalls
        && c.token_loop.residency_miss_attempts == t.token_loop.residency_miss_attempts
        && c.token_loop.residency_services == t.token_loop.residency_services
        && recovery_semantics_equal(c.recovery, t.recovery)
        && c.routed_execution.selected_routed_experts == t.routed_execution.selected_routed_experts
        && c.engine_storage.ram_hits == t.engine_storage.ram_hits
        && c.engine_storage.ram_misses == t.engine_storage.ram_misses
        && c.engine_storage.nvme_read_operations == t.engine_storage.nvme_read_operations
        && c.engine_storage.nvme_bytes_read == t.engine_storage.nvme_bytes_read
        && c.token_loop.queue_submissions == t.token_loop.queue_submissions
        && c.token_loop.boundary_maps == t.token_loop.boundary_maps
        && c.token_loop.boundary_readbacks == t.token_loop.boundary_readbacks
}

pub(super) fn work_errors_zero(w: &ArmWorkEvidence) -> bool {
    w.gpu_native_residency.stale_generation_rejections == 0
        && w.token_loop.fatal_failures == 0
        && w.token_loop.no_progress_failures == 0
        && w.token_loop.replay_attempts == 0
        && w.recovery.full_token_replay_attempts == 0
}

// Performance has no argument and cannot affect correctness/mechanism PASS.
fn qualification_pass(reconciliation_pass: bool, gates: &Gates) -> bool {
    reconciliation_pass && gates.passed
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    run_command_inner(args, false).await
}

pub(crate) async fn run_order_straggler_command(
    args: CommandArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.expected_adapter_name != "NVIDIA L4" {
        return Err("HMA-1E requires the frozen NVIDIA L4 adapter".into());
    }
    crate::gpu_native_source_order_straggler_production::begin_transcript();
    run_command_inner(args, true).await
}

fn emit_run_report(
    report: &Report,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(stores) = &report.order_straggler {
        let envelope = crate::gpu_native_source_order_straggler_production::Envelope::new(
            serde_json::to_value(report)?,
            stores.clone(),
        );
        envelope.emit(output)
    } else {
        emit_report(report, output)
    }
}

async fn run_command_inner(
    args: CommandArgs,
    order_straggler: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let prepared = prepare(&args)?;
    let mut timing_definitions = concurrency_timing_definitions();
    timing_definitions.common.physical_install_total_us = "both arms: sum of each post-reservation physical stage plus ordered commit service; excludes reservation time";
    timing_definitions.common.physical_slot_prepare_us = "control and RAM-hit fallback: validated no-zero CPU payload staging; fused treatment: validation, epoch write and GPU copy encoding; copy-set submission separately timed";
    timing_definitions.common.mapping_publication_us =
        "both arms: ordered logical mapping Queue::write_buffer time after staging; treatment additionally submits the fused copy set before publication";
    let mut report = Report {
        order_straggler: order_straggler.then(Vec::new),
        schema: SCHEMA,
        mode: MODE,
        control: None,
        treatment: None,
        control_path: "ordinary production source and concurrent no-zero Queue::write_buffer_with",
        treatment_path:
            "ordinary production-owned single-read NVMe to bounded mapped upload; shared logical/RAM payload; GPU-copy fused misses; ordinary RAM-hit physical fallback",
        both_arms_same_reservation_and_commit: true,
        source_scheduler_changed: false,
        staging_byte_count_changed: false,
        queue_ordering_changed: true,
        payload_offset_bytes: EPOCH_BYTES,
        logical_expert_bytes: LOGICAL_EXPERT_BYTES,
        slot_stride_bytes: SLOT_STRIDE_BYTES,
        tail_padding_bytes: 0,
        no_zero_requires_complete_coverage_before_first_write: true,
        frozen_workload: frozen_workload(args.expected_adapter_name.clone()),
        provenance: prepared.provenance.clone(),
        timing_definitions,
        logical_materialization_timing_definition: "logical cache transaction wall is logical_demand_admission_us; treatment fresh shared-host materialization occurs inside source acquisition and is separately counted/timed as logical_materialization_us; RAM-hit readmission materialization occurs inside the logical transaction",
        reconciliation: None,
        gates: None,
        performance: None,
        benchmark_complete: false,
        qualification_pass: false,
        performance_result: "not_measured",
        failure: None,
    };
    let (control_arm, treatment_arm) = qualification_arms();
    let execution_order = if order_straggler {
        crate::gpu_native_source_order_straggler_production::qualification_arms()
    } else {
        [control_arm, treatment_arm]
    };
    for arm in execution_order {
        let result = run_physical_install_arm_inner(
            &prepared,
            &args,
            PhysicalInstallQualificationRun::SourceToUpload(arm),
            order_straggler.then_some(crate::gpu_native_source_order_straggler_production::MODE),
        )
        .await;
        let run = match result {
            Ok(run) => run,
            Err(failure) => {
                report.failure = Some(failure.clone());
                emit_run_report(&report, &args.report_out)?;
                return Err(failure.to_string().into());
            }
        };
        if let (Some(stores), Some(observation)) =
            (&mut report.order_straggler, run.source_order_observer)
        {
            stores.push(observation);
        }
        let failure = run.common.failure.clone();
        let arm_report = UploadArmReport {
            run: ConcurrencyArmReport {
                common: run.common,
                warmup_mechanism: run.warmup_concurrency,
                mechanism: run.concurrency,
            },
            warmup_upload: run.warmup_upload,
            upload: run.upload,
        };
        let snapshots_present = arm_report.upload.is_some()
            && arm_report.warmup_upload.is_some()
            && arm_report.common.complete
            && arm_report.mechanism.is_some()
            && arm_report.warmup_mechanism.is_some()
            && arm_report.common.source.is_some()
            && arm_report.common.warmup_source.is_some()
            && arm_report.common.work.is_some()
            && arm_report.common.warmup_work.is_some()
            && arm_report.common.production.is_some()
            && arm_report.common.warmup_production.is_some()
            && arm_report.common.production_physical_install.is_some()
            && arm_report
                .common
                .warmup_production_physical_install
                .is_some();
        if arm == control_arm {
            report.control = Some(arm_report);
        } else {
            report.treatment = Some(arm_report);
        }
        if let Some(failure) = failure.or_else(|| {
            (!snapshots_present).then(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-source-upload-arm-evidence",
                    "complete warmup and measured evidence is required",
                )
            })
        }) {
            report.failure = Some(failure.clone());
            emit_run_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    }
    let c = report.control.as_ref().expect("stored control");
    let t = report.treatment.as_ref().expect("stored treatment");
    let reconciliation =
        production_reconciliation(reconcile(&c.common, &t.common), &c.common, &t.common);
    let (behavioral, work_equivalence) = common_gates(&reconciliation.common);
    let warmup_mechanism = pair_mechanism_gate(
        c.warmup_mechanism.as_ref().unwrap(),
        t.warmup_mechanism.as_ref().unwrap(),
        c.common
            .warmup_production_physical_install
            .as_ref()
            .unwrap(),
        t.common
            .warmup_production_physical_install
            .as_ref()
            .unwrap(),
        c.warmup_upload.as_ref().unwrap(),
        t.warmup_upload.as_ref().unwrap(),
        c.common.warmup_production.as_ref().unwrap(),
        t.common.warmup_production.as_ref().unwrap(),
    );
    let measured_mechanism = pair_mechanism_gate(
        c.mechanism.as_ref().unwrap(),
        t.mechanism.as_ref().unwrap(),
        c.common.production_physical_install.as_ref().unwrap(),
        t.common.production_physical_install.as_ref().unwrap(),
        c.upload.as_ref().unwrap(),
        t.upload.as_ref().unwrap(),
        c.common.production.as_ref().unwrap(),
        t.common.production.as_ref().unwrap(),
    );
    let cw = c.common.warmup_work.as_ref().unwrap();
    let tw = t.common.warmup_work.as_ref().unwrap();
    let cm = c.common.work.as_ref().unwrap();
    let tm = t.common.work.as_ref().unwrap();
    let warmup_and_measured_work_exact = work_pair_exact(cw, tw) && work_pair_exact(cm, tm);
    let stale_generation_and_install_errors_zero =
        [cw, tw, cm, tm].into_iter().all(work_errors_zero);
    let physical_installs_reconcile_with_residency = [
        (cw, c.warmup_mechanism.as_ref().unwrap()),
        (tw, t.warmup_mechanism.as_ref().unwrap()),
        (cm, c.mechanism.as_ref().unwrap()),
        (tm, t.mechanism.as_ref().unwrap()),
    ]
    .into_iter()
    .all(|(work, mechanism)| {
        work.gpu_native_residency.ram_to_vram_installs == mechanism.physical_install_completions
    });
    let source_bytes_and_logical_admissions_reconcile = [
        (cw,c.warmup_mechanism.as_ref().unwrap(),c.warmup_upload.as_ref().unwrap(),c.common.warmup_production.as_ref().unwrap()),
        (tw,t.warmup_mechanism.as_ref().unwrap(),t.warmup_upload.as_ref().unwrap(),t.common.warmup_production.as_ref().unwrap()),
        (cm,c.mechanism.as_ref().unwrap(),c.upload.as_ref().unwrap(),c.common.production.as_ref().unwrap()),
        (tm,t.mechanism.as_ref().unwrap(),t.upload.as_ref().unwrap(),t.common.production.as_ref().unwrap()),
    ].into_iter().all(|(work,source,upload,production)| {
        work.gpu_native_residency.logical_admissions_for_physical_misses == upload.metrics.logical_admissions
            && work.engine_storage.nvme_bytes_read == source.source_nvme_bytes
            // The inherited I/O histogram records one observation per batch,
            // not one per expert; reconcile its observation count explicitly.
            && source.source_nvme_reads.checked_sub(production.production_batch_experts)
                .and_then(|v| v.checked_add(production.production_batch_successes)) == Some(work.engine_storage.nvme_read_operations)
    });
    let token_ids_text_and_routes_exact = c.common.warmup_results.len() == FROZEN_WARMUP_RUNS
        && t.common.warmup_results.len() == FROZEN_WARMUP_RUNS
        && generated_results(&c.common).len() == FROZEN_MEASURED_RUNS
        && generated_results(&t.common).len() == FROZEN_MEASURED_RUNS
        && c.common
            .warmup_results
            .iter()
            .zip(&t.common.warmup_results)
            .all(|(a, b)| {
                a.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && b.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && a.generated_text_sha256 == b.generated_text_sha256
                    && a.generated_token_ids_sha256 == b.generated_token_ids_sha256
            })
        && generated_results(&c.common)
            .iter()
            .zip(generated_results(&t.common))
            .all(|(a, b)| {
                a.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && b.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && a.generated_text_sha256 == b.generated_text_sha256
                    && a.generated_token_ids_sha256 == b.generated_token_ids_sha256
            })
        && warmup_mechanism.source_and_route_streams_exact
        && measured_mechanism.source_and_route_streams_exact;
    let source_upload_fd_proof = source_upload_fd_proof_gate(
        c.warmup_upload.as_ref().unwrap(),
        t.warmup_upload.as_ref().unwrap(),
        c.upload.as_ref().unwrap(),
        t.upload.as_ref().unwrap(),
    );
    let passed = source_upload_fd_proof.passed
        && behavioral.passed
        && work_equivalence.passed
        && warmup_mechanism.passed
        && measured_mechanism.passed
        && warmup_and_measured_work_exact
        && token_ids_text_and_routes_exact
        && stale_generation_and_install_errors_zero
        && physical_installs_reconcile_with_residency
        && source_bytes_and_logical_admissions_reconcile;
    let gates = Gates {
        source_upload_fd_proof,
        behavioral,
        work_equivalence,
        warmup_mechanism,
        measured_mechanism,
        warmup_and_measured_work_exact,
        token_ids_text_and_routes_exact,
        stale_generation_and_install_errors_zero,
        physical_installs_reconcile_with_residency,
        source_bytes_and_logical_admissions_reconcile,
        passed,
    };
    let performance = concurrency_performance(c, t).map(|physical_install| Performance {
        fused_install_coverage_percent: t.upload.as_ref().unwrap().metrics.fused_installs as f64
            / t.mechanism.as_ref().unwrap().physical_install_completions as f64
            * 100.0,
        fused_payload_coverage_percent: t.upload.as_ref().unwrap().metrics.fused_gpu_copy_bytes
            as f64
            / (t.mechanism.as_ref().unwrap().physical_install_completions * LOGICAL_EXPERT_BYTES)
                as f64
            * 100.0,
        control_cpu_physical_payload_copy_bytes: c
            .upload
            .as_ref()
            .unwrap()
            .metrics
            .physical_cpu_payload_copy_bytes,
        control_cpu_physical_payload_copy_us: c
            .mechanism
            .as_ref()
            .unwrap()
            .physical_slot_payload_copy_us,
        treatment_mechanism: t.upload.as_ref().unwrap().metrics.clone(),
        physical_install,
        mean_time_to_first_token_seconds: comparison(
            generated_results(&c.common)
                .iter()
                .map(|r| r.timing.time_to_first_token_seconds)
                .sum::<f64>()
                / FROZEN_MEASURED_RUNS as f64,
            generated_results(&t.common)
                .iter()
                .map(|r| r.timing.time_to_first_token_seconds)
                .sum::<f64>()
                / FROZEN_MEASURED_RUNS as f64,
        ),
    });
    report.qualification_pass = qualification_pass(reconciliation.all_invariants_pass, &gates);
    report.reconciliation = Some(reconciliation);
    report.gates = Some(gates);
    match performance {
        Ok(performance) => {
            report.benchmark_complete = true;
            report.performance_result = performance.physical_install.common.performance_result;
            report.performance = Some(performance);
        }
        Err(failure) => {
            report.qualification_pass = false;
            report.failure = Some(failure.clone());
            emit_run_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    }
    report.qualification_pass = completed_pass(
        report.benchmark_complete,
        report.failure.is_some(),
        report.qualification_pass,
    );
    emit_run_report(&report, &args.report_out)?;
    if report.qualification_pass {
        Ok(())
    } else {
        Err(
            "source-to-upload production qualification gates did not all pass; see emitted report"
                .into(),
        )
    }
}

fn completed_pass(complete: bool, has_failure: bool, gates_pass: bool) -> bool {
    complete && !has_failure && gates_pass
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm as LegacyArm;
    fn physical_fixture() -> (Snapshot, GpuNativeProductionPhysicalInstallSnapshot) {
        let mut s = crate::engine::empty_physical_zero_fill_test_snapshot(
            LegacyArm::ProductionNoZeroFillTreatment,
        );
        s.physical_install_attempts = 3;
        s.physical_install_completions = 3;
        s.physical_install_experts = 3;
        s.direct_staging_writes = 3;
        s.reservation_attempts = 3;
        s.reservation_successes = 3;
        s.physical_stage_attempts = 3;
        s.physical_stage_completions = 3;
        s.ordered_commit_attempts = 3;
        s.ordered_commit_completions = 3;
        s.mapping_publications = 3;
        s.physical_slot_bytes_staged = 3 * SLOT_STRIDE_BYTES;
        s.physical_bytes_staged = s.physical_slot_bytes_staged;
        s.physical_slot_zero_fill_bytes = 0;
        s.physical_slot_epoch_write_bytes = 3 * EPOCH_BYTES;
        s.physical_slot_payload_copy_bytes = 3 * LOGICAL_EXPERT_BYTES;
        s.physical_install_sets = 1;
        s.install_set_width_min = 3;
        s.install_set_width_max = 3;
        s.install_set_width_mean = 3.0;
        s.parallel_eligible_sets = 1;
        s.parallel_eligible_experts = 3;
        s.parallel_staging_sets = 1;
        s.parallel_staging_experts = 3;
        s.max_in_flight_physical_staging = 3;
        s.rayon_num_threads = 4;
        s.physical_slot_prepare_us = 10;
        s.physical_queue_staging_us = 5;
        s.sum_individual_physical_stage_us = 30;
        s.physical_ordered_commit_us = 9;
        s.mapping_publication_us = 3;
        s.physical_install_total_us = 39;
        let p = GpuNativeProductionPhysicalInstallSnapshot {
            physical_install_sets: 1,
            physical_install_experts: 3,
            parallel_eligible_sets: 1,
            parallel_staging_sets: 1,
            parallel_staging_experts: 3,
            reservation_attempts: 3,
            reservation_successes: 3,
            physical_install_attempts: 3,
            physical_stage_attempts: 3,
            physical_stage_completions: 3,
            direct_staging_successes: 3,
            ordered_commit_attempts: 3,
            ordered_commit_completions: 3,
            max_in_flight_physical_staging: 3,
            ..GpuNativeProductionPhysicalInstallSnapshot::default()
        };
        (s, p)
    }

    fn source_fixture() -> ProductionDemandSourceSnapshot {
        ProductionDemandSourceSnapshot {
            ordinary_production_path_exercised: true,
            production_source_sets: 0,
            production_batch_eligible_sets: 0,
            production_batch_attempts: 0,
            production_batch_successes: 0,
            production_batch_experts: 0,
            production_sequential_fallback_mixed_ram: 0,
            production_sequential_fallback_single_item: 0,
            production_sequential_fallback_singleflight_contention: 0,
            production_sequential_fallback_reservation: 0,
            production_sequential_fallback_pool: 0,
            production_sequential_fallback_batch_read_error: 0,
            production_batch_width_min: 0,
            production_batch_width_max: 0,
            production_batch_width_mean: 0.0,
            production_singleflight_ids_claimed: 0,
            production_singleflight_claim_rollbacks: 0,
            production_singleflight_followers_observed: 0,
            production_cache_slots_reserved: 0,
            production_cache_reservations_consumed: 0,
            production_cache_reservations_released: 0,
            production_cache_reservation_leaks: 0,
            production_batch_commit_violations: 0,
            stale_singleflight_entries: 0,
        }
    }

    fn fixture(
        arm: Arm,
    ) -> (
        Snapshot,
        GpuNativeProductionPhysicalInstallSnapshot,
        UploadSnapshot,
    ) {
        let (mut s, p) = physical_fixture();
        s.source_nvme_reads = 2;
        s.source_nvme_bytes = 2 * FULL as u64;
        s.source_ram_misses = 2;
        s.source_ram_hits = 1;
        let mut u = if arm == Arm::Treatment {
            crate::gpu_native_source_upload::State::cpu_test_production_state().snapshot()
        } else {
            crate::gpu_native_source_upload::State::cpu_test_state(arm).snapshot()
        };
        let m = &mut u.metrics;
        m.logical_admissions = 2;
        m.logical_generation_observations = 3;
        m.logical_materialization_operations = 2;
        m.logical_materialization_bytes = 2 * PAYLOAD as u64;
        m.total_payload_bytes_staged = 3 * PAYLOAD as u64;
        if arm == Arm::Control {
            m.physical_cpu_payload_copy_bytes = 3 * PAYLOAD as u64;
        } else {
            u.ring_capacity = 16;
            s.direct_staging_writes = 1;
            s.physical_slot_payload_copy_bytes = PAYLOAD as u64;
            m.physical_cpu_payload_copy_bytes = PAYLOAD as u64;
            m.fallback_installs = 1;
            m.fallback_payload_copy_bytes = PAYLOAD as u64;
            m.fused_installs = 2;
            m.fused_gpu_copy_bytes = 2 * PAYLOAD as u64;
            m.shared_payload_constructions = 2;
            m.acquisition_attempts = 2;
            m.high_water = 2;
            m.map_attempts = 2;
            m.map_completions = 2;
            m.unmaps = 2;
            m.leases_created = 2;
            m.leases_consumed = 2;
            m.leases_released = 2;
            m.odirect_observations = 2;
            m.direct_source_reads = 2;
            m.direct_source_bytes = 2 * FULL as u64;
            m.direct_payload_bytes = 2 * PAYLOAD as u64;
            m.fused_install_sets = 1;
            m.copy_command_buffers = 1;
            m.copy_submissions = 1;
            m.copied_experts = 2;
            m.copied_bytes = 2 * PAYLOAD as u64;
        }
        (s, p, u)
    }
    #[test]
    fn source_upload_fd_proof_gate_accepts_churn_and_warm_cache_intervals() {
        let (_, _, mut c) = fixture(Arm::Control);
        let (_, _, mut t) = fixture(Arm::Treatment);
        use crate::io_provider::SourceUploadFdProofSnapshot as Proof;
        c.source_upload_fd_proof = Some(Proof::default());
        t.source_upload_fd_proof = Some(Proof {
            source_upload_fd_proof_requests: 2,
            source_upload_fd_proof_misses: 2,
            ..Proof::default()
        });
        let churn = source_upload_fd_proof_gate(&c, &t, &c, &t);
        assert!(churn.passed);
        assert!(!churn.measured_hits_observed);
        let mut measured = t.clone();
        measured.source_upload_fd_proof = Some(Proof {
            source_upload_fd_proof_requests: 2,
            source_upload_fd_proof_hits: 2,
            ..Proof::default()
        });
        let warm = source_upload_fd_proof_gate(&c, &t, &c, &measured);
        assert!(warm.passed);
        assert!(warm.measured_hits_observed);
        assert!(!source_upload_fd_proof_gate(&c, &measured, &c, &measured).passed);
        measured
            .source_upload_fd_proof
            .as_mut()
            .unwrap()
            .source_upload_fd_proof_hits = 1;
        measured
            .source_upload_fd_proof
            .as_mut()
            .unwrap()
            .source_upload_fd_proof_misses = 1;
        assert!(source_upload_fd_proof_gate(&c, &t, &c, &measured).passed);
    }

    #[test]
    fn source_upload_fd_proof_gate_rejects_missing_corrupt_failed_and_control_activity() {
        use crate::io_provider::SourceUploadFdProofSnapshot as Proof;
        let (_, _, mut c) = fixture(Arm::Control);
        let (_, _, mut t) = fixture(Arm::Treatment);
        c.source_upload_fd_proof = Some(Proof::default());
        let valid = Proof {
            source_upload_fd_proof_requests: 2,
            source_upload_fd_proof_hits: 1,
            source_upload_fd_proof_misses: 1,
            source_upload_fd_proof_failures: 0,
        };
        t.source_upload_fd_proof = Some(valid.clone());
        for mutation in [
            |p: &mut Proof| p.source_upload_fd_proof_requests += 1,
            |p: &mut Proof| p.source_upload_fd_proof_hits += 1,
            |p: &mut Proof| p.source_upload_fd_proof_misses += 1,
            |p: &mut Proof| p.source_upload_fd_proof_failures = 1,
            |p: &mut Proof| {
                p.source_upload_fd_proof_hits = u64::MAX;
                p.source_upload_fd_proof_misses = 3;
            },
            |p: &mut Proof| *p = Proof::default(),
        ] {
            let mut bad = t.clone();
            mutation(bad.source_upload_fd_proof.as_mut().unwrap());
            assert!(!source_upload_fd_proof_gate(&c, &t, &c, &bad).passed);
            assert!(!source_upload_fd_proof_gate(&c, &bad, &c, &t).passed);
        }
        for mutation in [
            |p: &mut Proof| p.source_upload_fd_proof_requests = 1,
            |p: &mut Proof| p.source_upload_fd_proof_hits = 1,
            |p: &mut Proof| p.source_upload_fd_proof_misses = 1,
            |p: &mut Proof| p.source_upload_fd_proof_failures = 1,
        ] {
            let mut bad = c.clone();
            mutation(bad.source_upload_fd_proof.as_mut().unwrap());
            assert!(!source_upload_fd_proof_gate(&c, &t, &bad, &t).passed);
            assert!(!source_upload_fd_proof_gate(&bad, &t, &c, &t).passed);
        }
        for arm in [Arm::Control, Arm::Treatment] {
            let (_, _, missing) = fixture(arm);
            assert!(!source_upload_fd_proof_gate(&missing, &t, &c, &t).passed);
            assert!(!source_upload_fd_proof_gate(&c, &missing, &c, &t).passed);
            assert!(!source_upload_fd_proof_gate(&c, &t, &missing, &t).passed);
            assert!(!source_upload_fd_proof_gate(&c, &t, &c, &missing).passed);
        }
        let json = serde_json::to_value(&t).unwrap();
        assert_eq!(
            json["source_upload_fd_proof"]["source_upload_fd_proof_requests"],
            2
        );
    }

    #[test]
    fn source_upload_fd_proof_engine_samples_and_resets_at_existing_idle_boundaries() {
        let source = include_str!("engine.rs");
        let reset = source
            .split("fn reset_gpu_native_demand_source_qualification(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn")
            .next()
            .unwrap();
        assert!(
            reset.find("upload.reset()?").unwrap()
                < reset
                    .find("reset_source_upload_fd_proof_telemetry()")
                    .unwrap()
        );
        assert!(reset.contains("current.active_demand_set.load"));
        let enable = source
            .split("fn enable_gpu_native_source_upload_qualification(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn")
            .next()
            .unwrap();
        assert!(enable.contains("self.core.in_flight.is_empty()"));
        assert!(enable.contains("reset_source_upload_fd_proof_telemetry()"));
        let snapshot = source
            .split("fn gpu_native_source_upload_snapshot(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn")
            .next()
            .unwrap();
        assert!(snapshot.contains("self.core.storage.source_upload_fd_proof_snapshot()"));
        assert!(!snapshot.contains("reset_source_upload_fd_proof_telemetry()"));
    }

    #[test]
    fn source_upload_pair_accepts_exact_hybrid_copy_accounting_and_extra_submit() {
        let (c, cp, cu) = fixture(Arm::Control);
        let (t, tp, tu) = fixture(Arm::Treatment);
        let source = source_fixture();
        let gate = pair_mechanism_gate(&c, &t, &cp, &tp, &cu, &tu, &source, &source);
        assert!(gate.passed, "{gate:?}");
        assert_ne!(cu.metrics.copy_submissions, tu.metrics.copy_submissions);
        assert_ne!(tu.metrics.physical_cpu_payload_copy_bytes, 0);
    }
    #[test]
    fn source_upload_pair_rejects_source_identity_logical_and_fallback_corruption() {
        let (c, cp, cu) = fixture(Arm::Control);
        let (t, tp, tu) = fixture(Arm::Treatment);
        let source = source_fixture();
        let mutations: &[fn(&mut UploadSnapshot)] = &[
            |s| s.metrics.fused_gpu_copy_bytes += 1,
            |s| s.metrics.fallback_payload_copy_bytes = 0,
            |s| s.metrics.physical_cpu_payload_copy_bytes = 0,
            |s| s.metrics.total_payload_bytes_staged += 1,
            |s| s.metrics.fused_installs += 1,
            |s| s.metrics.fallback_installs += 1,
            |s| s.metrics.direct_source_reads += 1,
            |s| s.metrics.direct_source_bytes += 1,
            |s| s.metrics.source_fallback_reads = 1,
            |s| s.metrics.odirect_observations -= 1,
            |s| s.metrics.shared_payload_constructions -= 1,
            |s| s.metrics.logical_materialization_operations -= 1,
            |s| s.metrics.logical_materialization_bytes -= 1,
            |s| s.metrics.logical_admissions += 1,
            |s| s.metrics.logical_generation_observations += 1,
            |s| s.metrics.non_shared_logical_fallbacks = 1,
            |s| s.ordered_nvme_ids_sha256.push('x'),
            |s| s.logical_admission_ids_sha256.push('x'),
            |s| s.logical_generation_ids_sha256.push('x'),
            |s| s.metrics.copy_submissions += 1,
            |s| s.metrics.copy_command_buffers = 0,
            |s| s.metrics.fused_install_sets += 1,
            |s| s.metrics.copied_experts += 1,
            |s| s.metrics.copied_bytes += 1,
            |s| s.metrics.leases_created += 1,
            |s| s.metrics.leases_consumed -= 1,
            |s| s.metrics.leases_released -= 1,
            |s| s.metrics.leases_dropped_unconsumed = 1,
            |s| s.metrics.map_attempts += 1,
            |s| s.metrics.map_completions -= 1,
            |s| s.metrics.unmaps -= 1,
            |s| s.metrics.remap_attempts += 1,
            |s| s.metrics.remap_completions = 3,
            |s| s.metrics.high_water = 17,
            |s| s.metrics.alignment_failures = 1,
            |s| s.metrics.mapped_direct_io_rejections = 1,
            |s| s.metrics.remap_failures = 1,
            |s| s.metrics.copy_failures = 1,
            |s| s.metrics.source_failures = 1,
            |s| s.metrics.accounting_errors = 1,
            |s| s.active_leases = 1,
            |s| s.pending_leases = 1,
            |s| s.ring_capacity = 17,
            |s| s.production_owned = false,
            |s| s.arm = Arm::Control,
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut bad = tu.clone();
            mutate(&mut bad);
            assert!(
                !pair_mechanism_gate(&c, &t, &cp, &tp, &cu, &bad, &source, &source).passed,
                "mutation {i}"
            );
        }
    }
    #[test]
    fn source_upload_pair_rejects_source_cache_physical_identity_and_failures() {
        let (c, cp, cu) = fixture(Arm::Control);
        let (t, tp, tu) = fixture(Arm::Treatment);
        let source = source_fixture();
        let mutations: &[fn(&mut Snapshot)] = &[
            |s| s.source_nvme_reads += 1,
            |s| s.source_nvme_bytes += 1,
            |s| s.source_ram_hits += 1,
            |s| s.source_ram_misses += 1,
            |s| s.demand_source_requests += 1,
            |s| s.ram_cache_inserts += 1,
            |s| s.ram_cache_evictions += 1,
            |s| s.demand_ram_insert_ids_sha256.push('x'),
            |s| s.demand_ram_eviction_ids_sha256.push('x'),
            |s| s.physical_missing_ids_sha256.push('x'),
            |s| s.demand_source_request_ids_sha256.push('x'),
            |s| s.selected_route_ids_sha256.push('x'),
            |s| s.reservation_identity_sha256.push('x'),
            |s| s.physical_victim_ids_sha256.push('x'),
            |s| s.physical_residency_identity_sha256.push('x'),
            |s| s.mapping_publications += 1,
            |s| s.mapping_unpublications += 1,
            |s| s.physical_install_completions += 1,
            |s| s.physical_install_sets += 1,
            |s| s.physical_slot_bytes_staged += 1,
            |s| s.physical_slot_payload_copy_bytes += 1,
            |s| s.reservation_failures = 1,
            |s| s.physical_stage_failures = 1,
            |s| s.ordered_commit_failures = 1,
            |s| s.evidence_accounting_errors = 1,
            |s| s.overlapping_demand_sets = 1,
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut bad = t.clone();
            mutate(&mut bad);
            assert!(
                !pair_mechanism_gate(&c, &bad, &cp, &tp, &cu, &tu, &source, &source).passed,
                "mutation {i}"
            );
        }
        let mut changed_source = source_fixture();
        changed_source.production_batch_successes += 1;
        assert!(!pair_mechanism_gate(&c, &t, &cp, &tp, &cu, &tu, &source, &changed_source).passed);
    }
    #[test]
    fn source_upload_pass_is_independent_of_speed_but_requires_complete_failure_free_run() {
        let (c, cp, cu) = fixture(Arm::Control);
        let (mut t, tp, tu) = fixture(Arm::Treatment);
        let source = source_fixture();
        t.sum_individual_physical_stage_us = 3_000_000;
        t.physical_install_total_us = 3_000_009;
        assert!(pair_mechanism_gate(&c, &t, &cp, &tp, &cu, &tu, &source, &source).passed);
        assert!(completed_pass(true, false, true));
        for (complete, failure, gates) in [
            (false, false, true),
            (true, true, true),
            (true, false, false),
            (false, true, false),
        ] {
            assert!(!completed_pass(complete, failure, gates));
        }
    }
    #[test]
    fn source_upload_cli_schema_and_frozen_workload_have_no_overrides() {
        use clap::Parser;
        let args = [
            "micro-expert-router",
            MODE,
            "--config",
            FROZEN_CONFIG_PATH,
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "/tmp/qualification-candidate.json",
        ];
        let parsed = crate::Cli::try_parse_from(args).unwrap();
        assert!(matches!(
            parsed.cmd,
            crate::Cmd::QualifyGpuNativeSourceToUploadCopyElisionProduction { .. }
        ));
        let mut invalid = args.to_vec();
        invalid.extend(["--output-tokens", "1"]);
        assert!(crate::Cli::try_parse_from(invalid).is_err());
        assert_eq!(
            SCHEMA,
            "mer.gpu-native-source-to-upload-copy-elision-production.v2"
        );
        assert_eq!(
            (
                FROZEN_OUTPUT_TOKENS,
                FROZEN_WARMUP_RUNS,
                FROZEN_MEASURED_RUNS
            ),
            (128, 1, 3)
        );
        assert_eq!(
            FROZEN_PROMPT,
            "Write a Rust function that adds two i32 values and returns the result."
        );
        let workload = frozen_workload("NVIDIA L4".into());
        assert_eq!(workload.cache_reset, "keep");
        assert_eq!(workload.sampling, "greedy");
    }
    pub(super) fn order_straggler_fixture(mut p: serde_json::Value) -> serde_json::Value {
        use serde_json::{json, Value};
        let mut reconciliation = serde_json::to_value(Reconciliation::default()).unwrap();
        for v in reconciliation.as_object_mut().unwrap().values_mut() {
            *v = Value::Bool(true);
        }
        let reconciliation = ProductionReconciliation {
            common: serde_json::from_value(reconciliation).unwrap(),
            warmup_production_cache_reservation_leaks_zero: true,
            warmup_production_batch_commit_violations_zero: true,
            warmup_stale_singleflight_entries_zero: true,
            production_cache_reservation_leaks_zero: true,
            production_batch_commit_violations_zero: true,
            stale_singleflight_entries_zero: true,
            all_invariants_pass: true,
        };
        let (behavioral, work_equivalence) = common_gates(&reconciliation.common);
        p["reconciliation"] = serde_json::to_value(reconciliation).unwrap();
        p["gates"]["behavioral"] = serde_json::to_value(behavioral).unwrap();
        p["gates"]["work_equivalence"] = serde_json::to_value(work_equivalence).unwrap();
        for key in [
            "warmup_and_measured_work_exact",
            "token_ids_text_and_routes_exact",
            "stale_generation_and_install_errors_zero",
            "physical_installs_reconcile_with_residency",
            "source_bytes_and_logical_admissions_reconcile",
            "passed",
        ] {
            p["gates"][key] = json!(true);
        }
        p["mode"] = json!(MODE);
        p["failure"] = Value::Null;
        p["frozen_workload"] = serde_json::to_value(frozen_workload("NVIDIA L4".into())).unwrap();
        for (key, b) in [
            ("both_arms_same_reservation_and_commit", true),
            ("source_scheduler_changed", false),
            ("staging_byte_count_changed", false),
            ("queue_ordering_changed", true),
            (
                "no_zero_requires_complete_coverage_before_first_write",
                true,
            ),
        ] {
            p[key] = json!(b);
        }
        for (key, n) in [
            ("payload_offset_bytes", EPOCH_BYTES),
            ("logical_expert_bytes", LOGICAL_EXPERT_BYTES),
            ("slot_stride_bytes", SLOT_STRIDE_BYTES),
            ("tail_padding_bytes", 0),
        ] {
            p[key] = json!(n);
        }
        p["provenance"] = json!({"build":{"dirty":false,"git_sha":"a".repeat(40)},"executable_sha256":"b".repeat(64),"artifacts":{"config":{"sha256":FROZEN_CONFIG_SHA256}}});
        let mut all_uploads = Vec::new();
        for prefix in ["warmup_", ""] {
            let uk = format!("{prefix}upload");
            let pk = format!("{prefix}production");
            let wk = format!("{prefix}work");
            let mk = if prefix.is_empty() {
                "mechanism"
            } else {
                "warmup_mechanism"
            };
            let mut arms = Vec::new();
            for (arm, key) in [(Arm::Control, "control"), (Arm::Treatment, "treatment")] {
                let n = p[key][mk]["source_nvme_reads"].as_u64().unwrap();
                assert_eq!(n % 2, 0);
                let factor = n / 2;
                let (s, pi, u) = fixture(arm);
                let scale = |value: Value| -> Value {
                    let mut value = value;
                    for (k, v) in value.as_object_mut().unwrap() {
                        if [
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
                            continue;
                        }
                        if let Some(n) = v.as_u64() {
                            *v = json!(n * factor);
                        }
                    }
                    value
                };
                let mut s = scale(serde_json::to_value(s).unwrap());
                let pi: GpuNativeProductionPhysicalInstallSnapshot =
                    serde_json::from_value(scale(serde_json::to_value(pi).unwrap())).unwrap();
                let mut u = scale(serde_json::to_value(u).unwrap());
                u["ordered_nvme_ids_sha256"] = p[key][&uk]["ordered_nvme_ids_sha256"].clone();
                u["fused_source_us"] = if arm == Arm::Treatment {
                    p[key][&uk]["fused_source_us"].clone()
                } else {
                    json!(0)
                };
                u["source_upload_fd_proof"] = serde_json::to_value(if arm == Arm::Control {
                    crate::io_provider::SourceUploadFdProofSnapshot::default()
                } else {
                    crate::io_provider::SourceUploadFdProofSnapshot {
                        source_upload_fd_proof_requests: n,
                        source_upload_fd_proof_hits: if prefix.is_empty() { n } else { 0 },
                        source_upload_fd_proof_misses: if prefix.is_empty() { 0 } else { n },
                        source_upload_fd_proof_failures: 0,
                    }
                })
                .unwrap();
                let u: UploadSnapshot = serde_json::from_value(u).unwrap();
                s["source_nvme_reads"] = json!(n);
                s["source_nvme_bytes"] = json!(n * FULL as u64);
                let s: Snapshot = serde_json::from_value(s).unwrap();
                p[key][format!("{prefix}source")] =
                    serde_json::to_value(concurrency_common_snapshot(&s)).unwrap();
                let mut source = serde_json::to_value(source_fixture()).unwrap();
                for (k, v) in p[key][&pk].as_object().unwrap() {
                    source[k] = v.clone();
                }
                let source: ProductionDemandSourceSnapshot =
                    serde_json::from_value(source).unwrap();
                let mut work = ArmWorkEvidence {
                    token_loop: Default::default(),
                    recovery: Default::default(),
                    routed_execution: Default::default(),
                    engine_storage: Default::default(),
                    gpu_expert_io: Default::default(),
                    gpu_expert_memory_before: Default::default(),
                    gpu_expert_memory_after: Default::default(),
                    gpu_native_residency: Default::default(),
                };
                work.gpu_native_residency.ram_to_vram_installs = s.physical_install_completions;
                work.gpu_native_residency
                    .logical_admissions_for_physical_misses = u.metrics.logical_admissions;
                work.engine_storage.nvme_bytes_read = n * FULL as u64;
                work.engine_storage.nvme_read_operations =
                    n - source.production_batch_experts + source.production_batch_successes;
                p[key][&wk] = serde_json::to_value(work).unwrap();
                p[key][mk] = serde_json::to_value(&s).unwrap();
                p[key][&uk] = serde_json::to_value(&u).unwrap();
                p[key][&pk] = serde_json::to_value(&source).unwrap();
                p[key][format!("{prefix}production_physical_install")] =
                    serde_json::to_value(&pi).unwrap();
                arms.push((s, pi, u, source));
            }
            let (c, cp, cu, cs) = &arms[0];
            let (t, tp, tu, ts) = &arms[1];
            let gate = pair_mechanism_gate(c, t, cp, tp, cu, tu, cs, ts);
            assert!(gate.passed, "{gate:?}");
            p["gates"][mk] = serde_json::to_value(gate).unwrap();
            all_uploads.push((cu.clone(), tu.clone()));
        }
        p["gates"]["source_upload_fd_proof"] = serde_json::to_value(source_upload_fd_proof_gate(
            &all_uploads[0].0,
            &all_uploads[0].1,
            &all_uploads[1].0,
            &all_uploads[1].1,
        ))
        .unwrap();
        for key in ["control", "treatment"] {
            p[key]["complete"] = json!(true);
            p[key]["isolated_runtime"] = json!(true);
            p[key]["failure"] = Value::Null;
            p[key]["warmup_ram_cache_state_sha256"] = json!("e".repeat(64));
            p[key]["warmup_results"] = json!([{"run_index":0,"generated_tokens":128,"generated_token_ids_sha256":"f".repeat(64),"generated_text_sha256":"c".repeat(64)}]);
            let provenance = p["provenance"].clone();
            let b = &mut p[key]["benchmark"];
            b["provenance"] = provenance;
            b["benchmark_complete"] = json!(true);
            b["failure"] = Value::Null;
            b["warmup_runs"] = json!(1);
            b["warmup_runs_completed"] = json!(1);
            b["measured_runs"] = json!(3);
            b["cache_reset"] = json!("keep");
            b["request"] = json!({"greedy":true,"requested_output_tokens":128});
            b["hardware"] = json!({"name":"NVIDIA L4","vendor_id":0x10de,"device_type":"DiscreteGpu","wgpu_backend":"vulkan","compute_plane":"wgpu-vulkan","software_adapter":false});
            b["runtime_shutdowns"] = json!([{"phase":key,"evidence":{"controlled_shutdown_requested":true,"all_runtime_resources_released":true}}]);
            for field in [
                "runtime_contract",
                "model_load",
                "model_identity",
                "production_configuration",
                "production_semantics",
            ] {
                b[field] = json!({});
            }
            for (field, value) in crate::gpu_native_real_benchmark::hma1e_test_runtime()
                .as_object()
                .unwrap()
            {
                b[field] = value.clone();
            }
            for run in b["per_run_results"].as_array_mut().unwrap() {
                let ids = vec![17u32; 128];
                run["generated_tokens"] = json!(128);
                run["generated_token_ids_sha256"] =
                    json!(crate::greedy_parity::token_ids_sha256(&ids));
                run["generated_token_ids"] = json!(ids);
                run["generated_text_sha256"] = json!("c".repeat(64));
            }
        }
        p
    }
}

/// Offline validation of HMA-1E's embedded production-v2 evidence. Reuse the
/// production pair/work/fd gates on typed snapshots, including all counter
/// fields; serialized PASS flags cannot override a failed reconstruction.
pub(crate) fn validate_recorded_authority(
    p: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::Value;
    fn require(ok: bool, message: &'static str) -> Result<(), Box<dyn std::error::Error>> {
        if ok {
            Ok(())
        } else {
            Err(message.into())
        }
    }
    fn all_true(v: &Value) -> bool {
        match v {
            Value::Bool(b) => *b,
            Value::Object(m) => !m.is_empty() && m.values().all(all_true),
            _ => false,
        }
    }
    fn hash(v: &Value) -> bool {
        v.as_str()
            .is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
    }
    require(
        p["schema"] == SCHEMA
            && p["mode"] == MODE
            && p["qualification_pass"] == true
            && p["benchmark_complete"] == true
            && p.get("failure") == Some(&Value::Null),
        "production-v2 incomplete/schema/mode/failure",
    )?;
    require(
        p["frozen_workload"] == serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
        "frozen workload mismatch",
    )?;
    for (key, expected) in [
        ("both_arms_same_reservation_and_commit", true),
        ("source_scheduler_changed", false),
        ("staging_byte_count_changed", false),
        ("queue_ordering_changed", true),
        (
            "no_zero_requires_complete_coverage_before_first_write",
            true,
        ),
    ] {
        require(p[key] == expected, "production semantics mismatch")?;
    }
    for (key, expected) in [
        ("payload_offset_bytes", EPOCH_BYTES),
        ("logical_expert_bytes", LOGICAL_EXPERT_BYTES),
        ("slot_stride_bytes", SLOT_STRIDE_BYTES),
        ("tail_padding_bytes", 0),
    ] {
        require(p[key] == expected, "production geometry mismatch")?;
    }
    // Deserialize the complete reconciliation shape; missing fields fail closed.
    let reconciliation: ProductionReconciliation =
        serde_json::from_value(p["reconciliation"].clone())?;
    require(
        all_true(&serde_json::to_value(&reconciliation)?),
        "production behavioral/work reconciliation failed",
    )?;
    let (behavioral, work_equivalence) = common_gates(&reconciliation.common);
    require(
        p["gates"]["behavioral"] == serde_json::to_value(behavioral)?
            && p["gates"]["work_equivalence"] == serde_json::to_value(work_equivalence)?,
        "behavioral/work gate mismatch",
    )?;
    for key in [
        "warmup_and_measured_work_exact",
        "token_ids_text_and_routes_exact",
        "stale_generation_and_install_errors_zero",
        "physical_installs_reconcile_with_residency",
        "source_bytes_and_logical_admissions_reconcile",
        "passed",
    ] {
        require(p["gates"][key] == true, "production gate failed/missing")?;
    }
    require(
        hash(&p["provenance"]["executable_sha256"])
            && p["provenance"]["build"]["dirty"] == false
            && p["provenance"]["build"]["git_sha"]
                .as_str()
                .is_some_and(|s| s.len() == 40),
        "missing clean production build provenance",
    )?;
    require(
        p["provenance"]["artifacts"]["config"]["sha256"] == FROZEN_CONFIG_SHA256,
        "config artifact hash mismatch",
    )?;
    let mut uploads = Vec::new();
    for warmup in [true, false] {
        let prefix = if warmup { "warmup_" } else { "" };
        let mechanism_key = if warmup {
            "warmup_mechanism"
        } else {
            "mechanism"
        };
        let work_key = format!("{prefix}work");
        let upload_key = format!("{prefix}upload");
        let physical_key = format!("{prefix}production_physical_install");
        let source_key = format!("{prefix}production");
        let c: Snapshot = serde_json::from_value(p["control"][mechanism_key].clone())?;
        let t: Snapshot = serde_json::from_value(p["treatment"][mechanism_key].clone())?;
        let cp: GpuNativeProductionPhysicalInstallSnapshot =
            serde_json::from_value(p["control"][&physical_key].clone())?;
        let tp: GpuNativeProductionPhysicalInstallSnapshot =
            serde_json::from_value(p["treatment"][&physical_key].clone())?;
        let cu: UploadSnapshot = serde_json::from_value(p["control"][&upload_key].clone())?;
        let tu: UploadSnapshot = serde_json::from_value(p["treatment"][&upload_key].clone())?;
        let cs: ProductionDemandSourceSnapshot =
            serde_json::from_value(p["control"][&source_key].clone())?;
        let ts: ProductionDemandSourceSnapshot =
            serde_json::from_value(p["treatment"][&source_key].clone())?;
        let cw: ArmWorkEvidence = serde_json::from_value(p["control"][&work_key].clone())?;
        let tw: ArmWorkEvidence = serde_json::from_value(p["treatment"][&work_key].clone())?;
        require(
            p["control"][format!("{prefix}source")]
                == serde_json::to_value(concurrency_common_snapshot(&c))?
                && p["treatment"][format!("{prefix}source")]
                    == serde_json::to_value(concurrency_common_snapshot(&t))?,
            "common source/mechanism snapshot mismatch",
        )?;
        let gate = pair_mechanism_gate(&c, &t, &cp, &tp, &cu, &tu, &cs, &ts);
        require(
            gate.passed && p["gates"][mechanism_key] == serde_json::to_value(&gate)?,
            "production mechanism reconstruction failed",
        )?;
        require(
            work_pair_exact(&cw, &tw)
                && work_errors_zero(&cw)
                && work_errors_zero(&tw)
                && arm_all_speculative_work_zero(&cw)
                && arm_all_speculative_work_zero(&tw),
            "production work/recovery/speculation reconstruction failed",
        )?;
        // Additional common reconciliation operands (work_pair_exact deliberately
        // leaves these to the shared production reconciliation).
        require(
            cw.gpu_expert_io.expert_weight_upload_bytes
                == tw.gpu_expert_io.expert_weight_upload_bytes
                && cw.gpu_native_residency.vram_hits == tw.gpu_native_residency.vram_hits
                && cw.gpu_native_residency.vram_misses == tw.gpu_native_residency.vram_misses
                && c.primary_pool_capacity == t.primary_pool_capacity,
            "production common work mismatch",
        )?;
        for (work, mechanism, upload, source) in [(&cw, &c, &cu, &cs), (&tw, &t, &tu, &ts)] {
            require(
                work.gpu_native_residency.ram_to_vram_installs
                    == mechanism.physical_install_completions
                    && work
                        .gpu_native_residency
                        .logical_admissions_for_physical_misses
                        == upload.metrics.logical_admissions
                    && work.engine_storage.nvme_bytes_read == mechanism.source_nvme_bytes
                    && mechanism
                        .source_nvme_reads
                        .checked_sub(source.production_batch_experts)
                        .and_then(|n| n.checked_add(source.production_batch_successes))
                        == Some(work.engine_storage.nvme_read_operations),
                "production source/install/work accounting mismatch",
            )?;
            require(
                source.ordinary_production_path_exercised
                    && source.production_cache_reservation_leaks == 0
                    && source.production_batch_commit_violations == 0
                    && source.stale_singleflight_entries == 0
                    && source.production_sequential_fallback_batch_read_error == 0,
                "production scheduler failure",
            )?;
        }
        uploads.push((cu, tu));
    }
    let proof =
        source_upload_fd_proof_gate(&uploads[0].0, &uploads[0].1, &uploads[1].0, &uploads[1].1);
    require(
        proof.passed && p["gates"]["source_upload_fd_proof"] == serde_json::to_value(proof)?,
        "fd proof reconstruction failed",
    )?;
    let c = &p["control"];
    let t = &p["treatment"];
    require(
        hash(&c["warmup_ram_cache_state_sha256"])
            && c["warmup_ram_cache_state_sha256"] == t["warmup_ram_cache_state_sha256"],
        "warmup cache identity mismatch",
    )?;
    for key in ["control", "treatment"] {
        let arm = &p[key];
        let b = &arm["benchmark"];
        require(
            arm["complete"] == true
                && arm.get("failure") == Some(&Value::Null)
                && arm["isolated_runtime"] == true,
            "arm incomplete/not isolated",
        )?;
        require(
            b["benchmark_complete"] == true
                && b.get("failure") == Some(&Value::Null)
                && b["provenance"] == p["provenance"],
            "benchmark/provenance incomplete",
        )?;
        require(
            b["warmup_runs"] == 1
                && b["warmup_runs_completed"] == 1
                && b["measured_runs"] == 3
                && b["cache_reset"] == "keep"
                && b["request"]["greedy"] == true
                && b["request"]["requested_output_tokens"] == 128,
            "benchmark workload mismatch",
        )?;
        require(
            b["hardware"]["name"] == "NVIDIA L4"
                && b["hardware"]["wgpu_backend"] == "vulkan"
                && b["hardware"]["device_type"] == "DiscreteGpu"
                && b["hardware"]["vendor_id"] == 0x10de
                && b["hardware"]["software_adapter"] == false
                && b["hardware"]["compute_plane"] == "wgpu-vulkan",
            "non-authoritative hardware identity",
        )?;
        crate::gpu_native_real_benchmark::audit_recorded_runtime(b)?;
        let shutdowns = b["runtime_shutdowns"]
            .as_array()
            .ok_or("missing shutdown evidence")?;
        require(
            shutdowns.len() == 1
                && shutdowns[0]["phase"] == key
                && shutdowns[0]["evidence"]["controlled_shutdown_requested"] == true
                && shutdowns[0]["evidence"]["all_runtime_resources_released"] == true,
            "runtime shutdown incomplete",
        )?;
    }
    for field in [
        "request",
        "production_configuration",
        "production_semantics",
        "model_identity",
        "hardware",
        "runtime_contract",
    ] {
        require(
            c["benchmark"][field].is_object() && c["benchmark"][field] == t["benchmark"][field],
            "arm runtime/config/model contract mismatch",
        )?;
    }
    for (key, len) in [("warmup_results", 1), ("measured", 3)] {
        let runs = |a: &Value| -> Result<Vec<Value>, Box<dyn std::error::Error>> {
            let v = if key == "measured" {
                &a["benchmark"]["per_run_results"]
            } else {
                &a[key]
            };
            Ok(v.as_array().ok_or("missing generated runs")?.clone())
        };
        let cr = runs(c)?;
        let tr = runs(t)?;
        require(
            cr.len() == len && tr.len() == len,
            "generated run count mismatch",
        )?;
        for (i, (a, b)) in cr.iter().zip(&tr).enumerate() {
            require(
                a["run_index"] == i
                    && b["run_index"] == i
                    && a["generated_tokens"] == 128
                    && b["generated_tokens"] == 128,
                "generated run index/token count mismatch",
            )?;
            for field in ["generated_token_ids_sha256", "generated_text_sha256"] {
                require(
                    hash(&a[field]) && a[field] == b[field],
                    "generated IDs/text mismatch",
                )?;
            }
            if key == "measured" {
                let ids: Vec<u32> = serde_json::from_value(a["generated_token_ids"].clone())?;
                require(
                    ids.len() == 128
                        && a["generated_token_ids"] == b["generated_token_ids"]
                        && a["generated_token_ids_sha256"]
                            == crate::greedy_parity::token_ids_sha256(&ids),
                    "generated token stream/hash corruption",
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn hma1e_test_production(p: serde_json::Value) -> serde_json::Value {
    tests::order_straggler_fixture(p)
}
