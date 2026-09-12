//! PR2-C.1 production qualification of the ordinary Q4 route-parallel encoder. No production config knob.
use super::*;
use crate::backend::gpu_native::q4_route_parallel::DispatchEvidence;
use parking_lot::Mutex;

pub(crate) const SCHEMA: &str = "mer.gpu-native-q4-route-parallel-production.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-q4-route-parallel-production";
const BASE_SHA: &str = "6be8881fbae7b2b8a4a8df0a612bf90923a7ef61";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Arm {
    Control,
    Treatment,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Mechanism {
    expert_layer_executions: u64,
    selected_routes_total: u64,
    top_k_min: u64,
    top_k_max: u64,
    control_serial_gate_up_dispatches: u64,
    control_serial_down_dispatches: u64,
    treatment_route_parallel_gate_up_dispatches: u64,
    treatment_route_parallel_down_dispatches: u64,
    treatment_route_parallel_routes_covered: u64,
    treatment_max_route_width: u64,
    activation_scratch_bytes: u64,
    route_output_bytes: u64,
    q4_gate_up_rows_covered: u64,
    q4_down_rows_covered: u64,
    expert_validation_dispatches: u64,
    expert_combine_dispatches: u64,
    expert_contain_dispatches: u64,
    unexpected_status_bits: u32,
    accounting_mismatch: bool,
}

pub(crate) struct Observation {
    pub(crate) arm: Arm,
    evidence: Mutex<Mechanism>,
}
impl Observation {
    pub(crate) fn new(arm: Arm) -> Self {
        Self {
            arm,
            evidence: Mutex::new(Mechanism::default()),
        }
    }
    pub(crate) fn record(&self, d: DispatchEvidence) {
        let mut m = self.evidence.lock();
        m.accounting_mismatch |= d.accounting_mismatch;
        macro_rules! add {
            ($field:ident, $value:expr) => {
                match m.$field.checked_add($value) {
                    Some(value) => m.$field = value,
                    None => m.accounting_mismatch = true,
                }
            };
        }
        add!(expert_layer_executions, d.layer_executions);
        add!(selected_routes_total, d.selected_routes);
        add!(
            control_serial_gate_up_dispatches,
            d.serial_gate_up_dispatches
        );
        add!(control_serial_down_dispatches, d.serial_down_dispatches);
        add!(
            treatment_route_parallel_gate_up_dispatches,
            d.parallel_gate_up_dispatches
        );
        add!(
            treatment_route_parallel_down_dispatches,
            d.parallel_down_dispatches
        );
        add!(treatment_route_parallel_routes_covered, d.routes_covered);
        add!(q4_gate_up_rows_covered, d.gate_up_rows);
        add!(q4_down_rows_covered, d.down_rows);
        add!(expert_validation_dispatches, d.validation_dispatches);
        add!(expert_combine_dispatches, d.combine_dispatches);
        add!(expert_contain_dispatches, d.contain_dispatches);
        m.top_k_min = if m.top_k_min == 0 {
            d.top_k
        } else {
            m.top_k_min.min(d.top_k)
        };
        m.top_k_max = m.top_k_max.max(d.top_k);
        m.treatment_max_route_width = m.treatment_max_route_width.max(d.max_route_width);
        if m.activation_scratch_bytes != 0 && m.activation_scratch_bytes != d.activation_bytes
            || m.route_output_bytes != 0 && m.route_output_bytes != d.route_output_bytes
        {
            m.accounting_mismatch = true;
        }
        m.activation_scratch_bytes = d.activation_bytes;
        m.route_output_bytes = d.route_output_bytes;
    }
    pub(crate) fn record_status(&self, statuses: &[u32], final_status: u32) {
        self.evidence.lock().unexpected_status_bits |=
            statuses.iter().fold(final_status, |a, b| a | b) & !4;
    }
    pub(crate) fn take(&self) -> Mechanism {
        std::mem::take(&mut *self.evidence.lock())
    }
}

impl Mechanism {
    pub(crate) fn unexpected_status_bits(&self) -> u32 {
        self.unexpected_status_bits
    }

    pub(crate) fn ordinary_production_route_parallel_exercised(&self) -> bool {
        let selected = self.expert_layer_executions.checked_mul(self.top_k_max);
        self.expert_layer_executions > 0
            && self.selected_routes_total > 0
            && self.top_k_min == self.top_k_max
            && self.top_k_max > 1
            && selected == Some(self.selected_routes_total)
            && self.control_serial_gate_up_dispatches == 0
            && self.control_serial_down_dispatches == 0
            && self.treatment_route_parallel_gate_up_dispatches == self.expert_layer_executions
            && self.treatment_route_parallel_down_dispatches == self.expert_layer_executions
            && self.treatment_route_parallel_routes_covered == self.selected_routes_total
            && self.treatment_max_route_width == self.top_k_max
            && !self.accounting_mismatch
    }
}

#[derive(Clone, Debug, Serialize)]
struct Q4ArmReport {
    q4_arm: Arm,
    common: ArmReport,
    warmup_mechanism: Mechanism,
    mechanism: Mechanism,
    final_ram_cache_state_sha256: Option<String>,
    warmup_baseline_physical_install_observer:
        Option<crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot>,
    baseline_physical_install_observer:
        Option<crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot>,
}

fn q4_source_snapshot(
    source: &crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot,
) -> GpuNativePhysicalInstallStagingQualificationSnapshot {
    let mut snapshot = concurrency_common_snapshot(source);
    snapshot.production_physical_install_changed = false;
    snapshot
}

async fn run_q4_arm(
    prepared: &Prepared,
    args: &CommandArgs,
    q4_arm: Arm,
) -> Result<Q4ArmReport, BenchmarkFailure> {
    // Both PR2-C.1 arms use the ordinary production source and physical-install
    // path. This existing treatment observer changes only qualification tracing.
    let (arm_name, arm) = match q4_arm {
        Arm::Control => (
            "control",
            GpuNativePhysicalInstallStagingQualificationArm::Control,
        ),
        Arm::Treatment => (
            "treatment",
            GpuNativePhysicalInstallStagingQualificationArm::Treatment,
        ),
    };
    let mode_name = MODE;
    let observation = Arc::new(Observation::new(q4_arm));
    let mut benchmark = benchmark_report(prepared);
    let runtime = crate::gpu_native_real_benchmark::construct_runtime(
        &prepared.spec,
        prepared.tokenizer.clone(),
        arm_name,
        None,
        &mut benchmark,
    )
    .await?;
    let enable_result = runtime
        .engine
        .enable_gpu_native_physical_install_concurrency_qualification(
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::Treatment,
        );
    let enable_result = enable_result.and_then(|()| {
        runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or_else(|| "missing GPU-native token loop".to_string())?
            .enable_q4_route_parallel_qualification(observation.clone())
    });
    if let Err(error) = enable_result {
        let failure = BenchmarkFailure::new("startup", "qualification-arm-enable-failed", error);
        let _ = crate::gpu_native_real_benchmark::shutdown_runtime(
            runtime,
            arm_name,
            None,
            &mut benchmark,
        )
        .await;
        return Err(failure);
    }
    if let Err(validation_error) = crate::gpu_native_real_benchmark::validate_and_record_runtime(
        &runtime,
        &prepared.resolved_config_sha256,
        &args.expected_adapter_name,
        &mut benchmark,
    ) {
        let shutdown = crate::gpu_native_real_benchmark::shutdown_runtime(
            runtime,
            arm_name,
            None,
            &mut benchmark,
        )
        .await;
        return match shutdown {
            Ok(()) => Err(validation_error),
            Err(shutdown_error) => Err(BenchmarkFailure::new(
                "postcondition",
                "runtime-validation-and-shutdown-failed",
                format!("{validation_error}; {shutdown_error}"),
            )),
        };
    }

    let mut warmup_results = Vec::with_capacity(FROZEN_WARMUP_RUNS);
    let mut execution_failure = None;
    let mut warmup_start = match ArmStart::capture(&runtime) {
        Ok(captured) => Some(captured),
        Err(error) => {
            execution_failure = Some(error);
            None
        }
    };
    if execution_failure.is_none() {
        for index in 0..FROZEN_WARMUP_RUNS {
            let result = crate::with_progress_timeout(
                format!("{mode_name} {arm_name} warmup {index}"),
                args.progress_watchdog,
                crate::gpu_native_real_benchmark::execute_request(
                    &runtime,
                    &prepared.prompt_ids,
                    FROZEN_OUTPUT_TOKENS,
                    index,
                ),
            )
            .await;
            match result {
                Ok(run) => {
                    warmup_results.push(WarmupEvidence {
                        run_index: index,
                        generated_tokens: run.generated_tokens,
                        generated_token_ids_sha256: run.generated_token_ids_sha256,
                        generated_text_sha256: run.generated_text_sha256,
                    });
                    benchmark.warmup_runs_completed += 1;
                }
                Err(error) => {
                    execution_failure = Some(BenchmarkFailure::new(
                        "inference",
                        "warmup-request-failed",
                        error.to_string(),
                    ));
                    break;
                }
            }
        }
    }

    let mut warmup_source = None;
    let mut warmup_concurrency = None;
    let mut warmup_production_physical_install = None;
    let mut warmup_production = None;
    let mut warmup_ram_cache_state_sha256 = None;
    let mut warmup_work = None;
    if execution_failure.is_none() {
        warmup_concurrency = runtime
            .engine
            .gpu_native_physical_install_concurrency_qualification_snapshot();
        warmup_source = warmup_concurrency.as_ref().map(q4_source_snapshot);
        if warmup_source.is_none() {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "missing-warmup-source-snapshot",
                "PR2-C qualification source counters disappeared after warmup",
            ));
        }
    }
    if execution_failure.is_none() {
        warmup_production = Some(runtime.engine.production_demand_source_snapshot());
    }
    if execution_failure.is_none() {
        warmup_production_physical_install = runtime.engine.production_physical_install_snapshot();
        if warmup_production_physical_install.is_none() {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "missing-warmup-production-physical-install-snapshot",
                "PR2-C production physical-install counters disappeared after warmup",
            ));
        }
    }
    if execution_failure.is_none() {
        warmup_ram_cache_state_sha256 = runtime
            .engine
            .gpu_native_demand_source_qualification_ram_cache_state_sha256();
        if warmup_ram_cache_state_sha256.is_none() {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "missing-warmup-ram-cache-state",
                "PR2-C RAM-cache state could not be hashed before the warmup counter reset",
            ));
        }
    }
    if execution_failure.is_none() {
        match warmup_start
            .take()
            .expect("warmup start captured")
            .finish(&runtime)
        {
            Ok(work) => warmup_work = Some(work),
            Err(error) => execution_failure = Some(error),
        }
    }

    let warmup_mechanism = observation.take();
    let mut start = None;
    if execution_failure.is_none() {
        if let Err(error) = runtime
            .engine
            .reset_gpu_native_demand_source_qualification()
        {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "qualification-counter-reset-failed",
                error,
            ));
        } else {
            match ArmStart::capture(&runtime) {
                Ok(captured) => start = Some(captured),
                Err(error) => execution_failure = Some(error),
            }
        }
    }

    if execution_failure.is_none() {
        for index in 0..FROZEN_MEASURED_RUNS {
            let result = crate::with_progress_timeout(
                format!("{mode_name} {arm_name} measured {index}"),
                args.progress_watchdog,
                crate::gpu_native_real_benchmark::execute_request(
                    &runtime,
                    &prepared.prompt_ids,
                    FROZEN_OUTPUT_TOKENS,
                    index,
                ),
            )
            .await;
            match result {
                Ok(run) => benchmark.per_run_results.push(run),
                Err(error) => {
                    execution_failure = Some(BenchmarkFailure::new(
                        "inference",
                        "measured-request-failed",
                        error.to_string(),
                    ));
                    break;
                }
            }
        }
    }

    let concurrency = runtime
        .engine
        .gpu_native_physical_install_concurrency_qualification_snapshot();
    let source = concurrency.as_ref().map(q4_source_snapshot);
    let production_physical_install = runtime.engine.production_physical_install_snapshot();
    if execution_failure.is_none() && production_physical_install.is_none() {
        execution_failure = Some(BenchmarkFailure::new(
            "postcondition",
            "missing-production-physical-install-snapshot",
            "PR2-C production physical-install counters disappeared after measurement",
        ));
    }
    let production = Some(runtime.engine.production_demand_source_snapshot());
    let work = if execution_failure.is_none() {
        match start.expect("measurement start captured").finish(&runtime) {
            Ok(work) => Some(work),
            Err(error) => {
                execution_failure = Some(error);
                None
            }
        }
    } else {
        None
    };

    let mechanism = observation.take();
    let final_ram_cache_state_sha256 = runtime
        .engine
        .gpu_native_demand_source_qualification_ram_cache_state_sha256();
    if final_ram_cache_state_sha256.is_none() && execution_failure.is_none() {
        execution_failure = Some(BenchmarkFailure::new(
            "postcondition",
            "missing-final-cache-state",
            "PR2-C.1 requires final RAM cache identity",
        ));
    }
    let shutdown =
        crate::gpu_native_real_benchmark::shutdown_runtime(runtime, arm_name, None, &mut benchmark)
            .await;
    if let Err(error) = shutdown {
        execution_failure = Some(match execution_failure {
            Some(previous) => BenchmarkFailure::new(
                "postcondition",
                "execution-and-shutdown-failed",
                format!("{previous}; {error}"),
            ),
            None => error,
        });
    }

    if execution_failure.is_none() {
        if let Err(error) = benchmark.finish() {
            execution_failure = Some(error);
        }
    }
    if let Some(failure) = execution_failure.clone() {
        benchmark.fail(failure.clone());
    }
    Ok(Q4ArmReport {
        q4_arm,
        warmup_mechanism,
        mechanism,
        final_ram_cache_state_sha256,
        common: ArmReport {
            arm,
            complete: execution_failure.is_none(),
            failure: execution_failure,
            isolated_runtime: true,
            warmup_results,
            warmup_source,
            warmup_production_physical_install,
            warmup_production,
            warmup_ram_cache_state_sha256,
            warmup_work,
            source,
            production_physical_install,
            production,
            work,
            benchmark,
        },
        warmup_baseline_physical_install_observer: warmup_concurrency,
        baseline_physical_install_observer: concurrency,
    })
}

#[derive(Clone, Debug, Serialize)]
struct Contract {
    qualification_only: bool,
    production_q4_route_parallel_changed: bool,
    control_uses_frozen_serial_reference_path: bool,
    treatment_uses_ordinary_production_route_parallel_path: bool,
    source_acquisition_changed: bool,
    ram_cache_policy_changed: bool,
    physical_residency_changed: bool,
    victim_policy_changed: bool,
    mapping_semantics_changed: bool,
    logical_admission_changed: bool,
    prefetch_or_speculation_changed: bool,
    recovery_semantics_changed: bool,
    routing_semantics_changed: bool,
    model_or_workload_changed: bool,
    expert_q4_arithmetic_changed: bool,
    expert_combine_order_changed: bool,
    expert_failure_semantics_changed: bool,
    queue_submission_contract_changed: bool,
    wgpu_version_changed: bool,
}
impl Default for Contract {
    fn default() -> Self {
        Self {
            qualification_only: false,
            production_q4_route_parallel_changed: true,
            control_uses_frozen_serial_reference_path: true,
            treatment_uses_ordinary_production_route_parallel_path: true,
            source_acquisition_changed: false,
            ram_cache_policy_changed: false,
            physical_residency_changed: false,
            victim_policy_changed: false,
            mapping_semantics_changed: false,
            logical_admission_changed: false,
            prefetch_or_speculation_changed: false,
            recovery_semantics_changed: false,
            routing_semantics_changed: false,
            model_or_workload_changed: false,
            expert_q4_arithmetic_changed: false,
            expert_combine_order_changed: false,
            expert_failure_semantics_changed: false,
            queue_submission_contract_changed: false,
            wgpu_version_changed: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct MechanismGate {
    control_frozen_serial_path_exercised: bool,
    treatment_ordinary_production_route_parallel_path_exercised: bool,
    control_dispatch_accounting_exact: bool,
    treatment_dispatch_accounting_exact: bool,
    treatment_route_width_gt_one: bool,
    expert_layer_execution_count_exact: bool,
    selected_route_count_exact: bool,
    gate_up_row_coverage_exact: bool,
    down_row_coverage_exact: bool,
    route_output_logical_shape_exact: bool,
    validation_dispatch_count_exact: bool,
    combine_dispatch_count_exact: bool,
    contain_dispatch_count_exact: bool,
    activation_scratch_geometry_exact: bool,
    unexpected_status_bits_zero: bool,
    mechanism_accounting_mismatch_zero: bool,
    passed: bool,
}

fn mechanism_gate(c: &Mechanism, t: &Mechanism, d_model: u64, d_ff: u64, k: u64) -> MechanismGate {
    let expected = c.expert_layer_executions.checked_mul(k);
    let gate_rows = expected.and_then(|v| v.checked_mul(d_ff));
    let down_rows = expected.and_then(|v| v.checked_mul(d_model));
    let serial = c.expert_layer_executions > 0
        && expected == Some(c.selected_routes_total)
        && expected == Some(c.control_serial_gate_up_dispatches)
        && expected == Some(c.control_serial_down_dispatches)
        && c.treatment_route_parallel_gate_up_dispatches == 0
        && c.treatment_route_parallel_down_dispatches == 0
        && c.treatment_route_parallel_routes_covered == 0
        && c.treatment_max_route_width == 0;
    let parallel = t.expert_layer_executions > 0
        && t.control_serial_gate_up_dispatches == 0
        && t.control_serial_down_dispatches == 0
        && t.treatment_route_parallel_gate_up_dispatches == t.expert_layer_executions
        && t.treatment_route_parallel_down_dispatches == t.expert_layer_executions
        && expected == Some(t.treatment_route_parallel_routes_covered)
        && t.treatment_max_route_width == k;
    let mut gate = MechanismGate {
        control_frozen_serial_path_exercised: serial,
        treatment_ordinary_production_route_parallel_path_exercised: parallel,
        control_dispatch_accounting_exact: serial,
        treatment_dispatch_accounting_exact: parallel,
        treatment_route_width_gt_one: t.treatment_max_route_width >= 2,
        expert_layer_execution_count_exact: c.expert_layer_executions == t.expert_layer_executions,
        selected_route_count_exact: c.selected_routes_total == t.selected_routes_total
            && c.top_k_min == k
            && c.top_k_max == k
            && t.top_k_min == k
            && t.top_k_max == k,
        gate_up_row_coverage_exact: gate_rows == Some(c.q4_gate_up_rows_covered)
            && gate_rows == Some(t.q4_gate_up_rows_covered),
        down_row_coverage_exact: down_rows == Some(c.q4_down_rows_covered)
            && down_rows == Some(t.q4_down_rows_covered),
        route_output_logical_shape_exact: k.checked_mul(d_model).and_then(|v| v.checked_mul(4))
            == Some(c.route_output_bytes)
            && c.route_output_bytes == t.route_output_bytes,
        validation_dispatch_count_exact: c.expert_validation_dispatches
            == c.expert_layer_executions
            && t.expert_validation_dispatches == c.expert_validation_dispatches,
        combine_dispatch_count_exact: c.expert_combine_dispatches == c.expert_layer_executions
            && t.expert_combine_dispatches == c.expert_combine_dispatches,
        contain_dispatch_count_exact: c.expert_contain_dispatches == c.expert_layer_executions
            && t.expert_contain_dispatches == c.expert_contain_dispatches,
        activation_scratch_geometry_exact: d_ff.checked_mul(4) == Some(c.activation_scratch_bytes)
            && k.checked_mul(d_ff).and_then(|v| v.checked_mul(4))
                == Some(t.activation_scratch_bytes),
        unexpected_status_bits_zero: c.unexpected_status_bits == 0 && t.unexpected_status_bits == 0,
        mechanism_accounting_mismatch_zero: !c.accounting_mismatch && !t.accounting_mismatch,
        passed: false,
    };
    gate.passed = gate.control_frozen_serial_path_exercised
        && gate.treatment_ordinary_production_route_parallel_path_exercised
        && gate.control_dispatch_accounting_exact
        && gate.treatment_dispatch_accounting_exact
        && gate.treatment_route_width_gt_one
        && gate.expert_layer_execution_count_exact
        && gate.selected_route_count_exact
        && gate.gate_up_row_coverage_exact
        && gate.down_row_coverage_exact
        && gate.route_output_logical_shape_exact
        && gate.validation_dispatch_count_exact
        && gate.combine_dispatch_count_exact
        && gate.contain_dispatch_count_exact
        && gate.activation_scratch_geometry_exact
        && gate.unexpected_status_bits_zero
        && gate.mechanism_accounting_mismatch_zero;
    gate
}

// Only these already-defined wall-time fields are excluded from exact work
// comparison. Every other field, including any future counter, must match.
const CONTEXT_TIMINGS: &[&str] = &[
    "physical_probe_us",
    "ssd_stall_us",
    "residency_service_us",
    "boundary_wait_us",
    "source_acquisition_wall_us",
    "logical_demand_admission_us",
    "physical_demand_install_us",
    "total_residency_service_us",
    "physical_slot_prepare_us",
    "physical_queue_staging_us",
    "mapping_publication_us",
    "physical_install_total_us",
];
fn work_projection(mut value: serde_json::Value) -> serde_json::Value {
    fn visit(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(fields) => {
                fields.retain(|key, _| !CONTEXT_TIMINGS.contains(&key.as_str()));
                for v in fields.values_mut() {
                    visit(v);
                }
            }
            serde_json::Value::Array(items) => {
                for v in items {
                    visit(v);
                }
            }
            _ => {}
        }
    }
    visit(&mut value);
    value
}
fn exact_work<T: Serialize>(c: &T, t: &T) -> bool {
    match (serde_json::to_value(c), serde_json::to_value(t)) {
        (Ok(c), Ok(t)) => !c.is_null() && !t.is_null() && work_projection(c) == work_projection(t),
        _ => false,
    }
}

#[derive(Clone, Debug, Serialize)]
struct Q4Reconciliation {
    production_and_work: ProductionReconciliation,
    warmup_all_source_counters_and_order_exact: bool,
    measured_all_source_counters_and_order_exact: bool,
    warmup_all_residency_recovery_and_boundary_work_exact: bool,
    measured_all_residency_recovery_and_boundary_work_exact: bool,
    generated_token_ids_exact: bool,
    generated_text_hashes_exact: bool,
    warmup_generated_text_hashes_exact: bool,
    final_ram_cache_state_exact: bool,
    all_invariants_pass: bool,
}

fn reconcile_q4(c: &Q4ArmReport, t: &Q4ArmReport) -> Q4Reconciliation {
    let production_and_work =
        production_reconciliation(reconcile(&c.common, &t.common), &c.common, &t.common);
    let generated_token_ids_exact = generated_results(&c.common)
        .iter()
        .map(|r| &r.generated_token_ids)
        .eq(generated_results(&t.common)
            .iter()
            .map(|r| &r.generated_token_ids));
    let generated_text_hashes_exact = generated_results(&c.common)
        .iter()
        .map(|r| &r.generated_text_sha256)
        .eq(generated_results(&t.common)
            .iter()
            .map(|r| &r.generated_text_sha256));
    let warmup_generated_text_hashes_exact = c
        .common
        .warmup_results
        .iter()
        .map(|r| &r.generated_text_sha256)
        .eq(t
            .common
            .warmup_results
            .iter()
            .map(|r| &r.generated_text_sha256));
    let mut r = Q4Reconciliation {
        production_and_work,
        warmup_all_source_counters_and_order_exact: exact_work(
            &c.common.warmup_source,
            &t.common.warmup_source,
        ),
        measured_all_source_counters_and_order_exact: exact_work(
            &c.common.source,
            &t.common.source,
        ),
        warmup_all_residency_recovery_and_boundary_work_exact: exact_work(
            &c.common.warmup_work,
            &t.common.warmup_work,
        ),
        measured_all_residency_recovery_and_boundary_work_exact: exact_work(
            &c.common.work,
            &t.common.work,
        ),
        generated_token_ids_exact,
        generated_text_hashes_exact,
        warmup_generated_text_hashes_exact,
        final_ram_cache_state_exact: c.final_ram_cache_state_sha256.is_some()
            && c.final_ram_cache_state_sha256 == t.final_ram_cache_state_sha256,
        all_invariants_pass: false,
    };
    r.all_invariants_pass = r.production_and_work.all_invariants_pass
        && r.warmup_all_source_counters_and_order_exact
        && r.measured_all_source_counters_and_order_exact
        && r.warmup_all_residency_recovery_and_boundary_work_exact
        && r.measured_all_residency_recovery_and_boundary_work_exact
        && r.generated_token_ids_exact
        && r.generated_text_hashes_exact
        && r.warmup_generated_text_hashes_exact
        && r.final_ram_cache_state_exact;
    r
}

#[derive(Clone, Debug, Serialize)]
struct Gates {
    behavioral: BehavioralGate,
    work_equivalence: WorkEquivalenceGate,
    warmup_mechanism: MechanismGate,
    measured_mechanism: MechanismGate,
    selected_route_weights_equality_supported_by_existing_boundary: bool,
    selected_route_weights_contract: &'static str,
    all_invariants_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Performance {
    measurements: ProductionPerformanceComparison,
    control_ttft_seconds: f64,
    treatment_ttft_seconds: f64,
    control_activation_scratch_bytes: u64,
    treatment_activation_scratch_bytes: u64,
    control_route_compute_dispatches_per_expert_layer: f64,
    treatment_route_compute_dispatches_per_expert_layer: f64,
    route_compute_dispatch_reduction_percent: f64,
    structural_metric_name: &'static str,
    classification_metric: &'static str,
    timing_claim: &'static str,
}
fn classify_performance(control_decode: f64, treatment_decode: f64) -> &'static str {
    if treatment_decode > control_decode {
        "improved"
    } else if treatment_decode < control_decode {
        "regressed"
    } else {
        "neutral"
    }
}
fn performance_q4(c: &Q4ArmReport, t: &Q4ArmReport) -> Result<Performance, BenchmarkFailure> {
    let mut measurements = production_performance(&c.common, &t.common)?;
    measurements.performance_result = classify_performance(
        measurements.control.decode_tps,
        measurements.treatment.decode_tps,
    );
    let dispatches = |m: &Mechanism| {
        (m.control_serial_gate_up_dispatches
            + m.control_serial_down_dispatches
            + m.treatment_route_parallel_gate_up_dispatches
            + m.treatment_route_parallel_down_dispatches) as f64
            / m.expert_layer_executions as f64
    };
    let control = dispatches(&c.mechanism);
    let treatment = dispatches(&t.mechanism);
    let ttft = |a: &ArmReport| {
        generated_results(a)
            .iter()
            .map(|r| r.timing.time_to_first_token_seconds)
            .sum::<f64>()
            / generated_results(a).len() as f64
    };
    Ok(Performance {
        measurements, control_ttft_seconds: ttft(&c.common), treatment_ttft_seconds: ttft(&t.common),
        control_activation_scratch_bytes: c.mechanism.activation_scratch_bytes,
        treatment_activation_scratch_bytes: t.mechanism.activation_scratch_bytes,
        control_route_compute_dispatches_per_expert_layer: control,
        treatment_route_compute_dispatches_per_expert_layer: treatment,
        route_compute_dispatch_reduction_percent: if control > 0.0 { (control - treatment) / control * 100.0 } else { 0.0 },
        structural_metric_name: "ROUTE-COMPUTE DISPATCH REDUCTION",
        classification_metric: "measured mean decode TPS; formal gates are independent of performance",
        timing_claim: "2D dispatch exposes routes to GPU scheduling; dispatch counts do not measure simultaneous execution or GPU kernel time",
    })
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    mode: &'static str,
    base_sha: &'static str,
    #[serde(flatten)]
    contract: Contract,
    frozen_workload: FrozenWorkload,
    d_model: usize,
    d_ff: usize,
    layers: usize,
    experts_per_layer: usize,
    top_k: usize,
    replacement: &'static str,
    dispatch_accounting_definition: &'static str,
    baseline_observer_contract: &'static str,
    benchmark_complete: bool,
    qualification_pass: bool,
    performance_result: &'static str,
    failure: Option<BenchmarkFailure>,
    provenance: Option<BenchmarkProvenance>,
    control: Option<Q4ArmReport>,
    treatment: Option<Q4ArmReport>,
    reconciliation: Option<Q4Reconciliation>,
    gates: Option<Gates>,
    performance: Option<Performance>,
}
impl Report {
    fn pending(adapter: String) -> Self {
        Self {
            schema: SCHEMA, mode: MODE, base_sha: BASE_SHA, contract: Contract::default(),
            frozen_workload: frozen_workload(adapter), d_model: 2048, d_ff: 768, layers: 48,
            experts_per_layer: 128, top_k: 8, replacement: "physical-lru-prediction-protected",
            baseline_observer_contract: "both PR2-C.1 arms enable the existing PR2-B-B treatment observer and use ordinary source/install behavior; nested baseline observer production-change flags describe PR2-B versus its historical baseline, not changes made by PR2-C.1",
            dispatch_accounting_definition: "host-encoded expert layer executions, including invalid tails and recovery segments; selected_routes_total counts encoded route lanes; accepted selected-route counts and ordered IDs are independently recorded by the existing boundary contract",
            benchmark_complete: false, qualification_pass: false, performance_result: "not_measured",
            failure: None, provenance: None, control: None, treatment: None,
            reconciliation: None, gates: None, performance: None,
        }
    }
}
fn write_report(report: &Report, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    use std::io::Write;
    file.write_all(serde_json::to_string_pretty(report)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = Report::pending(args.expected_adapter_name.clone());
    let result = run_qualification(&args, &mut report).await;
    if let Err(error) = &result {
        report.qualification_pass = false;
        if report.failure.is_none() {
            report.failure = Some(BenchmarkFailure::new(
                "qualification",
                "pr2c1-failed",
                error.to_string(),
            ));
        }
    }
    write_report(&report, &args.report_out)?;
    result
}
async fn run_qualification(
    args: &CommandArgs,
    report: &mut Report,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.report_out.exists() {
        return Err("PR2-C.1 requires a new report path".into());
    }
    if args.expected_adapter_name != "NVIDIA L4" {
        return Err("frozen PR2-C.1 adapter must be NVIDIA L4".into());
    }
    let prepared = prepare(args)?;
    report.provenance = Some(prepared.provenance.clone());
    for arm in [Arm::Control, Arm::Treatment] {
        let result = run_q4_arm(&prepared, args, arm).await?;
        let failure = result.common.failure.clone();
        match arm {
            Arm::Control => report.control = Some(result),
            Arm::Treatment => report.treatment = Some(result),
        }
        if let Some(failure) = failure {
            report.failure = Some(failure.clone());
            return Err(failure.into());
        }
    }
    let c = report.control.as_ref().expect("control stored");
    let t = report.treatment.as_ref().expect("treatment stored");
    let reconciliation = reconcile_q4(c, t);
    let (mut behavioral, mut work_equivalence) =
        common_gates(&reconciliation.production_and_work.common);
    behavioral.passed &= reconciliation.generated_token_ids_exact
        && reconciliation.generated_text_hashes_exact
        && reconciliation.warmup_generated_text_hashes_exact;
    work_equivalence.passed &= reconciliation.all_invariants_pass;
    let warmup_mechanism = mechanism_gate(&c.warmup_mechanism, &t.warmup_mechanism, 2048, 768, 8);
    let measured_mechanism = mechanism_gate(&c.mechanism, &t.mechanism, 2048, 768, 8);
    let all_invariants_pass = reconciliation.all_invariants_pass
        && behavioral.passed
        && work_equivalence.passed
        && warmup_mechanism.passed
        && measured_mechanism.passed;
    // Performance is calculated and reported even when formal parity fails.
    let performance = performance_q4(c, t)?;
    report.performance_result = performance.measurements.performance_result;
    report.benchmark_complete = c.common.complete && t.common.complete;
    report.qualification_pass = report.benchmark_complete && all_invariants_pass;
    report.reconciliation = Some(reconciliation);
    report.gates = Some(Gates {
        behavioral, work_equivalence, warmup_mechanism, measured_mechanism,
        selected_route_weights_equality_supported_by_existing_boundary: false,
        selected_route_weights_contract: "existing ordinary boundary exposes ordered selected IDs, not f32 selected weights; no additional readback is introduced; focused same-device tests compare route and combined f32 bits",
        all_invariants_pass,
    });
    report.performance = Some(performance);
    if !report.qualification_pass {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "formal-gate-failed",
            "PR2-C.1 correctness, work or mechanism reconciliation failed",
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evidence(arm: Arm, k: u64) -> DispatchEvidence {
        DispatchEvidence {
            layer_executions: 1,
            accounting_mismatch: false,
            selected_routes: k,
            top_k: k,
            serial_gate_up_dispatches: if arm == Arm::Control { k } else { 0 },
            serial_down_dispatches: if arm == Arm::Control { k } else { 0 },
            parallel_gate_up_dispatches: u64::from(arm == Arm::Treatment),
            parallel_down_dispatches: u64::from(arm == Arm::Treatment),
            routes_covered: if arm == Arm::Treatment { k } else { 0 },
            max_route_width: if arm == Arm::Treatment { k } else { 0 },
            activation_bytes: 768 * 4 * if arm == Arm::Treatment { k } else { 1 },
            route_output_bytes: k * 2048 * 4,
            gate_up_rows: k * 768,
            down_rows: k * 2048,
            validation_dispatches: 1,
            combine_dispatches: 1,
            contain_dispatches: 1,
        }
    }
    fn pair(k: u64) -> (Mechanism, Mechanism) {
        let c = Observation::new(Arm::Control);
        let t = Observation::new(Arm::Treatment);
        for _ in 0..3 {
            c.record(evidence(Arm::Control, k));
            t.record(evidence(Arm::Treatment, k));
        }
        (c.take(), t.take())
    }
    #[test]
    fn q4_route_parallel_accounting_reconciles_control_and_treatment() {
        for k in 2..=8 {
            let (c, t) = pair(k);
            assert!(mechanism_gate(&c, &t, 2048, 768, k).passed);
            assert_eq!(
                c.control_serial_gate_up_dispatches + c.control_serial_down_dispatches,
                6 * k
            );
            assert_eq!(
                t.treatment_route_parallel_gate_up_dispatches
                    + t.treatment_route_parallel_down_dispatches,
                6
            );
        }
    }
    #[test]
    fn q4_route_parallel_width_one_is_valid_math_but_not_mechanism_pass() {
        let (c, t) = pair(1);
        let g = mechanism_gate(&c, &t, 2048, 768, 1);
        assert!(g.control_dispatch_accounting_exact && g.treatment_dispatch_accounting_exact);
        assert!(!g.treatment_route_width_gt_one && !g.passed);
    }
    #[test]
    fn q4_route_parallel_mechanism_rejects_serial_fallback_and_empty_work() {
        let (c, mut t) = pair(8);
        t.control_serial_gate_up_dispatches = 24;
        t.control_serial_down_dispatches = 24;
        t.treatment_route_parallel_gate_up_dispatches = 0;
        t.treatment_route_parallel_down_dispatches = 0;
        assert!(!mechanism_gate(&c, &t, 2048, 768, 8).passed);
        assert!(!mechanism_gate(&Mechanism::default(), &Mechanism::default(), 2048, 768, 8).passed);
    }
    #[test]
    fn q4_route_parallel_mechanism_rejects_each_accounting_difference() {
        let (c, t) = pair(8);
        macro_rules! corrupt { ($($f:ident),+) => { $(let mut bad = t.clone(); bad.$f += 1; assert!(!mechanism_gate(&c, &bad, 2048, 768, 8).passed, stringify!($f));)+ }; }
        corrupt!(
            expert_layer_executions,
            selected_routes_total,
            top_k_min,
            top_k_max,
            control_serial_gate_up_dispatches,
            control_serial_down_dispatches,
            treatment_route_parallel_gate_up_dispatches,
            treatment_route_parallel_down_dispatches,
            treatment_route_parallel_routes_covered,
            treatment_max_route_width,
            activation_scratch_bytes,
            route_output_bytes,
            q4_gate_up_rows_covered,
            q4_down_rows_covered,
            expert_validation_dispatches,
            expert_combine_dispatches,
            expert_contain_dispatches,
            unexpected_status_bits
        );
    }
    #[test]
    fn q4_route_parallel_observation_reset_overflow_and_status_fail_closed() {
        let o = Observation::new(Arm::Treatment);
        o.record(evidence(Arm::Treatment, 8));
        o.record_status(&[0, 4, 4], 4);
        assert_eq!(o.take().unexpected_status_bits, 0);
        assert_eq!(o.take().expert_layer_executions, 0);
        o.record_status(&[4, 8, 1 << 20], 0);
        assert_ne!(o.take().unexpected_status_bits, 0);
        let mut e = evidence(Arm::Treatment, 8);
        e.layer_executions = u64::MAX;
        o.record(e);
        o.record(evidence(Arm::Treatment, 8));
        assert!(o.take().accounting_mismatch);
    }
    #[test]
    fn q4_route_parallel_work_gate_rejects_every_non_timing_field_and_absent_evidence() {
        let fields = [
            "demand_source_requests",
            "source_ram_hits",
            "source_ram_misses",
            "source_nvme_reads",
            "source_nvme_bytes",
            "ram_cache_inserts",
            "ram_cache_evictions",
            "logical_admissions",
            "ram_to_vram_installs",
            "expert_weight_upload_bytes",
            "physical_install_completions",
            "physical_evictions",
            "physical_victim_ids_sha256",
            "physical_residency_identity_sha256",
            "mapping_publications",
            "mapping_unpublications",
            "physical_reinstalls",
            "residency_miss_attempts",
            "residency_services",
            "recovery_segments",
            "queue_submissions",
            "boundary_maps",
            "boundary_readbacks",
            "speculative_requests",
            "demand_ram_insert_ids_sha256",
            "demand_ram_eviction_ids_sha256",
            "future_counter",
        ];
        let mut c = serde_json::json!({"source_acquisition_wall_us":1});
        for key in fields {
            c[key] = 1.into();
        }
        let mut timing = c.clone();
        for key in CONTEXT_TIMINGS {
            timing[*key] = 123.into();
        }
        assert!(exact_work(&c, &timing));
        for key in fields {
            let mut t = c.clone();
            t[key] = 2.into();
            assert!(!exact_work(&c, &t), "{key}");
        }
        assert!(!exact_work(&Option::<u64>::None, &None));
    }
    #[test]
    fn q4_route_parallel_contract_is_explicit_and_performance_independent() {
        let report = Report::pending("NVIDIA L4".into());
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(
            json["schema"],
            "mer.gpu-native-q4-route-parallel-production.v1"
        );
        assert_eq!(json["base_sha"], "6be8881fbae7b2b8a4a8df0a612bf90923a7ef61");
        for field in [
            "production_q4_route_parallel_changed",
            "control_uses_frozen_serial_reference_path",
            "treatment_uses_ordinary_production_route_parallel_path",
        ] {
            assert_eq!(json[field], true);
        }
        for field in [
            "qualification_only",
            "source_acquisition_changed",
            "ram_cache_policy_changed",
            "physical_residency_changed",
            "victim_policy_changed",
            "mapping_semantics_changed",
            "logical_admission_changed",
            "prefetch_or_speculation_changed",
            "recovery_semantics_changed",
            "routing_semantics_changed",
            "model_or_workload_changed",
            "expert_q4_arithmetic_changed",
            "expert_combine_order_changed",
            "expert_failure_semantics_changed",
            "queue_submission_contract_changed",
            "wgpu_version_changed",
            "benchmark_complete",
            "qualification_pass",
        ] {
            assert_eq!(json[field], false);
        }
        let (c, t) = pair(8);
        for (a, b, expected) in [
            (1.0, 2.0, "improved"),
            (2.0, 2.0, "neutral"),
            (3.0, 2.0, "regressed"),
        ] {
            assert_eq!(classify_performance(a, b), expected);
            assert!(mechanism_gate(&c, &t, 2048, 768, 8).passed);
        }
    }
    #[test]
    fn q4_route_parallel_source_residency_and_configuration_remain_frozen() {
        use sha2::Digest;
        // HMA-1D adds qualification-only source observations to these two
        // files. Retain the whole-file guard at the reviewed instrumented
        // source; residency/configuration and the Q4 mechanism stay frozen.
        for (source, expected) in [
            (
                include_str!("engine.rs"),
                "e748e3902b441ec1b83b1f6b8b6619e51b3bafddc8d38ffc5abdd20031dae04d",
            ),
            (
                include_str!("gpu_native_residency.rs"),
                "662feee8293175a06a26ee3a400fdff23ce245513d1ebaa44a51b3cc9f601cbe",
            ),
            (
                include_str!("config.rs"),
                "f57f8131c2f37976a5019cd16d9e8fdac83f75379bab324235e27f4a27395428",
            ),
            (
                include_str!("gpu_native_physical_install_staging.rs"),
                "28b4bdda34cf23caff4c6f8ecba8eeb66dbbb408803e28243b8fed216cc29612",
            ),
        ] {
            assert_eq!(
                format!("{:x}", sha2::Sha256::digest(source.as_bytes())),
                expected
            );
        }
    }
    #[test]
    fn q4_route_parallel_cli_is_explicit_and_has_no_workload_overrides() {
        use clap::Parser;
        let arguments = [
            "micro-expert-router",
            MODE,
            "--config",
            FROZEN_CONFIG_PATH,
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "/tmp/pr2c-new.json",
        ];
        let cli = crate::Cli::try_parse_from(arguments).unwrap();
        assert!(matches!(
            cli.cmd,
            crate::Cmd::QualifyGpuNativeQ4RouteParallelProduction { .. }
        ));
        for override_flag in [
            "--top-k",
            "--output-tokens",
            "--warmup-runs",
            "--measured-runs",
            "--request-json",
            "--cache-reset",
            "--parallel",
            "--prompt",
        ] {
            let mut invalid = arguments.to_vec();
            invalid.extend([override_flag, "1"]);
            assert!(crate::Cli::try_parse_from(invalid).is_err());
        }
    }
    #[tokio::test]
    async fn q4_route_parallel_preflight_failure_emits_typed_report_without_runtime() {
        let dir = std::env::temp_dir().join(format!(
            "mer-pr2c-preflight-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("failure.json");
        let result = run_command(CommandArgs {
            config: PathBuf::from(FROZEN_CONFIG_PATH),
            expected_adapter_name: "software-only-invalid-for-frozen-workload".into(),
            report_out: path.clone(),
            progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig::disabled(),
        })
        .await;
        assert!(result.is_err());
        let original = std::fs::read(&path).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&original).unwrap();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["qualification_pass"], false);
        assert_eq!(json["benchmark_complete"], false);
        assert!(json["failure"].is_object());
        assert!(json["control"].is_null() && json["treatment"].is_null());
        assert!(write_report(&Report::pending("NVIDIA L4".into()), &path).is_err());
        assert_eq!(std::fs::read(path).unwrap(), original);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
