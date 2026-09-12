//! PR2-B-A.1 production qualification for direct physical-install queue
//! staging. Control explicitly forces the legacy Vec writer; treatment uses
//! the ordinary production demand-install path in a fresh runtime.

#[path = "gpu_native_physical_staging_payload_copy.rs"]
pub(crate) mod payload_copy;

#[path = "gpu_native_physical_zero_fill_production.rs"]
pub(crate) mod zero_fill_production;

#[path = "gpu_native_source_to_upload_copy_elision_production.rs"]
pub(crate) mod source_to_upload_production;

#[path = "gpu_native_q4_route_parallel.rs"]
pub(crate) mod q4_route_parallel;

use crate::backend::gpu_native::GpuNativeProductionPhysicalInstallSnapshot;
use crate::backend::{GpuExpertIoSnapshot, GpuExpertMemorySnapshot};
use crate::engine::{
    GpuNativePhysicalInstallStagingQualificationArm,
    GpuNativePhysicalInstallStagingQualificationSnapshot, ProductionDemandSourceSnapshot,
    RoutedExpertExecutionSnapshot,
};
use crate::gpu_native_real_benchmark::{
    BenchmarkFailure, BenchmarkProvenance, BenchmarkReport, EngineStorageSnapshot,
    GpuNativeResidencyDelta, PerRunResult, ProductionConfiguration, RequestEvidence,
};
use crate::gpu_native_residency::GpuNativeTieredResidencySnapshot;
use crate::gpu_native_token_loop::{GpuNativeRecoverySnapshot, GpuNativeTokenLoopSnapshot};
use crate::qualification::{BuildProvenance, QualificationArtifacts};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) const PRODUCTION_SCHEMA: &str = "mer.gpu-native-physical-install-staging.v2";
pub(crate) const PRODUCTION_MODE: &str = "qualify-gpu-native-physical-install-staging-production";
pub(crate) const CONCURRENCY_SCHEMA: &str = "mer.gpu-native-physical-install-concurrency-production.v1";
pub(crate) const CONCURRENCY_MODE: &str = "qualify-gpu-native-physical-install-concurrency-production";
pub(crate) const FROZEN_PROMPT: &str =
    "Write a Rust function that adds two i32 values and returns the result.";
pub(crate) const FROZEN_OUTPUT_TOKENS: usize = 128;
pub(crate) const FROZEN_WARMUP_RUNS: usize = 1;
pub(crate) const FROZEN_MEASURED_RUNS: usize = 3;
pub(crate) const FROZEN_CONFIG_SHA256: &str =
    "33d7cf96328d9c68b0ff45448d91d597d2e3a757cb99e6e61c72998ceabdd056";
pub(crate) const FROZEN_CONFIG_PATH: &str = "/home/randyap8/slice11-qwen3-coder-gpu-native.toml";
const CONCURRENCY_PHYSICAL_INSTALL_TOTAL_DEFINITION: &str = "sum of per-expert post-reservation physical staging plus commit service; control uses the existing ordinary direct-install timer and treatment sums each individual staging service with its ordered commit service; reservation time is reported separately";
const CONCURRENCY_PHYSICAL_ORDERED_COMMIT_DEFINITION: &str = "sum of per-expert post-staging reservation recheck plus logical mapping publication plus arena Resident commit service";
const CONCURRENCY_PHYSICAL_TRANSACTION_DEFINITION: &str = "complete per-set wall-clock from before sequential reservation through parallel staging and ordered commit; this is the primary mechanism-wall diagnostic";
const CONCURRENCY_PARALLEL_STAGE_WALL_DEFINITION: &str = "per-set physical staging critical-path wall after reservation and before ordered commit";
const CONCURRENCY_INDIVIDUAL_STAGE_SUM_DEFINITION: &str = "sum of individual per-expert physical staging service; its ratio to parallel stage wall measures host preparation overlap, not DMA parallelism";

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: PathBuf,
    pub(crate) expected_adapter_name: String,
    pub(crate) report_out: PathBuf,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FrozenWorkload {
    config: &'static str,
    config_sha256: &'static str,
    model_path: &'static str,
    compatibility_path: &'static str,
    model: &'static str,
    quantization: &'static str,
    prompt: &'static str,
    output_tokens: usize,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: &'static str,
    sampling: &'static str,
    expected_adapter_name: String,
    backend: &'static str,
    datadog: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WarmupEvidence {
    run_index: usize,
    generated_tokens: usize,
    generated_token_ids_sha256: String,
    generated_text_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArmWorkEvidence {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed_execution: RoutedExpertExecutionSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_expert_io: GpuExpertIoSnapshot,
    gpu_expert_memory_before: GpuExpertMemorySnapshot,
    gpu_expert_memory_after: GpuExpertMemorySnapshot,
    gpu_native_residency: GpuNativeResidencyDelta,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArmReport {
    arm: GpuNativePhysicalInstallStagingQualificationArm,
    complete: bool,
    failure: Option<BenchmarkFailure>,
    isolated_runtime: bool,
    warmup_results: Vec<WarmupEvidence>,
    warmup_source: Option<GpuNativePhysicalInstallStagingQualificationSnapshot>,
    warmup_production_physical_install: Option<GpuNativeProductionPhysicalInstallSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup_production: Option<ProductionDemandSourceSnapshot>,
    warmup_ram_cache_state_sha256: Option<String>,
    warmup_work: Option<ArmWorkEvidence>,
    source: Option<GpuNativePhysicalInstallStagingQualificationSnapshot>,
    production_physical_install: Option<GpuNativeProductionPhysicalInstallSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production: Option<ProductionDemandSourceSnapshot>,
    work: Option<ArmWorkEvidence>,
    benchmark: BenchmarkReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Reconciliation {
    generated_tokens_exact: bool,
    generated_token_hashes_exact: bool,
    warmup_token_hashes_exact: bool,
    warmup_source_requests_exact: bool,
    warmup_ram_source_hits_exact: bool,
    warmup_ram_source_misses_exact: bool,
    warmup_nvme_reads_exact: bool,
    warmup_nvme_bytes_exact: bool,
    warmup_ram_cache_inserts_exact: bool,
    warmup_ram_cache_evictions_exact: bool,
    warmup_ordered_ram_insert_ids_exact: bool,
    warmup_ordered_ram_eviction_ids_exact: bool,
    warmup_physical_missing_sequence_exact: bool,
    warmup_demand_source_sequence_exact: bool,
    warmup_ram_cache_state_exact: bool,
    warmup_all_speculative_work_zero: bool,
    selected_route_sequence_exact: bool,
    selected_route_counts_exact: bool,
    physical_missing_sequence_exact: bool,
    physical_missing_counts_exact: bool,
    demand_source_request_sequence_exact: bool,
    demand_source_requests_exact: bool,
    ram_source_hits_exact: bool,
    ram_source_misses_exact: bool,
    demand_nvme_reads_exact: bool,
    demand_nvme_bytes_exact: bool,
    logical_admissions_exact: bool,
    ram_to_vram_installs_exact: bool,
    ram_to_vram_bytes_exact: bool,
    physical_evictions_exact: bool,
    physical_reinstalls_exact: bool,
    physical_victim_sequence_exact: bool,
    physical_residency_identity_stream_exact: bool,
    physical_install_attempts_exact: bool,
    physical_install_completions_exact: bool,
    physical_slot_bytes_staged_exact: bool,
    mapping_publications_exact: bool,
    mapping_unpublications_exact: bool,
    vram_hits_exact: bool,
    vram_misses_exact: bool,
    residency_miss_attempts_exact: bool,
    residency_services_exact: bool,
    recovery_segments_exact: bool,
    miss_boundaries_exact: bool,
    recovery_semantics_exact: bool,
    full_token_replay_zero: bool,
    fatal_and_no_progress_zero: bool,
    ram_cache_inserts_exact: bool,
    ram_cache_evictions_exact: bool,
    ordered_ram_insert_ids_exact: bool,
    ordered_ram_eviction_ids_exact: bool,
    all_speculative_work_zero: bool,
    primary_pool_capacity_exact: bool,
    all_invariants_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BehavioralGate {
    generated_token_parity_exact: bool,
    route_parity_exact: bool,
    physical_missing_sequence_exact: bool,
    full_token_replay_zero: bool,
    fatal_and_no_progress_zero: bool,
    speculation_zero: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WorkEquivalenceGate {
    source_requests_exact: bool,
    ram_hits_and_misses_exact: bool,
    nvme_reads_exact: bool,
    demand_nvme_bytes_exact: bool,
    ram_inserts_and_evictions_exact: bool,
    logical_admissions_exact: bool,
    demand_h2d_installs_exact: bool,
    demand_h2d_bytes_exact: bool,
    physical_evictions_exact: bool,
    physical_victim_order_exact: bool,
    physical_residency_identity_stream_exact: bool,
    slot_installs_and_mapping_publications_exact: bool,
    mapping_unpublications_exact: bool,
    physical_reinstalls_exact: bool,
    recovery_and_miss_boundaries_exact: bool,
    deterministic_ram_cache_order_exact: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MetricComparison {
    control: f64,
    treatment: f64,
    delta_percent: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionReconciliation {
    common: Reconciliation,
    warmup_production_cache_reservation_leaks_zero: bool,
    warmup_production_batch_commit_violations_zero: bool,
    warmup_stale_singleflight_entries_zero: bool,
    production_cache_reservation_leaks_zero: bool,
    production_batch_commit_violations_zero: bool,
    stale_singleflight_entries_zero: bool,
    all_invariants_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ConcurrencyReconciliation {
    production_and_work: ProductionReconciliation,
    warmup_generated_text_hashes_exact: bool,
    measured_generated_text_hashes_exact: bool,
    all_invariants_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionMechanismGate {
    warmup_mechanism_reconciled: bool,
    control_vec_materializations_gt_zero: bool,
    control_direct_staging_writes_zero: bool,
    treatment_direct_staging_writes_gt_zero: bool,
    treatment_vec_materializations_zero: bool,
    treatment_install_completions_gt_zero: bool,
    attempts_equal_completions: bool,
    staged_bytes_exact: bool,
    mapping_publications_exact: bool,
    treatment_staging_failures_zero: bool,
    control_production_direct_staging_successes_zero: bool,
    treatment_production_direct_staging_successes_gt_zero: bool,
    treatment_direct_allocation_fallbacks_zero: bool,
    treatment_production_install_failures_zero: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionGates {
    behavioral: BehavioralGate,
    work_equivalence: WorkEquivalenceGate,
    mechanism: ProductionMechanismGate,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionArmPerformance {
    decode_tps: f64,
    end_to_end_generated_tps: f64,
    mean_request_wall_seconds: f64,
    source_acquisition_wall_us: u64,
    logical_demand_admission_us: u64,
    physical_demand_install_us: u64,
    physical_slot_prepare_us: u64,
    physical_queue_staging_us: u64,
    mapping_publication_us: u64,
    physical_install_total_us: u64,
    total_residency_service_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionPerformanceComparison {
    control: ProductionArmPerformance,
    treatment: ProductionArmPerformance,
    decode_tps: MetricComparison,
    end_to_end_generated_tps: MetricComparison,
    mean_request_wall_seconds: MetricComparison,
    source_acquisition_wall_us: MetricComparison,
    logical_demand_admission_us: MetricComparison,
    physical_demand_install_us: MetricComparison,
    physical_slot_prepare_us: MetricComparison,
    physical_queue_staging_us: MetricComparison,
    mapping_publication_us: MetricComparison,
    physical_install_total_us: MetricComparison,
    total_residency_service_us: MetricComparison,
    performance_result: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionQualificationReport {
    schema: &'static str,
    mode: &'static str,
    production_physical_install_changed: bool,
    control_forces_legacy_full_slot_vec: bool,
    treatment_uses_ordinary_production_path: bool,
    normal_production_uses_direct_queue_staging: bool,
    allocation_only_fallback_implemented: bool,
    allocation_only_fallback_decision: &'static str,
    both_arms_use_ordinary_production_source: bool,
    pinned_wgpu_api_audit: WgpuApiAudit,
    timing_definitions: TimingDefinitions,
    benchmark_complete: bool,
    qualification_pass: bool,
    performance_result: &'static str,
    failure: Option<BenchmarkFailure>,
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    control: Option<ArmReport>,
    treatment: Option<ArmReport>,
    reconciliation: Option<ProductionReconciliation>,
    gates: Option<ProductionGates>,
    performance: Option<ProductionPerformanceComparison>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TimingDefinitions {
    physical_slot_prepare_us: &'static str,
    physical_queue_staging_us: &'static str,
    mapping_publication_us: &'static str,
    physical_install_total_us: &'static str,
    physical_demand_install_us: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct ConcurrencyTimingDefinitions {
    #[serde(flatten)]
    common: TimingDefinitions,
    physical_parallel_stage_wall_us: &'static str,
    sum_individual_physical_stage_us: &'static str,
    physical_ordered_commit_us: &'static str,
    physical_install_transaction_us: &'static str,
}

fn concurrency_timing_definitions() -> ConcurrencyTimingDefinitions {
    ConcurrencyTimingDefinitions {
        common: TimingDefinitions {
            physical_slot_prepare_us: "sum of qualification-only per-expert canonical validation plus arm-specific physical-slot fill time",
            physical_queue_staging_us: "sum of qualification-only Queue::write_buffer_with view acquisition plus Drop scheduling time",
            mapping_publication_us: "ordered logical mapping Queue::write_buffer time after all treatment staging jobs finish",
            physical_install_total_us: CONCURRENCY_PHYSICAL_INSTALL_TOTAL_DEFINITION,
            physical_demand_install_us: "existing aggregate Engine wall timer around the complete residency-manager demand-set transaction",
        },
        physical_parallel_stage_wall_us: CONCURRENCY_PARALLEL_STAGE_WALL_DEFINITION,
        sum_individual_physical_stage_us: CONCURRENCY_INDIVIDUAL_STAGE_SUM_DEFINITION,
        physical_ordered_commit_us: CONCURRENCY_PHYSICAL_ORDERED_COMMIT_DEFINITION,
        physical_install_transaction_us: CONCURRENCY_PHYSICAL_TRANSACTION_DEFINITION,
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WgpuApiAudit {
    requested_version: &'static str,
    resolved_version: &'static str,
    supported_api: &'static str,
    exact_destination_range: bool,
    none_distinguishes_allocation_from_validation: bool,
    schedules_at_next_submit: bool,
    adds_queue_submit: bool,
    adds_device_poll: bool,
    adds_map_async: bool,
    adds_readback: bool,
    uses_unsafe_internals: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ConcurrencyArmReport {
    #[serde(flatten)]
    common: ArmReport,
    warmup_mechanism:
        Option<crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot>,
    mechanism: Option<crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot>,
}

#[derive(Clone, Debug, Serialize)]
struct ConcurrencyMechanismGate {
    warmup_mechanism_reconciled: bool,
    control_production_parallel_staging_sets_zero: bool,
    treatment_ordinary_production_path_exercised: bool,
    treatment_production_parallel_staging_sets_gt_zero: bool,
    treatment_production_counters_reconcile: bool,
    control_direct_staging_writes_gt_zero: bool,
    control_full_slot_vec_materializations_zero: bool,
    control_parallel_staging_sets_zero: bool,
    control_max_in_flight_lte_one: bool,
    treatment_direct_staging_writes_gt_zero: bool,
    treatment_full_slot_vec_materializations_zero: bool,
    treatment_parallel_staging_sets_gt_zero: bool,
    treatment_parallel_staging_experts_gt_zero: bool,
    treatment_max_in_flight_gte_two: bool,
    reservation_failures_zero: bool,
    physical_stage_failures_zero: bool,
    ordered_commit_failures_zero: bool,
    ordered_commit_violations_zero: bool,
    unpublished_physical_writes_after_failure_zero: bool,
    attempts_and_completions_reconcile: bool,
    staged_bytes_exact: bool,
    mapping_publications_exact: bool,
    mapping_unpublications_exact: bool,
    victim_sequence_exact: bool,
    reservation_identity_exact: bool,
    ordered_physical_identity_exact: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ConcurrencyGates {
    behavioral: BehavioralGate,
    warmup_generated_text_hashes_exact: bool,
    measured_generated_text_hashes_exact: bool,
    text_hashes_passed: bool,
    work_equivalence: WorkEquivalenceGate,
    mechanism: ConcurrencyMechanismGate,
}

#[derive(Clone, Debug, Serialize)]
struct ConcurrencyPerformanceComparison {
    #[serde(flatten)]
    common: ProductionPerformanceComparison,
    physical_reservation_us: MetricComparison,
    physical_parallel_stage_wall_us: MetricComparison,
    sum_individual_physical_stage_us: MetricComparison,
    physical_ordered_commit_us: MetricComparison,
    physical_install_transaction_us: MetricComparison,
    control_stage_overlap_ratio: f64,
    treatment_stage_overlap_ratio: f64,
    overlap_ratio_definition: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct PinnedWgpuConcurrencyAudit {
    requested_version: &'static str,
    resolved_version: &'static str,
    native_queue_send_sync: bool,
    native_buffer_send_sync: bool,
    native_queue_write_buffer_view_send_sync: bool,
    validates_destination_before_staging_creation: bool,
    drop_schedules_buffer_write: bool,
    scheduled_writes_begin_at_next_submit: bool,
    concurrent_calls_use_internal_pending_write_lock: bool,
    distinct_reserved_slot_ranges_do_not_alias: bool,
    adds_queue_submit: bool,
    adds_device_poll: bool,
    adds_map_async: bool,
    adds_readback: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RayonIntegrationAudit {
    existing_process_wide_pool: bool,
    creates_new_thread_pool: bool,
    spawns_os_threads_per_set: bool,
    uses_tokio_spawn_blocking_per_expert: bool,
    changes_pool_sizing: bool,
    changes_rayon_num_threads: bool,
    parallel_width_is_exact_install_set_width: bool,
}

#[derive(Clone, Debug, Serialize)]
struct QwenPhysicalSlotGeometry {
    d_model: usize,
    d_ff: usize,
    q4_block_elements: usize,
    q4_block_bytes: usize,
    logical_expert_bytes: usize,
    slot_stride_bytes: usize,
    physical_tail_bytes: usize,
    destination_fill_zero_unchanged: bool,
}

#[derive(Clone, Debug, Serialize)]
struct PhysicalInstallConcurrencyQualificationReport {
    schema: &'static str,
    mode: &'static str,
    production_physical_install_concurrency_changed: bool,
    normal_production_uses_concurrent_physical_staging: bool,
    control_forces_sequential_direct_staging: bool,
    treatment_uses_ordinary_production_path: bool,
    speculative_physical_install_changed: bool,
    both_arms_use_ordinary_production_source: bool,
    victim_policy_changed: bool,
    slot_assignment_policy_changed: bool,
    mapping_publication_order_changed: bool,
    source_acquisition_changed: bool,
    expert_compute_changed: bool,
    adds_queue_submit: bool,
    adds_device_poll: bool,
    adds_map_async: bool,
    adds_readback: bool,
    wgpu_version_changed: bool,
    rayon_pool_sizing_changed: bool,
    pinned_wgpu_audit: PinnedWgpuConcurrencyAudit,
    rayon_integration: RayonIntegrationAudit,
    qwen_physical_slot_geometry: QwenPhysicalSlotGeometry,
    timing_definitions: ConcurrencyTimingDefinitions,
    benchmark_complete: bool,
    qualification_pass: bool,
    performance_result: &'static str,
    failure: Option<BenchmarkFailure>,
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    control: Option<ConcurrencyArmReport>,
    treatment: Option<ConcurrencyArmReport>,
    reconciliation: Option<ConcurrencyReconciliation>,
    gates: Option<ConcurrencyGates>,
    performance: Option<ConcurrencyPerformanceComparison>,
}

struct Prepared {
    spec: crate::ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_ids: Vec<u32>,
    resolved_config_sha256: String,
    provenance: BenchmarkProvenance,
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    request: RequestEvidence,
    production_configuration: ProductionConfiguration,
}

#[derive(Clone)]
struct ArmStart {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed: RoutedExpertExecutionSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_io: GpuExpertIoSnapshot,
    gpu_memory: GpuExpertMemorySnapshot,
    residency: GpuNativeTieredResidencySnapshot,
}

impl ArmStart {
    fn capture(runtime: &crate::BenchRealRuntime) -> Result<Self, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "startup",
                "missing-gpu-native-token-loop",
                "PR2-B-A arm did not construct the authoritative GPU-native token loop",
            )
        })?;
        Ok(Self {
            token_loop: token_loop.snapshot(),
            recovery: token_loop.recovery_snapshot(),
            routed: runtime.engine.routed_expert_execution_snapshot(),
            engine_storage: EngineStorageSnapshot::from_runtime(runtime),
            gpu_io: runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-io-snapshot",
                    "PR2-B-A arm did not expose GPU expert I/O counters",
                )
            })?,
            gpu_memory: runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-memory-snapshot",
                    "PR2-B-A arm did not expose GPU expert memory counters",
                )
            })?,
            residency: runtime
                .engine
                .gpu_native_residency_snapshot()
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "startup",
                        "missing-gpu-native-residency-snapshot",
                        "PR2-B-A arm did not expose GPU-native residency counters",
                    )
                })?,
        })
    }

    fn finish(
        self,
        runtime: &crate::BenchRealRuntime,
    ) -> Result<ArmWorkEvidence, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-native-token-loop",
                "PR2-B-A token loop disappeared after measurement",
            )
        })?;
        let engine_storage_after = EngineStorageSnapshot::from_runtime(runtime);
        let gpu_io_after = runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-io-snapshot",
                "PR2-B-A GPU expert I/O counters disappeared",
            )
        })?;
        let gpu_memory_after = runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-memory-snapshot",
                "PR2-B-A GPU expert memory counters disappeared",
            )
        })?;
        let residency_after = runtime
            .engine
            .gpu_native_residency_snapshot()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-gpu-native-residency-snapshot",
                    "PR2-B-A GPU-native residency counters disappeared",
                )
            })?;
        Ok(ArmWorkEvidence {
            token_loop: crate::gpu_native_real_benchmark::token_loop_delta(
                self.token_loop,
                token_loop.snapshot(),
            )?,
            recovery: crate::gpu_native_real_benchmark::recovery_delta(
                self.recovery,
                token_loop.recovery_snapshot(),
            )?,
            routed_execution: crate::gpu_native_real_benchmark::routed_delta(
                self.routed,
                runtime.engine.routed_expert_execution_snapshot(),
            )?,
            engine_storage: engine_storage_after.checked_delta(self.engine_storage)?,
            gpu_expert_io: crate::gpu_native_real_benchmark::gpu_io_delta(
                self.gpu_io,
                gpu_io_after,
            )?,
            gpu_expert_memory_before: self.gpu_memory,
            gpu_expert_memory_after: gpu_memory_after,
            gpu_native_residency: crate::gpu_native_real_benchmark::gpu_native_residency_delta(
                &self.residency,
                &residency_after,
            )?,
        })
    }
}

fn frozen_workload(expected_adapter_name: String) -> FrozenWorkload {
    FrozenWorkload {
        config: FROZEN_CONFIG_PATH,
        config_sha256: FROZEN_CONFIG_SHA256,
        model_path: "/mnt/localssd/data/qwen3-coder-q4",
        compatibility_path: "/mnt/mer-local-ssd/mer/qwen3-coder-q4",
        model: "Qwen3-Coder-30B-A3B-Instruct",
        quantization: "pure Q4_0",
        prompt: FROZEN_PROMPT,
        output_tokens: FROZEN_OUTPUT_TOKENS,
        warmup_runs: FROZEN_WARMUP_RUNS,
        measured_runs: FROZEN_MEASURED_RUNS,
        cache_reset: "keep",
        sampling: "greedy",
        expected_adapter_name,
        backend: "WGPU/Vulkan",
        datadog: "off",
    }
}

fn validate_isolation_config(cfg: &crate::config::Config) -> Result<(), BenchmarkFailure> {
    let predictive = &cfg.predictive;
    if cfg.storage.predict_fanout != 0
        || predictive.locality_enabled
        || predictive.speculator_enabled
        || predictive.affinity_enabled
        || predictive.pregate_enabled
        || predictive.static_residency_fraction != 0.0
        || predictive.static_residency_profile.is_some()
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "speculation-must-be-disabled",
            "PR2-B-A requires storage.predict_fanout=0 and every locality/speculator/affinity/pregate/static-residency arm disabled",
        ));
    }
    if cfg.storage.pin_after_observations != 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "observation-pinning-must-be-disabled",
            format!(
                "PR2-B-A requires storage.pin_after_observations=0; observed {}",
                cfg.storage.pin_after_observations
            ),
        ));
    }
    if predictive.cost_aware_eviction {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cost-aware-eviction-must-be-disabled",
            "PR2-B-A requires predictive.cost_aware_eviction=false so victim ordering is strict LRU",
        ));
    }
    if cfg.storage.packed_blob.is_some() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "packed-blob-must-be-disabled",
            "PR2-B-A requires storage.packed_blob to be unset so the treatment measures ordinary per-file reads",
        ));
    }
    if cfg.storage.packed_manifest.is_some() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "packed-manifest-must-be-disabled",
            "PR2-B-A requires storage.packed_manifest to be unset so the treatment measures ordinary per-file reads",
        ));
    }
    Ok(())
}

fn prepare(args: &CommandArgs) -> Result<Prepared, Box<dyn std::error::Error>> {
    if args.config != Path::new(FROZEN_CONFIG_PATH) {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-frozen-config-path",
            format!(
                "PR2-B-A requires config {FROZEN_CONFIG_PATH}; observed {}",
                args.config.display()
            ),
        )
        .into());
    }
    if args.expected_adapter_name.trim().is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "missing-expected-adapter",
            "PR2-B-A requires an exact nonempty adapter name",
        )
        .into());
    }
    let build = BuildProvenance::embedded();
    crate::gpu_native_real_benchmark::validate_preflight_provenance(&build)?;
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    validate_isolation_config(&cfg)?;
    let (artifacts, artifact_errors): (QualificationArtifacts, Vec<String>) =
        crate::qualification_artifacts(&args.config, &cfg);
    crate::gpu_native_real_benchmark::validate_artifacts(&artifacts, &artifact_errors)?;
    let observed_config_sha256 = artifacts
        .config
        .as_ref()
        .map(|digest| digest.sha256.as_str())
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "config-hash-unavailable",
                "PR2-B-A could not hash the frozen config",
            )
        })?;
    if observed_config_sha256 != FROZEN_CONFIG_SHA256 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-config-hash-mismatch",
            format!(
                "PR2-B-A config SHA256 {observed_config_sha256} did not match {FROZEN_CONFIG_SHA256}"
            ),
        )
        .into());
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_real_benchmark::validate_expert_metadata(&expert_metadata)?;
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-model-identity",
            format!("PR2-B-A requires exact Qwen3-Coder 30B-A3B Q4_0; observed {model_identity:?}"),
        )
        .into());
    }
    let resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let prompt_ids = tokenizer.encode(FROZEN_PROMPT)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "the frozen PR2-B-A prompt encoded to zero tokens",
        )
        .into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)
        .map_err(|error| {
            BenchmarkFailure::new(
                "preflight",
                "executable-provenance-unavailable",
                format!("failed to canonicalize {}: {error}", executable.display()),
            )
        })?
        .display()
        .to_string();
    if !crate::gpu_native_real_benchmark::is_hex(&executable_sha256, 64)
        || !crate::gpu_native_real_benchmark::is_hex(&resolved_config_sha256, 64)
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "PR2-B-A executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let production_configuration =
        ProductionConfiguration::from_config(&spec.cfg, &expert_metadata);
    Ok(Prepared {
        spec,
        tokenizer,
        prompt_ids: prompt_ids.clone(),
        resolved_config_sha256: resolved_config_sha256.clone(),
        provenance: BenchmarkProvenance {
            build,
            executable_canonical_path,
            executable_sha256,
            resolved_config_sha256,
            artifacts,
            expert_metadata,
        },
        model_identity,
        request: RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(FROZEN_PROMPT.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: FROZEN_OUTPUT_TOKENS,
            greedy: true,
        },
        production_configuration,
    })
}

fn benchmark_report(prepared: &Prepared) -> BenchmarkReport {
    BenchmarkReport::new(
        prepared.provenance.clone(),
        prepared.model_identity.clone(),
        prepared.request.clone(),
        crate::BenchRealCacheReset::Keep,
        FROZEN_WARMUP_RUNS,
        FROZEN_MEASURED_RUNS,
        prepared.production_configuration.clone(),
    )
}

#[derive(Clone, Copy)]
enum PhysicalInstallQualificationRun {
    Staging(GpuNativePhysicalInstallStagingQualificationArm),
    Concurrency(crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm),
    ZeroFillProduction(crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm),
    SourceToUpload(crate::gpu_native_source_upload::Arm),
}

struct PhysicalInstallArmRun {
    source_decomposition: Option<crate::gpu_native_source_path_decomposition::StoreSnapshot>,
    common: ArmReport,
    warmup_upload: Option<crate::gpu_native_source_upload::Snapshot>,
    upload: Option<crate::gpu_native_source_upload::Snapshot>,
    warmup_concurrency:
        Option<crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot>,
    concurrency: Option<crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot>,
}

fn concurrency_common_snapshot(
    source: &crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot,
) -> GpuNativePhysicalInstallStagingQualificationSnapshot {
    GpuNativePhysicalInstallStagingQualificationSnapshot {
        arm: match source.arm {
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::Control
            | crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::ConcurrentFullZeroControl
            | crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::SourceToUploadControl => {
                GpuNativePhysicalInstallStagingQualificationArm::Control
            }
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::Treatment
            | crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::ProductionNoZeroFillTreatment
            | crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::SourceToUploadTreatment => {
                GpuNativePhysicalInstallStagingQualificationArm::Treatment
            }
        },
        qualification_telemetry_only: true,
        production_physical_install_changed: true,
        control_forces_legacy_full_slot_vec: false,
        treatment_uses_ordinary_production_path: source.treatment_uses_ordinary_production_path,
        normal_production_uses_direct_queue_staging: true,
        single_request_stream: source.single_request_stream,
        overlapping_demand_sets: source.overlapping_demand_sets,
        primary_pool_capacity: source.primary_pool_capacity,
        shadow_pool_capacity: source.shadow_pool_capacity,
        demand_sets: source.demand_sets,
        physical_missing_experts: source.physical_missing_experts,
        physical_probe_us: source.physical_probe_us,
        source_sets: source.source_sets,
        source_experts: source.source_experts,
        demand_source_requests: source.demand_source_requests,
        source_ram_hits: source.source_ram_hits,
        source_ram_misses: source.source_ram_misses,
        source_nvme_reads: source.source_nvme_reads,
        source_nvme_bytes: source.source_nvme_bytes,
        source_acquisition_wall_us: source.source_acquisition_wall_us,
        logical_demand_admission_us: source.logical_demand_admission_us,
        physical_demand_install_us: source.physical_demand_install_us,
        total_residency_service_us: source.total_residency_service_us,
        ram_cache_inserts: source.ram_cache_inserts,
        ram_cache_evictions: source.ram_cache_evictions,
        selected_route_ids_sha256: source.selected_route_ids_sha256.clone(),
        physical_missing_ids_sha256: source.physical_missing_ids_sha256.clone(),
        demand_source_request_ids_sha256: source.demand_source_request_ids_sha256.clone(),
        demand_ram_insert_ids_sha256: source.demand_ram_insert_ids_sha256.clone(),
        demand_ram_eviction_ids_sha256: source.demand_ram_eviction_ids_sha256.clone(),
        physical_victim_ids_sha256: source.physical_victim_ids_sha256.clone(),
        physical_residency_identity_sha256: source.physical_residency_identity_sha256.clone(),
        physical_install_attempts: source.physical_install_attempts,
        physical_install_completions: source.physical_install_completions,
        full_slot_vec_materializations: source.full_slot_vec_materializations,
        direct_staging_writes: source.direct_staging_writes,
        direct_staging_failures: source.direct_staging_failures,
        physical_slot_bytes_staged: source.physical_slot_bytes_staged,
        mapping_publications: source.mapping_publications,
        mapping_unpublications: source.mapping_unpublications,
        physical_slot_prepare_us: source.physical_slot_prepare_us,
        physical_slot_validation_us: source.physical_slot_validation_us,
        physical_slot_epoch_write_us: source.physical_slot_epoch_write_us,
        physical_slot_payload_copy_us: source.physical_slot_payload_copy_us,
        physical_slot_prepare_residual_us: source.physical_slot_prepare_residual_us,
        physical_slot_subphase_observations: source.physical_slot_subphase_observations,
        physical_queue_staging_us: source.physical_queue_staging_us,
        mapping_publication_us: source.mapping_publication_us,
        physical_install_total_us: source.physical_install_total_us,
    }
}

async fn run_physical_install_arm(
    prepared: &Prepared,
    args: &CommandArgs,
    run: PhysicalInstallQualificationRun,
) -> Result<PhysicalInstallArmRun, BenchmarkFailure> {
    run_physical_install_arm_inner(prepared, args, run, None).await
}

async fn run_physical_install_arm_inner(
    prepared: &Prepared,
    args: &CommandArgs,
    run: PhysicalInstallQualificationRun,
    diagnostic_mode: Option<&'static str>,
) -> Result<PhysicalInstallArmRun, BenchmarkFailure> {
    use crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm as ConcurrentArm;
    let (arm_name, arm) = match run {
        PhysicalInstallQualificationRun::SourceToUpload(arm) => match arm {
            crate::gpu_native_source_upload::Arm::Control => (
                "control",
                GpuNativePhysicalInstallStagingQualificationArm::Control,
            ),
            crate::gpu_native_source_upload::Arm::Treatment => (
                "treatment",
                GpuNativePhysicalInstallStagingQualificationArm::Treatment,
            ),
        },
        PhysicalInstallQualificationRun::Staging(arm) => match arm {
            GpuNativePhysicalInstallStagingQualificationArm::Control => ("control", arm),
            GpuNativePhysicalInstallStagingQualificationArm::Treatment => ("treatment", arm),
        },
        PhysicalInstallQualificationRun::Concurrency(arm)
        | PhysicalInstallQualificationRun::ZeroFillProduction(arm) => match arm {
            ConcurrentArm::Control
            | ConcurrentArm::ConcurrentFullZeroControl
            | ConcurrentArm::SourceToUploadControl => (
                "control",
                GpuNativePhysicalInstallStagingQualificationArm::Control,
            ),
            ConcurrentArm::Treatment
            | ConcurrentArm::ProductionNoZeroFillTreatment
            | ConcurrentArm::SourceToUploadTreatment => (
                "treatment",
                GpuNativePhysicalInstallStagingQualificationArm::Treatment,
            ),
        },
    };
    let mode_name = diagnostic_mode.unwrap_or(match run {
        PhysicalInstallQualificationRun::Staging(_) => PRODUCTION_MODE,
        PhysicalInstallQualificationRun::Concurrency(_) => CONCURRENCY_MODE,
        PhysicalInstallQualificationRun::ZeroFillProduction(_) => zero_fill_production::MODE,
        PhysicalInstallQualificationRun::SourceToUpload(_) => source_to_upload_production::MODE,
    });
    let decomposition_capacity =
        if diagnostic_mode == Some(crate::gpu_native_source_path_decomposition::MODE) {
            if !matches!(run, PhysicalInstallQualificationRun::SourceToUpload(_)) {
                return Err(BenchmarkFailure::new(
                    "startup",
                    "invalid-decomposition-arm",
                    "HMA-1D requires production-v2 source/upload arms",
                ));
            }
            Some(
                prepared
                    .prompt_ids
                    .len()
                    .checked_add(FROZEN_OUTPUT_TOKENS)
                    .and_then(|n| n.checked_mul(48))
                    .and_then(|n| n.checked_mul(crate::gpu_native_source_path_decomposition::WIDTH))
                    .and_then(|n| n.checked_mul(FROZEN_WARMUP_RUNS + FROZEN_MEASURED_RUNS))
                    .ok_or_else(|| {
                        BenchmarkFailure::new(
                            "startup",
                            "decomposition-capacity-overflow",
                            "frozen workload source bound overflow",
                        )
                    })?,
            )
        } else {
            None
        };
    let mut benchmark = benchmark_report(prepared);
    let runtime = crate::gpu_native_real_benchmark::construct_runtime(
        &prepared.spec,
        prepared.tokenizer.clone(),
        arm_name,
        None,
        &mut benchmark,
    )
    .await?;
    let enable_result = match run {
        PhysicalInstallQualificationRun::SourceToUpload(arm) => runtime
            .engine
            .enable_gpu_native_source_upload_qualification(arm),
        PhysicalInstallQualificationRun::Staging(arm) => runtime
            .engine
            .enable_gpu_native_physical_install_staging_qualification(arm),
        PhysicalInstallQualificationRun::Concurrency(arm)
        | PhysicalInstallQualificationRun::ZeroFillProduction(arm) => runtime
            .engine
            .enable_gpu_native_physical_install_concurrency_qualification(arm),
    };
    let mut source_observer = None;
    let enable_result = enable_result.and_then(|()| {
        if let Some(capacity) = decomposition_capacity {
            source_observer = Some(runtime.engine.enable_source_decomposition(capacity)?);
        }
        Ok(())
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
            if let Some(observer) = &source_observer {
                observer.begin_request(
                    crate::gpu_native_source_path_decomposition::Phase::Warmup,
                    index,
                );
            }
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
            if let Some(observer) = &source_observer {
                observer.finish_request(
                    runtime
                        .engine
                        .gpu_native_source_upload_snapshot()
                        .map(|s| s.ordered_nvme_ids_sha256),
                );
            }
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

    let warmup_upload = runtime.engine.gpu_native_source_upload_snapshot();
    let mut warmup_source = None;
    let mut warmup_concurrency = None;
    let mut warmup_production_physical_install = None;
    let mut warmup_production = None;
    let mut warmup_ram_cache_state_sha256 = None;
    let mut warmup_work = None;
    if execution_failure.is_none() {
        match run {
            PhysicalInstallQualificationRun::Staging(_) => {
                warmup_source = runtime
                    .engine
                    .gpu_native_physical_install_staging_qualification_snapshot();
            }
            PhysicalInstallQualificationRun::Concurrency(_)
            | PhysicalInstallQualificationRun::ZeroFillProduction(_)
            | PhysicalInstallQualificationRun::SourceToUpload(_) => {
                warmup_concurrency = runtime
                    .engine
                    .gpu_native_physical_install_concurrency_qualification_snapshot();
                warmup_source = warmup_concurrency.as_ref().map(concurrency_common_snapshot);
            }
        }
        if warmup_source.is_none() {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "missing-warmup-source-snapshot",
                "PR2-B-A qualification source counters disappeared after warmup",
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
                "PR2-B-A.1 production physical-install counters disappeared after warmup",
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
                "PR2-B-A RAM-cache state could not be hashed before the warmup counter reset",
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
            if let Some(observer) = &source_observer {
                observer.begin_request(
                    crate::gpu_native_source_path_decomposition::Phase::Measured,
                    index,
                );
            }
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
            if let Some(observer) = &source_observer {
                observer.finish_request(
                    runtime
                        .engine
                        .gpu_native_source_upload_snapshot()
                        .map(|s| s.ordered_nvme_ids_sha256),
                );
            }
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

    let (source, concurrency) = match run {
        PhysicalInstallQualificationRun::Staging(_) => (
            runtime
                .engine
                .gpu_native_physical_install_staging_qualification_snapshot(),
            None,
        ),
        PhysicalInstallQualificationRun::Concurrency(_)
        | PhysicalInstallQualificationRun::ZeroFillProduction(_)
        | PhysicalInstallQualificationRun::SourceToUpload(_) => {
            let concurrency = runtime
                .engine
                .gpu_native_physical_install_concurrency_qualification_snapshot();
            (
                concurrency.as_ref().map(concurrency_common_snapshot),
                concurrency,
            )
        }
    };
    let upload = runtime.engine.gpu_native_source_upload_snapshot();
    let production_physical_install = runtime.engine.production_physical_install_snapshot();
    if execution_failure.is_none() && production_physical_install.is_none() {
        execution_failure = Some(BenchmarkFailure::new(
            "postcondition",
            "missing-production-physical-install-snapshot",
            "PR2-B-A.1 production physical-install counters disappeared after measurement",
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
    Ok(PhysicalInstallArmRun {
        source_decomposition: source_observer.map(|observer| observer.snapshot()),
        warmup_upload,
        upload,
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
        warmup_concurrency,
        concurrency,
    })
}

async fn run_arm(
    prepared: &Prepared,
    args: &CommandArgs,
    arm: GpuNativePhysicalInstallStagingQualificationArm,
) -> Result<ArmReport, BenchmarkFailure> {
    Ok(run_physical_install_arm(
        prepared,
        args,
        PhysicalInstallQualificationRun::Staging(arm),
    )
    .await?
    .common)
}

fn generated_results(arm: &ArmReport) -> &[PerRunResult] {
    &arm.benchmark.per_run_results
}

fn recovery_semantics_equal(a: GpuNativeRecoverySnapshot, b: GpuNativeRecoverySnapshot) -> bool {
    a.resume_attempts == b.resume_attempts
        && a.recovery_segments == b.recovery_segments
        && a.checkpoint_captures == b.checkpoint_captures
        && a.checkpoint_restores == b.checkpoint_restores
        && a.full_token_replay_attempts == b.full_token_replay_attempts
        && a.layers_encoded == b.layers_encoded
        && a.attention_layers_reexecuted == b.attention_layers_reexecuted
        && a.expert_layers_reexecuted == b.expert_layers_reexecuted
        && a.invalid_tail_layers_encoded == b.invalid_tail_layers_encoded
}

fn arm_all_speculative_work_zero(work: &ArmWorkEvidence) -> bool {
    work.gpu_native_residency.speculative_requests == 0
        && work.gpu_native_residency.speculative_vram_hits == 0
        && work.gpu_native_residency.speculative_ram_to_vram_installs == 0
        && work
            .gpu_native_residency
            .speculative_dropped_capacity_or_pressure
            == 0
        && work.engine_storage.prefetch_completed == 0
}

fn reconcile(control: &ArmReport, treatment: &ArmReport) -> Reconciliation {
    let c = control.work.as_ref().expect("complete control work");
    let t = treatment.work.as_ref().expect("complete treatment work");
    let cs = control.source.as_ref().expect("complete control source");
    let ts = treatment
        .source
        .as_ref()
        .expect("complete treatment source");
    let cws = control
        .warmup_source
        .as_ref()
        .expect("complete control warmup source");
    let tws = treatment
        .warmup_source
        .as_ref()
        .expect("complete treatment warmup source");
    let cww = control
        .warmup_work
        .as_ref()
        .expect("complete control warmup work");
    let tww = treatment
        .warmup_work
        .as_ref()
        .expect("complete treatment warmup work");
    let generated_tokens_exact = generated_results(control)
        .iter()
        .map(|run| run.generated_tokens)
        .eq(generated_results(treatment)
            .iter()
            .map(|run| run.generated_tokens));
    let generated_token_hashes_exact = generated_results(control)
        .iter()
        .map(|run| &run.generated_token_ids_sha256)
        .eq(generated_results(treatment)
            .iter()
            .map(|run| &run.generated_token_ids_sha256));
    let warmup_token_hashes_exact = control
        .warmup_results
        .iter()
        .map(|run| &run.generated_token_ids_sha256)
        .eq(treatment
            .warmup_results
            .iter()
            .map(|run| &run.generated_token_ids_sha256));
    let selected_route_sequence_exact =
        cs.selected_route_ids_sha256 == ts.selected_route_ids_sha256;
    let selected_route_counts_exact =
        c.routed_execution.selected_routed_experts == t.routed_execution.selected_routed_experts;
    let physical_missing_sequence_exact =
        cs.physical_missing_ids_sha256 == ts.physical_missing_ids_sha256;
    let physical_missing_counts_exact = cs.physical_missing_experts == ts.physical_missing_experts;
    let demand_source_request_sequence_exact =
        cs.demand_source_request_ids_sha256 == ts.demand_source_request_ids_sha256;
    let full_token_replay_zero = c.token_loop.replay_attempts == 0
        && t.token_loop.replay_attempts == 0
        && c.recovery.full_token_replay_attempts == 0
        && t.recovery.full_token_replay_attempts == 0;
    let fatal_and_no_progress_zero = c.token_loop.fatal_failures == 0
        && t.token_loop.fatal_failures == 0
        && c.token_loop.no_progress_failures == 0
        && t.token_loop.no_progress_failures == 0;
    let all_speculative_work_zero =
        arm_all_speculative_work_zero(c) && arm_all_speculative_work_zero(t);
    let warmup_all_speculative_work_zero =
        arm_all_speculative_work_zero(cww) && arm_all_speculative_work_zero(tww);
    let mut result = Reconciliation {
        generated_tokens_exact,
        generated_token_hashes_exact,
        warmup_token_hashes_exact,
        warmup_source_requests_exact: cws.demand_source_requests == tws.demand_source_requests,
        warmup_ram_source_hits_exact: cws.source_ram_hits == tws.source_ram_hits,
        warmup_ram_source_misses_exact: cws.source_ram_misses == tws.source_ram_misses,
        warmup_nvme_reads_exact: cws.source_nvme_reads == tws.source_nvme_reads,
        warmup_nvme_bytes_exact: cws.source_nvme_bytes == tws.source_nvme_bytes,
        warmup_ram_cache_inserts_exact: cws.ram_cache_inserts == tws.ram_cache_inserts,
        warmup_ram_cache_evictions_exact: cws.ram_cache_evictions == tws.ram_cache_evictions,
        warmup_ordered_ram_insert_ids_exact: cws.demand_ram_insert_ids_sha256
            == tws.demand_ram_insert_ids_sha256,
        warmup_ordered_ram_eviction_ids_exact: cws.demand_ram_eviction_ids_sha256
            == tws.demand_ram_eviction_ids_sha256,
        warmup_physical_missing_sequence_exact: cws.physical_missing_ids_sha256
            == tws.physical_missing_ids_sha256,
        warmup_demand_source_sequence_exact: cws.demand_source_request_ids_sha256
            == tws.demand_source_request_ids_sha256,
        warmup_ram_cache_state_exact: control.warmup_ram_cache_state_sha256
            == treatment.warmup_ram_cache_state_sha256,
        warmup_all_speculative_work_zero,
        selected_route_sequence_exact,
        selected_route_counts_exact,
        physical_missing_sequence_exact,
        physical_missing_counts_exact,
        demand_source_request_sequence_exact,
        demand_source_requests_exact: cs.demand_source_requests == ts.demand_source_requests,
        ram_source_hits_exact: cs.source_ram_hits == ts.source_ram_hits,
        ram_source_misses_exact: cs.source_ram_misses == ts.source_ram_misses,
        demand_nvme_reads_exact: cs.source_nvme_reads == ts.source_nvme_reads,
        demand_nvme_bytes_exact: cs.source_nvme_bytes == ts.source_nvme_bytes,
        logical_admissions_exact: c
            .gpu_native_residency
            .logical_admissions_for_physical_misses
            == t.gpu_native_residency
                .logical_admissions_for_physical_misses,
        ram_to_vram_installs_exact: c.gpu_native_residency.ram_to_vram_installs
            == t.gpu_native_residency.ram_to_vram_installs,
        ram_to_vram_bytes_exact: c.gpu_expert_io.expert_weight_upload_bytes
            == t.gpu_expert_io.expert_weight_upload_bytes,
        physical_evictions_exact: c.gpu_native_residency.physical_evictions
            == t.gpu_native_residency.physical_evictions,
        physical_reinstalls_exact: c.gpu_native_residency.physical_reinstalls
            == t.gpu_native_residency.physical_reinstalls,
        physical_victim_sequence_exact: cs.physical_victim_ids_sha256
            == ts.physical_victim_ids_sha256,
        physical_residency_identity_stream_exact: cs.physical_residency_identity_sha256
            == ts.physical_residency_identity_sha256,
        physical_install_attempts_exact: cs.physical_install_attempts
            == ts.physical_install_attempts,
        physical_install_completions_exact: cs.physical_install_completions
            == ts.physical_install_completions,
        physical_slot_bytes_staged_exact: cs.physical_slot_bytes_staged
            == ts.physical_slot_bytes_staged,
        mapping_publications_exact: cs.mapping_publications == ts.mapping_publications,
        mapping_unpublications_exact: cs.mapping_unpublications == ts.mapping_unpublications,
        vram_hits_exact: c.gpu_native_residency.vram_hits == t.gpu_native_residency.vram_hits,
        vram_misses_exact: c.gpu_native_residency.vram_misses == t.gpu_native_residency.vram_misses,
        residency_miss_attempts_exact: c.token_loop.residency_miss_attempts
            == t.token_loop.residency_miss_attempts,
        residency_services_exact: c.token_loop.residency_services
            == t.token_loop.residency_services,
        recovery_segments_exact: c.recovery.recovery_segments == t.recovery.recovery_segments,
        miss_boundaries_exact: c.token_loop.residency_miss_attempts
            == t.token_loop.residency_miss_attempts
            && c.token_loop.residency_services == t.token_loop.residency_services,
        recovery_semantics_exact: recovery_semantics_equal(c.recovery, t.recovery),
        full_token_replay_zero,
        fatal_and_no_progress_zero,
        ram_cache_inserts_exact: cs.ram_cache_inserts == ts.ram_cache_inserts,
        ram_cache_evictions_exact: cs.ram_cache_evictions == ts.ram_cache_evictions,
        ordered_ram_insert_ids_exact: cs.demand_ram_insert_ids_sha256
            == ts.demand_ram_insert_ids_sha256,
        ordered_ram_eviction_ids_exact: cs.demand_ram_eviction_ids_sha256
            == ts.demand_ram_eviction_ids_sha256,
        all_speculative_work_zero,
        primary_pool_capacity_exact: cs.primary_pool_capacity == ts.primary_pool_capacity,
        all_invariants_pass: false,
    };
    result.all_invariants_pass = result.generated_tokens_exact
        && result.generated_token_hashes_exact
        && result.warmup_token_hashes_exact
        && result.warmup_source_requests_exact
        && result.warmup_ram_source_hits_exact
        && result.warmup_ram_source_misses_exact
        && result.warmup_nvme_reads_exact
        && result.warmup_nvme_bytes_exact
        && result.warmup_ram_cache_inserts_exact
        && result.warmup_ram_cache_evictions_exact
        && result.warmup_ordered_ram_insert_ids_exact
        && result.warmup_ordered_ram_eviction_ids_exact
        && result.warmup_physical_missing_sequence_exact
        && result.warmup_demand_source_sequence_exact
        && result.warmup_ram_cache_state_exact
        && result.warmup_all_speculative_work_zero
        && result.selected_route_sequence_exact
        && result.selected_route_counts_exact
        && result.physical_missing_sequence_exact
        && result.physical_missing_counts_exact
        && result.demand_source_request_sequence_exact
        && result.demand_source_requests_exact
        && result.ram_source_hits_exact
        && result.ram_source_misses_exact
        && result.demand_nvme_reads_exact
        && result.demand_nvme_bytes_exact
        && result.logical_admissions_exact
        && result.ram_to_vram_installs_exact
        && result.ram_to_vram_bytes_exact
        && result.physical_evictions_exact
        && result.physical_reinstalls_exact
        && result.physical_victim_sequence_exact
        && result.physical_residency_identity_stream_exact
        && result.physical_install_attempts_exact
        && result.physical_install_completions_exact
        && result.physical_slot_bytes_staged_exact
        && result.mapping_publications_exact
        && result.mapping_unpublications_exact
        && result.vram_hits_exact
        && result.vram_misses_exact
        && result.residency_miss_attempts_exact
        && result.residency_services_exact
        && result.recovery_segments_exact
        && result.miss_boundaries_exact
        && result.recovery_semantics_exact
        && result.full_token_replay_zero
        && result.fatal_and_no_progress_zero
        && result.ram_cache_inserts_exact
        && result.ram_cache_evictions_exact
        && result.ordered_ram_insert_ids_exact
        && result.ordered_ram_eviction_ids_exact
        && result.all_speculative_work_zero
        && result.primary_pool_capacity_exact;
    result
}

fn common_gates(reconciliation: &Reconciliation) -> (BehavioralGate, WorkEquivalenceGate) {
    let behavioral_pass = reconciliation.generated_tokens_exact
        && reconciliation.generated_token_hashes_exact
        && reconciliation.selected_route_sequence_exact
        && reconciliation.selected_route_counts_exact
        && reconciliation.physical_missing_sequence_exact
        && reconciliation.physical_missing_counts_exact
        && reconciliation.full_token_replay_zero
        && reconciliation.fatal_and_no_progress_zero
        && reconciliation.all_speculative_work_zero;
    let work_pass = reconciliation.demand_source_requests_exact
        && reconciliation.ram_source_hits_exact
        && reconciliation.ram_source_misses_exact
        && reconciliation.demand_nvme_reads_exact
        && reconciliation.demand_nvme_bytes_exact
        && reconciliation.ram_cache_inserts_exact
        && reconciliation.ram_cache_evictions_exact
        && reconciliation.logical_admissions_exact
        && reconciliation.ram_to_vram_installs_exact
        && reconciliation.ram_to_vram_bytes_exact
        && reconciliation.physical_evictions_exact
        && reconciliation.physical_reinstalls_exact
        && reconciliation.physical_victim_sequence_exact
        && reconciliation.physical_residency_identity_stream_exact
        && reconciliation.physical_install_attempts_exact
        && reconciliation.physical_install_completions_exact
        && reconciliation.physical_slot_bytes_staged_exact
        && reconciliation.mapping_publications_exact
        && reconciliation.mapping_unpublications_exact
        && reconciliation.miss_boundaries_exact
        && reconciliation.recovery_semantics_exact
        && reconciliation.ordered_ram_insert_ids_exact
        && reconciliation.ordered_ram_eviction_ids_exact;
    (
        BehavioralGate {
            generated_token_parity_exact: reconciliation.generated_tokens_exact
                && reconciliation.generated_token_hashes_exact,
            route_parity_exact: reconciliation.selected_route_sequence_exact
                && reconciliation.selected_route_counts_exact,
            physical_missing_sequence_exact: reconciliation.physical_missing_sequence_exact
                && reconciliation.physical_missing_counts_exact,
            full_token_replay_zero: reconciliation.full_token_replay_zero,
            fatal_and_no_progress_zero: reconciliation.fatal_and_no_progress_zero,
            speculation_zero: reconciliation.all_speculative_work_zero,
            passed: behavioral_pass,
        },
        WorkEquivalenceGate {
            source_requests_exact: reconciliation.demand_source_requests_exact
                && reconciliation.demand_source_request_sequence_exact,
            ram_hits_and_misses_exact: reconciliation.ram_source_hits_exact
                && reconciliation.ram_source_misses_exact,
            nvme_reads_exact: reconciliation.demand_nvme_reads_exact,
            demand_nvme_bytes_exact: reconciliation.demand_nvme_bytes_exact,
            ram_inserts_and_evictions_exact: reconciliation.ram_cache_inserts_exact
                && reconciliation.ram_cache_evictions_exact
                && reconciliation.ordered_ram_insert_ids_exact
                && reconciliation.ordered_ram_eviction_ids_exact,
            logical_admissions_exact: reconciliation.logical_admissions_exact,
            demand_h2d_installs_exact: reconciliation.ram_to_vram_installs_exact,
            demand_h2d_bytes_exact: reconciliation.ram_to_vram_bytes_exact,
            physical_evictions_exact: reconciliation.physical_evictions_exact,
            physical_victim_order_exact: reconciliation.physical_victim_sequence_exact,
            physical_residency_identity_stream_exact: reconciliation
                .physical_residency_identity_stream_exact,
            slot_installs_and_mapping_publications_exact: reconciliation
                .physical_install_attempts_exact
                && reconciliation.physical_install_completions_exact
                && reconciliation.physical_slot_bytes_staged_exact
                && reconciliation.mapping_publications_exact,
            mapping_unpublications_exact: reconciliation.mapping_unpublications_exact,
            physical_reinstalls_exact: reconciliation.physical_reinstalls_exact,
            recovery_and_miss_boundaries_exact: reconciliation.miss_boundaries_exact
                && reconciliation.recovery_semantics_exact,
            deterministic_ram_cache_order_exact: reconciliation.ordered_ram_insert_ids_exact
                && reconciliation.ordered_ram_eviction_ids_exact,
            passed: work_pass,
        },
    )
}

fn comparison(control: f64, treatment: f64) -> MetricComparison {
    MetricComparison {
        control,
        treatment,
        delta_percent: if control == 0.0 {
            0.0
        } else {
            (treatment - control) / control * 100.0
        },
    }
}

fn production_safety_zero(snapshot: &ProductionDemandSourceSnapshot) -> bool {
    snapshot.production_cache_reservation_leaks == 0
        && snapshot.production_batch_commit_violations == 0
        && snapshot.stale_singleflight_entries == 0
}

fn production_reconciliation(
    common: Reconciliation,
    control: &ArmReport,
    treatment: &ArmReport,
) -> ProductionReconciliation {
    let cwp = control
        .warmup_production
        .as_ref()
        .expect("complete v2 control warmup production telemetry");
    let twp = treatment
        .warmup_production
        .as_ref()
        .expect("complete v2 treatment warmup production telemetry");
    let cp = control
        .production
        .as_ref()
        .expect("complete v2 control production telemetry");
    let tp = treatment
        .production
        .as_ref()
        .expect("complete v2 treatment production telemetry");
    let warmup_production_cache_reservation_leaks_zero =
        cwp.production_cache_reservation_leaks == 0 && twp.production_cache_reservation_leaks == 0;
    let warmup_production_batch_commit_violations_zero =
        cwp.production_batch_commit_violations == 0 && twp.production_batch_commit_violations == 0;
    let warmup_stale_singleflight_entries_zero =
        cwp.stale_singleflight_entries == 0 && twp.stale_singleflight_entries == 0;
    let production_cache_reservation_leaks_zero =
        cp.production_cache_reservation_leaks == 0 && tp.production_cache_reservation_leaks == 0;
    let production_batch_commit_violations_zero =
        cp.production_batch_commit_violations == 0 && tp.production_batch_commit_violations == 0;
    let stale_singleflight_entries_zero =
        cp.stale_singleflight_entries == 0 && tp.stale_singleflight_entries == 0;
    let all_invariants_pass = common.all_invariants_pass
        && warmup_production_cache_reservation_leaks_zero
        && warmup_production_batch_commit_violations_zero
        && warmup_stale_singleflight_entries_zero
        && production_cache_reservation_leaks_zero
        && production_batch_commit_violations_zero
        && stale_singleflight_entries_zero;
    ProductionReconciliation {
        common,
        warmup_production_cache_reservation_leaks_zero,
        warmup_production_batch_commit_violations_zero,
        warmup_stale_singleflight_entries_zero,
        production_cache_reservation_leaks_zero,
        production_batch_commit_violations_zero,
        stale_singleflight_entries_zero,
        all_invariants_pass,
    }
}

fn production_gates(
    reconciliation: &ProductionReconciliation,
    control: &ArmReport,
    treatment: &ArmReport,
) -> ProductionGates {
    let (behavioral, work_equivalence) = common_gates(&reconciliation.common);
    let cs = control
        .source
        .as_ref()
        .expect("complete control staging telemetry");
    let ts = treatment
        .source
        .as_ref()
        .expect("complete treatment staging telemetry");
    let cws = control
        .warmup_source
        .as_ref()
        .expect("complete control warmup staging telemetry");
    let tws = treatment
        .warmup_source
        .as_ref()
        .expect("complete treatment warmup staging telemetry");
    let cwpi = control
        .warmup_production_physical_install
        .as_ref()
        .expect("complete control warmup production physical-install telemetry");
    let twpi = treatment
        .warmup_production_physical_install
        .as_ref()
        .expect("complete treatment warmup production physical-install telemetry");
    let cpi = control
        .production_physical_install
        .as_ref()
        .expect("complete control production physical-install telemetry");
    let tpi = treatment
        .production_physical_install
        .as_ref()
        .expect("complete treatment production physical-install telemetry");
    let cp = control
        .production
        .as_ref()
        .expect("complete control source telemetry");
    let tp = treatment
        .production
        .as_ref()
        .expect("complete treatment source telemetry");
    let pair_reconciles =
        |control: &GpuNativePhysicalInstallStagingQualificationSnapshot,
         treatment: &GpuNativePhysicalInstallStagingQualificationSnapshot| {
            control.full_slot_vec_materializations > 0
                && control.direct_staging_writes == 0
                && treatment.direct_staging_writes > 0
                && treatment.full_slot_vec_materializations == 0
                && control.physical_install_attempts == control.physical_install_completions
                && treatment.physical_install_attempts == treatment.physical_install_completions
                && control.physical_install_completions == treatment.physical_install_completions
                && control.physical_slot_bytes_staged == treatment.physical_slot_bytes_staged
                && control.mapping_publications == treatment.mapping_publications
                && control.mapping_unpublications == treatment.mapping_unpublications
                && control.physical_victim_ids_sha256 == treatment.physical_victim_ids_sha256
                && control.physical_residency_identity_sha256
                    == treatment.physical_residency_identity_sha256
                && control.mapping_publications == control.physical_install_completions
                && treatment.mapping_publications == treatment.physical_install_completions
                && treatment.direct_staging_failures == 0
        };
    let warmup_mechanism_reconciled = pair_reconciles(cws, tws)
        && cwpi.physical_install_attempts == 0
        && cwpi.direct_staging_successes == 0
        && cwpi.direct_staging_unavailable == 0
        && cwpi.direct_staging_allocation_fallbacks == 0
        && cwpi.physical_install_failures == 0
        && twpi.physical_install_attempts == tws.physical_install_attempts
        && twpi.direct_staging_successes == tws.physical_install_completions
        && twpi.direct_staging_unavailable == 0
        && twpi.direct_staging_allocation_fallbacks == 0
        && twpi.physical_install_failures == 0;
    let control_vec_materializations_gt_zero = cs.full_slot_vec_materializations > 0;
    let control_direct_staging_writes_zero = cs.direct_staging_writes == 0;
    let treatment_direct_staging_writes_gt_zero = ts.direct_staging_writes > 0;
    let treatment_vec_materializations_zero = ts.full_slot_vec_materializations == 0;
    let treatment_install_completions_gt_zero = ts.physical_install_completions > 0;
    let attempts_equal_completions = cs.physical_install_attempts
        == cs.physical_install_completions
        && ts.physical_install_attempts == ts.physical_install_completions
        && cs.physical_install_completions == ts.physical_install_completions;
    let staged_bytes_exact = cs.physical_slot_bytes_staged == ts.physical_slot_bytes_staged;
    let mapping_publications_exact = cs.mapping_publications == ts.mapping_publications
        && cs.mapping_publications == cs.physical_install_completions
        && ts.mapping_publications == ts.physical_install_completions;
    let treatment_staging_failures_zero = ts.direct_staging_failures == 0;
    let control_production_direct_staging_successes_zero = cpi.physical_install_attempts == 0
        && cpi.direct_staging_successes == 0
        && cpi.direct_staging_unavailable == 0
        && cpi.direct_staging_allocation_fallbacks == 0
        && cpi.physical_install_failures == 0;
    let treatment_production_direct_staging_successes_gt_zero = tpi.direct_staging_successes > 0
        && tpi.physical_install_attempts == ts.physical_install_attempts
        && tpi.direct_staging_successes == ts.physical_install_completions;
    let treatment_direct_allocation_fallbacks_zero = tpi.direct_staging_allocation_fallbacks == 0;
    let treatment_production_install_failures_zero =
        tpi.direct_staging_unavailable == 0 && tpi.physical_install_failures == 0;
    let passed = warmup_mechanism_reconciled
        && control_vec_materializations_gt_zero
        && control_direct_staging_writes_zero
        && treatment_direct_staging_writes_gt_zero
        && treatment_vec_materializations_zero
        && treatment_install_completions_gt_zero
        && attempts_equal_completions
        && staged_bytes_exact
        && mapping_publications_exact
        && treatment_staging_failures_zero
        && control_production_direct_staging_successes_zero
        && treatment_production_direct_staging_successes_gt_zero
        && treatment_direct_allocation_fallbacks_zero
        && treatment_production_install_failures_zero
        && production_safety_zero(cp)
        && production_safety_zero(tp);
    ProductionGates {
        behavioral,
        work_equivalence,
        mechanism: ProductionMechanismGate {
            warmup_mechanism_reconciled,
            control_vec_materializations_gt_zero,
            control_direct_staging_writes_zero,
            treatment_direct_staging_writes_gt_zero,
            treatment_vec_materializations_zero,
            treatment_install_completions_gt_zero,
            attempts_equal_completions,
            staged_bytes_exact,
            mapping_publications_exact,
            treatment_staging_failures_zero,
            control_production_direct_staging_successes_zero,
            treatment_production_direct_staging_successes_gt_zero,
            treatment_direct_allocation_fallbacks_zero,
            treatment_production_install_failures_zero,
            passed,
        },
    }
}

fn production_arm_performance(
    arm: &ArmReport,
) -> Result<ProductionArmPerformance, BenchmarkFailure> {
    let aggregate = arm.benchmark.aggregate.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "postcondition",
            "missing-arm-aggregate",
            "complete PR2-B-A arm did not produce a benchmark aggregate",
        )
    })?;
    let runs = generated_results(arm);
    let mean_request_wall_seconds = runs
        .iter()
        .map(|run| run.timing.end_to_end_seconds)
        .sum::<f64>()
        / runs.len() as f64;
    let source = arm.source.as_ref().expect("complete arm source");
    Ok(ProductionArmPerformance {
        decode_tps: aggregate.decode_tps.mean,
        end_to_end_generated_tps: aggregate.end_to_end_generated_tps.mean,
        mean_request_wall_seconds,
        source_acquisition_wall_us: source.source_acquisition_wall_us,
        logical_demand_admission_us: source.logical_demand_admission_us,
        physical_demand_install_us: source.physical_demand_install_us,
        physical_slot_prepare_us: source.physical_slot_prepare_us,
        physical_queue_staging_us: source.physical_queue_staging_us,
        mapping_publication_us: source.mapping_publication_us,
        physical_install_total_us: source.physical_install_total_us,
        total_residency_service_us: source.total_residency_service_us,
    })
}

fn production_performance(
    control: &ArmReport,
    treatment: &ArmReport,
) -> Result<ProductionPerformanceComparison, BenchmarkFailure> {
    let control = production_arm_performance(control)?;
    let treatment = production_arm_performance(treatment)?;
    let performance_result = if treatment.physical_demand_install_us
        < control.physical_demand_install_us
        && treatment.decode_tps >= control.decode_tps
        && treatment.end_to_end_generated_tps >= control.end_to_end_generated_tps
    {
        "improved"
    } else if treatment.physical_demand_install_us > control.physical_demand_install_us
        && treatment.decode_tps <= control.decode_tps
        && treatment.end_to_end_generated_tps <= control.end_to_end_generated_tps
    {
        "regressed"
    } else {
        "neutral"
    };
    Ok(ProductionPerformanceComparison {
        decode_tps: comparison(control.decode_tps, treatment.decode_tps),
        end_to_end_generated_tps: comparison(
            control.end_to_end_generated_tps,
            treatment.end_to_end_generated_tps,
        ),
        mean_request_wall_seconds: comparison(
            control.mean_request_wall_seconds,
            treatment.mean_request_wall_seconds,
        ),
        source_acquisition_wall_us: comparison(
            control.source_acquisition_wall_us as f64,
            treatment.source_acquisition_wall_us as f64,
        ),
        logical_demand_admission_us: comparison(
            control.logical_demand_admission_us as f64,
            treatment.logical_demand_admission_us as f64,
        ),
        physical_demand_install_us: comparison(
            control.physical_demand_install_us as f64,
            treatment.physical_demand_install_us as f64,
        ),
        physical_slot_prepare_us: comparison(
            control.physical_slot_prepare_us as f64,
            treatment.physical_slot_prepare_us as f64,
        ),
        physical_queue_staging_us: comparison(
            control.physical_queue_staging_us as f64,
            treatment.physical_queue_staging_us as f64,
        ),
        mapping_publication_us: comparison(
            control.mapping_publication_us as f64,
            treatment.mapping_publication_us as f64,
        ),
        physical_install_total_us: comparison(
            control.physical_install_total_us as f64,
            treatment.physical_install_total_us as f64,
        ),
        total_residency_service_us: comparison(
            control.total_residency_service_us as f64,
            treatment.total_residency_service_us as f64,
        ),
        control,
        treatment,
        performance_result,
    })
}

fn emit_report<T: Serialize>(report: &T, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json)?;
    eprintln!("GPU-native physical-install qualification report written to {}", path.display());
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    let prepared = prepare(&args)?;
    let mut report = ProductionQualificationReport {
        schema: PRODUCTION_SCHEMA,
        mode: PRODUCTION_MODE,
        production_physical_install_changed: true,
        control_forces_legacy_full_slot_vec: true,
        treatment_uses_ordinary_production_path: true,
        normal_production_uses_direct_queue_staging: true,
        allocation_only_fallback_implemented: false,
        allocation_only_fallback_decision: "not implemented: wgpu 0.20 Queue::write_buffer_with returns None for either destination validation failure or staging-buffer creation failure, so allocation-only fallback cannot be distinguished and proved without a WGPU API change",
        both_arms_use_ordinary_production_source: true,
        pinned_wgpu_api_audit: WgpuApiAudit {
            requested_version: "0.20",
            resolved_version: "0.20.1",
            supported_api: "Queue::write_buffer_with",
            exact_destination_range: true,
            none_distinguishes_allocation_from_validation: false,
            schedules_at_next_submit: true,
            adds_queue_submit: false,
            adds_device_poll: false,
            adds_map_async: false,
            adds_readback: false,
            uses_unsafe_internals: false,
        },
        timing_definitions: TimingDefinitions {
            physical_slot_prepare_us: "qualification-only timing of validated host byte-layout preparation: control allocates/zeros/fills a full-slot Vec; treatment validates and fills the ordinary production direct queue view",
            physical_queue_staging_us: "qualification-only timing of control Queue::write_buffer or ordinary-production Queue::write_buffer_with view acquisition plus Drop scheduling, excluding view fill",
            mapping_publication_us: "existing logical mapping Queue::write_buffer call after physical staging",
            physical_install_total_us: "qualification-only per-install timer from a validated install permit through physical staging, mapping publication, and host arena commit; ordinary serving does not execute this timer",
            physical_demand_install_us: "existing aggregate Engine wall timer around the complete residency-manager demand-set transaction",
        },
        benchmark_complete: false,
        qualification_pass: false,
        performance_result: "not_measured",
        failure: None,
        frozen_workload: frozen_workload(args.expected_adapter_name.clone()),
        provenance: prepared.provenance.clone(),
        control: None,
        treatment: None,
        reconciliation: None,
        gates: None,
        performance: None,
    };

    let control = match run_arm(
        &prepared,
        &args,
        GpuNativePhysicalInstallStagingQualificationArm::Control,
    )
    .await
    {
        Ok(control) => control,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let control_failure = control.failure.clone();
    report.control = Some(control);
    if let Some(failure) = control_failure {
        report.failure = Some(failure.clone());
        emit_report(&report, &args.report_out)?;
        return Err(failure.to_string().into());
    }

    let treatment = match run_arm(
        &prepared,
        &args,
        GpuNativePhysicalInstallStagingQualificationArm::Treatment,
    )
    .await
    {
        Ok(treatment) => treatment,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let treatment_failure = treatment.failure.clone();
    report.treatment = Some(treatment);
    if let Some(failure) = treatment_failure {
        report.failure = Some(failure.clone());
        emit_report(&report, &args.report_out)?;
        return Err(failure.to_string().into());
    }

    let control = report.control.as_ref().expect("control stored");
    let treatment = report.treatment.as_ref().expect("treatment stored");
    let reconciliation =
        production_reconciliation(reconcile(control, treatment), control, treatment);
    let gates = production_gates(&reconciliation, control, treatment);
    let performance = match production_performance(control, treatment) {
        Ok(performance) => performance,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let qualification_pass = reconciliation.all_invariants_pass
        && gates.behavioral.passed
        && gates.work_equivalence.passed
        && gates.mechanism.passed;
    report.benchmark_complete = true;
    report.qualification_pass = qualification_pass;
    report.performance_result = performance.performance_result;
    report.reconciliation = Some(reconciliation);
    report.gates = Some(gates);
    report.performance = Some(performance);
    emit_report(&report, &args.report_out)?;
    if qualification_pass {
        Ok(())
    } else {
        Err("PR2-B-A production qualification gates did not all pass; see emitted report".into())
    }
}

fn production_physical_install_counters_reconcile(
    source: &GpuNativeProductionPhysicalInstallSnapshot,
) -> bool {
    source.physical_install_sets > 0
        && source.physical_install_experts == source.reservation_attempts
        && source.reservation_attempts == source.reservation_successes
        && source.reservation_failures == 0
        && source.physical_install_attempts == source.physical_stage_attempts
        && source.physical_stage_attempts == source.physical_stage_completions
        && source.physical_stage_failures == 0
        && source.physical_stage_completions == source.direct_staging_successes
        && source.direct_staging_unavailable == 0
        && source.direct_staging_allocation_fallbacks == 0
        && source.ordered_commit_attempts == source.ordered_commit_completions
        && source.ordered_commit_completions == source.physical_stage_completions
        && source.ordered_commit_failures == 0
        && source.ordered_commit_violations == 0
        && source.unpublished_physical_writes_after_failure == 0
        && source.physical_install_failures == 0
        && source.parallel_eligible_sets == source.parallel_staging_sets
        && source.physical_install_experts
            == source
                .parallel_staging_experts
                .saturating_add(source.singleton_staging_sets)
}

fn production_concurrency_exercised(source: &GpuNativeProductionPhysicalInstallSnapshot) -> bool {
    production_physical_install_counters_reconcile(source)
        && source.parallel_staging_sets > 0
        && source.parallel_staging_experts > 0
        && source.max_in_flight_physical_staging >= 2
}

fn concurrency_mechanism_gate(
    control: &ConcurrencyArmReport,
    treatment: &ConcurrencyArmReport,
) -> ConcurrencyMechanismGate {
    use crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot as Snapshot;

    fn internally_reconciles(source: &Snapshot) -> bool {
        source.physical_install_attempts == source.reservation_attempts
            && source.reservation_attempts == source.reservation_successes
            && source.reservation_failures == 0
            && source.physical_stage_attempts == source.physical_stage_completions
            && source.physical_stage_failures == 0
            && source.physical_stage_completions == source.direct_staging_writes
            && source.physical_stage_completions == source.physical_install_completions
            && source.physical_stage_completions == source.ordered_commit_attempts
            && source.ordered_commit_attempts == source.ordered_commit_completions
            && source.ordered_commit_failures == 0
            && source.ordered_commit_violations == 0
            && source.physical_bytes_staged == source.physical_slot_bytes_staged
            && source.mapping_publications == source.ordered_commit_completions
            && source.unpublished_physical_writes_after_failure == 0
    }

    fn pair_reconciles(control: &Snapshot, treatment: &Snapshot) -> bool {
        internally_reconciles(control)
            && internally_reconciles(treatment)
            && control.physical_install_completions == treatment.physical_install_completions
            && control.physical_slot_bytes_staged == treatment.physical_slot_bytes_staged
            && control.mapping_publications == treatment.mapping_publications
            && control.mapping_unpublications == treatment.mapping_unpublications
            && control.physical_victim_ids_sha256 == treatment.physical_victim_ids_sha256
            && control.reservation_identity_sha256 == treatment.reservation_identity_sha256
            && control.physical_residency_identity_sha256
                == treatment.physical_residency_identity_sha256
    }

    let c = control
        .mechanism
        .as_ref()
        .expect("complete control mechanism");
    let t = treatment
        .mechanism
        .as_ref()
        .expect("complete treatment mechanism");
    let cw = control
        .warmup_mechanism
        .as_ref()
        .expect("complete control warmup mechanism");
    let tw = treatment
        .warmup_mechanism
        .as_ref()
        .expect("complete treatment warmup mechanism");
    let cwpi = control
        .common
        .warmup_production_physical_install
        .as_ref()
        .expect("complete control warmup production physical-install telemetry");
    let twpi = treatment
        .common
        .warmup_production_physical_install
        .as_ref()
        .expect("complete treatment warmup production physical-install telemetry");
    let cpi = control
        .common
        .production_physical_install
        .as_ref()
        .expect("complete control production physical-install telemetry");
    let tpi = treatment
        .common
        .production_physical_install
        .as_ref()
        .expect("complete treatment production physical-install telemetry");
    let warmup_mechanism_reconciled = pair_reconciles(cw, tw)
        && cw.parallel_staging_sets == 0
        && cw.max_in_flight_physical_staging <= 1
        && tw.parallel_staging_sets > 0
        && tw.max_in_flight_physical_staging >= 2
        && cwpi.physical_install_sets == 0
        && cwpi.parallel_staging_sets == 0
        && production_concurrency_exercised(twpi);
    let control_production_parallel_staging_sets_zero = cpi.physical_install_sets == 0
        && cpi.parallel_staging_sets == 0
        && cpi.physical_install_attempts == 0;
    let treatment_ordinary_production_path_exercised = t.treatment_uses_ordinary_production_path
        && t.production_physical_install_concurrency_changed
        && t.normal_production_uses_concurrent_physical_staging
        && !c.treatment_uses_ordinary_production_path
        && c.control_forces_sequential_direct_staging
        && tpi.physical_install_sets > 0;
    let treatment_production_parallel_staging_sets_gt_zero = production_concurrency_exercised(tpi);
    let treatment_production_counters_reconcile =
        production_physical_install_counters_reconcile(tpi);
    let control_direct_staging_writes_gt_zero = c.direct_staging_writes > 0;
    let control_full_slot_vec_materializations_zero = c.full_slot_vec_materializations == 0;
    let control_parallel_staging_sets_zero = c.parallel_staging_sets == 0;
    let control_max_in_flight_lte_one = c.max_in_flight_physical_staging <= 1;
    let treatment_direct_staging_writes_gt_zero = t.direct_staging_writes > 0;
    let treatment_full_slot_vec_materializations_zero = t.full_slot_vec_materializations == 0;
    let treatment_parallel_staging_sets_gt_zero = t.parallel_staging_sets > 0;
    let treatment_parallel_staging_experts_gt_zero = t.parallel_staging_experts > 0;
    let treatment_max_in_flight_gte_two = t.max_in_flight_physical_staging >= 2;
    let reservation_failures_zero = c.reservation_failures == 0 && t.reservation_failures == 0;
    let physical_stage_failures_zero =
        c.physical_stage_failures == 0 && t.physical_stage_failures == 0;
    let ordered_commit_failures_zero =
        c.ordered_commit_failures == 0 && t.ordered_commit_failures == 0;
    let ordered_commit_violations_zero =
        c.ordered_commit_violations == 0 && t.ordered_commit_violations == 0;
    let unpublished_physical_writes_after_failure_zero =
        c.unpublished_physical_writes_after_failure == 0
            && t.unpublished_physical_writes_after_failure == 0;
    let attempts_and_completions_reconcile = pair_reconciles(c, t);
    let staged_bytes_exact = c.physical_bytes_staged == t.physical_bytes_staged;
    let mapping_publications_exact = c.mapping_publications == t.mapping_publications;
    let mapping_unpublications_exact = c.mapping_unpublications == t.mapping_unpublications;
    let victim_sequence_exact = c.physical_victim_ids_sha256 == t.physical_victim_ids_sha256;
    let reservation_identity_exact =
        c.reservation_identity_sha256 == t.reservation_identity_sha256;
    let ordered_physical_identity_exact =
        c.physical_residency_identity_sha256 == t.physical_residency_identity_sha256;
    let passed = warmup_mechanism_reconciled
        && control_production_parallel_staging_sets_zero
        && treatment_ordinary_production_path_exercised
        && treatment_production_parallel_staging_sets_gt_zero
        && treatment_production_counters_reconcile
        && control_direct_staging_writes_gt_zero
        && control_full_slot_vec_materializations_zero
        && control_parallel_staging_sets_zero
        && control_max_in_flight_lte_one
        && treatment_direct_staging_writes_gt_zero
        && treatment_full_slot_vec_materializations_zero
        && treatment_parallel_staging_sets_gt_zero
        && treatment_parallel_staging_experts_gt_zero
        && treatment_max_in_flight_gte_two
        && reservation_failures_zero
        && physical_stage_failures_zero
        && ordered_commit_failures_zero
        && ordered_commit_violations_zero
        && unpublished_physical_writes_after_failure_zero
        && attempts_and_completions_reconcile
        && staged_bytes_exact
        && mapping_publications_exact
        && mapping_unpublications_exact
        && victim_sequence_exact
        && reservation_identity_exact
        && ordered_physical_identity_exact;
    ConcurrencyMechanismGate {
        warmup_mechanism_reconciled,
        control_production_parallel_staging_sets_zero,
        treatment_ordinary_production_path_exercised,
        treatment_production_parallel_staging_sets_gt_zero,
        treatment_production_counters_reconcile,
        control_direct_staging_writes_gt_zero,
        control_full_slot_vec_materializations_zero,
        control_parallel_staging_sets_zero,
        control_max_in_flight_lte_one,
        treatment_direct_staging_writes_gt_zero,
        treatment_full_slot_vec_materializations_zero,
        treatment_parallel_staging_sets_gt_zero,
        treatment_parallel_staging_experts_gt_zero,
        treatment_max_in_flight_gte_two,
        reservation_failures_zero,
        physical_stage_failures_zero,
        ordered_commit_failures_zero,
        ordered_commit_violations_zero,
        unpublished_physical_writes_after_failure_zero,
        attempts_and_completions_reconcile,
        staged_bytes_exact,
        mapping_publications_exact,
        mapping_unpublications_exact,
        victim_sequence_exact,
        reservation_identity_exact,
        ordered_physical_identity_exact,
        passed,
    }
}

fn concurrency_performance(
    control: &ConcurrencyArmReport,
    treatment: &ConcurrencyArmReport,
) -> Result<ConcurrencyPerformanceComparison, BenchmarkFailure> {
    let common = production_performance(&control.common, &treatment.common)?;
    let c = control
        .mechanism
        .as_ref()
        .expect("complete control mechanism");
    let t = treatment
        .mechanism
        .as_ref()
        .expect("complete treatment mechanism");
    let overlap =
        |source: &crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot| {
            if source.physical_parallel_stage_wall_us == 0 {
                0.0
            } else {
                source.sum_individual_physical_stage_us as f64
                    / source.physical_parallel_stage_wall_us as f64
            }
        };
    Ok(ConcurrencyPerformanceComparison {
        physical_reservation_us: comparison(
            c.physical_reservation_us as f64,
            t.physical_reservation_us as f64,
        ),
        physical_parallel_stage_wall_us: comparison(
            c.physical_parallel_stage_wall_us as f64,
            t.physical_parallel_stage_wall_us as f64,
        ),
        sum_individual_physical_stage_us: comparison(
            c.sum_individual_physical_stage_us as f64,
            t.sum_individual_physical_stage_us as f64,
        ),
        physical_ordered_commit_us: comparison(
            c.physical_ordered_commit_us as f64,
            t.physical_ordered_commit_us as f64,
        ),
        physical_install_transaction_us: comparison(
            c.physical_install_transaction_us as f64,
            t.physical_install_transaction_us as f64,
        ),
        control_stage_overlap_ratio: overlap(c),
        treatment_stage_overlap_ratio: overlap(t),
        overlap_ratio_definition: "sum of per-expert direct-staging preparation wall times divided by install-set staging critical-path wall; this is host preparation overlap, not hardware DMA parallelism",
        common,
    })
}

pub(crate) async fn run_concurrency_command(
    args: CommandArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let prepared = prepare(&args)?;
    let mut report = PhysicalInstallConcurrencyQualificationReport {
        schema: CONCURRENCY_SCHEMA,
        mode: CONCURRENCY_MODE,
        production_physical_install_concurrency_changed: true,
        normal_production_uses_concurrent_physical_staging: true,
        control_forces_sequential_direct_staging: true,
        treatment_uses_ordinary_production_path: true,
        speculative_physical_install_changed: false,
        both_arms_use_ordinary_production_source: true,
        victim_policy_changed: false,
        slot_assignment_policy_changed: false,
        mapping_publication_order_changed: false,
        source_acquisition_changed: false,
        expert_compute_changed: false,
        adds_queue_submit: false,
        adds_device_poll: false,
        adds_map_async: false,
        adds_readback: false,
        wgpu_version_changed: false,
        rayon_pool_sizing_changed: false,
        pinned_wgpu_audit: PinnedWgpuConcurrencyAudit {
            requested_version: "0.20",
            resolved_version: "0.20.1",
            native_queue_send_sync: true,
            native_buffer_send_sync: true,
            native_queue_write_buffer_view_send_sync: true,
            validates_destination_before_staging_creation: true,
            drop_schedules_buffer_write: true,
            scheduled_writes_begin_at_next_submit: true,
            concurrent_calls_use_internal_pending_write_lock: true,
            distinct_reserved_slot_ranges_do_not_alias: true,
            adds_queue_submit: false,
            adds_device_poll: false,
            adds_map_async: false,
            adds_readback: false,
        },
        rayon_integration: RayonIntegrationAudit {
            existing_process_wide_pool: true,
            creates_new_thread_pool: false,
            spawns_os_threads_per_set: false,
            uses_tokio_spawn_blocking_per_expert: false,
            changes_pool_sizing: false,
            changes_rayon_num_threads: false,
            parallel_width_is_exact_install_set_width: true,
        },
        qwen_physical_slot_geometry: QwenPhysicalSlotGeometry {
            d_model: 2048,
            d_ff: 768,
            q4_block_elements: 32,
            q4_block_bytes: 18,
            logical_expert_bytes: 2_654_208,
            slot_stride_bytes: 2_654_212,
            physical_tail_bytes: 0,
            destination_fill_zero_unchanged: false,
        },
        timing_definitions: concurrency_timing_definitions(),
        benchmark_complete: false,
        qualification_pass: false,
        performance_result: "not_measured",
        failure: None,
        frozen_workload: frozen_workload(args.expected_adapter_name.clone()),
        provenance: prepared.provenance.clone(),
        control: None,
        treatment: None,
        reconciliation: None,
        gates: None,
        performance: None,
    };

    let control = match run_physical_install_arm(
        &prepared,
        &args,
        PhysicalInstallQualificationRun::Concurrency(
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::Control,
        ),
    )
    .await
    {
        Ok(control) => control,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let control_failure = control.common.failure.clone();
    report.control = Some(ConcurrencyArmReport {
        common: control.common,
        warmup_mechanism: control.warmup_concurrency,
        mechanism: control.concurrency,
    });
    if let Some(failure) = control_failure {
        report.failure = Some(failure.clone());
        emit_report(&report, &args.report_out)?;
        return Err(failure.to_string().into());
    }

    let treatment = match run_physical_install_arm(
        &prepared,
        &args,
        PhysicalInstallQualificationRun::Concurrency(
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::Treatment,
        ),
    )
    .await
    {
        Ok(treatment) => treatment,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let treatment_failure = treatment.common.failure.clone();
    report.treatment = Some(ConcurrencyArmReport {
        common: treatment.common,
        warmup_mechanism: treatment.warmup_concurrency,
        mechanism: treatment.concurrency,
    });
    if let Some(failure) = treatment_failure {
        report.failure = Some(failure.clone());
        emit_report(&report, &args.report_out)?;
        return Err(failure.to_string().into());
    }

    let control = report.control.as_ref().expect("control stored");
    let treatment = report.treatment.as_ref().expect("treatment stored");
    let production_and_work = production_reconciliation(
        reconcile(&control.common, &treatment.common),
        &control.common,
        &treatment.common,
    );
    let warmup_generated_text_hashes_exact = control
        .common
        .warmup_results
        .iter()
        .map(|run| &run.generated_text_sha256)
        .eq(treatment
            .common
            .warmup_results
            .iter()
            .map(|run| &run.generated_text_sha256));
    let measured_generated_text_hashes_exact = generated_results(&control.common)
        .iter()
        .map(|run| &run.generated_text_sha256)
        .eq(generated_results(&treatment.common)
            .iter()
            .map(|run| &run.generated_text_sha256));
    let reconciliation = ConcurrencyReconciliation {
        all_invariants_pass: production_and_work.all_invariants_pass
            && warmup_generated_text_hashes_exact
            && measured_generated_text_hashes_exact,
        production_and_work,
        warmup_generated_text_hashes_exact,
        measured_generated_text_hashes_exact,
    };
    let (behavioral, work_equivalence) =
        common_gates(&reconciliation.production_and_work.common);
    let mechanism = concurrency_mechanism_gate(control, treatment);
    let text_hashes_passed = reconciliation.warmup_generated_text_hashes_exact
        && reconciliation.measured_generated_text_hashes_exact;
    let gates = ConcurrencyGates {
        behavioral,
        warmup_generated_text_hashes_exact,
        measured_generated_text_hashes_exact,
        text_hashes_passed,
        work_equivalence,
        mechanism,
    };
    let performance = concurrency_performance(control, treatment)?;
    let qualification_pass = reconciliation.all_invariants_pass
        && gates.behavioral.passed
        && gates.text_hashes_passed
        && gates.work_equivalence.passed
        && gates.mechanism.passed;
    report.benchmark_complete = true;
    report.qualification_pass = qualification_pass;
    report.performance_result = performance.common.performance_result;
    report.reconciliation = Some(reconciliation);
    report.gates = Some(gates);
    report.performance = Some(performance);
    emit_report(&report, &args.report_out)?;
    if qualification_pass {
        Ok(())
    } else {
        Err("PR2-B-B.1 production physical-install concurrency qualification gates did not all pass; see emitted report".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_pr2ba1_production_contract_is_literal_and_versioned_separately() {
        assert_eq!(
            PRODUCTION_SCHEMA,
            "mer.gpu-native-physical-install-staging.v2"
        );
        assert_eq!(
            PRODUCTION_MODE,
            "qualify-gpu-native-physical-install-staging-production"
        );
        assert_eq!(
            FROZEN_CONFIG_PATH,
            "/home/randyap8/slice11-qwen3-coder-gpu-native.toml"
        );
        assert_eq!(
            FROZEN_CONFIG_SHA256,
            "33d7cf96328d9c68b0ff45448d91d597d2e3a757cb99e6e61c72998ceabdd056"
        );
        assert_eq!(FROZEN_OUTPUT_TOKENS, 128);
        assert_eq!(FROZEN_WARMUP_RUNS, 1);
        assert_eq!(FROZEN_MEASURED_RUNS, 3);
        assert_eq!(
            FROZEN_PROMPT,
            "Write a Rust function that adds two i32 values and returns the result."
        );
    }

    #[test]
    fn percentage_delta_preserves_direction_without_a_performance_gate() {
        assert_eq!(comparison(10.0, 12.0).delta_percent, 20.0);
        assert_eq!(comparison(10.0, 8.0).delta_percent, -20.0);
        assert_eq!(comparison(0.0, 8.0).delta_percent, 0.0);
    }

    #[test]
    fn pr2bb1_production_contract_is_literal_and_versioned_separately() {
        assert_eq!(
            CONCURRENCY_SCHEMA,
            "mer.gpu-native-physical-install-concurrency-production.v1"
        );
        assert_eq!(
            CONCURRENCY_MODE,
            "qualify-gpu-native-physical-install-concurrency-production"
        );
        let logical = (2048usize * 768 / 32) * 3 * 18;
        let stride = (4 + logical + 3) & !3;
        assert_eq!(logical, 2_654_208);
        assert_eq!(stride, 2_654_212);
        assert_eq!(stride - 4 - logical, 0);
    }

    #[test]
    fn pr2bb_timing_definitions_separate_per_expert_service_from_set_wall() {
        let definitions = concurrency_timing_definitions();

        assert_eq!(
            definitions.common.physical_install_total_us,
            CONCURRENCY_PHYSICAL_INSTALL_TOTAL_DEFINITION
        );
        assert_eq!(
            definitions.physical_ordered_commit_us,
            CONCURRENCY_PHYSICAL_ORDERED_COMMIT_DEFINITION
        );
        assert_eq!(
            definitions.physical_install_transaction_us,
            CONCURRENCY_PHYSICAL_TRANSACTION_DEFINITION
        );
        assert_eq!(
            definitions.physical_parallel_stage_wall_us,
            CONCURRENCY_PARALLEL_STAGE_WALL_DEFINITION
        );
        assert_eq!(
            definitions.sum_individual_physical_stage_us,
            CONCURRENCY_INDIVIDUAL_STAGE_SUM_DEFINITION
        );
        assert!(!definitions
            .common
            .physical_install_total_us
            .contains("reservation-through"));
        assert!(definitions
            .physical_install_transaction_us
            .contains("complete per-set wall-clock"));
        assert!(definitions
            .sum_individual_physical_stage_us
            .contains("not DMA parallelism"));
    }

    #[test]
    fn pr2bb_timing_metrics_do_not_enter_the_mechanism_gate_schema() {
        let serialized = serde_json::to_value(ConcurrencyMechanismGate {
            warmup_mechanism_reconciled: false,
            control_production_parallel_staging_sets_zero: false,
            treatment_ordinary_production_path_exercised: false,
            treatment_production_parallel_staging_sets_gt_zero: false,
            treatment_production_counters_reconcile: false,
            control_direct_staging_writes_gt_zero: false,
            control_full_slot_vec_materializations_zero: false,
            control_parallel_staging_sets_zero: false,
            control_max_in_flight_lte_one: false,
            treatment_direct_staging_writes_gt_zero: false,
            treatment_full_slot_vec_materializations_zero: false,
            treatment_parallel_staging_sets_gt_zero: false,
            treatment_parallel_staging_experts_gt_zero: false,
            treatment_max_in_flight_gte_two: false,
            reservation_failures_zero: false,
            physical_stage_failures_zero: false,
            ordered_commit_failures_zero: false,
            ordered_commit_violations_zero: false,
            unpublished_physical_writes_after_failure_zero: false,
            attempts_and_completions_reconcile: false,
            staged_bytes_exact: false,
            mapping_publications_exact: false,
            mapping_unpublications_exact: false,
            victim_sequence_exact: false,
            reservation_identity_exact: false,
            ordered_physical_identity_exact: false,
            passed: false,
        })
        .unwrap();
        let fields = serialized.as_object().unwrap();

        assert_eq!(fields.len(), 27);
        assert!(fields.contains_key("warmup_mechanism_reconciled"));
        assert!(fields.contains_key("treatment_ordinary_production_path_exercised"));
        assert!(fields.contains_key("control_parallel_staging_sets_zero"));
        assert!(fields.contains_key("treatment_parallel_staging_sets_gt_zero"));
        assert!(fields.contains_key("attempts_and_completions_reconcile"));
        assert!(fields.contains_key("reservation_identity_exact"));
        assert!(fields.contains_key("ordered_physical_identity_exact"));
        assert!(fields.contains_key("passed"));
        assert!(fields.keys().all(|field| !field.ends_with("_us")));
    }

    #[test]
    fn production_qualifier_fails_closed_without_observed_concurrency() {
        let sequential_only = GpuNativeProductionPhysicalInstallSnapshot {
            physical_install_sets: 2,
            physical_install_experts: 2,
            singleton_staging_sets: 2,
            reservation_attempts: 2,
            reservation_successes: 2,
            physical_install_attempts: 2,
            physical_stage_attempts: 2,
            physical_stage_completions: 2,
            direct_staging_successes: 2,
            ordered_commit_attempts: 2,
            ordered_commit_completions: 2,
            max_in_flight_physical_staging: 1,
            ..GpuNativeProductionPhysicalInstallSnapshot::default()
        };
        assert!(production_physical_install_counters_reconcile(
            &sequential_only
        ));
        assert!(!production_concurrency_exercised(&sequential_only));

        let concurrent = GpuNativeProductionPhysicalInstallSnapshot {
            physical_install_sets: 1,
            physical_install_experts: 2,
            parallel_eligible_sets: 1,
            parallel_staging_sets: 1,
            parallel_staging_experts: 2,
            reservation_attempts: 2,
            reservation_successes: 2,
            physical_install_attempts: 2,
            physical_stage_attempts: 2,
            physical_stage_completions: 2,
            direct_staging_successes: 2,
            ordered_commit_attempts: 2,
            ordered_commit_completions: 2,
            max_in_flight_physical_staging: 2,
            ..GpuNativeProductionPhysicalInstallSnapshot::default()
        };
        assert!(production_concurrency_exercised(&concurrent));
    }
}
