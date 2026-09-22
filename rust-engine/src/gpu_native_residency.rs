//! Tier-aware control plane for the future GPU-native token loop.
//!
//! The engine remains responsible for NVMe reads, RAM-cache ownership,
//! speculative admission, and logical [`GpuExpertCache`] generations. This
//! module owns only the preallocated RAM -> VRAM physical plane built from
//! Slice 8's mutable Q4 expert arenas.

#![allow(dead_code)]

use crate::backend::gpu_native::{
    GpuNativeBootstrapError, GpuNativeExecutorContext, GpuNativePhysicalInstallEvidence,
    GpuNativePhysicalSlotFillPolicy, GpuNativeProductionPhysicalInstallSnapshot,
    GpuNativeQ4ExpertAcquire, GpuNativeQ4ExpertArena, GpuNativeQ4ExpertGeometry,
    GpuNativeQ4ExpertKey, GpuNativeQ4ExpertPreparedInstall, GpuNativeQ4ExpertResidency,
    GpuNativeQ4ExpertRetire, GpuNativeQ4ExpertVramPlan,
};
use crate::backend::gpu_native::{GpuNativeP1jSidecar, P1jClaimRefusal, P1jPhase, P1jPublication};
use crate::expert_cache::{
    ExpertResident, GpuAdmission, GpuExpertCache, HostBackedLease, HostBackedLeaseResult,
};
use crate::predictor_v2::{p1e, P1jIdentity, P1jTerminal};
use lru::LruCache;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn qualification_elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn post_reservation_physical_install_total_us(individual_stage_us: u64, commit_us: u64) -> u64 {
    individual_stage_us.saturating_add(commit_us)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeTieredResidencyError {
    InvalidModelLayerCount,
    InvalidExpertsPerLayer,
    ExpertNamespaceOverflow {
        num_layers: usize,
        experts_per_layer: usize,
    },
    GlobalExpertOutOfRange {
        global_id: u32,
        num_layers: usize,
        experts_per_layer: u32,
    },
    LayerOutOfRange {
        layer_index: usize,
        num_layers: usize,
    },
    LocalExpertOutOfRange {
        local_expert_id: u32,
        experts_per_layer: u32,
    },
    ModelBudgetOverflow,
    ModelBudgetTooSmall {
        requested_bytes: u64,
        minimum_bytes: u64,
    },
    DuplicateDemandExpert {
        global_id: u32,
    },
    DemandLayerMismatch {
        requested_layer: usize,
        global_id: u32,
        actual_layer: usize,
    },
    DemandSetExceedsLayerCapacity {
        requested: usize,
        capacity: usize,
    },
    DemandSourceMissing {
        global_id: u32,
    },
    DemandSourceIdentityMismatch {
        global_id: u32,
    },
    LogicalAdmissionStale {
        global_id: u32,
        generation: u64,
    },
    InstallInProgress {
        global_id: u32,
        generation: u64,
    },
    NoPhysicalSlot {
        layer_index: usize,
    },
    NoEvictablePhysicalSlot {
        layer_index: usize,
    },
    StalePhysicalRequester {
        global_id: u32,
        generation: u64,
    },
    PhysicalIdentityCorrupt {
        global_id: u32,
    },
    UnsafeOracleBoundary {
        layer_index: usize,
        detail: String,
    },
    ResidencyPriorityMismatch,
    Backend(GpuNativeBootstrapError),
}

impl fmt::Display for GpuNativeTieredResidencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelLayerCount => f.write_str("GPU-native residency requires at least one MoE layer"),
            Self::InvalidExpertsPerLayer => f.write_str("GPU-native residency requires at least one expert per layer"),
            Self::ExpertNamespaceOverflow { num_layers, experts_per_layer } => write!(f, "global expert namespace overflows u32: layers={num_layers} experts_per_layer={experts_per_layer}"),
            Self::GlobalExpertOutOfRange { global_id, num_layers, experts_per_layer } => write!(f, "global expert {global_id} is outside {num_layers} layers x {experts_per_layer} experts"),
            Self::LayerOutOfRange { layer_index, num_layers } => write!(f, "layer {layer_index} is outside {num_layers} GPU-native arenas"),
            Self::LocalExpertOutOfRange { local_expert_id, experts_per_layer } => write!(f, "local expert {local_expert_id} is outside layer width {experts_per_layer}"),
            Self::ModelBudgetOverflow => f.write_str("model-wide GPU-native expert budget arithmetic overflowed"),
            Self::ModelBudgetTooSmall { requested_bytes, minimum_bytes } => write!(f, "model-wide expert budget {requested_bytes} bytes is below the executable minimum {minimum_bytes} bytes"),
            Self::DuplicateDemandExpert { global_id } => write!(f, "demand set repeats global expert {global_id}"),
            Self::DemandLayerMismatch { requested_layer, global_id, actual_layer } => write!(f, "demand for layer {requested_layer} contains global expert {global_id} from layer {actual_layer}"),
            Self::DemandSetExceedsLayerCapacity { requested, capacity } => write!(f, "demand set of {requested} experts exceeds physical layer capacity {capacity}"),
            Self::DemandSourceMissing { global_id } => write!(f, "physical miss for global expert {global_id} has no RAM/logical-admission source"),
            Self::DemandSourceIdentityMismatch { global_id } => write!(f, "RAM resident or logical admission does not match global expert {global_id}"),
            Self::LogicalAdmissionStale { global_id, generation } => write!(f, "logical generation {generation} for global expert {global_id} is no longer current"),
            Self::InstallInProgress { global_id, generation } => write!(f, "physical install is already in progress for global expert {global_id} generation {generation}"),
            Self::NoPhysicalSlot { layer_index } => write!(f, "layer {layer_index} has no free physical expert slot"),
            Self::NoEvictablePhysicalSlot { layer_index } => write!(f, "layer {layer_index} has no physical victim outside the protected demand set"),
            Self::StalePhysicalRequester { global_id, generation } => write!(f, "stale physical requester for global expert {global_id} generation {generation}"),
            Self::PhysicalIdentityCorrupt { global_id } => write!(f, "GPU-native physical metadata disagrees with the authoritative arena for global expert {global_id}"),
            Self::UnsafeOracleBoundary { layer_index, detail } => write!(f, "ORACLE-0B-S safe-boundary authorization failed for layer {layer_index}: {detail}"),
            Self::ResidencyPriorityMismatch => f.write_str("residency request used the wrong demand/speculative priority"),
            Self::Backend(error) => write!(f, "GPU-native residency backend error: {error}"),
        }
    }
}

impl std::error::Error for GpuNativeTieredResidencyError {}

impl From<GpuNativeBootstrapError> for GpuNativeTieredResidencyError {
    fn from(value: GpuNativeBootstrapError) -> Self {
        Self::Backend(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GpuNativeLayerExpertId {
    pub(crate) layer_index: usize,
    pub(crate) local_expert_id: u32,
}

pub(crate) fn global_to_layer_local(
    global_id: u32,
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<GpuNativeLayerExpertId, GpuNativeTieredResidencyError> {
    validate_namespace(num_layers, experts_per_layer as usize)?;
    let layer_index = (global_id / experts_per_layer) as usize;
    if layer_index >= num_layers {
        return Err(GpuNativeTieredResidencyError::GlobalExpertOutOfRange {
            global_id,
            num_layers,
            experts_per_layer,
        });
    }
    Ok(GpuNativeLayerExpertId {
        layer_index,
        local_expert_id: global_id % experts_per_layer,
    })
}

pub(crate) fn layer_local_to_global(
    layer_index: usize,
    local_expert_id: u32,
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<u32, GpuNativeTieredResidencyError> {
    validate_namespace(num_layers, experts_per_layer as usize)?;
    if layer_index >= num_layers {
        return Err(GpuNativeTieredResidencyError::LayerOutOfRange {
            layer_index,
            num_layers,
        });
    }
    if local_expert_id >= experts_per_layer {
        return Err(GpuNativeTieredResidencyError::LocalExpertOutOfRange {
            local_expert_id,
            experts_per_layer,
        });
    }
    let global = (layer_index as u64)
        .checked_mul(experts_per_layer as u64)
        .and_then(|base| base.checked_add(local_expert_id as u64))
        .and_then(|id| u32::try_from(id).ok())
        .ok_or(GpuNativeTieredResidencyError::ExpertNamespaceOverflow {
            num_layers,
            experts_per_layer: experts_per_layer as usize,
        })?;
    Ok(global)
}

pub(crate) fn global_to_q4_expert_key(
    global_id: u32,
    logical_generation: u64,
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<GpuNativeQ4ExpertKey, GpuNativeTieredResidencyError> {
    let identity = global_to_layer_local(global_id, num_layers, experts_per_layer)?;
    Ok(GpuNativeQ4ExpertKey::new(
        identity.layer_index,
        identity.local_expert_id,
        logical_generation,
    ))
}

fn validate_namespace(
    num_layers: usize,
    experts_per_layer: usize,
) -> Result<(), GpuNativeTieredResidencyError> {
    if num_layers == 0 {
        return Err(GpuNativeTieredResidencyError::InvalidModelLayerCount);
    }
    if experts_per_layer == 0 {
        return Err(GpuNativeTieredResidencyError::InvalidExpertsPerLayer);
    }
    let count = (num_layers as u128)
        .checked_mul(experts_per_layer as u128)
        .ok_or(GpuNativeTieredResidencyError::ExpertNamespaceOverflow {
            num_layers,
            experts_per_layer,
        })?;
    if count > u32::MAX as u128 + 1 {
        return Err(GpuNativeTieredResidencyError::ExpertNamespaceOverflow {
            num_layers,
            experts_per_layer,
        });
    }
    Ok(())
}

/// One deterministic model-wide post-headroom expert-VRAM plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeModelExpertVramPlan {
    geometry: GpuNativeQ4ExpertGeometry,
    total_expert_budget_bytes: u64,
    minimum_executable_budget_bytes: u64,
    layer_plans: Vec<GpuNativeQ4ExpertVramPlan>,
    total_arena_allocation_bytes: u64,
}

impl GpuNativeModelExpertVramPlan {
    pub(crate) fn try_new(
        num_layers: usize,
        geometry: GpuNativeQ4ExpertGeometry,
        total_expert_budget_bytes: u64,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeTieredResidencyError> {
        validate_namespace(num_layers, geometry.num_experts())?;
        let minimum_layer =
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, geometry.top_k(), limits)?;
        let minimum_executable_budget_bytes = minimum_layer
            .total_arena_allocation_bytes()
            .checked_mul(num_layers as u64)
            .ok_or(GpuNativeTieredResidencyError::ModelBudgetOverflow)?;
        if total_expert_budget_bytes < minimum_executable_budget_bytes {
            return Err(GpuNativeTieredResidencyError::ModelBudgetTooSmall {
                requested_bytes: total_expert_budget_bytes,
                minimum_bytes: minimum_executable_budget_bytes,
            });
        }

        let mut layer_plans = vec![minimum_layer; num_layers];
        let mut total_arena_allocation_bytes = minimum_executable_budget_bytes;
        loop {
            let remaining = total_expert_budget_bytes - total_arena_allocation_bytes;
            let mut best: Option<(u64, usize, GpuNativeQ4ExpertVramPlan)> = None;
            for (layer_index, current) in layer_plans.iter().copied().enumerate() {
                if current.slot_capacity() >= geometry.num_experts() {
                    continue;
                }
                let candidate = GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(
                    geometry,
                    current.slot_capacity() + 1,
                    limits,
                )?;
                let increment = candidate
                    .total_arena_allocation_bytes()
                    .checked_sub(current.total_arena_allocation_bytes())
                    .ok_or(GpuNativeTieredResidencyError::ModelBudgetOverflow)?;
                if increment > remaining {
                    continue;
                }
                if best.as_ref().is_none_or(|(best_increment, best_layer, _)| {
                    (increment, layer_index) < (*best_increment, *best_layer)
                }) {
                    best = Some((increment, layer_index, candidate));
                }
            }
            let Some((increment, layer_index, candidate)) = best else {
                break;
            };
            layer_plans[layer_index] = candidate;
            total_arena_allocation_bytes = total_arena_allocation_bytes
                .checked_add(increment)
                .ok_or(GpuNativeTieredResidencyError::ModelBudgetOverflow)?;
        }

        debug_assert!(total_arena_allocation_bytes <= total_expert_budget_bytes);
        Ok(Self {
            geometry,
            total_expert_budget_bytes,
            minimum_executable_budget_bytes,
            layer_plans,
            total_arena_allocation_bytes,
        })
    }

    pub(crate) const fn geometry(&self) -> GpuNativeQ4ExpertGeometry {
        self.geometry
    }

    pub(crate) const fn total_expert_budget_bytes(&self) -> u64 {
        self.total_expert_budget_bytes
    }

    pub(crate) const fn minimum_executable_budget_bytes(&self) -> u64 {
        self.minimum_executable_budget_bytes
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layer_plans.len()
    }

    pub(crate) fn layer_plans(&self) -> &[GpuNativeQ4ExpertVramPlan] {
        &self.layer_plans
    }

    pub(crate) const fn total_arena_allocation_bytes(&self) -> u64 {
        self.total_arena_allocation_bytes
    }

    pub(crate) const fn unused_remainder_bytes(&self) -> u64 {
        self.total_expert_budget_bytes - self.total_arena_allocation_bytes
    }

    pub(crate) fn model_slot_capacity(&self) -> usize {
        self.layer_plans
            .iter()
            .map(|plan| plan.slot_capacity())
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GpuNativeResidencyPriority {
    Demand,
    OracleSafeBoundary,
    Speculative { score: f64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DemandPhysicalInstallPath {
    ProductionConcurrentDirectStaging, // Complete overwrite, no explicit zero fill.
    QualificationConcurrentDirectStagingFullZero,
    QualificationConcurrentDirectStagingNoZeroFill,
    SequentialDirectStagingControl,
    LegacyFullSlotVecControl,
}

const fn ordinary_demand_install_path() -> DemandPhysicalInstallPath {
    DemandPhysicalInstallPath::ProductionConcurrentDirectStaging
}

const fn oracle_safe_boundary_install_path(
    fill_policy: GpuNativePhysicalSlotFillPolicy,
) -> DemandPhysicalInstallPath {
    match fill_policy {
        GpuNativePhysicalSlotFillPolicy::FullSlotZero => {
            DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
        }
        GpuNativePhysicalSlotFillPolicy::QualificationNoZeroFill => {
            DemandPhysicalInstallPath::QualificationConcurrentDirectStagingNoZeroFill
        }
    }
}

const fn production_concurrency_control_path() -> DemandPhysicalInstallPath {
    DemandPhysicalInstallPath::SequentialDirectStagingControl
}

const fn speculative_install_path() -> DemandPhysicalInstallPath {
    DemandPhysicalInstallPath::LegacyFullSlotVecControl
}

/// Qualification observer kept outside ordinary serving. It receives only
/// already-committed physical events in their production order, except that a
/// direct-staging unavailability is recorded explicitly before failing.
pub(crate) trait GpuNativePhysicalInstallObserver: Send + Sync {
    fn source_upload_state(&self) -> Option<&crate::gpu_native_source_upload::State> {
        None
    }
    fn record_physical_victim(&self, global_id: u32);
    fn record_physical_install_attempt(&self);
    fn record_direct_staging_failure(&self);
    fn record_physical_install_completion(
        &self,
        global_id: u32,
        residency: GpuNativeQ4ExpertResidency,
        evidence: GpuNativePhysicalInstallEvidence,
        physical_install_total_us: u64,
    );
    fn record_physical_install_set(
        &self,
        width: usize,
        parallel: bool,
        caller_in_rayon_worker: bool,
        rayon_threads: usize,
    );
    fn record_reservation_attempt(&self);
    fn record_reservation_success(
        &self,
        global_id: u32,
        residency: GpuNativeQ4ExpertResidency,
        install_ticket: u64,
    );
    fn record_reservation_failure(&self);
    fn record_physical_stage_started(&self);
    fn record_physical_stage_completed(
        &self,
        evidence: GpuNativePhysicalInstallEvidence,
        individual_stage_us: u64,
    );
    fn record_physical_stage_failed(&self);
    fn record_parallel_stage_wall(&self, wall_us: u64);
    fn record_ordered_commit_attempt(&self);
    fn record_ordered_commit_completed(&self, commit_us: u64);
    fn record_ordered_commit_failed(&self, violation: bool);
    fn record_physical_reservation_wall(&self, wall_us: u64);
    fn record_physical_install_transaction_wall(&self, wall_us: u64);
    fn record_unpublished_physical_writes_after_failure(&self, count: u64);
}

struct ReservedPhysicalInstall<'a> {
    demand_index: usize,
    global_id: u32,
    resident: Arc<ExpertResident>,
    admission: GpuAdmission,
    key: GpuNativeQ4ExpertKey,
    permit: crate::backend::gpu_native::GpuNativeQ4ExpertInstallPermit<'a>,
}

struct StagedPhysicalInstall<'a> {
    demand_index: usize,
    global_id: u32,
    admission: GpuAdmission,
    key: GpuNativeQ4ExpertKey,
    prepared: GpuNativeQ4ExpertPreparedInstall<'a>,
    evidence: GpuNativePhysicalInstallEvidence,
    individual_stage_us: u64,
}

/// Rayon indexed collection retains the input order even when workers finish
/// out of order. Keeping this seam small makes that ordering contract directly
/// testable and ensures the task count is exactly the install-set width.
fn collect_physical_stage_results_in_request_order<T, R, F>(
    reserved: Vec<T>,
    parallel: bool,
    stage_one: F,
) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(T) -> R + Send + Sync,
{
    if parallel {
        use rayon::prelude::*;
        reserved.into_par_iter().map(stage_one).collect()
    } else {
        reserved.into_iter().map(stage_one).collect()
    }
}

#[derive(Clone)]
pub(crate) enum GpuNativeDemandExpert {
    Current {
        global_id: u32,
    },
    Install {
        global_id: u32,
        resident: Arc<ExpertResident>,
        admission: GpuAdmission,
    },
}

impl GpuNativeDemandExpert {
    pub(crate) const fn current(global_id: u32) -> Self {
        Self::Current { global_id }
    }

    pub(crate) fn install(
        global_id: u32,
        resident: Arc<ExpertResident>,
        admission: GpuAdmission,
    ) -> Self {
        Self::Install {
            global_id,
            resident,
            admission,
        }
    }

    const fn global_id(&self) -> u32 {
        match self {
            Self::Current { global_id } | Self::Install { global_id, .. } => *global_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeSpeculativeProbe {
    Hit(GpuNativeQ4ExpertResidency),
    Miss,
    DroppedPressure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeSpeculativeInstall {
    Hit(GpuNativeQ4ExpertResidency),
    Installed(GpuNativeQ4ExpertResidency),
    DroppedCapacityOrPressure,
    StaleLogicalGeneration,
}

#[derive(Clone, Copy)]
struct PhysicalRecord {
    key: GpuNativeQ4ExpertKey,
    residency: GpuNativeQ4ExpertResidency,
}

struct LayerResidencyState {
    p1e_shadow: Option<Box<crate::predictor_v2::p1e::Shadow>>,
    residents: LruCache<u32, PhysicalRecord>,
    last_installed_generations: HashMap<u32, u64>,
    physical_evictions: u64,
}

impl Default for LayerResidencyState {
    fn default() -> Self {
        Self {
            residents: LruCache::unbounded(),
            p1e_shadow: None,
            last_installed_generations: HashMap::new(),
            physical_evictions: 0,
        }
    }
}

/// Copy LRU to MRU through an immutable reference. The verifier is used only
/// inside this module to check an existing record, never to fetch or touch it.
fn copy_p1e_snapshot<T>(
    namespace: crate::predictor_v2::p1e::Namespace,
    residents: &LruCache<u32, T>,
    arena_resident_count: usize,
    verify: impl Fn(u32, &T) -> Option<crate::predictor_v2::p1e::Resident>,
) -> Result<crate::predictor_v2::p1e::PhysicalSnapshot, crate::predictor_v2::p1e::Error> {
    use crate::predictor_v2::p1e::{Error, PhysicalSnapshot, CAPACITY, EXPERTS, LAYER};
    if residents.len() > CAPACITY || arena_resident_count != residents.len() {
        return Err(Error::PhysicalEvidence);
    }
    let mut copied = [None; CAPACITY];
    for (dst, (&global_id, record)) in copied.iter_mut().zip(residents.iter().rev()) {
        let resident = verify(global_id, record).ok_or(Error::PhysicalEvidence)?;
        if global_id as usize != LAYER * EXPERTS + resident.expert as usize {
            return Err(Error::PhysicalEvidence);
        }
        *dst = Some(resident);
    }
    let snapshot = PhysicalSnapshot {
        namespace,
        residents: copied,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn p1e_resident(record: PhysicalRecord) -> crate::predictor_v2::p1e::Resident {
    crate::predictor_v2::p1e::Resident {
        expert: record.key.expert_id(),
        generation: record.key.logical_generation(),
        bank: record.residency.location().bank(),
        slot: record.residency.location().slot(),
        epoch: record.residency.slot_epoch(),
    }
}

fn touch_physical_record<T: Copy>(residents: &mut LruCache<u32, T>, global_id: u32) -> Option<T> {
    residents.get(&global_id).copied()
}

fn validate_physical_install_source(
    gpu_cache: &GpuExpertCache,
    global_id: u32,
    resident: &Arc<ExpertResident>,
    admission: &GpuAdmission,
) -> Result<(), GpuNativeTieredResidencyError> {
    if resident.id != global_id || admission.resident().id != global_id {
        return Err(GpuNativeTieredResidencyError::DemandSourceIdentityMismatch { global_id });
    }
    if !gpu_cache.contains_generation(global_id, admission.generation()) {
        return Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
            global_id,
            generation: admission.generation(),
        });
    }
    Ok(())
}

struct LayerResidency {
    arena: Arc<GpuNativeQ4ExpertArena>,
    state: Mutex<LayerResidencyState>,
}

#[derive(Default)]
struct TieredResidencyCounters {
    vram_hits: AtomicU64,
    vram_misses: AtomicU64,
    physical_current_hits: AtomicU64,
    physical_source_acquisitions: AtomicU64,
    logical_admissions_for_physical_misses: AtomicU64,
    ram_to_vram_installs: AtomicU64,
    physical_evictions: AtomicU64,
    physical_reinstalls: AtomicU64,
    stale_generation_rejections: AtomicU64,
    demand_requests: AtomicU64,
    speculative_requests: AtomicU64,
    speculative_vram_hits: AtomicU64,
    speculative_ram_to_vram_installs: AtomicU64,
    speculative_dropped_capacity_or_pressure: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeTieredLayerSnapshot {
    pub(crate) slot_capacity: usize,
    pub(crate) resident_slots: usize,
    pub(crate) free_slots: usize,
    pub(crate) physical_evictions: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeTieredResidencySnapshot {
    pub(crate) model_expert_budget_bytes: u64,
    pub(crate) model_arena_allocation_bytes: u64,
    pub(crate) model_slot_capacity: usize,
    pub(crate) resident_physical_slots: usize,
    pub(crate) free_physical_slots: usize,
    pub(crate) vram_hits: u64,
    pub(crate) vram_misses: u64,
    pub(crate) physical_current_hits: u64,
    pub(crate) physical_source_acquisitions: u64,
    pub(crate) logical_admissions_for_physical_misses: u64,
    pub(crate) ram_to_vram_installs: u64,
    pub(crate) physical_evictions: u64,
    pub(crate) physical_reinstalls: u64,
    pub(crate) stale_generation_rejections: u64,
    pub(crate) demand_requests: u64,
    pub(crate) speculative_requests: u64,
    pub(crate) speculative_vram_hits: u64,
    pub(crate) speculative_ram_to_vram_installs: u64,
    pub(crate) speculative_dropped_capacity_or_pressure: u64,
    pub(crate) layers: Vec<GpuNativeTieredLayerSnapshot>,
}

/// Read-only setup evidence; local namespace IDs are retained for audit and
/// normalized only by the performance driver's structural comparison.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct P1qResourceSnapshot {
    pub(crate) sidecar_initialized: bool,
    pub(crate) sidecar_allocated: bool,
    pub(crate) sidecar_stride_bytes: u64,
    pub(crate) sidecar_bank: u32,
    pub(crate) sidecar_slot: u32,
    pub(crate) namespace: Option<p1e::Namespace>,
    pub(crate) ordinary_total_expert_budget_bytes: u64,
    pub(crate) ordinary_arena_allocation_bytes: u64,
    pub(crate) ordinary_layer_capacities: Vec<usize>,
    pub(crate) ordinary_layer_resident_counts: Vec<usize>,
    pub(crate) p1e_shadow_present: bool,
    pub(crate) activity_counters: [u64; 14],
}

/// Model-scoped owner of the preallocated per-layer Q4 expert arenas.
pub(crate) struct GpuNativeTieredResidencyManager {
    executor: Arc<GpuNativeExecutorContext>,
    gpu_cache: Arc<GpuExpertCache>,
    plan: GpuNativeModelExpertVramPlan,
    layers: Vec<LayerResidency>,
    counters: TieredResidencyCounters,
    p1j: std::sync::OnceLock<Result<Arc<P1jSidecarOwner>, String>>,
}

/// One model-owned resource. No request history or ordinary capacity lives here.
pub(crate) struct P1jSidecarOwner {
    executor: Arc<GpuNativeExecutorContext>,
    arena: Arc<GpuNativeQ4ExpertArena>,
    resource: GpuNativeP1jSidecar,
    namespace: p1e::Namespace,
    runtime: tokio::runtime::Handle,
    retirement: P1jRetirement,
}

/// One latest scalar cleanup request, not a queue of predictions. Epochs are
/// unique for this model resource, so an older cancellation cannot replace it.
#[derive(Default)]
struct P1jRetirement {
    requested: AtomicU64,
    running: std::sync::atomic::AtomicBool,
}

impl P1jRetirement {
    fn request(&self, epoch: u32, reason: P1jTerminal) {
        let code = match reason {
            P1jTerminal::UsedMatching => 1,
            P1jTerminal::Unused => 2,
            P1jTerminal::StaleLogicalGeneration => 3,
            P1jTerminal::OrdinarySuperseded => 4,
            P1jTerminal::Cancelled => 5,
            P1jTerminal::RequestEnded => 6,
            P1jTerminal::WriteFailure => 7,
        };
        let packed = (u64::from(epoch) << 8) | code;
        let _ = self
            .requested
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                (old >> 8 < u64::from(epoch)).then_some(packed)
            });
    }

    fn cancelled(&self, epoch: u32) -> bool {
        self.requested.load(Ordering::Acquire) >> 8 >= u64::from(epoch)
    }

    fn start(&self) -> bool {
        self.running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Called only by a blocking worker. Clearing the dispatch flag before
    /// rechecking the request closes the lost-wakeup window: either this worker
    /// handles a newer cancellation or its caller owns the next dispatch.
    fn drain(&self, mut retire: impl FnMut(u32, P1jTerminal)) {
        loop {
            let packed = self.requested.load(Ordering::Acquire);
            let reason = match packed & 0xff {
                1 => P1jTerminal::UsedMatching,
                2 => P1jTerminal::Unused,
                3 => P1jTerminal::StaleLogicalGeneration,
                4 => P1jTerminal::OrdinarySuperseded,
                5 => P1jTerminal::Cancelled,
                6 => P1jTerminal::RequestEnded,
                7 => P1jTerminal::WriteFailure,
                _ => unreachable!("cleanup dispatch follows a retirement request"),
            };
            retire((packed >> 8) as u32, reason);
            self.running.store(false, Ordering::Release);
            if self.requested.load(Ordering::Acquire) == packed || !self.start() {
                break;
            }
        }
    }
}

/// Existing prepare stages, in their original evaluation order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum P1jPrepareRefusal {
    RetirementBusy,
    CandidateIdentityInvalid,
    FreezeIncomplete,
    CandidateAlreadyCurrentAtF,
    PhysicalEvidenceMissingOrInvalid,
    NotLogicalMaterialized,
    MissingLogicalGeneration,
    HostLeaseBusy,
    HostLeaseMissing,
    HostLeaseStale,
    HostLeaseWrongPayloadKind,
    HostLeaseWrongDtype,
    HostLeaseWrongLength,
    SidecarLockBusy,
    SidecarOccupied,
    SidecarIdentityRejected,
    SidecarEpochExhausted,
    SidecarWriterSequenceExhausted,
}

impl From<P1jClaimRefusal> for P1jPrepareRefusal {
    fn from(value: P1jClaimRefusal) -> Self {
        match value {
            P1jClaimRefusal::LockBusy => Self::SidecarLockBusy,
            P1jClaimRefusal::Occupied => Self::SidecarOccupied,
            P1jClaimRefusal::IdentityRejected => Self::SidecarIdentityRejected,
            P1jClaimRefusal::EpochExhausted => Self::SidecarEpochExhausted,
            P1jClaimRefusal::WriterSequenceExhausted => Self::SidecarWriterSequenceExhausted,
        }
    }
}

fn p1j_host_lease(result: HostBackedLeaseResult) -> Result<HostBackedLease, P1jPrepareRefusal> {
    match result {
        HostBackedLeaseResult::Acquired(lease) => Ok(lease),
        HostBackedLeaseResult::Busy => Err(P1jPrepareRefusal::HostLeaseBusy),
        HostBackedLeaseResult::Missing => Err(P1jPrepareRefusal::HostLeaseMissing),
        HostBackedLeaseResult::Stale => Err(P1jPrepareRefusal::HostLeaseStale),
        HostBackedLeaseResult::WrongPayloadKind => {
            Err(P1jPrepareRefusal::HostLeaseWrongPayloadKind)
        }
        HostBackedLeaseResult::WrongDtype => Err(P1jPrepareRefusal::HostLeaseWrongDtype),
        HostBackedLeaseResult::WrongLength => Err(P1jPrepareRefusal::HostLeaseWrongLength),
    }
}

// Scalar-only validation shared by production and CPU fixtures. Keep the
// original F/source/physical/generation order; never acquire a source here.
pub(crate) fn p1j_prepare_generation(
    namespace: p1e::Namespace,
    freeze: p1e::Freeze,
) -> Result<u64, P1jPrepareRefusal> {
    use P1jPrepareRefusal::*;
    let c = freeze.candidate;
    if c.namespace != namespace
        || c.source_layer != 47
        || c.target_layer != 47
        || c.position_distance != 1
        || c.source_position.absolute_position.checked_add(1)
            != Some(c.target_position.absolute_position)
    {
        return Err(CandidateIdentityInvalid);
    }
    if freeze.incomplete.is_some() || freeze.current.is_none() {
        return Err(FreezeIncomplete);
    }
    if freeze.current != Some(false) {
        return Err(CandidateAlreadyCurrentAtF);
    }
    if !freeze.source.logical_materialized {
        return Err(NotLogicalMaterialized);
    }
    let physical = freeze.physical.ok_or(PhysicalEvidenceMissingOrInvalid)?;
    if physical
        .snapshot
        .current(c.namespace, c.expert)
        .map_err(|_| PhysicalEvidenceMissingOrInvalid)?
    {
        return Err(PhysicalEvidenceMissingOrInvalid);
    }
    freeze
        .source
        .logical_generation
        .ok_or(MissingLogicalGeneration)
}

// The exact real prepare gates also accept a CPU arena claim in software tests.
pub(crate) fn p1j_prepare_source(
    retirement_busy: bool,
    namespace: p1e::Namespace,
    freeze: p1e::Freeze,
    cache: &GpuExpertCache,
    claim: impl FnOnce(p1e::Candidate, u64) -> Result<P1jIdentity, P1jClaimRefusal>,
) -> Result<(P1jIdentity, HostBackedLease), P1jPrepareRefusal> {
    if retirement_busy {
        return Err(P1jPrepareRefusal::RetirementBusy);
    }
    let c = freeze.candidate;
    let generation = p1j_prepare_generation(namespace, freeze)?;
    let lease = p1j_host_lease(cache.try_lease_host_backed((47 * 128) + c.expert, generation))?;
    let id = claim(c, generation)?;
    debug_assert_eq!(lease.global_id(), id.global_id());
    debug_assert_eq!(lease.generation(), id.logical_generation);
    Ok((id, lease))
}

pub(crate) struct P1jWriter {
    owner: Arc<P1jSidecarOwner>,
    pub(crate) id: P1jIdentity,
    lease: Option<HostBackedLease>,
    closed: bool,
}

/// Shared by the real worker and CPU queue doubles. Even an unwinding write
/// drops its view before closure is published, then relinquishes the source.
fn p1j_stage_lease(
    lease: HostBackedLease,
    write: impl FnOnce(&[u8]) -> Result<(), GpuNativeBootstrapError>,
    close: impl FnOnce(bool),
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(lease.data())));
    close(matches!(result, Ok(Ok(()))));
    drop(lease);
}

impl P1jWriter {
    pub(crate) fn spawn(self) {
        let runtime = self.owner.runtime.clone();
        // Claim and lease already exist before dispatch. There is no prediction queue.
        runtime.spawn_blocking(move || {
            let mut writer = self;
            let lease = writer.lease.take().expect("writer owns source");
            let owner = &writer.owner;
            p1j_stage_lease(
                lease,
                |payload| {
                    owner
                        .executor
                        .write_p1j_payload(&owner.resource, writer.id, payload)
                },
                |success| {
                    if !success {
                        owner
                            .retirement
                            .request(writer.id.epoch, P1jTerminal::WriteFailure);
                    }
                    owner.arena.close_p1j_writer(writer.id, success);
                },
            );
            writer.closed = true;
        });
    }
    fn close(&mut self, success: bool) {
        if self.closed {
            return;
        }
        if !success {
            self.owner
                .retirement
                .request(self.id.epoch, P1jTerminal::WriteFailure);
        }
        if !self.owner.arena.try_close_p1j_writer(self.id, success) {
            // A never-dispatched writer can be dropped on the foreground path.
            // Its enqueue capability is gone; keep WRITING claimed until this
            // scalar-only closer runs. There can be only one such closer.
            let owner = self.owner.clone();
            let id = self.id;
            self.owner.runtime.spawn_blocking(move || {
                owner.arena.close_p1j_writer(id, success);
            });
        }
        // No code may access source bytes after ENQUEUED_CLOSED. Drop immediately.
        drop(self.lease.take());
        self.closed = true;
    }
}

impl Drop for P1jWriter {
    fn drop(&mut self) {
        self.close(false);
    }
}

impl P1jSidecarOwner {
    pub(crate) fn try_prepare(
        self: &Arc<Self>,
        freeze: p1e::Freeze,
        cache: &GpuExpertCache,
    ) -> Result<P1jWriter, P1jPrepareRefusal> {
        let (id, lease) = p1j_prepare_source(
            self.retirement.running.load(Ordering::Acquire),
            self.namespace,
            freeze,
            cache,
            |c, generation| self.arena.try_claim_p1j(c, generation),
        )?;
        Ok(P1jWriter {
            owner: self.clone(),
            id,
            lease: Some(lease),
            closed: false,
        })
    }

    pub(crate) fn try_publish(&self, id: P1jIdentity, cache: &GpuExpertCache) -> P1jPublication {
        if id.candidate.namespace != self.namespace || self.cancelled(id) {
            return P1jPublication::IdentityMismatch;
        }
        cache
            .try_with_host_generation(id.global_id(), id.logical_generation, |current| {
                self.executor.try_publish_p1j(&self.arena, id, current)
            })
            .unwrap_or(P1jPublication::Busy)
    }

    pub(crate) fn cancelled(&self, id: P1jIdentity) -> bool {
        self.retirement.cancelled(id.epoch)
    }

    pub(crate) fn phase(&self, id: P1jIdentity) -> Option<P1jPhase> {
        self.arena.try_p1j_phase(id)
    }
    pub(crate) fn resource(&self) -> &GpuNativeP1jSidecar {
        &self.resource
    }

    /// Foreground cleanup never waits. At most one deferred cleanup may exist
    /// while this exact occupant prevents destination reuse; it performs only
    /// conditional unpublication, never payload writes or mapping publication.
    pub(crate) fn retire(self: &Arc<Self>, id: P1jIdentity, reason: P1jTerminal) {
        if id.candidate.namespace != self.namespace {
            return;
        }
        self.retirement.request(id.epoch, reason);
        if self
            .executor
            .retire_p1j(&self.arena, id, reason, true)
            .is_some()
        {
            return;
        }
        if !self.retirement.start() {
            return;
        }
        let owner = self.clone();
        self.runtime.spawn_blocking(move || {
            owner.retirement.drain(|epoch, reason| {
                owner
                    .executor
                    .retire_p1j_epoch(&owner.arena, owner.namespace, epoch, reason);
            });
        });
    }

    fn current(&self, global: u32) -> Option<(P1jIdentity, GpuNativeQ4ExpertResidency)> {
        let (id, residency) = self.arena.p1j_current()?;
        (id.global_id() == global
            && id.candidate.namespace == self.namespace
            && !self.cancelled(id))
        .then_some((id, residency))
    }
}

impl GpuNativeTieredResidencyManager {
    pub(crate) fn try_new(
        executor: Arc<GpuNativeExecutorContext>,
        gpu_cache: Arc<GpuExpertCache>,
        num_layers: usize,
        geometry: GpuNativeQ4ExpertGeometry,
        total_expert_budget_bytes: u64,
    ) -> Result<Self, GpuNativeTieredResidencyError> {
        let limits = executor.device_limits()?;
        let plan = GpuNativeModelExpertVramPlan::try_new(
            num_layers,
            geometry,
            total_expert_budget_bytes,
            &limits,
        )?;
        let mut layers = Vec::with_capacity(num_layers);
        for (layer_index, layer_plan) in plan.layer_plans().iter().copied().enumerate() {
            let arena = executor.create_q4_expert_arena(layer_index, layer_plan, &[])?;
            layers.push(LayerResidency {
                arena: Arc::new(arena),
                state: Mutex::new(LayerResidencyState::default()),
            });
        }
        Ok(Self {
            executor,
            gpu_cache,
            plan,
            layers,
            counters: TieredResidencyCounters::default(),
            p1j: std::sync::OnceLock::new(),
        })
    }

    /// Explicit internal future-qualification opt-in. No ordinary constructor,
    /// observer, CLI, environment variable or serving path calls this method.
    pub(crate) fn enable_p1j_sidecar(&self, runtime: u64) -> Result<Arc<P1jSidecarOwner>, String> {
        let namespace = self.p1e_namespace(runtime).map_err(|e| format!("{e:?}"))?;
        let result = self.p1j.get_or_init(|| {
            let handle = tokio::runtime::Handle::try_current().map_err(|e| e.to_string())?;
            let arena = self.layers[47].arena.clone();
            let resource = self
                .executor
                .create_p1j_sidecar(&arena)
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(P1jSidecarOwner {
                executor: self.executor.clone(),
                arena,
                resource,
                namespace,
                runtime: handle,
                retirement: P1jRetirement::default(),
            }))
        });
        let owner = result.as_ref().map_err(Clone::clone)?;
        if owner.namespace != namespace {
            return Err("P1J model/runtime namespace mismatch".into());
        }
        Ok(owner.clone())
    }

    /// Bounded CPU-only setup snapshot. Contention fails closed; this never
    /// touches an LRU, waits for a writer, allocates a GPU resource or submits.
    pub(crate) fn p1q_resource_snapshot(&self) -> Result<P1qResourceSnapshot, String> {
        let mut resident_counts = Vec::with_capacity(self.layers.len());
        let mut p1e_shadow_present = false;
        for (index, layer) in self.layers.iter().enumerate() {
            let state = layer
                .state
                .try_lock()
                .ok_or("P1Q residency snapshot busy")?;
            resident_counts.push(state.residents.len());
            if index == p1e::LAYER {
                p1e_shadow_present = state.p1e_shadow.is_some();
            }
        }
        let owner = self
            .p1j
            .get()
            .map(|v| v.as_ref().map_err(Clone::clone))
            .transpose()?;
        Ok(P1qResourceSnapshot {
            sidecar_initialized: self.p1j.get().is_some(),
            sidecar_allocated: owner.is_some(),
            // Successful allocation validates precisely this frozen backend layout.
            sidecar_stride_bytes: crate::backend::gpu_native::P1J_STRIDE_BYTES as u64,
            sidecar_bank: 1,
            sidecar_slot: 0,
            namespace: owner.map(|o| o.namespace),
            ordinary_total_expert_budget_bytes: self.plan.total_expert_budget_bytes(),
            ordinary_arena_allocation_bytes: self.plan.total_arena_allocation_bytes(),
            ordinary_layer_capacities: self
                .plan
                .layer_plans()
                .iter()
                .map(|p| p.slot_capacity())
                .collect(),
            ordinary_layer_resident_counts: resident_counts,
            p1e_shadow_present,
            activity_counters: [
                self.counters.vram_hits.load(Ordering::Relaxed),
                self.counters.vram_misses.load(Ordering::Relaxed),
                self.counters.physical_current_hits.load(Ordering::Relaxed),
                self.counters
                    .physical_source_acquisitions
                    .load(Ordering::Relaxed),
                self.counters
                    .logical_admissions_for_physical_misses
                    .load(Ordering::Relaxed),
                self.counters.ram_to_vram_installs.load(Ordering::Relaxed),
                self.counters.physical_evictions.load(Ordering::Relaxed),
                self.counters.physical_reinstalls.load(Ordering::Relaxed),
                self.counters
                    .stale_generation_rejections
                    .load(Ordering::Relaxed),
                self.counters.demand_requests.load(Ordering::Relaxed),
                self.counters.speculative_requests.load(Ordering::Relaxed),
                self.counters.speculative_vram_hits.load(Ordering::Relaxed),
                self.counters
                    .speculative_ram_to_vram_installs
                    .load(Ordering::Relaxed),
                self.counters
                    .speculative_dropped_capacity_or_pressure
                    .load(Ordering::Relaxed),
            ],
        })
    }

    fn p1j_current(&self, global: u32) -> Option<(P1jIdentity, GpuNativeQ4ExpertResidency)> {
        if global / 128 != 47 {
            return None;
        }
        self.p1j.get()?.as_ref().ok()?.current(global)
    }

    /// Nonblocking variant only for the opted-in P1J D seam. P1E's ordinary
    /// observation path remains unchanged. A busy read marks evidence unavailable.
    pub(crate) fn try_observe_p1j_physical(
        &self,
        namespace: p1e::Namespace,
    ) -> Result<p1e::PhysicalEvidence, p1e::Error> {
        if self.p1e_namespace(namespace.runtime)? != namespace {
            return Err(p1e::Error::Identity);
        }
        let layer = &self.layers[47];
        let mut state = layer.state.try_lock().ok_or(p1e::Error::PhysicalEvidence)?;
        let actual = copy_p1e_snapshot(
            namespace,
            &state.residents,
            state.residents.len(),
            |_, record| Some(p1e_resident(*record)),
        )?;
        if layer.arena.try_p1j_verify_snapshot(&actual) != Some(true) {
            return Err(p1e::Error::PhysicalEvidence);
        }
        state
            .p1e_shadow
            .as_deref_mut()
            .ok_or(p1e::Error::Incomplete)?
            .evidence(actual)
    }

    pub(crate) fn executor(&self) -> &Arc<GpuNativeExecutorContext> {
        &self.executor
    }

    pub(crate) fn production_physical_install_snapshot(
        &self,
    ) -> GpuNativeProductionPhysicalInstallSnapshot {
        self.executor.production_physical_install_snapshot()
    }

    pub(crate) fn reset_production_physical_install_telemetry(&self) {
        self.executor.reset_production_physical_install_telemetry();
    }

    pub(crate) fn gpu_cache(&self) -> &Arc<GpuExpertCache> {
        &self.gpu_cache
    }

    pub(crate) fn plan(&self) -> &GpuNativeModelExpertVramPlan {
        &self.plan
    }

    /// Re-run the production model-wide slot planner for an alternate expert
    /// budget without allocating buffers or mutating residency. Qualification
    /// analyzers use the authoritative executor limits rather than guessing
    /// per-layer capacities from the budget.
    pub(crate) fn plan_for_budget(
        &self,
        total_expert_budget_bytes: u64,
    ) -> Result<GpuNativeModelExpertVramPlan, GpuNativeTieredResidencyError> {
        let limits = self.executor.device_limits()?;
        GpuNativeModelExpertVramPlan::try_new(
            self.plan.num_layers(),
            self.plan.geometry(),
            total_expert_budget_bytes,
            &limits,
        )
    }

    pub(crate) fn arena(&self, layer_index: usize) -> Option<&Arc<GpuNativeQ4ExpertArena>> {
        self.layers.get(layer_index).map(|layer| &layer.arena)
    }

    fn p1e_namespace(
        &self,
        runtime: u64,
    ) -> Result<crate::predictor_v2::p1e::Namespace, crate::predictor_v2::p1e::Error> {
        use crate::predictor_v2::p1e::{Error, Namespace, CAPACITY, EXPERTS, LAYER};
        let layer = self.layers.get(LAYER).ok_or(Error::PhysicalEvidence)?;
        if self.layers.len() != LAYER + 1
            || self.plan.geometry().num_experts() != EXPERTS
            || self.plan.geometry().top_k() != CAPACITY
            || layer.arena.slot_capacity() != CAPACITY
            || layer.arena.layer_index() != LAYER
            || runtime == 0
        {
            return Err(Error::PhysicalEvidence);
        }
        Ok(Namespace {
            runtime,
            context: self.executor.context_id(),
            arena: Arc::as_ptr(&layer.arena) as usize,
            layer: LAYER,
            capacity: CAPACITY,
        })
    }

    fn p1e_copy_locked(
        &self,
        namespace: crate::predictor_v2::p1e::Namespace,
        state: &LayerResidencyState,
    ) -> Result<crate::predictor_v2::p1e::PhysicalSnapshot, crate::predictor_v2::p1e::Error> {
        use crate::predictor_v2::p1e::{Error, LAYER};
        if self.p1e_namespace(namespace.runtime)? != namespace {
            return Err(Error::PhysicalEvidence);
        }
        let layer = &self.layers[LAYER];
        copy_p1e_snapshot(
            namespace,
            &state.residents,
            layer.arena.resident_experts(),
            |global, record| {
                let identity = self.identity(global).ok()?;
                (identity.layer_index == LAYER
                    && record.key.layer_index() == LAYER
                    && record.key.expert_id() == identity.local_expert_id
                    && record.residency.key() == record.key
                    && layer
                        .arena
                        .contains_exact_residency(self.executor.context_id(), record.residency))
                .then(|| p1e_resident(*record))
            },
        )
    }

    /// Internal observation opt-in only. Seeds once from verified copied state;
    /// subsequent requests inherit the physical shadow, never temporal history.
    pub(crate) fn enable_p1e_shadow(
        &self,
        runtime: u64,
    ) -> Result<crate::predictor_v2::p1e::Namespace, crate::predictor_v2::p1e::Error> {
        use crate::predictor_v2::p1e::{Error, Shadow, LAYER};
        let namespace = self.p1e_namespace(runtime)?;
        let mut state = self.layers[LAYER].state.lock();
        let actual = self.p1e_copy_locked(namespace, &state)?;
        if let Some(shadow) = state.p1e_shadow.as_deref_mut() {
            if shadow.namespace() != namespace {
                return Err(Error::PhysicalEvidence);
            }
            shadow.evidence(actual)?;
        } else {
            state.p1e_shadow = Some(Box::new(Shadow::new(actual)?));
        }
        Ok(namespace)
    }

    /// Bounded non-touching evidence at F/D (and clean completion). The only
    /// mutable object is CPU observation accounting; physical/cache state and
    /// counters are never mutated. No source access or GPU synchronization.
    pub(crate) fn observe_p1e_physical(
        &self,
        namespace: crate::predictor_v2::p1e::Namespace,
    ) -> Result<crate::predictor_v2::p1e::PhysicalEvidence, crate::predictor_v2::p1e::Error> {
        use crate::predictor_v2::p1e::{Error, LAYER};
        let layer = self.layers.get(LAYER).ok_or(Error::PhysicalEvidence)?;
        let mut state = layer.state.lock();
        let actual = self.p1e_copy_locked(namespace, &state);
        let shadow = state.p1e_shadow.as_deref_mut().ok_or(Error::Incomplete)?;
        match actual {
            Ok(actual) => shadow.evidence(actual),
            Err(e) => {
                shadow.mark_incomplete(e);
                Err(e)
            }
        }
    }

    fn identity(
        &self,
        global_id: u32,
    ) -> Result<GpuNativeLayerExpertId, GpuNativeTieredResidencyError> {
        global_to_layer_local(
            global_id,
            self.plan.num_layers(),
            self.plan.geometry().num_experts() as u32,
        )
    }

    /// Read-only physical tier-selection probe for the engine's async demand
    /// path. Host logical-admission LRU state is intentionally not consulted:
    /// an exact, internally current arena residency owns immutable executable
    /// bytes for the lifetime of this manager.
    pub(crate) fn has_current_for_demand(
        &self,
        global_id: u32,
    ) -> Result<bool, GpuNativeTieredResidencyError> {
        let identity = self.identity(global_id)?;
        let layer = &self.layers[identity.layer_index];
        let mut state = layer.state.lock();
        Ok(self
            .current_record_locked(global_id, layer, &mut state, false)?
            .is_some()
            || self.p1j_current(global_id).is_some())
    }

    pub(crate) fn record_physical_source_acquisition(&self) {
        self.counters
            .physical_source_acquisitions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_logical_admissions_for_physical_misses(&self, count: usize) {
        self.counters
            .logical_admissions_for_physical_misses
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    /// VRAM-first speculative probe. It never waits behind a demand mutation;
    /// contention is an immediate best-effort drop.
    pub(crate) fn probe_speculative(
        &self,
        global_id: u32,
        priority: GpuNativeResidencyPriority,
    ) -> Result<GpuNativeSpeculativeProbe, GpuNativeTieredResidencyError> {
        let GpuNativeResidencyPriority::Speculative { score: _ } = priority else {
            return Err(GpuNativeTieredResidencyError::ResidencyPriorityMismatch);
        };
        self.counters
            .speculative_requests
            .fetch_add(1, Ordering::Relaxed);
        let identity = self.identity(global_id)?;
        let layer = &self.layers[identity.layer_index];
        let Some(mut state) = layer.state.try_lock() else {
            self.record_speculative_drop();
            return Ok(GpuNativeSpeculativeProbe::DroppedPressure);
        };
        if let Some(record) = self.current_record_locked(global_id, layer, &mut state, true)? {
            self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
            self.counters
                .speculative_vram_hits
                .fetch_add(1, Ordering::Relaxed);
            Ok(GpuNativeSpeculativeProbe::Hit(record.residency))
        } else {
            self.counters.vram_misses.fetch_add(1, Ordering::Relaxed);
            Ok(GpuNativeSpeculativeProbe::Miss)
        }
    }

    pub(crate) fn ensure_demand_set(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        self.ensure_demand_set_inner::<false>(
            priority,
            layer_index,
            demands,
            ordinary_demand_install_path(),
            None,
        )
    }

    pub(crate) fn ensure_demand_set_source_upload(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        source_upload: &crate::gpu_native_source_upload::State,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        if source_upload.arm != crate::gpu_native_source_upload::Arm::Treatment {
            return Err(GpuNativeTieredResidencyError::Backend(
                GpuNativeBootstrapError::QualificationSourceUpload {
                    detail: "ordinary production source/upload state must use treatment".into(),
                },
            ));
        }
        self.ensure_demand_set_inner_with_source_upload::<false>(
            priority,
            layer_index,
            demands,
            ordinary_demand_install_path(),
            None,
            Some(source_upload),
        )
    }

    pub(crate) fn ensure_demand_set_source_upload_observed(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        source_upload: &crate::gpu_native_source_upload::State,
        observer: &dyn GpuNativePhysicalInstallObserver,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        if source_upload.arm != crate::gpu_native_source_upload::Arm::Treatment
            || observer
                .source_upload_state()
                .is_none_or(|observed| !std::ptr::eq(observed, source_upload))
        {
            return Err(GpuNativeTieredResidencyError::Backend(
                GpuNativeBootstrapError::QualificationSourceUpload {
                    detail:
                        "observed source/upload state must be the production-owned treatment state"
                            .into(),
                },
            ));
        }
        self.ensure_demand_set_inner_with_source_upload::<true>(
            priority,
            layer_index,
            demands,
            ordinary_demand_install_path(),
            Some(observer),
            Some(source_upload),
        )
    }

    pub(crate) fn ensure_demand_set_legacy_control(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        observer: &dyn GpuNativePhysicalInstallObserver,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        self.ensure_demand_set_inner::<true>(
            priority,
            layer_index,
            demands,
            DemandPhysicalInstallPath::LegacyFullSlotVecControl,
            Some(observer),
        )
    }

    pub(crate) fn ensure_demand_set_production_observed(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        observer: &dyn GpuNativePhysicalInstallObserver,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        self.ensure_demand_set_inner::<true>(
            priority,
            layer_index,
            demands,
            ordinary_demand_install_path(),
            Some(observer),
        )
    }

    /// Same concurrent reserve/stage/commit transaction as production, with
    /// explicit full-slot zero writes for the zero-fill production qualifier.
    pub(crate) fn ensure_demand_set_full_zero_control_observed(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        observer: &dyn GpuNativePhysicalInstallObserver,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        self.ensure_demand_set_inner::<true>(
            priority,
            layer_index,
            demands,
            DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero,
            Some(observer),
        )
    }

    /// ORACLE-0B-S qualification-only destructive replacement. The caller
    /// must present the non-cloneable witness minted by the token loop after
    /// the previous token's successful boundary readback. Physical staging is
    /// the shared concurrent direct-staging transaction with an explicit
    /// full-zero control or historical qualification no-zero fill.
    /// The demand-request counter is kept separate. The resulting queue writes
    /// are deliberately left pending: next-token mapping writes follow them,
    /// then the next token's one normal command-buffer submit orders both sets
    /// of writes before compute. No extra empty submit is used to flush H2D.
    pub(crate) fn ensure_oracle_future_set_at_safe_boundary(
        &self,
        boundary: &mut crate::gpu_native_token_loop::GpuNativeSafeTokenBoundary,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        fill_policy: GpuNativePhysicalSlotFillPolicy,
        observer: &dyn GpuNativePhysicalInstallObserver,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        boundary
            .authorize_layer_once(layer_index)
            .map_err(
                |detail| GpuNativeTieredResidencyError::UnsafeOracleBoundary {
                    layer_index,
                    detail: detail.to_string(),
                },
            )?;
        self.ensure_demand_set_inner::<true>(
            GpuNativeResidencyPriority::OracleSafeBoundary,
            layer_index,
            demands,
            oracle_safe_boundary_install_path(fill_policy),
            Some(observer),
        )
    }

    /// PR2-B-B.1 control: the exact pre-B-B sequential production direct-
    /// staging path, observed only by the dedicated production qualifier.
    pub(crate) fn ensure_demand_set_physical_install_concurrency_control(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        observer: &dyn GpuNativePhysicalInstallObserver,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        self.ensure_demand_set_inner::<true>(
            priority,
            layer_index,
            demands,
            production_concurrency_control_path(),
            Some(observer),
        )
    }

    fn ensure_demand_set_inner<const OBSERVE: bool>(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        install_path: DemandPhysicalInstallPath,
        observer: Option<&dyn GpuNativePhysicalInstallObserver>,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        self.ensure_demand_set_inner_with_source_upload::<OBSERVE>(
            priority,
            layer_index,
            demands,
            install_path,
            observer,
            None,
        )
    }

    fn ensure_demand_set_inner_with_source_upload<const OBSERVE: bool>(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
        install_path: DemandPhysicalInstallPath,
        observer: Option<&dyn GpuNativePhysicalInstallObserver>,
        production_source_upload: Option<&crate::gpu_native_source_upload::State>,
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        if !matches!(
            priority,
            GpuNativeResidencyPriority::Demand | GpuNativeResidencyPriority::OracleSafeBoundary
        ) {
            return Err(GpuNativeTieredResidencyError::ResidencyPriorityMismatch);
        }
        let layer =
            self.layers
                .get(layer_index)
                .ok_or(GpuNativeTieredResidencyError::LayerOutOfRange {
                    layer_index,
                    num_layers: self.layers.len(),
                })?;
        if demands.len() > layer.arena.slot_capacity() {
            return Err(
                GpuNativeTieredResidencyError::DemandSetExceedsLayerCapacity {
                    requested: demands.len(),
                    capacity: layer.arena.slot_capacity(),
                },
            );
        }
        let mut protected = HashSet::with_capacity(demands.len());
        for demand in demands {
            let global_id = demand.global_id();
            if !protected.insert(global_id) {
                return Err(GpuNativeTieredResidencyError::DuplicateDemandExpert { global_id });
            }
            let identity = self.identity(global_id)?;
            if identity.layer_index != layer_index {
                return Err(GpuNativeTieredResidencyError::DemandLayerMismatch {
                    requested_layer: layer_index,
                    global_id,
                    actual_layer: identity.layer_index,
                });
            }
            if let GpuNativeDemandExpert::Install {
                resident,
                admission,
                ..
            } = demand
            {
                self.validate_source(global_id, resident, admission)?;
            }
        }

        if matches!(priority, GpuNativeResidencyPriority::Demand) {
            self.counters
                .demand_requests
                .fetch_add(demands.len() as u64, Ordering::Relaxed);
        }
        let mut state = layer.state.lock();
        let sidecar = demands
            .iter()
            .find_map(|d| self.p1j_current(d.global_id()))
            .map(|(_, residency)| {
                p1e_resident(PhysicalRecord {
                    key: residency.key(),
                    residency,
                })
            });
        if let Some(shadow) = state.p1e_shadow.as_deref_mut() {
            let local_ids = demands
                .iter()
                .map(|d| d.global_id() % self.plan.geometry().num_experts() as u32)
                .collect::<Vec<_>>();
            if sidecar.is_some() {
                shadow.demand_with_sidecar(&local_ids, sidecar);
            } else {
                shadow.demand(&local_ids);
            }
        }
        let mut resolved = vec![None; demands.len()];
        let mut misses = Vec::new();
        for (index, demand) in demands.iter().enumerate() {
            let global_id = demand.global_id();
            if let Some(record) = self.current_record_locked(global_id, layer, &mut state, true)? {
                self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .physical_current_hits
                    .fetch_add(1, Ordering::Relaxed);
                resolved[index] = Some(record.residency);
            } else if let Some((_, residency)) = self.p1j_current(global_id) {
                // The token loop's execution guard spans this target recovery.
                // No logical acquisition, ordinary LRU touch or payload rewrite.
                self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .physical_current_hits
                    .fetch_add(1, Ordering::Relaxed);
                resolved[index] = Some(residency);
            } else {
                self.counters.vram_misses.fetch_add(1, Ordering::Relaxed);
                match demand {
                    GpuNativeDemandExpert::Install { .. } => misses.push(index),
                    GpuNativeDemandExpert::Current { .. } => {
                        return Err(GpuNativeTieredResidencyError::DemandSourceMissing {
                            global_id,
                        });
                    }
                }
            }
        }

        while state.residents.len().saturating_add(misses.len()) > layer.arena.slot_capacity() {
            let victim = oldest_unprotected(&state.residents, &protected)
                .ok_or(GpuNativeTieredResidencyError::NoEvictablePhysicalSlot { layer_index })?;
            self.retire_metadata_record_locked(victim, layer, &mut state, true)?;
            if let Some(observer) = observer {
                observer.record_physical_victim(victim);
            }
        }

        if matches!(
            install_path,
            DemandPhysicalInstallPath::ProductionConcurrentDirectStaging
                | DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
                | DemandPhysicalInstallPath::QualificationConcurrentDirectStagingNoZeroFill
        ) {
            if install_path
                == DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
            {
                self.install_parallel_physical_misses_locked::<true, true>(
                    demands,
                    &misses,
                    layer,
                    &mut state,
                    &mut resolved,
                    observer,
                    production_source_upload,
                )?;
            } else {
                self.install_parallel_physical_misses_locked::<OBSERVE, false>(
                    demands,
                    &misses,
                    layer,
                    &mut state,
                    &mut resolved,
                    observer,
                    production_source_upload,
                )?;
            }
        } else {
            let transaction_started = (OBSERVE && !misses.is_empty()).then(Instant::now);
            if OBSERVE && !misses.is_empty() {
                if let Some(observer) = observer {
                    observer.record_physical_install_set(
                        misses.len(),
                        false,
                        crate::parallel::in_rayon_worker(),
                        rayon::current_num_threads(),
                    );
                }
            }
            for index in misses {
                let GpuNativeDemandExpert::Install {
                    global_id,
                    resident,
                    admission,
                } = &demands[index]
                else {
                    unreachable!("miss list contains only install sources")
                };
                let residency = self.install_locked::<OBSERVE>(
                    *global_id,
                    resident,
                    admission,
                    layer,
                    &mut state,
                    false,
                    install_path,
                    observer,
                )?;
                resolved[index] = Some(residency);
            }
            if let (Some(observer), Some(started)) = (observer, transaction_started) {
                observer
                    .record_physical_install_transaction_wall(qualification_elapsed_us(started));
            }
        }
        resolved
            .into_iter()
            .enumerate()
            .map(|(index, residency)| {
                residency.ok_or(GpuNativeTieredResidencyError::DemandSourceMissing {
                    global_id: demands[index].global_id(),
                })
            })
            .collect()
    }

    fn install_parallel_physical_misses_locked<
        'a,
        const OBSERVE: bool,
        const FULL_ZERO_CONTROL: bool,
    >(
        &self,
        demands: &[GpuNativeDemandExpert],
        misses: &[usize],
        layer: &'a LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        resolved: &mut [Option<GpuNativeQ4ExpertResidency>],
        observer: Option<&dyn GpuNativePhysicalInstallObserver>,
        production_source_upload: Option<&crate::gpu_native_source_upload::State>,
    ) -> Result<(), GpuNativeTieredResidencyError> {
        if misses.is_empty() {
            return Ok(());
        }
        let transaction_started = OBSERVE.then(Instant::now);
        let parallel = misses.len() >= 2;
        self.executor
            .record_production_physical_install_set(misses.len());
        if OBSERVE {
            observer
                .expect("observed production install has an observer")
                .record_physical_install_set(
                    misses.len(),
                    parallel,
                    crate::parallel::in_rayon_worker(),
                    rayon::current_num_threads(),
                );
        }

        let reservation_started = OBSERVE.then(Instant::now);
        let mut reserved = Vec::with_capacity(misses.len());
        for &demand_index in misses {
            let GpuNativeDemandExpert::Install {
                global_id,
                resident,
                admission,
            } = &demands[demand_index]
            else {
                unreachable!("miss list contains only install sources")
            };
            let identity = self.identity(*global_id)?;
            let key = global_to_q4_expert_key(
                *global_id,
                admission.generation(),
                self.plan.num_layers(),
                self.plan.geometry().num_experts() as u32,
            )?;
            self.executor
                .record_production_physical_reservation_attempt();
            if OBSERVE {
                let observer = observer.expect("observed production install has an observer");
                observer.record_physical_install_attempt();
                observer.record_reservation_attempt();
            }
            let acquired = match self.executor.acquire_q4_expert_residency(&layer.arena, key) {
                Ok(acquired) => acquired,
                Err(error) => {
                    self.executor
                        .record_production_physical_reservation_failure();
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer.record_reservation_failure();
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(error.into());
                }
            };
            let permit = match acquired {
                GpuNativeQ4ExpertAcquire::Install(permit) => permit,
                GpuNativeQ4ExpertAcquire::Hit(_) => {
                    self.executor
                        .record_production_physical_reservation_failure();
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer.record_reservation_failure();
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(GpuNativeTieredResidencyError::PhysicalIdentityCorrupt {
                        global_id: *global_id,
                    });
                }
                GpuNativeQ4ExpertAcquire::InstallInProgress => {
                    self.executor
                        .record_production_physical_reservation_failure();
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer.record_reservation_failure();
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(GpuNativeTieredResidencyError::InstallInProgress {
                        global_id: *global_id,
                        generation: admission.generation(),
                    });
                }
                GpuNativeQ4ExpertAcquire::StaleRequester => {
                    self.executor
                        .record_production_physical_reservation_failure();
                    self.counters
                        .stale_generation_rejections
                        .fetch_add(1, Ordering::Relaxed);
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer.record_reservation_failure();
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(GpuNativeTieredResidencyError::StalePhysicalRequester {
                        global_id: *global_id,
                        generation: admission.generation(),
                    });
                }
                GpuNativeQ4ExpertAcquire::NoPhysicalSlot => {
                    self.executor
                        .record_production_physical_reservation_failure();
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer.record_reservation_failure();
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(GpuNativeTieredResidencyError::NoPhysicalSlot {
                        layer_index: identity.layer_index,
                    });
                }
            };
            self.executor
                .record_production_physical_reservation_success();
            if OBSERVE {
                observer
                    .expect("observed production install has an observer")
                    .record_reservation_success(
                        *global_id,
                        permit.reserved_residency(),
                        permit.install_ticket(),
                    );
            }
            reserved.push(ReservedPhysicalInstall {
                demand_index,
                global_id: *global_id,
                resident: resident.clone(),
                admission: admission.clone(),
                key,
                permit,
            });
        }
        if OBSERVE {
            observer
                .expect("observed production install has an observer")
                .record_physical_reservation_wall(qualification_elapsed_us(
                    reservation_started.expect("observed reservation has a timer"),
                ));
        }

        let upload = if let Some(production_source_upload) = production_source_upload {
            if OBSERVE
                && observer
                    .and_then(|o| o.source_upload_state())
                    .is_none_or(|observed| !std::ptr::eq(observed, production_source_upload))
            {
                return Err(GpuNativeTieredResidencyError::Backend(
                    GpuNativeBootstrapError::QualificationSourceUpload {
                        detail: "explicit production source/upload state disagrees with observer"
                            .into(),
                    },
                ));
            }
            Some(production_source_upload)
        } else if OBSERVE {
            observer.and_then(|o| o.source_upload_state())
        } else {
            None
        };
        if let Some(upload) =
            upload.filter(|u| u.arm == crate::gpu_native_source_upload::Arm::Treatment)
        {
            if reserved.iter().any(|r| upload.has_pending(r.global_id)) {
                upload.add(|m| &mut m.fused_install_sets, 1);
            }
        }
        let copies = upload
            .filter(|u| u.arm == crate::gpu_native_source_upload::Arm::Treatment)
            .map(crate::gpu_native_source_upload::CopySet::new);
        let stage_one = |reserved: ReservedPhysicalInstall<'a>| {
            if OBSERVE {
                observer
                    .expect("observed production install has an observer")
                    .record_physical_stage_started();
            }
            let stage_started = OBSERVE.then(Instant::now);
            let lease = upload
                .map(|u| u.take_lease(reserved.global_id, &reserved.resident))
                .transpose();
            let fused_source_upload = matches!(&lease, Ok(Some(Some(_))));
            let result = match lease {
                Err(detail) => Err(crate::backend::gpu_native::GpuNativeBootstrapError::QualificationSourceUpload { detail }),
                Ok(Some(Some(lease))) => self.executor.stage_q4_expert_source_upload(
                    reserved.permit, reserved.resident.data(), lease, copies.as_ref().expect("treatment copy set")),
                _ => if FULL_ZERO_CONTROL {
                self.executor
                    .stage_q4_expert_residency_full_zero_control_observed(
                        reserved.permit,
                        reserved.resident.data(),
                    )
            } else if OBSERVE {
                self.executor.stage_q4_expert_residency_production_observed(
                    reserved.permit,
                    reserved.resident.data(),
                )
            } else {
                self.executor
                    .stage_q4_expert_residency_production(reserved.permit, reserved.resident.data())
                    .map(|prepared| (prepared, GpuNativePhysicalInstallEvidence::default()))
            }};
            match result {
                Ok((prepared, mut evidence)) => {
                    let individual_stage_us = stage_started.map_or(0, qualification_elapsed_us);
                    if !OBSERVE {
                        if let Some(upload) = upload {
                            upload.add(
                                |m| &mut m.total_payload_bytes_staged,
                                crate::gpu_native_source_upload::PAYLOAD as u64,
                            );
                            if fused_source_upload {
                                upload.add(|m| &mut m.fused_installs, 1);
                                upload.add(
                                    |m| &mut m.fused_gpu_copy_bytes,
                                    crate::gpu_native_source_upload::PAYLOAD as u64,
                                );
                            } else {
                                upload.add(|m| &mut m.fallback_installs, 1);
                                upload.add(
                                    |m| &mut m.fallback_payload_copy_bytes,
                                    crate::gpu_native_source_upload::PAYLOAD as u64,
                                );
                                upload.add(
                                    |m| &mut m.physical_cpu_payload_copy_bytes,
                                    crate::gpu_native_source_upload::PAYLOAD as u64,
                                );
                            }
                        }
                    }
                    if OBSERVE {
                        evidence.individual_physical_stage_us = individual_stage_us;
                        observer
                            .expect("observed production install has an observer")
                            .record_physical_stage_completed(evidence, individual_stage_us);
                    }
                    Ok(StagedPhysicalInstall {
                        demand_index: reserved.demand_index,
                        global_id: reserved.global_id,
                        admission: reserved.admission,
                        key: reserved.key,
                        prepared,
                        evidence,
                        individual_stage_us,
                    })
                }
                Err(error) => {
                    if OBSERVE {
                        observer
                            .expect("observed production install has an observer")
                            .record_physical_stage_failed();
                    }
                    Err(GpuNativeTieredResidencyError::from(error))
                }
            }
        };
        let parallel_stage_started = OBSERVE.then(Instant::now);
        let staged = collect_physical_stage_results_in_request_order(reserved, parallel, stage_one);
        if OBSERVE {
            observer
                .expect("observed production install has an observer")
                .record_parallel_stage_wall(qualification_elapsed_us(
                    parallel_stage_started.expect("observed stage has a timer"),
                ));
        }

        // Queue submission is deliberately separate from token compute and
        // precedes every executable mapping publication, including a prefix
        // retained by the existing ordered failure semantics.
        if let Some(copies) = &copies {
            copies.submit(&self.executor).map_err(|detail| {
                upload.expect("copy audit").add(|m| &mut m.copy_failures, 1);
                GpuNativeTieredResidencyError::from(crate::backend::gpu_native::GpuNativeBootstrapError::QualificationSourceUpload { detail })
            })?;
        }
        let first_stage_failure = staged.iter().position(Result::is_err);
        let mut remaining_successful = staged.iter().filter(|result| result.is_ok()).count() as u64;
        for (position, staged_result) in staged.into_iter().enumerate() {
            if first_stage_failure.is_some_and(|failed| position >= failed) {
                if position == first_stage_failure.expect("failure position is present") {
                    let error = match staged_result {
                        Err(error) => error,
                        Ok(_) => unreachable!("earliest failed stage is an error"),
                    };
                    self.executor
                        .record_production_unpublished_physical_writes_after_failure(
                            remaining_successful,
                        );
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer
                            .record_unpublished_physical_writes_after_failure(remaining_successful);
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(error);
                }
                unreachable!("earliest stage failure returns before later entries")
            }
            let staged = staged_result.expect("successful prefix precedes first stage failure");
            if OBSERVE {
                observer
                    .expect("observed production install has an observer")
                    .record_ordered_commit_attempt();
            }
            let commit_started = OBSERVE.then(Instant::now);
            let commit_result = if OBSERVE {
                self.executor
                    .commit_q4_expert_residency_production_observed(
                        staged.prepared,
                        staged.evidence,
                    )
            } else {
                self.executor
                    .commit_q4_expert_residency_production(staged.prepared)
                    .map(|residency| (residency, staged.evidence))
            };
            let (residency, evidence) = match commit_result {
                Ok(committed) => committed,
                Err(error) => {
                    self.executor
                        .record_production_unpublished_physical_writes_after_failure(
                            remaining_successful,
                        );
                    if OBSERVE {
                        let observer =
                            observer.expect("observed production install has an observer");
                        observer.record_ordered_commit_failed(matches!(
                            error,
                            GpuNativeBootstrapError::ExpertInstallReservationLost
                        ));
                        observer
                            .record_unpublished_physical_writes_after_failure(remaining_successful);
                        observer.record_physical_install_transaction_wall(
                            qualification_elapsed_us(
                                transaction_started.expect("observed transaction has a timer"),
                            ),
                        );
                    }
                    return Err(error.into());
                }
            };
            remaining_successful = remaining_successful.saturating_sub(1);
            if !self
                .gpu_cache
                .contains_generation(staged.global_id, staged.admission.generation())
            {
                let _ = self
                    .executor
                    .retire_q4_expert_residency(&layer.arena, staged.key)?;
                self.counters
                    .stale_generation_rejections
                    .fetch_add(1, Ordering::Relaxed);
                self.executor
                    .record_production_ordered_commit_failure(false);
                self.executor
                    .record_production_unpublished_physical_writes_after_failure(
                        remaining_successful,
                    );
                if OBSERVE {
                    let observer = observer.expect("observed production install has an observer");
                    observer.record_ordered_commit_failed(false);
                    observer.record_unpublished_physical_writes_after_failure(remaining_successful);
                    observer.record_physical_install_transaction_wall(qualification_elapsed_us(
                        transaction_started.expect("observed transaction has a timer"),
                    ));
                }
                return Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
                    global_id: staged.global_id,
                    generation: staged.admission.generation(),
                });
            }
            let reinstall = state
                .last_installed_generations
                .insert(staged.global_id, staged.admission.generation())
                == Some(staged.admission.generation());
            state.residents.put(
                staged.global_id,
                PhysicalRecord {
                    key: staged.key,
                    residency,
                },
            );
            if let Some(shadow) = state.p1e_shadow.as_deref_mut() {
                shadow.committed_install(p1e_resident(PhysicalRecord {
                    key: staged.key,
                    residency,
                }));
            }
            self.counters
                .ram_to_vram_installs
                .fetch_add(1, Ordering::Relaxed);
            if reinstall {
                self.counters
                    .physical_reinstalls
                    .fetch_add(1, Ordering::Relaxed);
            }
            if OBSERVE {
                let commit_us =
                    qualification_elapsed_us(commit_started.expect("observed commit has a timer"));
                let observer = observer.expect("observed production install has an observer");
                observer.record_ordered_commit_completed(commit_us);
                observer.record_physical_install_completion(
                    staged.global_id,
                    residency,
                    evidence,
                    post_reservation_physical_install_total_us(
                        staged.individual_stage_us,
                        commit_us,
                    ),
                );
            }
            resolved[staged.demand_index] = Some(residency);
        }
        if OBSERVE {
            observer
                .expect("observed production install has an observer")
                .record_physical_install_transaction_wall(qualification_elapsed_us(
                    transaction_started.expect("observed transaction has a timer"),
                ));
        }
        Ok(())
    }

    pub(crate) fn ensure_speculative_resident(
        &self,
        global_id: u32,
        resident: &Arc<ExpertResident>,
        admission: &GpuAdmission,
        priority: GpuNativeResidencyPriority,
    ) -> Result<GpuNativeSpeculativeInstall, GpuNativeTieredResidencyError> {
        let GpuNativeResidencyPriority::Speculative { score: _ } = priority else {
            return Err(GpuNativeTieredResidencyError::ResidencyPriorityMismatch);
        };
        if let Err(error) = self.validate_source(global_id, resident, admission) {
            if matches!(
                error,
                GpuNativeTieredResidencyError::LogicalAdmissionStale { .. }
            ) {
                return Ok(GpuNativeSpeculativeInstall::StaleLogicalGeneration);
            }
            return Err(error);
        }
        let identity = self.identity(global_id)?;
        let layer = &self.layers[identity.layer_index];
        let Some(mut state) = layer.state.try_lock() else {
            self.record_speculative_drop();
            return Ok(GpuNativeSpeculativeInstall::DroppedCapacityOrPressure);
        };
        if let Some(record) = self.current_record_locked(global_id, layer, &mut state, true)? {
            self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
            self.counters
                .speculative_vram_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(GpuNativeSpeculativeInstall::Hit(record.residency));
        }
        if !self
            .gpu_cache
            .contains_generation(global_id, admission.generation())
        {
            self.counters
                .stale_generation_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Ok(GpuNativeSpeculativeInstall::StaleLogicalGeneration);
        }
        if state.residents.len() >= layer.arena.slot_capacity() {
            self.record_speculative_drop();
            return Ok(GpuNativeSpeculativeInstall::DroppedCapacityOrPressure);
        }
        let residency = match self.install_locked::<false>(
            global_id,
            resident,
            admission,
            layer,
            &mut state,
            true,
            speculative_install_path(),
            None,
        ) {
            Ok(residency) => residency,
            Err(GpuNativeTieredResidencyError::NoPhysicalSlot { .. }) => {
                return Ok(GpuNativeSpeculativeInstall::DroppedCapacityOrPressure);
            }
            Err(error) => return Err(error),
        };
        Ok(GpuNativeSpeculativeInstall::Installed(residency))
    }

    pub(crate) fn snapshot(&self) -> GpuNativeTieredResidencySnapshot {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let arena = layer.arena.residency_snapshot();
                let state = layer.state.lock();
                GpuNativeTieredLayerSnapshot {
                    slot_capacity: arena.slot_capacity,
                    resident_slots: arena.resident_slots,
                    free_slots: arena.free_slots,
                    physical_evictions: state.physical_evictions,
                }
            })
            .collect::<Vec<_>>();
        GpuNativeTieredResidencySnapshot {
            model_expert_budget_bytes: self.plan.total_expert_budget_bytes(),
            model_arena_allocation_bytes: self.plan.total_arena_allocation_bytes(),
            model_slot_capacity: self.plan.model_slot_capacity(),
            resident_physical_slots: layers.iter().map(|layer| layer.resident_slots).sum(),
            free_physical_slots: layers.iter().map(|layer| layer.free_slots).sum(),
            vram_hits: self.counters.vram_hits.load(Ordering::Relaxed),
            vram_misses: self.counters.vram_misses.load(Ordering::Relaxed),
            physical_current_hits: self.counters.physical_current_hits.load(Ordering::Relaxed),
            physical_source_acquisitions: self
                .counters
                .physical_source_acquisitions
                .load(Ordering::Relaxed),
            logical_admissions_for_physical_misses: self
                .counters
                .logical_admissions_for_physical_misses
                .load(Ordering::Relaxed),
            ram_to_vram_installs: self.counters.ram_to_vram_installs.load(Ordering::Relaxed),
            physical_evictions: self.counters.physical_evictions.load(Ordering::Relaxed),
            physical_reinstalls: self.counters.physical_reinstalls.load(Ordering::Relaxed),
            stale_generation_rejections: self
                .counters
                .stale_generation_rejections
                .load(Ordering::Relaxed),
            demand_requests: self.counters.demand_requests.load(Ordering::Relaxed),
            speculative_requests: self.counters.speculative_requests.load(Ordering::Relaxed),
            speculative_vram_hits: self.counters.speculative_vram_hits.load(Ordering::Relaxed),
            speculative_ram_to_vram_installs: self
                .counters
                .speculative_ram_to_vram_installs
                .load(Ordering::Relaxed),
            speculative_dropped_capacity_or_pressure: self
                .counters
                .speculative_dropped_capacity_or_pressure
                .load(Ordering::Relaxed),
            layers,
        }
    }

    fn validate_source(
        &self,
        global_id: u32,
        resident: &Arc<ExpertResident>,
        admission: &GpuAdmission,
    ) -> Result<(), GpuNativeTieredResidencyError> {
        let result = validate_physical_install_source(
            self.gpu_cache.as_ref(),
            global_id,
            resident,
            admission,
        );
        if matches!(
            result,
            Err(GpuNativeTieredResidencyError::LogicalAdmissionStale { .. })
        ) {
            self.counters
                .stale_generation_rejections
                .fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn current_record_locked(
        &self,
        global_id: u32,
        layer: &LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        touch: bool,
    ) -> Result<Option<PhysicalRecord>, GpuNativeTieredResidencyError> {
        let Some(record) = state.residents.peek(&global_id).copied() else {
            return Ok(None);
        };
        let identity = self.identity(global_id)?;
        if record.key.layer_index() != identity.layer_index
            || record.key.expert_id() != identity.local_expert_id
            || record.residency.key() != record.key
            || layer.arena.layer_index() != identity.layer_index
            || !layer
                .arena
                .contains_exact_residency(self.executor.context_id(), record.residency)
        {
            return Err(GpuNativeTieredResidencyError::PhysicalIdentityCorrupt { global_id });
        }
        if touch {
            let record = touch_physical_record(&mut state.residents, global_id)
                .expect("peeked current physical record remains under layer lock");
            Ok(Some(record))
        } else {
            Ok(Some(record))
        }
    }

    fn retire_metadata_record_locked(
        &self,
        global_id: u32,
        layer: &LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        capacity_eviction: bool,
    ) -> Result<(), GpuNativeTieredResidencyError> {
        let Some(record) = state.residents.peek(&global_id).copied() else {
            return Ok(());
        };
        match self
            .executor
            .retire_q4_expert_residency(&layer.arena, record.key)?
        {
            GpuNativeQ4ExpertRetire::Retired
            | GpuNativeQ4ExpertRetire::CancelledInstall
            | GpuNativeQ4ExpertRetire::NotResident
            | GpuNativeQ4ExpertRetire::StaleRequester => {}
        }
        state.residents.pop(&global_id);
        if let Some(shadow) = state.p1e_shadow.as_deref_mut() {
            shadow.victim(record.key.expert_id());
        }
        if capacity_eviction {
            state.physical_evictions = state.physical_evictions.saturating_add(1);
            self.counters
                .physical_evictions
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn install_locked<const OBSERVE: bool>(
        &self,
        global_id: u32,
        resident: &Arc<ExpertResident>,
        admission: &GpuAdmission,
        layer: &LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        speculative: bool,
        install_path: DemandPhysicalInstallPath,
        observer: Option<&dyn GpuNativePhysicalInstallObserver>,
    ) -> Result<GpuNativeQ4ExpertResidency, GpuNativeTieredResidencyError> {
        let identity = self.identity(global_id)?;
        let key = global_to_q4_expert_key(
            global_id,
            admission.generation(),
            self.plan.num_layers(),
            self.plan.geometry().num_experts() as u32,
        )?;
        let reservation_started = OBSERVE.then(Instant::now);
        let (residency, installed) = match self
            .executor
            .acquire_q4_expert_residency(&layer.arena, key)?
        {
            GpuNativeQ4ExpertAcquire::Hit(hit) => (hit, false),
            GpuNativeQ4ExpertAcquire::Install(permit) => {
                if OBSERVE {
                    let observer = observer.expect("observed sequential install has an observer");
                    observer.record_physical_install_attempt();
                    observer.record_reservation_attempt();
                    observer.record_reservation_success(
                        global_id,
                        permit.reserved_residency(),
                        permit.install_ticket(),
                    );
                    observer.record_physical_reservation_wall(qualification_elapsed_us(
                        reservation_started.expect("observed reservation has a timer"),
                    ));
                    let started = Instant::now();
                    let result = match install_path {
                        DemandPhysicalInstallPath::SequentialDirectStagingControl => self
                            .executor
                            .install_q4_expert_residency_sequential_control_observed(
                                permit,
                                resident.data(),
                            ),
                        DemandPhysicalInstallPath::LegacyFullSlotVecControl => self
                            .executor
                            .install_q4_expert_residency_legacy_observed(permit, resident.data()),
                        DemandPhysicalInstallPath::ProductionConcurrentDirectStaging
                        | DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
                        | DemandPhysicalInstallPath::QualificationConcurrentDirectStagingNoZeroFill => {
                            unreachable!("ordinary production installs use the split transaction")
                        }
                    };
                    match result {
                        Ok((residency, evidence)) => {
                            observer.record_physical_install_completion(
                                global_id,
                                residency,
                                evidence,
                                u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
                            );
                            (residency, true)
                        }
                        Err(error) => {
                            if matches!(
                                error,
                                GpuNativeBootstrapError::ExpertDirectStagingUnavailable { .. }
                            ) {
                                observer.record_direct_staging_failure();
                            }
                            return Err(error.into());
                        }
                    }
                } else {
                    let residency = match install_path {
                        DemandPhysicalInstallPath::LegacyFullSlotVecControl => self
                            .executor
                            .install_q4_expert_residency_legacy(permit, resident.data())?,
                        DemandPhysicalInstallPath::SequentialDirectStagingControl => {
                            unreachable!("sequential control is qualifier-only")
                        }
                        DemandPhysicalInstallPath::ProductionConcurrentDirectStaging
                        | DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
                        | DemandPhysicalInstallPath::QualificationConcurrentDirectStagingNoZeroFill => {
                            unreachable!("ordinary production installs use the split transaction")
                        }
                    };
                    (residency, true)
                }
            }
            GpuNativeQ4ExpertAcquire::InstallInProgress => {
                return Err(GpuNativeTieredResidencyError::InstallInProgress {
                    global_id,
                    generation: admission.generation(),
                });
            }
            GpuNativeQ4ExpertAcquire::StaleRequester => {
                self.counters
                    .stale_generation_rejections
                    .fetch_add(1, Ordering::Relaxed);
                return Err(GpuNativeTieredResidencyError::StalePhysicalRequester {
                    global_id,
                    generation: admission.generation(),
                });
            }
            GpuNativeQ4ExpertAcquire::NoPhysicalSlot => {
                if speculative {
                    self.record_speculative_drop();
                }
                return Err(GpuNativeTieredResidencyError::NoPhysicalSlot {
                    layer_index: identity.layer_index,
                });
            }
        };
        if !self
            .gpu_cache
            .contains_generation(global_id, admission.generation())
        {
            let _ = self
                .executor
                .retire_q4_expert_residency(&layer.arena, key)?;
            self.counters
                .stale_generation_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
                global_id,
                generation: admission.generation(),
            });
        }
        let reinstall = installed
            && state
                .last_installed_generations
                .insert(global_id, admission.generation())
                == Some(admission.generation());
        state
            .residents
            .put(global_id, PhysicalRecord { key, residency });
        if let Some(shadow) = state.p1e_shadow.as_deref_mut() {
            shadow.committed_install(p1e_resident(PhysicalRecord { key, residency }));
        }
        if installed {
            self.counters
                .ram_to_vram_installs
                .fetch_add(1, Ordering::Relaxed);
            if reinstall {
                self.counters
                    .physical_reinstalls
                    .fetch_add(1, Ordering::Relaxed);
            }
            if speculative {
                self.counters
                    .speculative_ram_to_vram_installs
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(residency)
    }

    fn record_speculative_drop(&self) {
        self.counters
            .speculative_dropped_capacity_or_pressure
            .fetch_add(1, Ordering::Relaxed);
    }
}

fn oldest_unprotected<T>(cache: &LruCache<u32, T>, protected: &HashSet<u32>) -> Option<u32> {
    cache
        .iter()
        .rev()
        .find_map(|(&global_id, _)| (!protected.contains(&global_id)).then_some(global_id))
}

#[cfg(test)]
pub(crate) fn validate_qualification_physical_source_for_test(
    gpu_cache: &GpuExpertCache,
    global_id: u32,
    resident: &Arc<ExpertResident>,
    admission: &GpuAdmission,
) -> Result<(), GpuNativeTieredResidencyError> {
    validate_physical_install_source(gpu_cache, global_id, resident, admission)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn p1m_every_host_lease_and_claim_refusal_maps_exactly() {
        use HostBackedLeaseResult as H;
        use P1jPrepareRefusal as P;
        for (input, expected) in [
            (H::Busy, P::HostLeaseBusy),
            (H::Missing, P::HostLeaseMissing),
            (H::Stale, P::HostLeaseStale),
            (H::WrongPayloadKind, P::HostLeaseWrongPayloadKind),
            (H::WrongDtype, P::HostLeaseWrongDtype),
            (H::WrongLength, P::HostLeaseWrongLength),
        ] {
            assert!(matches!(p1j_host_lease(input), Err(actual) if actual == expected));
        }
        for (input, expected) in [
            (P1jClaimRefusal::LockBusy, P::SidecarLockBusy),
            (P1jClaimRefusal::Occupied, P::SidecarOccupied),
            (
                P1jClaimRefusal::IdentityRejected,
                P::SidecarIdentityRejected,
            ),
            (P1jClaimRefusal::EpochExhausted, P::SidecarEpochExhausted),
            (
                P1jClaimRefusal::WriterSequenceExhausted,
                P::SidecarWriterSequenceExhausted,
            ),
        ] {
            assert_eq!(P::from(input), expected);
        }
    }

    #[test]
    fn p1m_real_host_payload_length_diagnosed_and_lease_retained_without_lru_touch() {
        use crate::expert_cache::{GpuResident, HOST_BACKED_Q4_BYTES};
        for len in [HOST_BACKED_Q4_BYTES - 1, HOST_BACKED_Q4_BYTES] {
            let cache = GpuExpertCache::new(HOST_BACKED_Q4_BYTES * 2, 0.0, 0);
            assert!(cache.promote_sync(Arc::new(GpuResident::new_with_dtype(
                47 * 128 + 8,
                vec![0; len],
                crate::inference::WeightDtype::Q4_0,
            ))));
            // A newer resident lets subsequent eviction expose any lease LRU touch.
            assert!(cache.promote_sync(Arc::new(GpuResident::new_with_dtype(
                47 * 128 + 9,
                vec![0; HOST_BACKED_Q4_BYTES],
                crate::inference::WeightDtype::Q4_0,
            ))));
            let generation = cache.current_generation(47 * 128 + 8).unwrap();
            let result = p1j_host_lease(cache.try_lease_host_backed(47 * 128 + 8, generation));
            if len != HOST_BACKED_Q4_BYTES {
                assert!(matches!(
                    result,
                    Err(P1jPrepareRefusal::HostLeaseWrongLength)
                ));
                assert_eq!(cache.host_backed_lease_snapshot().active, 0);
            } else {
                let lease = result.unwrap_or_else(|e| panic!("{e:?}"));
                assert_eq!(lease.generation(), generation);
                assert_eq!(cache.host_backed_lease_snapshot().active, 1);
                drop(lease);
                assert_eq!(cache.host_backed_lease_snapshot().active, 0);
            }
            assert_eq!(cache.current_generation(47 * 128 + 8), Some(generation));
            assert!(cache.promote_sync(Arc::new(GpuResident::new_with_dtype(
                47 * 128 + 10,
                vec![0; HOST_BACKED_Q4_BYTES],
                crate::inference::WeightDtype::Q4_0,
            ))));
            assert!(cache.current_generation(47 * 128 + 8).is_none());
            assert!(cache.current_generation(47 * 128 + 9).is_some());
        }
    }

    #[test]
    fn p1j_spawn_blocking_writer_enqueues_closes_then_releases_exact_source() {
        use crate::expert_cache::{GpuResident, HOST_BACKED_Q4_BYTES};
        let cache = Arc::new(GpuExpertCache::new(HOST_BACKED_Q4_BYTES, 0.0, 0));
        let resident = Arc::new(GpuResident::new_with_dtype(
            47 * 128 + 8,
            vec![19; HOST_BACKED_Q4_BYTES],
            crate::inference::WeightDtype::Q4_0,
        ));
        let ptr = resident.data().as_ptr() as usize;
        assert!(cache.promote_sync(resident));
        let id = 47 * 128 + 8;
        let HostBackedLeaseResult::Acquired(lease) =
            cache.try_lease_host_backed(id, cache.current_generation(id).unwrap())
        else {
            panic!("lease");
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let copy_events = events.clone();
        let copy_cache = cache.clone();
        let task = runtime.spawn_blocking(move || {
            p1j_stage_lease(
                lease,
                |bytes| {
                    assert_eq!(bytes.as_ptr() as usize, ptr);
                    assert_eq!(bytes.len(), HOST_BACKED_Q4_BYTES);
                    copy_events.lock().push("payload-enqueue");
                    Ok(())
                },
                |success| {
                    assert!(success);
                    assert_eq!(copy_cache.host_backed_lease_snapshot().active, 1);
                    copy_events.lock().push("closed");
                },
            );
            assert_eq!(copy_cache.host_backed_lease_snapshot().retained_bytes, 0);
            copy_events.lock().push("released");
        });
        runtime.block_on(task).unwrap(); // CPU test only; never the D path.
        assert_eq!(*events.lock(), ["payload-enqueue", "closed", "released"]);
        assert_eq!(cache.host_backed_lease_snapshot().releases, 1);
    }

    #[test]
    fn p1j_contended_cleanup_coalesces_newer_cancellation_without_losing_it() {
        let fence = P1jRetirement::default();
        fence.request(1, P1jTerminal::Cancelled);
        assert!(fence.start());
        let mut seen = Vec::new();
        fence.drain(|epoch, reason| {
            seen.push((epoch, reason));
            if epoch == 1 {
                // A stale cleanup worker was already dispatched when the next
                // occupant's foreground cleanup hit contention.
                fence.request(2, P1jTerminal::UsedMatching);
                assert!(!fence.start());
            }
        });
        assert_eq!(
            seen,
            [(1, P1jTerminal::Cancelled), (2, P1jTerminal::UsedMatching)]
        );
        assert!(!fence.running.load(Ordering::Acquire));
        assert!(fence.cancelled(2));
        assert!(!fence.cancelled(3));
    }

    #[test]
    fn p1j_stale_or_duplicate_cleanup_preserves_latest_epoch_and_first_terminal_reason() {
        let fence = P1jRetirement::default();
        fence.request(2, P1jTerminal::Unused);
        fence.request(1, P1jTerminal::Cancelled);
        fence.request(2, P1jTerminal::Cancelled);
        assert!(fence.start());
        let mut seen = Vec::new();
        fence.drain(|epoch, reason| seen.push((epoch, reason)));
        assert_eq!(seen, [(2, P1jTerminal::Unused)]);
        fence.request(3, P1jTerminal::RequestEnded);
        assert!(fence.start());
        fence.drain(|epoch, reason| seen.push((epoch, reason)));
        assert_eq!(seen[1], (3, P1jTerminal::RequestEnded));
    }
    #[test]
    fn p1j_writer_panic_closes_after_view_drop_and_releases_lease() {
        use crate::expert_cache::{GpuResident, HOST_BACKED_Q4_BYTES};
        let cache = GpuExpertCache::new(HOST_BACKED_Q4_BYTES, 0.0, 0);
        assert!(cache.promote_sync(Arc::new(GpuResident::new_with_dtype(
            1,
            vec![0; HOST_BACKED_Q4_BYTES],
            crate::inference::WeightDtype::Q4_0
        ))));
        let HostBackedLeaseResult::Acquired(lease) =
            cache.try_lease_host_backed(1, cache.current_generation(1).unwrap())
        else {
            panic!("lease");
        };
        let events = std::cell::RefCell::new(Vec::new());
        struct View<'a>(&'a std::cell::RefCell<Vec<&'static str>>);
        impl Drop for View<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().push("payload-enqueue");
            }
        }
        p1j_stage_lease(
            lease,
            |_| {
                let _view = View(&events);
                panic!("injected staging panic");
            },
            |success| {
                assert!(!success);
                events.borrow_mut().push("failure-closed");
            },
        );
        assert_eq!(*events.borrow(), ["payload-enqueue", "failure-closed"]);
        assert_eq!(
            (
                cache.host_backed_lease_snapshot().active,
                cache.host_backed_lease_snapshot().releases
            ),
            (0, 1)
        );
    }
    #[test]
    fn p1j_launch_and_writer_have_no_source_io_or_demand_buffer_capability() {
        let source = include_str!("gpu_native_residency.rs");
        let body = source
            .split("impl P1jSidecarOwner {")
            .nth(1)
            .unwrap()
            .split("impl GpuNativeTieredResidencyManager {")
            .next()
            .unwrap();
        for forbidden in [
            "ExpertResident",
            "gpu_native_source_upload",
            "engine.",
            "fetch_with_retry",
            "read_expert",
            "Nvme",
            "Semaphore",
            ".to_vec(",
        ] {
            assert!(!body.contains(forbidden), "{forbidden}");
        }
        let backend = include_str!("backend/gpu_native.rs");
        let writer = backend
            .split("pub(crate) fn write_p1j_payload(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn try_publish_p1j(")
            .next()
            .unwrap();
        assert!(writer.contains("CheckedPhysicalQ4ExpertSlot::new"));
        assert!(writer.contains("drop(view)"));
        for forbidden in [
            "arena.mapping",
            ".submit(",
            ".poll(",
            "Vec::",
            "vec![",
            ".to_vec(",
            "ExpertResident",
        ] {
            assert!(!writer.contains(forbidden), "{forbidden}");
        }
    }
    #[test]
    fn p1j_allocation_is_internal_once_and_ordinary_model_plan_stays_frozen() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(2048, 768, 128, 8).unwrap();
        let plan =
            GpuNativeModelExpertVramPlan::try_new(48, geometry, 2048 * 1024 * 1024, &limits())
                .unwrap();
        assert_eq!(plan.layer_plans()[47].slot_capacity(), 8);
        let source = include_str!("gpu_native_residency.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert_eq!(source.matches(".create_p1j_sidecar(").count(), 1);
        let setup = source
            .split("pub(crate) fn enable_p1j_sidecar(")
            .nth(1)
            .unwrap()
            .split("fn p1j_current(")
            .next()
            .unwrap();
        assert!(setup.contains("get_or_init"));
        let backend = include_str!("backend/gpu_native.rs");
        let allocation = backend
            .split("pub(crate) fn create_p1j_sidecar(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn write_p1j_payload(")
            .next()
            .unwrap();
        assert!(
            allocation.find("validate_startup_buffer").unwrap()
                < allocation
                    .find("let buffer = create_startup_buffer")
                    .unwrap()
        );
    }
    #[test]
    fn p1e_frozen_plan_has_exactly_eight_layer47_slots_without_device_construction() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(2048, 768, 128, 8).unwrap();
        let limits = limits();
        let plan = GpuNativeModelExpertVramPlan::try_new(48, geometry, 2048 * 1024 * 1024, &limits)
            .unwrap();
        assert_eq!(plan.layer_plans()[47].slot_capacity(), 8);
    }

    #[test]
    fn p1e_read_only_snapshot_and_shadow_match_real_lru_victim_helpers() {
        use crate::predictor_v2::p1e::{Namespace, Resident, Shadow};
        let namespace = Namespace {
            runtime: 1,
            context: 2,
            arena: 3,
            layer: 47,
            capacity: 8,
        };
        let mut residents = LruCache::unbounded();
        for expert in 0..8 {
            residents.put(
                47 * 128 + expert,
                Resident {
                    expert,
                    generation: 10,
                    bank: 0,
                    slot: expert,
                    epoch: 1,
                },
            );
        }
        let copy = |cache: &LruCache<u32, Resident>| {
            copy_p1e_snapshot(namespace, cache, cache.len(), |_, r| Some(*r)).unwrap()
        };
        let initial = copy(&residents);
        let mut shadow = Shadow::new(initial).unwrap();
        for _ in 0..20 {
            assert_eq!(copy(&residents), initial);
            assert!(!shadow
                .evidence(copy(&residents))
                .unwrap()
                .snapshot
                .current(namespace, 8)
                .unwrap());
        }
        let ids = [8, 0, 1, 2, 3, 4, 5, 9];
        shadow.demand(&ids);
        let protected = ids.iter().map(|&id| 47 * 128 + id).collect::<HashSet<_>>();
        let mut missing = Vec::new();
        for &id in &ids {
            if touch_physical_record(&mut residents, 47 * 128 + id).is_none() {
                missing.push(id);
            }
        }
        let mut free = Vec::new();
        while residents.len() + missing.len() > 8 {
            let victim = oldest_unprotected(&residents, &protected).unwrap();
            let old = residents.pop(&victim).unwrap();
            shadow.victim(old.expert);
            free.push(old.slot);
        }
        for (id, slot) in missing.into_iter().zip(free) {
            let record = Resident {
                expert: id,
                generation: 11,
                bank: 0,
                slot,
                epoch: 2,
            };
            residents.put(47 * 128 + id, record);
            shadow.committed_install(record);
        }
        let after = copy(&residents);
        assert_eq!(shadow.evidence(after).unwrap().snapshot, after);
        assert_eq!(
            after.residents.map(|r| r.unwrap().expert),
            [0, 1, 2, 3, 4, 5, 8, 9]
        );
        // Real host logical eviction has no physical event or shadow mutation.
        let logical = GpuExpertCache::new(16, 0.0, 0);
        assert!(
            logical.promote_sync(Arc::new(crate::expert_cache::GpuResident::new(
                47 * 128 + 8,
                vec![0; 16]
            )))
        );
        assert!(
            logical.promote_sync(Arc::new(crate::expert_cache::GpuResident::new(
                999,
                vec![0; 16]
            )))
        );
        assert!(!logical.contains(47 * 128 + 8));
        assert_eq!(copy(&residents), after);
        assert!(shadow
            .evidence(after)
            .unwrap()
            .snapshot
            .current(namespace, 8)
            .unwrap());
        assert!(copy_p1e_snapshot(namespace, &residents, 7, |_, r| Some(*r)).is_err());
        assert!(copy_p1e_snapshot(namespace, &residents, 8, |_, _| None).is_err());
        let mut wrong = LruCache::unbounded();
        wrong.put(
            8,
            Resident {
                expert: 8,
                generation: 1,
                bank: 0,
                slot: 0,
                epoch: 1,
            },
        );
        assert!(copy_p1e_snapshot(namespace, &wrong, 1, |_, r| Some(*r)).is_err());
    }

    #[test]
    fn p1e_physical_seam_uses_exact_non_touching_arena_validation() {
        let source = include_str!("gpu_native_residency.rs");
        let seam = source
            .split("    fn p1e_copy_locked(")
            .nth(1)
            .unwrap()
            .split("    /// Internal observation opt-in")
            .next()
            .unwrap();
        for required in [
            "record.key.layer_index()",
            "record.key.expert_id()",
            "record.residency.key() == record.key",
            "contains_exact_residency(self.executor.context_id(), record.residency)",
            "&state.residents",
        ] {
            assert!(seam.contains(required), "{required}");
        }
        for forbidden in [
            ".get(&",
            "touch_physical_record",
            "acquire_q4",
            "retire_q4",
            ".fetch_add",
            "device.poll",
            "queue.submit",
        ] {
            assert!(!seam.contains(forbidden), "{forbidden}");
        }
        let copy = source
            .split("fn copy_p1e_snapshot<T>(")
            .nth(1)
            .unwrap()
            .split("fn p1e_resident")
            .next()
            .unwrap();
        assert!(copy.contains("residents: &LruCache"));
        assert!(!copy.contains("&mut"));
    }

    use crate::buffer_pool::BufferPool;
    use crate::expert_cache::GpuResident;

    fn geometry() -> GpuNativeQ4ExpertGeometry {
        GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 2).unwrap()
    }

    fn limits() -> wgpu::Limits {
        wgpu::Limits {
            max_push_constant_size: 32,
            max_storage_buffers_per_shader_stage: 8,
            max_compute_workgroup_size_x: 64,
            max_compute_invocations_per_workgroup: 64,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn ordinary_and_qualification_paths_select_production_concurrency_and_legacy_speculation() {
        assert_eq!(
            ordinary_demand_install_path(),
            DemandPhysicalInstallPath::ProductionConcurrentDirectStaging
        );
        assert_eq!(
            oracle_safe_boundary_install_path(GpuNativePhysicalSlotFillPolicy::FullSlotZero),
            DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
        );
        assert_eq!(
            production_concurrency_control_path(),
            DemandPhysicalInstallPath::SequentialDirectStagingControl
        );
        assert_eq!(
            speculative_install_path(),
            DemandPhysicalInstallPath::LegacyFullSlotVecControl
        );
        assert_ne!(
            production_concurrency_control_path(),
            speculative_install_path()
        );
        assert_ne!(
            oracle_safe_boundary_install_path(GpuNativePhysicalSlotFillPolicy::FullSlotZero),
            speculative_install_path()
        );
    }

    #[test]
    fn oracle_full_zero_control_is_explicit_and_concurrent() {
        assert_ne!(
            oracle_safe_boundary_install_path(GpuNativePhysicalSlotFillPolicy::FullSlotZero),
            ordinary_demand_install_path()
        );
        assert_eq!(
            oracle_safe_boundary_install_path(GpuNativePhysicalSlotFillPolicy::FullSlotZero),
            DemandPhysicalInstallPath::QualificationConcurrentDirectStagingFullZero
        );
    }

    #[test]
    fn oracle_direct_staging_never_uses_legacy_full_slot_vec_control() {
        assert_ne!(
            oracle_safe_boundary_install_path(GpuNativePhysicalSlotFillPolicy::FullSlotZero),
            speculative_install_path()
        );
    }

    #[test]
    fn oracle_no_zero_fill_is_a_qualification_only_concurrent_path() {
        let treatment = oracle_safe_boundary_install_path(
            GpuNativePhysicalSlotFillPolicy::QualificationNoZeroFill,
        );
        assert_eq!(
            treatment,
            DemandPhysicalInstallPath::QualificationConcurrentDirectStagingNoZeroFill
        );
        assert_ne!(treatment, ordinary_demand_install_path());
        assert_ne!(treatment, production_concurrency_control_path());
        assert_ne!(treatment, speculative_install_path());
        // Production and historical treatment share checked no-zero staging;
        // the concurrent control alone explicitly selects full zero fill.
        let backend = include_str!("backend/gpu_native.rs");
        for (wrapper, specialization) in [
            (
                "stage_q4_expert_residency_production<'a>",
                "::<false, true, true>",
            ),
            (
                "stage_q4_expert_residency_production_observed<'a>",
                "::<true, true, true>",
            ),
            (
                "stage_q4_expert_residency_full_zero_control_observed<'a>",
                "::<true, true, false>",
            ),
            (
                "stage_q4_expert_residency_qualification<'a>",
                "::<true, false, false>",
            ),
            (
                "stage_q4_expert_residency_qualification_no_zero_fill<'a>",
                "::<true, true, true>",
            ),
        ] {
            let body = backend
                .split(&format!("pub(crate) fn {wrapper}"))
                .nth(1)
                .unwrap()
                .split("\n    }")
                .next()
                .unwrap();
            assert!(
                body.contains(&format!(
                    "stage_q4_expert_residency_inner{specialization}(permit, payload)"
                )),
                "{wrapper}"
            );
        }
    }

    #[test]
    fn treatment_physical_install_total_is_post_reservation_stage_plus_commit() {
        let demand_set_transaction_us = 10_000;
        let individual_stage_us = 37;
        let commit_us = 11;
        let per_expert_total =
            post_reservation_physical_install_total_us(individual_stage_us, commit_us);

        assert_eq!(per_expert_total, 48);
        assert_ne!(per_expert_total, demand_set_transaction_us);
        assert_eq!(
            post_reservation_physical_install_total_us(u64::MAX, 1),
            u64::MAX
        );
    }

    #[test]
    fn treatment_widths_one_through_eight_collect_and_touch_lru_in_request_order() {
        for width in 1usize..=8 {
            let task_count = AtomicU64::new(0);
            let request_order = (0..width as u32).collect::<Vec<_>>();
            let staged = collect_physical_stage_results_in_request_order(
                request_order.clone(),
                width >= 2,
                |global_id| {
                    task_count.fetch_add(1, Ordering::Relaxed);
                    global_id
                },
            );
            assert_eq!(task_count.load(Ordering::Relaxed), width as u64);
            assert_eq!(staged, request_order);

            let mut residents = LruCache::unbounded();
            for global_id in staged {
                residents.put(global_id, ());
            }
            assert_eq!(
                residents.iter().map(|(&id, _)| id).collect::<Vec<_>>(),
                request_order.into_iter().rev().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn global_layer_local_identity_is_checked_without_clamping() {
        assert_eq!(
            global_to_layer_local(0, 2, 128).unwrap(),
            GpuNativeLayerExpertId {
                layer_index: 0,
                local_expert_id: 0,
            }
        );
        assert_eq!(global_to_layer_local(127, 2, 128).unwrap().layer_index, 0);
        assert_eq!(
            global_to_layer_local(128, 2, 128).unwrap(),
            GpuNativeLayerExpertId {
                layer_index: 1,
                local_expert_id: 0,
            }
        );
        assert_eq!(layer_local_to_global(1, 127, 2, 128).unwrap(), 255);
        assert!(matches!(
            global_to_layer_local(256, 2, 128),
            Err(GpuNativeTieredResidencyError::GlobalExpertOutOfRange { .. })
        ));
        assert!(layer_local_to_global(2, 0, 2, 128).is_err());
        assert!(layer_local_to_global(1, 128, 2, 128).is_err());
        let generation = 0xfedc_ba98_7654_3210;
        let key = global_to_q4_expert_key(129, generation, 2, 128).unwrap();
        assert_eq!(key.layer_index(), 1);
        assert_eq!(key.expert_id(), 1);
        assert_eq!(key.logical_generation(), generation);
    }

    #[test]
    fn logical_admission_generation_reaches_physical_key_unchanged() {
        let cache = GpuExpertCache::new(8, 0.0, 0);
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![1; 8])))
            .unwrap();
        let first = cache.current_admission(7).unwrap();
        let first_key = global_to_q4_expert_key(7, first.generation(), 2, 128).unwrap();
        assert_eq!(first_key.logical_generation(), first.generation());
        assert!(cache.contains_generation(7, first_key.logical_generation()));

        // Fill the one-entry LRU with another identity, then readmit the same
        // global expert. Only GpuExpertCache advances the logical generation;
        // the physical key is a lossless consumer of that identity.
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![2; 8])))
            .unwrap();
        assert!(!cache.contains_generation(7, first.generation()));
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![3; 8])))
            .unwrap();
        let newer = cache.current_admission(7).unwrap();
        assert!(newer.generation() > first.generation());
        let newer_key = global_to_q4_expert_key(7, newer.generation(), 2, 128).unwrap();
        assert_eq!(newer_key.logical_generation(), newer.generation());
        assert!(!cache.contains_generation(7, first_key.logical_generation()));
        assert!(cache.contains_generation(7, newer_key.logical_generation()));
    }

    #[test]
    fn physical_install_source_requires_current_matching_logical_admission() {
        let cache = GpuExpertCache::new(8, 0.0, 0);
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![1; 8])))
            .unwrap();
        let admission = cache.current_admission(7).unwrap();
        let pool = BufferPool::new(2, 8, 4);
        let resident = Arc::new(ExpertResident::new(7, pool.try_acquire().unwrap()));
        assert_eq!(
            validate_physical_install_source(&cache, 7, &resident, &admission),
            Ok(())
        );

        let wrong_resident = Arc::new(ExpertResident::new(8, pool.try_acquire().unwrap()));
        assert_eq!(
            validate_physical_install_source(&cache, 7, &wrong_resident, &admission),
            Err(GpuNativeTieredResidencyError::DemandSourceIdentityMismatch { global_id: 7 })
        );

        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![2; 8])))
            .unwrap();
        assert_eq!(
            validate_physical_install_source(&cache, 7, &resident, &admission),
            Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
                global_id: 7,
                generation: admission.generation(),
            })
        );
    }

    #[test]
    fn qualification_logical_only_physical_validation_uses_real_source_and_rejects_stale_or_wrong_ids(
    ) {
        let cache = GpuExpertCache::new(8, 0.0, 0);
        let audit = Arc::new(AtomicU64::new(0));
        let pool = BufferPool::new_qualification_oracle_future_source(2, 8, 4);
        let mut buffer = pool.try_acquire().unwrap();
        buffer
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let resident = Arc::new(ExpertResident::new(7, buffer));
        cache
            .demand_admit_lru(Arc::new(GpuResident::new_qualification_logical_only(
                7,
                resident.data().len(),
                crate::inference::WeightDtype::Q4_0,
                audit.clone(),
            )))
            .unwrap();
        let admission = cache.current_admission(7).unwrap();
        assert_eq!(
            validate_physical_install_source(&cache, 7, &resident, &admission),
            Ok(())
        );
        let demand = GpuNativeDemandExpert::install(7, resident.clone(), admission.clone());
        match &demand {
            GpuNativeDemandExpert::Install {
                resident: source,
                admission: logical,
                ..
            } => {
                assert!(Arc::ptr_eq(source, &resident));
                assert_eq!(source.data(), &[1, 2, 3, 4, 5, 6, 7, 8]);
                assert_eq!(logical.generation(), admission.generation());
            }
            _ => panic!("install must carry the real source"),
        }
        let wrong = Arc::new(ExpertResident::new(8, pool.try_acquire().unwrap()));
        assert_eq!(
            validate_physical_install_source(&cache, 7, &wrong, &admission),
            Err(GpuNativeTieredResidencyError::DemandSourceIdentityMismatch { global_id: 7 })
        );
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![0; 8])))
            .unwrap();
        assert!(matches!(
            validate_physical_install_source(&cache, 7, &resident, &admission),
            Err(GpuNativeTieredResidencyError::LogicalAdmissionStale { .. })
        ));
        assert_eq!(audit.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn qualification_logical_only_physical_stage_source_witness() {
        // Both production calls and both qualification calls validate the
        // ExpertResident field of ReservedPhysicalInstall. Pin this contract
        // without changing the frozen engine/hash witnesses or requiring a GPU.
        let source = include_str!("gpu_native_residency.rs");
        let stage = source
            .split("let stage_one = |reserved:")
            .nth(1)
            .unwrap()
            .split("let parallel_stage_started =")
            .next()
            .unwrap();
        assert_eq!(stage.matches("reserved.resident.data()").count(), 4);
        assert_eq!(stage.matches("stage_q4_expert_source_upload(").count(), 1);
        let production = stage.split("} else if OBSERVE {").nth(1).unwrap();
        assert_eq!(production.matches("reserved.resident.data()").count(), 2);
        assert!(stage.contains("stage_q4_expert_residency_full_zero_control_observed("));
        assert!(stage.contains("stage_q4_expert_residency_production_observed("));
        assert!(stage.contains(".stage_q4_expert_residency_production("));
        assert!(!stage.contains("admission.resident()"));
        let owners = source
            .split("struct ReservedPhysicalInstall<'a> {")
            .nth(1)
            .unwrap()
            .split('}')
            .next()
            .unwrap();
        assert!(owners.contains("resident: Arc<ExpertResident>"));
    }

    #[test]
    fn pre_fix_foreground_admission_self_evicts_a_current_selected_source() {
        let cache = GpuExpertCache::new(16, 0.0, 0);
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![1; 8])))
            .unwrap();
        let selected_a_generation = cache.current_generation(7).unwrap();
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(99, vec![9; 8])))
            .unwrap();

        let selected_a = GpuNativeDemandExpert::current(7);
        assert!(cache.contains_generation(7, selected_a_generation));

        // This is the old engine ordering: A was classified as Current, then
        // selected B was admitted independently. A is the logical LRU even
        // though non-selected 99 is an eligible victim and A+B fit together.
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![2; 8])))
            .unwrap();
        assert!(!cache.contains_generation(7, selected_a_generation));
        assert!(cache.contains(99));
        assert!(cache.contains(8));

        let final_resolution = if cache.contains_generation(7, selected_a_generation) {
            Ok(())
        } else {
            match selected_a {
                GpuNativeDemandExpert::Current { global_id } => {
                    Err(GpuNativeTieredResidencyError::DemandSourceMissing { global_id })
                }
                GpuNativeDemandExpert::Install { .. } => unreachable!(),
            }
        };
        assert_eq!(
            final_resolution,
            Err(GpuNativeTieredResidencyError::DemandSourceMissing { global_id: 7 })
        );
    }

    #[test]
    fn model_plan_requires_top_k_slots_per_layer_and_counts_every_arena() {
        let limits = limits();
        let geometry = geometry();
        let layer_min =
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, geometry.top_k(), &limits)
                .unwrap();
        let exact = layer_min.total_arena_allocation_bytes() * 2;
        assert!(matches!(
            GpuNativeModelExpertVramPlan::try_new(2, geometry, exact - 1, &limits),
            Err(GpuNativeTieredResidencyError::ModelBudgetTooSmall { .. })
        ));
        let plan = GpuNativeModelExpertVramPlan::try_new(2, geometry, exact, &limits).unwrap();
        assert_eq!(plan.minimum_executable_budget_bytes(), exact);
        assert_eq!(plan.total_arena_allocation_bytes(), exact);
        assert_eq!(plan.layer_plans().len(), 2);
        assert!(plan
            .layer_plans()
            .iter()
            .all(|layer| layer.slot_capacity() == geometry.top_k()));
        assert!(plan.layer_plans().iter().all(|layer| {
            layer.total_arena_allocation_bytes()
                >= layer.physical_bank_allocation_bytes() + layer.mapping_metadata_bytes()
        }));
        assert!(plan.layer_plans().iter().all(|layer| {
            layer.physical_bank_allocation_bytes() - layer.active_bank_allocation_bytes()
                == 3 * std::mem::size_of::<u32>() as u64
        }));
        assert!(plan
            .layer_plans()
            .iter()
            .all(|layer| layer.total_arena_allocation_bytes() < exact));
    }

    #[test]
    fn model_plan_is_deterministic_monotonic_bounded_and_saturating() {
        let limits = limits();
        let geometry = geometry();
        let minimum = GpuNativeModelExpertVramPlan::try_new(
            3,
            geometry,
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, geometry.top_k(), &limits)
                .unwrap()
                .total_arena_allocation_bytes()
                * 3,
            &limits,
        )
        .unwrap();
        let extra = geometry.slot_stride_bytes() as u64 * 7;
        let larger = GpuNativeModelExpertVramPlan::try_new(
            3,
            geometry,
            minimum.total_expert_budget_bytes() + extra,
            &limits,
        )
        .unwrap();
        let repeat = GpuNativeModelExpertVramPlan::try_new(
            3,
            geometry,
            minimum.total_expert_budget_bytes() + extra,
            &limits,
        )
        .unwrap();
        assert_eq!(larger, repeat);
        assert!(larger.total_arena_allocation_bytes() <= larger.total_expert_budget_bytes());
        for (small, large) in minimum.layer_plans().iter().zip(larger.layer_plans()) {
            assert!(large.slot_capacity() >= small.slot_capacity());
            assert!(large.slot_capacity() <= geometry.num_experts());
        }
        for layer in larger.layer_plans() {
            if layer.slot_capacity() < geometry.num_experts() {
                let next = GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(
                    geometry,
                    layer.slot_capacity() + 1,
                    &limits,
                )
                .unwrap();
                let increment =
                    next.total_arena_allocation_bytes() - layer.total_arena_allocation_bytes();
                assert!(increment > larger.unused_remainder_bytes());
            }
        }
    }

    #[test]
    fn protected_demand_members_are_never_lru_victims() {
        let mut lru = LruCache::unbounded();
        for id in 0..8 {
            lru.put(id, ());
        }
        let protected = HashSet::from([0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(oldest_unprotected(&lru, &protected), Some(7));
        let all = (0..8).collect::<HashSet<_>>();
        assert_eq!(oldest_unprotected(&lru, &all), None);

        let mut other_layer = LruCache::unbounded();
        other_layer.put(128, ());
        other_layer.put(129, ());
        let other_before = other_layer.iter().map(|(&id, _)| id).collect::<Vec<_>>();
        let _ = oldest_unprotected(&lru, &protected);
        assert_eq!(
            other_layer.iter().map(|(&id, _)| id).collect::<Vec<_>>(),
            other_before
        );
    }

    #[test]
    fn normal_physical_demand_hit_promotes_lru_recency() {
        let mut residents = LruCache::unbounded();
        residents.put(7, 70);
        residents.put(8, 80);
        assert_eq!(oldest_unprotected(&residents, &HashSet::new()), Some(7));

        assert_eq!(touch_physical_record(&mut residents, 7), Some(70));
        assert_eq!(oldest_unprotected(&residents, &HashSet::new()), Some(8));
        assert_eq!(residents.peek(&7), Some(&70));
        assert_eq!(residents.peek(&8), Some(&80));
    }
}
