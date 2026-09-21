//! Predictor-v2 P0/P1E: bounded, request-owned CPU observation, never scheduling.
//!
//! Inputs are copied local route IDs and value metadata. There are deliberately
//! no crate imports, callbacks, execution handles, I/O, or asynchronous methods.
//! Live completion observation only records truth and request-local transitions.
//! The synthetic lifecycle below models work; it cannot perform that work.
#![allow(dead_code)] // Internal P0 seam; no CLI/configuration activation.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) enum RequestPhase {
    Serving,
    Warmup,
    Measured,
    Fixture,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct RequestIdentity {
    /// Process-local, never-reused runtime incarnation. Not a persistent ID.
    pub runtime_namespace: u64,
    pub request_sequence: u64,
    pub phase: RequestPhase,
    pub phase_run_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) enum PositionKind {
    Prompt,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct PositionIdentity {
    pub absolute_position: u64,
    pub position_kind: PositionKind,
    pub decode_index: Option<u64>,
}

impl PositionIdentity {
    pub fn from_prompt_length(position: usize, prompt_length: usize) -> AccountingResult<Self> {
        if prompt_length == 0 {
            return Err(AccountingError::InvalidIdentity);
        }
        Ok(Self {
            absolute_position: u64::try_from(position).map_err(|_| AccountingError::Overflow)?,
            position_kind: if position < prompt_length {
                PositionKind::Prompt
            } else {
                PositionKind::Decode
            },
            decode_index: if position < prompt_length {
                None
            } else {
                Some(
                    u64::try_from(
                        position
                            .checked_sub(prompt_length)
                            .ok_or(AccountingError::Overflow)?,
                    )
                    .map_err(|_| AccountingError::Overflow)?,
                )
            },
        })
    }

    fn valid(self) -> bool {
        match (self.position_kind, self.decode_index) {
            (PositionKind::Prompt, None) => true,
            (PositionKind::Decode, Some(index)) => index < self.absolute_position,
            _ => false,
        }
    }

    fn follows(self, source: Self) -> bool {
        if !self.valid()
            || !source.valid()
            || source.absolute_position.checked_add(1) != Some(self.absolute_position)
        {
            return false;
        }
        match (source.position_kind, self.position_kind) {
            (PositionKind::Prompt, PositionKind::Prompt) => true,
            (PositionKind::Prompt, PositionKind::Decode) => self.decode_index == Some(0),
            (PositionKind::Decode, PositionKind::Decode) => {
                source.decode_index.and_then(|n| n.checked_add(1)) == self.decode_index
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct ModelMetadata {
    pub num_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

impl ModelMetadata {
    fn valid(self) -> bool {
        self.num_layers > 0
            && self.num_experts > 0
            && self.top_k > 0
            && self.top_k <= self.num_experts
            && u32::try_from(self.num_experts).is_ok()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct RoutePoint {
    pub request: RequestIdentity,
    pub position: PositionIdentity,
    pub layer: usize,
}

fn adjacent(source: RoutePoint, target: RoutePoint, model: ModelMetadata) -> bool {
    source.request == target.request
        && source.position.valid()
        && target.position.valid()
        && source.layer < model.num_layers
        && target.layer < model.num_layers
        && if source.position == target.position {
            source.layer.checked_add(1) == Some(target.layer)
        } else {
            source.layer.checked_add(1) == Some(model.num_layers)
                && target.layer == 0
                && target.position.follows(source.position)
        }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) enum PredictorSource {
    CpuFixture,
    CompletedPositionReplay,
    Layer47Temporal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct PredictionIdentity {
    pub request: RequestIdentity,
    pub source_position: PositionIdentity,
    pub source_layer: usize,
    pub target_position: PositionIdentity,
    pub target_layer: usize,
    pub expert_local_id: u32,
    pub prediction_generation: u64,
    pub candidate_sequence: u64,
    pub predictor_source: PredictorSource,
    pub predictor_revision: u64,
}

impl PredictionIdentity {
    fn target(self) -> RoutePoint {
        RoutePoint {
            request: self.request,
            position: self.target_position,
            layer: self.target_layer,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct SourceAcquisitionIdentity {
    pub owner_request: RequestIdentity,
    pub acquisition_sequence: u64,
    pub layer: usize,
    pub expert_local_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct ReservationIdentity {
    pub acquisition: SourceAcquisitionIdentity,
    pub ticket_sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct PhysicalInstallIdentity {
    pub runtime_namespace: u64,
    pub executor_namespace: u64,
    pub model_namespace: u64,
    pub layer: usize,
    pub expert_local_id: u32,
    pub logical_generation: u64,
    pub bank: u32,
    pub slot: u32,
    pub slot_epoch: u64,
    pub reservation: ReservationIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct PhysicalSlot {
    runtime: u64,
    executor: u64,
    model: u64,
    bank: u32,
    slot: u32,
}

impl PhysicalInstallIdentity {
    fn slot_key(self) -> PhysicalSlot {
        PhysicalSlot {
            runtime: self.runtime_namespace,
            executor: self.executor_namespace,
            model: self.model_namespace,
            bank: self.bank,
            slot: self.slot,
        }
    }
}

/// All fields describe a synthetic fixture, not an executable reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstallLocation {
    pub executor_namespace: u64,
    pub model_namespace: u64,
    pub logical_generation: u64,
    pub bank: u32,
    pub slot: u32,
    pub slot_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) struct DemandIdentity {
    pub request: RequestIdentity,
    pub target_position: PositionIdentity,
    pub layer: usize,
    pub expert_local_id: u32,
    pub demand_sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) enum SkipReason {
    Policy,
    Retrospective,
    RequestEnd,
    Cancelled,
    Superseded,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum AdmissionDisposition {
    Accepted,
    Rejected,
    Skipped(SkipReason),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub(crate) enum TerminalReason {
    ConsumedByMatchingRoute,
    EvictedUnused,
    Cancelled,
    Superseded,
    RequestEnded,
    Rejected,
    Skipped(SkipReason),
    SourceFailed,
    SourceCancelled,
    InstallFailed,
    ReservationAborted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PredictionStage {
    Emitted,
    AdmissionDecision,
    SourceRequested,
    SourceCompleted,
    ResidencyReserved,
    Installed,
    Available,
    Terminal(TerminalReason),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceKind {
    SyntheticRead,
    SyntheticAlreadyAvailable,
    HostBackedLogical,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceState {
    Requested,
    Completed,
    Failed,
    Cancelled,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReservationState {
    Live,
    Committed,
    Aborted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstallOrigin {
    SpeculativeFixture,
    RestorationFixture,
    PredictorV2Sidecar,
}

/// Exact scalar identity of the isolated writer, independent of ordinary slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct P1jIdentity {
    pub candidate: p1e::Candidate,
    pub logical_generation: u64,
    pub epoch: u32,
    pub writer_sequence: u64,
}

impl P1jIdentity {
    pub(crate) fn global_id(self) -> u32 {
        (p1e::LAYER * p1e::EXPERTS) as u32 + self.candidate.expert
    }
    pub(crate) fn prediction(self) -> PredictionIdentity {
        let c = self.candidate;
        PredictionIdentity {
            request: c.request,
            source_position: c.source_position,
            source_layer: c.source_layer,
            target_position: c.target_position,
            target_layer: c.target_layer,
            expert_local_id: c.expert,
            prediction_generation: c.generation,
            candidate_sequence: c.sequence,
            predictor_source: PredictorSource::Layer47Temporal,
            predictor_revision: c.signal_revision,
        }
    }
    pub(crate) fn location(self) -> InstallLocation {
        InstallLocation {
            executor_namespace: self.candidate.namespace.context,
            model_namespace: self.candidate.namespace.arena as u64,
            logical_generation: self.logical_generation,
            bank: 1,
            slot: 0,
            slot_epoch: u64::from(self.epoch),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum P1jTerminal {
    UsedMatching,
    Unused,
    StaleLogicalGeneration,
    OrdinarySuperseded,
    Cancelled,
    RequestEnded,
    WriteFailure,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InstallState {
    Installed,
    Available,
    Evicted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DemandStatus {
    CleanCommitted,
    FailedAttempt,
    ReplayedSegment,
}

#[derive(Clone, Debug)]
struct PredictionRecord {
    // Immutable map key is the full frozen identity; rank cannot be rewritten.
    stage: PredictionStage,
    admission: Option<AdmissionDisposition>,
    acquisition: Option<SourceAcquisitionIdentity>,
    install: Option<PhysicalInstallIdentity>,
    consumed_by: Option<DemandIdentity>,
    causal: bool,
}
#[derive(Clone, Debug)]
struct SourceRecord {
    leader: PredictionIdentity,
    nominations: BTreeSet<PredictionIdentity>,
    kind: SourceKind,
    state: SourceState,
    reservation: Option<PhysicalInstallIdentity>,
}
#[derive(Clone, Debug)]
struct ReservationRecord {
    owner: PredictionIdentity,
    followers: BTreeSet<PredictionIdentity>,
    state: ReservationState,
    origin: InstallOrigin,
}
#[derive(Clone, Debug)]
struct InstallRecord {
    state: InstallState,
    origin: InstallOrigin,
    available_event: Option<u64>,
    credited_demand: Option<DemandIdentity>,
}
#[derive(Clone, Debug)]
struct ObservedRoute {
    ids: Vec<u32>,
    event: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) enum AccountingError {
    InvalidIdentity,
    InvalidTransition,
    ReusedIdentity,
    Overflow,
    Capacity,
    Incomplete,
    Closed,
}
type AccountingResult<T> = Result<T, AccountingError>;

fn checked_len(n: usize) -> AccountingResult<u64> {
    u64::try_from(n).map_err(|_| AccountingError::Overflow)
}
fn increment(n: &mut u64) -> AccountingResult<()> {
    *n = n.checked_add(1).ok_or(AccountingError::Overflow)?;
    Ok(())
}
fn require(value: bool, error: AccountingError) -> AccountingResult<()> {
    if value {
        Ok(())
    } else {
        Err(error)
    }
}

/// Counts are derived from retained records, with checked arithmetic. A capacity
/// limit applies to EACH collection; terminal records are never pruned/reused.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct ReconciliationSnapshot {
    pub incomplete: Option<AccountingError>,
    pub emitted: u64,
    pub terminal_predictions: u64,
    pub live_predictions: u64,
    pub admission_pending: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub skipped: u64,
    pub terminal_categories: Vec<(TerminalReason, u64)>,
    pub source_leaders: u64,
    pub source_followers: u64,
    pub source_completed: u64,
    pub source_failed: u64,
    pub source_cancelled: u64,
    pub source_live: u64,
    pub reservations: u64,
    pub reservations_committed: u64,
    pub reservations_aborted: u64,
    pub reservations_live: u64,
    pub install_owners: u64,
    pub install_followers: u64,
    pub available_installs: u64,
    pub direct_matching_demand_credits: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct LifecycleLedger {
    request: RequestIdentity,
    model: ModelMetadata,
    capacity: usize,
    incomplete: Option<AccountingError>,
    closed: bool,
    event: u64,
    generation: u64,
    candidate_sequence: u64,
    acquisition_sequence: u64,
    ticket_sequence: u64,
    demand_sequence: u64,
    predictions: BTreeMap<PredictionIdentity, PredictionRecord>,
    sources: BTreeMap<SourceAcquisitionIdentity, SourceRecord>,
    reservations: BTreeMap<PhysicalInstallIdentity, ReservationRecord>,
    installs: BTreeMap<PhysicalInstallIdentity, InstallRecord>,
    current: BTreeMap<PhysicalSlot, PhysicalInstallIdentity>,
    demands: BTreeMap<DemandIdentity, PhysicalInstallIdentity>,
    observations: BTreeMap<RoutePoint, ObservedRoute>,
}

impl LifecycleLedger {
    pub fn new(
        request: RequestIdentity,
        model: ModelMetadata,
        capacity: usize,
    ) -> AccountingResult<Self> {
        require(
            request.runtime_namespace > 0 && request.request_sequence > 0 && model.valid(),
            AccountingError::InvalidIdentity,
        )?;
        Ok(Self {
            request,
            model,
            capacity,
            incomplete: None,
            closed: false,
            event: 0,
            generation: 0,
            candidate_sequence: 0,
            acquisition_sequence: 0,
            ticket_sequence: 0,
            demand_sequence: 0,
            predictions: BTreeMap::new(),
            sources: BTreeMap::new(),
            reservations: BTreeMap::new(),
            installs: BTreeMap::new(),
            current: BTreeMap::new(),
            demands: BTreeMap::new(),
            observations: BTreeMap::new(),
        })
    }

    pub fn mark_incomplete(&mut self, error: AccountingError) {
        self.incomplete.get_or_insert(error);
    }

    // Private, synchronous bookkeeping only. Operations validate before mutation.
    // New attribution stops on error; snapshots still reconcile retained work.
    fn account<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> AccountingResult<T>,
    ) -> AccountingResult<T> {
        if self.incomplete.is_some() {
            return Err(AccountingError::Incomplete);
        }
        let result = increment(&mut self.event).and_then(|()| operation(self));
        if let Err(error) = &result {
            self.mark_incomplete(*error);
        }
        result
    }

    fn room(&self, len: usize, extra: usize) -> AccountingResult<()> {
        require(
            len.checked_add(extra).ok_or(AccountingError::Overflow)? <= self.capacity,
            AccountingError::Capacity,
        )
    }
    fn valid_point(&self, point: RoutePoint) -> AccountingResult<()> {
        require(
            point.request == self.request
                && point.position.valid()
                && point.layer < self.model.num_layers,
            AccountingError::InvalidIdentity,
        )
    }
    fn prediction(
        &self,
        id: PredictionIdentity,
        stage: PredictionStage,
    ) -> AccountingResult<&PredictionRecord> {
        let record = self
            .predictions
            .get(&id)
            .ok_or(AccountingError::InvalidIdentity)?;
        require(record.stage == stage, AccountingError::InvalidTransition)?;
        Ok(record)
    }

    /// Truth must represent a clean committed observation. This API has no
    /// physical identity, so observing a matching expert cannot award utility.
    pub fn observe_route(&mut self, point: RoutePoint, ids: &[u32]) -> AccountingResult<()> {
        self.account(|s| {
            require(!s.closed, AccountingError::Closed)?;
            s.valid_point(point)?;
            require(
                ids.len() == s.model.top_k
                    && ids.iter().all(|id| (*id as usize) < s.model.num_experts)
                    && ids.iter().copied().collect::<BTreeSet<_>>().len() == ids.len(),
                AccountingError::InvalidIdentity,
            )?;
            require(
                !s.observations.contains_key(&point),
                AccountingError::ReusedIdentity,
            )?;
            s.room(s.observations.len(), 1)?;
            s.observations.insert(
                point,
                ObservedRoute {
                    ids: ids.to_vec(),
                    event: s.event,
                },
            );
            Ok(())
        })
    }

    /// Freeze the complete ranked batch at once, before target truth. The only
    /// exception is explicitly retrospective replay, which can only be skipped.
    pub fn freeze_candidates(
        &mut self,
        source: RoutePoint,
        target: RoutePoint,
        candidates: &[u32],
        predictor_source: PredictorSource,
        predictor_revision: u64,
    ) -> AccountingResult<Vec<PredictionIdentity>> {
        self.account(|s| {
            require(!s.closed, AccountingError::Closed)?;
            s.valid_point(source)?;
            s.valid_point(target)?;
            require(
                adjacent(source, target, s.model)
                    && s.observations.contains_key(&source)
                    && predictor_revision > 0
                    && !candidates.is_empty()
                    && candidates.len() <= s.model.num_experts
                    && candidates
                        .iter()
                        .all(|id| (*id as usize) < s.model.num_experts)
                    && candidates.iter().copied().collect::<BTreeSet<_>>().len()
                        == candidates.len(),
                AccountingError::InvalidIdentity,
            )?;
            let causal = predictor_source != PredictorSource::CompletedPositionReplay;
            if causal {
                require(
                    !s.observations.contains_key(&target),
                    AccountingError::InvalidTransition,
                )?;
            }
            s.room(s.predictions.len(), candidates.len())?;
            let generation = s
                .generation
                .checked_add(1)
                .ok_or(AccountingError::Overflow)?;
            let final_sequence = s
                .candidate_sequence
                .checked_add(checked_len(candidates.len())?)
                .ok_or(AccountingError::Overflow)?;
            let mut sequence = s.candidate_sequence;
            let mut frozen = Vec::with_capacity(candidates.len());
            for &expert_local_id in candidates {
                increment(&mut sequence)?;
                frozen.push(PredictionIdentity {
                    request: s.request,
                    source_position: source.position,
                    source_layer: source.layer,
                    target_position: target.position,
                    target_layer: target.layer,
                    expert_local_id,
                    prediction_generation: generation,
                    candidate_sequence: sequence,
                    predictor_source,
                    predictor_revision,
                });
            }
            for &id in &frozen {
                s.predictions.insert(
                    id,
                    PredictionRecord {
                        stage: PredictionStage::Emitted,
                        admission: None,
                        acquisition: None,
                        install: None,
                        consumed_by: None,
                        causal,
                    },
                );
            }
            s.generation = generation;
            s.candidate_sequence = final_sequence;
            Ok(frozen)
        })
    }

    pub fn admit(
        &mut self,
        id: PredictionIdentity,
        disposition: AdmissionDisposition,
    ) -> AccountingResult<()> {
        self.account(|s| {
            let p = s.prediction(id, PredictionStage::Emitted)?;
            require(
                p.causal
                    || matches!(
                        disposition,
                        AdmissionDisposition::Skipped(SkipReason::Retrospective)
                    ),
                AccountingError::InvalidTransition,
            )?;
            let stage = match disposition {
                AdmissionDisposition::Accepted => PredictionStage::AdmissionDecision,
                AdmissionDisposition::Rejected => {
                    PredictionStage::Terminal(TerminalReason::Rejected)
                }
                AdmissionDisposition::Skipped(reason) => {
                    PredictionStage::Terminal(TerminalReason::Skipped(reason))
                }
            };
            let p = s.predictions.get_mut(&id).unwrap();
            p.admission = Some(disposition);
            p.stage = stage;
            Ok(())
        })
    }

    pub fn request_source(
        &mut self,
        id: PredictionIdentity,
        kind: SourceKind,
    ) -> AccountingResult<SourceAcquisitionIdentity> {
        self.account(|s| {
            s.prediction(id, PredictionStage::AdmissionDecision)?;
            s.room(s.sources.len(), 1)?;
            let sequence = s
                .acquisition_sequence
                .checked_add(1)
                .ok_or(AccountingError::Overflow)?;
            let work = SourceAcquisitionIdentity {
                owner_request: s.request,
                acquisition_sequence: sequence,
                layer: id.target_layer,
                expert_local_id: id.expert_local_id,
            };
            s.sources.insert(
                work,
                SourceRecord {
                    leader: id,
                    nominations: BTreeSet::from([id]),
                    kind,
                    state: SourceState::Requested,
                    reservation: None,
                },
            );
            let p = s.predictions.get_mut(&id).unwrap();
            p.stage = PredictionStage::SourceRequested;
            p.acquisition = Some(work);
            s.acquisition_sequence = sequence;
            Ok(work)
        })
    }

    pub fn join_source(
        &mut self,
        id: PredictionIdentity,
        work: SourceAcquisitionIdentity,
    ) -> AccountingResult<()> {
        self.account(|s| {
            s.prediction(id, PredictionStage::AdmissionDecision)?;
            let source = s
                .sources
                .get(&work)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                work.owner_request == id.request
                    && work.layer == id.target_layer
                    && work.expert_local_id == id.expert_local_id,
                AccountingError::InvalidIdentity,
            )?;
            require(
                source.reservation.is_none()
                    && matches!(
                        source.state,
                        SourceState::Requested | SourceState::Completed
                    ),
                AccountingError::InvalidTransition,
            )?;
            let stage = if source.state == SourceState::Requested {
                PredictionStage::SourceRequested
            } else {
                PredictionStage::SourceCompleted
            };
            s.sources.get_mut(&work).unwrap().nominations.insert(id);
            let p = s.predictions.get_mut(&id).unwrap();
            p.acquisition = Some(work);
            p.stage = stage;
            Ok(())
        })
    }

    pub fn complete_source(
        &mut self,
        work: SourceAcquisitionIdentity,
        outcome: SourceState,
    ) -> AccountingResult<()> {
        self.account(|s| {
            let source = s
                .sources
                .get(&work)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                source.state == SourceState::Requested && outcome != SourceState::Requested,
                AccountingError::InvalidTransition,
            )?;
            let stage = match outcome {
                SourceState::Completed => PredictionStage::SourceCompleted,
                SourceState::Failed => PredictionStage::Terminal(TerminalReason::SourceFailed),
                SourceState::Cancelled => {
                    PredictionStage::Terminal(TerminalReason::SourceCancelled)
                }
                SourceState::Requested => return Err(AccountingError::InvalidTransition),
            };
            for id in &source.nominations {
                let p = s.predictions.get_mut(id).unwrap();
                if p.stage == PredictionStage::SourceRequested {
                    p.stage = stage;
                }
            }
            s.sources.get_mut(&work).unwrap().state = outcome;
            Ok(())
        })
    }

    pub fn reserve(
        &mut self,
        owner: PredictionIdentity,
        location: InstallLocation,
        origin: InstallOrigin,
    ) -> AccountingResult<PhysicalInstallIdentity> {
        self.account(|s| {
            let p = s.prediction(owner, PredictionStage::SourceCompleted)?;
            let work = p.acquisition.ok_or(AccountingError::InvalidIdentity)?;
            let source = s
                .sources
                .get(&work)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                source.state == SourceState::Completed && source.reservation.is_none(),
                AccountingError::InvalidTransition,
            )?;
            require(
                location.executor_namespace > 0
                    && location.model_namespace > 0
                    && location.logical_generation > 0
                    && location.slot_epoch > 0,
                AccountingError::InvalidIdentity,
            )?;
            s.room(s.reservations.len(), 1)?;
            let sequence = s
                .ticket_sequence
                .checked_add(1)
                .ok_or(AccountingError::Overflow)?;
            let install = PhysicalInstallIdentity {
                runtime_namespace: s.request.runtime_namespace,
                executor_namespace: location.executor_namespace,
                model_namespace: location.model_namespace,
                layer: work.layer,
                expert_local_id: work.expert_local_id,
                logical_generation: location.logical_generation,
                bank: location.bank,
                slot: location.slot,
                slot_epoch: location.slot_epoch,
                reservation: ReservationIdentity {
                    acquisition: work,
                    ticket_sequence: sequence,
                },
            };
            // A new ticket cannot disguise reuse of a slot epoch or generation.
            require(
                !s.reservations.keys().any(|old| {
                    old.slot_key() == install.slot_key() && old.slot_epoch >= install.slot_epoch
                }),
                AccountingError::ReusedIdentity,
            )?;
            require(
                !s.reservations.keys().any(|old| {
                    old.runtime_namespace == install.runtime_namespace
                        && old.executor_namespace == install.executor_namespace
                        && old.model_namespace == install.model_namespace
                        && old.layer == install.layer
                        && old.expert_local_id == install.expert_local_id
                        && (old.logical_generation > install.logical_generation
                            || (old.logical_generation == install.logical_generation
                                && origin != InstallOrigin::PredictorV2Sidecar))
                }),
                AccountingError::ReusedIdentity,
            )?;
            let followers: BTreeSet<_> = source
                .nominations
                .iter()
                .copied()
                .filter(|id| {
                    *id != owner && s.predictions[id].stage == PredictionStage::SourceCompleted
                })
                .collect();
            for id in std::iter::once(&owner).chain(followers.iter()) {
                let p = s.predictions.get_mut(id).unwrap();
                p.stage = PredictionStage::ResidencyReserved;
                p.install = Some(install);
            }
            s.reservations.insert(
                install,
                ReservationRecord {
                    owner,
                    followers,
                    state: ReservationState::Live,
                    origin,
                },
            );
            s.sources.get_mut(&work).unwrap().reservation = Some(install);
            s.ticket_sequence = sequence;
            Ok(install)
        })
    }

    pub fn finish_reservation(
        &mut self,
        install: PhysicalInstallIdentity,
        failure: Option<TerminalReason>,
    ) -> AccountingResult<()> {
        self.account(|s| {
            let reservation = s
                .reservations
                .get(&install)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                reservation.state == ReservationState::Live,
                AccountingError::InvalidTransition,
            )?;
            require(
                failure.is_none()
                    || matches!(
                        failure,
                        Some(TerminalReason::InstallFailed | TerminalReason::ReservationAborted)
                    ),
                AccountingError::InvalidTransition,
            )?;
            if failure.is_none() {
                s.room(s.installs.len(), 1)?;
                require(
                    !s.current.contains_key(&install.slot_key()),
                    AccountingError::InvalidTransition,
                )?;
                require(
                    !s.current.values().any(|old| {
                        old.runtime_namespace == install.runtime_namespace
                            && old.executor_namespace == install.executor_namespace
                            && old.model_namespace == install.model_namespace
                            && old.layer == install.layer
                            && old.expert_local_id == install.expert_local_id
                    }),
                    AccountingError::InvalidTransition,
                )?;
            }
            for id in std::iter::once(&reservation.owner).chain(reservation.followers.iter()) {
                let p = s.predictions.get_mut(id).unwrap();
                if p.stage == PredictionStage::ResidencyReserved {
                    p.stage = failure
                        .map(PredictionStage::Terminal)
                        .unwrap_or(PredictionStage::Installed);
                }
            }
            if failure.is_none() {
                s.installs.insert(
                    install,
                    InstallRecord {
                        state: InstallState::Installed,
                        origin: reservation.origin,
                        available_event: None,
                        credited_demand: None,
                    },
                );
                s.current.insert(install.slot_key(), install);
            }
            s.reservations.get_mut(&install).unwrap().state = if failure.is_some() {
                ReservationState::Aborted
            } else {
                ReservationState::Committed
            };
            Ok(())
        })
    }

    pub fn make_available(&mut self, install: PhysicalInstallIdentity) -> AccountingResult<()> {
        self.account(|s| {
            let record = s
                .installs
                .get(&install)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                record.state == InstallState::Installed
                    && s.current.get(&install.slot_key()) == Some(&install),
                AccountingError::InvalidTransition,
            )?;
            let record = s.installs.get_mut(&install).unwrap();
            record.state = InstallState::Available;
            record.available_event = Some(s.event);
            for p in s.predictions.values_mut() {
                if p.install == Some(install) && p.stage == PredictionStage::Installed {
                    p.stage = PredictionStage::Available;
                }
            }
            Ok(())
        })
    }

    pub fn consume(
        &mut self,
        demand: DemandIdentity,
        install: PhysicalInstallIdentity,
        status: DemandStatus,
    ) -> AccountingResult<()> {
        self.account(|s| {
            let record = s
                .installs
                .get(&install)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                status == DemandStatus::CleanCommitted
                    && demand.request == s.request
                    && demand.layer == install.layer
                    && demand.expert_local_id == install.expert_local_id,
                AccountingError::InvalidIdentity,
            )?;
            let truth = s
                .observations
                .get(&RoutePoint {
                    request: demand.request,
                    position: demand.target_position,
                    layer: demand.layer,
                })
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                record.state == InstallState::Available
                    && matches!(
                        record.origin,
                        InstallOrigin::SpeculativeFixture | InstallOrigin::PredictorV2Sidecar
                    )
                    && record.credited_demand.is_none()
                    && s.current.get(&install.slot_key()) == Some(&install)
                    && record
                        .available_event
                        .is_some_and(|event| event < truth.event)
                    && truth.ids.contains(&demand.expert_local_id),
                AccountingError::InvalidTransition,
            )?;
            let matches: Vec<_> = s
                .predictions
                .iter()
                .filter(|(id, p)| {
                    p.stage == PredictionStage::Available
                        && p.install == Some(install)
                        && p.causal
                        && id.target()
                            == RoutePoint {
                                request: demand.request,
                                position: demand.target_position,
                                layer: demand.layer,
                            }
                        && id.expert_local_id == demand.expert_local_id
                })
                .map(|(id, _)| *id)
                .collect();
            require(!matches.is_empty(), AccountingError::InvalidIdentity)?;
            let sequence = s
                .demand_sequence
                .checked_add(1)
                .ok_or(AccountingError::Overflow)?;
            require(
                demand.demand_sequence == sequence
                    && !s.demands.contains_key(&demand)
                    && !s.demands.keys().any(|old| {
                        old.request == demand.request
                            && old.target_position == demand.target_position
                            && old.layer == demand.layer
                            && old.expert_local_id == demand.expert_local_id
                    }),
                AccountingError::ReusedIdentity,
            )?;
            s.room(s.demands.len(), 1)?;
            for id in matches {
                let p = s.predictions.get_mut(&id).unwrap();
                p.stage = PredictionStage::Terminal(TerminalReason::ConsumedByMatchingRoute);
                p.consumed_by = Some(demand);
            }
            s.installs.get_mut(&install).unwrap().credited_demand = Some(demand);
            s.demands.insert(demand, install);
            s.demand_sequence = sequence;
            Ok(())
        })
    }

    pub fn evict_unused(&mut self, install: PhysicalInstallIdentity) -> AccountingResult<()> {
        self.account(|s| {
            let record = s
                .installs
                .get(&install)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                matches!(
                    record.state,
                    InstallState::Installed | InstallState::Available
                ) && record.credited_demand.is_none()
                    && s.current.get(&install.slot_key()) == Some(&install),
                AccountingError::InvalidTransition,
            )?;
            for p in s.predictions.values_mut() {
                if p.install == Some(install)
                    && matches!(
                        p.stage,
                        PredictionStage::Installed | PredictionStage::Available
                    )
                {
                    p.stage = PredictionStage::Terminal(TerminalReason::EvictedUnused);
                }
            }
            s.installs.get_mut(&install).unwrap().state = InstallState::Evicted;
            s.current.remove(&install.slot_key());
            Ok(())
        })
    }

    fn terminate_record(p: &mut PredictionRecord, reason: TerminalReason) {
        if p.admission.is_none() {
            p.admission = Some(AdmissionDisposition::Skipped(match reason {
                TerminalReason::Cancelled => SkipReason::Cancelled,
                TerminalReason::Superseded => SkipReason::Superseded,
                _ => SkipReason::RequestEnd,
            }));
        }
        p.stage = PredictionStage::Terminal(reason);
    }

    pub fn terminate(
        &mut self,
        id: PredictionIdentity,
        reason: TerminalReason,
    ) -> AccountingResult<()> {
        self.account(|s| {
            require(
                matches!(
                    reason,
                    TerminalReason::Cancelled
                        | TerminalReason::Superseded
                        | TerminalReason::RequestEnded
                ),
                AccountingError::InvalidTransition,
            )?;
            let p = s
                .predictions
                .get_mut(&id)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                !matches!(p.stage, PredictionStage::Terminal(_)),
                AccountingError::InvalidTransition,
            )?;
            Self::terminate_record(p, reason);
            Ok(())
        })
    }

    /// Ends nominations only. Acquisitions/reservations/install ownership remain
    /// present, and their late synthetic completions can still be accounted.
    pub fn end_request(&mut self, cancelled: bool) -> AccountingResult<()> {
        self.account(|s| {
            require(!s.closed, AccountingError::Closed)?;
            for p in s.predictions.values_mut() {
                if !matches!(p.stage, PredictionStage::Terminal(_)) {
                    Self::terminate_record(
                        p,
                        if cancelled {
                            TerminalReason::Cancelled
                        } else {
                            TerminalReason::RequestEnded
                        },
                    );
                }
            }
            s.closed = true;
            Ok(())
        })
    }

    pub fn reconcile(&self) -> AccountingResult<ReconciliationSnapshot> {
        let mut out = ReconciliationSnapshot {
            incomplete: self.incomplete,
            emitted: checked_len(self.predictions.len())?,
            source_leaders: checked_len(self.sources.len())?,
            reservations: checked_len(self.reservations.len())?,
            install_owners: checked_len(self.installs.len())?,
            ..Default::default()
        };
        let mut terminal_categories = BTreeMap::new();
        for (id, p) in &self.predictions {
            require(id.request == self.request, AccountingError::InvalidIdentity)?;
            match p.stage {
                PredictionStage::Terminal(reason) => {
                    increment(&mut out.terminal_predictions)?;
                    increment(terminal_categories.entry(reason).or_default())?;
                    require(p.admission.is_some(), AccountingError::InvalidTransition)?;
                    require(
                        (reason == TerminalReason::ConsumedByMatchingRoute)
                            == p.consumed_by.is_some(),
                        AccountingError::InvalidTransition,
                    )?;
                    match reason {
                        TerminalReason::Rejected => require(
                            p.admission == Some(AdmissionDisposition::Rejected),
                            AccountingError::InvalidTransition,
                        )?,
                        TerminalReason::Skipped(why) => require(
                            p.admission == Some(AdmissionDisposition::Skipped(why)),
                            AccountingError::InvalidTransition,
                        )?,
                        _ => {}
                    }
                }
                _ => {
                    increment(&mut out.live_predictions)?;
                    require(p.consumed_by.is_none(), AccountingError::InvalidTransition)?;
                    require(
                        (p.stage == PredictionStage::Emitted) == p.admission.is_none(),
                        AccountingError::InvalidTransition,
                    )?;
                    if p.stage != PredictionStage::Emitted {
                        require(
                            p.admission == Some(AdmissionDisposition::Accepted),
                            AccountingError::InvalidTransition,
                        )?;
                    }
                }
            }
            match p.admission {
                None => increment(&mut out.admission_pending)?,
                Some(AdmissionDisposition::Accepted) => increment(&mut out.accepted)?,
                Some(AdmissionDisposition::Rejected) => increment(&mut out.rejected)?,
                Some(AdmissionDisposition::Skipped(_)) => increment(&mut out.skipped)?,
            }
            if p.acquisition.is_some() || p.install.is_some() {
                require(
                    p.admission == Some(AdmissionDisposition::Accepted) && p.causal,
                    AccountingError::InvalidTransition,
                )?;
            }
            if let Some(work) = p.acquisition {
                require(
                    self.sources
                        .get(&work)
                        .is_some_and(|s| s.nominations.contains(id)),
                    AccountingError::InvalidIdentity,
                )?;
            }
            if let Some(install) = p.install {
                let r = self
                    .reservations
                    .get(&install)
                    .ok_or(AccountingError::InvalidIdentity)?;
                require(
                    (r.owner == *id || r.followers.contains(id))
                        && p.acquisition == Some(install.reservation.acquisition),
                    AccountingError::InvalidIdentity,
                )?;
            }
            if let Some(demand) = p.consumed_by {
                require(
                    self.demands.get(&demand) == p.install.as_ref()
                        && demand.request == id.request
                        && demand.target_position == id.target_position
                        && demand.layer == id.target_layer
                        && demand.expert_local_id == id.expert_local_id,
                    AccountingError::InvalidIdentity,
                )?;
            }
        }
        for (id, source) in &self.sources {
            require(
                id.owner_request == self.request && source.nominations.contains(&source.leader),
                AccountingError::InvalidIdentity,
            )?;
            out.source_followers = out
                .source_followers
                .checked_add(
                    checked_len(source.nominations.len())?
                        .checked_sub(1)
                        .ok_or(AccountingError::Overflow)?,
                )
                .ok_or(AccountingError::Overflow)?;
            match source.state {
                SourceState::Requested => increment(&mut out.source_live)?,
                SourceState::Completed => increment(&mut out.source_completed)?,
                SourceState::Failed => increment(&mut out.source_failed)?,
                SourceState::Cancelled => increment(&mut out.source_cancelled)?,
            }
            for nomination in &source.nominations {
                let p = self
                    .predictions
                    .get(nomination)
                    .ok_or(AccountingError::InvalidIdentity)?;
                require(
                    p.acquisition == Some(*id)
                        && nomination.target_layer == id.layer
                        && nomination.expert_local_id == id.expert_local_id,
                    AccountingError::InvalidIdentity,
                )?;
            }
            if let Some(install) = source.reservation {
                require(
                    source.state == SourceState::Completed
                        && install.reservation.acquisition == *id
                        && self.reservations.contains_key(&install),
                    AccountingError::InvalidIdentity,
                )?;
            }
        }
        for (id, reservation) in &self.reservations {
            require(
                !reservation.followers.contains(&reservation.owner)
                    && self
                        .sources
                        .get(&id.reservation.acquisition)
                        .is_some_and(|s| s.reservation == Some(*id)),
                AccountingError::InvalidIdentity,
            )?;
            for nomination in
                std::iter::once(&reservation.owner).chain(reservation.followers.iter())
            {
                require(
                    self.predictions
                        .get(nomination)
                        .is_some_and(|p| p.install == Some(*id)),
                    AccountingError::InvalidIdentity,
                )?;
            }
            match reservation.state {
                ReservationState::Live => {
                    increment(&mut out.reservations_live)?;
                    require(
                        !self.installs.contains_key(id),
                        AccountingError::InvalidTransition,
                    )?;
                }
                ReservationState::Aborted => {
                    increment(&mut out.reservations_aborted)?;
                    require(
                        !self.installs.contains_key(id),
                        AccountingError::InvalidTransition,
                    )?;
                }
                ReservationState::Committed => {
                    increment(&mut out.reservations_committed)?;
                    require(
                        self.installs.contains_key(id),
                        AccountingError::InvalidIdentity,
                    )?;
                    out.install_followers = out
                        .install_followers
                        .checked_add(checked_len(reservation.followers.len())?)
                        .ok_or(AccountingError::Overflow)?;
                }
            }
        }
        for (id, install) in &self.installs {
            require(
                self.reservations.get(id).is_some_and(|r| {
                    r.state == ReservationState::Committed && r.origin == install.origin
                }),
                AccountingError::InvalidIdentity,
            )?;
            require(
                (self.current.get(&id.slot_key()) == Some(id))
                    == (install.state != InstallState::Evicted),
                AccountingError::InvalidIdentity,
            )?;
            if install.state == InstallState::Available {
                increment(&mut out.available_installs)?;
            }
            if let Some(demand) = install.credited_demand {
                require(
                    self.demands.get(&demand) == Some(id),
                    AccountingError::InvalidIdentity,
                )?;
                increment(&mut out.direct_matching_demand_credits)?;
            }
        }
        for (slot, id) in &self.current {
            require(
                *slot == id.slot_key()
                    && self
                        .installs
                        .get(id)
                        .is_some_and(|i| i.state != InstallState::Evicted),
                AccountingError::InvalidIdentity,
            )?;
        }
        for (demand, id) in &self.demands {
            require(
                self.installs
                    .get(id)
                    .is_some_and(|i| i.credited_demand == Some(*demand)),
                AccountingError::InvalidIdentity,
            )?;
        }
        let sum = |parts: &[u64]| {
            parts.iter().try_fold(0u64, |n, part| {
                n.checked_add(*part).ok_or(AccountingError::Overflow)
            })
        };
        require(
            out.emitted == sum(&[out.live_predictions, out.terminal_predictions])?
                && out.emitted
                    == sum(&[
                        out.admission_pending,
                        out.accepted,
                        out.rejected,
                        out.skipped,
                    ])?
                && out.terminal_predictions
                    == sum(&terminal_categories.values().copied().collect::<Vec<_>>())?
                && out.source_leaders
                    == sum(&[
                        out.source_live,
                        out.source_completed,
                        out.source_failed,
                        out.source_cancelled,
                    ])?
                && out.reservations
                    == sum(&[
                        out.reservations_live,
                        out.reservations_committed,
                        out.reservations_aborted,
                    ])?
                && out.install_owners == out.reservations_committed
                && out.direct_matching_demand_credits == checked_len(self.demands.len())?,
            AccountingError::InvalidTransition,
        )?;
        out.terminal_categories = terminal_categories.into_iter().collect();
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ObservationConfig {
    pub phase: RequestPhase,
    pub phase_run_index: u64,
    /// Explicit prompt extent; sampling the final prompt token is still prompt.
    pub prompt_length: usize,
    pub capacity_per_collection: usize,
}

#[derive(Clone, Debug)]
struct Predecessor {
    point: RoutePoint,
    ids: Vec<u32>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct TransitionKey {
    source_layer: usize,
    target_layer: usize,
    source_kind: PositionKind,
    target_kind: PositionKind,
    source_expert: u32,
    target_expert: u32,
}

/// No Clone implementation: history is owned by one native request. There is no
/// shared predecessor ring or shared learned table. Completed-position replay
/// trains descriptive transitions only. P1E has a separate explicit opt-in for
/// temporal opportunity records; neither observer schedules work.
#[derive(Debug)]
pub(crate) struct RequestObserver {
    ledger: LifecycleLedger,
    predecessor: Option<Predecessor>,
    transition_counts: BTreeMap<TransitionKey, u64>,
    completed_positions: u64,
    temporal: Option<Box<p1e::Temporal>>,
}

impl RequestObserver {
    pub fn new(
        request: RequestIdentity,
        model: ModelMetadata,
        capacity: usize,
    ) -> AccountingResult<Self> {
        Ok(Self {
            ledger: LifecycleLedger::new(request, model, capacity)?,
            predecessor: None,
            transition_counts: BTreeMap::new(),
            completed_positions: 0,
            temporal: None,
        })
    }
    pub fn request_identity(&self) -> RequestIdentity {
        self.ledger.request
    }
    pub fn snapshot(&self) -> AccountingResult<ReconciliationSnapshot> {
        self.ledger.reconcile()
    }
    pub fn completed_positions(&self) -> u64 {
        self.completed_positions
    }
    pub fn mark_incomplete(&mut self, error: AccountingError) {
        if let Some(temporal) = self.temporal.as_deref_mut() {
            temporal.mark_incomplete(p1e::Error::Incomplete);
        }
        self.ledger.mark_incomplete(error);
        self.predecessor = None;
    }
    pub fn finish(&mut self, cancelled: bool) {
        if let Some(temporal) = self.temporal.as_deref_mut() {
            temporal.finish();
        }
        let _ = self.ledger.end_request(cancelled);
        self.predecessor = None;
    }

    pub fn enable_temporal(&mut self, namespace: p1e::Namespace) -> Result<(), p1e::Error> {
        if self.completed_positions != 0 || self.temporal.is_some() || self.ledger.closed {
            return Err(p1e::Error::Identity);
        }
        self.temporal = Some(Box::new(p1e::Temporal::new(
            self.ledger.request,
            self.ledger.model,
            namespace,
            self.ledger.capacity,
        )?));
        Ok(())
    }
    pub fn temporal(&self) -> Option<&p1e::Temporal> {
        self.temporal.as_deref()
    }
    pub fn temporal_mut(&mut self) -> Option<&mut p1e::Temporal> {
        self.temporal.as_deref_mut()
    }

    pub(crate) fn p1j_ready(&self) -> bool {
        self.ledger.incomplete.is_none()
            && !self.ledger.closed
            && self.temporal().is_some_and(|t| t.active())
    }

    /// Import the already-frozen P1E identity; this never scores or chooses a candidate.
    pub(crate) fn p1j_acquired(
        &mut self,
        id: P1jIdentity,
    ) -> AccountingResult<PhysicalInstallIdentity> {
        let freeze = self.temporal().and_then(|t| {
            t.pending_freeze(id.candidate.target_position.absolute_position as usize)
        });
        let valid = self.p1j_ready()
            && freeze.is_some_and(|f| {
                f.candidate == id.candidate
                    && f.current == Some(false)
                    && f.incomplete.is_none()
                    && f.source.logical_materialized
                    && f.source.logical_generation == Some(id.logical_generation)
            });
        let prediction = id.prediction();
        self.ledger.account(|s| {
            require(
                valid && id.epoch != 0 && id.writer_sequence != 0,
                AccountingError::InvalidIdentity,
            )?;
            require(
                prediction.request == s.request
                    && id.candidate.model == s.model
                    && prediction.source_layer == p1e::LAYER
                    && prediction.target_layer == p1e::LAYER
                    && prediction
                        .target_position
                        .follows(prediction.source_position)
                    && s.observations.contains_key(&RoutePoint {
                        request: s.request,
                        position: prediction.source_position,
                        layer: p1e::LAYER,
                    })
                    && !s.observations.contains_key(&prediction.target())
                    && !s.predictions.contains_key(&prediction),
                AccountingError::InvalidIdentity,
            )?;
            s.room(s.predictions.len(), 1)?;
            s.predictions.insert(
                prediction,
                PredictionRecord {
                    stage: PredictionStage::Emitted,
                    admission: None,
                    acquisition: None,
                    install: None,
                    consumed_by: None,
                    causal: true,
                },
            );
            Ok(())
        })?;
        self.ledger
            .admit(prediction, AdmissionDisposition::Accepted)?;
        let source = self
            .ledger
            .request_source(prediction, SourceKind::HostBackedLogical)?;
        self.ledger
            .complete_source(source, SourceState::Completed)?;
        self.ledger
            .reserve(prediction, id.location(), InstallOrigin::PredictorV2Sidecar)
    }

    pub(crate) fn p1j_published(
        &mut self,
        install: PhysicalInstallIdentity,
    ) -> AccountingResult<()> {
        self.ledger.finish_reservation(install, None)?;
        self.ledger.make_available(install)
    }

    pub(crate) fn p1j_finish(
        &mut self,
        id: P1jIdentity,
        install: PhysicalInstallIdentity,
        terminal: P1jTerminal,
    ) -> AccountingResult<()> {
        if terminal == P1jTerminal::UsedMatching {
            let sequence = self
                .ledger
                .demand_sequence
                .checked_add(1)
                .ok_or(AccountingError::Overflow)?;
            self.ledger.consume(
                DemandIdentity {
                    request: id.candidate.request,
                    target_position: id.candidate.target_position,
                    layer: p1e::LAYER,
                    expert_local_id: id.candidate.expert,
                    demand_sequence: sequence,
                },
                install,
                DemandStatus::CleanCommitted,
            )?;
        }
        self.ledger.account(|s| {
            let reservation = s
                .reservations
                .get_mut(&install)
                .ok_or(AccountingError::InvalidIdentity)?;
            require(
                reservation.origin == InstallOrigin::PredictorV2Sidecar,
                AccountingError::InvalidIdentity,
            )?;
            if reservation.state == ReservationState::Live {
                reservation.state = ReservationState::Aborted;
            }
            if let Some(record) = s.installs.get_mut(&install) {
                record.state = InstallState::Evicted;
                if s.current.get(&install.slot_key()) == Some(&install) {
                    s.current.remove(&install.slot_key());
                }
            }
            let p = s
                .predictions
                .get_mut(&id.prediction())
                .ok_or(AccountingError::InvalidIdentity)?;
            if !matches!(p.stage, PredictionStage::Terminal(_)) {
                Self::p1j_terminate(p, terminal);
            }
            Ok(())
        })
    }

    fn p1j_terminate(p: &mut PredictionRecord, terminal: P1jTerminal) {
        LifecycleLedger::terminate_record(
            p,
            match terminal {
                P1jTerminal::UsedMatching => TerminalReason::ConsumedByMatchingRoute,
                P1jTerminal::Unused => TerminalReason::EvictedUnused,
                P1jTerminal::OrdinarySuperseded | P1jTerminal::StaleLogicalGeneration => {
                    TerminalReason::Superseded
                }
                P1jTerminal::Cancelled => TerminalReason::Cancelled,
                P1jTerminal::RequestEnded => TerminalReason::RequestEnded,
                P1jTerminal::WriteFailure => TerminalReason::InstallFailed,
            },
        );
    }
    pub fn prepare_temporal(
        &mut self,
        position: PositionIdentity,
        target: PositionIdentity,
    ) -> Option<p1e::Candidate> {
        let temporal = self.temporal.as_deref_mut()?;
        if self.ledger.incomplete.is_some()
            || self.ledger.closed
            || position.absolute_position.checked_add(1) != Some(self.completed_positions)
        {
            temporal.mark_incomplete(p1e::Error::Incomplete);
            return None;
        }
        let Some(previous) = self.predecessor.as_ref().filter(|p| {
            p.point.layer == p1e::LAYER
                && p.point.position == position
                && p.point.request == self.ledger.request
        }) else {
            temporal.mark_incomplete(p1e::Error::Identity);
            return None;
        };
        temporal.committed(position, target, &previous.ids)
    }

    fn observe_layer(&mut self, point: RoutePoint, ids: &[u32]) -> AccountingResult<()> {
        let mut updates = Vec::new();
        if let Some(previous) = &self.predecessor {
            require(
                adjacent(previous.point, point, self.ledger.model),
                AccountingError::InvalidIdentity,
            )?;
            for &source_expert in &previous.ids {
                for &target_expert in ids {
                    let key = TransitionKey {
                        source_layer: previous.point.layer,
                        target_layer: point.layer,
                        source_kind: previous.point.position.position_kind,
                        target_kind: point.position.position_kind,
                        source_expert,
                        target_expert,
                    };
                    let count = self
                        .transition_counts
                        .get(&key)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(1)
                        .ok_or(AccountingError::Overflow)?;
                    updates.push((key, count));
                }
            }
        } else {
            require(
                point.request == self.ledger.request
                    && point.layer == 0
                    && point.position.absolute_position == 0
                    && point.position.position_kind == PositionKind::Prompt,
                AccountingError::InvalidIdentity,
            )?;
        }
        let extra = updates
            .iter()
            .filter(|(key, _)| !self.transition_counts.contains_key(key))
            .count();
        self.ledger.room(self.transition_counts.len(), extra)?;
        self.ledger.observe_route(point, ids)?;
        for (key, count) in updates {
            self.transition_counts.insert(key, count);
        }
        self.predecessor = Some(Predecessor {
            point,
            ids: ids.to_vec(),
        });
        Ok(())
    }

    /// Infallible to the inference caller. No source/install/consumption methods
    /// are invoked here. Routes are borrowed read-only and copied into CPU state.
    pub fn observe_completed_position(
        &mut self,
        request: RequestIdentity,
        position: PositionIdentity,
        model: ModelMetadata,
        routes: &[Vec<u32>],
    ) {
        if let Some(temporal) = self.temporal.as_deref_mut() {
            temporal.note_p0_truth(position);
        }
        let result = (|| {
            require(
                self.ledger.incomplete.is_none(),
                AccountingError::Incomplete,
            )?;
            require(!self.ledger.closed, AccountingError::Closed)?;
            require(
                request == self.ledger.request
                    && model == self.ledger.model
                    && position.valid()
                    && position.absolute_position == self.completed_positions
                    && routes.len() == model.num_layers,
                AccountingError::InvalidIdentity,
            )?;
            for ids in routes {
                require(
                    ids.len() == model.top_k
                        && ids.iter().all(|id| (*id as usize) < model.num_experts)
                        && ids.iter().copied().collect::<BTreeSet<_>>().len() == ids.len(),
                    AccountingError::InvalidIdentity,
                )?;
            }
            self.ledger
                .room(self.ledger.observations.len(), routes.len())?;
            let completed = self
                .completed_positions
                .checked_add(1)
                .ok_or(AccountingError::Overflow)?;
            for (layer, ids) in routes.iter().enumerate() {
                self.observe_layer(
                    RoutePoint {
                        request,
                        position,
                        layer,
                    },
                    ids,
                )?;
            }
            self.completed_positions = completed;
            Ok(())
        })();
        if let Err(error) = result {
            self.mark_incomplete(error);
        }
    }
}

/// Frozen P1E geometry and signal. This module accepts values only, never an
/// executor, source, cache, callback or task. Its shadow describes ordinary work.
pub(crate) mod p1e {
    use super::{ModelMetadata, PositionIdentity, PositionKind, RequestIdentity};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};

    pub const LAYER: usize = 47;
    pub const CAPACITY: usize = 8;
    pub const EXPERTS: usize = 128;
    pub const REVISION: u64 = 1;
    const MAX_REPORT_RECORDS: usize = 4096;
    const MAX_RECOVERY_EVENTS: usize = 512;

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum Error {
        Identity,
        Capacity,
        Overflow,
        PhysicalEvidence,
        ShadowDisagreement,
        Chronology,
        MissingDeadline,
        Incomplete,
        Closed,
    }
    type Result<T> = std::result::Result<T, Error>;
    fn check(ok: bool, error: Error) -> Result<()> {
        if ok {
            Ok(())
        } else {
            Err(error)
        }
    }
    fn increment(n: u64) -> Result<u64> {
        n.checked_add(1).ok_or(Error::Overflow)
    }
    fn selected(ids: &[u32]) -> Result<[u32; CAPACITY]> {
        check(
            ids.len() == CAPACITY
                && ids.iter().all(|&e| (e as usize) < EXPERTS)
                && ids.iter().copied().collect::<BTreeSet<_>>().len() == CAPACITY,
            Error::Identity,
        )?;
        let mut out = [0; CAPACITY];
        out.copy_from_slice(ids);
        out.sort_unstable();
        Ok(out)
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Namespace {
        pub runtime: u64,
        pub context: u64,
        /// Process-local scalar address of the manager-owned arena. The manager
        /// retains that arena for this namespace's lifetime; this is not a handle.
        pub arena: usize,
        pub layer: usize,
        pub capacity: usize,
    }
    impl Namespace {
        fn valid(self) -> bool {
            self.runtime != 0
                && self.context != 0
                && self.arena != 0
                && self.layer == LAYER
                && self.capacity == CAPACITY
        }
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Resident {
        pub expert: u32,
        pub generation: u64,
        pub bank: u32,
        pub slot: u32,
        pub epoch: u32,
    }
    impl Resident {
        fn valid(self) -> bool {
            (self.expert as usize) < EXPERTS
                && self.generation != 0
                && self.bank < 4
                && (self.slot as usize) < CAPACITY
                && self.epoch != 0
        }
    }
    /// LRU to MRU. Every entry was checked against the authoritative arena
    /// owner before being copied. There is no reservation or payload here.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct PhysicalSnapshot {
        pub namespace: Namespace,
        pub residents: [Option<Resident>; CAPACITY],
    }
    impl PhysicalSnapshot {
        pub fn validate(&self) -> Result<()> {
            check(self.namespace.valid(), Error::PhysicalEvidence)?;
            let mut ids = BTreeSet::new();
            let mut slots = BTreeSet::new();
            let mut gap = false;
            for entry in self.residents {
                match entry {
                    Some(r) => check(
                        !gap && r.valid() && ids.insert(r.expert) && slots.insert((r.bank, r.slot)),
                        Error::PhysicalEvidence,
                    )?,
                    None => gap = true,
                }
            }
            Ok(())
        }
        pub fn current(&self, namespace: Namespace, expert: u32) -> Result<bool> {
            self.validate()?;
            check(
                self.namespace == namespace && (expert as usize) < EXPERTS,
                Error::PhysicalEvidence,
            )?;
            Ok(self.residents.iter().flatten().any(|r| r.expert == expert))
        }
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct PhysicalEvidence {
        pub snapshot: PhysicalSnapshot,
        pub event_cutoff: u64,
        pub committed_installs: u64,
        pub physical_victims: u64,
    }

    /// Persistent runtime-owned CPU description, independently replaying the
    /// capacity-8 policy. Request history resets must not reset this object.
    #[derive(Debug)]
    pub struct Shadow {
        namespace: Namespace,
        residents: Vec<Resident>,
        victims: Vec<u32>,
        installs: Vec<u32>,
        event: u64,
        committed_installs: u64,
        physical_victims: u64,
        incomplete: Option<Error>,
    }
    impl Shadow {
        pub fn new(initial: PhysicalSnapshot) -> Result<Self> {
            initial.validate()?;
            Ok(Self {
                namespace: initial.namespace,
                residents: initial.residents.into_iter().flatten().collect(),
                victims: Vec::new(),
                installs: Vec::new(),
                event: 0,
                committed_installs: 0,
                physical_victims: 0,
                incomplete: None,
            })
        }
        pub fn namespace(&self) -> Namespace {
            self.namespace
        }
        pub fn mark_incomplete(&mut self, e: Error) {
            self.incomplete.get_or_insert(e);
        }
        fn apply(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) {
            if self.incomplete.is_none() {
                if let Err(e) = f(self) {
                    self.mark_incomplete(e);
                }
            }
        }
        fn ready(&self) -> Result<()> {
            self.incomplete.map_or(Ok(()), Err)
        }
        fn values(&self) -> PhysicalSnapshot {
            let mut residents = [None; CAPACITY];
            for (dst, src) in residents.iter_mut().zip(&self.residents) {
                *dst = Some(*src);
            }
            PhysicalSnapshot {
                namespace: self.namespace,
                residents,
            }
        }
        pub fn evidence(&mut self, actual: PhysicalSnapshot) -> Result<PhysicalEvidence> {
            let result = (|| {
                self.ready()?;
                actual.validate()?;
                check(
                    self.victims.is_empty() && self.installs.is_empty() && self.values() == actual,
                    Error::ShadowDisagreement,
                )?;
                Ok(PhysicalEvidence {
                    snapshot: actual,
                    event_cutoff: self.event,
                    committed_installs: self.committed_installs,
                    physical_victims: self.physical_victims,
                })
            })();
            if let Err(e) = result {
                self.mark_incomplete(e);
            }
            result
        }
        /// Actual demand order, not sorted route order. Probe-only lookups never
        /// call this. Full protection is computed before choosing any victim.
        pub fn demand(&mut self, ids: &[u32]) {
            self.demand_with_sidecar(ids, None);
        }
        /// The shadow still describes only ordinary capacity. The exact
        /// published sidecar satisfies one selected id without a shadow install.
        pub fn demand_with_sidecar(&mut self, ids: &[u32], sidecar: Option<Resident>) {
            self.apply(|s| {
                selected(ids)?;
                if let Some(r) = sidecar {
                    check(
                        r.valid()
                            && r.bank == 1
                            && r.slot == 0
                            && !s
                                .residents
                                .iter()
                                .any(|ordinary| ordinary.expert == r.expert),
                        Error::PhysicalEvidence,
                    )?;
                }
                check(
                    s.victims.is_empty() && s.installs.is_empty(),
                    Error::ShadowDisagreement,
                )?;
                let event = increment(s.event)?;
                let mut residents = s.residents.clone();
                let mut installs = Vec::new();
                for &id in ids {
                    if sidecar.is_some_and(|r| r.expert == id) {
                        continue;
                    }
                    if let Some(index) = residents.iter().position(|r| r.expert == id) {
                        let r = residents.remove(index);
                        residents.push(r);
                    } else {
                        installs.push(id);
                    }
                }
                let mut preview = residents.clone();
                let mut victims = Vec::new();
                while preview
                    .len()
                    .checked_add(installs.len())
                    .ok_or(Error::Overflow)?
                    > CAPACITY
                {
                    let index = preview
                        .iter()
                        .position(|r| !ids.contains(&r.expert))
                        .ok_or(Error::ShadowDisagreement)?;
                    victims.push(preview.remove(index).expert);
                }
                s.residents = residents;
                s.victims = victims;
                s.installs = installs;
                s.event = event;
                Ok(())
            });
        }
        pub fn victim(&mut self, expert: u32) {
            self.apply(|s| {
                check(
                    s.victims.first() == Some(&expert),
                    Error::ShadowDisagreement,
                )?;
                let index = s
                    .residents
                    .iter()
                    .position(|r| r.expert == expert)
                    .ok_or(Error::ShadowDisagreement)?;
                let event = increment(s.event)?;
                let count = increment(s.physical_victims)?;
                s.residents.remove(index);
                s.victims.remove(0);
                s.event = event;
                s.physical_victims = count;
                Ok(())
            });
        }
        /// Called only after successful committed installation and current
        /// generation validation. Reserved, failed and stale work emits nothing.
        pub fn committed_install(&mut self, resident: Resident) {
            self.apply(|s| {
                check(
                    resident.valid()
                        && s.victims.is_empty()
                        && s.installs.first() == Some(&resident.expert)
                        && s.residents.len() < CAPACITY
                        && !s.residents.iter().any(|r| {
                            r.expert == resident.expert
                                || (r.bank, r.slot) == (resident.bank, resident.slot)
                        }),
                    Error::ShadowDisagreement,
                )?;
                let event = increment(s.event)?;
                let count = increment(s.committed_installs)?;
                s.residents.push(resident);
                s.installs.remove(0);
                s.event = event;
                s.committed_installs = count;
                Ok(())
            });
        }
        // Logical eviction has deliberately no physical event or mutation.
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
    struct Key {
        revision: u64,
        source_layer: usize,
        target_layer: usize,
        position_delta: u64,
        source_kind: PositionKind,
        target_kind: PositionKind,
        source_expert: u32,
        target_expert: u32,
    }
    impl Key {
        fn new(source: PositionKind, target: PositionKind, s: u32, e: u32) -> Self {
            Self {
                revision: REVISION,
                source_layer: LAYER,
                target_layer: LAYER,
                position_delta: 1,
                source_kind: source,
                target_kind: target,
                source_expert: s,
                target_expert: e,
            }
        }
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum Permanence {
        Unknown,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct HostSource {
        pub logical_generation: Option<u64>,
        pub logical_materialized: bool,
        pub ram_resident: bool,
        pub permanence: Permanence,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Candidate {
        pub request: RequestIdentity,
        pub model: ModelMetadata,
        pub namespace: Namespace,
        pub source_position: PositionIdentity,
        pub target_position: PositionIdentity,
        pub source_layer: usize,
        pub target_layer: usize,
        pub position_distance: u64,
        pub nominal_layer_lead: usize,
        pub source_set: [u32; CAPACITY],
        pub expert: u32,
        pub score: u64,
        pub signal_revision: u64,
        pub generation: u64,
        pub sequence: u64,
        /// Completed clean positions and 64-cell updates included in scoring.
        pub committed_position_cutoff: u64,
        pub table_update_cutoff: u64,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Freeze {
        pub candidate: Candidate,
        pub timestamp_ns: u64,
        pub physical: Option<PhysicalEvidence>,
        pub current: Option<bool>,
        pub source: HostSource,
        pub incomplete: Option<Error>,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Deadline {
        pub request: RequestIdentity,
        pub position: PositionIdentity,
        pub timestamp_ns: u64,
        pub physical: Option<PhysicalEvidence>,
        pub current: Option<bool>,
        pub host_lead_ns: Option<u64>,
        pub incomplete: Option<Error>,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum Outcome {
        Pending,
        Resolved { prediction_hit: bool },
        Censored,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct RecoveryEvent {
        pub attempt: usize,
        pub attempted_start: usize,
        pub attempted_end: usize,
        pub first_failure_layer: Option<usize>,
        pub final_status: u32,
        pub layer47_demand_service_completed: bool,
    }
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Observation {
        pub freeze: Freeze,
        pub deadline: Option<Deadline>,
        pub outcome: Outcome,
        pub recovery: Vec<RecoveryEvent>,
        pub completion_physical: Option<PhysicalEvidence>,
        pub incomplete: Option<Error>,
        initial_attempt_seen: bool,
        deadline_eligible: bool,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Opportunities {
        pub prediction_hit: Option<bool>,
        pub physical_miss_at_f: Option<bool>,
        pub physical_miss_at_d: Option<bool>,
        pub useful: Option<bool>,
        pub target_confirmed_useful: Option<bool>,
        pub already_resident: Option<bool>,
        pub redundant_route_hit: Option<bool>,
    }
    impl Observation {
        pub fn opportunities(&self) -> Opportunities {
            let hit = match self.outcome {
                Outcome::Resolved { prediction_hit } => Some(prediction_hit),
                _ => None,
            };
            let f = self.freeze.current.map(|v| !v);
            let d = self.deadline.and_then(|d| d.current).map(|v| !v);
            // Unknown inputs stay outside boolean partitions, even when another
            // operand is false. No target failure is converted into a route miss.
            let useful = hit.zip(f).map(|(h, f)| h && f);
            Opportunities {
                prediction_hit: hit,
                physical_miss_at_f: f,
                physical_miss_at_d: d,
                useful,
                target_confirmed_useful: useful.zip(d).map(|(u, d)| u && d),
                already_resident: self.freeze.current,
                redundant_route_hit: hit.zip(self.freeze.current).map(|(h, f)| h && f),
            }
        }
        /// Hypothetical host preparation budget only. This is neither GPU
        /// execution lead nor measured quarantine readiness.
        pub fn timely(&self, budget_ns: u64) -> Option<bool> {
            self.opportunities()
                .target_confirmed_useful
                .zip(self.deadline?.host_lead_ns)
                .map(|(useful, lead)| useful && lead > budget_ns)
        }
        pub fn late(&self, budget_ns: u64) -> Option<bool> {
            self.opportunities()
                .target_confirmed_useful
                .zip(self.deadline?.host_lead_ns)
                .map(|(useful, lead)| useful && lead <= budget_ns)
        }
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum NoEmissionReason {
        NoPositiveHistory,
        Incomplete,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct NoEmission {
        pub source: PositionIdentity,
        pub target: PositionIdentity,
        pub reason: NoEmissionReason,
    }
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Partitions {
        pub emitted: usize,
        pub resolved: usize,
        pub pending: usize,
        pub censored: usize,
        pub prediction_hits: usize,
        pub route_misses: usize,
        pub valid_f: usize,
        pub current_at_f: usize,
        pub absent_at_f: usize,
        pub useful: usize,
        pub target_confirmed_useful: usize,
    }
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub struct Report {
        pub request: RequestIdentity,
        pub observations: Vec<Observation>,
        /// Per-candidate derived fields, joined by the immutable sequence.
        pub opportunities: Vec<(u64, Opportunities)>,
        pub no_emissions: Vec<NoEmission>,
        pub incomplete: Option<Error>,
        pub partitions: Partitions,
        pub readiness_measured: bool,
    }
    /// No Clone/shared state. Only this request's already committed truth trains
    /// this table; it is independent from P0 adjacent transitions and lifecycle.
    #[derive(Debug)]
    pub struct Temporal {
        request: RequestIdentity,
        model: ModelMetadata,
        namespace: Namespace,
        capacity: usize,
        table: BTreeMap<Key, u64>,
        previous: Option<(PositionIdentity, [u32; CAPACITY])>,
        committed: u64,
        updates: u64,
        generation: u64,
        sequence: u64,
        prepared: Option<Candidate>,
        records: Vec<Observation>,
        no_emissions: Vec<NoEmission>,
        incomplete: Option<Error>,
        closed: bool,
    }
    impl Temporal {
        pub fn new(
            request: RequestIdentity,
            model: ModelMetadata,
            namespace: Namespace,
            capacity: usize,
        ) -> Result<Self> {
            check(
                namespace.valid()
                    && namespace.runtime == request.runtime_namespace
                    && model.num_layers == LAYER + 1
                    && model.num_experts == EXPERTS
                    && model.top_k == CAPACITY
                    && capacity > 0,
                Error::Identity,
            )?;
            Ok(Self {
                request,
                model,
                namespace,
                capacity,
                table: BTreeMap::new(),
                previous: None,
                committed: 0,
                updates: 0,
                generation: 0,
                sequence: 0,
                prepared: None,
                records: Vec::new(),
                no_emissions: Vec::new(),
                incomplete: None,
                closed: false,
            })
        }
        pub fn mark_incomplete(&mut self, e: Error) {
            self.incomplete.get_or_insert(e);
        }
        pub fn namespace(&self) -> Namespace {
            self.namespace
        }
        pub(super) fn note_p0_truth(&mut self, position: PositionIdentity) {
            if self
                .prepared
                .is_some_and(|c| position.absolute_position >= c.target_position.absolute_position)
            {
                self.abandon_prepared(Error::Identity);
            }
        }
        pub fn abandon_prepared(&mut self, e: Error) {
            if let Some(c) = self.prepared.take() {
                self.note_no_emission(
                    c.source_position,
                    c.target_position,
                    NoEmissionReason::Incomplete,
                );
            }
            self.mark_incomplete(e);
        }
        pub fn censor_target(&mut self, position: usize) {
            if let Some(r) = self.pending_mut(position) {
                r.outcome = Outcome::Censored;
            }
            self.mark_incomplete(Error::Closed);
        }
        pub fn service_completed(&mut self, position: usize, attempt: usize) {
            if let Some(r) = self.pending_mut(position) {
                if let Some(e) = r
                    .recovery
                    .last_mut()
                    .filter(|e| e.attempt == attempt && e.first_failure_layer == Some(LAYER))
                {
                    e.layer47_demand_service_completed = true;
                }
            }
        }
        pub fn active(&self) -> bool {
            !self.closed && self.incomplete.is_none()
        }
        fn record_capacity(&self) -> Result<()> {
            check(
                self.records
                    .len()
                    .checked_add(self.no_emissions.len())
                    .ok_or(Error::Overflow)?
                    < self.capacity.min(MAX_REPORT_RECORDS),
                Error::Capacity,
            )
        }
        fn note_no_emission(
            &mut self,
            source: PositionIdentity,
            target: PositionIdentity,
            reason: NoEmissionReason,
        ) {
            if self.record_capacity().is_ok() {
                self.no_emissions.push(NoEmission {
                    source,
                    target,
                    reason,
                });
            }
        }
        /// Only called through RequestObserver after P0 has accepted an entire
        /// clean committed position. Target truth resolves the old immutable
        /// record first, then trains historical pairs, then scores the NEXT target.
        pub(super) fn committed(
            &mut self,
            position: PositionIdentity,
            target: PositionIdentity,
            ids: &[u32],
        ) -> Option<Candidate> {
            let result = (|| {
                check(!self.closed, Error::Closed)?;
                let set = selected(ids)?;
                check(
                    position.valid()
                        && position.absolute_position == self.committed
                        && target.follows(position)
                        && self.prepared.is_none(),
                    Error::Identity,
                )?;
                if let Some(last) = self
                    .records
                    .last_mut()
                    .filter(|r| r.freeze.candidate.target_position == position)
                {
                    check(last.outcome == Outcome::Pending, Error::Identity)?;
                    last.outcome = Outcome::Resolved {
                        prediction_hit: set.contains(&last.freeze.candidate.expert),
                    };
                    if last.deadline.is_none() {
                        last.incomplete.get_or_insert(Error::MissingDeadline);
                        return Err(Error::MissingDeadline);
                    }
                }
                check(self.incomplete.is_none(), Error::Incomplete)?;
                self.record_capacity()?;
                let committed = increment(self.committed)?;
                let generation = increment(self.generation)?;
                let mut changes = Vec::new();
                let mut updates = self.updates;
                if let Some((previous_position, previous_set)) = self.previous {
                    check(position.follows(previous_position), Error::Identity)?;
                    updates = increment(updates)?;
                    for s in previous_set {
                        for e in set {
                            let key = Key::new(
                                previous_position.position_kind,
                                position.position_kind,
                                s,
                                e,
                            );
                            let count = increment(self.table.get(&key).copied().unwrap_or(0))?;
                            changes.push((key, count));
                        }
                    }
                    let extra = changes
                        .iter()
                        .filter(|(k, _)| !self.table.contains_key(k))
                        .count();
                    check(
                        self.table.len().checked_add(extra).ok_or(Error::Overflow)?
                            <= self.capacity,
                        Error::Capacity,
                    )?;
                } else {
                    check(
                        position.absolute_position == 0
                            && position.position_kind == PositionKind::Prompt,
                        Error::Identity,
                    )?;
                }
                // Every one of the 64 increments, update cutoff and capacity
                // checks succeeded before the first table cell is changed.
                for (key, count) in changes {
                    self.table.insert(key, count);
                }
                self.previous = Some((position, set));
                self.committed = committed;
                self.updates = updates;
                self.generation = generation;
                let mut best = None;
                for expert in 0..EXPERTS as u32 {
                    let mut score = 0u64;
                    for s in set {
                        score = score
                            .checked_add(
                                self.table
                                    .get(&Key::new(
                                        position.position_kind,
                                        target.position_kind,
                                        s,
                                        expert,
                                    ))
                                    .copied()
                                    .unwrap_or(0),
                            )
                            .ok_or(Error::Overflow)?;
                    }
                    // Ascending iteration retains the smallest ID on ties.
                    if score > best.map_or(0, |(_, score)| score) {
                        best = Some((expert, score));
                    }
                }
                let Some((expert, score)) = best else {
                    return Ok(None);
                };
                let sequence = increment(self.sequence)?;
                let candidate = Candidate {
                    request: self.request,
                    model: self.model,
                    namespace: self.namespace,
                    source_position: position,
                    target_position: target,
                    source_layer: LAYER,
                    target_layer: LAYER,
                    position_distance: 1,
                    nominal_layer_lead: LAYER,
                    source_set: set,
                    expert,
                    score,
                    signal_revision: REVISION,
                    generation,
                    sequence,
                    committed_position_cutoff: committed,
                    table_update_cutoff: updates,
                };
                self.prepared = Some(candidate);
                Ok(Some(candidate))
            })();
            match result {
                Ok(Some(candidate)) => Some(candidate),
                Ok(None) => {
                    self.note_no_emission(position, target, NoEmissionReason::NoPositiveHistory);
                    None
                }
                Err(e) => {
                    self.mark_incomplete(e);
                    self.note_no_emission(position, target, NoEmissionReason::Incomplete);
                    None
                }
            }
        }
        pub fn freeze(
            &mut self,
            candidate: Candidate,
            timestamp_ns: u64,
            physical: Result<PhysicalEvidence>,
            source: HostSource,
        ) {
            let result = (|| {
                check(
                    self.active() && self.prepared == Some(candidate),
                    Error::Identity,
                )?;
                self.record_capacity()?;
                let current =
                    physical.and_then(|p| p.snapshot.current(self.namespace, candidate.expert));
                let error = current.err();
                self.records.push(Observation {
                    freeze: Freeze {
                        candidate,
                        timestamp_ns,
                        physical: physical.ok(),
                        current: current.ok(),
                        source,
                        incomplete: error,
                    },
                    deadline: None,
                    outcome: Outcome::Pending,
                    recovery: Vec::new(),
                    completion_physical: None,
                    incomplete: error,
                    initial_attempt_seen: false,
                    deadline_eligible: false,
                });
                self.sequence = candidate.sequence;
                self.prepared = None;
                if let Some(e) = error {
                    self.mark_incomplete(e);
                }
                Ok(())
            })();
            if let Err(e) = result {
                self.mark_incomplete(e);
            }
        }
        fn pending_mut(&mut self, position: usize) -> Option<&mut Observation> {
            let request = self.request;
            self.records.last_mut().filter(|r| {
                r.outcome == Outcome::Pending
                    && r.freeze.candidate.request == request
                    && r.freeze.candidate.target_position.absolute_position == position as u64
            })
        }
        /// Close eligibility at the first attempt, even if encoding fails before
        /// reaching layer 47. Recovery/full-token replay can never reopen it.
        pub fn begin_attempt(&mut self, position: usize, initial_fresh: bool) -> bool {
            if self
                .prepared
                .is_some_and(|c| position as u64 >= c.target_position.absolute_position)
            {
                self.abandon_prepared(Error::Identity);
            }
            let Some(record) = self.pending_mut(position) else {
                return false;
            };
            let first = !record.initial_attempt_seen;
            record.initial_attempt_seen = true;
            record.deadline_eligible = first && initial_fresh;
            record.deadline_eligible
        }
        pub fn pending_candidate(&self, position: usize) -> Option<Candidate> {
            self.records
                .last()
                .filter(|r| {
                    r.outcome == Outcome::Pending
                        && r.freeze.candidate.target_position.absolute_position == position as u64
                })
                .map(|r| r.freeze.candidate)
        }
        pub fn pending_freeze(&self, position: usize) -> Option<Freeze> {
            self.records
                .last()
                .filter(|r| {
                    r.outcome == Outcome::Pending
                        && r.freeze.candidate.target_position.absolute_position == position as u64
                })
                .map(|r| r.freeze)
        }
        pub fn p1j_deadline_valid(&self, candidate: Candidate) -> bool {
            self.active()
                && self.records.last().is_some_and(|r| {
                    r.freeze.candidate == candidate
                        && r.incomplete.is_none()
                        && r.deadline.is_some_and(|d| {
                            d.incomplete.is_none()
                                && d.current.is_some()
                                && d.request == candidate.request
                                && d.position == candidate.target_position
                        })
                })
        }
        pub fn deadline(
            &mut self,
            request: RequestIdentity,
            position: usize,
            timestamp_ns: u64,
            physical: Result<PhysicalEvidence>,
        ) {
            let Some(record) = self.pending_mut(position) else {
                self.mark_incomplete(Error::Identity);
                return;
            };
            if record.deadline.is_some() {
                return;
            }
            let c = record.freeze.candidate;
            let lead = timestamp_ns
                .checked_sub(record.freeze.timestamp_ns)
                .filter(|&v| v > 0);
            let current = (|| {
                check(
                    record.deadline_eligible && request == c.request,
                    Error::Identity,
                )?;
                check(lead.is_some(), Error::Chronology)?;
                physical?.snapshot.current(c.namespace, c.expert)
            })();
            let error = current.err();
            record.deadline = Some(Deadline {
                request,
                position: c.target_position,
                timestamp_ns,
                physical: physical.ok(),
                current: current.ok(),
                host_lead_ns: lead,
                incomplete: error,
            });
            if let Some(e) = error {
                record.incomplete.get_or_insert(e);
                self.mark_incomplete(e);
            }
        }
        pub fn recovery_event(&mut self, position: usize, event: RecoveryEvent) {
            if let Some(record) = self.pending_mut(position) {
                if record.recovery.len() >= MAX_RECOVERY_EVENTS {
                    record.incomplete.get_or_insert(Error::Capacity);
                    self.mark_incomplete(Error::Capacity);
                } else {
                    record.recovery.push(event);
                }
            }
        }
        pub fn completion_evidence(&mut self, position: usize, physical: Result<PhysicalEvidence>) {
            if let Some(record) = self
                .records
                .last_mut()
                .filter(|r| r.freeze.candidate.target_position.absolute_position == position as u64)
            {
                record.completion_physical = physical.ok();
                if let Err(e) = physical {
                    record.incomplete.get_or_insert(e);
                    self.mark_incomplete(e);
                }
            }
        }
        pub fn finish(&mut self) {
            self.closed = true;
            self.previous = None;
            self.prepared = None;
            for r in &mut self.records {
                if r.outcome == Outcome::Pending {
                    r.outcome = Outcome::Censored;
                }
            }
        }
        pub fn report(&self) -> Report {
            let mut p = Partitions {
                emitted: self.records.len(),
                ..Partitions::default()
            };
            for r in &self.records {
                match r.outcome {
                    Outcome::Pending => p.pending += 1,
                    Outcome::Censored => p.censored += 1,
                    Outcome::Resolved { prediction_hit } => {
                        p.resolved += 1;
                        if prediction_hit {
                            p.prediction_hits += 1;
                        } else {
                            p.route_misses += 1;
                        }
                    }
                }
                if let Some(current) = r.freeze.current {
                    p.valid_f += 1;
                    if current {
                        p.current_at_f += 1;
                    } else {
                        p.absent_at_f += 1;
                    }
                }
                let o = r.opportunities();
                p.useful += usize::from(o.useful == Some(true));
                p.target_confirmed_useful += usize::from(o.target_confirmed_useful == Some(true));
            }
            let reconciled = p.emitted == p.resolved + p.pending + p.censored
                && p.resolved == p.prediction_hits + p.route_misses
                && p.valid_f == p.current_at_f + p.absent_at_f
                && p.useful <= p.prediction_hits
                && p.useful <= p.absent_at_f
                && p.target_confirmed_useful <= p.useful;
            Report {
                request: self.request,
                observations: self.records.clone(),
                opportunities: self
                    .records
                    .iter()
                    .map(|r| (r.freeze.candidate.sequence, r.opportunities()))
                    .collect(),
                no_emissions: self.no_emissions.clone(),
                incomplete: self.incomplete.or(if reconciled {
                    None
                } else {
                    Some(Error::Identity)
                }),
                partitions: p,
                readiness_measured: false,
            }
        }
    }
    #[cfg(test)]
    mod tests {
        use super::super::{RequestObserver, RequestPhase};
        use super::*;

        fn request(n: u64, phase: RequestPhase) -> RequestIdentity {
            RequestIdentity {
                runtime_namespace: 1,
                request_sequence: n,
                phase,
                phase_run_index: n,
            }
        }
        fn namespace() -> Namespace {
            Namespace {
                runtime: 1,
                context: 2,
                arena: 3,
                layer: LAYER,
                capacity: CAPACITY,
            }
        }
        fn model() -> ModelMetadata {
            ModelMetadata {
                num_layers: 48,
                num_experts: EXPERTS,
                top_k: CAPACITY,
            }
        }
        fn observer(n: u64, phase: RequestPhase) -> RequestObserver {
            let mut observer = RequestObserver::new(request(n, phase), model(), 100_000).unwrap();
            observer.enable_temporal(namespace()).unwrap();
            observer
        }
        fn set(start: u32) -> [u32; 8] {
            std::array::from_fn(|i| start + i as u32)
        }
        fn position(p: usize, prompt: usize) -> PositionIdentity {
            PositionIdentity::from_prompt_length(p, prompt).unwrap()
        }
        fn resident(expert: u32, slot: u32, epoch: u32) -> Resident {
            Resident {
                expert,
                generation: 1,
                bank: 0,
                slot,
                epoch,
            }
        }
        fn snapshot(ids: &[u32]) -> PhysicalSnapshot {
            let mut residents = [None; 8];
            for (slot, &id) in ids.iter().enumerate() {
                residents[slot] = Some(resident(id, slot as u32, 1));
            }
            PhysicalSnapshot {
                namespace: namespace(),
                residents,
            }
        }
        fn evidence(ids: &[u32]) -> PhysicalEvidence {
            PhysicalEvidence {
                snapshot: snapshot(ids),
                event_cutoff: 0,
                committed_installs: 0,
                physical_victims: 0,
            }
        }
        fn source() -> HostSource {
            HostSource {
                logical_generation: None,
                logical_materialized: false,
                ram_resident: false,
                permanence: Permanence::Unknown,
            }
        }
        fn commit(
            o: &mut RequestObserver,
            p: usize,
            prompt: usize,
            ids: &[u32],
        ) -> Option<Candidate> {
            let pos = position(p, prompt);
            o.observe_completed_position(
                o.request_identity(),
                pos,
                model(),
                &vec![ids.to_vec(); 48],
            );
            o.prepare_temporal(pos, position(p + 1, prompt))
        }
        fn step(
            o: &mut RequestObserver,
            p: usize,
            prompt: usize,
            ids: &[u32],
        ) -> Option<Candidate> {
            // Model the initial-attempt deadline before delivering any target truth.
            let t = o.temporal_mut().unwrap();
            if let Some(candidate) = t.pending_candidate(p) {
                assert!(t.begin_attempt(p, true));
                t.deadline(
                    candidate.request,
                    p,
                    p as u64 * 100,
                    Ok(evidence(&candidate.source_set)),
                );
            }
            let candidate = commit(o, p, prompt, ids);
            if let Some(c) = candidate {
                o.temporal_mut().unwrap().freeze(
                    c,
                    p as u64 * 100 + 1,
                    Ok(evidence(ids)),
                    source(),
                );
            }
            candidate
        }
        fn alternating() -> RequestObserver {
            let mut o = observer(1, RequestPhase::Fixture);
            assert_eq!(step(&mut o, 0, 10, &set(0)), None);
            assert_eq!(step(&mut o, 1, 10, &set(8)), None);
            let c = step(&mut o, 2, 10, &set(0)).unwrap();
            assert_eq!((c.expert, c.score), (8, 8));
            o
        }

        #[test]
        fn p1e_a_b_a_freezes_expert8_absent_before_truth_and_reconciles_usefulness() {
            let mut o = alternating();
            let before = o.temporal().unwrap().report().observations[0].freeze;
            assert_eq!(before.current, Some(false));
            assert_eq!(before.candidate.source_set, set(0));
            assert_eq!(before.candidate.table_update_cutoff, 2);
            assert_eq!(before.candidate.committed_position_cutoff, 3);
            assert_eq!(before.candidate.target_position, position(3, 10));
            assert_eq!(o.temporal().unwrap().report().partitions.resolved, 0);
            step(&mut o, 3, 10, &set(8));
            let report = o.temporal().unwrap().report();
            let r = &report.observations[0];
            assert_eq!(r.freeze, before);
            assert_eq!(
                r.opportunities(),
                Opportunities {
                    prediction_hit: Some(true),
                    physical_miss_at_f: Some(true),
                    physical_miss_at_d: Some(true),
                    useful: Some(true),
                    target_confirmed_useful: Some(true),
                    already_resident: Some(false),
                    redundant_route_hit: Some(false)
                }
            );
            assert_eq!(r.deadline.unwrap().host_lead_ns, Some(99));
            assert_eq!(r.timely(98), Some(true));
            assert_eq!(r.late(99), Some(true));
            assert_eq!(
                (
                    report.partitions.emitted,
                    report.partitions.resolved,
                    report.partitions.pending
                ),
                (2, 1, 1)
            );
            assert_eq!(
                (
                    report.partitions.prediction_hits,
                    report.partitions.useful,
                    report.partitions.target_confirmed_useful
                ),
                (1, 1, 1)
            );
            assert_eq!(report.incomplete, None);
            assert!(!report.readiness_measured);
        }

        #[test]
        fn p1e_redundant_top1_is_retained_with_no_runner_up() {
            let mut o = observer(1, RequestPhase::Fixture);
            step(&mut o, 0, 10, &set(0));
            let c = step(&mut o, 1, 10, &set(0)).unwrap();
            assert_eq!(c.expert, 0);
            // A lower scoring absent expert must not replace the resident winner.
            o.temporal_mut().unwrap().table.insert(
                Key::new(PositionKind::Prompt, PositionKind::Prompt, 0, 8),
                1,
            );
            let next = step(&mut o, 2, 10, &set(0)).unwrap();
            assert_eq!(next.expert, 0);
            let r = &o.temporal().unwrap().report().observations[0];
            assert_eq!(r.opportunities().redundant_route_hit, Some(true));
            assert_eq!(r.opportunities().useful, Some(false));
            assert_eq!(r.freeze.current, Some(true));
        }

        #[test]
        fn p1e_unfrozen_draft_cannot_be_frozen_after_target_truth_or_attempt() {
            for truth in [false, true] {
                let mut o = observer(1, RequestPhase::Fixture);
                step(&mut o, 0, 10, &set(0));
                let draft = commit(&mut o, 1, 10, &set(0)).unwrap();
                if truth {
                    o.observe_completed_position(
                        o.request_identity(),
                        position(2, 10),
                        model(),
                        &vec![set(8).to_vec(); 48],
                    );
                } else {
                    assert!(!o.temporal_mut().unwrap().begin_attempt(2, true));
                }
                o.temporal_mut()
                    .unwrap()
                    .freeze(draft, 201, Ok(evidence(&set(8))), source());
                let report = o.temporal().unwrap().report();
                assert!(report.observations.is_empty());
                assert!(report.incomplete.is_some());
                assert_eq!(
                    report.no_emissions.last().unwrap().reason,
                    NoEmissionReason::Incomplete
                );
            }
        }

        #[test]
        fn p1e_request_phase_and_run_reset_isolation() {
            let old = alternating();
            let frozen = old.temporal().unwrap().report();
            for (n, phase) in [
                (2, RequestPhase::Warmup),
                (3, RequestPhase::Measured),
                (4, RequestPhase::Serving),
                (5, RequestPhase::Measured),
            ] {
                let mut fresh = observer(n, phase);
                assert!(fresh.temporal().unwrap().table.is_empty());
                assert_eq!(step(&mut fresh, 0, 10, &set(0)), None);
                assert_eq!(
                    fresh.temporal().unwrap().report().request,
                    request(n, phase)
                );
                fresh.finish(false);
                assert!(!fresh.temporal().unwrap().active());
            }
            assert_eq!(old.temporal().unwrap().report(), frozen);
        }

        #[test]
        fn p1e_kind_pairs_start_cold_and_never_borrow() {
            let mut o = observer(1, RequestPhase::Fixture);
            assert_eq!(step(&mut o, 0, 3, &set(0)), None);
            assert!(step(&mut o, 1, 3, &set(0)).is_some()); // prompt->prompt
            assert_eq!(step(&mut o, 2, 3, &set(0)), None); // prompt->decode cold
            assert_eq!(step(&mut o, 3, 3, &set(0)), None); // decode->decode cold
            assert!(step(&mut o, 4, 3, &set(0)).is_some());
            let t = o.temporal().unwrap();
            let count = |a, b| {
                t.table
                    .iter()
                    .filter(|(k, _)| k.source_kind == a && k.target_kind == b)
                    .map(|(_, &v)| v)
                    .sum::<u64>()
            };
            assert_eq!(count(PositionKind::Prompt, PositionKind::Prompt), 128);
            assert_eq!(count(PositionKind::Prompt, PositionKind::Decode), 64);
            assert_eq!(count(PositionKind::Decode, PositionKind::Decode), 64);
        }

        #[test]
        fn p1e_zero_history_ties_order_invariance_and_exact_64_cells() {
            let mut forward = observer(1, RequestPhase::Fixture);
            let mut reverse = observer(1, RequestPhase::Fixture);
            for (p, ids) in [set(0), set(8), set(0)].iter().enumerate() {
                let mut reversed = *ids;
                reversed.reverse();
                assert_eq!(
                    step(&mut forward, p, 10, ids),
                    step(&mut reverse, p, 10, &reversed)
                );
                assert_eq!(
                    forward.temporal().unwrap().table,
                    reverse.temporal().unwrap().table
                );
                assert_eq!(forward.temporal().unwrap().table.len(), p * 64);
                assert!(forward.temporal().unwrap().table.values().all(|&v| v == 1));
            }
            assert_eq!(forward.temporal().unwrap().report().no_emissions.len(), 2);
            assert_eq!(
                forward.temporal().unwrap().report().observations[0]
                    .freeze
                    .candidate
                    .expert,
                8
            );
        }

        #[test]
        fn p1e_checked_update_overflow_and_capacity_are_atomic() {
            for capacity_failure in [false, true] {
                let mut o = observer(1, RequestPhase::Fixture);
                step(&mut o, 0, 10, &set(0));
                let t = o.temporal_mut().unwrap();
                // Last Cartesian cell would overflow: none of the first 63 may change.
                if capacity_failure {
                    t.capacity = 63;
                } else {
                    t.table.insert(
                        Key::new(PositionKind::Prompt, PositionKind::Prompt, 7, 15),
                        u64::MAX,
                    );
                }
                let before = (t.table.clone(), t.updates, t.committed, t.previous);
                assert!(commit(&mut o, 1, 10, &set(8)).is_none());
                let t = o.temporal().unwrap();
                assert_eq!(
                    (t.table.clone(), t.updates, t.committed, t.previous),
                    before
                );
                assert_eq!(
                    t.incomplete,
                    Some(if capacity_failure {
                        Error::Capacity
                    } else {
                        Error::Overflow
                    })
                );
                assert_eq!(
                    o.completed_positions(),
                    2,
                    "observation failure cannot reject P0 completion"
                );
                assert_eq!(commit(&mut o, 2, 10, &set(0)), None);
                assert_eq!(o.completed_positions(), 3);
            }
        }

        #[test]
        fn p1e_checked_score_and_identity_counter_overflows_stop_emission() {
            for mode in 0..4 {
                let mut o = observer(1, RequestPhase::Fixture);
                step(&mut o, 0, 10, &set(0));
                let t = o.temporal_mut().unwrap();
                match mode {
                    0 => {
                        for s in 0..8 {
                            t.table.insert(
                                Key::new(PositionKind::Prompt, PositionKind::Prompt, s, 99),
                                u64::MAX,
                            );
                        }
                    }
                    1 => t.generation = u64::MAX,
                    2 => t.sequence = u64::MAX,
                    _ => t.updates = u64::MAX,
                }
                assert_eq!(commit(&mut o, 1, 10, &set(0)), None);
                assert_eq!(o.temporal().unwrap().incomplete, Some(Error::Overflow));
                assert_eq!(o.completed_positions(), 2);
            }
        }

        #[test]
        fn p1e_duplicate_out_of_order_and_malformed_positions_rejected() {
            for mode in 0..4 {
                let mut o = observer(1, RequestPhase::Fixture);
                step(&mut o, 0, 10, &set(0));
                let before = o.temporal().unwrap().table.clone();
                let p = if mode == 0 {
                    0
                } else if mode == 1 {
                    2
                } else {
                    1
                };
                let mut ids = set(0).to_vec();
                if mode == 2 {
                    ids[7] = ids[0];
                }
                if mode == 3 {
                    ids.pop();
                }
                assert!(commit(&mut o, p, 10, &ids).is_none());
                assert_eq!(o.temporal().unwrap().table, before);
                assert!(o.temporal().unwrap().incomplete.is_some());
            }
            let mut o = observer(1, RequestPhase::Fixture);
            assert!(o
                .prepare_temporal(position(0, 10), position(1, 10))
                .is_none());
            assert_eq!(o.temporal().unwrap().incomplete, Some(Error::Incomplete));
        }

        #[test]
        fn p1e_recovery_deadline_closes_once_and_final_clean_truth_trains_once() {
            let mut o = alternating();
            let t = o.temporal_mut().unwrap();
            let frozen = t.records[0].freeze;
            assert!(t.begin_attempt(3, true));
            t.deadline(
                request(1, RequestPhase::Fixture),
                3,
                250,
                Ok(evidence(&set(0))),
            );
            let deadline = t.records[0].deadline;
            let table = t.table.clone();
            for attempt in 1..=3 {
                t.recovery_event(
                    3,
                    RecoveryEvent {
                        attempt,
                        attempted_start: 0,
                        attempted_end: 48,
                        first_failure_layer: Some(LAYER),
                        final_status: 0,
                        layer47_demand_service_completed: false,
                    },
                );
                t.service_completed(3, attempt);
                assert!(!t.begin_attempt(3, true));
                t.deadline(
                    request(1, RequestPhase::Fixture),
                    3,
                    999,
                    Ok(evidence(&set(8))),
                );
                assert_eq!(t.table, table);
                assert_eq!(t.records[0].freeze, frozen);
                assert_eq!(t.records[0].deadline, deadline);
            }
            let c = commit(&mut o, 3, 10, &set(8)).unwrap();
            let t = o.temporal().unwrap();
            assert_eq!(t.updates, 3);
            assert_eq!(
                t.records[0].outcome,
                Outcome::Resolved {
                    prediction_hit: true
                }
            );
            assert!(t.records[0]
                .recovery
                .iter()
                .all(|e| e.layer47_demand_service_completed));
            assert_eq!(c.source_position, position(3, 10));
            assert!(commit(&mut o, 3, 10, &set(8)).is_none());
            assert_eq!(o.temporal().unwrap().updates, 3);
        }

        #[test]
        fn p1e_initial_attempt_failure_before_layer47_cannot_reopen_deadline() {
            let mut o = alternating();
            let t = o.temporal_mut().unwrap();
            assert!(t.begin_attempt(3, true));
            // No layer 47 hook was reached before initial encoding failure.
            assert!(!t.begin_attempt(3, true));
            t.deadline(
                request(1, RequestPhase::Fixture),
                3,
                250,
                Ok(evidence(&set(0))),
            );
            assert_eq!(t.records[0].deadline.unwrap().current, None);
            assert_eq!(t.incomplete, Some(Error::Identity));
        }

        #[test]
        fn p1e_equal_negative_and_wrong_identity_deadlines_are_incomplete() {
            for (time, wrong_request) in [(201, false), (200, false), (250, true)] {
                let mut o = alternating();
                let t = o.temporal_mut().unwrap();
                t.begin_attempt(3, true);
                t.deadline(
                    request(if wrong_request { 2 } else { 1 }, RequestPhase::Fixture),
                    3,
                    time,
                    Ok(evidence(&set(0))),
                );
                assert!(t.incomplete.is_some());
                assert_eq!(t.records[0].deadline.unwrap().current, None);
                assert_eq!(t.records[0].opportunities().target_confirmed_useful, None);
                // Clean target truth remains usable independently of bad timing.
                assert!(commit(&mut o, 3, 10, &set(8)).is_none());
                assert_eq!(
                    o.temporal().unwrap().records[0]
                        .opportunities()
                        .prediction_hit,
                    Some(true)
                );
            }
        }

        #[test]
        fn p1e_route_miss_redundancy_and_unknown_partitions_reconcile() {
            for target in [set(8), set(16)] {
                for d_current in [false, true] {
                    let mut o = alternating();
                    let t = o.temporal_mut().unwrap();
                    t.begin_attempt(3, true);
                    t.deadline(
                        request(1, RequestPhase::Fixture),
                        3,
                        250,
                        Ok(evidence(&if d_current { set(8) } else { set(0) })),
                    );
                    commit(&mut o, 3, 10, &target);
                    let r = o.temporal().unwrap().report();
                    let expected_hit = target == set(8);
                    assert_eq!(r.partitions.prediction_hits, usize::from(expected_hit));
                    assert_eq!(r.partitions.route_misses, usize::from(!expected_hit));
                    assert_eq!(
                        r.partitions.target_confirmed_useful,
                        usize::from(expected_hit && !d_current)
                    );
                    assert_eq!(r.incomplete, None);
                }
            }
            let mut o = alternating();
            o.temporal_mut().unwrap().censor_target(3);
            let r = o.temporal().unwrap().report();
            assert_eq!(
                (
                    r.partitions.emitted,
                    r.partitions.censored,
                    r.partitions.resolved
                ),
                (1, 1, 0)
            );
            assert_eq!(r.observations[0].opportunities().prediction_hit, None);
            assert_eq!(r.observations[0].opportunities().useful, None);
            let mut cancelled = alternating();
            cancelled.finish(true);
            assert_eq!(
                cancelled.temporal().unwrap().report().partitions.censored,
                1
            );
        }

        #[test]
        fn p1e_bad_f_evidence_does_not_become_absence_or_hide_clean_route_truth() {
            let mut o = observer(1, RequestPhase::Fixture);
            step(&mut o, 0, 10, &set(0));
            let c = commit(&mut o, 1, 10, &set(0)).unwrap();
            o.temporal_mut()
                .unwrap()
                .freeze(c, 101, Err(Error::ShadowDisagreement), source());
            let t = o.temporal_mut().unwrap();
            t.begin_attempt(2, true);
            t.deadline(c.request, 2, 200, Err(Error::ShadowDisagreement));
            assert!(commit(&mut o, 2, 10, &set(0)).is_none());
            let r = o.temporal().unwrap().report();
            assert_eq!(r.partitions.prediction_hits, 1);
            assert_eq!(r.partitions.valid_f, 0);
            assert_eq!(r.observations[0].opportunities().useful, None);
        }

        #[test]
        fn p1e_shadow_protects_full_set_evicts_oldest_and_waits_for_commits() {
            let initial = snapshot(&set(0));
            let mut shadow = Shadow::new(initial).unwrap();
            assert_eq!(shadow.evidence(initial).unwrap().snapshot, initial);
            // Hit 0 is oldest but protected, while 6,7 are evictable. Demand
            // ordering controls recency; planned installs do not become current.
            let ids = [8, 0, 1, 2, 3, 4, 5, 9];
            shadow.demand(&ids);
            assert_eq!(shadow.victims, [6, 7]);
            assert!(!shadow.values().current(namespace(), 8).unwrap());
            shadow.victim(6);
            shadow.victim(7);
            assert_eq!(shadow.residents.len(), 6);
            shadow.committed_install(resident(8, 6, 2));
            shadow.committed_install(resident(9, 7, 2));
            let actual = shadow.values();
            assert_eq!(
                actual.residents.map(|r| r.unwrap().expert),
                [0, 1, 2, 3, 4, 5, 8, 9]
            );
            let evidence = shadow.evidence(actual).unwrap();
            assert_eq!(
                (evidence.physical_victims, evidence.committed_installs),
                (2, 2)
            );
            // Starting another request and dropping host logical admission has
            // no shadow reset/event: the same runtime's physical bytes persist.
            drop(observer(2, RequestPhase::Measured));
            assert_eq!(shadow.evidence(actual).unwrap(), evidence);
            let mut bad = Shadow::new(initial).unwrap();
            bad.demand(&ids);
            bad.victim(0);
            assert_eq!(bad.incomplete, Some(Error::ShadowDisagreement));
        }

        #[test]
        fn p1e_shadow_disagreement_and_invalid_namespace_records_fail_closed() {
            let initial = snapshot(&set(0));
            for mode in 0..9 {
                let mut corrupted = initial;
                match mode {
                    0 => corrupted.namespace.context += 1,
                    1 => corrupted.namespace.arena += 1,
                    2 => corrupted.namespace.layer = 0,
                    3 => corrupted.namespace.capacity = 9,
                    4 => corrupted.residents[0].as_mut().unwrap().generation += 1,
                    5 => corrupted.residents[0].as_mut().unwrap().epoch += 1,
                    6 => corrupted.residents[0].as_mut().unwrap().expert = 128,
                    7 => corrupted.residents.swap(0, 1),
                    _ => corrupted.residents[0].as_mut().unwrap().slot = 1,
                }
                let mut shadow = Shadow::new(initial).unwrap();
                assert!(shadow.evidence(corrupted).is_err());
                assert!(
                    shadow.evidence(initial).is_err(),
                    "never guess or repair after disagreement"
                );
            }
            let mut reserved = Shadow::new(snapshot(&[])).unwrap();
            reserved.demand(&set(8));
            assert!(reserved.values().residents.iter().all(Option::is_none));
            assert!(
                reserved.evidence(snapshot(&[])).is_err(),
                "failed/uncommitted work cannot reconcile as installed"
            );
        }

        #[test]
        fn p1e_default_observer_has_no_temporal_state_and_report_is_bounded() {
            let o =
                RequestObserver::new(request(1, RequestPhase::Fixture), model(), 100_000).unwrap();
            assert!(o.temporal().is_none());
            let mut t =
                Temporal::new(request(1, RequestPhase::Fixture), model(), namespace(), 1).unwrap();
            assert!(t
                .committed(position(0, 10), position(1, 10), &set(0))
                .is_none());
            assert!(t
                .committed(position(1, 10), position(2, 10), &set(8))
                .is_none());
            assert_eq!(t.no_emissions.len(), 1);
            assert_eq!(t.incomplete, Some(Error::Capacity));
            assert!(t.records.is_empty());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p1j_observer_fixture() -> (RequestObserver, P1jIdentity) {
        let r = request(1);
        let model = ModelMetadata {
            num_layers: 48,
            num_experts: 128,
            top_k: 8,
        };
        let namespace = p1e::Namespace {
            runtime: 1,
            context: 7,
            arena: 3,
            layer: 47,
            capacity: 8,
        };
        let mut observer = RequestObserver::new(r, model, 100_000).unwrap();
        observer.enable_temporal(namespace).unwrap();
        for (n, first) in [0, 8, 0].into_iter().enumerate() {
            let source = PositionIdentity::from_prompt_length(n, 10).unwrap();
            let target = PositionIdentity::from_prompt_length(n + 1, 10).unwrap();
            observer.observe_completed_position(
                r,
                source,
                model,
                &vec![(first..first + 8).collect(); 48],
            );
            if let Some(candidate) = observer.prepare_temporal(source, target) {
                observer.temporal_mut().unwrap().freeze(
                    candidate,
                    100 + n as u64,
                    Ok(p1e::PhysicalEvidence {
                        snapshot: p1e::PhysicalSnapshot {
                            namespace,
                            residents: [None; 8],
                        },
                        event_cutoff: 0,
                        committed_installs: 0,
                        physical_victims: 0,
                    }),
                    p1e::HostSource {
                        logical_generation: Some(11),
                        logical_materialized: true,
                        ram_resident: false,
                        permanence: p1e::Permanence::Unknown,
                    },
                );
            }
        }
        let candidate = observer.temporal().unwrap().pending_candidate(3).unwrap();
        assert_eq!(candidate.expert, 8);
        (
            observer,
            P1jIdentity {
                candidate,
                logical_generation: 11,
                epoch: 1,
                writer_sequence: 1,
            },
        )
    }
    #[test]
    fn p1j_p0_host_source_and_sidecar_install_identity_reconcile_truthfully() {
        let (mut observer, id) = p1j_observer_fixture();
        let install = observer.p1j_acquired(id).unwrap();
        assert_eq!(
            (
                install.bank,
                install.slot,
                install.slot_epoch,
                install.logical_generation
            ),
            (1, 0, 1, 11)
        );
        assert_eq!(
            (
                install.runtime_namespace,
                install.executor_namespace,
                install.model_namespace
            ),
            (1, 7, 3)
        );
        assert_eq!(
            observer.ledger.sources[&install.reservation.acquisition].kind,
            SourceKind::HostBackedLogical
        );
        assert_eq!(
            observer.ledger.reservations[&install].origin,
            InstallOrigin::PredictorV2Sidecar
        );
        let s = observer.snapshot().unwrap();
        assert_eq!(
            (
                s.emitted,
                s.source_completed,
                s.reservations_live,
                s.direct_matching_demand_credits
            ),
            (1, 1, 1, 0)
        );
        observer.p1j_published(install).unwrap();
        assert_eq!(observer.snapshot().unwrap().available_installs, 1);
    }
    #[test]
    fn p1j_matching_clean_target_credit_is_once_and_survives_retirement() {
        let (mut observer, id) = p1j_observer_fixture();
        let install = observer.p1j_acquired(id).unwrap();
        observer.p1j_published(install).unwrap();
        observer.observe_completed_position(
            id.candidate.request,
            id.candidate.target_position,
            id.candidate.model,
            &vec![(8..16).collect(); 48],
        );
        observer
            .p1j_finish(id, install, P1jTerminal::UsedMatching)
            .unwrap();
        let snapshot = observer.snapshot().unwrap();
        assert_eq!(
            (
                snapshot.direct_matching_demand_credits,
                snapshot.terminal_predictions,
                snapshot.available_installs
            ),
            (1, 1, 0)
        );
        assert!(observer.ledger.current.is_empty());
        assert!(observer
            .p1j_finish(id, install, P1jTerminal::UsedMatching)
            .is_err());
        assert_eq!(
            observer.snapshot().unwrap().direct_matching_demand_credits,
            1
        );
    }
    #[test]
    fn p1j_wrong_target_retires_unused_without_credit() {
        let (mut observer, id) = p1j_observer_fixture();
        let install = observer.p1j_acquired(id).unwrap();
        observer.p1j_published(install).unwrap();
        observer.observe_completed_position(
            id.candidate.request,
            id.candidate.target_position,
            id.candidate.model,
            &vec![(0..8).collect(); 48],
        );
        observer
            .p1j_finish(id, install, P1jTerminal::Unused)
            .unwrap();
        let s = observer.snapshot().unwrap();
        assert_eq!(
            (
                s.terminal_predictions,
                s.direct_matching_demand_credits,
                s.available_installs
            ),
            (1, 0, 0)
        );
        assert_eq!(
            observer.ledger.predictions[&id.prediction()].stage,
            PredictionStage::Terminal(TerminalReason::EvictedUnused)
        );
    }
    #[test]
    fn p1j_incomplete_accounting_stops_admission_and_keeps_reconciliation() {
        let (mut observer, id) = p1j_observer_fixture();
        observer.mark_incomplete(AccountingError::Capacity);
        assert!(!observer.p1j_ready());
        assert!(observer.p1j_acquired(id).is_err());
        assert_eq!(observer.snapshot().unwrap().emitted, 0);
    }
    #[test]
    fn p1j_cancelled_source_reservation_is_terminal_without_fabricated_install() {
        let (mut observer, id) = p1j_observer_fixture();
        let install = observer.p1j_acquired(id).unwrap();
        observer
            .p1j_finish(id, install, P1jTerminal::Cancelled)
            .unwrap();
        let s = observer.snapshot().unwrap();
        assert_eq!(
            (
                s.source_completed,
                s.reservations_aborted,
                s.install_owners,
                s.terminal_predictions
            ),
            (1, 1, 0, 1)
        );
    }
    #[test]
    fn p1j_same_generation_distinct_epoch_is_a_new_real_install() {
        let (mut observer, id) = p1j_observer_fixture();
        let install = observer.p1j_acquired(id).unwrap();
        observer.p1j_published(install).unwrap();
        observer
            .p1j_finish(id, install, P1jTerminal::Unused)
            .unwrap();
        // Exercise the actual ledger's epoch rule independently of scoring.
        let mut next = id.prediction();
        next.candidate_sequence += 1;
        observer.ledger.predictions.insert(
            next,
            PredictionRecord {
                stage: PredictionStage::Emitted,
                admission: None,
                acquisition: None,
                install: None,
                consumed_by: None,
                causal: true,
            },
        );
        observer
            .ledger
            .admit(next, AdmissionDisposition::Accepted)
            .unwrap();
        let source = observer
            .ledger
            .request_source(next, SourceKind::HostBackedLogical)
            .unwrap();
        observer
            .ledger
            .complete_source(source, SourceState::Completed)
            .unwrap();
        let mut location = id.location();
        location.slot_epoch += 1;
        let next_install = observer
            .ledger
            .reserve(next, location, InstallOrigin::PredictorV2Sidecar)
            .unwrap();
        assert_ne!(next_install, install);
        assert_eq!(next_install.logical_generation, install.logical_generation);
        assert_eq!(observer.snapshot().unwrap().reservations, 2);
    }
    #[test]
    fn p1j_recovery_shadow_excludes_sidecar_from_installs_victims_and_capacity() {
        let (observer, id) = p1j_observer_fixture();
        let namespace = observer.temporal().unwrap().namespace();
        let initial = p1e::PhysicalSnapshot {
            namespace,
            residents: std::array::from_fn(|i| {
                Some(p1e::Resident {
                    expert: i as u32,
                    generation: 1,
                    bank: 0,
                    slot: i as u32,
                    epoch: 1,
                })
            }),
        };
        let mut shadow = p1e::Shadow::new(initial).unwrap();
        let sidecar = p1e::Resident {
            expert: 8,
            generation: 11,
            bank: 1,
            slot: 0,
            epoch: id.epoch,
        };
        // Seven ordinary hits + sidecar; no ordinary install or eviction.
        shadow.demand_with_sidecar(&[0, 1, 2, 3, 4, 5, 6, 8], Some(sidecar));
        let reordered = p1e::PhysicalSnapshot {
            namespace,
            residents: [7, 0, 1, 2, 3, 4, 5, 6].map(|e| Some(initial.residents[e].unwrap())),
        };
        let evidence = shadow.evidence(reordered).unwrap();
        assert_eq!(
            (evidence.committed_installs, evidence.physical_victims),
            (0, 0)
        );
        assert_eq!(evidence.snapshot.residents.iter().flatten().count(), 8);
        assert!(!evidence
            .snapshot
            .residents
            .iter()
            .flatten()
            .any(|r| r.expert == 8));
    }

    fn request(sequence: u64) -> RequestIdentity {
        RequestIdentity {
            runtime_namespace: 1,
            request_sequence: sequence,
            phase: RequestPhase::Fixture,
            phase_run_index: 0,
        }
    }
    fn model() -> ModelMetadata {
        ModelMetadata {
            num_layers: 3,
            num_experts: 8,
            top_k: 2,
        }
    }
    fn position(n: usize) -> PositionIdentity {
        PositionIdentity::from_prompt_length(n, 1).unwrap()
    }
    fn point(r: RequestIdentity, n: usize, layer: usize) -> RoutePoint {
        RoutePoint {
            request: r,
            position: position(n),
            layer,
        }
    }
    fn location() -> InstallLocation {
        InstallLocation {
            executor_namespace: 7,
            model_namespace: 11,
            logical_generation: 1,
            bank: 0,
            slot: 4,
            slot_epoch: 1,
        }
    }
    fn checked(ledger: &LifecycleLedger) -> ReconciliationSnapshot {
        let snap = ledger.reconcile().unwrap();
        assert_eq!(
            snap.emitted,
            snap.live_predictions + snap.terminal_predictions
        );
        assert_eq!(
            snap.emitted,
            snap.admission_pending + snap.accepted + snap.rejected + snap.skipped
        );
        assert_eq!(
            snap.source_leaders,
            snap.source_live + snap.source_completed + snap.source_failed + snap.source_cancelled
        );
        assert_eq!(
            snap.reservations,
            snap.reservations_live + snap.reservations_committed + snap.reservations_aborted
        );
        assert_eq!(snap.install_owners, snap.reservations_committed);
        // All categories, including payload-bearing skip reasons, serialize in order.
        assert_eq!(
            serde_json::to_vec(&snap).unwrap(),
            serde_json::to_vec(&snap).unwrap()
        );
        snap
    }
    macro_rules! step {
        ($ledger:ident, $op:expr) => {{
            let value = $op.unwrap();
            checked(&$ledger);
            value
        }};
    }
    macro_rules! rejected {
        ($ledger:ident, $op:expr) => {{
            assert!($op.is_err());
            assert!(checked(&$ledger).incomplete.is_some());
        }};
    }
    fn emitted() -> (LifecycleLedger, PredictionIdentity) {
        let mut ledger = LifecycleLedger::new(request(1), model(), 256).unwrap();
        step!(
            ledger,
            ledger.observe_route(point(request(1), 0, 0), &[1, 2])
        );
        let ids = step!(
            ledger,
            ledger.freeze_candidates(
                point(request(1), 0, 0),
                point(request(1), 0, 1),
                &[3],
                PredictorSource::CpuFixture,
                1
            )
        );
        (ledger, ids[0])
    }
    fn accepted() -> (LifecycleLedger, PredictionIdentity) {
        let (mut ledger, id) = emitted();
        step!(ledger, ledger.admit(id, AdmissionDisposition::Accepted));
        (ledger, id)
    }
    fn available(
        origin: InstallOrigin,
    ) -> (LifecycleLedger, PredictionIdentity, PhysicalInstallIdentity) {
        let (mut ledger, id) = accepted();
        let source = step!(ledger, ledger.request_source(id, SourceKind::SyntheticRead));
        step!(
            ledger,
            ledger.complete_source(source, SourceState::Completed)
        );
        let install = step!(ledger, ledger.reserve(id, location(), origin));
        step!(ledger, ledger.finish_reservation(install, None));
        step!(ledger, ledger.make_available(install));
        (ledger, id, install)
    }
    fn demand(id: PredictionIdentity) -> DemandIdentity {
        DemandIdentity {
            request: id.request,
            target_position: id.target_position,
            layer: id.target_layer,
            expert_local_id: id.expert_local_id,
            demand_sequence: 1,
        }
    }

    #[test]
    fn concurrent_requests_keep_predecessors_separate() {
        // Old shared-ring trap: B0, A0, B1, A1, A2, B2. A0 -> B1
        // looks adjacent and has identical expert sets, but is a different request.
        // Threads exist only in this CPU test, never in the production module.
        use std::sync::{Arc, Barrier};
        let barrier = Arc::new(Barrier::new(2));
        let schedule = [2, 1, 2, 1, 1, 2];
        let workers: Vec<_> = [1, 2]
            .into_iter()
            .map(|sequence| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut observer =
                        RequestObserver::new(request(sequence), model(), 256).unwrap();
                    let mut layer = 0;
                    for turn in schedule {
                        barrier.wait();
                        if turn == sequence {
                            observer
                                .observe_layer(point(request(sequence), 0, layer), &[1, 2])
                                .unwrap();
                            assert_eq!(
                                observer.predecessor.as_ref().unwrap().point.request,
                                request(sequence)
                            );
                            layer += 1;
                        }
                        barrier.wait();
                    }
                    assert_eq!(observer.ledger.observations.len(), 3);
                    assert!(observer
                        .ledger
                        .observations
                        .keys()
                        .all(|p| p.request == request(sequence)));
                    assert_eq!(observer.transition_counts.len(), 8);
                    assert!(observer.transition_counts.values().all(|count| *count == 1));
                    assert_eq!(observer.snapshot().unwrap().emitted, 0);
                    observer
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn request_and_position_distinguish_same_expert() {
        let (mut ledger, original) = emitted();
        let next = step!(
            ledger,
            ledger.freeze_candidates(
                point(request(1), 0, 0),
                point(request(1), 0, 1),
                &[3],
                PredictorSource::CpuFixture,
                1
            )
        )[0];
        assert_ne!(next.prediction_generation, original.prediction_generation);
        assert_ne!(next.candidate_sequence, original.candidate_sequence);
        let mut variants = BTreeSet::from([original, next]);
        let mut id = original;
        id.request = request(2);
        assert!(variants.insert(id));
        id = original;
        id.request.runtime_namespace = 2;
        assert!(variants.insert(id));
        id = original;
        id.request.phase = RequestPhase::Measured;
        assert!(variants.insert(id));
        id = original;
        id.request.phase_run_index = 1;
        assert!(variants.insert(id));
        id = original;
        id.source_position = position(1);
        id.target_position = position(1);
        assert!(variants.insert(id));
        id = original;
        id.source_position = PositionIdentity::from_prompt_length(1, 2).unwrap();
        id.target_position = id.source_position;
        assert!(variants.insert(id));
        id = original;
        id.predictor_revision = 2;
        assert!(variants.insert(id));
        for id in variants {
            let bytes = serde_json::to_vec(&id).unwrap();
            let decoded: PredictionIdentity = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(id, decoded);
            assert_eq!(bytes, serde_json::to_vec(&decoded).unwrap());
        }
    }

    #[test]
    fn warmup_measured_and_request_restart_reset_history() {
        for (sequence, phase, run) in [
            (1, RequestPhase::Warmup, 0),
            (2, RequestPhase::Measured, 0),
            (3, RequestPhase::Measured, 1),
            (4, RequestPhase::Measured, 1),
        ] {
            let mut identity = request(sequence);
            identity.phase = phase;
            identity.phase_run_index = run;
            let mut observer = RequestObserver::new(identity, model(), 256).unwrap();
            assert!(observer.predecessor.is_none());
            assert!(observer.transition_counts.is_empty());
            observer.observe_completed_position(
                identity,
                position(0),
                model(),
                &vec![vec![1, 2]; 3],
            );
            assert_eq!(observer.completed_positions(), 1);
            assert_eq!(observer.transition_counts.len(), 8);
            observer.finish(sequence % 2 == 0);
            assert!(observer.predecessor.is_none());
            // Neither cancellation nor finish permits reuse of the same observer.
            observer.observe_completed_position(
                identity,
                position(0),
                model(),
                &vec![vec![1, 2]; 3],
            );
            assert_eq!(observer.completed_positions(), 1);
            assert!(observer.snapshot().unwrap().incomplete.is_some());
        }
        let mut observer = RequestObserver::new(request(5), model(), 256).unwrap();
        observer.observe_completed_position(request(5), position(0), model(), &vec![vec![1, 2]; 3]);
        observer.observe_completed_position(request(5), position(0), model(), &vec![vec![1, 2]; 3]);
        assert!(observer.predecessor.is_none());
        assert_eq!(
            observer.snapshot().unwrap().incomplete,
            Some(AccountingError::InvalidIdentity)
        );
    }

    #[test]
    fn position_layer_and_request_relationships_are_explicit() {
        assert!(adjacent(
            point(request(1), 0, 0),
            point(request(1), 0, 1),
            model()
        ));
        assert!(!adjacent(
            point(request(1), 0, 0),
            point(request(2), 0, 1),
            model()
        ));
        assert!(!adjacent(
            point(request(1), 0, 0),
            point(request(1), 0, 2),
            model()
        ));
        assert!(!adjacent(
            point(request(1), 0, 2),
            point(request(1), 0, 0),
            model()
        ));
        assert!(adjacent(
            point(request(1), 0, 2),
            point(request(1), 1, 0),
            model()
        ));
        assert!(!adjacent(
            point(request(1), 0, 2),
            point(request(1), 2, 0),
            model()
        ));
        let mut bad_decode = point(request(1), 1, 0);
        bad_decode.position.decode_index = Some(1);
        assert!(!adjacent(point(request(1), 0, 2), bad_decode, model()));
        let mut overflow = point(request(1), 0, 2);
        overflow.position.absolute_position = u64::MAX;
        assert!(!adjacent(overflow, point(request(1), 0, 0), model()));
    }

    #[test]
    fn prediction_frozen_before_target_observation() {
        let (mut ledger, _) = emitted();
        let frozen = step!(
            ledger,
            ledger.freeze_candidates(
                point(request(1), 0, 0),
                point(request(1), 0, 1),
                &[5, 3, 7],
                PredictorSource::CpuFixture,
                9
            )
        );
        let before = serde_json::to_vec(&frozen).unwrap();
        step!(
            ledger,
            ledger.observe_route(point(request(1), 0, 1), &[3, 5])
        );
        assert_eq!(before, serde_json::to_vec(&frozen).unwrap());
        assert_eq!(
            frozen.iter().map(|p| p.expert_local_id).collect::<Vec<_>>(),
            vec![5, 3, 7]
        );
        assert!(frozen
            .windows(2)
            .all(|w| w[0].candidate_sequence < w[1].candidate_sequence));
        let mut late = ledger.clone();
        rejected!(
            late,
            late.freeze_candidates(
                point(request(1), 0, 0),
                point(request(1), 0, 1),
                &[3],
                PredictorSource::CpuFixture,
                1
            )
        );
        let retrospective = step!(
            ledger,
            ledger.freeze_candidates(
                point(request(1), 0, 0),
                point(request(1), 0, 1),
                &[3],
                PredictorSource::CompletedPositionReplay,
                1
            )
        )[0];
        let mut cannot_admit = ledger.clone();
        rejected!(
            cannot_admit,
            cannot_admit.admit(retrospective, AdmissionDisposition::Accepted)
        );
        step!(
            ledger,
            ledger.admit(
                retrospective,
                AdmissionDisposition::Skipped(SkipReason::Retrospective)
            )
        );
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
        assert_eq!(checked(&ledger).source_leaders, 0);
    }

    #[test]
    fn lifecycle_reconciles_each_transition() {
        for kind in [
            SourceKind::SyntheticRead,
            SourceKind::SyntheticAlreadyAvailable,
        ] {
            for outcome in [
                SourceState::Completed,
                SourceState::Failed,
                SourceState::Cancelled,
            ] {
                let (mut ledger, id) = accepted();
                let source = step!(ledger, ledger.request_source(id, kind));
                step!(ledger, ledger.complete_source(source, outcome));
                let mut duplicate = ledger.clone();
                rejected!(duplicate, duplicate.complete_source(source, outcome));
                if outcome == SourceState::Completed {
                    for failure in [
                        None,
                        Some(TerminalReason::InstallFailed),
                        Some(TerminalReason::ReservationAborted),
                    ] {
                        let mut arm = ledger.clone();
                        let install = step!(
                            arm,
                            arm.reserve(id, location(), InstallOrigin::SpeculativeFixture)
                        );
                        step!(arm, arm.finish_reservation(install, failure));
                        if failure.is_none() {
                            step!(arm, arm.make_available(install));
                            let mut eviction = arm.clone();
                            step!(eviction, eviction.evict_unused(install));
                            step!(arm, arm.observe_route(id.target(), &[3, 4]));
                            step!(
                                arm,
                                arm.consume(demand(id), install, DemandStatus::CleanCommitted)
                            );
                            assert_eq!(checked(&arm).direct_matching_demand_credits, 1);
                        } else {
                            assert_eq!(checked(&arm).reservations_aborted, 1);
                        }
                    }
                } else {
                    assert_eq!(checked(&ledger).terminal_predictions, 1);
                }
            }
        }
        for disposition in [
            AdmissionDisposition::Rejected,
            AdmissionDisposition::Skipped(SkipReason::Policy),
        ] {
            let (mut ledger, id) = emitted();
            step!(ledger, ledger.admit(id, disposition));
            assert_eq!(checked(&ledger).terminal_predictions, 1);
        }
        for reason in [
            TerminalReason::Cancelled,
            TerminalReason::Superseded,
            TerminalReason::RequestEnded,
        ] {
            let (mut ledger, id) = emitted();
            step!(ledger, ledger.terminate(id, reason));
            assert_eq!(checked(&ledger).admission_pending, 0);
            let mut duplicate = ledger.clone();
            rejected!(duplicate, duplicate.terminate(id, reason));
        }
        let (mut ledger, _) = accepted();
        step!(ledger, ledger.end_request(false));
        assert_eq!(checked(&ledger).live_predictions, 0);
    }

    #[test]
    fn rejected_prediction_is_terminal_without_work() {
        for disposition in [
            AdmissionDisposition::Rejected,
            AdmissionDisposition::Skipped(SkipReason::Policy),
        ] {
            let (mut ledger, id) = emitted();
            step!(ledger, ledger.admit(id, disposition));
            step!(ledger, ledger.observe_route(id.target(), &[3, 4]));
            let snap = checked(&ledger);
            assert_eq!(
                (
                    snap.source_leaders,
                    snap.reservations,
                    snap.install_owners,
                    snap.direct_matching_demand_credits
                ),
                (0, 0, 0, 0)
            );
            let mut copy = ledger.clone();
            rejected!(copy, copy.request_source(id, SourceKind::SyntheticRead));
            let mut copy = ledger.clone();
            rejected!(
                copy,
                copy.reserve(id, location(), InstallOrigin::SpeculativeFixture)
            );
            let mut copy = ledger.clone();
            rejected!(copy, copy.admit(id, AdmissionDisposition::Accepted));
        }
    }

    #[test]
    fn unused_install_eviction_uses_exact_identity() {
        let (ledger, _, install) = available(InstallOrigin::SpeculativeFixture);
        let mut variants = Vec::new();
        let mut id = install;
        id.logical_generation += 1;
        variants.push(id);
        id = install;
        id.bank += 1;
        variants.push(id);
        id = install;
        id.slot += 1;
        variants.push(id);
        id = install;
        id.slot_epoch += 1;
        variants.push(id);
        id = install;
        id.executor_namespace += 1;
        variants.push(id);
        id = install;
        id.model_namespace += 1;
        variants.push(id);
        id = install;
        id.reservation.ticket_sequence += 1;
        variants.push(id);
        for wrong in variants {
            let mut copy = ledger.clone();
            rejected!(copy, copy.evict_unused(wrong));
            assert_eq!(copy.installs[&install].state, InstallState::Available);
        }
        let mut ledger = ledger;
        step!(ledger, ledger.evict_unused(install));
        assert_eq!(checked(&ledger).terminal_predictions, 1);
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
    }

    #[test]
    fn matching_demand_consumes_exact_install_once() {
        let (mut ledger, id, install) = available(InstallOrigin::SpeculativeFixture);
        step!(ledger, ledger.observe_route(id.target(), &[3, 4]));
        let correct = demand(id);
        let mut variants = Vec::new();
        let mut wrong = correct;
        wrong.request = request(2);
        variants.push(wrong);
        wrong = correct;
        wrong.target_position = position(1);
        variants.push(wrong);
        wrong = correct;
        wrong.layer = 2;
        variants.push(wrong);
        wrong = correct;
        wrong.expert_local_id = 4;
        variants.push(wrong);
        for wrong in variants {
            let mut copy = ledger.clone();
            rejected!(
                copy,
                copy.consume(wrong, install, DemandStatus::CleanCommitted)
            );
            assert_eq!(checked(&copy).direct_matching_demand_credits, 0);
        }
        for status in [DemandStatus::FailedAttempt, DemandStatus::ReplayedSegment] {
            let mut copy = ledger.clone();
            rejected!(copy, copy.consume(correct, install, status));
        }
        let mut wrong_install = install;
        wrong_install.slot_epoch += 1;
        let mut copy = ledger.clone();
        rejected!(
            copy,
            copy.consume(correct, wrong_install, DemandStatus::CleanCommitted)
        );
        step!(
            ledger,
            ledger.consume(correct, install, DemandStatus::CleanCommitted)
        );
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 1);
        rejected!(
            ledger,
            ledger.consume(correct, install, DemandStatus::CleanCommitted)
        );
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 1);
        let (mut restoration, restored_prediction, restored_install) =
            available(InstallOrigin::RestorationFixture);
        step!(
            restoration,
            restoration.observe_route(restored_prediction.target(), &[3, 4])
        );
        rejected!(
            restoration,
            restoration.consume(
                demand(restored_prediction),
                restored_install,
                DemandStatus::CleanCommitted
            )
        );
        assert_eq!(checked(&restoration).direct_matching_demand_credits, 0);
    }

    #[test]
    fn coalesced_followers_never_multiply_work_or_payoff() {
        let (mut ledger, first) = accepted();
        let second = step!(
            ledger,
            ledger.freeze_candidates(
                point(request(1), 0, 0),
                first.target(),
                &[3],
                PredictorSource::CpuFixture,
                2
            )
        )[0];
        step!(ledger, ledger.admit(second, AdmissionDisposition::Accepted));
        let source = step!(
            ledger,
            ledger.request_source(first, SourceKind::SyntheticRead)
        );
        step!(ledger, ledger.join_source(second, source));
        step!(
            ledger,
            ledger.complete_source(source, SourceState::Completed)
        );
        let install = step!(
            ledger,
            ledger.reserve(first, location(), InstallOrigin::SpeculativeFixture)
        );
        step!(ledger, ledger.finish_reservation(install, None));
        step!(ledger, ledger.make_available(install));
        step!(ledger, ledger.observe_route(first.target(), &[3, 4]));
        step!(
            ledger,
            ledger.consume(demand(first), install, DemandStatus::CleanCommitted)
        );
        let snap = checked(&ledger);
        assert_eq!(
            (
                snap.emitted,
                snap.terminal_predictions,
                snap.source_leaders,
                snap.source_followers,
                snap.install_owners,
                snap.install_followers,
                snap.direct_matching_demand_credits
            ),
            (2, 2, 1, 1, 1, 1, 1)
        );
        assert_eq!(
            ledger.predictions[&first].consumed_by,
            ledger.predictions[&second].consumed_by
        );
        rejected!(
            ledger,
            ledger.consume(demand(second), install, DemandStatus::CleanCommitted)
        );
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 1);
    }

    #[test]
    fn cancellation_does_not_erase_inflight_work() {
        for reason in [TerminalReason::Cancelled, TerminalReason::Superseded] {
            let (mut ledger, id) = accepted();
            let source = step!(ledger, ledger.request_source(id, SourceKind::SyntheticRead));
            step!(ledger, ledger.terminate(id, reason));
            assert_eq!(checked(&ledger).source_live, 1);
            step!(
                ledger,
                ledger.complete_source(source, SourceState::Completed)
            );
            assert_eq!(checked(&ledger).source_completed, 1);
            assert_eq!(
                ledger.predictions[&id].stage,
                PredictionStage::Terminal(reason)
            );
            assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
        }
        let (mut ledger, id) = accepted();
        let source = step!(ledger, ledger.request_source(id, SourceKind::SyntheticRead));
        step!(
            ledger,
            ledger.complete_source(source, SourceState::Completed)
        );
        let install = step!(
            ledger,
            ledger.reserve(id, location(), InstallOrigin::SpeculativeFixture)
        );
        step!(ledger, ledger.end_request(true));
        assert_eq!(checked(&ledger).reservations_live, 1);
        step!(ledger, ledger.finish_reservation(install, None));
        step!(ledger, ledger.make_available(install));
        assert_eq!(
            ledger.predictions[&id].stage,
            PredictionStage::Terminal(TerminalReason::Cancelled)
        );
        assert_eq!(checked(&ledger).install_owners, 1);
        step!(ledger, ledger.evict_unused(install));
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
    }

    #[test]
    fn observation_only_has_no_movement_capabilities() {
        let mut observer = RequestObserver::new(request(1), model(), 256).unwrap();
        for n in 0..4 {
            observer.observe_completed_position(
                request(1),
                position(n),
                model(),
                &vec![vec![1, 2]; 3],
            );
            let snap = observer.snapshot().unwrap();
            assert_eq!(snap.incomplete, None);
            assert_eq!(
                (
                    snap.emitted,
                    snap.source_leaders,
                    snap.reservations,
                    snap.install_owners,
                    snap.direct_matching_demand_credits
                ),
                (0, 0, 0, 0, 0)
            );
        }
        assert_eq!(observer.completed_positions(), 4);
        // Production dependencies are deliberately limited to serde and ordered
        // CPU collections. The same file also builds as a standalone CPU crate.
        let production = include_str!("predictor_v2.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "use crate::",
            "wgpu::",
            "tokio::",
            "std::fs",
            "std::thread",
            "unsafe",
            "async fn",
        ] {
            assert!(
                !production.contains(forbidden),
                "movement-capable dependency: {forbidden}"
            );
        }
    }

    #[test]
    fn overflow_and_capacity_fail_accounting_only() {
        let (ledger, _) = emitted();
        for which in 0..3 {
            let mut copy = ledger.clone();
            match which {
                0 => copy.generation = u64::MAX,
                1 => copy.candidate_sequence = u64::MAX,
                _ => copy.event = u64::MAX,
            }
            let before = copy.predictions.clone();
            rejected!(
                copy,
                copy.freeze_candidates(
                    point(request(1), 0, 0),
                    point(request(1), 0, 1),
                    &[4],
                    PredictorSource::CpuFixture,
                    1
                )
            );
            assert_eq!(
                copy.predictions.keys().collect::<Vec<_>>(),
                before.keys().collect::<Vec<_>>()
            );
            assert_eq!(checked(&copy).incomplete, Some(AccountingError::Overflow));
        }
        let (mut copy, id) = accepted();
        copy.acquisition_sequence = u64::MAX;
        rejected!(copy, copy.request_source(id, SourceKind::SyntheticRead));
        assert!(copy.sources.is_empty());
        let (mut copy, id) = accepted();
        let source = step!(copy, copy.request_source(id, SourceKind::SyntheticRead));
        step!(copy, copy.complete_source(source, SourceState::Completed));
        copy.ticket_sequence = u64::MAX;
        rejected!(
            copy,
            copy.reserve(id, location(), InstallOrigin::SpeculativeFixture)
        );
        assert!(copy.reservations.is_empty());
        let (mut copy, id, install) = available(InstallOrigin::SpeculativeFixture);
        step!(copy, copy.observe_route(id.target(), &[3, 4]));
        copy.demand_sequence = u64::MAX;
        rejected!(
            copy,
            copy.consume(demand(id), install, DemandStatus::CleanCommitted)
        );
        assert!(copy.demands.is_empty());
        let (mut full, _) = emitted();
        full.capacity = 1;
        rejected!(
            full,
            full.freeze_candidates(
                point(request(1), 0, 0),
                point(request(1), 0, 1),
                &[4],
                PredictorSource::CpuFixture,
                1
            )
        );
        assert_eq!(checked(&full).emitted, 1);
        let mut observer = RequestObserver::new(request(1), model(), 2).unwrap();
        let routes = vec![vec![1, 2]; 3];
        let original = routes.clone();
        observer.observe_completed_position(request(1), position(0), model(), &routes);
        assert_eq!(routes, original);
        assert_eq!(
            observer.snapshot().unwrap().incomplete,
            Some(AccountingError::Capacity)
        );
        assert_eq!(observer.completed_positions(), 0);
        let mut observer = RequestObserver::new(request(1), model(), 256).unwrap();
        observer.observe_completed_position(request(1), position(0), model(), &routes);
        for value in observer.transition_counts.values_mut() {
            *value = u64::MAX;
        }
        observer.observe_completed_position(request(1), position(1), model(), &routes);
        assert_eq!(observer.snapshot().unwrap().incomplete, None); // distinct prompt/decode keys
        observer.observe_completed_position(request(1), position(2), model(), &routes);
        assert_eq!(observer.snapshot().unwrap().incomplete, None);
        for value in observer.transition_counts.values_mut() {
            *value = u64::MAX;
        }
        observer.observe_completed_position(request(1), position(3), model(), &routes);
        assert_eq!(
            observer.snapshot().unwrap().incomplete,
            Some(AccountingError::Overflow)
        );
    }

    #[test]
    fn reinstallation_and_restoration_cannot_credit_original_install() {
        let (mut ledger, original_prediction, original_install) =
            available(InstallOrigin::SpeculativeFixture);
        let replacement = step!(
            ledger,
            ledger.freeze_candidates(
                point(request(1), 0, 0),
                original_prediction.target(),
                &[3],
                PredictorSource::CpuFixture,
                2
            )
        )[0];
        step!(
            ledger,
            ledger.admit(replacement, AdmissionDisposition::Accepted)
        );
        let source = step!(
            ledger,
            ledger.request_source(replacement, SourceKind::SyntheticAlreadyAvailable)
        );
        step!(
            ledger,
            ledger.complete_source(source, SourceState::Completed)
        );
        let mut next_location = location();
        next_location.logical_generation = 2;
        next_location.slot = 5;
        let restored = step!(
            ledger,
            ledger.reserve(
                replacement,
                next_location,
                InstallOrigin::RestorationFixture
            )
        );
        let mut simultaneous = ledger.clone();
        rejected!(
            simultaneous,
            simultaneous.finish_reservation(restored, None)
        );
        step!(ledger, ledger.evict_unused(original_install));
        step!(ledger, ledger.finish_reservation(restored, None));
        step!(ledger, ledger.make_available(restored));
        step!(
            ledger,
            ledger.observe_route(original_prediction.target(), &[3, 4])
        );
        let mut stale = ledger.clone();
        rejected!(
            stale,
            stale.consume(
                demand(original_prediction),
                original_install,
                DemandStatus::CleanCommitted
            )
        );
        rejected!(
            ledger,
            ledger.consume(demand(replacement), restored, DemandStatus::CleanCommitted)
        );
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
    }

    #[test]
    fn invalid_transitions_do_not_mutate_work_or_reuse_identifiers() {
        let (mut ledger, id) = accepted();
        let source = step!(ledger, ledger.request_source(id, SourceKind::SyntheticRead));
        let mut backwards = ledger.clone();
        rejected!(
            backwards,
            backwards.admit(id, AdmissionDisposition::Accepted)
        );
        let mut duplicate = ledger.clone();
        rejected!(
            duplicate,
            duplicate.request_source(id, SourceKind::SyntheticRead)
        );
        let mut premature = ledger.clone();
        rejected!(
            premature,
            premature.reserve(id, location(), InstallOrigin::SpeculativeFixture)
        );
        step!(
            ledger,
            ledger.complete_source(source, SourceState::Completed)
        );
        let install = step!(
            ledger,
            ledger.reserve(id, location(), InstallOrigin::SpeculativeFixture)
        );
        let mut premature = ledger.clone();
        rejected!(premature, premature.make_available(install));
        step!(ledger, ledger.finish_reservation(install, None));
        let mut duplicate = ledger.clone();
        rejected!(duplicate, duplicate.finish_reservation(install, None));
        // Truth that arrived before availability cannot create causal payoff.
        step!(ledger, ledger.observe_route(id.target(), &[3, 4]));
        step!(ledger, ledger.make_available(install));
        let mut duplicate = ledger.clone();
        rejected!(duplicate, duplicate.make_available(install));
        rejected!(
            ledger,
            ledger.consume(demand(id), install, DemandStatus::CleanCommitted)
        );
        assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
    }

    #[test]
    fn closed_request_preserves_late_work_success_failure_and_cancel() {
        for outcome in [
            SourceState::Completed,
            SourceState::Failed,
            SourceState::Cancelled,
        ] {
            let (mut ledger, id) = accepted();
            let source = step!(ledger, ledger.request_source(id, SourceKind::SyntheticRead));
            step!(ledger, ledger.end_request(false));
            step!(ledger, ledger.complete_source(source, outcome));
            assert_eq!(
                ledger.predictions[&id].stage,
                PredictionStage::Terminal(TerminalReason::RequestEnded)
            );
            assert_eq!(checked(&ledger).source_live, 0);
            assert_eq!(checked(&ledger).source_leaders, 1);
            assert_eq!(checked(&ledger).direct_matching_demand_credits, 0);
        }
    }
}
