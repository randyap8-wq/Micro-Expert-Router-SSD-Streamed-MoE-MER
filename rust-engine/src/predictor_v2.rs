//! Predictor-v2 P0: bounded, request-owned CPU attribution, never scheduling.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
                        && old.logical_generation >= install.logical_generation
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
                    && record.origin == InstallOrigin::SpeculativeFixture
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
/// trains descriptive transitions only; it emits no predictions or utility.
#[derive(Debug)]
pub(crate) struct RequestObserver {
    ledger: LifecycleLedger,
    predecessor: Option<Predecessor>,
    transition_counts: BTreeMap<TransitionKey, u64>,
    completed_positions: u64,
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
        self.ledger.mark_incomplete(error);
        self.predecessor = None;
    }
    pub fn finish(&mut self, cancelled: bool) {
        let _ = self.ledger.end_request(cancelled);
        self.predecessor = None;
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

#[cfg(test)]
mod tests {
    use super::*;

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
