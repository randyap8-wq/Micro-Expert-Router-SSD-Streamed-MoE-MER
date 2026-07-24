//! Prompt 2 Phase 4B routing, speculative-I/O lifecycle, and critical-path trace.
//!
//! This module is deliberately opt-in. When an [`Engine`](crate::engine::Engine)
//! has no [`Phase4bTrace`] installed, the runtime pays only an `Option` check:
//! no event allocation, JSON serialization, writer lock, or diagnostic state
//! mutation occurs.

use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub const PHASE4B_SCHEMA_NAME: &str = "mer-prompt2-phase4b-routing-trace";
pub const PHASE4B_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictorArm {
    FirstOrderMarkov,
    SecondOrderMarkov,
    FallbackPriorFill,
    Affinity,
    NeuralSpeculator,
    Locality,
    Combined,
    Other,
}

impl PredictorArm {
    fn label(self) -> &'static str {
        match self {
            Self::FirstOrderMarkov => "first_order_markov",
            Self::SecondOrderMarkov => "second_order_markov",
            Self::FallbackPriorFill => "fallback_prior_fill",
            Self::Affinity => "affinity",
            Self::NeuralSpeculator => "neural_speculator",
            Self::Locality => "locality",
            Self::Combined => "combined",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialLookupClass {
    OrdinaryResident,
    ReadyPrefetchedResident,
    SpeculativeReadInFlight,
    OrdinaryMiss,
}

impl InitialLookupClass {
    fn label(self) -> &'static str {
        match self {
            Self::OrdinaryResident => "ordinary_resident",
            Self::ReadyPrefetchedResident => "ready_prefetched_resident",
            Self::SpeculativeReadInFlight => "speculative_read_in_flight",
            Self::OrdinaryMiss => "ordinary_miss",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadClass {
    Demand,
    Speculative,
}

impl ReadClass {
    fn label(self) -> &'static str {
        match self {
            Self::Demand => "demand",
            Self::Speculative => "speculative",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CandidateTicket {
    pub lifecycle_id: u64,
    pub request_id: u64,
    pub token_index: u64,
    pub source_layer: u32,
    pub expected_target_layer: Option<u32>,
    pub expert_id: u32,
    pub arm: PredictorArm,
    pub rank: usize,
    pub score: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct LookupTicket {
    pub lookup_id: u64,
    pub request_id: u64,
    pub token_index: u64,
    pub layer: u32,
    pub routed_slot: usize,
    pub expert_id: u32,
    pub classification: InitialLookupClass,
    lifecycle_id: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReadTicket {
    read_id: u64,
    lifecycle_id: Option<u64>,
    lookup_id: Option<u64>,
    class: ReadClass,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LifecycleSnapshot {
    pub candidate_generated: u64,
    pub filtered_resident: u64,
    pub filtered_current_target: u64,
    pub filtered_global_in_flight: u64,
    pub rejected_governor: u64,
    pub rejected_concurrency_limit: u64,
    pub rejected_shadow_pool_exhaustion: u64,
    pub admitted: u64,
    pub task_spawned: u64,
    pub cache_race_found_resident: u64,
    pub singleflight_follower: u64,
    pub singleflight_leader: u64,
    pub physical_read_issued: u64,
    pub physical_read_completed: u64,
    pub physical_read_failed: u64,
    pub physical_read_inflight_at_sample: u64,
    pub publication_attempted: u64,
    pub published: u64,
    pub publication_rejected: u64,
    pub completion_not_yet_published_at_sample: u64,
    pub first_use: u64,
    pub evicted_before_first_use: u64,
    pub still_resident_unused_at_sample: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct InitialLookupSnapshot {
    pub ordinary_resident: u64,
    pub ready_prefetched_resident: u64,
    pub speculative_read_in_flight: u64,
    pub ordinary_miss: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TimelinessSnapshot {
    pub ready_before_lookup: u64,
    pub in_flight_completed_before_unrelated_misses: u64,
    pub in_flight_completed_as_final_straggler: u64,
    pub in_flight_completed_after_another_demand_read: u64,
    pub published_but_evicted_before_lookup: u64,
    pub published_but_never_requested: u64,
    pub completed_but_publication_rejected: u64,
    pub failed: u64,
    pub still_in_flight_at_sample: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CriticalPathSnapshot {
    pub ready_prefetch_removed_miss: u64,
    pub prefetch_join_avoided_duplicate_physical_read: u64,
    pub prefetched_expert_used_but_noncritical: u64,
    pub prefetched_expert_was_final_straggler: u64,
    pub prefetched_expert_shortened_final_straggler_controlled_replay_only: u64,
    pub layer_with_no_missing_experts: u64,
    pub layer_with_one_or_more_missing_experts: u64,
    pub layer_where_all_missing_experts_were_speculative_joins: u64,
    pub layer_blocked_by_ordinary_demand_read: u64,
    pub final_straggler_source: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ReadOverlapSnapshot {
    pub demand_physical_reads: u64,
    pub speculative_physical_reads: u64,
    pub active_speculative_reads_observed_at_demand_issue: u64,
    pub peak_demand_reads_in_flight: u64,
    pub peak_speculative_reads_in_flight: u64,
    pub peak_total_reads_in_flight: u64,
    /// Live overlap is observational. Only deterministic replay/tests may
    /// populate a causal delay count.
    pub causally_delayed_demand_reads: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PredictorArmSnapshot {
    pub candidate_lists: u64,
    pub candidates_generated: u64,
    pub unique_candidates_generated: u64,
    pub candidates_in_correct_next_layer: u64,
    pub candidates_in_actual_next_top8: u64,
    pub actual_next_top8_slots: u64,
    pub timely_candidates: u64,
    pub ready_before_lookup_candidates: u64,
    pub candidate_precision: f64,
    pub candidate_set_recall: f64,
    pub correct_layer_precision: f64,
    pub timely_precision: f64,
    pub timely_recall: f64,
    pub ready_before_lookup_precision: f64,
    pub ready_before_lookup_recall: f64,
    pub candidate_rank_distribution: BTreeMap<usize, u64>,
    pub wrong_layer_candidates: u64,
    pub fallback_prior_fill_candidates: u64,
    pub tied_probability_candidates: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PredictorQualitySnapshot {
    pub by_arm: BTreeMap<String, PredictorArmSnapshot>,
    pub candidates_lost_after_truncation_to_resident_filter: u64,
    pub candidates_lost_after_truncation_to_global_inflight_filter: u64,
    pub cross_request_history_use_count: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Phase4bSnapshot {
    pub schema_name: &'static str,
    pub schema_version: u32,
    pub trace_path: String,
    pub max_events: u64,
    pub events_written: u64,
    pub events_dropped: u64,
    pub trace_truncated: bool,
    pub trace_write_failed: bool,
    pub lifecycle_reconciliation_passed: bool,
    pub lifecycle_reconciliation_errors: Vec<String>,
    pub lifecycle: LifecycleSnapshot,
    pub initial_lookup: InitialLookupSnapshot,
    pub timeliness: TimelinessSnapshot,
    pub layer_critical_path: CriticalPathSnapshot,
    pub predictor_quality: PredictorQualitySnapshot,
    pub read_overlap: ReadOverlapSnapshot,
    pub request_boundary_contamination_count: u64,
}

#[derive(Clone, Debug)]
struct RequestContext {
    fixture_id: String,
    repetition_index: usize,
    measured: bool,
}

#[derive(Clone, Debug)]
struct LifecycleState {
    ticket: CandidateTicket,
    admitted: bool,
    task_spawned: bool,
    singleflight_leader: bool,
    read_issued_ns: Option<u64>,
    read_completed_ns: Option<u64>,
    read_failed: bool,
    publication_attempted: bool,
    published_ns: Option<u64>,
    publication_rejected: bool,
    first_use_ns: Option<u64>,
    evicted_ns: Option<u64>,
    evicted_miss_observed: bool,
}

impl LifecycleState {
    fn new(ticket: CandidateTicket) -> Self {
        Self {
            ticket,
            admitted: false,
            task_spawned: false,
            singleflight_leader: false,
            read_issued_ns: None,
            read_completed_ns: None,
            read_failed: false,
            publication_attempted: false,
            published_ns: None,
            publication_rejected: false,
            first_use_ns: None,
            evicted_ns: None,
            evicted_miss_observed: false,
        }
    }
}

#[derive(Clone, Debug)]
struct LookupState {
    ticket: LookupTicket,
    lookup_ns: u64,
    speculative_issue_ns: Option<u64>,
    speculative_completion_ns: Option<u64>,
    publication_ns: Option<u64>,
    demand_read_start_ns: Option<u64>,
    availability_ns: Option<u64>,
    consumption_ns: Option<u64>,
    layer_compute_start_ns: Option<u64>,
    timeliness_classified: bool,
}

#[derive(Clone, Debug)]
struct ReadState {
    class: ReadClass,
    expert_id: u32,
    requested_bytes: u64,
    actual_bytes: Option<u64>,
    issue_ns: u64,
    completion_ns: Option<u64>,
    failed: bool,
    active_demand_at_issue: u64,
    active_speculative_at_issue: u64,
}

#[derive(Clone, Debug)]
struct PendingPrediction {
    lifecycle_id: u64,
    arm: PredictorArm,
    expert_id: u32,
}

#[derive(Default)]
struct MutableState {
    requests: HashMap<u64, RequestContext>,
    lifecycles: HashMap<u64, LifecycleState>,
    expert_lifecycles: HashMap<u32, Vec<u64>>,
    lookups: HashMap<u64, LookupState>,
    reads: HashMap<u64, ReadState>,
    pending_predictions: HashMap<(u64, u32), Vec<PendingPrediction>>,
    lifecycle: LifecycleSnapshot,
    initial_lookup: InitialLookupSnapshot,
    timeliness: TimelinessSnapshot,
    critical_path: CriticalPathSnapshot,
    predictor: PredictorQualitySnapshot,
    active_demand: u64,
    active_speculative: u64,
    read_overlap: ReadOverlapSnapshot,
    request_boundary_contamination_count: u64,
}

struct WriterState {
    writer: BufWriter<File>,
    failed: bool,
}

/// Opt-in, bounded Phase 4B trace collector.
pub struct Phase4bTrace {
    origin: Instant,
    path: PathBuf,
    max_events: u64,
    next_event_id: AtomicU64,
    next_lifecycle_id: AtomicU64,
    next_lookup_id: AtomicU64,
    next_read_id: AtomicU64,
    next_request_id: AtomicU64,
    current_request_id: AtomicU64,
    events_written: AtomicU64,
    events_dropped: AtomicU64,
    truncated: AtomicBool,
    writer_failed: AtomicBool,
    writer: parking_lot::Mutex<WriterState>,
    state: parking_lot::Mutex<MutableState>,
}

impl Phase4bTrace {
    pub fn open(path: &Path, max_events: u64) -> io::Result<Self> {
        if max_events == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS must be greater than zero",
            ));
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        Ok(Self {
            origin: Instant::now(),
            path: path.to_path_buf(),
            max_events,
            next_event_id: AtomicU64::new(1),
            next_lifecycle_id: AtomicU64::new(1),
            next_lookup_id: AtomicU64::new(1),
            next_read_id: AtomicU64::new(1),
            next_request_id: AtomicU64::new(1),
            current_request_id: AtomicU64::new(0),
            events_written: AtomicU64::new(0),
            events_dropped: AtomicU64::new(0),
            truncated: AtomicBool::new(false),
            writer_failed: AtomicBool::new(false),
            writer: parking_lot::Mutex::new(WriterState {
                writer: BufWriter::new(file),
                failed: false,
            }),
            state: parking_lot::Mutex::new(MutableState::default()),
        })
    }

    #[inline]
    pub fn now_ns(&self) -> u64 {
        self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    #[inline]
    pub fn current_request_id(&self) -> u64 {
        self.current_request_id.load(Ordering::Acquire)
    }

    pub fn begin_request(&self, fixture_id: &str, repetition_index: usize, measured: bool) -> u64 {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.current_request_id.store(request_id, Ordering::Release);
        if self.accept_new_state() {
            self.state.lock().requests.insert(
                request_id,
                RequestContext {
                    fixture_id: fixture_id.to_string(),
                    repetition_index,
                    measured,
                },
            );
        }
        self.emit(
            "request_begin",
            json!({
                "request_id": request_id,
                "stream_id": request_id,
                "prompt_fixture_id": fixture_id,
                "benchmark_repetition_index": repetition_index,
                "measured": measured
            }),
        );
        request_id
    }

    pub fn end_request(&self, request_id: u64) {
        self.emit("request_end", json!({"request_id": request_id}));
        let _ = self.current_request_id.compare_exchange(
            request_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub fn record_routing(
        &self,
        request_id: u64,
        token_index: u64,
        layer: u32,
        global_experts: &[u32],
        local_experts: &[u32],
        routing_weights: &[f32],
        history_request_id: Option<u64>,
    ) {
        let now = self.now_ns();
        let (fixture_id, repetition_index) = {
            let mut state = self.state.lock();
            if history_request_id.is_some_and(|prior| prior != 0 && prior != request_id) {
                state.request_boundary_contamination_count =
                    state.request_boundary_contamination_count.saturating_add(1);
                state.predictor.cross_request_history_use_count = state
                    .predictor
                    .cross_request_history_use_count
                    .saturating_add(1);
            }
            let actual: HashSet<u32> = global_experts.iter().copied().collect();
            if let Some(pending) = state.pending_predictions.remove(&(request_id, layer)) {
                let mut by_arm: HashMap<PredictorArm, (HashSet<u32>, u64)> = HashMap::new();
                for candidate in pending {
                    let entry = by_arm
                        .entry(candidate.arm)
                        .or_insert_with(|| (HashSet::new(), 0));
                    entry.0.insert(candidate.expert_id);
                    if actual.contains(&candidate.expert_id) {
                        entry.1 = entry.1.saturating_add(1);
                        let arm = state
                            .predictor
                            .by_arm
                            .entry(candidate.arm.label().to_string())
                            .or_default();
                        arm.candidates_in_actual_next_top8 =
                            arm.candidates_in_actual_next_top8.saturating_add(1);
                    }
                    if let Some(lifecycle) = state.lifecycles.get(&candidate.lifecycle_id) {
                        let _ = lifecycle.ticket.lifecycle_id;
                    }
                }
                for (arm_key, (_unique, _hits)) in by_arm {
                    let arm = state
                        .predictor
                        .by_arm
                        .entry(arm_key.label().to_string())
                        .or_default();
                    arm.actual_next_top8_slots = arm
                        .actual_next_top8_slots
                        .saturating_add(actual.len() as u64);
                }
            }
            state
                .requests
                .get(&request_id)
                .map(|ctx| (ctx.fixture_id.clone(), ctx.repetition_index))
                .unwrap_or_else(|| ("unknown".to_string(), 0))
        };
        self.emit(
            "routing",
            json!({
                "request_id": request_id,
                "stream_id": request_id,
                "prompt_fixture_id": fixture_id,
                "benchmark_repetition_index": repetition_index,
                "token_index": token_index,
                "layer_index": layer,
                "ordered_local_topk_expert_ids": local_experts,
                "ordered_global_topk_expert_ids": global_experts,
                "ordered_routing_weights_or_scores": routing_weights,
                "routing_timestamp_ns": now,
                "history_request_id": history_request_id
            }),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn candidate_generated(
        &self,
        request_id: u64,
        token_index: u64,
        source_layer: u32,
        expected_target_layer: Option<u32>,
        context_experts: &[u32],
        arm: PredictorArm,
        rank: usize,
        score: f64,
        expert_id: u32,
        decoded_layer: Option<u32>,
        decoded_local_id: Option<u32>,
        belongs_to_expected_layer: Option<bool>,
        part_of_current_target: bool,
        already_resident: bool,
        globally_in_flight: bool,
    ) -> CandidateTicket {
        let lifecycle_id = self.next_lifecycle_id.fetch_add(1, Ordering::Relaxed);
        let ticket = CandidateTicket {
            lifecycle_id,
            request_id,
            token_index,
            source_layer,
            expected_target_layer,
            expert_id,
            arm,
            rank,
            score,
        };
        if self.accept_new_state() {
            let mut state = self.state.lock();
            state.lifecycle.candidate_generated =
                state.lifecycle.candidate_generated.saturating_add(1);
            let arm_stats = state
                .predictor
                .by_arm
                .entry(arm.label().to_string())
                .or_default();
            arm_stats.candidates_generated = arm_stats.candidates_generated.saturating_add(1);
            *arm_stats
                .candidate_rank_distribution
                .entry(rank)
                .or_default() += 1;
            match belongs_to_expected_layer {
                Some(true) => {
                    arm_stats.candidates_in_correct_next_layer =
                        arm_stats.candidates_in_correct_next_layer.saturating_add(1)
                }
                Some(false) => {
                    arm_stats.wrong_layer_candidates =
                        arm_stats.wrong_layer_candidates.saturating_add(1)
                }
                None => {}
            }
            if arm == PredictorArm::FallbackPriorFill {
                arm_stats.fallback_prior_fill_candidates =
                    arm_stats.fallback_prior_fill_candidates.saturating_add(1);
            }
            if let Some(target_layer) = expected_target_layer {
                state
                    .pending_predictions
                    .entry((request_id, target_layer))
                    .or_default()
                    .push(PendingPrediction {
                        lifecycle_id,
                        arm,
                        expert_id,
                    });
            }
            state
                .expert_lifecycles
                .entry(expert_id)
                .or_default()
                .push(lifecycle_id);
            state
                .lifecycles
                .insert(lifecycle_id, LifecycleState::new(ticket));
        }
        self.emit(
            "prediction_candidate",
            json!({
                "lifecycle_id": lifecycle_id,
                "request_id": request_id,
                "token_index": token_index,
                "source_layer": source_layer,
                "expected_target_layer": expected_target_layer,
                "predictor_context_expert_ids": context_experts,
                "predictor_arm": arm,
                "candidate_rank": rank,
                "candidate_probability_or_score": score,
                "global_expert_id": expert_id,
                "decoded_expert_layer": decoded_layer,
                "decoded_local_expert_id": decoded_local_id,
                "belongs_to_expected_next_layer_slice": belongs_to_expected_layer,
                "part_of_current_routed_target_set": part_of_current_target,
                "already_resident": already_resident,
                "globally_in_flight": globally_in_flight
            }),
        );
        ticket
    }

    pub fn finish_candidate_list(&self, arm: PredictorArm, candidates: &[(u32, f64)]) {
        if !self.accept_new_state() {
            return;
        }
        let mut state = self.state.lock();
        let stats = state
            .predictor
            .by_arm
            .entry(arm.label().to_string())
            .or_default();
        stats.candidate_lists = stats.candidate_lists.saturating_add(1);
        stats.unique_candidates_generated = stats.unique_candidates_generated.saturating_add(
            candidates
                .iter()
                .map(|(id, _)| *id)
                .collect::<HashSet<_>>()
                .len() as u64,
        );
        let mut tied = HashSet::new();
        for (idx, (_, score)) in candidates.iter().enumerate() {
            for (other_idx, (_, other_score)) in candidates.iter().enumerate().skip(idx + 1) {
                if score.to_bits() == other_score.to_bits() {
                    tied.insert(idx);
                    tied.insert(other_idx);
                }
            }
        }
        stats.tied_probability_candidates = stats
            .tied_probability_candidates
            .saturating_add(tied.len() as u64);
        let tied_count = tied.len();
        drop(state);
        self.emit(
            "prediction_candidate_list",
            json!({
                "predictor_arm": arm,
                "ordered_candidates": candidates,
                "tied_probability_candidate_count": tied_count
            }),
        );
    }

    pub fn transition(&self, ticket: CandidateTicket, transition: &'static str) {
        {
            let mut state = self.state.lock();
            if state.lifecycles.contains_key(&ticket.lifecycle_id) {
                match transition {
                    "filtered_resident" => {
                        state.lifecycle.filtered_resident =
                            state.lifecycle.filtered_resident.saturating_add(1);
                        if ticket.arm == PredictorArm::Combined {
                            state
                                .predictor
                                .candidates_lost_after_truncation_to_resident_filter = state
                                .predictor
                                .candidates_lost_after_truncation_to_resident_filter
                                .saturating_add(1);
                        }
                    }
                    "filtered_current_target" => {
                        state.lifecycle.filtered_current_target =
                            state.lifecycle.filtered_current_target.saturating_add(1)
                    }
                    "filtered_global_in_flight" => {
                        state.lifecycle.filtered_global_in_flight =
                            state.lifecycle.filtered_global_in_flight.saturating_add(1);
                        if ticket.arm == PredictorArm::Combined {
                            state
                                .predictor
                                .candidates_lost_after_truncation_to_global_inflight_filter = state
                                .predictor
                                .candidates_lost_after_truncation_to_global_inflight_filter
                                .saturating_add(1);
                        }
                    }
                    "rejected_governor" => {
                        state.lifecycle.rejected_governor =
                            state.lifecycle.rejected_governor.saturating_add(1)
                    }
                    "rejected_concurrency_limit" => {
                        state.lifecycle.rejected_concurrency_limit =
                            state.lifecycle.rejected_concurrency_limit.saturating_add(1)
                    }
                    "rejected_shadow_pool_exhaustion" => {
                        state.lifecycle.rejected_shadow_pool_exhaustion = state
                            .lifecycle
                            .rejected_shadow_pool_exhaustion
                            .saturating_add(1)
                    }
                    "admitted" => {
                        state.lifecycle.admitted = state.lifecycle.admitted.saturating_add(1);
                        if let Some(lifecycle) = state.lifecycles.get_mut(&ticket.lifecycle_id) {
                            lifecycle.admitted = true;
                        }
                    }
                    "task_spawned" => {
                        state.lifecycle.task_spawned =
                            state.lifecycle.task_spawned.saturating_add(1);
                        if let Some(lifecycle) = state.lifecycles.get_mut(&ticket.lifecycle_id) {
                            lifecycle.task_spawned = true;
                        }
                    }
                    "cache_race_found_resident" => {
                        state.lifecycle.cache_race_found_resident =
                            state.lifecycle.cache_race_found_resident.saturating_add(1)
                    }
                    "singleflight_follower" => {
                        state.lifecycle.singleflight_follower =
                            state.lifecycle.singleflight_follower.saturating_add(1)
                    }
                    "singleflight_leader" => {
                        state.lifecycle.singleflight_leader =
                            state.lifecycle.singleflight_leader.saturating_add(1);
                        if let Some(lifecycle) = state.lifecycles.get_mut(&ticket.lifecycle_id) {
                            lifecycle.singleflight_leader = true;
                        }
                    }
                    "publication_attempted" => {
                        state.lifecycle.publication_attempted =
                            state.lifecycle.publication_attempted.saturating_add(1);
                        if let Some(lifecycle) = state.lifecycles.get_mut(&ticket.lifecycle_id) {
                            lifecycle.publication_attempted = true;
                        }
                    }
                    "publication_rejected" => {
                        state.lifecycle.publication_rejected =
                            state.lifecycle.publication_rejected.saturating_add(1);
                        if let Some(lifecycle) = state.lifecycles.get_mut(&ticket.lifecycle_id) {
                            lifecycle.publication_rejected = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        self.emit(
            "lifecycle_transition",
            json!({
                "lifecycle_id": ticket.lifecycle_id,
                "request_id": ticket.request_id,
                "expert_id": ticket.expert_id,
                "transition": transition
            }),
        );
    }

    pub fn speculative_read_issued(
        &self,
        ticket: CandidateTicket,
        requested_bytes: u64,
    ) -> ReadTicket {
        self.read_issued(
            ReadClass::Speculative,
            ticket.expert_id,
            requested_bytes,
            Some(ticket.lifecycle_id),
            None,
        )
    }

    pub fn demand_read_issued(&self, lookup: LookupTicket, requested_bytes: u64) -> ReadTicket {
        self.read_issued(
            ReadClass::Demand,
            lookup.expert_id,
            requested_bytes,
            None,
            Some(lookup.lookup_id),
        )
    }

    fn read_issued(
        &self,
        class: ReadClass,
        expert_id: u32,
        requested_bytes: u64,
        lifecycle_id: Option<u64>,
        lookup_id: Option<u64>,
    ) -> ReadTicket {
        let read_id = self.next_read_id.fetch_add(1, Ordering::Relaxed);
        let issue_ns = self.now_ns();
        let (active_demand, active_speculative, total) = {
            let mut state = self.state.lock();
            let active_demand = state.active_demand;
            let active_speculative = state.active_speculative;
            let tracked = match class {
                ReadClass::Demand => lookup_id.is_some_and(|id| state.lookups.contains_key(&id)),
                ReadClass::Speculative => {
                    lifecycle_id.is_some_and(|id| state.lifecycles.contains_key(&id))
                }
            };
            if tracked {
                match class {
                    ReadClass::Demand => {
                        state.active_demand = state.active_demand.saturating_add(1);
                        state.read_overlap.demand_physical_reads =
                            state.read_overlap.demand_physical_reads.saturating_add(1);
                        state
                            .read_overlap
                            .active_speculative_reads_observed_at_demand_issue = state
                            .read_overlap
                            .active_speculative_reads_observed_at_demand_issue
                            .saturating_add(active_speculative);
                        if let Some(id) = lookup_id {
                            if let Some(lookup) = state.lookups.get_mut(&id) {
                                lookup.demand_read_start_ns = Some(issue_ns);
                            }
                        }
                    }
                    ReadClass::Speculative => {
                        state.active_speculative = state.active_speculative.saturating_add(1);
                        state.read_overlap.speculative_physical_reads = state
                            .read_overlap
                            .speculative_physical_reads
                            .saturating_add(1);
                        state.lifecycle.physical_read_issued =
                            state.lifecycle.physical_read_issued.saturating_add(1);
                        if let Some(id) = lifecycle_id {
                            if let Some(lifecycle) = state.lifecycles.get_mut(&id) {
                                lifecycle.read_issued_ns = Some(issue_ns);
                            }
                        }
                    }
                }
                let current_demand = state.active_demand;
                let current_speculative = state.active_speculative;
                state.read_overlap.peak_demand_reads_in_flight = state
                    .read_overlap
                    .peak_demand_reads_in_flight
                    .max(current_demand);
                state.read_overlap.peak_speculative_reads_in_flight = state
                    .read_overlap
                    .peak_speculative_reads_in_flight
                    .max(current_speculative);
                state.read_overlap.peak_total_reads_in_flight = state
                    .read_overlap
                    .peak_total_reads_in_flight
                    .max(current_demand.saturating_add(current_speculative));
                state.reads.insert(
                    read_id,
                    ReadState {
                        class,
                        expert_id,
                        requested_bytes,
                        actual_bytes: None,
                        issue_ns,
                        completion_ns: None,
                        failed: false,
                        active_demand_at_issue: active_demand,
                        active_speculative_at_issue: active_speculative,
                    },
                );
            }
            (
                active_demand,
                active_speculative,
                active_demand.saturating_add(active_speculative),
            )
        };
        self.emit(
            "physical_read_issued",
            json!({
                "read_id": read_id,
                "lifecycle_id": lifecycle_id,
                "lookup_id": lookup_id,
                "expert_id": expert_id,
                "read_classification": class,
                "issue_timestamp_ns": issue_ns,
                "requested_bytes": requested_bytes,
                "active_demand_reads_at_issue": active_demand,
                "active_speculative_reads_at_issue": active_speculative,
                "total_reads_in_flight_at_issue": total
            }),
        );
        ReadTicket {
            read_id,
            lifecycle_id,
            lookup_id,
            class,
        }
    }

    pub fn physical_read_completed(&self, ticket: ReadTicket, actual_bytes: u64) {
        let completion_ns = self.now_ns();
        let mut event = None;
        {
            let mut state = self.state.lock();
            if state.reads.contains_key(&ticket.read_id) {
                match ticket.class {
                    ReadClass::Demand => {
                        state.active_demand = state.active_demand.saturating_sub(1)
                    }
                    ReadClass::Speculative => {
                        state.active_speculative = state.active_speculative.saturating_sub(1);
                        state.lifecycle.physical_read_completed =
                            state.lifecycle.physical_read_completed.saturating_add(1);
                    }
                }
            }
            if let Some(read) = state.reads.get_mut(&ticket.read_id) {
                read.actual_bytes = Some(actual_bytes);
                read.completion_ns = Some(completion_ns);
                event = Some((
                    read.expert_id,
                    read.issue_ns,
                    read.requested_bytes,
                    read.active_demand_at_issue,
                    read.active_speculative_at_issue,
                ));
            }
            if let Some(id) = ticket.lifecycle_id {
                if let Some(lifecycle) = state.lifecycles.get_mut(&id) {
                    lifecycle.read_completed_ns = Some(completion_ns);
                }
            }
            if let Some(id) = ticket.lookup_id {
                if let Some(lookup) = state.lookups.get_mut(&id) {
                    lookup.availability_ns = Some(completion_ns);
                }
            }
        }
        if let Some((expert_id, issue_ns, requested, active_demand, active_speculative)) = event {
            self.emit(
                "physical_read_completed",
                json!({
                    "read_id": ticket.read_id,
                    "lifecycle_id": ticket.lifecycle_id,
                    "lookup_id": ticket.lookup_id,
                    "expert_id": expert_id,
                    "read_classification": ticket.class,
                    "issue_timestamp_ns": issue_ns,
                    "completion_timestamp_ns": completion_ns,
                    "requested_bytes": requested,
                    "actual_bytes": actual_bytes,
                    "active_demand_reads_at_issue": active_demand,
                    "active_speculative_reads_at_issue": active_speculative,
                    "total_reads_in_flight_at_issue": active_demand.saturating_add(active_speculative)
                }),
            );
        }
    }

    pub fn physical_read_failed(&self, ticket: ReadTicket, error: &str) {
        let completion_ns = self.now_ns();
        let expert_id = {
            let mut state = self.state.lock();
            if state.reads.contains_key(&ticket.read_id) {
                match ticket.class {
                    ReadClass::Demand => {
                        state.active_demand = state.active_demand.saturating_sub(1)
                    }
                    ReadClass::Speculative => {
                        state.active_speculative = state.active_speculative.saturating_sub(1);
                        state.lifecycle.physical_read_failed =
                            state.lifecycle.physical_read_failed.saturating_add(1);
                        state.timeliness.failed = state.timeliness.failed.saturating_add(1);
                    }
                }
            }
            let expert_id = state
                .reads
                .get(&ticket.read_id)
                .map(|read| read.expert_id)
                .unwrap_or(0);
            if let Some(read) = state.reads.get_mut(&ticket.read_id) {
                read.completion_ns = Some(completion_ns);
                read.failed = true;
            }
            if let Some(id) = ticket.lifecycle_id {
                if let Some(lifecycle) = state.lifecycles.get_mut(&id) {
                    lifecycle.read_failed = true;
                }
            }
            expert_id
        };
        self.emit(
            "physical_read_failed",
            json!({
                "read_id": ticket.read_id,
                "lifecycle_id": ticket.lifecycle_id,
                "lookup_id": ticket.lookup_id,
                "expert_id": expert_id,
                "read_classification": ticket.class,
                "completion_timestamp_ns": completion_ns,
                "error": error
            }),
        );
    }

    pub fn published(&self, ticket: CandidateTicket) {
        let now = self.now_ns();
        let tracked = {
            let mut state = self.state.lock();
            if let Some(lifecycle) = state.lifecycles.get_mut(&ticket.lifecycle_id) {
                lifecycle.published_ns = Some(now);
                state.lifecycle.published = state.lifecycle.published.saturating_add(1);
                true
            } else {
                false
            }
        };
        if !tracked {
            return;
        }
        self.emit(
            "lifecycle_transition",
            json!({
                "lifecycle_id": ticket.lifecycle_id,
                "request_id": ticket.request_id,
                "expert_id": ticket.expert_id,
                "transition": "published",
                "publication_timestamp_ns": now
            }),
        );
    }

    pub fn record_eviction(&self, expert_id: u32) {
        let now = self.now_ns();
        let lifecycle = {
            let mut state = self.state.lock();
            let ids = state
                .expert_lifecycles
                .get(&expert_id)
                .cloned()
                .unwrap_or_default();
            let id = ids.into_iter().rev().find(|id| {
                state.lifecycles.get(id).is_some_and(|lifecycle| {
                    lifecycle.published_ns.is_some() && lifecycle.evicted_ns.is_none()
                })
            });
            let mut unused = false;
            if let Some(id) = id {
                if let Some(lifecycle) = state.lifecycles.get_mut(&id) {
                    unused = lifecycle.first_use_ns.is_none();
                    lifecycle.evicted_ns = Some(now);
                }
                if unused {
                    state.lifecycle.evicted_before_first_use =
                        state.lifecycle.evicted_before_first_use.saturating_add(1);
                }
            }
            id.map(|id| (id, unused))
        };
        if let Some((id, unused)) = lifecycle {
            self.emit(
                "lifecycle_transition",
                json!({
                    "lifecycle_id": id,
                    "expert_id": expert_id,
                    "transition": if unused {
                        "evicted_before_first_use"
                    } else {
                        "evicted_after_first_use"
                    },
                    "timestamp_ns": now
                }),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initial_lookup(
        &self,
        request_id: u64,
        token_index: u64,
        layer: u32,
        routed_slot: usize,
        expert_id: u32,
        cache_hit: bool,
    ) -> LookupTicket {
        let now = self.now_ns();
        let lookup_id = self.next_lookup_id.fetch_add(1, Ordering::Relaxed);
        let (classification, lifecycle_id, issue, completion, publication) = {
            let mut state = self.state.lock();
            let track_lookup = self.accept_new_state();
            let ids = state
                .expert_lifecycles
                .get(&expert_id)
                .cloned()
                .unwrap_or_default();
            let active = ids.iter().rev().find_map(|id| {
                state.lifecycles.get(id).and_then(|lifecycle| {
                    (lifecycle.read_issued_ns.is_some()
                        && lifecycle.read_completed_ns.is_none()
                        && !lifecycle.read_failed)
                        .then_some((*id, lifecycle.clone()))
                })
            });
            let published = ids.iter().rev().find_map(|id| {
                state.lifecycles.get(id).and_then(|lifecycle| {
                    (lifecycle.published_ns.is_some()
                        && lifecycle.evicted_ns.is_none()
                        && lifecycle.first_use_ns.is_none())
                    .then_some((*id, lifecycle.clone()))
                })
            });
            let evicted_unused = ids.iter().rev().find_map(|id| {
                state.lifecycles.get(id).and_then(|lifecycle| {
                    (lifecycle.published_ns.is_some()
                        && lifecycle.evicted_ns.is_some()
                        && lifecycle.first_use_ns.is_none()
                        && !lifecycle.evicted_miss_observed)
                        .then_some(*id)
                })
            });
            let (classification, lifecycle) = if cache_hit {
                if let Some((id, lifecycle)) = published {
                    (
                        InitialLookupClass::ReadyPrefetchedResident,
                        Some((id, lifecycle)),
                    )
                } else {
                    (InitialLookupClass::OrdinaryResident, None)
                }
            } else if let Some((id, lifecycle)) = active {
                (
                    InitialLookupClass::SpeculativeReadInFlight,
                    Some((id, lifecycle)),
                )
            } else {
                if let Some(lifecycle_id) = evicted_unused.filter(|_| track_lookup) {
                    state.timeliness.published_but_evicted_before_lookup = state
                        .timeliness
                        .published_but_evicted_before_lookup
                        .saturating_add(1);
                    if let Some(lifecycle) = state.lifecycles.get_mut(&lifecycle_id) {
                        lifecycle.evicted_miss_observed = true;
                    }
                }
                (InitialLookupClass::OrdinaryMiss, None)
            };
            if track_lookup {
                match classification {
                    InitialLookupClass::OrdinaryResident => {
                        state.initial_lookup.ordinary_resident =
                            state.initial_lookup.ordinary_resident.saturating_add(1)
                    }
                    InitialLookupClass::ReadyPrefetchedResident => {
                        state.initial_lookup.ready_prefetched_resident = state
                            .initial_lookup
                            .ready_prefetched_resident
                            .saturating_add(1);
                        state.critical_path.ready_prefetch_removed_miss = state
                            .critical_path
                            .ready_prefetch_removed_miss
                            .saturating_add(1);
                    }
                    InitialLookupClass::SpeculativeReadInFlight => {
                        state.initial_lookup.speculative_read_in_flight = state
                            .initial_lookup
                            .speculative_read_in_flight
                            .saturating_add(1);
                        state
                            .critical_path
                            .prefetch_join_avoided_duplicate_physical_read = state
                            .critical_path
                            .prefetch_join_avoided_duplicate_physical_read
                            .saturating_add(1);
                    }
                    InitialLookupClass::OrdinaryMiss => {
                        state.initial_lookup.ordinary_miss =
                            state.initial_lookup.ordinary_miss.saturating_add(1)
                    }
                }
            }
            let lifecycle_id = lifecycle.as_ref().map(|(id, _)| *id);
            let issue = lifecycle.as_ref().and_then(|(_, l)| l.read_issued_ns);
            let completion = lifecycle.as_ref().and_then(|(_, l)| l.read_completed_ns);
            let publication = lifecycle.as_ref().and_then(|(_, l)| l.published_ns);
            let ticket = LookupTicket {
                lookup_id,
                request_id,
                token_index,
                layer,
                routed_slot,
                expert_id,
                classification,
                lifecycle_id,
            };
            if track_lookup {
                state.lookups.insert(
                    lookup_id,
                    LookupState {
                        ticket,
                        lookup_ns: now,
                        speculative_issue_ns: issue,
                        speculative_completion_ns: completion,
                        publication_ns: publication,
                        demand_read_start_ns: None,
                        availability_ns: cache_hit.then_some(now),
                        consumption_ns: None,
                        layer_compute_start_ns: None,
                        timeliness_classified: false,
                    },
                );
            }
            (classification, lifecycle_id, issue, completion, publication)
        };
        let ticket = LookupTicket {
            lookup_id,
            request_id,
            token_index,
            layer,
            routed_slot,
            expert_id,
            classification,
            lifecycle_id,
        };
        self.emit(
            "initial_demand_lookup",
            json!({
                "lookup_id": lookup_id,
                "lifecycle_id": lifecycle_id,
                "request_id": request_id,
                "token_index": token_index,
                "layer_index": layer,
                "routed_slot_index": routed_slot,
                "expert_id": expert_id,
                "classification": classification,
                "lookup_timestamp_ns": now,
                "speculative_issue_timestamp_ns": issue,
                "speculative_completion_timestamp_ns": completion,
                "publication_timestamp_ns": publication
            }),
        );
        ticket
    }

    pub fn lookup_singleflight_follower(&self, lookup: LookupTicket) {
        self.emit(
            "lookup_singleflight_follower",
            json!({
                "lookup_id": lookup.lookup_id,
                "request_id": lookup.request_id,
                "expert_id": lookup.expert_id
            }),
        );
    }

    pub fn lookup_available(&self, lookup: LookupTicket) {
        let now = self.now_ns();
        let mut state = self.state.lock();
        let lifecycle_times = lookup.lifecycle_id.and_then(|lifecycle_id| {
            state
                .lifecycles
                .get(&lifecycle_id)
                .map(|lifecycle| (lifecycle.read_completed_ns, lifecycle.published_ns))
        });
        if let Some(lookup_state) = state.lookups.get_mut(&lookup.lookup_id) {
            lookup_state.availability_ns.get_or_insert(now);
            if let Some((completion, publication)) = lifecycle_times {
                lookup_state.speculative_completion_ns = completion;
                lookup_state.publication_ns = publication;
            }
        }
    }

    pub fn layer_compute_start(
        &self,
        request_id: u64,
        token_index: u64,
        layer: u32,
        ordered_experts: &[u32],
    ) {
        let now = self.now_ns();
        let (
            initial_resident,
            initially_missing,
            joins,
            demand,
            availability,
            straggler,
            layer_lookup_start,
            first_uses,
            lookup_timings,
        ) = {
            let mut state = self.state.lock();
            let mut ids: Vec<u64> = state
                .lookups
                .iter()
                .filter_map(|(id, lookup)| {
                    (lookup.ticket.request_id == request_id
                        && lookup.ticket.token_index == token_index
                        && lookup.ticket.layer == layer)
                        .then_some(*id)
                })
                .collect();
            ids.sort_by_key(|id| state.lookups[id].ticket.routed_slot);
            let layer_lookup_start = ids
                .iter()
                .filter_map(|id| state.lookups.get(id).map(|lookup| lookup.lookup_ns))
                .min()
                .unwrap_or(now);
            let mut initial_resident = Vec::new();
            let mut initially_missing = Vec::new();
            let mut joins = Vec::new();
            let mut demand = Vec::new();
            let mut availability = Vec::new();
            let mut latest: Option<(u64, u32, InitialLookupClass)> = None;
            let mut missing_count = 0usize;
            let mut join_count = 0usize;
            let mut first_uses = Vec::new();
            for id in ids {
                let Some((expert_id, classification, lifecycle_id, available)) =
                    state.lookups.get_mut(&id).map(|lookup| {
                        lookup.layer_compute_start_ns = Some(now);
                        lookup.consumption_ns.get_or_insert(now);
                        (
                            lookup.ticket.expert_id,
                            lookup.ticket.classification,
                            lookup.ticket.lifecycle_id,
                            lookup.availability_ns.unwrap_or(now),
                        )
                    })
                else {
                    continue;
                };
                availability.push((expert_id, available));
                match classification {
                    InitialLookupClass::OrdinaryResident
                    | InitialLookupClass::ReadyPrefetchedResident => {
                        initial_resident.push(expert_id);
                    }
                    InitialLookupClass::SpeculativeReadInFlight => {
                        initially_missing.push(expert_id);
                        joins.push(expert_id);
                        missing_count += 1;
                        join_count += 1;
                    }
                    InitialLookupClass::OrdinaryMiss => {
                        initially_missing.push(expert_id);
                        demand.push(expert_id);
                        missing_count += 1;
                    }
                }
                if latest.as_ref().is_none_or(|(ts, _, _)| available >= *ts) {
                    latest = Some((available, expert_id, classification));
                }
                if let Some(lifecycle_id) = lifecycle_id {
                    let mut first = false;
                    if let Some(lifecycle) = state.lifecycles.get_mut(&lifecycle_id) {
                        if lifecycle.first_use_ns.is_none() {
                            lifecycle.first_use_ns = Some(now);
                            first = true;
                        }
                    }
                    if first {
                        state.lifecycle.first_use = state.lifecycle.first_use.saturating_add(1);
                        first_uses.push((lifecycle_id, expert_id));
                        let arm_name = state
                            .lifecycles
                            .get(&lifecycle_id)
                            .map(|lifecycle| lifecycle.ticket.arm.label().to_string());
                        if let Some(arm_name) = arm_name {
                            let arm = state.predictor.by_arm.entry(arm_name).or_default();
                            arm.timely_candidates = arm.timely_candidates.saturating_add(1);
                            if classification == InitialLookupClass::ReadyPrefetchedResident {
                                arm.ready_before_lookup_candidates =
                                    arm.ready_before_lookup_candidates.saturating_add(1);
                            }
                        }
                    }
                }
            }
            if missing_count == 0 {
                state.critical_path.layer_with_no_missing_experts = state
                    .critical_path
                    .layer_with_no_missing_experts
                    .saturating_add(1);
            } else {
                state.critical_path.layer_with_one_or_more_missing_experts = state
                    .critical_path
                    .layer_with_one_or_more_missing_experts
                    .saturating_add(1);
                if join_count == missing_count {
                    state
                        .critical_path
                        .layer_where_all_missing_experts_were_speculative_joins = state
                        .critical_path
                        .layer_where_all_missing_experts_were_speculative_joins
                        .saturating_add(1);
                }
                if !demand.is_empty() {
                    state.critical_path.layer_blocked_by_ordinary_demand_read = state
                        .critical_path
                        .layer_blocked_by_ordinary_demand_read
                        .saturating_add(1);
                }
            }
            if let Some((_, _, class)) = latest {
                *state
                    .critical_path
                    .final_straggler_source
                    .entry(class.label().to_string())
                    .or_default() += 1;
                if class == InitialLookupClass::SpeculativeReadInFlight {
                    state.critical_path.prefetched_expert_was_final_straggler = state
                        .critical_path
                        .prefetched_expert_was_final_straggler
                        .saturating_add(1);
                }
            }
            for id in state.lookups.keys().copied().collect::<Vec<_>>() {
                let Some(lookup_ro) = state.lookups.get(&id) else {
                    continue;
                };
                if lookup_ro.ticket.request_id != request_id
                    || lookup_ro.ticket.token_index != token_index
                    || lookup_ro.ticket.layer != layer
                    || lookup_ro.timeliness_classified
                {
                    continue;
                }
                let class = lookup_ro.ticket.classification;
                let expert_id = lookup_ro.ticket.expert_id;
                let available = lookup_ro.availability_ns.unwrap_or(now);
                let demand_completions: Vec<u64> = state
                    .lookups
                    .values()
                    .filter(|other| {
                        other.ticket.request_id == request_id
                            && other.ticket.token_index == token_index
                            && other.ticket.layer == layer
                            && other.ticket.classification == InitialLookupClass::OrdinaryMiss
                    })
                    .filter_map(|other| other.availability_ns)
                    .collect();
                let is_final = latest.as_ref().is_some_and(|(_, id, _)| *id == expert_id);
                match class {
                    InitialLookupClass::ReadyPrefetchedResident => {
                        state.timeliness.ready_before_lookup =
                            state.timeliness.ready_before_lookup.saturating_add(1);
                        if !is_final {
                            state.critical_path.prefetched_expert_used_but_noncritical = state
                                .critical_path
                                .prefetched_expert_used_but_noncritical
                                .saturating_add(1);
                        }
                    }
                    InitialLookupClass::SpeculativeReadInFlight if is_final => {
                        state.timeliness.in_flight_completed_as_final_straggler = state
                            .timeliness
                            .in_flight_completed_as_final_straggler
                            .saturating_add(1);
                    }
                    InitialLookupClass::SpeculativeReadInFlight
                        if demand_completions.iter().any(|ts| *ts < available) =>
                    {
                        state
                            .timeliness
                            .in_flight_completed_after_another_demand_read = state
                            .timeliness
                            .in_flight_completed_after_another_demand_read
                            .saturating_add(1);
                        state.critical_path.prefetched_expert_used_but_noncritical = state
                            .critical_path
                            .prefetched_expert_used_but_noncritical
                            .saturating_add(1);
                    }
                    InitialLookupClass::SpeculativeReadInFlight => {
                        state.timeliness.in_flight_completed_before_unrelated_misses = state
                            .timeliness
                            .in_flight_completed_before_unrelated_misses
                            .saturating_add(1);
                        state.critical_path.prefetched_expert_used_but_noncritical = state
                            .critical_path
                            .prefetched_expert_used_but_noncritical
                            .saturating_add(1);
                    }
                    _ => {}
                }
                if let Some(lookup) = state.lookups.get_mut(&id) {
                    lookup.timeliness_classified = true;
                }
            }
            let lookup_timings = state
                .lookups
                .values()
                .filter(|lookup| {
                    lookup.ticket.request_id == request_id
                        && lookup.ticket.token_index == token_index
                        && lookup.ticket.layer == layer
                })
                .map(|lookup| {
                    json!({
                        "lookup_id": lookup.ticket.lookup_id,
                        "request_id": request_id,
                        "token_index": token_index,
                        "layer_index": layer,
                        "routed_slot_index": lookup.ticket.routed_slot,
                        "expert_id": lookup.ticket.expert_id,
                        "classification": lookup.ticket.classification,
                        "lookup_timestamp_ns": lookup.lookup_ns,
                        "speculative_issue_timestamp_ns": lookup.speculative_issue_ns,
                        "speculative_completion_timestamp_ns": lookup.speculative_completion_ns,
                        "publication_timestamp_ns": lookup.publication_ns,
                        "demand_read_start_timestamp_ns": lookup.demand_read_start_ns,
                        "availability_timestamp_ns": lookup.availability_ns,
                        "consumption_timestamp_ns": lookup.consumption_ns,
                        "layer_compute_start_timestamp_ns": lookup.layer_compute_start_ns
                    })
                })
                .collect::<Vec<_>>();
            (
                initial_resident,
                initially_missing,
                joins,
                demand,
                availability,
                latest,
                layer_lookup_start,
                first_uses,
                lookup_timings,
            )
        };
        for (lifecycle_id, expert_id) in first_uses {
            self.emit(
                "lifecycle_transition",
                json!({
                    "lifecycle_id": lifecycle_id,
                    "request_id": request_id,
                    "expert_id": expert_id,
                    "transition": "first_use",
                    "timestamp_ns": now
                }),
            );
        }
        for timing in lookup_timings {
            self.emit("demand_lookup_timing", timing);
        }
        let (straggler_expert, straggler_source) = straggler
            .map(|(_, expert, class)| (Some(expert), Some(class.label())))
            .unwrap_or((None, None));
        self.emit(
            "layer_fetch_complete",
            json!({
                "request_id": request_id,
                "token_index": token_index,
                "layer_index": layer,
                "ordered_topk_routed_experts": ordered_experts,
                "initially_resident_experts": initial_resident,
                "initially_missing_experts": initially_missing,
                "speculative_join_experts": joins,
                "ordinary_demand_read_experts": demand,
                "expert_availability_timestamps_ns": availability,
                "final_fetch_straggler_expert": straggler_expert,
                "final_fetch_straggler_source": straggler_source,
                "layer_lookup_start_timestamp_ns": layer_lookup_start,
                "layer_fetch_completion_timestamp_ns": now,
                "layer_compute_start_timestamp_ns": now
            }),
        );
    }

    pub fn layer_compute_complete(&self, request_id: u64, token_index: u64, layer: u32) {
        self.emit(
            "layer_compute_complete",
            json!({
                "request_id": request_id,
                "token_index": token_index,
                "layer_index": layer,
                "layer_compute_completion_timestamp_ns": self.now_ns()
            }),
        );
    }

    pub fn snapshot(&self) -> Phase4bSnapshot {
        {
            let mut writer = self.writer.lock();
            if !writer.failed && writer.writer.flush().is_err() {
                writer.failed = true;
                self.writer_failed.store(true, Ordering::Release);
                self.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        let (
            lifecycle,
            initial_lookup,
            timeliness,
            critical_path,
            mut predictor,
            read_overlap,
            contamination,
        ) = {
            let state = self.state.lock();
            let mut lifecycle = state.lifecycle.clone();
            lifecycle.physical_read_inflight_at_sample = state
                .lifecycles
                .values()
                .filter(|l| {
                    l.read_issued_ns.is_some() && l.read_completed_ns.is_none() && !l.read_failed
                })
                .count() as u64;
            lifecycle.completion_not_yet_published_at_sample = state
                .lifecycles
                .values()
                .filter(|l| {
                    l.read_completed_ns.is_some()
                        && l.published_ns.is_none()
                        && !l.publication_rejected
                })
                .count() as u64;
            lifecycle.first_use = 0;
            lifecycle.evicted_before_first_use = 0;
            lifecycle.still_resident_unused_at_sample = 0;
            for published in state
                .lifecycles
                .values()
                .filter(|lifecycle| lifecycle.published_ns.is_some())
            {
                match (
                    published.first_use_ns.is_some(),
                    published.evicted_ns.is_some(),
                ) {
                    (true, _) => {
                        lifecycle.first_use = lifecycle.first_use.saturating_add(1);
                    }
                    (false, true) => {
                        lifecycle.evicted_before_first_use =
                            lifecycle.evicted_before_first_use.saturating_add(1);
                    }
                    (false, false) => {
                        lifecycle.still_resident_unused_at_sample =
                            lifecycle.still_resident_unused_at_sample.saturating_add(1);
                    }
                }
            }
            let mut timeliness = state.timeliness.clone();
            timeliness.published_but_never_requested = lifecycle.still_resident_unused_at_sample;
            timeliness.completed_but_publication_rejected = lifecycle.publication_rejected;
            timeliness.still_in_flight_at_sample = lifecycle.physical_read_inflight_at_sample;
            (
                lifecycle,
                state.initial_lookup.clone(),
                timeliness,
                state.critical_path.clone(),
                state.predictor.clone(),
                state.read_overlap.clone(),
                state.request_boundary_contamination_count,
            )
        };
        for stats in predictor.by_arm.values_mut() {
            stats.candidate_precision = ratio(
                stats.candidates_in_actual_next_top8,
                stats.candidates_generated,
            );
            stats.candidate_set_recall = ratio(
                stats.candidates_in_actual_next_top8,
                stats.actual_next_top8_slots,
            );
            stats.correct_layer_precision = ratio(
                stats.candidates_in_correct_next_layer,
                stats.candidates_generated,
            );
            stats.timely_precision = ratio(stats.timely_candidates, stats.candidates_generated);
            stats.timely_recall = ratio(stats.timely_candidates, stats.actual_next_top8_slots);
            stats.ready_before_lookup_precision = ratio(
                stats.ready_before_lookup_candidates,
                stats.candidates_generated,
            );
            stats.ready_before_lookup_recall = ratio(
                stats.ready_before_lookup_candidates,
                stats.actual_next_top8_slots,
            );
        }
        let mut reconciliation_errors = Vec::new();
        if lifecycle.physical_read_issued
            != lifecycle
                .physical_read_completed
                .saturating_add(lifecycle.physical_read_failed)
                .saturating_add(lifecycle.physical_read_inflight_at_sample)
        {
            reconciliation_errors.push(
                "physical_read_issued != completed + failed + inflight_at_sample".to_string(),
            );
        }
        if lifecycle.physical_read_completed
            != lifecycle
                .published
                .saturating_add(lifecycle.publication_rejected)
                .saturating_add(lifecycle.completion_not_yet_published_at_sample)
        {
            reconciliation_errors.push(
                "physical_read_completed != published + publication_rejected + completion_not_yet_published_at_sample"
                    .to_string(),
            );
        }
        if lifecycle.published
            != lifecycle
                .first_use
                .saturating_add(lifecycle.evicted_before_first_use)
                .saturating_add(lifecycle.still_resident_unused_at_sample)
        {
            reconciliation_errors.push(
                "published != first_use + evicted_before_first_use + still_resident_unused_at_sample"
                    .to_string(),
            );
        }
        let snapshot = Phase4bSnapshot {
            schema_name: PHASE4B_SCHEMA_NAME,
            schema_version: PHASE4B_SCHEMA_VERSION,
            trace_path: self.path.display().to_string(),
            max_events: self.max_events,
            events_written: self.events_written.load(Ordering::Relaxed),
            events_dropped: self.events_dropped.load(Ordering::Relaxed),
            trace_truncated: self.truncated.load(Ordering::Relaxed),
            trace_write_failed: self.writer_failed.load(Ordering::Relaxed),
            lifecycle_reconciliation_passed: reconciliation_errors.is_empty(),
            lifecycle_reconciliation_errors: reconciliation_errors,
            lifecycle,
            initial_lookup,
            timeliness,
            layer_critical_path: critical_path,
            predictor_quality: predictor,
            read_overlap,
            request_boundary_contamination_count: contamination,
        };
        snapshot
    }

    #[inline]
    fn accept_new_state(&self) -> bool {
        !self.truncated.load(Ordering::Acquire) && !self.writer_failed.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn inject_writer_failure(&self) {
        self.writer.lock().failed = true;
        self.writer_failed.store(true, Ordering::Release);
    }

    fn emit(&self, event_type: &'static str, payload: Value) {
        if self.truncated.load(Ordering::Acquire) {
            self.events_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let event_id = self.next_event_id.fetch_add(1, Ordering::Relaxed);
        let written = self.events_written.load(Ordering::Acquire);
        if written.saturating_add(1) >= self.max_events {
            self.events_dropped.fetch_add(1, Ordering::Relaxed);
            if !self.truncated.swap(true, Ordering::AcqRel) {
                let truncation = json!({
                    "schema_name": PHASE4B_SCHEMA_NAME,
                    "schema_version": PHASE4B_SCHEMA_VERSION,
                    "event_id": event_id,
                    "event_type": "trace_truncated",
                    "monotonic_timestamp_ns": self.now_ns(),
                    "max_events": self.max_events,
                    "events_dropped_at_indicator": self.events_dropped.load(Ordering::Relaxed)
                });
                self.write_json_line(&truncation);
            }
            return;
        }
        let envelope = json!({
            "schema_name": PHASE4B_SCHEMA_NAME,
            "schema_version": PHASE4B_SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "monotonic_timestamp_ns": self.now_ns(),
            "payload": payload
        });
        self.write_json_line(&envelope);
    }

    fn write_json_line(&self, value: &Value) {
        let mut writer = self.writer.lock();
        if writer.failed {
            self.writer_failed.store(true, Ordering::Release);
            self.events_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let result = serde_json::to_writer(&mut writer.writer, value).and_then(|_| {
            writer
                .writer
                .write_all(b"\n")
                .map_err(serde_json::Error::io)
        });
        match result {
            Ok(()) => {
                self.events_written.fetch_add(1, Ordering::Release);
            }
            Err(_) => {
                writer.failed = true;
                self.writer_failed.store(true, Ordering::Release);
                self.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
pub(crate) mod fake_io {
    //! Deterministic, test-only one-or-many-lane storage seam. Completion is
    //! barrier controlled; no test uses a wall-clock sleep as its oracle.

    use super::ReadClass;
    use std::collections::{HashMap, HashSet, VecDeque};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct FakeRead {
        pub sequence: u64,
        pub expert_id: u32,
        pub class: ReadClass,
        pub lane: usize,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum FakeAdmission {
        Issued(FakeRead),
        InitialCacheRace,
        GlobalSingleflightFollower,
        NoShadowBuffer,
        SemaphoreRejected,
    }

    pub struct DeterministicFakeIo {
        lanes: Vec<VecDeque<FakeRead>>,
        next_sequence: u64,
        permits_total: usize,
        permits_available: usize,
        buffers_total: usize,
        buffers_available: usize,
        resources: HashMap<u64, (bool, bool)>,
        singleflight: HashMap<u32, usize>,
        followers_notified: u64,
        initial_cache_races: HashSet<u32>,
        globally_occupied: HashSet<u32>,
        no_shadow_buffers: HashSet<u32>,
        semaphore_rejections: HashSet<u32>,
        failures: HashMap<u32, &'static str>,
        cancellations: HashSet<u32>,
        publication_rejections: HashMap<u32, bool>,
    }

    impl DeterministicFakeIo {
        pub fn new(lanes: usize) -> Self {
            Self::with_resources(lanes, usize::MAX, usize::MAX)
        }

        pub fn with_resources(lanes: usize, permits: usize, buffers: usize) -> Self {
            Self {
                lanes: (0..lanes.max(1)).map(|_| VecDeque::new()).collect(),
                next_sequence: 1,
                permits_total: permits,
                permits_available: permits,
                buffers_total: buffers,
                buffers_available: buffers,
                resources: HashMap::new(),
                singleflight: HashMap::new(),
                followers_notified: 0,
                initial_cache_races: HashSet::new(),
                globally_occupied: HashSet::new(),
                no_shadow_buffers: HashSet::new(),
                semaphore_rejections: HashSet::new(),
                failures: HashMap::new(),
                cancellations: HashSet::new(),
                publication_rejections: HashMap::new(),
            }
        }

        pub fn issue(&mut self, expert_id: u32, class: ReadClass) -> FakeRead {
            let lane = self
                .lanes
                .iter()
                .enumerate()
                .min_by_key(|(_, queue)| queue.len())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let read = FakeRead {
                sequence: self.next_sequence,
                expert_id,
                class,
                lane,
            };
            self.next_sequence += 1;
            self.lanes[lane].push_back(read);
            read
        }

        pub fn attempt_speculative(&mut self, expert_id: u32) -> FakeAdmission {
            if self.initial_cache_races.contains(&expert_id) {
                return FakeAdmission::InitialCacheRace;
            }
            if let Some(followers) = self.singleflight.get_mut(&expert_id) {
                *followers += 1;
                return FakeAdmission::GlobalSingleflightFollower;
            }
            if self.semaphore_rejections.contains(&expert_id) || self.permits_available == 0 {
                return FakeAdmission::SemaphoreRejected;
            }

            self.permits_available -= 1;
            if self.no_shadow_buffers.contains(&expert_id) || self.buffers_available == 0 {
                self.permits_available += 1;
                return FakeAdmission::NoShadowBuffer;
            }

            self.buffers_available -= 1;
            self.singleflight.insert(expert_id, 0);
            let read = self.issue(expert_id, ReadClass::Speculative);
            self.resources.insert(read.sequence, (true, true));
            FakeAdmission::Issued(read)
        }

        pub fn observe_issue_order(&self) -> Vec<FakeRead> {
            let mut all: Vec<FakeRead> = self
                .lanes
                .iter()
                .flat_map(|queue| queue.iter().copied())
                .collect();
            all.sort_by_key(|read| read.sequence);
            all
        }

        pub fn release_next(&mut self, lane: usize) -> Option<Result<FakeRead, &'static str>> {
            let read = self.lanes.get_mut(lane)?.pop_front()?;
            let result = if self.cancellations.contains(&read.expert_id) {
                Err("injected cancellation or panic")
            } else {
                match self.failures.get(&read.expert_id) {
                    Some(error) => Err(*error),
                    None => Ok(read),
                }
            };
            self.release_resources(read);
            Some(result)
        }

        fn release_resources(&mut self, read: FakeRead) {
            if let Some((permit, buffer)) = self.resources.remove(&read.sequence) {
                if permit {
                    self.permits_available += 1;
                }
                if buffer {
                    self.buffers_available += 1;
                }
                if let Some(followers) = self.singleflight.remove(&read.expert_id) {
                    self.followers_notified += followers as u64;
                }
            }
        }

        pub fn inject_initial_cache_race(&mut self, expert_id: u32) {
            self.initial_cache_races.insert(expert_id);
        }

        pub fn inject_global_singleflight(&mut self, expert_id: u32) {
            self.globally_occupied.insert(expert_id);
            self.singleflight.insert(expert_id, 0);
        }

        pub fn settle_global_singleflight(&mut self, expert_id: u32) {
            self.globally_occupied.remove(&expert_id);
            if let Some(followers) = self.singleflight.remove(&expert_id) {
                self.followers_notified += followers as u64;
            }
        }

        pub fn inject_no_shadow_buffer(&mut self, expert_id: u32) {
            self.no_shadow_buffers.insert(expert_id);
        }

        pub fn inject_semaphore_rejection(&mut self, expert_id: u32) {
            self.semaphore_rejections.insert(expert_id);
        }

        pub fn inject_failure(&mut self, expert_id: u32, error: &'static str) {
            self.failures.insert(expert_id, error);
        }

        pub fn inject_cancellation_or_panic(&mut self, expert_id: u32) {
            self.cancellations.insert(expert_id);
        }

        pub fn inject_publication_rejection(&mut self, expert_id: u32) {
            self.publication_rejections.insert(expert_id, true);
        }

        pub fn publication_rejected(&self, expert_id: u32) -> bool {
            self.publication_rejections
                .get(&expert_id)
                .copied()
                .unwrap_or(false)
        }

        pub fn is_idle(&self) -> bool {
            self.lanes.iter().all(VecDeque::is_empty)
        }

        pub fn resources_reconciled(&self) -> bool {
            self.permits_available == self.permits_total
                && self.buffers_available == self.buffers_total
                && self.resources.is_empty()
                && self.singleflight.is_empty()
                && self.globally_occupied.is_empty()
        }

        pub fn followers_notified(&self) -> u64 {
            self.followers_notified
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake_io::{DeterministicFakeIo, FakeAdmission};
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        std::env::temp_dir().join(format!(
            "mer-phase4b-{label}-{}-{}.jsonl",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn candidate(trace: &Phase4bTrace, request: u64, expert: u32, rank: usize) -> CandidateTicket {
        trace.candidate_generated(
            request,
            0,
            0,
            Some(1),
            &[7],
            PredictorArm::Combined,
            rank,
            0.5,
            expert,
            Some(1),
            Some(expert),
            Some(true),
            false,
            false,
            false,
        )
    }

    fn publish(trace: &Phase4bTrace, ticket: CandidateTicket) {
        trace.transition(ticket, "admitted");
        trace.transition(ticket, "task_spawned");
        trace.transition(ticket, "singleflight_leader");
        let read = trace.speculative_read_issued(ticket, 4096);
        trace.physical_read_completed(read, 4096);
        trace.transition(ticket, "publication_attempted");
        trace.published(ticket);
    }

    #[test]
    fn bounded_trace_emits_explicit_truncation_indicator() {
        let path = temp_path("bounded");
        let trace = Phase4bTrace::open(&path, 3).expect("trace");
        let request = trace.begin_request("fixture", 0, true);
        trace.end_request(request);
        trace.begin_request("fixture", 1, true);
        let snapshot = trace.snapshot();
        assert!(snapshot.trace_truncated);
        assert!(snapshot.events_dropped > 0);
        let body = std::fs::read_to_string(&path).expect("trace body");
        assert!(body.contains("\"event_type\":\"trace_truncated\""));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ready_prefetch_is_ready_before_lookup_and_issues_no_demand_read() {
        let path = temp_path("ready");
        let trace = Phase4bTrace::open(&path, 256).unwrap();
        let request = trace.begin_request("fixture", 0, true);
        let ticket = candidate(&trace, request, 9, 0);
        publish(&trace, ticket);
        let lookup = trace.initial_lookup(request, 0, 1, 0, 9, true);
        assert_eq!(
            lookup.classification,
            InitialLookupClass::ReadyPrefetchedResident
        );
        trace.layer_compute_start(request, 0, 1, &[9]);
        let reused = trace.initial_lookup(request, 1, 1, 0, 9, true);
        assert_eq!(
            reused.classification,
            InitialLookupClass::OrdinaryResident,
            "a prefetched lifecycle receives ready credit only on first use"
        );
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.initial_lookup.ready_prefetched_resident, 1);
        assert_eq!(snapshot.initial_lookup.ordinary_resident, 1);
        assert_eq!(snapshot.initial_lookup.speculative_read_in_flight, 0);
        assert_eq!(snapshot.read_overlap.demand_physical_reads, 0);
        assert_eq!(snapshot.timeliness.ready_before_lookup, 1);
        assert!(snapshot.lifecycle_reconciliation_passed);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn late_speculative_join_is_not_credited_as_ready() {
        let path = temp_path("late-join");
        let trace = Phase4bTrace::open(&path, 256).unwrap();
        let request = trace.begin_request("fixture", 0, true);
        let ticket = candidate(&trace, request, 9, 0);
        trace.transition(ticket, "admitted");
        trace.transition(ticket, "task_spawned");
        trace.transition(ticket, "singleflight_leader");
        let read = trace.speculative_read_issued(ticket, 4096);
        let lookup = trace.initial_lookup(request, 0, 1, 0, 9, false);
        assert_eq!(
            lookup.classification,
            InitialLookupClass::SpeculativeReadInFlight
        );
        trace.lookup_singleflight_follower(lookup);
        trace.physical_read_completed(read, 4096);
        trace.transition(ticket, "publication_attempted");
        trace.published(ticket);
        trace.lookup_available(lookup);
        trace.layer_compute_start(request, 0, 1, &[9]);
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.initial_lookup.ready_prefetched_resident, 0);
        assert_eq!(snapshot.initial_lookup.speculative_read_in_flight, 1);
        assert_eq!(
            snapshot
                .layer_critical_path
                .prefetch_join_avoided_duplicate_physical_read,
            1
        );
        assert_eq!(snapshot.read_overlap.demand_physical_reads, 0);
        assert!(snapshot.lifecycle_reconciliation_passed);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn eviction_before_use_gets_no_timely_credit_and_later_lookup_misses() {
        let path = temp_path("evicted");
        let trace = Phase4bTrace::open(&path, 256).unwrap();
        let request = trace.begin_request("fixture", 0, true);
        let ticket = candidate(&trace, request, 9, 0);
        publish(&trace, ticket);
        trace.record_eviction(9);
        let lookup = trace.initial_lookup(request, 0, 1, 0, 9, false);
        assert_eq!(lookup.classification, InitialLookupClass::OrdinaryMiss);
        let repeated_lookup = trace.initial_lookup(request, 0, 1, 1, 9, false);
        assert_eq!(
            repeated_lookup.classification,
            InitialLookupClass::OrdinaryMiss
        );
        let demand = trace.demand_read_issued(lookup, 4096);
        trace.physical_read_completed(demand, 4096);
        trace.lookup_available(lookup);
        trace.layer_compute_start(request, 0, 1, &[9]);
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.lifecycle.evicted_before_first_use, 1);
        assert_eq!(snapshot.timeliness.published_but_evicted_before_lookup, 1);
        assert_eq!(snapshot.timeliness.ready_before_lookup, 0);
        assert_eq!(snapshot.read_overlap.demand_physical_reads, 1);
        assert!(snapshot.lifecycle_reconciliation_passed);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn all_eight_barrier_attributes_blocked_eighth_without_claiming_removal() {
        let path = temp_path("all-eight");
        let trace = Phase4bTrace::open(&path, 1024).unwrap();
        let request = trace.begin_request("fixture", 0, true);
        for expert in 0..7u32 {
            let ticket = candidate(&trace, request, expert, expert as usize);
            publish(&trace, ticket);
            let lookup = trace.initial_lookup(request, 0, 1, expert as usize, expert, true);
            assert_eq!(
                lookup.classification,
                InitialLookupClass::ReadyPrefetchedResident
            );
        }
        let blocked = trace.initial_lookup(request, 0, 1, 7, 7, false);
        let demand = trace.demand_read_issued(blocked, 4096);
        trace.physical_read_completed(demand, 4096);
        trace.lookup_available(blocked);
        trace.layer_compute_start(request, 0, 1, &[0, 1, 2, 3, 4, 5, 6, 7]);
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.initial_lookup.ready_prefetched_resident, 7);
        assert_eq!(
            snapshot
                .layer_critical_path
                .prefetched_expert_shortened_final_straggler_controlled_replay_only,
            0
        );
        assert_eq!(
            snapshot
                .layer_critical_path
                .final_straggler_source
                .get("ordinary_miss"),
            Some(&1)
        );
        assert!(snapshot.lifecycle_reconciliation_passed);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn request_boundary_history_contamination_is_detected_not_corrected() {
        let path = temp_path("boundary");
        let trace = Phase4bTrace::open(&path, 64).unwrap();
        let first = trace.begin_request("first", 0, true);
        trace.record_routing(first, 0, 47, &[1], &[1], &[1.0], None);
        trace.end_request(first);
        let second = trace.begin_request("second", 1, true);
        trace.record_routing(second, 0, 0, &[2], &[2], &[1.0], Some(first));
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.request_boundary_contamination_count, 1);
        assert_eq!(
            snapshot.predictor_quality.cross_request_history_use_count,
            1
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn published_terminal_snapshot_excludes_unpublished_first_use_observations() {
        let path = temp_path("published-terminal-population");
        let trace = Phase4bTrace::open(&path, 256).unwrap();
        let request = trace.begin_request("fixture", 0, true);

        let consumed_before_publication = candidate(&trace, request, 1, 0);
        trace.transition(consumed_before_publication, "admitted");
        trace.transition(consumed_before_publication, "task_spawned");
        trace.transition(consumed_before_publication, "singleflight_leader");
        let pending_read = trace.speculative_read_issued(consumed_before_publication, 4096);
        let lookup = trace.initial_lookup(request, 0, 1, 0, 1, false);
        assert_eq!(
            lookup.classification,
            InitialLookupClass::SpeculativeReadInFlight
        );
        trace.physical_read_completed(pending_read, 4096);
        trace.lookup_available(lookup);
        trace.layer_compute_start(request, 0, 1, &[1]);

        let published_but_unused = candidate(&trace, request, 2, 1);
        publish(&trace, published_but_unused);

        assert_eq!(
            trace.state.lock().lifecycle.first_use,
            1,
            "the cumulative observation counter retains the unpublished first use"
        );
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.lifecycle.physical_read_completed, 2);
        assert_eq!(snapshot.lifecycle.published, 1);
        assert_eq!(snapshot.lifecycle.completion_not_yet_published_at_sample, 1);
        assert_eq!(
            snapshot.lifecycle.first_use, 0,
            "an unpublished first-use observation is outside the published terminal population"
        );
        assert_eq!(snapshot.lifecycle.evicted_before_first_use, 0);
        assert_eq!(snapshot.lifecycle.still_resident_unused_at_sample, 1);
        assert_eq!(
            snapshot.lifecycle.first_use
                + snapshot.lifecycle.evicted_before_first_use
                + snapshot.lifecycle.still_resident_unused_at_sample,
            snapshot.lifecycle.published,
            "published terminal categories must be mutually exclusive and exhaustive"
        );
        assert!(snapshot.lifecycle_reconciliation_passed);
        assert!(snapshot.lifecycle_reconciliation_errors.is_empty());

        let body = std::fs::read_to_string(&path).expect("trace body");
        assert!(
            body.contains("\"transition\":\"first_use\""),
            "the cumulative first-use observation must remain in the emitted trace"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cleanup_state_matrix_reconciles_failure_rejection_inflight_and_resident() {
        let path = temp_path("cleanup");
        let trace = Phase4bTrace::open(&path, 512).unwrap();
        let request = trace.begin_request("fixture", 0, true);

        for (expert, transition) in [
            (10, "filtered_resident"),
            (11, "filtered_current_target"),
            (12, "filtered_global_in_flight"),
            (13, "rejected_governor"),
            (14, "rejected_concurrency_limit"),
            (15, "rejected_shadow_pool_exhaustion"),
            (16, "cache_race_found_resident"),
            (17, "singleflight_follower"),
        ] {
            let dropped = candidate(&trace, request, expert, expert as usize);
            trace.transition(dropped, transition);
        }

        let resident = candidate(&trace, request, 1, 0);
        publish(&trace, resident);

        let failed = candidate(&trace, request, 2, 1);
        trace.transition(failed, "admitted");
        trace.transition(failed, "task_spawned");
        trace.transition(failed, "singleflight_leader");
        let failed_read = trace.speculative_read_issued(failed, 4096);
        trace.physical_read_failed(failed_read, "injected");

        let rejected = candidate(&trace, request, 3, 2);
        trace.transition(rejected, "admitted");
        trace.transition(rejected, "task_spawned");
        trace.transition(rejected, "singleflight_leader");
        let rejected_read = trace.speculative_read_issued(rejected, 4096);
        trace.physical_read_completed(rejected_read, 4096);
        trace.transition(rejected, "publication_attempted");
        trace.transition(rejected, "publication_rejected");

        let inflight = candidate(&trace, request, 4, 3);
        trace.transition(inflight, "admitted");
        trace.transition(inflight, "task_spawned");
        trace.transition(inflight, "singleflight_leader");
        let _inflight_read = trace.speculative_read_issued(inflight, 4096);

        let snapshot = trace.snapshot();
        assert_eq!(snapshot.lifecycle.physical_read_issued, 4);
        assert_eq!(snapshot.lifecycle.physical_read_completed, 2);
        assert_eq!(snapshot.lifecycle.physical_read_failed, 1);
        assert_eq!(snapshot.lifecycle.physical_read_inflight_at_sample, 1);
        assert_eq!(snapshot.lifecycle.publication_rejected, 1);
        assert_eq!(snapshot.lifecycle.still_resident_unused_at_sample, 1);
        assert_eq!(snapshot.lifecycle.filtered_resident, 1);
        assert_eq!(snapshot.lifecycle.filtered_current_target, 1);
        assert_eq!(snapshot.lifecycle.filtered_global_in_flight, 1);
        assert_eq!(snapshot.lifecycle.rejected_governor, 1);
        assert_eq!(snapshot.lifecycle.rejected_concurrency_limit, 1);
        assert_eq!(snapshot.lifecycle.rejected_shadow_pool_exhaustion, 1);
        assert_eq!(snapshot.lifecycle.cache_race_found_resident, 1);
        assert_eq!(snapshot.lifecycle.singleflight_follower, 1);
        assert!(snapshot.lifecycle_reconciliation_passed);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn deterministic_one_lane_speculation_delays_unrelated_demand() {
        let mut with_speculation = DeterministicFakeIo::new(1);
        let spec = with_speculation.issue(10, ReadClass::Speculative);
        let demand = with_speculation.issue(20, ReadClass::Demand);
        assert_eq!(
            with_speculation.observe_issue_order(),
            vec![spec, demand],
            "the fake seam exposes deterministic issue order and read class"
        );
        assert_eq!(
            with_speculation.release_next(0).unwrap().unwrap(),
            spec,
            "FIFO one-lane service completes speculation first"
        );
        assert_eq!(with_speculation.release_next(0).unwrap().unwrap(), demand);

        let mut control = DeterministicFakeIo::new(1);
        let control_demand = control.issue(20, ReadClass::Demand);
        assert_eq!(control.release_next(0).unwrap().unwrap(), control_demand);
        assert!(
            demand.sequence > control_demand.sequence,
            "the controlled speculative enqueue adds one service position"
        );
    }

    #[test]
    fn deterministic_fake_io_supports_failure_publication_rejection_and_cleanup() {
        let mut io = DeterministicFakeIo::with_resources(2, 4, 4);
        io.inject_initial_cache_race(1);
        io.inject_global_singleflight(2);
        io.inject_no_shadow_buffer(3);
        io.inject_semaphore_rejection(4);
        io.inject_failure(5, "injected failure");
        io.inject_publication_rejection(6);
        io.inject_cancellation_or_panic(7);

        assert_eq!(io.attempt_speculative(1), FakeAdmission::InitialCacheRace);
        assert_eq!(
            io.attempt_speculative(2),
            FakeAdmission::GlobalSingleflightFollower
        );
        assert_eq!(io.attempt_speculative(3), FakeAdmission::NoShadowBuffer);
        assert_eq!(io.attempt_speculative(4), FakeAdmission::SemaphoreRejected);
        let FakeAdmission::Issued(failed) = io.attempt_speculative(5) else {
            panic!("failure injection must admit a read");
        };
        let FakeAdmission::Issued(rejected) = io.attempt_speculative(6) else {
            panic!("publication rejection must admit a read");
        };
        let FakeAdmission::Issued(cancelled) = io.attempt_speculative(7) else {
            panic!("cancellation injection must admit a read");
        };

        assert_eq!(
            io.release_next(failed.lane).unwrap(),
            Err("injected failure")
        );
        assert_eq!(io.release_next(rejected.lane).unwrap().unwrap(), rejected);
        assert!(io.publication_rejected(6));
        assert_eq!(
            io.release_next(cancelled.lane).unwrap(),
            Err("injected cancellation or panic")
        );
        io.settle_global_singleflight(2);

        assert!(io.is_idle());
        assert_eq!(io.followers_notified(), 1);
        assert!(
            io.resources_reconciled(),
            "permits, buffers, and singleflight ownership must all be returned"
        );
    }
}
