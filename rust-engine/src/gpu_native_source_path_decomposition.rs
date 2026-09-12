//! HMA-1D qualification-only observations of the existing production stream.
//! Timestamp capture never owns storage, a GPU, a lock, or a source scheduler.
use crate::gpu_native_source_upload::Arm;
use serde::Serialize;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-source-path-decomposition-production.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-source-path-decomposition-production";
// Frozen Qwen production geometry: eight selected experts per layer. This is
// deliberately the source width, not the sixteen-slot upload-ring capacity.
pub(crate) const WIDTH: usize = 8;
const _: () = assert!(crate::io_provider::STORAGE_RETRY_ATTEMPTS == 3);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawRead {
    pub(crate) start: Option<Instant>,
    pub(crate) end: Option<Instant>,
    pub(crate) attempt_starts: [Option<Instant>; 3],
    pub(crate) attempt_ends: [Option<Instant>; 3],
    pub(crate) transient_events: u32,
    pub(crate) breaker_event: bool,
    pub(crate) success: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawBatch {
    pub(crate) start: Option<Instant>,
    pub(crate) fd_end: Option<Instant>,
    pub(crate) scheduler_start: Option<Instant>,
    pub(crate) scheduler_end: Option<Instant>,
    pub(crate) end: Option<Instant>,
    pub(crate) reads: [RawRead; WIDTH],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Helper {
    ControlSingleFileExt,
    ControlBatchScopedFileExt,
    TreatmentAlignedBatchScopedFileExt,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct Totals {
    pub(crate) helper_total_ns: u64,
    pub(crate) fd_resolve_proof_ns: u64,
    pub(crate) read_critical_span_ns: u64,
    pub(crate) scheduler_shell_ns: u64,
    pub(crate) residual_helper_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Read {
    pub(crate) wrapper_start_ns: u64,
    pub(crate) wrapper_end_ns: u64,
    pub(crate) wrapper_wall_ns: u64,
    pub(crate) attempt_wall_ns: [Option<u64>; 3],
    pub(crate) retry_attempt_count: u32,
    pub(crate) transient_events: u32,
    pub(crate) breaker_event: bool,
    pub(crate) success: bool,
}

fn ns(a: Instant, b: Instant) -> Result<u64, &'static str> {
    u64::try_from(
        b.checked_duration_since(a)
            .ok_or("timestamp order")?
            .as_nanos(),
    )
    .map_err(|_| "nanosecond overflow")
}

fn stamp(value: Option<Instant>) -> Result<Instant, &'static str> {
    value.ok_or("missing timestamp")
}

impl RawBatch {
    /// All arithmetic and reconstruction occur after BOTH helper timers stop.
    pub(crate) fn reconstruct(
        &self,
        width: usize,
    ) -> Result<(Totals, [Read; WIDTH]), &'static str> {
        if width == 0 || width > WIDTH {
            return Err("source width");
        }
        let start = stamp(self.start)?;
        let fd = stamp(self.fd_end)?;
        let sched_start = stamp(self.scheduler_start)?;
        let sched_end = stamp(self.scheduler_end)?;
        let end = stamp(self.end)?;
        ns(start, fd)?;
        ns(fd, sched_start)?;
        ns(sched_start, sched_end)?;
        ns(sched_end, end)?;
        let mut reads = [Read::default(); WIDTH];
        let mut earliest = sched_end;
        let mut latest = sched_start;
        for (raw, read) in self.reads[..width].iter().zip(&mut reads) {
            let a = stamp(raw.start)?;
            let b = stamp(raw.end)?;
            ns(sched_start, a)?;
            ns(b, sched_end)?;
            earliest = earliest.min(a);
            latest = latest.max(b);
            read.wrapper_start_ns = ns(start, a)?;
            read.wrapper_end_ns = ns(start, b)?;
            read.wrapper_wall_ns = ns(a, b)?;
            read.success = raw.success;
            read.breaker_event = raw.breaker_event;
            read.transient_events = raw.transient_events;
            let mut attempts = 0u32;
            let mut previous_end = a;
            let mut gap = false;
            for i in 0..3 {
                match (raw.attempt_starts[i], raw.attempt_ends[i]) {
                    (Some(x), Some(y)) if !gap => {
                        ns(previous_end, x)?;
                        ns(y, b)?;
                        read.attempt_wall_ns[i] = Some(ns(x, y)?);
                        previous_end = y;
                        attempts += 1;
                    }
                    (None, None) => gap = true,
                    _ => return Err("read attempt timestamps"),
                }
            }
            read.retry_attempt_count = attempts.saturating_sub(1);
            if raw.success && attempts == 0 {
                return Err("successful read without attempt");
            }
        }
        let helper_total_ns = ns(start, end)?;
        let fd_resolve_proof_ns = ns(start, fd)?;
        let read_critical_span_ns = ns(earliest, latest)?;
        let scheduler_shell_ns = ns(sched_start, earliest)?
            .checked_add(ns(latest, sched_end)?)
            .ok_or("scheduler overflow")?;
        let residual_helper_ns = helper_total_ns
            .checked_sub(fd_resolve_proof_ns)
            .and_then(|n| n.checked_sub(read_critical_span_ns))
            .and_then(|n| n.checked_sub(scheduler_shell_ns))
            .ok_or("non-additive helper")?;
        Ok((
            Totals {
                helper_total_ns,
                fd_resolve_proof_ns,
                read_critical_span_ns,
                scheduler_shell_ns,
                residual_helper_ns,
            },
            reads,
        ))
    }
}

use parking_lot::Mutex;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Phase {
    Warmup,
    Measured,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct TreatmentState {
    pub(crate) active_slots: usize,
    pub(crate) source_set_width: usize,
    pub(crate) slot_indices: [Option<usize>; WIDTH],
    pub(crate) first_map: [Option<bool>; WIDTH],
    pub(crate) map_wait_us: u64,
    pub(crate) remap_wait_us: u64,
    pub(crate) mapped_leases_complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Record {
    pub(crate) phase: Phase,
    pub(crate) request_index: usize,
    pub(crate) source_call_ordinal_within_request: usize,
    pub(crate) global_source_call_ordinal: usize,
    pub(crate) arm: Arm,
    pub(crate) helper: Helper,
    pub(crate) source_set_width: usize,
    pub(crate) ordered_expert_ids: [u32; WIDTH],
    pub(crate) returned_bytes: Option<u64>,
    pub(crate) source_error: Option<String>,
    pub(crate) timing_error: Option<String>,
    pub(crate) timing: Totals,
    pub(crate) raw_helper_total_ns: Option<u64>,
    pub(crate) scheduler_start_ns: u64,
    pub(crate) scheduler_end_ns: u64,
    pub(crate) caller_helper_ns: u64,
    pub(crate) reads: [Read; WIDTH],
    pub(crate) batch_max_read_wall_ns: u64,
    pub(crate) batch_sum_read_wall_ns: u64,
    pub(crate) diagnostic_retry_count: u64,
    pub(crate) treatment_pre_helper: Option<TreatmentState>,
    pub(crate) diagnostic_record_commit_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct StoreSnapshot {
    pub(crate) arm: Arm,
    pub(crate) capacity: usize,
    pub(crate) overflow: u64,
    pub(crate) context_errors: u64,
    pub(crate) request_begins: Vec<(Phase, usize)>,
    pub(crate) request_source_id_witnesses: Vec<(Phase, usize, String)>,
    pub(crate) records: Vec<Record>,
}

struct Store {
    snapshot: StoreSnapshot,
    context: Option<(Phase, usize)>,
    ordinal: usize,
}

/// The sole lock is acquired before requests or AFTER source helper completion.
/// Neither a helper nor any of its workers receives this observer.
pub(crate) struct Observer {
    inner: Mutex<Store>,
}
impl Observer {
    pub(crate) fn new(arm: Arm, capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Store {
                snapshot: StoreSnapshot {
                    arm,
                    capacity,
                    overflow: 0,
                    context_errors: 0,
                    request_begins: Vec::with_capacity(4),
                    request_source_id_witnesses: Vec::with_capacity(4),
                    records: Vec::with_capacity(capacity),
                },
                context: None,
                ordinal: 0,
            }),
        })
    }
    pub(crate) fn begin_request(&self, phase: Phase, index: usize) {
        let mut s = self.inner.lock();
        s.context = Some((phase, index));
        s.ordinal = 0;
        s.snapshot.request_begins.push((phase, index));
    }
    pub(crate) fn finish_request(&self, existing_cumulative_ids: Option<String>) {
        let mut s = self.inner.lock();
        if let (Some((phase, index)), Some(hash)) = (s.context, existing_cumulative_ids) {
            s.snapshot
                .request_source_id_witnesses
                .push((phase, index, hash));
        } else {
            s.snapshot.context_errors = s.snapshot.context_errors.saturating_add(1);
        }
    }
    pub(crate) fn snapshot(&self) -> StoreSnapshot {
        self.inner.lock().snapshot.clone()
    }
    pub(crate) fn commit(
        &self,
        ids: &[u32],
        helper: Helper,
        raw: &RawBatch,
        caller_start: Instant,
        caller_end: Instant,
        result: &std::io::Result<usize>,
        treatment_pre_helper: Option<TreatmentState>,
    ) {
        // This timestamp starts a DIFFERENT interval, after caller_end/raw.end.
        let commit_start = Instant::now();
        let mut s = self.inner.lock();
        let Some((phase, request_index)) = s.context else {
            s.snapshot.context_errors = s.snapshot.context_errors.saturating_add(1);
            return;
        };
        if s.snapshot.records.len() >= s.snapshot.capacity {
            s.snapshot.overflow = s.snapshot.overflow.saturating_add(1);
            return;
        }
        let mut ordered_expert_ids = [0; WIDTH];
        if ids.len() <= WIDTH {
            ordered_expert_ids[..ids.len()].copy_from_slice(ids);
        }
        let rebuilt = raw.reconstruct(ids.len());
        let timing_error = rebuilt.as_ref().err().map(|e| e.to_string());
        let raw_helper_total_ns = raw.start.zip(raw.end).and_then(|(a, b)| ns(a, b).ok());
        let (timing, reads) = rebuilt.unwrap_or_else(|_| {
            // Keep retry/failure evidence even if a failed/panicked worker lacks
            // a completion timestamp. Never turn incomplete timing into zero
            // retries or an apparently successful source record.
            let reads = raw.reads.map(|r| {
                let attempts = r.attempt_starts.iter().filter(|t| t.is_some()).count();
                Read {
                    retry_attempt_count: attempts.saturating_sub(1) as u32,
                    transient_events: r.transient_events,
                    breaker_event: r.breaker_event,
                    success: r.success,
                    ..Default::default()
                }
            });
            (Totals::default(), reads)
        });
        let sum = reads
            .iter()
            .try_fold(0u64, |n, r| n.checked_add(r.wrapper_wall_ns));
        let max = reads.iter().map(|r| r.wrapper_wall_ns).max().unwrap_or(0);
        let mut record = Record {
            phase,
            request_index,
            source_call_ordinal_within_request: s.ordinal,
            global_source_call_ordinal: s.snapshot.records.len(),
            arm: s.snapshot.arm,
            helper,
            source_set_width: ids.len(),
            ordered_expert_ids,
            returned_bytes: result.as_ref().ok().and_then(|n| u64::try_from(*n).ok()),
            source_error: result.as_ref().err().map(ToString::to_string),
            timing_error,
            timing,
            raw_helper_total_ns,
            reads,
            scheduler_start_ns: raw
                .start
                .zip(raw.scheduler_start)
                .and_then(|(a, b)| ns(a, b).ok())
                .unwrap_or(0),
            scheduler_end_ns: raw
                .start
                .zip(raw.scheduler_end)
                .and_then(|(a, b)| ns(a, b).ok())
                .unwrap_or(0),
            caller_helper_ns: ns(caller_start, caller_end).unwrap_or(0),
            batch_max_read_wall_ns: max,
            batch_sum_read_wall_ns: sum.unwrap_or(0),
            diagnostic_retry_count: reads.iter().map(|r| u64::from(r.retry_attempt_count)).sum(),
            treatment_pre_helper,
            diagnostic_record_commit_ns: 0,
        };
        if sum.is_none() {
            record.timing_error = Some("read sum overflow".into());
        }
        s.ordinal += 1;
        s.snapshot.records.push(record); // capacity was allocated before any request
        if let Some(last) = s.snapshot.records.last_mut() {
            last.diagnostic_record_commit_ns = ns(commit_start, Instant::now()).unwrap_or(u64::MAX);
        }
    }
}

impl Totals {
    fn fields(self) -> [u64; 5] {
        [
            self.helper_total_ns,
            self.fd_resolve_proof_ns,
            self.read_critical_span_ns,
            self.scheduler_shell_ns,
            self.residual_helper_ns,
        ]
    }
    fn from_fields(v: [u64; 5]) -> Self {
        Self {
            helper_total_ns: v[0],
            fd_resolve_proof_ns: v[1],
            read_critical_span_ns: v[2],
            scheduler_shell_ns: v[3],
            residual_helper_ns: v[4],
        }
    }
    fn add(self, other: Self) -> Result<Self, &'static str> {
        let mut v = [0; 5];
        for (i, (a, b)) in self.fields().into_iter().zip(other.fields()).enumerate() {
            v[i] = a.checked_add(b).ok_or("total overflow")?;
        }
        Ok(Self::from_fields(v))
    }
    fn exact(self) -> bool {
        self.fields()[1..]
            .iter()
            .try_fold(0u64, |n, v| n.checked_add(*v))
            == Some(self.helper_total_ns)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Comparison {
    control: Totals,
    treatment: Totals,
    // Signed i128 nanoseconds preserve the full difference between u64 totals.
    deltas_ns: [i128; 5],
    percentages_of_control_helper: [Option<Rational>; 5],
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Rational {
    numerator: i128,
    denominator: u64,
}
fn compare(control: Totals, treatment: Totals) -> Result<Comparison, &'static str> {
    if !control.exact() || !treatment.exact() {
        return Err("non-additive arm total");
    }
    let mut deltas_ns = [0; 5];
    for (i, (c, t)) in control
        .fields()
        .into_iter()
        .zip(treatment.fields())
        .enumerate()
    {
        deltas_ns[i] = i128::from(t) - i128::from(c);
    }
    if deltas_ns[0] != deltas_ns[1..].iter().sum::<i128>() {
        return Err("non-additive delta");
    }
    Ok(Comparison {
        control,
        treatment,
        deltas_ns,
        percentages_of_control_helper: deltas_ns.map(|n| {
            (control.helper_total_ns > 0).then_some(Rational {
                numerator: 100 * n,
                denominator: control.helper_total_ns,
            })
        }),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Stratum {
    kind: String,
    key: String,
    control_records: usize,
    treatment_records: usize,
    comparison: Comparison,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Stream {
    arm: Arm,
    phase: Phase,
    records: usize,
    reads: u64,
    bytes: u64,
    ordered_ids_sha256: String,
    ordered_widths_sha256: String,
    totals: Totals,
    caller_helper_ns: u64,
    diagnostic_record_commit_ns: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Analysis {
    pub(crate) disposition: String,
    source_gap: String,
    components: Vec<String>,
    errors: Vec<String>,
    external_retry_warning_occurrences: Option<u64>,
    primary: Option<Comparison>,
    streams: Vec<Stream>,
    strata: Vec<Stratum>,
    decision: String,
}

const COMPONENTS: [&str; 4] = [
    "FD_PROOF_DOMINANT",
    "READ_CRITICAL_DOMINANT",
    "SCHEDULER_SHELL_DOMINANT",
    "RESIDUAL_HELPER_DOMINANT",
];

/// Frozen integer decision tree. Display percentages never enter this function.
fn classify(
    primary: &Comparison,
    requests: &[[i128; 5]; 3],
    halves: &[[i128; 5]; 2],
) -> (&'static str, &'static str, Vec<String>) {
    let d = primary.deltas_ns;
    let c = i128::from(primary.control.helper_total_ns);
    if c == 0 {
        return ("NON_AUTHORITATIVE", "NON_AUTHORITATIVE", vec![]);
    }
    if 100 * d[0] <= -c {
        return ("SOURCE_GAP_REVERSED", "SOURCE_GAP_REVERSED", vec![]);
    }
    if 100 * d[0] < c {
        return (
            "SOURCE_GAP_NOT_REPRODUCED",
            "SOURCE_GAP_NOT_REPRODUCED",
            vec![],
        );
    }
    if 100 * d[0] < 3 * c {
        return ("DIRECTIONAL_ONLY", "DIRECTIONAL_SOURCE_GAP", vec![]);
    }
    let consistent = |i: usize| requests.iter().filter(|r| r[i] > 0).count() >= 2;
    let reaches60 = |i: usize| d[i] > 0 && 100 * d[i] >= 60 * d[0];
    let dominant: Vec<_> = (1..5)
        .filter(|&i| reaches60(i) && consistent(i) && halves.iter().all(|r| r[i] > 0))
        .collect();
    if dominant.len() == 1 {
        let label = COMPONENTS[dominant[0] - 1];
        return (label, "MATERIAL_SOURCE_GAP", vec![label.into()]);
    }
    // Offsetting negative deltas can leave more than one >=60% component.
    // The freeze specifies a single dominant component, not an arbitrary tie.
    if dominant.len() > 1 {
        return ("AMBIGUOUS_DECOMPOSITION", "MATERIAL_SOURCE_GAP", vec![]);
    }
    if !(1..5).any(reaches60) {
        let mut positive: Vec<_> = (1..5).filter(|&i| d[i] > 0).collect();
        positive.sort_by(|&a, &b| d[b].cmp(&d[a]).then(a.cmp(&b)));
        if positive.len() >= 2 {
            let (a, b) = (positive[0], positive[1]);
            if 100 * (d[a] + d[b]) >= 80 * d[0] && consistent(a) && consistent(b) {
                return (
                    "MIXED_TWO_COMPONENT",
                    "MATERIAL_SOURCE_GAP",
                    vec![COMPONENTS[a - 1].into(), COMPONENTS[b - 1].into()],
                );
            }
        }
    }
    ("AMBIGUOUS_DECOMPOSITION", "MATERIAL_SOURCE_GAP", vec![])
}

fn decision(label: &str) -> &'static str {
    match label {
        "FD_PROOF_DOMINANT"=>"Inspect remaining treatment-specific fd/proof integration; HMA-1A and fd-cache policy remain frozen pending a separate design.",
        "READ_CRITICAL_DOMINANT"=>"Review the actual production storage/runtime interaction under treatment state; no raw-backing replay.",
        "SCHEDULER_SHELL_DOMINANT"=>"Review source-helper/backend integration preserving routing, source QD and exact source work.",
        "RESIDUAL_HELPER_DOMINANT"=>"Inspect the measured residual scaffolding before proposing code.",
        "MIXED_TWO_COMPONENT"=>"Review only the two named components in one design, or close if expected gain is too small.",
        "NON_AUTHORITATIVE"=>"No performance interpretation; reconcile authority failures and external retry evidence. Do not retry the FIRST.",
        _=>"No further HMA micro-discriminator by default; close this source lane unless evidence identifies one explicit previously untested production mechanism for separate review.",
    }
}

/// Half/quartile membership is by ordinal RECORD COUNT independently per arm.
/// Floor(p*n/groups) boundaries make odd counts explicit and deterministic.
fn bucket(ordinal: usize, count: usize, groups: usize) -> usize {
    (1..groups)
        .find(|&g| ordinal < count * g / groups)
        .map_or(groups - 1, |g| g - 1)
}

fn validate_record(r: &Record) -> Result<(), &'static str> {
    let k = r.source_set_width;
    if k == 0 || k > WIDTH || r.ordered_expert_ids[k..].iter().any(|&id| id != 0) {
        return Err("width/unused ID corruption");
    }
    if r.ordered_expert_ids[..k]
        .iter()
        .enumerate()
        .any(|(i, id)| r.ordered_expert_ids[..i].contains(id))
    {
        return Err("duplicate source ID within batch");
    }
    if r.source_error.is_some()
        || r.timing_error.is_some()
        || r.returned_bytes != Some(k as u64 * crate::gpu_native_source_upload::FULL as u64)
    {
        return Err("source result/timing failure");
    }
    match (r.arm, r.helper) {
        (Arm::Control, Helper::ControlSingleFileExt) if k == 1 => {}
        (Arm::Control, Helper::ControlBatchScopedFileExt) if k > 1 => {}
        (Arm::Treatment, Helper::TreatmentAlignedBatchScopedFileExt) => {}
        _ => return Err("actual helper/arm/width mismatch"),
    }
    if !r.timing.exact()
        || r.raw_helper_total_ns != Some(r.timing.helper_total_ns)
        || r.caller_helper_ns < r.timing.helper_total_ns
    {
        return Err("helper additive/caller corruption");
    }
    let (mut first, mut last, mut sum, mut max) = (u64::MAX, 0, 0u64, 0);
    let mut retry_count = 0;
    for (i, read) in r.reads.iter().enumerate() {
        if i >= k {
            if serde_json::to_value(read).ok() != serde_json::to_value(Read::default()).ok() {
                return Err("unused read corruption");
            }
            continue;
        }
        if !read.success
            || read.breaker_event
            || read.transient_events != 0
            || read.retry_attempt_count != 0
        {
            return Err("read/retry/breaker authority failure");
        }
        retry_count += u64::from(read.retry_attempt_count);
        if read.wrapper_end_ns.checked_sub(read.wrapper_start_ns) != Some(read.wrapper_wall_ns)
            || read.attempt_wall_ns[0].is_none()
            || read.attempt_wall_ns[1..].iter().any(Option::is_some)
            || read.attempt_wall_ns[0].is_some_and(|n| n > read.wrapper_wall_ns)
        {
            return Err("read timing corruption");
        }
        first = first.min(read.wrapper_start_ns);
        last = last.max(read.wrapper_end_ns);
        sum = sum
            .checked_add(read.wrapper_wall_ns)
            .ok_or("read sum overflow")?;
        max = max.max(read.wrapper_wall_ns);
    }
    if retry_count != r.diagnostic_retry_count || retry_count != 0 {
        return Err("diagnostic retry count");
    }
    if last.checked_sub(first) != Some(r.timing.read_critical_span_ns)
        || sum != r.batch_sum_read_wall_ns
        || max != r.batch_max_read_wall_ns
    {
        return Err("read critical-span/sum/max corruption");
    }
    if r.timing.fd_resolve_proof_ns > r.scheduler_start_ns
        || r.scheduler_end_ns > r.timing.helper_total_ns
        || first < r.scheduler_start_ns
        || last > r.scheduler_end_ns
    {
        return Err("scheduler boundary corruption");
    }
    if (first - r.scheduler_start_ns).checked_add(r.scheduler_end_ns - last)
        != Some(r.timing.scheduler_shell_ns)
    {
        return Err("scheduler shell corruption");
    }
    match (r.arm, &r.treatment_pre_helper) {
        (Arm::Control, None) => {}
        (Arm::Treatment, Some(t)) => {
            if t.source_set_width != k
                || t.active_slots < k
                || t.active_slots > crate::gpu_native_source_upload::CAPACITY
                || !t.mapped_leases_complete
                || t.remap_wait_us > t.map_wait_us
            {
                return Err("treatment pre-helper state corruption");
            }
            for i in 0..WIDTH {
                if i < k {
                    if t.slot_indices[i]
                        .is_none_or(|n| n >= crate::gpu_native_source_upload::CAPACITY)
                        || t.first_map[i].is_none()
                        || t.slot_indices[..i].contains(&t.slot_indices[i])
                    {
                        return Err("treatment slot corruption");
                    }
                } else if t.slot_indices[i].is_some() || t.first_map[i].is_some() {
                    return Err("unused treatment slot corruption");
                }
            }
        }
        _ => return Err("treatment state missing/unexpected"),
    }
    Ok(())
}

fn total(records: &[&Record]) -> Result<Totals, &'static str> {
    records
        .iter()
        .try_fold(Totals::default(), |n, r| n.add(r.timing))
}
fn checked_sum(mut values: impl Iterator<Item = u64>) -> Result<u64, &'static str> {
    values.try_fold(0u64, |n, v| n.checked_add(v).ok_or("sum overflow"))
}
fn reconcile_timer(internal: u64, caller: u64) -> bool {
    // max(arm wall / 400, 5ms), without truncating the 0.25% term.
    let diff = u128::from(internal.abs_diff(caller));
    400 * diff <= u128::from(internal) || diff <= 5_000_000
}
fn json_u64(v: &serde_json::Value, key: &str) -> Result<u64, &'static str> {
    v[key].as_u64().ok_or("missing production integer")
}
fn hash_ids(records: &[&Record], widths: bool) -> String {
    let mut hash = Sha256::new();
    for r in records {
        if widths {
            hash.update((r.source_set_width as u64).to_le_bytes());
        } else {
            for id in &r.ordered_expert_ids[..r.source_set_width] {
                hash.update(id.to_le_bytes());
            }
        }
    }
    format!("{:x}", hash.finalize())
}

fn audit_stream(
    store: &StoreSnapshot,
    phase: Phase,
    p: &serde_json::Value,
) -> Result<Stream, &'static str> {
    let records: Vec<_> = store.records.iter().filter(|r| r.phase == phase).collect();
    let (upload, source, scheduler) = if phase == Phase::Warmup {
        (
            &p["warmup_upload"],
            &p["warmup_mechanism"],
            &p["warmup_production"],
        )
    } else {
        (&p["upload"], &p["mechanism"], &p["production"])
    };
    let reads = checked_sum(records.iter().map(|r| r.source_set_width as u64))?;
    let bytes = checked_sum(records.iter().map(|r| r.returned_bytes.unwrap_or(0)))?;
    let ordered_ids_sha256 = hash_ids(&records, false);
    let ordered_widths_sha256 = hash_ids(&records, true);
    if reads != json_u64(source, "source_nvme_reads")?
        || bytes != json_u64(source, "source_nvme_bytes")?
        || upload["ordered_nvme_ids_sha256"].as_str() != Some(&ordered_ids_sha256)
    {
        return Err("source stream count/bytes/ordered identity mismatch");
    }
    let batched: Vec<_> = records.iter().filter(|r| r.source_set_width > 1).collect();
    let batch_experts = checked_sum(batched.iter().map(|r| r.source_set_width as u64))?;
    if records.len() as u64
        != reads
            .checked_sub(batch_experts)
            .and_then(|n| n.checked_add(batched.len() as u64))
            .ok_or("source call accounting overflow")?
    {
        return Err("source call count reconstruction mismatch");
    }
    if batched.len() as u64 != json_u64(scheduler, "production_batch_successes")?
        || batch_experts != json_u64(scheduler, "production_batch_experts")?
    {
        return Err("source scheduler batch count/width mismatch");
    }
    if let (Some(min), Some(max)) = (
        batched.iter().map(|r| r.source_set_width).min(),
        batched.iter().map(|r| r.source_set_width).max(),
    ) {
        if min as u64 != json_u64(scheduler, "production_batch_width_min")?
            || max as u64 != json_u64(scheduler, "production_batch_width_max")?
        {
            return Err("source scheduler width range mismatch");
        }
    }
    let totals = total(&records)?;
    let caller_helper_ns = checked_sum(records.iter().map(|r| r.caller_helper_ns))?;
    if !reconcile_timer(totals.helper_total_ns, caller_helper_ns) {
        return Err("caller/internal helper timer reconciliation");
    }
    if store.arm == Arm::Treatment {
        let fused = json_u64(upload, "fused_source_us")?
            .checked_mul(1000)
            .ok_or("fused timer overflow")?;
        if !reconcile_timer(totals.helper_total_ns, fused) {
            return Err("fused_source_us timer reconciliation");
        }
        if reads != json_u64(upload, "direct_source_reads")?
            || bytes != json_u64(upload, "direct_source_bytes")?
        {
            return Err("treatment direct source reconciliation");
        }
    }
    Ok(Stream {
        arm: store.arm,
        phase,
        records: records.len(),
        reads,
        bytes,
        ordered_ids_sha256,
        ordered_widths_sha256,
        totals,
        caller_helper_ns,
        diagnostic_record_commit_ns: checked_sum(
            records.iter().map(|r| r.diagnostic_record_commit_ns),
        )?,
    })
}

/// Reconstruct from raw records and retained production-v2 witnesses. Cached
/// display totals/classification in a JSON artifact are never trusted.
fn analyze_inner(
    stores: &[StoreSnapshot],
    production: &serde_json::Value,
) -> Result<
    (
        Comparison,
        Vec<Stream>,
        Vec<Stratum>,
        [[i128; 5]; 3],
        [[i128; 5]; 2],
    ),
    &'static str,
> {
    if production["schema"] != "mer.gpu-native-source-to-upload-copy-elision-production.v2"
        || production["qualification_pass"] != true
        || production["benchmark_complete"] != true
        || production["gates"]["passed"] != true
        || production["reconciliation"]["all_invariants_pass"] != true
    {
        return Err("existing production-v2 gates incomplete/failed");
    }
    if stores.len() != 2 || stores[0].arm != Arm::Control || stores[1].arm != Arm::Treatment {
        return Err("missing/duplicate arm");
    }
    for key in ["warmup_production", "production"] {
        if production["control"][key].is_null()
            || production["control"][key] != production["treatment"][key]
        {
            return Err("existing source scheduler snapshots differ");
        }
    }
    let mut streams = Vec::new();
    for (s, key) in stores.iter().zip(["control", "treatment"]) {
        if s.overflow != 0 || s.records.len() > s.capacity {
            return Err("record capacity overflow");
        }
        if s.context_errors != 0
            || s.request_begins
                != [
                    (Phase::Warmup, 0),
                    (Phase::Measured, 0),
                    (Phase::Measured, 1),
                    (Phase::Measured, 2),
                ]
        {
            return Err("request context mismatch");
        }
        let mut request_counts = BTreeMap::new();
        let mut previous = None;
        for (ordinal, r) in s.records.iter().enumerate() {
            let key = (r.phase, r.request_index);
            if !s.request_begins.contains(&key) || previous.is_some_and(|p| p > key) {
                return Err("request order corruption");
            }
            previous = Some(key);
            let count = request_counts.entry(key).or_insert(0usize);
            if r.arm != s.arm
                || r.global_source_call_ordinal != ordinal
                || r.source_call_ordinal_within_request != *count
            {
                return Err("missing/duplicate source record or ordinal corruption");
            }
            *count += 1;
            validate_record(r)?;
        }
        if s.request_source_id_witnesses.len() != 4 {
            return Err("missing request source identity witnesses");
        }
        for ((phase, index), (wp, wi, hash)) in
            s.request_begins.iter().zip(&s.request_source_id_witnesses)
        {
            if (phase, index) != (wp, wi) {
                return Err("request identity witness order");
            }
            let cumulative: Vec<_> = s
                .records
                .iter()
                .filter(|r| r.phase == *phase && r.request_index <= *index)
                .collect();
            if hash_ids(&cumulative, false) != *hash {
                return Err("request source identity witness mismatch");
            }
        }
        let runs = production[key]["benchmark"]["per_run_results"]
            .as_array()
            .ok_or("missing production requests")?;
        if runs.len() != 3 {
            return Err("measured request count");
        }
        for (index, run) in runs.iter().enumerate() {
            let records: Vec<_> = s
                .records
                .iter()
                .filter(|r| r.phase == Phase::Measured && r.request_index == index)
                .collect();
            let bytes = checked_sum(records.iter().map(|r| r.returned_bytes.unwrap_or(0)))?;
            let counters = &run["counters"]["engine_storage_delta"];
            if json_u64(run, "run_index")? != index as u64
                || json_u64(counters, "nvme_read_operations")? != records.len() as u64
                || json_u64(counters, "nvme_bytes_read")? != bytes
            {
                return Err("per-request production read/byte reconciliation");
            }
        }
        for phase in [Phase::Warmup, Phase::Measured] {
            streams.push(audit_stream(s, phase, &production[key])?);
        }
    }
    let c = &stores[0].records;
    let t = &stores[1].records;
    if c.len() != t.len() {
        return Err("control/treatment source request counts mismatch");
    }
    for (a, b) in c.iter().zip(t) {
        if (
            a.phase,
            a.request_index,
            a.source_call_ordinal_within_request,
            a.global_source_call_ordinal,
            a.source_set_width,
            a.ordered_expert_ids,
            a.returned_bytes,
        ) != (
            b.phase,
            b.request_index,
            b.source_call_ordinal_within_request,
            b.global_source_call_ordinal,
            b.source_set_width,
            b.ordered_expert_ids,
            b.returned_bytes,
        ) {
            return Err("control/treatment source stream mismatch");
        }
    }
    let cm: Vec<_> = c.iter().filter(|r| r.phase == Phase::Measured).collect();
    let tm: Vec<_> = t.iter().filter(|r| r.phase == Phase::Measured).collect();
    let primary = compare(total(&cm)?, total(&tm)?)?;
    if primary.control.helper_total_ns == 0 {
        return Err("empty measured helper wall");
    }
    // Reconcile complete-arm caller totals as well as the separate phases.
    for store in stores {
        let all: Vec<_> = store.records.iter().collect();
        if !reconcile_timer(
            total(&all)?.helper_total_ns,
            checked_sum(all.iter().map(|r| r.caller_helper_ns))?,
        ) {
            return Err("complete arm caller timer mismatch");
        }
    }
    let mut strata = Vec::new();
    let mut requests = [[0; 5]; 3];
    let mut halves = [[0; 5]; 2];
    let mut add = |kind: &str,
                   key: String,
                   a: Vec<&Record>,
                   b: Vec<&Record>|
     -> Result<[i128; 5], &'static str> {
        let comparison = compare(total(&a)?, total(&b)?)?;
        let d = comparison.deltas_ns;
        strata.push(Stratum {
            kind: kind.into(),
            key,
            control_records: a.len(),
            treatment_records: b.len(),
            comparison,
        });
        Ok(d)
    };
    for i in 0..3 {
        let a: Vec<_> = cm
            .iter()
            .copied()
            .filter(|r| r.request_index == i)
            .collect();
        let b: Vec<_> = tm
            .iter()
            .copied()
            .filter(|r| r.request_index == i)
            .collect();
        if a.is_empty() || b.is_empty() {
            return Err("missing measured request records");
        }
        requests[i] = add("request_index", i.to_string(), a, b)?;
    }
    for groups in [2, 4] {
        for i in 0..groups {
            let a = cm
                .iter()
                .enumerate()
                .filter(|(n, _)| bucket(*n, cm.len(), groups) == i)
                .map(|(_, r)| *r)
                .collect();
            let b = tm
                .iter()
                .enumerate()
                .filter(|(n, _)| bucket(*n, tm.len(), groups) == i)
                .map(|(_, r)| *r)
                .collect();
            let d = add(
                if groups == 2 {
                    "ordinal_half"
                } else {
                    "ordinal_quartile"
                },
                (i + 1).to_string(),
                a,
                b,
            )?;
            if groups == 2 {
                halves[i] = d;
            }
        }
    }
    for width in 1..=WIDTH {
        let a: Vec<_> = cm
            .iter()
            .copied()
            .filter(|r| r.source_set_width == width)
            .collect();
        let b: Vec<_> = tm
            .iter()
            .copied()
            .filter(|r| r.source_set_width == width)
            .collect();
        if !a.is_empty() || !b.is_empty() {
            add("source_set_width", width.to_string(), a, b)?;
        }
    }
    let mut buckets: BTreeMap<String, Vec<&Record>> = BTreeMap::new();
    for r in &tm {
        if let Some(state) = &r.treatment_pre_helper {
            let first = state.first_map[..r.source_set_width]
                .iter()
                .filter(|v| **v == Some(true))
                .count();
            let key = if first == 0 {
                "all_remap"
            } else if first == r.source_set_width {
                "all_first_map"
            } else {
                "mixed_first_map_remap"
            };
            buckets.entry(key.into()).or_default().push(r);
        }
    }
    for (key, records) in buckets {
        add("treatment_first_map_remap", key, vec![], records)?;
    }
    for slot in 0..crate::gpu_native_source_upload::CAPACITY {
        for first in [true, false] {
            let records: Vec<_> = tm
                .iter()
                .copied()
                .filter(|r| {
                    r.treatment_pre_helper.as_ref().is_some_and(|s| {
                        s.slot_indices
                            .iter()
                            .zip(s.first_map)
                            .any(|(i, f)| *i == Some(slot) && f == Some(first))
                    })
                })
                .collect();
            if !records.is_empty() {
                add(
                    "treatment_slot_reuse",
                    format!("slot={slot},first_map={first}"),
                    vec![],
                    records,
                )?;
            }
        }
    }
    Ok((primary, streams, strata, requests, halves))
}

pub(crate) fn analyze(
    stores: &[StoreSnapshot],
    production: &serde_json::Value,
    warnings: Option<u64>,
) -> Analysis {
    let mut result = Analysis {
        disposition: "NON_AUTHORITATIVE".into(),
        source_gap: "NON_AUTHORITATIVE".into(),
        components: vec![],
        errors: vec![],
        external_retry_warning_occurrences: warnings,
        primary: None,
        streams: vec![],
        strata: vec![],
        decision: decision("NON_AUTHORITATIVE").into(),
    };
    match analyze_inner(stores, production) {
        Ok((primary, streams, strata, requests, halves)) => {
            let (label, gap, components) = classify(&primary, &requests, &halves);
            result.primary = Some(primary);
            result.streams = streams;
            result.strata = strata;
            match warnings {
                Some(0) => {
                    result.disposition = label.into();
                    result.source_gap = gap.into();
                    result.components = components;
                    result.decision = decision(label).into();
                }
                Some(_) => result
                    .errors
                    .push("external transient I/O error; retrying occurrence".into()),
                None => result
                    .errors
                    .push("completed external retry transcript audit pending".into()),
            }
        }
        Err(e) => result.errors.push(e.into()),
    }
    result
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Envelope {
    schema: String,
    mode: String,
    pub(crate) production_v2: serde_json::Value,
    pub(crate) observations: Vec<StoreSnapshot>,
    pub(crate) analysis: Analysis,
    helper_definitions: BTreeMap<String, String>,
    timing_definitions: BTreeMap<String, String>,
    external_retry_log_sha256: Option<String>,
    source_report_sha256: Option<String>,
    external_existing_breaker_retry_events: Option<u64>,
}
impl Envelope {
    pub(crate) fn new(production_v2: serde_json::Value, observations: Vec<StoreSnapshot>) -> Self {
        let analysis = analyze(&observations, &production_v2, None);
        Self { schema:SCHEMA.into(),mode:MODE.into(),production_v2,observations,analysis,external_retry_log_sha256:None,source_report_sha256:None,external_existing_breaker_retry_events:None,
            helper_definitions:BTreeMap::from([
                ("ControlSingleFileExt".into(),"NvmeStorage::read_expert_observed; original read_expert body; per-file fd, block_in_place, read_at_with_retries, FileExt::read_at".into()),
                ("ControlBatchScopedFileExt".into(),"NvmeStorage::read_experts_batch; per-file fds, block_in_place, scoped threads, ordered joins, read_at_with_retries, FileExt::read_at".into()),
                ("TreatmentAlignedBatchScopedFileExt".into(),"NvmeStorage::read_experts_batch_into_aligned_slices; resolved-fd HMA-1A proof, block_in_place, scoped threads, ordered joins, read_at_with_retries, FileExt::read_at".into())]),
            timing_definitions:BTreeMap::from([
                ("helper_total_ns".into(),"first helper timestamp through completion of the original helper body and local drops; stops before caller completion and diagnostic commit".into()),
                ("fd_resolve_proof_ns".into(),"helper entry through completion of all file/fd resolution and applicable HMA-1A proof".into()),
                ("read_critical_span_ns".into(),"earliest actual retry-wrapper entry through latest wrapper completion; never summed parallel read wall".into()),
                ("scheduler_shell_ns".into(),"before block_in_place through earliest read start plus latest read completion through block_in_place return; includes worker launches/joins/completion tail outside read critical span".into()),
                ("residual_helper_ns".into(),"helper_total_ns - fd_resolve_proof_ns - read_critical_span_ns - scheduler_shell_ns, exact integer nanoseconds".into()),
                ("caller_helper_ns".into(),"immediately around the actual awaited source helper, before record commit; max(0.25% of internal helper wall,5ms) reconciliation per phase and arm".into()),
                ("legacy_control_timing".into(),"individual_source_service_us and batch wall include additional engine/cache/pool work; they have different scopes and are not forced equal".into()),
                ("fused_source_us".into(),"existing treatment caller interval around aligned source helper; microsecond quantization retained; reconciled to internal totals".into()),
                ("ordinal_groups".into(),"measured-only ordinal record count, independently per arm; boundaries floor(p*N/groups); warmup excluded".into()),
                ("treatment_strata".into(),"treatment-only descriptive totals, no matched-source replay or new reuse generation; slot strata overlap across multi-slot records".into())]),
        }
    }
}

pub(crate) fn retry_warning_count(bytes: &[u8]) -> u64 {
    const PATTERN: &[u8] = b"transient I/O error; retrying";
    bytes
        .windows(PATTERN.len())
        .filter(|w| *w == PATTERN)
        .count() as u64
}

fn existing_breaker_retry_event_count(bytes: &[u8]) -> u64 {
    // These are existing production messages, not newly emitted diagnostics.
    [
        b"expert fetch recovered after retry".as_slice(),
        b"expert fetch failed; will retry".as_slice(),
        b"circuit breaker".as_slice(),
    ]
    .into_iter()
    .map(|pattern| {
        bytes
            .windows(pattern.len())
            .filter(|w| *w == pattern)
            .count() as u64
    })
    .sum()
}

/// Offline, non-consuming finalization AFTER the source process and its full
/// external stderr/stdout capture have completed. Never constructs a runtime.
/// Retain a content hash of the exact supplied log bytes; callers must supply
/// the complete transcript, not a selected excerpt. The original run artifact
/// is immutable and output must use a new path.
pub(crate) fn audit_command(
    input: &std::path::Path,
    log: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if input == output || output.exists() {
        return Err("audit output must be a new artifact path".into());
    }
    let bytes = std::fs::read(input)?;
    let mut report: Envelope = serde_json::from_slice(&bytes)?;
    if report.schema != SCHEMA || report.mode != MODE {
        return Err("unexpected decomposition schema/mode".into());
    }
    let log_bytes = std::fs::read(log)?;
    if log_bytes.is_empty() {
        return Err("completed external transcript must not be empty".into());
    }
    report.analysis = analyze(
        &report.observations,
        &report.production_v2,
        Some(retry_warning_count(&log_bytes)),
    );
    report.external_retry_log_sha256 = Some(format!("{:x}", Sha256::digest(&log_bytes)));
    report.source_report_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
    let events = existing_breaker_retry_event_count(&log_bytes);
    report.external_existing_breaker_retry_events = Some(events);
    if events != 0 {
        report.analysis.disposition = "NON_AUTHORITATIVE".into();
        report.analysis.source_gap = "NON_AUTHORITATIVE".into();
        report.analysis.components.clear();
        report
            .analysis
            .errors
            .push("external existing breaker/fetch-retry event".into());
        report.analysis.decision = decision("NON_AUTHORITATIVE").into();
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    if report.analysis.disposition == "NON_AUTHORITATIVE" {
        return Err("HMA-1D remains NON_AUTHORITATIVE; see audited report".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    fn fixture_record(
        arm: Arm,
        phase: Phase,
        request: usize,
        ordinal: usize,
        global: usize,
        width: usize,
    ) -> Record {
        let t = if arm == Arm::Control {
            Totals::from_fields([1_000_000, 200_000, 400_000, 300_000, 100_000])
        } else {
            Totals::from_fields([1_050_000, 230_000, 410_000, 305_000, 105_000])
        };
        let mut ids = [0; WIDTH];
        let mut reads = [Read::default(); WIDTH];
        let scheduler_start_ns = t.fd_resolve_proof_ns + 10_000;
        let first = scheduler_start_ns + 100_000;
        let last = first + t.read_critical_span_ns;
        let scheduler_end_ns = last + t.scheduler_shell_ns - 100_000;
        for i in 0..width {
            ids[i] = (global * WIDTH + i) as u32;
            let a = first + i as u64 * 100;
            let b = last - (width - i - 1) as u64 * 100;
            reads[i] = Read {
                wrapper_start_ns: a,
                wrapper_end_ns: b,
                wrapper_wall_ns: b - a,
                attempt_wall_ns: [Some(b - a - 100), None, None],
                success: true,
                ..Default::default()
            };
        }
        let treatment_pre_helper = (arm == Arm::Treatment).then(|| {
            let mut state = TreatmentState {
                active_slots: width,
                source_set_width: width,
                mapped_leases_complete: true,
                map_wait_us: 50,
                remap_wait_us: if global == 0 { 0 } else { 50 },
                ..Default::default()
            };
            for i in 0..width {
                state.slot_indices[i] = Some(i);
                state.first_map[i] = Some(global == 0);
            }
            state
        });
        Record {
            phase,
            request_index: request,
            source_call_ordinal_within_request: ordinal,
            global_source_call_ordinal: global,
            arm,
            helper: if arm == Arm::Treatment {
                Helper::TreatmentAlignedBatchScopedFileExt
            } else if width == 1 {
                Helper::ControlSingleFileExt
            } else {
                Helper::ControlBatchScopedFileExt
            },
            source_set_width: width,
            ordered_expert_ids: ids,
            returned_bytes: Some(width as u64 * crate::gpu_native_source_upload::FULL as u64),
            source_error: None,
            timing_error: None,
            timing: t,
            raw_helper_total_ns: Some(t.helper_total_ns),
            scheduler_start_ns,
            scheduler_end_ns,
            caller_helper_ns: t.helper_total_ns + 100,
            reads,
            batch_max_read_wall_ns: reads.iter().map(|r| r.wrapper_wall_ns).max().unwrap(),
            batch_sum_read_wall_ns: reads.iter().map(|r| r.wrapper_wall_ns).sum(),
            diagnostic_retry_count: 0,
            treatment_pre_helper,
            diagnostic_record_commit_ns: 80,
        }
    }
    fn fixture() -> (Vec<StoreSnapshot>, serde_json::Value) {
        let mut stores = Vec::new();
        let mut p = json!({"schema":"mer.gpu-native-source-to-upload-copy-elision-production.v2","qualification_pass":true,"benchmark_complete":true,"gates":{"passed":true},"reconciliation":{"all_invariants_pass":true}});
        for (arm, key) in [(Arm::Control, "control"), (Arm::Treatment, "treatment")] {
            let begins = vec![
                (Phase::Warmup, 0),
                (Phase::Measured, 0),
                (Phase::Measured, 1),
                (Phase::Measured, 2),
            ];
            let mut records = Vec::new();
            for &(phase, request) in &begins {
                for (ordinal, width) in [1, 2, 3, 8].into_iter().enumerate() {
                    records.push(fixture_record(
                        arm,
                        phase,
                        request,
                        ordinal,
                        records.len(),
                        width,
                    ));
                }
            }
            let witnesses = begins
                .iter()
                .map(|&(phase, index)| {
                    let cumulative: Vec<_> = records
                        .iter()
                        .filter(|r| r.phase == phase && r.request_index <= index)
                        .collect();
                    (phase, index, hash_ids(&cumulative, false))
                })
                .collect();
            let store = StoreSnapshot {
                arm,
                capacity: records.len(),
                overflow: 0,
                context_errors: 0,
                request_begins: begins,
                request_source_id_witnesses: witnesses,
                records,
            };
            let mut arm_json = json!({});
            for phase in [Phase::Warmup, Phase::Measured] {
                let r: Vec<_> = store.records.iter().filter(|r| r.phase == phase).collect();
                let reads = r.iter().map(|r| r.source_set_width as u64).sum::<u64>();
                let bytes = reads * crate::gpu_native_source_upload::FULL as u64;
                let source = json!({"source_nvme_reads":reads,"source_nvme_bytes":bytes});
                let scheduler = json!({"production_batch_successes":r.iter().filter(|r|r.source_set_width>1).count(),"production_batch_experts":r.iter().filter(|r|r.source_set_width>1).map(|r|r.source_set_width as u64).sum::<u64>(),"production_batch_width_min":2,"production_batch_width_max":8});
                let upload = json!({"ordered_nvme_ids_sha256":hash_ids(&r,false),"fused_source_us":r.iter().map(|r|r.timing.helper_total_ns/1000).sum::<u64>(),"direct_source_reads":reads,"direct_source_bytes":bytes});
                let keys = if phase == Phase::Warmup {
                    ["warmup_mechanism", "warmup_production", "warmup_upload"]
                } else {
                    ["mechanism", "production", "upload"]
                };
                arm_json[keys[0]] = source;
                arm_json[keys[1]] = scheduler;
                arm_json[keys[2]] = upload;
            }
            arm_json["benchmark"] = json!({"per_run_results":(0..3).map(|i|{
                let records:Vec<_>=store.records.iter().filter(|r|r.phase==Phase::Measured&&r.request_index==i).collect();
                json!({"run_index":i,"counters":{"engine_storage_delta":{"nvme_read_operations":records.len(),"nvme_bytes_read":records.iter().map(|r|r.returned_bytes.unwrap()).sum::<u64>()}}})
            }).collect::<Vec<_>>()});
            p[key] = arm_json;
            stores.push(store);
        }
        (stores, p)
    }
    fn comparison_with_delta(d: i128, components: [i128; 4]) -> Comparison {
        let control =
            Totals::from_fields([100_000_000, 25_000_000, 25_000_000, 25_000_000, 25_000_000]);
        assert_eq!(d, components.iter().sum::<i128>());
        let treatment = Totals::from_fields([
            (100_000_000i128 + d) as u64,
            (25_000_000i128 + components[0]) as u64,
            (25_000_000i128 + components[1]) as u64,
            (25_000_000i128 + components[2]) as u64,
            (25_000_000i128 + components[3]) as u64,
        ]);
        compare(control, treatment).unwrap()
    }
    fn label(c: &Comparison) -> &'static str {
        classify(c, &[c.deltas_ns; 3], &[c.deltas_ns; 2]).0
    }

    #[test]
    fn hma1d_exact_one_and_three_percent_boundaries_and_negative_one() {
        for (d, expected) in [
            (-1_000_001, "SOURCE_GAP_REVERSED"),
            (-1_000_000, "SOURCE_GAP_REVERSED"),
            (-999_999, "SOURCE_GAP_NOT_REPRODUCED"),
            (999_999, "SOURCE_GAP_NOT_REPRODUCED"),
            (1_000_000, "DIRECTIONAL_ONLY"),
            (1_000_001, "DIRECTIONAL_ONLY"),
            (2_999_999, "DIRECTIONAL_ONLY"),
            (3_000_000, "FD_PROOF_DOMINANT"),
            (3_000_001, "FD_PROOF_DOMINANT"),
        ] {
            let c = comparison_with_delta(d, [d, 0, 0, 0]);
            assert_eq!(label(&c), expected, "{d}");
        }
        assert_eq!(
            format!("{:.6}", 999_999f64 / 100_000_000.0 * 100.0),
            "0.999999"
        );
        // Both round to 1.000000% at six decimal places but classify differently.
        let base = 100_000_000_000u64;
        for (delta, expected) in [
            (999_999_999u64, "SOURCE_GAP_NOT_REPRODUCED"),
            (1_000_000_000, "DIRECTIONAL_ONLY"),
        ] {
            let c = compare(
                Totals::from_fields([base, base, 0, 0, 0]),
                Totals::from_fields([base + delta, base + delta, 0, 0, 0]),
            )
            .unwrap();
            assert_eq!(
                format!("{:.6}", delta as f64 / base as f64 * 100.0),
                "1.000000"
            );
            assert_eq!(label(&c), expected);
        }
    }
    #[test]
    fn hma1d_exact_sixty_percent_boundary() {
        for (fd, expected) in [
            (2_399_999, "AMBIGUOUS_DECOMPOSITION"),
            (2_400_000, "FD_PROOF_DOMINANT"),
            (2_400_001, "FD_PROOF_DOMINANT"),
        ] {
            let remaining = 4_000_000 - fd;
            let c = comparison_with_delta(
                4_000_000,
                [
                    fd,
                    remaining / 3,
                    remaining / 3,
                    remaining - 2 * (remaining / 3),
                ],
            );
            assert_eq!(label(&c), expected);
        }
    }
    #[test]
    fn hma1d_exact_eighty_percent_boundary() {
        for (second, expected) in [
            (1_399_999, "AMBIGUOUS_DECOMPOSITION"),
            (1_400_000, "MIXED_TWO_COMPONENT"),
            (1_400_001, "MIXED_TWO_COMPONENT"),
        ] {
            let left = 4_000_000 - 1_800_000 - second;
            let c =
                comparison_with_delta(4_000_000, [1_800_000, second, left / 2, left - left / 2]);
            assert_eq!(label(&c), expected);
        }
    }
    #[test]
    fn hma1d_dominance_requires_request_and_both_halves_consistency() {
        let c = comparison_with_delta(4_000_000, [3_000_000, 1_000_000, 0, 0]);
        let mut requests = [c.deltas_ns; 3];
        requests[0][1] = 0;
        requests[1][1] = -1;
        assert_eq!(
            classify(&c, &requests, &[c.deltas_ns; 2]).0,
            "AMBIGUOUS_DECOMPOSITION"
        );
        requests[1][1] = 1;
        assert_eq!(
            classify(&c, &requests, &[c.deltas_ns; 2]).0,
            "FD_PROOF_DOMINANT"
        );
        let mut halves = [c.deltas_ns; 2];
        halves[1][1] = 0;
        assert_eq!(
            classify(&c, &requests, &halves).0,
            "AMBIGUOUS_DECOMPOSITION"
        );
    }
    #[test]
    fn hma1d_mixed_requires_both_components_in_two_requests_no_fallback_from_failed_sixty() {
        let c = comparison_with_delta(4_000_000, [1_800_000, 1_400_000, 400_000, 400_000]);
        let mut requests = [c.deltas_ns; 3];
        requests[0][2] = 0;
        requests[1][2] = -1;
        assert_eq!(
            classify(&c, &requests, &[c.deltas_ns; 2]).0,
            "AMBIGUOUS_DECOMPOSITION"
        );
        let c = comparison_with_delta(4_000_000, [2_400_000, 1_000_000, 300_000, 300_000]);
        let mut halves = [c.deltas_ns; 2];
        halves[1][1] = 0;
        assert_eq!(
            classify(&c, &[c.deltas_ns; 3], &halves).0,
            "AMBIGUOUS_DECOMPOSITION"
        );
    }
    #[test]
    fn hma1d_all_four_dominant_labels_and_directional_gate() {
        for i in 0..4 {
            let mut ds = [0; 4];
            ds[i] = 4_000_000;
            assert_eq!(label(&comparison_with_delta(4_000_000, ds)), COMPONENTS[i]);
        }
        assert_eq!(
            label(&comparison_with_delta(2_000_000, [2_000_000, 0, 0, 0])),
            "DIRECTIONAL_ONLY"
        );
    }
    #[test]
    fn hma1d_raw_timestamps_reconstruct_critical_path_not_parallel_sum() {
        let start = Instant::now();
        let time = |n| Some(start + Duration::from_nanos(n));
        let mut raw = RawBatch {
            start: time(0),
            fd_end: time(100),
            scheduler_start: time(130),
            scheduler_end: time(900),
            end: time(1000),
            ..Default::default()
        };
        for (i, (a, b)) in [(200, 700), (300, 800)].into_iter().enumerate() {
            raw.reads[i] = RawRead {
                start: time(a),
                end: time(b),
                attempt_starts: [time(a + 10), None, None],
                attempt_ends: [time(b - 10), None, None],
                success: true,
                ..Default::default()
            };
        }
        let (totals, reads) = raw.reconstruct(2).unwrap();
        assert_eq!(totals, Totals::from_fields([1000, 100, 600, 170, 130]));
        assert_eq!(reads.iter().map(|r| r.wrapper_wall_ns).sum::<u64>(), 1000);
        assert_eq!(raw.reconstruct(1).unwrap().0.read_critical_span_ns, 500);
        raw.scheduler_end = time(750);
        assert!(raw.reconstruct(2).is_err());
        raw.scheduler_end = time(900);
        raw.reads[1].attempt_ends[0] = None;
        assert!(raw.reconstruct(2).is_err());
    }
    #[test]
    fn hma1d_exact_additive_totals_and_delta_synthetic_fixture() {
        let (s, p) = fixture();
        let a = analyze(&s, &p, Some(0));
        assert_eq!(a.disposition, "FD_PROOF_DOMINANT", "{:?}", a.errors);
        let c = a.primary.as_ref().unwrap();
        assert_eq!(c.control.helper_total_ns, 12_000_000);
        assert_eq!(c.treatment.helper_total_ns, 12_600_000);
        assert_eq!(c.deltas_ns, [600_000, 360_000, 120_000, 60_000, 60_000]);
        assert_eq!(c.deltas_ns[0], c.deltas_ns[1..].iter().sum());
        let percent = c.percentages_of_control_helper[0].unwrap();
        assert_eq!(percent.numerator, 60_000_000);
        assert_eq!(percent.denominator, 12_000_000);
        if let Ok(path) = std::env::var("MER_HMA1D_SYNTHETIC_JSON") {
            let mut envelope = Envelope::new(p, s);
            envelope.analysis = a;
            std::fs::write(path, serde_json::to_vec_pretty(&envelope).unwrap()).unwrap();
        }
    }
    #[test]
    fn hma1d_request_width_half_quartile_and_map_strata() {
        let (s, p) = fixture();
        let a = analyze(&s, &p, Some(0));
        assert!(a.errors.is_empty());
        for r in a.strata.iter().filter(|r| r.kind == "request_index") {
            assert_eq!(r.control_records, 4);
            assert_eq!(r.comparison.deltas_ns[0], 200_000);
        }
        for r in a.strata.iter().filter(|r| r.kind == "source_set_width") {
            assert_eq!(r.control_records, 3);
        }
        for r in a.strata.iter().filter(|r| r.kind == "ordinal_half") {
            assert_eq!(r.control_records, 6);
        }
        for r in a.strata.iter().filter(|r| r.kind == "ordinal_quartile") {
            assert_eq!(r.control_records, 3);
        }
        assert!(a
            .strata
            .iter()
            .any(|r| r.kind == "treatment_first_map_remap" && r.key == "all_remap"));
        assert!(a.strata.iter().any(|r| r.kind == "treatment_slot_reuse"));
        assert_eq!(
            (0..7).map(|i| bucket(i, 7, 4)).collect::<Vec<_>>(),
            [0, 1, 1, 2, 2, 3, 3]
        );
        assert_eq!(
            (0..7).map(|i| bucket(i, 7, 2)).collect::<Vec<_>>(),
            [0, 0, 0, 1, 1, 1, 1]
        );
    }
    #[test]
    fn hma1d_warmup_is_reconciled_but_excluded_from_primary() {
        let (mut s, mut p) = fixture();
        for r in s[1].records.iter_mut().filter(|r| r.phase == Phase::Warmup) {
            r.timing.helper_total_ns += 99_000_000;
            r.raw_helper_total_ns = Some(r.timing.helper_total_ns);
            r.timing.residual_helper_ns += 99_000_000;
            r.caller_helper_ns += 99_000_000;
        }
        p["treatment"]["warmup_upload"]["fused_source_us"] = json!(400_200);
        let a = analyze(&s, &p, Some(0));
        assert_eq!(a.disposition, "FD_PROOF_DOMINANT", "{:?}", a.errors);
        assert_eq!(a.primary.unwrap().deltas_ns[0], 600_000);
    }
    #[test]
    fn hma1d_missing_duplicate_overflow_and_corruption_rejected() {
        type Mutation = fn(&mut Vec<StoreSnapshot>, &mut serde_json::Value);
        let mutations: [Mutation; 24] = [
            |s, _| {
                s[0].records.remove(3);
            },
            |s, _| {
                let duplicate = s[0].records[3].clone();
                s[0].records.insert(3, duplicate);
            },
            |s, _| s[0].overflow = 1,
            |s, _| s[0].capacity = 0,
            |s, _| s[0].records[3].ordered_expert_ids[0] += 1,
            |s, _| s[0].records[0].ordered_expert_ids[7] = 1,
            |s, _| s[0].records[3].source_set_width = 9,
            |s, _| s[0].records[3].returned_bytes = Some(1),
            |s, _| s[0].records[3].timing.helper_total_ns += 1,
            |s, _| s[0].records[3].timing.read_critical_span_ns += 1,
            |s, _| s[0].records[3].scheduler_start_ns += 1,
            |s, _| s[0].records[3].reads[0].wrapper_wall_ns += 1,
            |s, _| s[0].records[3].batch_sum_read_wall_ns += 1,
            |s, _| s[0].records[3].global_source_call_ordinal += 1,
            |s, _| s[0].records[3].source_call_ordinal_within_request += 1,
            |s, _| s[0].records[3].request_index = 7,
            |s, _| s[0].records[3].phase = Phase::Measured,
            |s, _| s[0].records[3].helper = Helper::TreatmentAlignedBatchScopedFileExt,
            |s, _| s[1].records[3].treatment_pre_helper = None,
            |s, _| s[0].records[3].caller_helper_ns += 6_000_000,
            |_, p| p["control"]["warmup_upload"]["ordered_nvme_ids_sha256"] = json!("bad"),
            |_, p| p["control"]["warmup_production"]["production_batch_experts"] = json!(0),
            |_, p| p["control"]["warmup_mechanism"]["source_nvme_reads"] = json!(0),
            |_, p| p["gates"]["passed"] = json!(false),
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let (mut s, mut p) = fixture();
            mutate(&mut s, &mut p);
            assert_eq!(
                analyze(&s, &p, Some(0)).disposition,
                "NON_AUTHORITATIVE",
                "corruption {i}"
            );
        }
    }
    #[test]
    fn hma1d_retry_breaker_failure_and_external_warning_reject_authority() {
        for mode in 0..5 {
            let (mut s, p) = fixture();
            let r = &mut s[0].records[4];
            match mode {
                0 => r.diagnostic_retry_count = 1,
                1 => r.reads[0].transient_events = 1,
                2 => r.reads[0].breaker_event = true,
                3 => r.reads[0].retry_attempt_count = 1,
                _ => r.reads[0].success = false,
            };
            assert_eq!(analyze(&s, &p, Some(0)).disposition, "NON_AUTHORITATIVE");
        }
        let (s, p) = fixture();
        assert_eq!(analyze(&s, &p, None).disposition, "NON_AUTHORITATIVE");
        assert_eq!(analyze(&s, &p, Some(1)).disposition, "NON_AUTHORITATIVE");
        assert_eq!(
            retry_warning_count(
                b"ok\ntransient I/O error; retrying\ntransient I/O error; retrying"
            ),
            2
        );
        assert_eq!(retry_warning_count(b"ok\ncompleted"), 0);
    }
    #[test]
    fn hma1d_record_store_never_grows_and_rejects_overflow() {
        let observer = Observer::new(Arm::Control, 1);
        observer.begin_request(Phase::Warmup, 0);
        let now = Instant::now();
        let raw = RawBatch::default();
        observer.commit(
            &[1],
            Helper::ControlSingleFileExt,
            &raw,
            now,
            now,
            &Ok(2_658_304),
            None,
        );
        observer.commit(
            &[2],
            Helper::ControlSingleFileExt,
            &raw,
            now,
            now,
            &Ok(2_658_304),
            None,
        );
        let s = observer.snapshot();
        assert_eq!(s.records.len(), 1);
        assert_eq!(s.overflow, 1);
        assert!(s.records[0].timing_error.is_some());
        assert_eq!(observer.inner.lock().snapshot.records.capacity(), 1);
    }
    #[test]
    fn hma1d_checked_aggregation_rejects_overflow() {
        assert!(Totals::from_fields([u64::MAX, u64::MAX, 0, 0, 0])
            .add(Totals::from_fields([1, 1, 0, 0, 0]))
            .is_err());
        assert!(compare(Totals::from_fields([1, 2, 0, 0, 0]), Totals::default()).is_err());
    }
    #[test]
    fn hma1d_caller_reconciliation_exact_tolerance_boundary() {
        assert!(reconcile_timer(100_000_000, 105_000_000));
        assert!(!reconcile_timer(100_000_000, 105_000_001));
        assert!(reconcile_timer(4_000_000_000, 4_010_000_000));
        assert!(!reconcile_timer(4_000_000_000, 4_010_000_001));
    }
    #[test]
    fn hma1d_cli_reuses_production_config_and_offline_audit_has_no_runtime() {
        use clap::Parser;
        let cli = crate::Cli::try_parse_from([
            "mer",
            MODE,
            "--config",
            "frozen.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "out.json",
        ])
        .unwrap();
        assert_eq!(
            crate::startup_config_path(&cli.cmd),
            Some(std::path::Path::new("frozen.toml"))
        );
        let source = include_str!("gpu_native_physical_install_staging.rs");
        let runner = source
            .split("async fn run_physical_install_arm_inner(")
            .nth(1)
            .unwrap()
            .split("async fn run_arm(")
            .next()
            .unwrap();
        assert_eq!(
            runner
                .matches("crate::gpu_native_real_benchmark::execute_request(")
                .count(),
            2
        );
        assert_eq!(
            runner.matches("for index in 0..FROZEN_WARMUP_RUNS").count(),
            1
        );
        assert_eq!(
            runner
                .matches("for index in 0..FROZEN_MEASURED_RUNS")
                .count(),
            1
        );
        let source = include_str!("gpu_native_source_to_upload_copy_elision_production.rs");
        assert_eq!(source.matches("run_physical_install_arm_inner(").count(), 1);
        let this = include_str!("gpu_native_source_path_decomposition.rs");
        let audit = this
            .split("pub(crate) fn audit_command(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "construct_runtime(",
            "read_expert(",
            "device.poll(",
            "execute_request(",
        ] {
            assert!(!audit.contains(forbidden));
        }
    }
    #[test]
    fn hma1d_display_rounding_cannot_change_three_sixty_or_eighty_boundaries() {
        let base = 100_000_000_000_000u64;
        let control = Totals::from_fields([base, base / 4, base / 4, base / 4, base / 4]);
        let build = |d: u64, parts: [u64; 4]| {
            compare(
                control,
                Totals::from_fields([
                    base + d,
                    base / 4 + parts[0],
                    base / 4 + parts[1],
                    base / 4 + parts[2],
                    base / 4 + parts[3],
                ]),
            )
            .unwrap()
        };
        for (d, expected) in [
            (2_999_999_999_999, "DIRECTIONAL_ONLY"),
            (3_000_000_000_000, "FD_PROOF_DOMINANT"),
            (3_000_000_000_001, "FD_PROOF_DOMINANT"),
        ] {
            assert_eq!(format!("{:.6}", d as f64 / base as f64 * 100.0), "3.000000");
            assert_eq!(label(&build(d, [d, 0, 0, 0])), expected);
        }
        let d = 4_000_000_000_000u64;
        for (fd, expected) in [
            (2_399_999_999_999, "AMBIGUOUS_DECOMPOSITION"),
            (2_400_000_000_000, "FD_PROOF_DOMINANT"),
            (2_400_000_000_001, "FD_PROOF_DOMINANT"),
        ] {
            assert_eq!(format!("{:.6}", fd as f64 / d as f64 * 100.0), "60.000000");
            let left = d - fd;
            assert_eq!(
                label(&build(d, [fd, left / 3, left / 3, left - 2 * (left / 3)])),
                expected
            );
        }
        for (second, expected) in [
            (1_399_999_999_999, "AMBIGUOUS_DECOMPOSITION"),
            (1_400_000_000_000, "MIXED_TWO_COMPONENT"),
            (1_400_000_000_001, "MIXED_TWO_COMPONENT"),
        ] {
            let first = 1_800_000_000_000;
            let left = d - first - second;
            assert_eq!(
                format!("{:.6}", (first + second) as f64 / d as f64 * 100.0),
                "80.000000"
            );
            assert_eq!(
                label(&build(d, [first, second, left / 2, left - left / 2])),
                expected
            );
        }
    }

    #[test]
    fn hma1d_external_existing_breaker_and_fetch_retry_events_are_rejected() {
        assert_eq!(existing_breaker_retry_event_count(b"expert fetch failed; will retry\nexpert fetch recovered after retry\ndrive circuit breaker half-open"), 3);
        assert_eq!(
            existing_breaker_retry_event_count(b"completed clean source stream"),
            0
        );
    }

    #[test]
    fn hma1d_incomplete_worker_timing_never_hides_retry_evidence() {
        let observer = Observer::new(Arm::Control, 1);
        observer.begin_request(Phase::Warmup, 0);
        let now = Instant::now();
        let mut raw = RawBatch {
            start: Some(now),
            end: Some(now),
            ..Default::default()
        };
        raw.reads[0].attempt_starts = [Some(now), Some(now), None];
        raw.reads[0].transient_events = 2;
        observer.commit(
            &[1],
            Helper::ControlSingleFileExt,
            &raw,
            now,
            now,
            &Err(std::io::Error::other("failed")),
            None,
        );
        let snapshot = observer.snapshot();
        let record = &snapshot.records[0];
        assert!(record.timing_error.is_some());
        assert_eq!(record.diagnostic_retry_count, 1);
        assert_eq!(record.reads[0].transient_events, 2);
        assert_eq!(record.raw_helper_total_ns, Some(0));
    }

    #[test]
    fn hma1d_offline_audit_reconstructs_and_does_not_overwrite_source() {
        let root = std::env::temp_dir().join(format!(
            "hma1d-audit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let (stores, p) = fixture();
        let mut envelope = Envelope::new(p, stores);
        envelope.analysis.disposition = "UNTRUSTED_CACHED_LABEL".into();
        let bytes = serde_json::to_vec(&envelope).unwrap();
        let input = root.join("source.json");
        let log = root.join("complete.log");
        let output = root.join("audited.json");
        std::fs::write(&input, &bytes).unwrap();
        std::fs::write(&log, b"synthetic test complete transcript\n").unwrap();
        audit_command(&input, &log, &output).unwrap();
        let audited: Envelope = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(audited.analysis.disposition, "FD_PROOF_DOMINANT");
        assert_eq!(std::fs::read(&input).unwrap(), bytes);
        assert!(audit_command(&input, &log, &output).is_err());
        std::fs::write(&log, b"transient I/O error; retrying\n").unwrap();
        let failed = root.join("retry.json");
        assert!(audit_command(&input, &log, &failed).is_err());
        let audited: Envelope = serde_json::from_slice(&std::fs::read(&failed).unwrap()).unwrap();
        assert_eq!(audited.analysis.disposition, "NON_AUTHORITATIVE");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
