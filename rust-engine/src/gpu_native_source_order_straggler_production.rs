//! HMA-1E qualification-only observations of the existing production stream.
//! Timestamp capture never owns storage, a GPU, a lock, or a source scheduler.
use crate::gpu_native_physical_install_staging::source_to_upload_production::{
    runtime_contracts_equal_except_context_id, validate_recorded_authority_with_contract_equality,
    ArmContractEquality, MeasuredMechanismGateKey,
};
#[cfg(test)]
use crate::gpu_native_physical_install_staging::source_to_upload_production::validate_recorded_authority_with_measured_gate_key;
use crate::gpu_native_source_upload::Arm;
use serde::Serialize;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-source-order-straggler-production.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-source-order-straggler-production";
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
        if s.context.is_some() || s.snapshot.request_begins.len() >= 4 {
            s.snapshot.context_errors = s.snapshot.context_errors.saturating_add(1);
            return;
        }
        s.context = Some((phase, index));
        s.ordinal = 0;
        s.snapshot.request_begins.push((phase, index));
    }
    pub(crate) fn finish_request(&self, existing_cumulative_ids: Option<String>) {
        let mut s = self.inner.lock();
        if let (Some((phase, index)), Some(hash)) = (s.context.take(), existing_cumulative_ids) {
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

pub(crate) const fn qualification_arms() -> [Arm; 2] {
    [Arm::Treatment, Arm::Control]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
struct EndpointTotals {
    records: u64,
    read_critical_span_ns: u64,
    batch_max_read_wall_ns: u64,
    batch_sum_read_wall_ns: u64,
    critical_minus_max_wrapper_ns: u64,
    overlap_factor: Option<f64>,
}
fn endpoint_total(records: &[&Record]) -> Result<EndpointTotals, &'static str> {
    let critical = checked_sum(records.iter().map(|r| r.timing.read_critical_span_ns))?;
    let max = checked_sum(records.iter().map(|r| r.batch_max_read_wall_ns))?;
    let sum = checked_sum(records.iter().map(|r| r.batch_sum_read_wall_ns))?;
    Ok(EndpointTotals {
        records: records.len() as u64,
        read_critical_span_ns: critical,
        batch_max_read_wall_ns: max,
        batch_sum_read_wall_ns: sum,
        critical_minus_max_wrapper_ns: critical
            .checked_sub(max)
            .ok_or("critical below max wrapper")?,
        overlap_factor: (critical > 0).then(|| sum as f64 / critical as f64),
    })
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Comparison {
    control: EndpointTotals,
    treatment: EndpointTotals,
    dcrit_ns: i128,
    dmax_ns: i128,
    dcrit_percent_of_control: Option<f64>,
    dmax_percent_of_control: Option<f64>,
}
fn compare(c: EndpointTotals, t: EndpointTotals) -> Comparison {
    let dcrit = i128::from(t.read_critical_span_ns) - i128::from(c.read_critical_span_ns);
    let dmax = i128::from(t.batch_max_read_wall_ns) - i128::from(c.batch_max_read_wall_ns);
    Comparison {
        control: c,
        treatment: t,
        dcrit_ns: dcrit,
        dmax_ns: dmax,
        dcrit_percent_of_control: (c.read_critical_span_ns > 0)
            .then(|| 100.0 * dcrit as f64 / c.read_critical_span_ns as f64),
        dmax_percent_of_control: (c.batch_max_read_wall_ns > 0)
            .then(|| 100.0 * dmax as f64 / c.batch_max_read_wall_ns as f64),
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Stratum {
    kind: String,
    key: usize,
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
    errors: Vec<String>,
    external_retry_warning_occurrences: Option<u64>,
    external_existing_breaker_retry_events: Option<u64>,
    primary: Option<Comparison>,
    streams: Vec<Stream>,
    strata: Vec<Stratum>,
    decision: String,
}

fn decision(label: &str) -> &'static str {
    match label {
        "ORDER_ROBUST_STRAGGLER" => "The concurrent treatment straggler survives first-arm temporal position. Only this result permits a later separate kernel/page-pinning/NUMA/fixed-buffer mechanism design.",
        "ORDER_DOMINATED_REVERSAL" => "Reversing full-arm order reverses the material straggler. Do not pursue mapped-destination/kernel optimization from HMA-1D; close the HMA source micro-lane by default.",
        "STRAGGLER_NOT_REPRODUCED" => "The HMA-1D material straggler does not reproduce under reversed order. Close the source micro-lane by default.",
        "DIRECTIONAL_ORDER_ROBUST" => "Direction survives below the material threshold. No production change; close by default unless a later explicit design justifies the expected gain.",
        "AMBIGUOUS_ORDER_INTERACTION" => "Do not infer mapped-memory/kernel causality. No automatic next micro-discriminator.",
        _ => "Authority failed or completed external transcript audit is pending. No performance classification or interpretation is permitted.",
    }
}

fn classify(p: &Comparison, strata: &[Stratum]) -> Result<&'static str, &'static str> {
    // u64 arm totals -> i128 deltas, then checked cross-products. Display f64
    // percentages NEVER control disposition, including rounded threshold ties.
    let dc = p
        .dcrit_ns
        .checked_mul(100)
        .ok_or("critical threshold overflow")?;
    let dm = p.dmax_ns.checked_mul(100).ok_or("max threshold overflow")?;
    let c = i128::from(p.control.read_critical_span_ns);
    let m = i128::from(p.control.batch_max_read_wall_ns);
    if c == 0 || m == 0 {
        return Err("empty primary control denominator");
    }
    let c3 = c.checked_mul(3).ok_or("critical threshold overflow")?;
    let m3 = m.checked_mul(3).ok_or("max threshold overflow")?;
    let positive = |s: &&Stratum| s.comparison.dcrit_ns > 0 && s.comparison.dmax_ns > 0;
    let requests = strata
        .iter()
        .filter(|s| s.kind == "request_index")
        .filter(positive)
        .count();
    let halves = strata
        .iter()
        .filter(|s| s.kind == "ordinal_half")
        .filter(positive)
        .count();
    let widths_positive = strata
        .iter()
        .filter(|s| {
            s.kind == "source_set_width" && (2..=8).contains(&s.key) && s.comparison.dcrit_ns > 0
        })
        .count();
    let widths_negative = strata
        .iter()
        .filter(|s| {
            s.kind == "source_set_width" && (2..=8).contains(&s.key) && s.comparison.dcrit_ns < 0
        })
        .count();
    let consistent = requests >= 2 && halves == 2 && widths_positive >= 5;
    Ok(if dc >= c3 && dm >= m3 && consistent {
        "ORDER_ROBUST_STRAGGLER"
    } else if dc <= -c3 && dm <= -m3 && widths_negative >= 5 {
        "ORDER_DOMINATED_REVERSAL"
    } else if dc > -c && dc < c && dm > -m && dm < m {
        "STRAGGLER_NOT_REPRODUCED"
    } else if dc >= c && dc < c3 && dm >= m && dm < m3 && consistent {
        "DIRECTIONAL_ORDER_ROBUST"
    } else {
        "AMBIGUOUS_ORDER_INTERACTION"
    })
}
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
fn validate_streams(
    stores: &[StoreSnapshot],
    production: &serde_json::Value,
) -> Result<Vec<Stream>, &'static str> {
    if stores.len() != 2 || stores[0].arm != Arm::Treatment || stores[1].arm != Arm::Control {
        return Err("missing/duplicate/reordered arm: required Treatment then Control");
    }
    for key in ["warmup_production", "production"] {
        if production["control"][key].is_null()
            || production["control"][key] != production["treatment"][key]
        {
            return Err("existing source scheduler snapshots differ");
        }
    }
    let mut streams = Vec::new();
    for (s, key) in stores.iter().zip(["treatment", "control"]) {
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
    let c = &stores[1].records;
    let t = &stores[0].records;
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
    // Reconcile both phases separately above and the full arm here.
    for store in stores {
        let all: Vec<_> = store.records.iter().collect();
        if !reconcile_timer(
            total(&all)?.helper_total_ns,
            checked_sum(all.iter().map(|r| r.caller_helper_ns))?,
        ) {
            return Err("complete arm caller timer mismatch");
        }
    }
    Ok(streams)
}
fn endpoints(stores: &[StoreSnapshot]) -> Result<(Comparison, Vec<Stratum>), &'static str> {
    let c: Vec<_> = stores[1]
        .records
        .iter()
        .filter(|r| r.phase == Phase::Measured && r.source_set_width > 1)
        .collect();
    let t: Vec<_> = stores[0]
        .records
        .iter()
        .filter(|r| r.phase == Phase::Measured && r.source_set_width > 1)
        .collect();
    let primary = compare(endpoint_total(&c)?, endpoint_total(&t)?);
    if primary.control.read_critical_span_ns == 0
        || primary.control.batch_max_read_wall_ns == 0
        || primary.treatment.read_critical_span_ns == 0
        || primary.treatment.batch_max_read_wall_ns == 0
    {
        return Err("empty measured K>1 primary");
    }
    let mut strata = Vec::new();
    let mut add =
        |kind: &str, key: usize, a: Vec<&Record>, b: Vec<&Record>| -> Result<(), &'static str> {
            strata.push(Stratum {
                kind: kind.into(),
                key,
                comparison: compare(endpoint_total(&a)?, endpoint_total(&b)?),
            });
            Ok(())
        };
    for i in 0..3 {
        add(
            "request_index",
            i,
            c.iter().copied().filter(|r| r.request_index == i).collect(),
            t.iter().copied().filter(|r| r.request_index == i).collect(),
        )?;
    }
    // Halves split the measured K>1 sequence, floor(N/2), independently by arm.
    // Exact stream equivalence above makes the boundaries identical.
    for i in 0..2 {
        add(
            "ordinal_half",
            i + 1,
            c.iter()
                .enumerate()
                .filter(|(n, _)| bucket(*n, c.len(), 2) == i)
                .map(|(_, r)| *r)
                .collect(),
            t.iter()
                .enumerate()
                .filter(|(n, _)| bucket(*n, t.len(), 2) == i)
                .map(|(_, r)| *r)
                .collect(),
        )?;
    }
    // Keep all seven frozen width strata, including empty ones. Empty widths
    // cannot supply a positive/negative consistency vote. K=1 is descriptive.
    for width in 1..=WIDTH {
        add(
            "source_set_width",
            width,
            stores[1]
                .records
                .iter()
                .filter(|r| r.phase == Phase::Measured && r.source_set_width == width)
                .collect(),
            stores[0]
                .records
                .iter()
                .filter(|r| r.phase == Phase::Measured && r.source_set_width == width)
                .collect(),
        )?;
    }
    Ok((primary, strata))
}

fn non_authoritative() -> Analysis {
    Analysis {
        disposition: "NON_AUTHORITATIVE".into(),
        errors: vec![],
        external_retry_warning_occurrences: None,
        external_existing_breaker_retry_events: None,
        primary: None,
        streams: vec![],
        strata: vec![],
        decision: decision("NON_AUTHORITATIVE").into(),
    }
}
fn reconstruct(stores: &[StoreSnapshot], production: &serde_json::Value) -> Analysis {
    reconstruct_with_measured_gate_key(stores, production, MeasuredMechanismGateKey::Legacy)
}
fn reconstruct_with_measured_gate_key(
    stores: &[StoreSnapshot],
    production: &serde_json::Value,
    measured_gate_key: MeasuredMechanismGateKey,
) -> Analysis {
    reconstruct_with_contract_equality(
        stores,
        production,
        measured_gate_key,
        ArmContractEquality::Exact,
    )
}
fn reconstruct_with_contract_equality(
    stores: &[StoreSnapshot],
    production: &serde_json::Value,
    measured_gate_key: MeasuredMechanismGateKey,
    contract_equality: ArmContractEquality,
) -> Analysis {
    let mut result = non_authoritative();
    if let Err(e) = validate_recorded_authority_with_contract_equality(
        production,
        measured_gate_key,
        contract_equality,
    ) {
        result.errors.push(e.to_string());
    }
    match validate_streams(stores, production) {
        Ok(streams) => {
            result.streams = streams;
            match endpoints(stores) {
                Ok((primary, strata)) => {
                    result.primary = Some(primary);
                    result.strata = strata;
                }
                Err(e) => result.errors.push(e.into()),
            }
        }
        Err(e) => result.errors.push(e.into()),
    }
    result
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Envelope {
    schema: String,
    mode: String,
    execution_order: [Arm; 2],
    primary_definition: String,
    ordinal_half_definition: String,
    production_v2: serde_json::Value,
    observations: Vec<StoreSnapshot>,
    analysis: Analysis,
}
const PRIMARY: &str = "Measured source_set_width > 1 only; Dcrit and Dmax are Treatment minus Control; each percentage uses its corresponding Control K>1 total. K=1 is diagnostic only.";
const HALVES: &str = "Measured K>1 source-call sequence independently per arm, first half [0,floor(N/2)), second half [floor(N/2),N); warmup and K=1 excluded.";
const BEGIN: &str = "HMA1E_QUALIFIER_BEGIN order=treatment,control";
const END: &str = "HMA1E_QUALIFIER_COMPLETE raw_report_sha256=";
impl Envelope {
    pub(crate) fn new(production_v2: serde_json::Value, observations: Vec<StoreSnapshot>) -> Self {
        let mut analysis = reconstruct(&observations, &production_v2);
        analysis
            .errors
            .push("completed external transcript audit pending".into());
        Self {
            schema: SCHEMA.into(),
            mode: MODE.into(),
            execution_order: qualification_arms(),
            primary_definition: PRIMARY.into(),
            ordinal_half_definition: HALVES.into(),
            production_v2,
            observations,
            analysis,
        }
    }
    pub(crate) fn emit(&self, output: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        eprintln!("{END}{:x}", Sha256::digest(&bytes));
        Ok(())
    }
}
pub(crate) fn begin_transcript() {
    eprintln!("{BEGIN}");
}

pub(crate) fn retry_warning_count(bytes: &[u8]) -> u64 {
    count(bytes, b"transient I/O error; retrying")
}
fn count(bytes: &[u8], pattern: &[u8]) -> u64 {
    bytes
        .windows(pattern.len())
        .filter(|w| *w == pattern)
        .count() as u64
}
fn existing_breaker_retry_event_count(bytes: &[u8]) -> u64 {
    [
        b"expert fetch recovered after retry".as_slice(),
        b"expert fetch failed; will retry".as_slice(),
        b"circuit breaker".as_slice(),
    ]
    .into_iter()
    .map(|p| count(bytes, p))
    .sum()
}
#[derive(Serialize)]
struct AuditedReport {
    schema: &'static str,
    mode: &'static str,
    source_report_sha256: String,
    completed_transcript_sha256: String,
    source_report_bytes: usize,
    completed_transcript_bytes: usize,
    analysis: Analysis,
}
fn audit_bytes(bytes: &[u8], transcript: &[u8]) -> AuditedReport {
    audit_bytes_with_measured_gate_key(bytes, transcript, MeasuredMechanismGateKey::Legacy)
}
fn audit_bytes_with_measured_gate_key(
    bytes: &[u8],
    transcript: &[u8],
    measured_gate_key: MeasuredMechanismGateKey,
) -> AuditedReport {
    audit_bytes_with_contract_equality(
        bytes,
        transcript,
        measured_gate_key,
        ArmContractEquality::Exact,
    )
}
fn audit_bytes_with_contract_equality(
    bytes: &[u8],
    transcript: &[u8],
    measured_gate_key: MeasuredMechanismGateKey,
    contract_equality: ArmContractEquality,
) -> AuditedReport {
    let source_report_sha256 = format!("{:x}", Sha256::digest(bytes));
    let mut a = non_authoritative();
    match serde_json::from_slice::<Envelope>(bytes) {
        Err(e) => a.errors.push(format!("raw report structure: {e}")),
        Ok(raw) => {
            a = reconstruct_with_contract_equality(
                &raw.observations,
                &raw.production_v2,
                measured_gate_key,
                contract_equality,
            );
            if raw.schema != SCHEMA
                || raw.mode != MODE
                || raw.execution_order != qualification_arms()
                || raw.primary_definition != PRIMARY
                || raw.ordinal_half_definition != HALVES
            {
                a.errors
                    .push("raw schema/mode/order/endpoint definition mismatch".into());
            }
            // Validate ALL retained cached reconstruction, including strata,
            // raw NON_AUTHORITATIVE disposition, decision and pending status.
            // Cached raw analysis ALWAYS belongs to the legacy reconstruction,
            // including when this shared auditor reconstructs corrected authority.
            let expected = Envelope::new(raw.production_v2.clone(), raw.observations.clone());
            let expected_json: serde_json::Value = serde_json::from_slice(
                &serde_json::to_vec(&expected.analysis).expect("analysis serialization"),
            )
            .expect("analysis JSON");
            if serde_json::to_value(&raw.analysis).ok() != Some(expected_json) {
                a.errors
                    .push("raw analysis/primary/strata reconstruction mismatch".into());
            }
        }
    }
    a.external_retry_warning_occurrences = Some(retry_warning_count(transcript));
    a.external_existing_breaker_retry_events = Some(existing_breaker_retry_event_count(transcript));
    if a.external_retry_warning_occurrences != Some(0) {
        a.errors
            .push("external transient I/O error; retrying occurrence".into());
    }
    if a.external_existing_breaker_retry_events != Some(0) {
        a.errors
            .push("external existing breaker/fetch-retry event".into());
    }
    let end = format!("{END}{source_report_sha256}");
    let text = String::from_utf8_lossy(transcript);
    let lines: Vec<_> = text.lines().collect();
    let begin_at = lines.iter().position(|l| *l == BEGIN);
    let end_at = lines.iter().position(|l| *l == end);
    if count(transcript, BEGIN.as_bytes()) != 1
        || count(transcript, END.as_bytes()) != 1
        || begin_at.zip(end_at).is_none_or(|(b, e)| b >= e)
    {
        a.errors.push("missing/duplicated/out-of-order completed transcript evidence or raw report hash mismatch; complete external stdout+stderr required".into());
    }
    // Only AFTER every authority gate, report-integrity check and transcript
    // check has passed can the primary endpoint select a performance label.
    if a.errors.is_empty() {
        match a
            .primary
            .as_ref()
            .ok_or("missing primary")
            .and_then(|p| classify(p, &a.strata))
        {
            Ok(label) => {
                a.disposition = label.into();
                a.decision = decision(label).into();
            }
            Err(e) => a.errors.push(e.into()),
        }
    }
    AuditedReport {
        schema: SCHEMA,
        mode: "audit-gpu-native-source-order-straggler-production",
        source_report_sha256,
        completed_transcript_sha256: format!("{:x}", Sha256::digest(transcript)),
        source_report_bytes: bytes.len(),
        completed_transcript_bytes: transcript.len(),
        analysis: a,
    }
}
/// Offline only: read once into bytes, hash and deserialize that exact report
/// snapshot; scan the complete external transcript, including any trailing
/// shutdown/errors. No runtime, model loading, config reads or GPU construction.
pub(crate) fn audit_command(
    input: &std::path::Path,
    log: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if input == output || log == output || output.exists() {
        return Err("audit output must be a distinct new artifact path".into());
    }
    let bytes = std::fs::read(input)?;
    let transcript = std::fs::read(log)?;
    let report = audit_bytes(&bytes, &transcript);
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    if report.analysis.disposition == "NON_AUTHORITATIVE" {
        return Err("HMA-1E remains NON_AUTHORITATIVE; see audited report".into());
    }
    Ok(())
}
const REPAIR2_SCHEMA: &str =
    "mer.gpu-native-source-order-straggler-production.audit-repair-context-id.v1";
const REPAIR2_MODE: &str = "repair2-audit-gpu-native-source-order-straggler-production";
const REPAIR1_ERROR: &str = "arm runtime/config/model contract mismatch";
const CONSUMED_REPAIR1_SHA256: &str =
    "cd8bd4af368aa8b624077172e315c51f5a9a660faf59bb6d5e5f9371c9143742";
const CONTEXT_ID_PATH: &str = "runtime_contract.legacy_execution_plan.context_id";
const CONTRACT_FIELDS: [&str; 6] = [
    "request",
    "production_configuration",
    "production_semantics",
    "model_identity",
    "hardware",
    "runtime_contract",
];

#[derive(Serialize)]
struct ContractDifference {
    path: String,
    // Retain unambiguous components even if an unexpected JSON key contains dots.
    path_components: Vec<String>,
    control: Option<serde_json::Value>,
    treatment: Option<serde_json::Value>,
}

/// Complete structural diff: objects use the union of keys, arrays include every
/// index, and missing is distinct from null. Nothing is excluded here.
fn contract_differences(
    control: Option<&serde_json::Value>,
    treatment: Option<&serde_json::Value>,
    path: &mut Vec<String>,
    differences: &mut Vec<ContractDifference>,
) {
    use serde_json::Value;
    if control == treatment {
        return;
    }
    match (control, treatment) {
        (Some(Value::Object(c)), Some(Value::Object(t))) => {
            let keys: std::collections::BTreeSet<_> = c.keys().chain(t.keys()).collect();
            for key in keys {
                path.push(key.clone());
                contract_differences(c.get(key), t.get(key), path, differences);
                path.pop();
            }
        }
        (Some(Value::Array(c)), Some(Value::Array(t))) => {
            for i in 0..c.len().max(t.len()) {
                path.push(i.to_string());
                contract_differences(c.get(i), t.get(i), path, differences);
                path.pop();
            }
        }
        _ => differences.push(ContractDifference {
            path: path.join("."),
            path_components: path.clone(),
            control: control.cloned(),
            treatment: treatment.cloned(),
        }),
    }
}

#[derive(Default, Serialize)]
struct ContextIdProof {
    control_context_id: Option<String>,
    treatment_context_id: Option<String>,
    context_ids_parseable: bool,
    context_ids_nonzero: bool,
    context_ids_distinct: bool,
    cross_arm_difference_count_before_exclusion: Option<usize>,
    cross_arm_differences_before_exclusion: Vec<ContractDifference>,
    all_other_runtime_contract_fields_exact: bool,
    other_five_contract_objects_exact: bool,
}
impl ContextIdProof {
    fn of(production: &serde_json::Value) -> Self {
        let c = &production["control"]["benchmark"];
        let t = &production["treatment"]["benchmark"];
        let mut proof = Self::default();
        // Independently enumerate the COMPLETE six-object diff before removing
        // any leaf or using the repair2 equality helper.
        for field in CONTRACT_FIELDS {
            contract_differences(
                c.get(field),
                t.get(field),
                &mut vec![field.into()],
                &mut proof.cross_arm_differences_before_exclusion,
            );
        }
        proof.cross_arm_difference_count_before_exclusion =
            Some(proof.cross_arm_differences_before_exclusion.len());
        proof.control_context_id = c["runtime_contract"]["legacy_execution_plan"]["context_id"]
            .as_str()
            .map(str::to_owned);
        proof.treatment_context_id = t["runtime_contract"]["legacy_execution_plan"]["context_id"]
            .as_str()
            .map(str::to_owned);
        let ids = proof
            .control_context_id
            .as_deref()
            .and_then(|v| v.parse::<u64>().ok())
            .zip(
                proof
                    .treatment_context_id
                    .as_deref()
                    .and_then(|v| v.parse::<u64>().ok()),
            );
        proof.context_ids_parseable = ids.is_some();
        proof.context_ids_nonzero = ids.is_some_and(|(c, t)| c > 0 && t > 0);
        proof.context_ids_distinct = ids.is_some_and(|(c, t)| c != t);
        proof.other_five_contract_objects_exact = CONTRACT_FIELDS[..5]
            .iter()
            .all(|field| c[*field].is_object() && c[*field] == t[*field]);
        proof.all_other_runtime_contract_fields_exact = runtime_contracts_equal_except_context_id(
            &c["runtime_contract"],
            &t["runtime_contract"],
        );
        proof
    }
    fn passed(&self) -> bool {
        self.cross_arm_difference_count_before_exclusion == Some(1)
            && self.cross_arm_differences_before_exclusion[0].path_components
                == ["runtime_contract", "legacy_execution_plan", "context_id"]
            && self.control_context_id.as_deref() == Some("3")
            && self.treatment_context_id.as_deref() == Some("2")
            && self.context_ids_parseable
            && self.context_ids_nonzero
            && self.context_ids_distinct
            && self.all_other_runtime_contract_fields_exact
            && self.other_five_contract_objects_exact
    }
}

#[derive(Serialize)]
struct Repair2Report {
    schema: &'static str,
    mode: &'static str,
    raw_report: ArtifactIdentity,
    completed_transcript: ArtifactIdentity,
    original_audit: ArtifactIdentity,
    repair1_audit: ArtifactIdentity,
    legacy_reproduction_pass: bool,
    repair1_reproduction_pass: bool,
    defect: &'static str,
    excluded_path: &'static str,
    #[serde(flatten)]
    context_id_proof: ContextIdProof,
    // None means an earlier proof stage refused final reconstruction.
    final_analysis: Option<Analysis>,
    final_disposition: String,
    final_errors: Vec<String>,
    #[serde(flatten)]
    result: Analysis,
    hardware_rerun_performed: bool,
    runtime_constructed: bool,
    model_loaded: bool,
    gpu_constructed: bool,
}
impl Repair2Report {
    fn finish(mut self) -> Self {
        self.final_disposition = self.result.disposition.clone();
        self.final_errors = self.result.errors.clone();
        self
    }
}

/// Production caller supplies ONLY frozen constants. Private bindings permit
/// deterministic synthetic tests; no CLI/env/config replacement exists.
fn repair2_bytes(
    raw: &[u8],
    transcript: &[u8],
    original: &[u8],
    repair1: &[u8],
    expected: &EvidenceHashes<'_>,
    expected_repair1: &str,
) -> Repair2Report {
    let mut report = Repair2Report {
        schema: REPAIR2_SCHEMA,
        mode: REPAIR2_MODE,
        raw_report: ArtifactIdentity::of(raw),
        completed_transcript: ArtifactIdentity::of(transcript),
        original_audit: ArtifactIdentity::of(original),
        repair1_audit: ArtifactIdentity::of(repair1),
        legacy_reproduction_pass: false,
        repair1_reproduction_pass: false,
        defect: "ISOLATED_EXECUTION_CONTEXT_ID_EQUALITY",
        excluded_path: CONTEXT_ID_PATH,
        context_id_proof: ContextIdProof::default(),
        final_analysis: None,
        final_disposition: "NON_AUTHORITATIVE".into(),
        final_errors: vec![],
        result: non_authoritative(),
        hardware_rerun_performed: false,
        runtime_constructed: false,
        model_loaded: false,
        gpu_constructed: false,
    };
    // Stage 0: all four exact byte identities before ANY parsing/reconstruction.
    for (actual, expected, label) in [
        (&report.raw_report.sha256, expected.raw, "raw report"),
        (
            &report.completed_transcript.sha256,
            expected.transcript,
            "transcript",
        ),
        (
            &report.original_audit.sha256,
            expected.original_audit,
            "original audit",
        ),
        (
            &report.repair1_audit.sha256,
            expected_repair1,
            "repair-v1 audit",
        ),
    ] {
        if actual != expected {
            report
                .result
                .errors
                .push(format!("consumed FIRST {label} SHA-256 mismatch"));
        }
    }
    if !report.result.errors.is_empty() {
        return report.finish();
    }

    // Stages 1 and 2: invoke repair-v1 itself UNCHANGED. It reproduces the entire
    // historical audit before its unchanged measured-mechanism reconstruction.
    let reproduced = repair_bytes(raw, transcript, original, expected);
    report.legacy_reproduction_pass = reproduced.legacy_reproduction_pass;
    report.result = reproduced.result.clone();
    if !report.legacy_reproduction_pass {
        return report.finish();
    }
    let supplied: serde_json::Value = match serde_json::from_slice(repair1) {
        Ok(value) => value,
        Err(e) => {
            report.result = non_authoritative();
            report.result.external_retry_warning_occurrences =
                reproduced.result.external_retry_warning_occurrences;
            report.result.external_existing_breaker_retry_events =
                reproduced.result.external_existing_breaker_retry_events;
            report
                .result
                .errors
                .push(format!("repair-v1 audit structure: {e}"));
            return report.finish();
        }
    };
    // Canonical semantic comparison uses the same serialized-then-parsed form
    // as repair-v1. No field filtering, tolerance, rounding or normalization.
    let reproduced_json = serde_json::to_vec(&reproduced)
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes));
    let known_failure = reproduced.corrected_disposition.as_deref() == Some("NON_AUTHORITATIVE")
        && reproduced.corrected_errors.as_deref() == Some(&[REPAIR1_ERROR.to_string()][..])
        && reproduced.result.disposition == "NON_AUTHORITATIVE"
        && reproduced.result.errors == [REPAIR1_ERROR]
        && reproduced.result.external_retry_warning_occurrences == Some(0)
        && reproduced.result.external_existing_breaker_retry_events == Some(0);
    if !known_failure || reproduced_json.ok().as_ref() != Some(&supplied) {
        report.result = non_authoritative();
        report.result.external_retry_warning_occurrences =
            reproduced.result.external_retry_warning_occurrences;
        report.result.external_existing_breaker_retry_events =
            reproduced.result.external_existing_breaker_retry_events;
        report.result.errors.push("repair-v1 must exactly reproduce the sole arm contract error and zero retry/breaker events".into());
        return report.finish();
    }
    report.repair1_reproduction_pass = true;

    // Stage 3: this SAME snapshot has passed the historical envelope audit.
    let envelope: Envelope = match serde_json::from_slice(raw) {
        Ok(value) => value,
        Err(e) => {
            report
                .result
                .errors
                .push(format!("raw report structure: {e}"));
            return report.finish();
        }
    };
    report.context_id_proof = ContextIdProof::of(&envelope.production_v2);
    if !report.context_id_proof.passed() {
        report.result.errors.push("expected exactly the isolated context_id difference: control=3 treatment=2; all other contract fields exact".into());
        return report.finish();
    }
    // Stage 4: all shared checks and per-arm runtime validators are retained.
    // Only cross-arm runtime equivalence excludes the single proven leaf.
    let final_audit = audit_bytes_with_contract_equality(
        raw,
        transcript,
        MeasuredMechanismGateKey::Repaired,
        ArmContractEquality::IsolatedContextId,
    );
    report.final_analysis = Some(final_audit.analysis.clone());
    report.result = final_audit.analysis;
    report.finish()
}

/// Offline/pre-startup: four read-once byte snapshots, frozen bindings, new output.
pub(crate) fn repair2_audit_command(
    input: &std::path::Path,
    log: &std::path::Path,
    original_audit: &std::path::Path,
    repair1_audit: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if [input, log, original_audit, repair1_audit].contains(&output) {
        return Err("repair2 output must be distinct from all inputs".into());
    }
    match std::fs::symlink_metadata(output) {
        Ok(_) => return Err("repair2 output must be a new artifact path".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let raw = std::fs::read(input)?;
    let transcript = std::fs::read(log)?;
    let original = std::fs::read(original_audit)?;
    let repair1 = std::fs::read(repair1_audit)?;
    let report = repair2_bytes(
        &raw,
        &transcript,
        &original,
        &repair1,
        &CONSUMED_FIRST,
        CONSUMED_REPAIR1_SHA256,
    );
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    if report.result.disposition == "NON_AUTHORITATIVE" {
        return Err("HMA-1E repair2 remains NON_AUTHORITATIVE; see new repaired audit".into());
    }
    Ok(())
}

const REPAIR_SCHEMA: &str = "mer.gpu-native-source-order-straggler-production.audit-repair.v1";
const REPAIR_MODE: &str = "repair-audit-gpu-native-source-order-straggler-production";
const HISTORICAL_ERROR: &str = "production mechanism reconstruction failed";
const EVIDENCE_CHILD: &str = "66e73d921c6eb2bbe1e7ba7eebde743553078a67";
const EVIDENCE_PARENT: &str = "cbc9f4b7e5fd34dbed6315206c0d578e848c96ee";
const EVIDENCE_TREE: &str = "9eb50b1bacedb40b8777fff43791c3001830d9d9";

struct EvidenceHashes<'a> {
    raw: &'a str,
    transcript: &'a str,
    original_audit: &'a str,
}
const CONSUMED_FIRST: EvidenceHashes<'static> = EvidenceHashes {
    raw: "0737f4c8258614df7e2a13fa40c5cce47dbaa7af2ab84558e0a2db9fd5894cfd",
    transcript: "d09f527da2cc299aca5097b940811298d6269f2d19b090e0dacca6e7d49c9db0",
    original_audit: "6c942ab83a4e907c2be1a3726a0c4807d7a7ec8b9b83635fd80e8f06d5e80ebf",
};

#[derive(Serialize)]
struct ArtifactIdentity {
    sha256: String,
    bytes: usize,
}
impl ArtifactIdentity {
    fn of(bytes: &[u8]) -> Self {
        Self {
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len(),
        }
    }
}

#[derive(Serialize)]
struct RepairReport {
    schema: &'static str,
    mode: &'static str,
    evidence_child_sha: &'static str,
    evidence_parent_sha: &'static str,
    evidence_tree_sha: &'static str,
    raw_report: ArtifactIdentity,
    completed_transcript: ArtifactIdentity,
    original_audit: ArtifactIdentity,
    legacy_reproduction_pass: bool,
    defect: &'static str,
    legacy_lookup: &'static str,
    corrected_lookup: &'static str,
    original_disposition: Option<String>,
    original_errors: Option<Vec<String>>,
    corrected_analysis_origin: &'static str,
    // None means Stage 1 refused the evidence; correction never ran.
    corrected_analysis: Option<Analysis>,
    corrected_disposition: Option<String>,
    corrected_errors: Option<Vec<String>>,
    #[serde(flatten)]
    result: Analysis,
    hardware_rerun_performed: bool,
    runtime_constructed: bool,
    model_loaded: bool,
    gpu_constructed: bool,
}

/// Only repair_audit_command supplies production bindings, always CONSUMED_FIRST.
/// The private parameter lets unit tests exercise both stages on synthetic bytes
/// without accepting replacement evidence through a CLI, environment or config.
fn repair_bytes(
    raw: &[u8],
    transcript: &[u8],
    original_audit: &[u8],
    expected: &EvidenceHashes<'_>,
) -> RepairReport {
    let mut report = RepairReport {
        schema: REPAIR_SCHEMA,
        mode: REPAIR_MODE,
        evidence_child_sha: EVIDENCE_CHILD,
        evidence_parent_sha: EVIDENCE_PARENT,
        evidence_tree_sha: EVIDENCE_TREE,
        raw_report: ArtifactIdentity::of(raw),
        completed_transcript: ArtifactIdentity::of(transcript),
        original_audit: ArtifactIdentity::of(original_audit),
        legacy_reproduction_pass: false,
        defect: "MEASURED_MECHANISM_GATE_KEY",
        legacy_lookup: "mechanism",
        corrected_lookup: "measured_mechanism",
        original_disposition: None,
        original_errors: None,
        corrected_analysis_origin: "Separate offline repair reconstruction over the immutable consumed FIRST; cached raw analysis is validated only against legacy reconstruction.",
        corrected_analysis: None,
        corrected_disposition: None,
        corrected_errors: None,
        result: non_authoritative(),
        hardware_rerun_performed: false,
        runtime_constructed: false,
        model_loaded: false,
        gpu_constructed: false,
    };
    // Bind every exact byte snapshot BEFORE deserialization or reconstruction.
    for (actual, expected, label) in [
        (&report.raw_report.sha256, expected.raw, "raw report"),
        (
            &report.completed_transcript.sha256,
            expected.transcript,
            "transcript",
        ),
        (
            &report.original_audit.sha256,
            expected.original_audit,
            "original audit",
        ),
    ] {
        if actual != expected {
            report
                .result
                .errors
                .push(format!("consumed FIRST {label} SHA-256 mismatch"));
        }
    }
    if !report.result.errors.is_empty() {
        return report;
    }
    let original: serde_json::Value = match serde_json::from_slice(original_audit) {
        Ok(value) => value,
        Err(e) => {
            report
                .result
                .errors
                .push(format!("original audit structure: {e}"));
            return report;
        }
    };
    report.original_disposition = original["analysis"]["disposition"]
        .as_str()
        .map(str::to_owned);
    report.original_errors = serde_json::from_value(original["analysis"]["errors"].clone()).ok();

    // Stage 1 invokes the historical auditor itself. Its checks include exact
    // cached legacy analysis, envelope definitions, all streams and transcript
    // marker/order/hash/retry binding. Compare the ENTIRE canonical audit value,
    // including schema, mode, byte counts and every authority-bearing field.
    let legacy = audit_bytes(raw, transcript);
    report.result = legacy.analysis.clone();
    let known_failure = report.original_disposition.as_deref() == Some("NON_AUTHORITATIVE")
        && report.original_errors.as_deref() == Some(&[HISTORICAL_ERROR.to_string()][..])
        && legacy.analysis.disposition == "NON_AUTHORITATIVE"
        && legacy.analysis.errors == [HISTORICAL_ERROR]
        && legacy.analysis.external_retry_warning_occurrences == Some(0)
        && legacy.analysis.external_existing_breaker_retry_events == Some(0);
    if !known_failure {
        report.result.errors.push("legacy audit must contain only the historical measured mechanism error and zero retry/breaker events".into());
    }
    // Compare the historical SERIALIZED result, parsed identically on both
    // sides (as cached raw analysis already does). Direct to_value retains
    // in-memory f64 representations that can differ after JSON parsing.
    // This is exact semantic equality, with no numeric tolerance or rounding.
    let legacy_json = serde_json::to_vec(&legacy)
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes));
    if legacy_json.ok().as_ref() != Some(&original) {
        report
            .result
            .errors
            .push("legacy audited result does not exactly reproduce original audit".into());
    }
    if !known_failure || report.result.errors != [HISTORICAL_ERROR] {
        report.result.disposition = "NON_AUTHORITATIVE".into();
        report.result.decision = decision("NON_AUTHORITATIVE").into();
        return report;
    }
    report.legacy_reproduction_pass = true;

    // Stage 2 uses the SAME byte snapshots and shared checks/classifier. The
    // sole semantic correction is gates.measured_mechanism. Cached analysis
    // remains checked against LEGACY, never against this corrected result.
    let corrected =
        audit_bytes_with_measured_gate_key(raw, transcript, MeasuredMechanismGateKey::Repaired);
    report.corrected_disposition = Some(corrected.analysis.disposition.clone());
    report.corrected_errors = Some(corrected.analysis.errors.clone());
    report.corrected_analysis = Some(corrected.analysis.clone());
    report.result = corrected.analysis;
    report
}

/// Read each evidence artifact once, then hash/parse/reconstruct those bytes.
/// No production runner, storage object, runtime, config, model or GPU exists.
pub(crate) fn repair_audit_command(
    input: &std::path::Path,
    log: &std::path::Path,
    original_audit: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Any existing entry is forbidden, including hard links, symlinks (also
    // dangling ones), directories and alternate spellings of an input path.
    if [input, log, original_audit].contains(&output) {
        return Err("repair output must be distinct from all inputs".into());
    }
    match std::fs::symlink_metadata(output) {
        Ok(_) => return Err("repair output must be a new artifact path".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let raw = std::fs::read(input)?;
    let transcript = std::fs::read(log)?;
    let original = std::fs::read(original_audit)?;
    let report = repair_bytes(&raw, &transcript, &original, &CONSUMED_FIRST);
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    if report.result.disposition == "NON_AUTHORITATIVE" {
        return Err("HMA-1E repair remains NON_AUTHORITATIVE; see new repaired audit".into());
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
                for (ordinal, width) in [1, 2, 3, 4, 5, 6, 7, 8].into_iter().enumerate() {
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
        stores.reverse();
        (stores, crate::gpu_native_physical_install_staging::source_to_upload_production::hma1e_test_production(p))
    }
    fn complete_log(bytes: &[u8]) -> Vec<u8> {
        format!(
            "{BEGIN}\nsynthetic portable fixture only\n{END}{:x}\n",
            Sha256::digest(bytes)
        )
        .into_bytes()
    }
    fn audited(s: Vec<StoreSnapshot>, p: serde_json::Value) -> AuditedReport {
        let raw = serde_json::to_vec(&Envelope::new(p, s)).unwrap();
        audit_bytes(&raw, &complete_log(&raw))
    }
    fn point(dc: i128, dm: i128) -> Comparison {
        compare(
            EndpointTotals {
                read_critical_span_ns: 100_000_000,
                batch_max_read_wall_ns: 100_000_000,
                ..Default::default()
            },
            EndpointTotals {
                read_critical_span_ns: (100_000_000 + dc) as u64,
                batch_max_read_wall_ns: (100_000_000 + dm) as u64,
                ..Default::default()
            },
        )
    }
    fn consistent(p: &Comparison) -> Vec<Stratum> {
        [
            ("request_index", 3),
            ("ordinal_half", 2),
            ("source_set_width", 7),
        ]
        .into_iter()
        .flat_map(|(kind, n)| {
            (0..n).map(move |i| Stratum {
                kind: kind.into(),
                key: if kind == "source_set_width" { i + 2 } else { i },
                comparison: p.clone(),
            })
        })
        .collect()
    }
    #[test]
    fn hma1e_frozen_order_and_existing_production_order() {
        assert_eq!(qualification_arms(), [Arm::Treatment, Arm::Control]);
        assert_eq!(crate::gpu_native_physical_install_staging::source_to_upload_production::qualification_arms(),(Arm::Control,Arm::Treatment));
        let source = include_str!("gpu_native_source_to_upload_copy_elision_production.rs");
        let loop_body = source
            .split("for arm in execution_order {")
            .nth(1)
            .unwrap()
            .split("let c = report.control")
            .next()
            .unwrap();
        assert!(loop_body.contains("if arm == control_arm"));
    }
    #[test]
    fn hma1e_exact_plus_three_boundary() {
        for d in [2_999_999, 3_000_000, 3_000_001] {
            let p = point(d, d);
            assert_eq!(
                classify(&p, &consistent(&p)).unwrap(),
                if d < 3_000_000 {
                    "DIRECTIONAL_ORDER_ROBUST"
                } else {
                    "ORDER_ROBUST_STRAGGLER"
                }
            );
        }
    }
    #[test]
    fn hma1e_exact_minus_three_boundary() {
        for d in [-3_000_001, -3_000_000, -2_999_999] {
            let p = point(d, d);
            assert_eq!(
                classify(&p, &consistent(&p)).unwrap(),
                if d <= -3_000_000 {
                    "ORDER_DOMINATED_REVERSAL"
                } else {
                    "AMBIGUOUS_ORDER_INTERACTION"
                }
            );
        }
    }
    #[test]
    fn hma1e_strict_one_percent_and_directional() {
        for (d, label) in [
            (-1_000_001, "AMBIGUOUS_ORDER_INTERACTION"),
            (-1_000_000, "AMBIGUOUS_ORDER_INTERACTION"),
            (-999_999, "STRAGGLER_NOT_REPRODUCED"),
            (0, "STRAGGLER_NOT_REPRODUCED"),
            (999_999, "STRAGGLER_NOT_REPRODUCED"),
            (1_000_000, "DIRECTIONAL_ORDER_ROBUST"),
            (1_000_001, "DIRECTIONAL_ORDER_ROBUST"),
        ] {
            let p = point(d, d);
            assert_eq!(classify(&p, &consistent(&p)).unwrap(), label, "{d}");
        }
    }
    #[test]
    fn hma1e_critical_max_disagreement_is_ambiguous() {
        for (dc, dm) in [
            (3_000_000, -3_000_000),
            (3_000_000, 2_000_000),
            (2_000_000, 3_000_000),
            (1_000_000, 999_999),
        ] {
            let p = point(dc, dm);
            assert_eq!(
                classify(&p, &consistent(&p)).unwrap(),
                "AMBIGUOUS_ORDER_INTERACTION"
            );
        }
    }
    #[test]
    fn hma1e_request_both_endpoints_same_two_requests() {
        for d in [2_000_000, 3_000_000] {
            let p = point(d, d);
            let mut s = consistent(&p);
            s[0].comparison.dcrit_ns = 0;
            assert_ne!(classify(&p, &s).unwrap(), "AMBIGUOUS_ORDER_INTERACTION");
            s[1].comparison.dmax_ns = 0;
            assert_eq!(classify(&p, &s).unwrap(), "AMBIGUOUS_ORDER_INTERACTION");
        }
    }
    #[test]
    fn hma1e_both_ordinal_halves_required() {
        for d in [2_000_000, 3_000_000] {
            let p = point(d, d);
            let mut s = consistent(&p);
            s[3].comparison.dmax_ns = 0;
            assert_eq!(classify(&p, &s).unwrap(), "AMBIGUOUS_ORDER_INTERACTION");
        }
    }
    #[test]
    fn hma1e_five_of_seven_width_votes() {
        for d in [2_000_000, 3_000_000, -3_000_000] {
            let p = point(d, d);
            let mut s = consistent(&p);
            for i in 5..7 {
                s[i].comparison.dcrit_ns = 0;
            }
            assert_ne!(classify(&p, &s).unwrap(), "AMBIGUOUS_ORDER_INTERACTION");
            s[7].comparison.dcrit_ns = 0;
            assert_eq!(classify(&p, &s).unwrap(), "AMBIGUOUS_ORDER_INTERACTION");
        }
    }
    #[test]
    fn hma1e_k1_excluded_k2_through_k8_and_t_minus_c() {
        let (mut s, p) = fixture();
        let a = audited(s.clone(), p);
        assert!(a.analysis.errors.is_empty(), "{:?}", a.analysis.errors);
        let primary = a.analysis.primary.unwrap();
        assert_eq!(primary.control.records, 21);
        assert_eq!(primary.control.read_critical_span_ns, 8_400_000);
        assert_eq!(primary.treatment.read_critical_span_ns, 8_610_000);
        assert_eq!(primary.dcrit_ns, 210_000);
        assert_eq!(primary.dmax_ns, 210_000);
        assert_eq!(primary.control.batch_max_read_wall_ns, 8_391_600);
        for r in &mut s[0].records {
            if r.source_set_width == 1 || r.phase == Phase::Warmup {
                r.timing.read_critical_span_ns = 1_000_000_000_000;
                r.batch_max_read_wall_ns = 1_000_000_000_000;
            }
        }
        assert_eq!(endpoints(&s).unwrap().0, primary);
        for stratum in a
            .analysis
            .strata
            .iter()
            .filter(|s| s.kind == "source_set_width")
        {
            assert_eq!(stratum.comparison.control.records, 3);
        }
    }
    #[test]
    fn hma1e_request_halves_and_partition_reconcile() {
        let (s, p) = fixture();
        let a = audited(s, p);
        assert!(a.analysis.errors.is_empty(), "{:?}", a.analysis.errors);
        for kind in ["request_index", "ordinal_half"] {
            assert_eq!(
                a.analysis
                    .strata
                    .iter()
                    .filter(|s| s.kind == kind)
                    .map(|s| s.comparison.control.read_critical_span_ns)
                    .sum::<u64>(),
                8_400_000
            );
        }
        let halves: Vec<_> = a
            .analysis
            .strata
            .iter()
            .filter(|s| s.kind == "ordinal_half")
            .map(|s| s.comparison.control.records)
            .collect();
        assert_eq!(halves, [10, 11]);
    }
    #[test]
    fn hma1e_authority_corruptions_fail_closed() {
        type Mutation = fn(&mut Vec<StoreSnapshot>, &mut serde_json::Value);
        let mutations: &[Mutation] = &[
            |s, _| {
                s[0].records.remove(3);
            },
            |s, _| {
                let copy = s[0].records[3].clone();
                s[0].records.insert(3, copy);
            },
            |s, _| s.reverse(),
            |s, _| s[0].overflow = 1,
            |s, _| s[0].capacity = 0,
            |s, _| s[0].context_errors = 1,
            |s, _| s[0].records[3].ordered_expert_ids[0] += 1,
            |s, _| s[0].records[0].ordered_expert_ids[7] = 1,
            |s, _| s[0].records[3].source_set_width = 9,
            |s, _| s[0].records[3].returned_bytes = Some(1),
            |s, _| s[0].records[3].timing.helper_total_ns += 1,
            |s, _| s[0].records[3].timing.read_critical_span_ns += 1,
            |s, _| s[0].records[3].scheduler_start_ns += 1,
            |s, _| s[0].records[3].reads[0].wrapper_wall_ns += 1,
            |s, _| s[0].records[3].batch_sum_read_wall_ns += 1,
            |s, _| s[0].records[3].batch_max_read_wall_ns += 1,
            |s, _| s[0].records[3].global_source_call_ordinal += 1,
            |s, _| s[0].records[3].source_call_ordinal_within_request += 1,
            |s, _| s[0].records[3].request_index = 7,
            |s, _| s[0].records[3].phase = Phase::Measured,
            |s, _| s[0].records[3].helper = Helper::ControlSingleFileExt,
            |s, _| s[0].records[3].caller_helper_ns += 6_000_000,
            |s, _| s[0].request_source_id_witnesses[0].2.push('x'),
            |s, _| {
                s[0].request_begins.pop();
            },
            |_, p| p["gates"]["passed"] = json!(false),
            |_, p| p["gates"]["warmup_mechanism"]["passed"] = json!(false),
            |_, p| p["control"]["warmup_upload"]["ordered_nvme_ids_sha256"] = json!("bad"),
            |_, p| p["control"]["production"]["production_batch_experts"] = json!(0),
            |_, p| p["control"]["mechanism"]["source_nvme_reads"] = json!(0),
            |_, p| p["treatment"]["upload"]["accounting_errors"] = json!(1),
            |_, p| p["treatment"]["work"]["token_loop"]["fatal_failures"] = json!(1),
            |_, p| p["treatment"]["mechanism"]["reservation_identity_sha256"] = json!("changed"),
            |_, p| {
                p["control"]["benchmark"]["per_run_results"][0]["generated_token_ids"][0] =
                    json!(999)
            },
            |_, p| p["frozen_workload"]["measured_runs"] = json!(2),
            |_, p| {
                p["treatment"]["upload"]["source_upload_fd_proof"]
                    ["source_upload_fd_proof_failures"] = json!(1)
            },
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let (mut s, mut p) = fixture();
            mutate(&mut s, &mut p);
            let a = audited(s, p);
            assert_eq!(a.analysis.disposition, "NON_AUTHORITATIVE", "{i}");
            assert!(!a.analysis.errors.is_empty(), "{i}");
        }
    }
    #[test]
    fn hma1e_any_read_retry_transient_breaker_failure_rejects() {
        for mode in 0..7 {
            let (mut s, p) = fixture();
            let r = &mut s[0].records[4];
            match mode {
                0 => r.diagnostic_retry_count = 1,
                1 => r.reads[0].transient_events = 1,
                2 => r.reads[0].breaker_event = true,
                3 => r.reads[0].retry_attempt_count = 1,
                4 => r.reads[0].success = false,
                5 => r.reads[0].attempt_wall_ns[1] = Some(0),
                _ => r.reads[0].attempt_wall_ns[0] = None,
            };
            assert_eq!(audited(s, p).analysis.disposition, "NON_AUTHORITATIVE");
        }
    }
    #[test]
    fn hma1e_transcript_complete_hash_and_all_retry_messages() {
        let (s, p) = fixture();
        let bytes = serde_json::to_vec(&Envelope::new(p, s)).unwrap();
        let log = complete_log(&bytes);
        assert_ne!(
            audit_bytes(&bytes, &log).analysis.disposition,
            "NON_AUTHORITATIVE"
        );
        for msg in [
            "transient I/O error; retrying",
            "expert fetch failed; will retry",
            "expert fetch recovered after retry",
            "drive circuit breaker",
        ] {
            let mut bad = log.clone();
            bad.extend_from_slice(msg.as_bytes());
            assert_eq!(
                audit_bytes(&bytes, &bad).analysis.disposition,
                "NON_AUTHORITATIVE"
            );
        }
        for bad in [
            Vec::new(),
            b"unrelated clean text".to_vec(),
            format!("{BEGIN}\n").into_bytes(),
            [log.clone(), log].concat(),
        ] {
            assert_eq!(
                audit_bytes(&bytes, &bad).analysis.disposition,
                "NON_AUTHORITATIVE"
            );
        }
        assert_eq!(
            retry_warning_count(b"transient I/O error; retrying\ntransient I/O error; retrying"),
            2
        );
    }
    #[test]
    fn hma1e_raw_is_non_authoritative_and_cached_corruption_rejected() {
        let (s, p) = fixture();
        let raw = Envelope::new(p, s);
        assert_eq!(raw.analysis.disposition, "NON_AUTHORITATIVE");
        assert_eq!(
            raw.analysis.errors,
            ["completed external transcript audit pending"]
        );
        let value = serde_json::to_value(raw).unwrap();
        for path in [
            "/schema",
            "/analysis/disposition",
            "/analysis/primary/dcrit_ns",
            "/analysis/strata/0/comparison/dmax_ns",
            "/analysis/decision",
            "/execution_order/0",
        ] {
            let mut bad = value.clone();
            *bad.pointer_mut(path).unwrap() = json!("corrupt");
            let bytes = serde_json::to_vec(&bad).unwrap();
            assert_eq!(
                audit_bytes(&bytes, &complete_log(&bytes))
                    .analysis
                    .disposition,
                "NON_AUTHORITATIVE"
            );
        }
    }
    #[test]
    fn hma1e_audit_files_are_immutable_and_malformed_report_is_explicit() {
        let root = std::env::temp_dir().join(format!(
            "hma1e-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let (s, p) = fixture();
        let bytes = serde_json::to_vec(&Envelope::new(p, s)).unwrap();
        let input = root.join("raw.json");
        let log = root.join("log.txt");
        let out = root.join("audit.json");
        std::fs::write(&input, &bytes).unwrap();
        std::fs::write(&log, complete_log(&bytes)).unwrap();
        audit_command(&input, &log, &out).unwrap();
        assert_eq!(std::fs::read(&input).unwrap(), bytes);
        assert!(audit_command(&input, &log, &out).is_err());
        assert!(audit_command(&input, &log, &input).is_err());
        std::fs::write(&input, b"malformed").unwrap();
        let bad = root.join("bad.json");
        assert!(audit_command(&input, &log, &bad).is_err());
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(bad).unwrap()).unwrap();
        assert_eq!(report["analysis"]["disposition"], "NON_AUTHORITATIVE");
        assert!(report["analysis"]["errors"].as_array().unwrap().len() > 0);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn hma1e_display_rounding_does_not_control_thresholds() {
        let base = 100_000_000_000_000u64;
        for (d, label) in [
            (2_999_999_999_999, "DIRECTIONAL_ORDER_ROBUST"),
            (3_000_000_000_000, "ORDER_ROBUST_STRAGGLER"),
        ] {
            let p = compare(
                EndpointTotals {
                    read_critical_span_ns: base,
                    batch_max_read_wall_ns: base,
                    ..Default::default()
                },
                EndpointTotals {
                    read_critical_span_ns: base + d,
                    batch_max_read_wall_ns: base + d,
                    ..Default::default()
                },
            );
            assert_eq!(
                format!("{:.6}", p.dcrit_percent_of_control.unwrap()),
                "3.000000"
            );
            assert_eq!(classify(&p, &consistent(&p)).unwrap(), label);
        }
    }
    #[test]
    fn hma1e_record_store_never_grows_and_rejects_overflow() {
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
        );
        observer.commit(
            &[2],
            Helper::ControlSingleFileExt,
            &raw,
            now,
            now,
            &Ok(2_658_304),
        );
        let s = observer.snapshot();
        assert_eq!(s.records.len(), 1);
        assert_eq!(s.overflow, 1);
        assert!(s.records[0].timing_error.is_some());
        assert_eq!(observer.inner.lock().snapshot.records.capacity(), 1);
    }
    #[test]
    fn hma1e_caller_reconciliation_exact_tolerance_boundary() {
        assert!(reconcile_timer(100_000_000, 105_000_000));
        assert!(!reconcile_timer(100_000_000, 105_000_001));
        assert!(reconcile_timer(4_000_000_000, 4_010_000_000));
        assert!(!reconcile_timer(4_000_000_000, 4_010_000_001));
    }
    #[test]
    fn hma1e_cli_reuses_production_config_and_offline_audit_has_no_runtime() {
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
        let this = include_str!("gpu_native_source_order_straggler_production.rs");
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
    fn hma1e_incomplete_worker_timing_never_hides_retry_evidence() {
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
        );
        let snapshot = observer.snapshot();
        let record = &snapshot.records[0];
        assert!(record.timing_error.is_some());
        assert_eq!(record.diagnostic_retry_count, 1);
        assert_eq!(record.reads[0].transient_events, 2);
        assert_eq!(record.raw_helper_total_ns, Some(0));
    }
    #[test]
    fn hma1e_raw_timestamps_reconstruct_critical_path_not_parallel_sum() {
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
    fn hma1e_checked_aggregation_and_empty_primary_rejected() {
        let mut r = fixture_record(Arm::Control, Phase::Measured, 0, 0, 0, 2);
        r.timing.read_critical_span_ns = u64::MAX;
        assert!(endpoint_total(&[&r, &r]).is_err());
        assert!(classify(
            &compare(EndpointTotals::default(), EndpointTotals::default()),
            &[]
        )
        .is_err());
    }
    #[test]
    fn hma1e_observer_context_never_grows_beyond_four_requests() {
        let observer = Observer::new(Arm::Control, 0);
        observer.finish_request(None);
        for _ in 0..6 {
            observer.begin_request(Phase::Warmup, 0);
            observer.finish_request(Some(String::new()));
        }
        let s = observer.snapshot();
        assert_eq!(s.request_begins.len(), 4);
        assert_eq!(s.request_source_id_witnesses.len(), 4);
        assert!(s.context_errors > 0);
    }

    mod repair2_tests {
        use super::*;

        fn production() -> (Vec<StoreSnapshot>, serde_json::Value) {
            let (stores, mut p) = fixture();
            p["gates"]["measured_mechanism"] = p["gates"]
                .as_object_mut()
                .unwrap()
                .remove("mechanism")
                .unwrap();
            for (arm, id) in [("control", "3"), ("treatment", "2")] {
                p[arm]["benchmark"]["runtime_contract"]["legacy_execution_plan"]["context_id"] =
                    json!(id);
            }
            (stores, p)
        }
        fn prior(raw: Vec<u8>, log: Vec<u8>) -> [Vec<u8>; 4] {
            let original = serde_json::to_vec(&audit_bytes(&raw, &log)).unwrap();
            let ids = [
                ArtifactIdentity::of(&raw),
                ArtifactIdentity::of(&log),
                ArtifactIdentity::of(&original),
            ];
            let repair1 = repair_bytes(
                &raw,
                &log,
                &original,
                &EvidenceHashes {
                    raw: &ids[0].sha256,
                    transcript: &ids[1].sha256,
                    original_audit: &ids[2].sha256,
                },
            );
            [raw, log, original, serde_json::to_vec(&repair1).unwrap()]
        }
        fn evidence_from(stores: Vec<StoreSnapshot>, p: serde_json::Value) -> [Vec<u8>; 4] {
            let raw = serde_json::to_vec(&Envelope::new(p, stores)).unwrap();
            let log = complete_log(&raw);
            prior(raw, log)
        }
        fn evidence() -> [Vec<u8>; 4] {
            let (s, p) = production();
            evidence_from(s, p)
        }
        fn bound(e: &[Vec<u8>; 4]) -> Repair2Report {
            let ids = e.each_ref().map(|b| ArtifactIdentity::of(b));
            repair2_bytes(
                &e[0],
                &e[1],
                &e[2],
                &e[3],
                &EvidenceHashes {
                    raw: &ids[0].sha256,
                    transcript: &ids[1].sha256,
                    original_audit: &ids[2].sha256,
                },
                &ids[3].sha256,
            )
        }
        fn rejected(report: &Repair2Report) {
            assert_eq!(report.result.disposition, "NON_AUTHORITATIVE");
            assert_eq!(report.final_disposition, "NON_AUTHORITATIVE");
            assert_eq!(report.result.decision, decision("NON_AUTHORITATIVE"));
            assert!(!report.result.errors.is_empty());
            assert_eq!(report.final_errors, report.result.errors);
        }
        fn mutate_contract(field: &str, edit: impl FnOnce(&mut serde_json::Value)) {
            let (s, mut p) = production();
            edit(&mut p["treatment"]["benchmark"][field]);
            assert!(!ContextIdProof::of(&p).passed());
            rejected(&bound(&evidence_from(s, p)));
        }

        #[test]
        fn hma1e_repair2_sole_context_id_accepted_with_both_historical_failures() {
            let e = evidence();
            let legacy = audit_bytes(&e[0], &e[1]);
            assert_eq!(legacy.analysis.disposition, "NON_AUTHORITATIVE");
            assert_eq!(legacy.analysis.errors, [HISTORICAL_ERROR]);
            let repair1: serde_json::Value = serde_json::from_slice(&e[3]).unwrap();
            assert_eq!(repair1["legacy_reproduction_pass"], true);
            assert_eq!(repair1["corrected_disposition"], "NON_AUTHORITATIVE");
            assert_eq!(repair1["corrected_errors"], json!([REPAIR1_ERROR]));
            let r = bound(&e);
            assert!(r.legacy_reproduction_pass && r.repair1_reproduction_pass);
            assert!(r.context_id_proof.passed());
            assert_eq!(r.result.disposition, "DIRECTIONAL_ORDER_ROBUST");
            assert!(r.result.errors.is_empty());
            assert_eq!(r.result.external_retry_warning_occurrences, Some(0));
            assert_eq!(r.result.external_existing_breaker_retry_events, Some(0));
            let value = serde_json::to_value(&r).unwrap();
            assert_eq!(value["schema"], REPAIR2_SCHEMA);
            assert_eq!(value["excluded_path"], CONTEXT_ID_PATH);
            assert_eq!(
                value["cross_arm_differences_before_exclusion"][0]["path"],
                CONTEXT_ID_PATH
            );
            assert_eq!(
                value["cross_arm_differences_before_exclusion"][0]["control"],
                "3"
            );
            assert_eq!(
                value["cross_arm_differences_before_exclusion"][0]["treatment"],
                "2"
            );
            for field in [
                "disposition",
                "errors",
                "primary",
                "streams",
                "strata",
                "decision",
                "external_retry_warning_occurrences",
                "external_existing_breaker_retry_events",
            ] {
                assert_eq!(value["final_analysis"][field], value[field]);
            }
            assert!(
                !r.hardware_rerun_performed
                    && !r.runtime_constructed
                    && !r.model_loaded
                    && !r.gpu_constructed
            );
        }
        #[test]
        fn hma1e_repair2_equal_context_ids_reject() {
            mutate_contract("runtime_contract", |r| {
                r["legacy_execution_plan"]["context_id"] = json!("3")
            });
        }
        #[test]
        fn hma1e_repair2_zero_context_id_rejects() {
            mutate_contract("runtime_contract", |r| {
                r["legacy_execution_plan"]["context_id"] = json!("0")
            });
        }
        #[test]
        fn hma1e_repair2_nonnumeric_overflow_or_nonstring_context_id_rejects() {
            for id in [
                json!("test"),
                json!("18446744073709551616"),
                json!("-2"),
                json!(2),
                json!(null),
            ] {
                mutate_contract("runtime_contract", |r| {
                    r["legacy_execution_plan"]["context_id"] = id
                });
            }
        }
        #[test]
        fn hma1e_repair2_missing_context_id_rejects() {
            mutate_contract("runtime_contract", |r| {
                r["legacy_execution_plan"]
                    .as_object_mut()
                    .unwrap()
                    .remove("context_id");
            });
        }
        #[test]
        fn hma1e_repair2_positive_distinct_but_not_frozen_ids_reject() {
            for id in ["4", "02", "+2", " 2"] {
                mutate_contract("runtime_contract", |r| {
                    r["legacy_execution_plan"]["context_id"] = json!(id)
                });
            }
        }
        #[test]
        fn hma1e_repair2_second_runtime_difference_rejects() {
            mutate_contract("runtime_contract", |r| r["compute_offload"] = json!("cpu"));
        }
        #[test]
        fn hma1e_repair2_request_difference_rejects() {
            mutate_contract("request", |r| r["extra"] = json!(1));
        }
        #[test]
        fn hma1e_repair2_production_configuration_difference_rejects() {
            mutate_contract("production_configuration", |r| r["extra"] = json!(1));
        }
        #[test]
        fn hma1e_repair2_production_semantics_difference_rejects() {
            mutate_contract("production_semantics", |r| r["extra"] = json!(1));
        }
        #[test]
        fn hma1e_repair2_model_identity_difference_rejects() {
            mutate_contract("model_identity", |r| r["extra"] = json!(1));
        }
        #[test]
        fn hma1e_repair2_hardware_difference_rejects() {
            mutate_contract("hardware", |r| r["name"] = json!("other"));
        }
        #[test]
        fn hma1e_repair2_comparison_cannot_ignore_any_second_field() {
            // Exercise the equality helper directly, independently of Stage 3.
            // Alter/remove EVERY other existing node, and insert unknown fields
            // at every object. A later second exclusion must break this test.
            fn visit(
                c: &serde_json::Value,
                t: &serde_json::Value,
                node: &serde_json::Value,
                pointer: String,
            ) {
                if pointer == "/legacy_execution_plan/context_id" {
                    return;
                }
                if !pointer.is_empty() {
                    let mut changed = t.clone();
                    *changed.pointer_mut(&pointer).unwrap() = if node.is_null() {
                        json!(false)
                    } else {
                        json!(null)
                    };
                    assert!(
                        !runtime_contracts_equal_except_context_id(c, &changed),
                        "ignored {pointer}"
                    );
                }
                match node {
                    serde_json::Value::Object(m) => {
                        let mut added = t.clone();
                        added
                            .pointer_mut(&pointer)
                            .unwrap()
                            .as_object_mut()
                            .unwrap()
                            .insert("future_field".into(), json!(1));
                        assert!(
                            !runtime_contracts_equal_except_context_id(c, &added),
                            "ignored new field at {pointer}"
                        );
                        for (k, v) in m {
                            if pointer == "/legacy_execution_plan" && k == "context_id" {
                                continue;
                            }
                            let mut removed = t.clone();
                            removed
                                .pointer_mut(&pointer)
                                .unwrap()
                                .as_object_mut()
                                .unwrap()
                                .remove(k);
                            assert!(
                                !runtime_contracts_equal_except_context_id(c, &removed),
                                "ignored removed {pointer}/{k}"
                            );
                            visit(
                                c,
                                t,
                                v,
                                format!("{pointer}/{}", k.replace('~', "~0").replace('/', "~1")),
                            );
                        }
                    }
                    serde_json::Value::Array(a) => {
                        for (i, v) in a.iter().enumerate() {
                            visit(c, t, v, format!("{pointer}/{i}"));
                        }
                    }
                    _ => {}
                }
            }
            let (_, p) = production();
            let c = &p["control"]["benchmark"]["runtime_contract"];
            let t = &p["treatment"]["benchmark"]["runtime_contract"];
            let original = (c.clone(), t.clone());
            assert!(runtime_contracts_equal_except_context_id(c, t));
            visit(c, t, t, String::new());
            assert_eq!((c, t), (&original.0, &original.1));
        }
        #[test]
        fn hma1e_repair2_deep_diff_complete_arrays_missing_null_and_dotted_keys() {
            let c = json!({"a":[1,2], "missing":null, "nested":{"x":1}, "legacy_execution_plan.context_id":"3"});
            let t = json!({"a":[3,2,4], "nested":{"x":2}, "legacy_execution_plan.context_id":"2"});
            let mut diff = vec![];
            contract_differences(
                Some(&c),
                Some(&t),
                &mut vec!["runtime_contract".into()],
                &mut diff,
            );
            assert_eq!(diff.len(), 5);
            assert_eq!(
                diff.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
                [
                    "runtime_contract.a.0",
                    "runtime_contract.a.2",
                    "runtime_contract.legacy_execution_plan.context_id",
                    "runtime_contract.missing",
                    "runtime_contract.nested.x"
                ]
            );
            assert_eq!(diff[2].path_components.len(), 2);
            assert_eq!(diff[3].control, Some(json!(null)));
            assert_eq!(diff[3].treatment, None);
        }
        #[test]
        fn hma1e_repair2_all_four_hashes_checked_before_deserialization() {
            let e = evidence();
            let ids = e.each_ref().map(|b| ArtifactIdentity::of(b));
            for bad in 0..4 {
                let mut hashes = ids.each_ref().map(|id| id.sha256.as_str());
                hashes[bad] = "wrong";
                let r = repair2_bytes(
                    &e[0],
                    &e[1],
                    &e[2],
                    &e[3],
                    &EvidenceHashes {
                        raw: hashes[0],
                        transcript: hashes[1],
                        original_audit: hashes[2],
                    },
                    hashes[3],
                );
                rejected(&r);
                assert!(!r.legacy_reproduction_pass && !r.repair1_reproduction_pass);
                assert!(r.final_analysis.is_none());
                assert_eq!(r.result.errors.len(), 1);
                assert!(r.result.errors[0].ends_with("SHA-256 mismatch"));
            }
            let r = repair2_bytes(
                b"not JSON",
                b"not a transcript",
                b"bad",
                b"bad",
                &CONSUMED_FIRST,
                CONSUMED_REPAIR1_SHA256,
            );
            assert_eq!(r.result.errors.len(), 4);
            assert!(r
                .result
                .errors
                .iter()
                .all(|e| e.ends_with("SHA-256 mismatch")));
        }
        fn corrupt_repair1(edit: impl FnOnce(&mut serde_json::Value)) {
            let mut e = evidence();
            let mut v: serde_json::Value = serde_json::from_slice(&e[3]).unwrap();
            edit(&mut v);
            e[3] = serde_json::to_vec(&v).unwrap();
            let r = bound(&e);
            rejected(&r);
            assert!(r.legacy_reproduction_pass && !r.repair1_reproduction_pass);
            assert_eq!(r.result.external_retry_warning_occurrences, Some(0));
            assert_eq!(r.result.external_existing_breaker_retry_events, Some(0));
            assert!(r.final_analysis.is_none());
            assert!(r
                .context_id_proof
                .cross_arm_difference_count_before_exclusion
                .is_none());
        }
        #[test]
        fn hma1e_repair2_wrong_repair1_disposition_rejects() {
            corrupt_repair1(|v| v["corrected_disposition"] = json!("ORDER_ROBUST_STRAGGLER"));
        }
        #[test]
        fn hma1e_repair2_repair1_extra_error_rejects() {
            corrupt_repair1(|v| v["corrected_errors"] = json!([REPAIR1_ERROR, "extra"]));
        }
        #[test]
        fn hma1e_repair2_repair1_missing_error_rejects() {
            corrupt_repair1(|v| v["corrected_errors"] = json!([]));
        }
        #[test]
        fn hma1e_repair2_entire_repair1_semantic_result_required() {
            for pointer in [
                "/schema",
                "/mode",
                "/raw_report/sha256",
                "/completed_transcript/bytes",
                "/original_audit/sha256",
                "/evidence_child_sha",
                "/legacy_reproduction_pass",
                "/corrected_analysis/primary/dcrit_ns",
                "/primary/dmax_ns",
                "/streams",
                "/strata",
                "/decision",
                "/external_retry_warning_occurrences",
                "/external_existing_breaker_retry_events",
                "/runtime_constructed",
                "/hardware_rerun_performed",
                "/model_loaded",
                "/gpu_constructed",
            ] {
                corrupt_repair1(|v| *v.pointer_mut(pointer).unwrap() = json!("corrupt"));
            }
            corrupt_repair1(|v| v["unexpected"] = json!(true));
            let mut e = evidence();
            e[3] = b"invalid JSON".to_vec();
            rejected(&bound(&e));
            let mut e = evidence();
            let v: serde_json::Value = serde_json::from_slice(&e[3]).unwrap();
            e[3] = serde_json::to_vec_pretty(&v).unwrap();
            assert!(bound(&e).repair1_reproduction_pass);
        }
        #[test]
        fn hma1e_repair2_original_audit_reproduction_still_exact() {
            for pointer in [
                "/analysis/errors",
                "/analysis/primary/dcrit_ns",
                "/schema",
                "/source_report_bytes",
            ] {
                let mut e = evidence();
                let mut v: serde_json::Value = serde_json::from_slice(&e[2]).unwrap();
                *v.pointer_mut(pointer).unwrap() = json!("corrupt");
                e[2] = serde_json::to_vec(&v).unwrap();
                let r = bound(&e);
                rejected(&r);
                assert!(!r.legacy_reproduction_pass);
            }
        }
        #[test]
        fn hma1e_repair2_retry_and_breaker_contamination_rejects() {
            for message in [
                "transient I/O error; retrying",
                "expert fetch failed; will retry",
                "expert fetch recovered after retry",
                "drive circuit breaker",
            ] {
                let [raw, mut log, _, _] = evidence();
                log.extend_from_slice(message.as_bytes());
                let r = bound(&prior(raw, log));
                rejected(&r);
                assert!(!r.legacy_reproduction_pass);
            }
            for mutation in 0..3 {
                let (mut s, p) = production();
                match mutation {
                    0 => s[0].records[4].reads[0].retry_attempt_count = 1,
                    1 => s[0].records[4].reads[0].breaker_event = true,
                    _ => s[0].records[4].reads[0].transient_events = 1,
                }
                rejected(&bound(&evidence_from(s, p)));
            }
        }
        #[test]
        fn hma1e_repair2_transcript_markers_and_hash_still_required() {
            let [raw, log, _, _] = evidence();
            let log = String::from_utf8(log).unwrap();
            for corrupt in [
                log.replace(BEGIN, "bad"),
                log.replace(END, "bad"),
                format!("{BEGIN}\n{log}"),
                format!("{END}{:x}\n{BEGIN}\n", Sha256::digest(&raw)),
                log.replace(&format!("{:x}", Sha256::digest(&raw)), &"0".repeat(64)),
            ] {
                rejected(&bound(&prior(raw.clone(), corrupt.into_bytes())));
            }
        }
        #[test]
        fn hma1e_repair2_final_stage_preserves_generated_evidence_validation() {
            let (s, mut p) = production();
            p["treatment"]["benchmark"]["per_run_results"][0]["generated_token_ids"][0] =
                json!(999);
            let r = bound(&evidence_from(s, p));
            assert!(
                r.legacy_reproduction_pass
                    && r.repair1_reproduction_pass
                    && r.context_id_proof.passed()
            );
            assert!(r.final_analysis.is_some());
            rejected(&r);
            assert!(r
                .result
                .errors
                .iter()
                .any(|e| e.contains("generated token")));
        }
        #[test]
        fn hma1e_repair2_classifier_boundaries_unchanged() {
            hma1e_exact_plus_three_boundary();
            hma1e_exact_minus_three_boundary();
            hma1e_strict_one_percent_and_directional();
            hma1e_critical_max_disagreement_is_ambiguous();
            hma1e_request_both_endpoints_same_two_requests();
            hma1e_both_ordinal_halves_required();
            hma1e_five_of_seven_width_votes();
        }
        #[test]
        fn hma1e_repair2_all_input_aliases_existing_outputs_and_links_reject() {
            let dir =
                std::env::temp_dir().join(format!("hma1e-repair2-paths-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let paths = ["raw", "log", "original", "repair1"].map(|n| dir.join(n));
            let e = evidence();
            for (path, bytes) in paths.iter().zip(&e) {
                std::fs::write(path, bytes).unwrap();
            }
            let run = |out: &std::path::Path| {
                repair2_audit_command(&paths[0], &paths[1], &paths[2], &paths[3], out)
            };
            for path in &paths {
                assert!(run(path).is_err());
                assert!(run(&dir.join(".").join(path.file_name().unwrap())).is_err());
                let hard = dir.join("hard");
                std::fs::hard_link(path, &hard).unwrap();
                assert!(run(&hard).is_err());
                std::fs::remove_file(&hard).unwrap();
                #[cfg(unix)]
                {
                    let link = dir.join("link");
                    std::os::unix::fs::symlink(path, &link).unwrap();
                    assert!(run(&link).is_err());
                    std::fs::remove_file(link).unwrap();
                }
            }
            #[cfg(unix)]
            {
                let link = dir.join("dangling");
                std::os::unix::fs::symlink(dir.join("absent"), &link).unwrap();
                assert!(run(&link).is_err());
            }
            let out = dir.join("new.json");
            assert!(run(&out).is_err()); // Frozen production hashes reject synthetic bytes.
            let first = std::fs::read(&out).unwrap();
            let v: serde_json::Value = serde_json::from_slice(&first).unwrap();
            assert_eq!(v["final_disposition"], "NON_AUTHORITATIVE");
            assert_eq!(v["final_errors"].as_array().unwrap().len(), 4);
            assert_eq!(v["legacy_reproduction_pass"], false);
            assert_eq!(v["repair1_reproduction_pass"], false);
            assert!(v["final_analysis"].is_null());
            assert!(run(&out).is_err());
            assert_eq!(std::fs::read(&out).unwrap(), first);
            assert!(run(&dir).is_err());
            for (path, bytes) in paths.iter().zip(&e) {
                assert_eq!(std::fs::read(path).unwrap(), *bytes);
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
        #[test]
        fn hma1e_repair2_cli_requires_only_five_paths_and_returns_before_startup() {
            use clap::Parser;
            let args = [
                "mer",
                REPAIR2_MODE,
                "--report-in",
                "raw",
                "--completed-run-log",
                "log",
                "--original-audit-in",
                "original",
                "--repair1-audit-in",
                "repair1",
                "--report-out",
                "new",
            ];
            let cli = crate::Cli::try_parse_from(args).unwrap();
            assert!(crate::startup_config_path(&cli.cmd).is_none());
            assert!(matches!(
                cli.cmd,
                crate::Cmd::Repair2AuditGpuNativeSourceOrderStragglerProduction { .. }
            ));
            for start in (2..12).step_by(2) {
                let mut missing = args.to_vec();
                missing.drain(start..start + 2);
                assert!(crate::Cli::try_parse_from(missing).is_err());
            }
            for flag in [
                "--config",
                "--model",
                "--expected-adapter-name",
                "--storage",
                "--raw-sha256",
                "--repair1-sha256",
            ] {
                let mut extra = args.to_vec();
                extra.extend([flag, "override"]);
                assert!(crate::Cli::try_parse_from(extra).is_err());
            }
            let main = include_str!("main.rs")
                .split("fn main() ->")
                .nth(1)
                .unwrap();
            let early = main.split("let worker_protocol_stdout").next().unwrap();
            assert!(early.contains("return crate::gpu_native_source_order_straggler_production::repair2_audit_command("));
            let source = include_str!("gpu_native_source_order_straggler_production.rs");
            let repair2 = source
                .split("const REPAIR2_SCHEMA:")
                .nth(1)
                .unwrap()
                .split("const REPAIR_SCHEMA:")
                .next()
                .unwrap();
            assert_eq!(repair2.matches("std::fs::read(").count(), 4);
            assert!(
                repair2.contains("&CONSUMED_FIRST, CONSUMED_REPAIR1_SHA256")
                    || repair2.contains("&CONSUMED_FIRST,\n        CONSUMED_REPAIR1_SHA256")
            );
            assert!(repair2.contains(".create_new(true)"));
            let validator = include_str!("gpu_native_source_to_upload_copy_elision_production.rs")
                .split("pub(crate) fn validate_recorded_authority(")
                .nth(1)
                .unwrap()
                .split("#[cfg(test)]")
                .next()
                .unwrap();
            for region in [early, repair2, validator] {
                for forbidden in [
                    "run_command(",
                    "run_order_straggler_command(",
                    "construct_runtime(",
                    "Config::from_file(",
                    "new_multi_thread(",
                    "install_default(",
                    "RealModel::",
                    "from_dir(",
                    "NvmeStorage::",
                    "read_expert(",
                    "execute_request(",
                    "wgpu::",
                    "request_adapter(",
                    "request_device(",
                    "std::env::",
                ] {
                    if region == early && forbidden == "std::env::" {
                        continue;
                    }
                    assert!(
                        !region.contains(forbidden),
                        "offline path contains {forbidden}"
                    );
                }
            }
        }
    }

    mod repair_tests {
        use super::*;
        use crate::gpu_native_physical_install_staging::source_to_upload_production::validate_recorded_authority;

        fn repaired_fixture() -> (Vec<StoreSnapshot>, serde_json::Value) {
            let (stores, mut p) = fixture();
            // Match the actual production-v2 serialized gate shape, independently
            // of the historical fixture (whose arm snapshots remain unchanged).
            let gate = p["gates"]
                .as_object_mut()
                .unwrap()
                .remove("mechanism")
                .unwrap();
            p["gates"]["measured_mechanism"] = gate;
            (stores, p)
        }
        fn repaired_validator(p: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
            validate_recorded_authority_with_measured_gate_key(
                p,
                MeasuredMechanismGateKey::Repaired,
            )
        }
        fn evidence() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            let (stores, p) = repaired_fixture();
            let raw = serde_json::to_vec(&Envelope::new(p, stores)).unwrap();
            let log = complete_log(&raw);
            let original = serde_json::to_vec(&audit_bytes(&raw, &log)).unwrap();
            (raw, log, original)
        }
        fn bound_repair(raw: &[u8], log: &[u8], original: &[u8]) -> RepairReport {
            let r = ArtifactIdentity::of(raw);
            let t = ArtifactIdentity::of(log);
            let o = ArtifactIdentity::of(original);
            repair_bytes(
                raw,
                log,
                original,
                &EvidenceHashes {
                    raw: &r.sha256,
                    transcript: &t.sha256,
                    original_audit: &o.sha256,
                },
            )
        }
        fn stage1_rejected(report: RepairReport) {
            assert!(!report.legacy_reproduction_pass);
            assert!(report.corrected_analysis.is_none());
            assert!(report.corrected_disposition.is_none());
            assert_eq!(report.result.disposition, "NON_AUTHORITATIVE");
            assert!(!report.result.errors.is_empty());
        }

        #[test]
        fn hma1e_repair_valid_measured_gate_and_historical_failure() {
            let (_, p) = repaired_fixture();
            repaired_validator(&p).unwrap();
            assert_eq!(
                validate_recorded_authority(&p).unwrap_err().to_string(),
                HISTORICAL_ERROR
            );
            let (raw, log, original) = evidence();
            assert_eq!(audit_bytes(&raw, &log).analysis.errors, [HISTORICAL_ERROR]);
            let report = bound_repair(&raw, &log, &original);
            assert!(report.legacy_reproduction_pass);
            assert_eq!(report.result.disposition, "DIRECTIONAL_ORDER_ROBUST");
            assert!(report.result.errors.is_empty());
            assert_eq!(report.original_errors.unwrap(), [HISTORICAL_ERROR]);
            assert_eq!(report.original_disposition.unwrap(), "NON_AUTHORITATIVE");
            assert_eq!(report.result.external_retry_warning_occurrences, Some(0));
            assert_eq!(
                report.result.external_existing_breaker_retry_events,
                Some(0)
            );
            assert!(
                !report.hardware_rerun_performed
                    && !report.runtime_constructed
                    && !report.model_loaded
                    && !report.gpu_constructed
            );
        }
        #[test]
        fn hma1e_repair_missing_measured_gate_rejected_even_with_legacy_gate() {
            let (_, p) = fixture();
            validate_recorded_authority(&p).unwrap();
            assert_eq!(
                repaired_validator(&p).unwrap_err().to_string(),
                HISTORICAL_ERROR
            );
        }
        #[test]
        fn hma1e_repair_corrupt_serialized_measured_gate_rejected() {
            let (_, mut p) = repaired_fixture();
            p["gates"]["measured_mechanism"]["passed"] = json!(false);
            assert_eq!(
                repaired_validator(&p).unwrap_err().to_string(),
                HISTORICAL_ERROR
            );
        }
        #[test]
        fn hma1e_repair_serialized_pass_cannot_override_reconstructed_failure() {
            let (_, mut p) = repaired_fixture();
            p["treatment"]["upload"]["accounting_errors"] = json!(1);
            assert_eq!(p["gates"]["measured_mechanism"]["passed"], true);
            assert_eq!(
                repaired_validator(&p).unwrap_err().to_string(),
                HISTORICAL_ERROR
            );
        }
        #[test]
        fn hma1e_repair_warmup_still_requires_warmup_mechanism() {
            let (_, mut p) = repaired_fixture();
            p["gates"]
                .as_object_mut()
                .unwrap()
                .remove("warmup_mechanism");
            assert_eq!(
                repaired_validator(&p).unwrap_err().to_string(),
                HISTORICAL_ERROR
            );
            let (_, mut p) = repaired_fixture();
            p["treatment"]["warmup_upload"]["accounting_errors"] = json!(1);
            assert_eq!(
                repaired_validator(&p).unwrap_err().to_string(),
                HISTORICAL_ERROR
            );
        }
        #[test]
        fn hma1e_repair_cached_analysis_is_legacy_and_required() {
            let (raw, _, _) = evidence();
            let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(
                value["analysis"]["errors"],
                json!([
                    HISTORICAL_ERROR,
                    "completed external transcript audit pending"
                ])
            );
            value["analysis"]["strata"][0]["key"] = json!(999);
            let raw = serde_json::to_vec(&value).unwrap();
            let log = complete_log(&raw);
            let original = serde_json::to_vec(&audit_bytes(&raw, &log)).unwrap();
            let report = bound_repair(&raw, &log, &original);
            assert!(report
                .result
                .errors
                .iter()
                .any(|s| s == "raw analysis/primary/strata reconstruction mismatch"));
            stage1_rejected(report);
        }
        #[test]
        fn hma1e_repair_corrected_analysis_never_compared_to_cached_legacy() {
            let (raw, log, original) = evidence();
            let raw_before = raw.clone();
            let raw_value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            let report = bound_repair(&raw, &log, &original);
            assert!(report.legacy_reproduction_pass);
            let corrected = report.corrected_analysis.unwrap();
            assert!(corrected.errors.is_empty());
            assert_ne!(
                raw_value["analysis"],
                serde_json::to_value(&corrected).unwrap()
            );
            assert_eq!(raw, raw_before);
        }
        #[test]
        fn hma1e_repair_each_exact_input_hash_is_mandatory() {
            let (raw, log, original) = evidence();
            let r = ArtifactIdentity::of(&raw);
            let t = ArtifactIdentity::of(&log);
            let o = ArtifactIdentity::of(&original);
            for field in 0..3 {
                let mut expected = EvidenceHashes {
                    raw: &r.sha256,
                    transcript: &t.sha256,
                    original_audit: &o.sha256,
                };
                match field {
                    0 => expected.raw = CONSUMED_FIRST.raw,
                    1 => expected.transcript = CONSUMED_FIRST.transcript,
                    _ => expected.original_audit = CONSUMED_FIRST.original_audit,
                }
                let report = repair_bytes(&raw, &log, &original, &expected);
                assert_eq!(report.result.errors.len(), 1);
                assert!(report.result.errors[0]
                    .contains(["raw report", "transcript", "original audit"][field]));
                stage1_rejected(report);
            }
            stage1_rejected(repair_bytes(&raw, &log, &original, &CONSUMED_FIRST));
        }
        #[test]
        fn hma1e_repair_original_extra_missing_or_different_error_rejected() {
            let (raw, log, original) = evidence();
            for errors in [
                json!([]),
                json!(["other"]),
                json!([HISTORICAL_ERROR, "other"]),
                json!([HISTORICAL_ERROR, HISTORICAL_ERROR]),
            ] {
                let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
                value["analysis"]["errors"] = errors;
                stage1_rejected(bound_repair(
                    &raw,
                    &log,
                    &serde_json::to_vec(&value).unwrap(),
                ));
            }
        }
        #[test]
        fn hma1e_repair_original_entire_canonical_result_must_match() {
            let (raw, log, original) = evidence();
            for pointer in [
                "/schema",
                "/mode",
                "/source_report_sha256",
                "/completed_transcript_sha256",
                "/source_report_bytes",
                "/completed_transcript_bytes",
                "/analysis/disposition",
                "/analysis/decision",
                "/analysis/primary",
                "/analysis/streams",
                "/analysis/strata",
                "/analysis/external_retry_warning_occurrences",
                "/analysis/external_existing_breaker_retry_events",
            ] {
                let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
                *value.pointer_mut(pointer).unwrap() = json!("corrupt");
                stage1_rejected(bound_repair(
                    &raw,
                    &log,
                    &serde_json::to_vec(&value).unwrap(),
                ));
            }
            let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
            value["unrecognized_authority"] = json!(true);
            stage1_rejected(bound_repair(
                &raw,
                &log,
                &serde_json::to_vec(&value).unwrap(),
            ));
            // Whitespace/key ordering are not authority fields; semantics are.
            let value = audit_bytes(&raw, &log);
            assert!(
                bound_repair(&raw, &log, &serde_json::to_vec_pretty(&value).unwrap())
                    .legacy_reproduction_pass
            );
        }
        #[test]
        fn hma1e_repair_retry_and_all_breaker_fetch_retry_contamination_rejected() {
            let (raw, log, _) = evidence();
            for warning in [
                "transient I/O error; retrying",
                "expert fetch recovered after retry",
                "expert fetch failed; will retry",
                "circuit breaker",
            ] {
                // Trailing warnings after COMPLETE still invalidate authority.
                let mut log = log.clone();
                log.extend_from_slice(warning.as_bytes());
                let original = serde_json::to_vec(&audit_bytes(&raw, &log)).unwrap();
                stage1_rejected(bound_repair(&raw, &log, &original));
            }
        }
        #[test]
        fn hma1e_repair_begin_end_order_and_completion_hash_mandatory() {
            let (raw, log, _) = evidence();
            let end = format!("{END}{:x}", Sha256::digest(&raw));
            for log in [
                format!("{end}\n"),
                format!("{BEGIN}\n"),
                format!("{end}\n{BEGIN}\n"),
                format!("{BEGIN}\n{BEGIN}\n{end}\n"),
                format!("{BEGIN}\n{end}\n{end}\n"),
                format!("{BEGIN}\n{END}{}\n", "0".repeat(64)),
                String::from_utf8(log)
                    .unwrap()
                    .replace(BEGIN, "HMA1E_QUALIFIER_BEGIN order=control,treatment"),
            ] {
                let original = serde_json::to_vec(&audit_bytes(&raw, log.as_bytes())).unwrap();
                let report = bound_repair(&raw, log.as_bytes(), &original);
                assert!(report
                    .result
                    .errors
                    .iter()
                    .any(|e| e.contains("completed transcript evidence")));
                stage1_rejected(report);
            }
        }
        #[test]
        fn hma1e_repair_raw_schema_mode_order_endpoints_mandatory() {
            let (raw, _, _) = evidence();
            for key in [
                "schema",
                "mode",
                "primary_definition",
                "ordinal_half_definition",
                "execution_order",
            ] {
                let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                value[key] = if key == "execution_order" {
                    json!(["control", "treatment"])
                } else {
                    json!("corrupt")
                };
                let raw = serde_json::to_vec(&value).unwrap();
                let log = complete_log(&raw);
                let original = serde_json::to_vec(&audit_bytes(&raw, &log)).unwrap();
                stage1_rejected(bound_repair(&raw, &log, &original));
            }
        }
        #[test]
        fn hma1e_repair_later_authority_failure_survives_legacy_reproduction() {
            // The old key failure short-circuits the validator. A later hidden
            // failure MUST still invalidate Stage 2 after legacy proof succeeds.
            let (stores, mut p) = repaired_fixture();
            p["control"]["warmup_ram_cache_state_sha256"] = json!("corrupt");
            let raw = serde_json::to_vec(&Envelope::new(p, stores)).unwrap();
            let log = complete_log(&raw);
            let original = serde_json::to_vec(&audit_bytes(&raw, &log)).unwrap();
            let report = bound_repair(&raw, &log, &original);
            assert!(report.legacy_reproduction_pass);
            assert_eq!(report.result.disposition, "NON_AUTHORITATIVE");
            assert_eq!(
                report.corrected_errors.unwrap(),
                ["warmup cache identity mismatch"]
            );
        }
        #[test]
        fn hma1e_repair_all_other_production_authority_checks_retained() {
            for (pointer, bad) in [
                ("/schema", json!("bad")),
                ("/mode", json!("bad")),
                ("/failure", json!("failed")),
                ("/qualification_pass", json!(false)),
                ("/benchmark_complete", json!(false)),
                ("/frozen_workload/measured_runs", json!(2)),
                ("/source_scheduler_changed", json!(true)),
                ("/slot_stride_bytes", json!(1)),
                ("/reconciliation/all_invariants_pass", json!(false)),
                ("/gates/behavioral", json!({})),
                ("/gates/work_equivalence", json!({})),
                ("/provenance/build/dirty", json!(true)),
                ("/provenance/artifacts/config/sha256", json!("bad")),
                ("/control/warmup_mechanism", json!({})),
                ("/control/mechanism", json!({})),
                ("/control/source", json!({})),
                ("/treatment/work/token_loop/fatal_failures", json!(1)),
                ("/treatment/work/engine_storage/nvme_bytes_read", json!(0)),
                (
                    "/treatment/production/production_batch_commit_violations",
                    json!(1),
                ),
                ("/gates/source_upload_fd_proof", json!({})),
                ("/control/warmup_ram_cache_state_sha256", json!("bad")),
                ("/control/complete", json!(false)),
                ("/control/isolated_runtime", json!(false)),
                ("/control/benchmark/benchmark_complete", json!(false)),
                ("/control/benchmark/failure", json!("failed")),
                ("/control/benchmark/runtime_shutdowns", json!([])),
                ("/control/benchmark/hardware/name", json!("other")),
                ("/control/benchmark/runtime_contract", json!({})),
                ("/control/benchmark/model_identity", json!({"corrupt":true})),
                (
                    "/control/benchmark/production_configuration",
                    json!({"corrupt":true}),
                ),
                (
                    "/control/benchmark/per_run_results/0/generated_token_ids/0",
                    json!(999),
                ),
                (
                    "/control/benchmark/per_run_results/0/generated_text_sha256",
                    json!("bad"),
                ),
            ] {
                let (_, mut p) = repaired_fixture();
                *p.pointer_mut(pointer)
                    .unwrap_or_else(|| panic!("missing fixture {pointer}")) = bad;
                assert!(repaired_validator(&p).is_err(), "accepted {pointer}");
            }
        }
        #[test]
        fn hma1e_repair_frozen_classifier_exact_boundaries() {
            for (delta, expected) in [
                (-3_000_001, "ORDER_DOMINATED_REVERSAL"),
                (-3_000_000, "ORDER_DOMINATED_REVERSAL"),
                (-2_999_999, "AMBIGUOUS_ORDER_INTERACTION"),
                (-1_000_000, "AMBIGUOUS_ORDER_INTERACTION"),
                (-999_999, "STRAGGLER_NOT_REPRODUCED"),
                (999_999, "STRAGGLER_NOT_REPRODUCED"),
                (1_000_000, "DIRECTIONAL_ORDER_ROBUST"),
                (2_999_999, "DIRECTIONAL_ORDER_ROBUST"),
                (3_000_000, "ORDER_ROBUST_STRAGGLER"),
                (3_000_001, "ORDER_ROBUST_STRAGGLER"),
            ] {
                let mut p = point(delta, delta);
                // Deliberately lie in display floats; integer thresholds win.
                p.dcrit_percent_of_control = Some(999.0);
                p.dmax_percent_of_control = Some(-999.0);
                assert_eq!(classify(&p, &consistent(&p)).unwrap(), expected);
            }
        }
        #[test]
        fn hma1e_repair_output_aliases_existing_links_and_dangling_links_rejected() {
            let dir =
                std::env::temp_dir().join(format!("hma1e-repair-paths-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let paths = [dir.join("raw"), dir.join("log"), dir.join("original")];
            for (i, path) in paths.iter().enumerate() {
                std::fs::write(path, format!("immutable-{i}")).unwrap();
            }
            let call =
                |out: &std::path::Path| repair_audit_command(&paths[0], &paths[1], &paths[2], out);
            for path in &paths {
                assert!(call(path).is_err());
                assert!(call(&dir.join(".").join(path.file_name().unwrap())).is_err());
                let hard = dir.join("hard");
                std::fs::hard_link(path, &hard).unwrap();
                assert!(call(&hard).is_err());
                std::fs::remove_file(hard).unwrap();
                #[cfg(unix)]
                {
                    let link = dir.join("link");
                    std::os::unix::fs::symlink(path, &link).unwrap();
                    assert!(call(&link).is_err());
                    std::fs::remove_file(link).unwrap();
                }
            }
            let out = dir.join("existing");
            std::fs::write(&out, b"preserve").unwrap();
            assert!(call(&out).is_err());
            assert_eq!(std::fs::read(&out).unwrap(), b"preserve");
            #[cfg(unix)]
            {
                let link = dir.join("dangling");
                std::os::unix::fs::symlink(dir.join("absent"), &link).unwrap();
                assert!(call(&link).is_err());
                assert!(!dir.join("absent").exists());
            }
            for (i, path) in paths.iter().enumerate() {
                assert_eq!(
                    std::fs::read(path).unwrap(),
                    format!("immutable-{i}").as_bytes()
                );
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
        #[test]
        fn hma1e_repair_command_frozen_binding_and_create_new_failure_artifact() {
            let dir =
                std::env::temp_dir().join(format!("hma1e-repair-command-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let (raw, log, original) = evidence();
            for (name, bytes) in [("raw", &raw), ("log", &log), ("original", &original)] {
                std::fs::write(dir.join(name), bytes).unwrap();
            }
            let out = dir.join("new");
            let run = || {
                repair_audit_command(
                    &dir.join("raw"),
                    &dir.join("log"),
                    &dir.join("original"),
                    &out,
                )
            };
            assert!(run().is_err()); // Synthetic data cannot pass production pins.
            let first = std::fs::read(&out).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&first).unwrap();
            assert_eq!(value["schema"], REPAIR_SCHEMA);
            assert_eq!(value["disposition"], "NON_AUTHORITATIVE");
            assert_eq!(value["errors"].as_array().unwrap().len(), 3);
            assert_eq!(value["legacy_reproduction_pass"], false);
            assert!(value["corrected_analysis"].is_null());
            assert!(run().is_err());
            assert_eq!(std::fs::read(&out).unwrap(), first);
            for (name, bytes) in [("raw", &raw), ("log", &log), ("original", &original)] {
                assert_eq!(&std::fs::read(dir.join(name)).unwrap(), bytes);
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
        #[test]
        fn hma1e_repair_cli_parsing_and_pre_startup_offline_source_proof() {
            use clap::Parser;
            let cli = crate::Cli::try_parse_from([
                "mer",
                REPAIR_MODE,
                "--report-in",
                "raw",
                "--completed-run-log",
                "log",
                "--original-audit-in",
                "original",
                "--report-out",
                "new",
            ])
            .unwrap();
            assert!(crate::startup_config_path(&cli.cmd).is_none());
            assert!(matches!(
                cli.cmd,
                crate::Cmd::RepairAuditGpuNativeSourceOrderStragglerProduction { .. }
            ));
            let main = include_str!("main.rs")
                .split("fn main() ->")
                .nth(1)
                .unwrap();
            let early = main.split("let worker_protocol_stdout").next().unwrap();
            assert!(early.contains(
                "return crate::gpu_native_source_order_straggler_production::repair_audit_command("
            ));
            let source = include_str!("gpu_native_source_order_straggler_production.rs");
            let repair = source
                .split("fn repair_bytes(")
                .nth(1)
                .unwrap()
                .split("#[cfg(test)]")
                .next()
                .unwrap();
            assert!(repair.contains("&CONSUMED_FIRST"));
            assert!(repair.contains(".create_new(true)"));
            assert_eq!(repair.matches("std::fs::read(").count(), 3);
            let production = include_str!("gpu_native_source_to_upload_copy_elision_production.rs");
            let validator = production
                .split("pub(crate) fn validate_recorded_authority(")
                .nth(1)
                .unwrap()
                .split("#[cfg(test)]")
                .next()
                .unwrap();
            for region in [early, repair, validator] {
                for forbidden in [
                    "run_command(",
                    "run_order_straggler_command(",
                    "construct_runtime(",
                    "Config::from_file(",
                    "new_multi_thread(",
                    "install_default(",
                    "RealModel::",
                    "from_dir(",
                    "NvmeStorage::",
                    "read_expert(",
                    "execute_request(",
                    "wgpu::",
                    "request_adapter(",
                    "request_device(",
                ] {
                    assert!(
                        !region.contains(forbidden),
                        "offline path contains {forbidden}"
                    );
                }
            }
        }
    }
}
