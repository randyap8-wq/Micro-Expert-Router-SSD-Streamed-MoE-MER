//! HMA-1E qualification-only observations of the existing production stream.
//! Timestamp capture never owns storage, a GPU, a lock, or a source scheduler.
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
    let mut result = non_authoritative();
    if let Err(e) = crate::gpu_native_physical_install_staging::source_to_upload_production::validate_recorded_authority(production) {
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
    let source_report_sha256 = format!("{:x}", Sha256::digest(bytes));
    let mut a = non_authoritative();
    match serde_json::from_slice::<Envelope>(bytes) {
        Err(e) => a.errors.push(format!("raw report structure: {e}")),
        Ok(raw) => {
            a = reconstruct(&raw.observations, &raw.production_v2);
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
}
