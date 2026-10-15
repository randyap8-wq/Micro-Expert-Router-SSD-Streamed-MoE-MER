//! Request-local, bounded host observations for PORT-PERF-2 C1.
//!
//! Enabled only by MER_RECOVERY_PHASE_TELEMETRY=1 when a request is created.
//! OFF allocates no recorder, takes no diagnostic lock and reads no clock.
//! Tokio task scope follows polling; scoped synchronous workers explicitly
//! inherit the recorder. Spawned background tasks do not inherit it.
//! Durations are inclusive, potentially overlapping host scopes. Worker sums
//! must never be added to critical-path wall time. No device timestamps or
//! NAND/controller latency are measured. Historical v4 receipts lack this data.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SETTING: &str = "MER_RECOVERY_PHASE_TELEMETRY";
macro_rules! keys {
    ($name:ident { $($key:ident),+ $(,)? }) => {
        #[derive(Clone, Copy)]
        pub(crate) enum $name { $($key,)+ Count }
        impl $name {
            const NAMES: &'static [&'static str] = &[$(stringify!($key),)+];
        }
    };
}
keys!(Counter {
    ServicesStarted,
    ServicesCompleted,
    ServicesFailed,
    EligibilityChecks,
    EligibleServices,
    FallbackPermitBusy,
    FallbackUnsupportedConfiguration,
    FallbackServiceCache,
    FallbackRam,
    FallbackNonsharedLogical,
    FallbackOther,
    MappingUnavailableFailures,
    FusedAttempts,
    FusedCompletions,
    FusedFailures,
    FusedStageAttempts,
    FusedStageCompletions,
    FusedStageFailures,
    ReadExpertCalls,
    ReadBatchCalls,
    ReadMappedBatchCalls,
    ReadSyscalls,
    ReadRequestedBytes,
    ReadReturnedBytes,
    ReadRetryCalls,
    ReadShortCalls,
    ReadErrors,
    DirectFdProofs,
    MapAttempts,
    MapCompletions,
    RemapAttempts,
    RemapCompletions,
    ArcMaterializations,
    ArcMaterializationBytes,
    EqualityComparisons,
    EqualityInputBytes,
    CopyBatches,
    CopyBytes,
    UploadCopySubmissions,
    ConventionalStageAttempts,
    ConventionalStageCompletions,
    ConventionalStageFailures,
    WriteBufferWithCompleted,
    CpuPayloadCopyBytes,
    PhysicalCommits,
    TokenSubmissions,
    FreshSegments,
    ResumeExpertSegments,
    ContinueSegments
});
keys!(Phase {
    FusionEligibilityHost,
    FusedStageHost,
    ReadVectoredSyscallHostWorkerSum,
    DemandServiceWall,
    ServiceMapRamSourceWall,
    SingleflightWaitWall,
    PoolWaitWall,
    FusedSourceWall,
    BatchDispatchJoinWall,
    ReadAtSyscallHostWorkerSum,
    UploadFirstMapWall,
    UploadRemapWall,
    UploadMapApiHost,
    UploadPollWaitHost,
    UploadChannelWaitHost,
    PayloadEqualityHost,
    ArcMaterializationHost,
    CpuValidationHostWorkerSum,
    CpuPayloadCopyHostWorkerSum,
    StagingWriteHostWorkerSum,
    CopyEncodeHostWorkerSum,
    UploadCopySubmitHost,
    PhysicalReservationWall,
    PhysicalStageHostWorkerSum,
    PhysicalStageJoinWall,
    ConventionalStageHost,
    PhysicalCommitHost,
    TokenSubmitHost
});

pub(crate) struct Collector {
    counters: [AtomicU64; Counter::Count as usize],
    samples: [AtomicU64; Phase::Count as usize],
    nanos: [AtomicU64; Phase::Count as usize],
    invalid: AtomicBool,
}
impl Collector {
    pub(crate) fn from_environment() -> Option<Arc<Self>> {
        Self::configured(std::env::var_os(SETTING).as_deref() == Some(std::ffi::OsStr::new("1")))
    }
    pub(crate) fn configured(enabled: bool) -> Option<Arc<Self>> {
        enabled.then(|| {
            Arc::new(Self {
                counters: std::array::from_fn(|_| AtomicU64::new(0)),
                samples: std::array::from_fn(|_| AtomicU64::new(0)),
                nanos: std::array::from_fn(|_| AtomicU64::new(0)),
                invalid: AtomicBool::new(false),
            })
        })
    }
    fn accumulate(&self, value: &AtomicU64, n: u64) {
        if value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(n))
            .is_err()
        {
            self.invalid.store(true, Ordering::Relaxed);
        }
    }
    fn add(&self, counter: Counter, n: u64) {
        self.accumulate(&self.counters[counter as usize], n);
    }
    fn observe(&self, phase: Phase, nanos: u64) {
        self.accumulate(&self.samples[phase as usize], 1);
        self.accumulate(&self.nanos[phase as usize], nanos);
    }
    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            counters: self
                .counters
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect(),
            samples: self
                .samples
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect(),
            nanos: self
                .nanos
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect(),
            invalid: self.invalid.load(Ordering::Relaxed),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Snapshot {
    counters: Vec<u64>,
    samples: Vec<u64>,
    nanos: Vec<u64>,
    invalid: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct HostTiming {
    samples: u64,
    /// None means no sample, never a measured zero.
    inclusive_host_ns: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Envelope {
    schema: String,
    enabled: bool,
    setting: String,
    status: String,
    scope: String,
    timing_semantics: String,
    device_copy_completion_ns: Option<u64>,
    ssd_device_latency_ns: Option<u64>,
    before: Option<Snapshot>,
    after: Option<Snapshot>,
    counters_delta: Option<BTreeMap<String, u64>>,
    host_timings_delta: Option<BTreeMap<String, HostTiming>>,
}
impl Envelope {
    pub(crate) fn between(before: Option<Snapshot>, after: Option<Snapshot>) -> Option<Self> {
        if before.is_none() && after.is_none() {
            return None;
        }
        let delta = before.as_ref().zip(after.as_ref()).and_then(|(b, a)| {
            if b.invalid || a.invalid {
                return None;
            }
            fn delta(b: &[u64], a: &[u64], len: usize) -> Option<Vec<u64>> {
                if b.len() != len || a.len() != len {
                    return None;
                }
                a.iter().zip(b).map(|(a, b)| a.checked_sub(*b)).collect()
            }
            Some((
                delta(&b.counters, &a.counters, Counter::Count as usize)?,
                delta(&b.samples, &a.samples, Phase::Count as usize)?,
                delta(&b.nanos, &a.nanos, Phase::Count as usize)?,
            ))
        });
        let status = if delta.is_some() {
            "measured_request_delta"
        } else {
            "unavailable_or_invalid_delta"
        };
        let counters_delta = delta.as_ref().map(|(c, _, _)| {
            Counter::NAMES
                .iter()
                .zip(c)
                .map(|(k, v)| ((*k).to_owned(), *v))
                .collect()
        });
        let host_timings_delta = delta.map(|(_, s, n)| {
            Phase::NAMES
                .iter()
                .zip(s)
                .zip(n)
                .map(|((k, s), n)| {
                    (
                        (*k).to_owned(),
                        HostTiming {
                            samples: s,
                            inclusive_host_ns: (s > 0).then_some(n),
                        },
                    )
                })
                .collect()
        });
        Some(Self {
            schema:"mer.recovery-phase-telemetry.v1".into(),enabled:true,
            setting:format!("{SETTING}=1"),status:status.into(),
            scope:"request demand; explicitly inherited workers; excludes background tasks".into(),
            timing_semantics:"inclusive host durations; nested scopes overlap; worker sums are not wall time; syscall issue-to-return includes retries separately, excludes retry backoff; queue submit is not device completion; equality bytes are input lengths, early mismatch may scan fewer".into(),
            device_copy_completion_ns:None,ssd_device_latency_ns:None,
            before,after,counters_delta,host_timings_delta,
        })
    }
}
tokio::task_local! { static TASK: Arc<Collector>; }
thread_local! { static WORKER: RefCell<Option<Arc<Collector>>> = const { RefCell::new(None) }; }
pub(crate) fn current() -> Option<Arc<Collector>> {
    WORKER
        .with(|v| v.borrow().clone())
        .or_else(|| TASK.try_with(Arc::clone).ok())
}
pub(crate) async fn scope<F: Future>(collector: Option<Arc<Collector>>, operation: F) -> F::Output {
    match collector {
        Some(c) => TASK.scope(c, operation).await,
        None => operation.await,
    }
}
pub(crate) fn with_sync<T>(collector: Option<Arc<Collector>>, operation: impl FnOnce() -> T) -> T {
    let Some(c) = collector else {
        return operation();
    };
    struct Restore(Option<Arc<Collector>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            WORKER.with(|v| *v.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(WORKER.with(|v| v.replace(Some(c))));
    operation()
}
pub(crate) fn add(counter: Counter, n: u64) {
    if let Some(c) = current() {
        c.add(counter, n);
    }
}
pub(crate) struct Span {
    collector: Option<Arc<Collector>>,
    phase: Phase,
    start: Option<Instant>,
}
pub(crate) fn span(phase: Phase) -> Span {
    let collector = current();
    let start = collector.as_ref().map(|_| Instant::now());
    Span {
        collector,
        phase,
        start,
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let (Some(c), Some(start)) = (&self.collector, self.start) {
            match u64::try_from(start.elapsed().as_nanos()) {
                Ok(n) => c.observe(self.phase, n),
                Err(_) => c.invalid.store(true, Ordering::Relaxed),
            }
        }
    }
}
pub(crate) fn observed_microseconds(phase: Phase, micros: u64) {
    if let Some(c) = current() {
        match micros.checked_mul(1000) {
            Some(n) => c.observe(phase, n),
            None => c.invalid.store(true, Ordering::Relaxed),
        }
    }
}
pub(crate) struct Outcome {
    span: Span,
    success: Counter,
    failure: Counter,
    done: bool,
}
pub(crate) fn outcome(
    phase: Phase,
    attempt: Counter,
    success: Counter,
    failure: Counter,
) -> Outcome {
    let span = span(phase);
    if let Some(c) = &span.collector {
        c.add(attempt, 1);
    }
    Outcome {
        span,
        success,
        failure,
        done: false,
    }
}
impl Outcome {
    pub(crate) fn succeed(&mut self) {
        self.done = true;
    }
}
impl Drop for Outcome {
    fn drop(&mut self) {
        if let Some(c) = &self.span.collector {
            c.add(
                if self.done {
                    self.success
                } else {
                    self.failure
                },
                1,
            );
        }
    }
}
pub(crate) fn record_read(requested: usize, attempt: u32, result: &std::io::Result<usize>) {
    if let Some(c) = current() {
        c.add(Counter::ReadSyscalls, 1);
        c.add(Counter::ReadRequestedBytes, requested as u64);
        if attempt > 0 {
            c.add(Counter::ReadRetryCalls, 1);
        }
        match result {
            Ok(n) => {
                c.add(Counter::ReadReturnedBytes, *n as u64);
                if *n < requested {
                    c.add(Counter::ReadShortCalls, 1);
                }
            }
            Err(_) => c.add(Counter::ReadErrors, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovery_telemetry_default_off_has_no_recorder_clock_or_envelope() {
        assert!(Collector::configured(false).is_none());
        let timer = span(Phase::DemandServiceWall);
        assert!(timer.collector.is_none() && timer.start.is_none());
        assert!(Envelope::between(None, None).is_none());
    }
    #[test]
    fn recovery_telemetry_short_read_retry_errors_and_nested_math_are_distinct() {
        let c = Collector::configured(true).unwrap();
        let before = c.snapshot();
        with_sync(Some(c.clone()), || {
            record_read(
                10,
                0,
                &Err(std::io::Error::from(std::io::ErrorKind::Interrupted)),
            );
            record_read(10, 1, &Ok(4));
            c.observe(Phase::DemandServiceWall, 20);
            c.observe(Phase::ReadAtSyscallHostWorkerSum, 30);
            c.observe(Phase::ReadAtSyscallHostWorkerSum, 40);
            add(Counter::TokenSubmissions, 2);
            add(Counter::UploadCopySubmissions, 1);
        });
        let e = Envelope::between(Some(before), Some(c.snapshot())).unwrap();
        let d = e.counters_delta.unwrap();
        assert_eq!(d["ReadRequestedBytes"], 20);
        assert_eq!(d["ReadReturnedBytes"], 4);
        assert_eq!(d["ReadRetryCalls"], 1);
        assert_eq!(d["ReadShortCalls"], 1);
        assert_eq!(d["ReadErrors"], 1);
        assert_eq!(d["TokenSubmissions"], 2);
        assert_eq!(d["UploadCopySubmissions"], 1);
        let t = e.host_timings_delta.unwrap();
        assert_eq!(t["DemandServiceWall"].inclusive_host_ns, Some(20));
        assert_eq!(t["ReadAtSyscallHostWorkerSum"].inclusive_host_ns, Some(70));
        assert_eq!(t["UploadPollWaitHost"].inclusive_host_ns, None);
    }
    #[test]
    fn recovery_telemetry_missing_overflow_and_reset_never_become_zero() {
        let c = Collector::configured(true).unwrap();
        let before = c.snapshot();
        c.counters[0].store(u64::MAX, Ordering::Relaxed);
        c.add(Counter::ServicesStarted, 1);
        let e = Envelope::between(Some(before.clone()), Some(c.snapshot())).unwrap();
        assert!(e.counters_delta.is_none());
        let e = Envelope::between(Some(c.snapshot()), Some(before.clone())).unwrap();
        assert!(e.counters_delta.is_none());
        assert!(Envelope::between(Some(before), None)
            .unwrap()
            .host_timings_delta
            .is_none());
    }
    #[test]
    fn recovery_telemetry_fused_outcomes_and_fallback_reasons_are_bounded() {
        let c = Collector::configured(true).unwrap();
        let before = c.snapshot();
        with_sync(Some(c.clone()), || {
            {
                let mut done = outcome(
                    Phase::FusedSourceWall,
                    Counter::FusedAttempts,
                    Counter::FusedCompletions,
                    Counter::FusedFailures,
                );
                done.succeed();
            }
            {
                let _failed = outcome(
                    Phase::FusedSourceWall,
                    Counter::FusedAttempts,
                    Counter::FusedCompletions,
                    Counter::FusedFailures,
                );
            }
            add(Counter::FallbackPermitBusy, 1);
            add(Counter::FallbackNonsharedLogical, 2);
        });
        let e = Envelope::between(Some(before), Some(c.snapshot())).unwrap();
        let d = e.counters_delta.as_ref().unwrap();
        assert_eq!(d["FusedAttempts"], 2);
        assert_eq!(d["FusedCompletions"], 1);
        assert_eq!(d["FusedFailures"], 1);
        assert_eq!(d["FallbackPermitBusy"], 1);
        assert_eq!(d["FallbackNonsharedLogical"], 2);
        let json = serde_json::to_value(&e).unwrap();
        assert!(json["device_copy_completion_ns"].is_null());
        assert!(json["ssd_device_latency_ns"].is_null());
        assert_eq!(serde_json::from_value::<Envelope>(json).unwrap(), e);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_telemetry_task_and_worker_isolation_do_not_inherit_background() {
        let a = Collector::configured(true).unwrap();
        let b = Collector::configured(true).unwrap();
        let operation = |c: Arc<Collector>| async move {
            scope(Some(c), async {
                add(Counter::TokenSubmissions, 1);
                tokio::task::yield_now().await;
                assert!(tokio::spawn(async { current().is_none() }).await.unwrap());
                let context = current();
                std::thread::scope(|s| {
                    s.spawn(move || with_sync(context, || add(Counter::UploadCopySubmissions, 1)))
                        .join()
                        .unwrap();
                });
            })
            .await;
        };
        tokio::join!(operation(a.clone()), operation(b.clone()));
        for c in [a, b] {
            assert_eq!(c.snapshot().counters[Counter::TokenSubmissions as usize], 1);
            assert_eq!(
                c.snapshot().counters[Counter::UploadCopySubmissions as usize],
                1
            );
        }
        assert!(current().is_none());
    }
}
