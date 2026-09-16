//! HMA-1F-B qualification-only registration state. No source I/O lives here.
use crate::gpu_native_mapped_lock::Timing;
use crate::gpu_native_source_upload::{ALIGN, FULL};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::io;

pub(crate) const WIDTH: usize = 8;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Mode {
    MappedBaseline,
    MappedPinned,
}
pub(crate) const ORDER: [Mode; 4] = [
    Mode::MappedBaseline,
    Mode::MappedPinned,
    Mode::MappedPinned,
    Mode::MappedBaseline,
];

pub(crate) fn require_platform() -> Result<(), String> {
    if cfg!(all(target_os = "linux", feature = "io_uring")) {
        Ok(())
    } else {
        Err("HMA-1F-B execution requires Linux + io_uring; portable audits use fixtures".into())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Range {
    pub(crate) pointer: usize,
    pub(crate) length: usize,
    pub(crate) pointer_mod_4096: usize,
}
impl Range {
    pub(crate) fn new(pointer: usize, length: usize) -> Self {
        Self {
            pointer,
            length,
            pointer_mod_4096: pointer % ALIGN,
        }
    }
}
pub(crate) fn pin_bytes(width: usize) -> Result<u64, String> {
    u64::try_from(width)
        .ok()
        .and_then(|k| k.checked_mul(FULL as u64))
        .ok_or_else(|| "pin byte arithmetic overflow".into())
}
fn validate_ranges(ranges: &[Range]) -> Result<(), String> {
    if !(1..=WIDTH).contains(&ranges.len()) {
        return Err("pin source width must be 1..=8".into());
    }
    for (i, r) in ranges.iter().enumerate() {
        if r.pointer == 0 || r.pointer % ALIGN != 0 || r.pointer_mod_4096 != 0 || r.length != FULL {
            return Err("pin destination must be nonzero, 4096-aligned, exact FULL".into());
        }
        let end = r
            .pointer
            .checked_add(r.length)
            .ok_or("pin range overflow")?;
        for a in &ranges[..i] {
            let a_end = a
                .pointer
                .checked_add(a.length)
                .ok_or("pin range overflow")?;
            if a.pointer < end && r.pointer < a_end {
                return Err("overlapping pin destinations".into());
            }
        }
    }
    pin_bytes(ranges.len())?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProcessMemory {
    pub(crate) vmpin_bytes: u64,
    pub(crate) vmlck_bytes: u64,
}
impl ProcessMemory {
    pub(crate) fn parse(status: &str) -> Result<Self, String> {
        fn field(status: &str, name: &str) -> Result<u64, String> {
            let rows: Vec<_> = status.lines().filter(|l| l.starts_with(name)).collect();
            if rows.len() != 1 {
                return Err(format!("missing/duplicate {name}"));
            }
            let parts: Vec<_> = rows[0].split_whitespace().collect();
            if parts.len() != 3 || parts[0] != name || parts[2] != "kB" {
                return Err(format!("invalid {name} units"));
            }
            parts[1]
                .parse::<u64>()
                .ok()
                .and_then(|v| v.checked_mul(1024))
                .ok_or_else(|| format!("invalid/overflow {name}"))
        }
        Ok(Self {
            vmpin_bytes: field(status, "VmPin:")?,
            vmlck_bytes: field(status, "VmLck:")?,
        })
    }
    pub(crate) fn capture() -> Result<Self, String> {
        require_platform()?;
        Self::parse(&std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?)
    }
    pub(crate) fn cleaned(&self, after: &Self) -> Result<(), String> {
        if self == after {
            Ok(())
        } else {
            Err("process VmPin/VmLck did not return exactly to pre".into())
        }
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Operation {
    pub(crate) attempts: u64,
    pub(crate) successes: u64,
    pub(crate) failures: u64,
    pub(crate) raw_os_error: Option<i32>,
    pub(crate) error_kind: Option<String>,
    pub(crate) error_message: Option<String>,
}
impl Operation {
    fn record(&mut self, result: io::Result<()>) -> Result<(), String> {
        self.attempts += 1;
        match result {
            Ok(()) => {
                self.successes += 1;
                Ok(())
            }
            Err(e) => {
                self.failures += 1;
                self.raw_os_error = e.raw_os_error();
                self.error_kind = Some(format!("{:?}", e.kind()));
                self.error_message = Some(e.to_string());
                Err(e.to_string())
            }
        }
    }
    fn exact(&self, n: u64) -> bool {
        self.attempts == n
            && self.successes == n
            && self.failures == 0
            && self.raw_os_error.is_none()
            && self.error_kind.is_none()
            && self.error_message.is_none()
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RingEvidence {
    pub(crate) queue_depth: u32,
    pub(crate) create: Operation,
}
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProcessEvidence {
    pub(crate) pre: Option<ProcessMemory>,
    pub(crate) active: Option<ProcessMemory>,
    pub(crate) after: Option<ProcessMemory>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PinEvidence {
    pub(crate) mode: Mode,
    pub(crate) ranges: Vec<Range>,
    pub(crate) ring: RingEvidence,
    pub(crate) registration: Operation,
    pub(crate) unregister: Operation,
    pub(crate) iovec_count: usize,
    pub(crate) requested_pin_bytes: u64,
    pub(crate) process: ProcessEvidence,
    /// Closed source boundary: this module has no queue access or I/O API.
    pub(crate) sqes_submitted: u64,
    pub(crate) ring_dropped: bool,
}
impl PinEvidence {
    pub(crate) fn new(mode: Mode, ranges: Vec<Range>) -> Self {
        Self {
            mode,
            ranges,
            ring: RingEvidence {
                queue_depth: 2,
                create: Operation::default(),
            },
            registration: Operation::default(),
            unregister: Operation::default(),
            iovec_count: 0,
            requested_pin_bytes: 0,
            process: ProcessEvidence::default(),
            sqes_submitted: 0,
            ring_dropped: false,
        }
    }
    fn validate_active(&self) -> Result<(), String> {
        let pre = self.process.pre.as_ref().ok_or("missing pre status")?;
        let active = self
            .process
            .active
            .as_ref()
            .ok_or("missing active status")?;
        let expected = if self.mode == Mode::MappedPinned {
            pre.vmpin_bytes
                .checked_add(pin_bytes(self.ranges.len())?)
                .ok_or("active VmPin overflow")?
        } else {
            pre.vmpin_bytes
        };
        if active.vmpin_bytes != expected || active.vmlck_bytes != pre.vmlck_bytes {
            return Err("exact active VmPin/VmLck transition failed; source read forbidden".into());
        }
        Ok(())
    }
    pub(crate) fn validate(&self, mode: Mode, width: usize) -> Result<(), String> {
        validate_ranges(&self.ranges)?;
        let n = u64::from(mode == Mode::MappedPinned);
        if self.mode != mode
            || self.ranges.len() != width
            || self.ring.queue_depth != 2
            || !self.ring.create.exact(1)
            || !self.registration.exact(n)
            || !self.unregister.exact(n)
            || self.iovec_count != if n == 1 { width } else { 0 }
            || self.requested_pin_bytes != if n == 1 { pin_bytes(width)? } else { 0 }
            || self.sqes_submitted != 0
            || !self.ring_dropped
        {
            return Err("pin/ring/registration authority failed".into());
        }
        self.validate_active()?;
        self.process
            .pre
            .as_ref()
            .ok_or("missing pre status")?
            .cleaned(self.process.after.as_ref().ok_or("missing after status")?)
    }
}

/// Implementations expose only ring construction, registration, and status.
/// Fixtures implement this boundary without invoking any OS/GPU operation.
pub(crate) trait Registration: Sized {
    fn create() -> io::Result<Self>;
    unsafe fn register(&self, ranges: &[Range]) -> io::Result<()>;
    fn unregister(&self) -> io::Result<()>;
    fn status(&self) -> Result<ProcessMemory, String>;
}
pub(crate) struct SystemRegistration {
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    ring: io_uring::IoUring,
}
impl Registration for SystemRegistration {
    fn create() -> io::Result<Self> {
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            Ok(Self {
                ring: io_uring::IoUring::new(2)?,
            })
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "requires Linux + io_uring",
            ))
        }
    }
    unsafe fn register(&self, ranges: &[Range]) -> io::Result<()> {
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            let iovecs: Vec<libc::iovec> = ranges
                .iter()
                .map(|r| {
                    let pointer = r.pointer as *mut u8;
                    libc::iovec {
                        iov_base: pointer.cast(),
                        iov_len: FULL,
                    }
                })
                .collect();
            let ring = &self.ring;
            // SAFETY: caller keeps the exact validated destination views live
            // until explicit unregister and ring destruction have completed.
            unsafe { ring.submitter().register_buffers(&iovecs) }
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            let _ = ranges;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "requires Linux + io_uring",
            ))
        }
    }
    fn unregister(&self) -> io::Result<()> {
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            let ring = &self.ring;
            ring.submitter().unregister_buffers()
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "requires Linux + io_uring",
            ))
        }
    }
    fn status(&self) -> Result<ProcessMemory, String> {
        ProcessMemory::capture()
    }
}

/// Declared after destination views. Explicit finish precedes their teardown.
pub(crate) struct PinGuard<'a, R: Registration> {
    ring: Option<R>,
    evidence: &'a mut PinEvidence,
    registered: bool,
    finished: bool,
}
impl<'a, R: Registration> PinGuard<'a, R> {
    /// SAFETY: every destination remains live until this guard is destroyed.
    pub(crate) unsafe fn acquire(evidence: &'a mut PinEvidence) -> Result<Self, String> {
        validate_ranges(&evidence.ranges)?;
        let result = R::create();
        evidence
            .ring
            .create
            .record(result.as_ref().map(|_| ()).map_err(|e| {
                e.raw_os_error()
                    .map(io::Error::from_raw_os_error)
                    .unwrap_or_else(|| io::Error::new(e.kind(), e.to_string()))
            }))?;
        let mut guard = Self {
            ring: Some(result.map_err(|e| e.to_string())?),
            evidence,
            registered: false,
            finished: false,
        };
        let setup = (|| -> Result<(), String> {
            let ring = guard.ring.as_ref().unwrap();
            guard.evidence.process.pre = Some(ring.status()?);
            if guard.evidence.mode == Mode::MappedPinned {
                guard.evidence.iovec_count = guard.evidence.ranges.len();
                guard.evidence.requested_pin_bytes = pin_bytes(guard.evidence.ranges.len())?;
                guard
                    .evidence
                    .registration
                    .record(unsafe { ring.register(&guard.evidence.ranges) })?;
                guard.registered = true;
            }
            guard.evidence.process.active = Some(ring.status()?);
            guard.evidence.validate_active()
        })();
        if let Err(error) = setup {
            let cleanup = guard.finish();
            return Err(match cleanup {
                Ok(()) => error,
                Err(e) => format!("{error}; cleanup: {e}"),
            });
        }
        Ok(guard)
    }
    pub(crate) fn finish(&mut self) -> Result<(), String> {
        if self.finished {
            return Err("pin cleanup already attempted".into());
        }
        self.finished = true;
        let ring = self.ring.as_ref().ok_or("missing pin ring")?;
        // Exactly one explicit attempt, even if source I/O or observation failed.
        let unregister = if self.registered {
            self.registered = false;
            self.evidence.unregister.record(ring.unregister())
        } else {
            Ok(())
        };
        let after = ring.status();
        let observation = match after {
            Ok(after) => {
                self.evidence.process.after = Some(after);
                Ok(())
            }
            Err(e) => Err(e),
        };
        let cleanup = observation.and_then(|()| {
            self.evidence
                .process
                .pre
                .as_ref()
                .ok_or("missing pre status")?
                .cleaned(self.evidence.process.after.as_ref().unwrap())
        });
        drop(self.ring.take());
        self.evidence.ring_dropped = true;
        unregister.and(cleanup)
    }
}
impl<R: Registration> Drop for PinGuard<'_, R> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish();
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    pub(crate) mode: Mode,
    pub(crate) measured: bool,
    pub(crate) request_index: usize,
    pub(crate) source_index: usize,
    pub(crate) ids: Vec<u32>,
    pub(crate) width: usize,
    pub(crate) bytes: usize,
    pub(crate) caller_ns: u64,
    pub(crate) timing: Option<Timing>,
    pub(crate) pin: PinEvidence,
    pub(crate) failure: Option<String>,
}
#[derive(Default)]
struct Store {
    measured: bool,
    request_index: usize,
    source_index: usize,
    failed: bool,
    records: Vec<Record>,
}
pub(crate) struct Observer {
    pub(crate) mode: Mode,
    store: Mutex<Store>,
}
impl Observer {
    pub(crate) fn new(mode: Mode) -> Self {
        Self {
            mode,
            store: Mutex::new(Store::default()),
        }
    }
    pub(crate) fn begin_request(&self, measured: bool, request_index: usize) {
        let mut s = self.store.lock();
        s.measured = measured;
        s.request_index = request_index;
        s.source_index = 0;
    }
    pub(crate) fn record(
        &self,
        ids: &[u32],
        bytes: usize,
        caller_ns: u64,
        timing: Option<Timing>,
        pin: PinEvidence,
        failure: Option<String>,
    ) {
        let mut s = self.store.lock();
        let r = Record {
            mode: self.mode,
            measured: s.measured,
            request_index: s.request_index,
            source_index: s.source_index,
            ids: ids.to_vec(),
            width: ids.len(),
            bytes,
            caller_ns,
            timing,
            pin,
            failure,
        };
        s.failed |= r.failure.is_some();
        s.source_index += 1;
        s.records.push(r);
    }
    pub(crate) fn failed(&self) -> bool {
        self.store.lock().failed
    }
    pub(crate) fn records(&self) -> Vec<Record> {
        self.store.lock().records.clone()
    }
}

#[cfg(test)]
mod hma1fb_tests {
    use super::*;
    use std::cell::RefCell;
    #[derive(Default)]
    struct Fixture {
        calls: Vec<&'static str>,
        ranges: Vec<Range>,
        registered: bool,
        create_error: bool,
        register_error: bool,
        unregister_error: bool,
        status_error_at: Option<usize>,
        statuses: usize,
        bad_active: bool,
        bad_after: bool,
        bad_lock: bool,
    }
    thread_local! { static FIXTURE: RefCell<Fixture> = RefCell::new(Fixture::default()); }
    struct Mock;
    impl Registration for Mock {
        fn create() -> io::Result<Self> {
            FIXTURE.with(|f| {
                let mut f = f.borrow_mut();
                f.calls.push("create");
                if f.create_error {
                    Err(io::Error::from_raw_os_error(libc::EPERM))
                } else {
                    Ok(Self)
                }
            })
        }
        unsafe fn register(&self, ranges: &[Range]) -> io::Result<()> {
            FIXTURE.with(|f| {
                let mut f = f.borrow_mut();
                f.calls.push("register");
                f.ranges = ranges.to_vec();
                if f.register_error {
                    Err(io::Error::from_raw_os_error(libc::ENOMEM))
                } else {
                    f.registered = true;
                    Ok(())
                }
            })
        }
        fn unregister(&self) -> io::Result<()> {
            FIXTURE.with(|f| {
                let mut f = f.borrow_mut();
                f.calls.push("unregister");
                if f.unregister_error {
                    Err(io::Error::from_raw_os_error(libc::EIO))
                } else {
                    f.registered = false;
                    Ok(())
                }
            })
        }
        fn status(&self) -> Result<ProcessMemory, String> {
            FIXTURE.with(|f| {
                let mut f = f.borrow_mut();
                f.calls.push("status");
                let index = f.statuses;
                f.statuses += 1;
                if f.status_error_at == Some(index) {
                    return Err("injected proc capture/parse failure".into());
                }
                Ok(ProcessMemory {
                    vmpin_bytes: 4096
                        + if f.registered {
                            pin_bytes(f.ranges.len()).unwrap()
                        } else {
                            0
                        }
                        + u64::from((f.bad_active && index == 1) || (f.bad_after && index == 2)),
                    vmlck_bytes: 8192 + u64::from(f.bad_lock && index == 1),
                })
            })
        }
    }
    impl Drop for Mock {
        fn drop(&mut self) {
            FIXTURE.with(|f| {
                let mut f = f.borrow_mut();
                f.calls.push("drop");
                f.registered = false;
            });
        }
    }
    fn reset(f: Fixture) {
        FIXTURE.with(|state| *state.borrow_mut() = f);
    }
    fn evidence(mode: Mode, width: usize) -> PinEvidence {
        PinEvidence::new(
            mode,
            (0..width)
                .map(|i| Range::new(ALIGN + i * FULL, FULL))
                .collect(),
        )
    }
    fn success(mode: Mode, width: usize) -> PinEvidence {
        reset(Fixture::default());
        let mut e = evidence(mode, width);
        {
            let mut g = unsafe { PinGuard::<Mock>::acquire(&mut e) }.unwrap();
            g.finish().unwrap();
        }
        e.validate(mode, width).unwrap();
        e
    }
    #[test]
    fn hma1fb_width_1_2_8_exact_k_ranges_and_pin_bytes() {
        for width in [1, 2, 8] {
            let e = success(Mode::MappedPinned, width);
            assert_eq!(e.requested_pin_bytes, (width * FULL) as u64);
            assert_eq!(e.iovec_count, width);
            FIXTURE.with(|f| {
                let f = f.borrow();
                assert_eq!(f.ranges, e.ranges);
                assert_eq!(
                    f.calls,
                    [
                        "create",
                        "status",
                        "register",
                        "status",
                        "unregister",
                        "status",
                        "drop"
                    ]
                );
            });
        }
    }
    #[test]
    fn hma1fb_baseline_same_ring_no_registration_or_unregistration() {
        for width in [1, 2, 8] {
            let e = success(Mode::MappedBaseline, width);
            assert_eq!(e.registration, Operation::default());
            assert_eq!(e.unregister, Operation::default());
            assert_eq!(e.process.pre, e.process.active);
            assert_eq!(e.process.pre, e.process.after);
            FIXTURE.with(|f| {
                assert_eq!(
                    f.borrow().calls,
                    ["create", "status", "status", "status", "drop"]
                )
            });
        }
    }
    #[test]
    fn hma1fb_range_geometry_and_overlap_reject_before_ring() {
        for ranges in [
            vec![],
            vec![Range::new(4096, FULL); 9],
            vec![Range::new(0, FULL)],
            vec![Range::new(4097, FULL)],
            vec![Range::new(4096, FULL - 4096)],
            vec![Range::new(4096, FULL + 4096)],
            vec![Range::new(4096, FULL), Range::new(8192, FULL)],
            vec![Range::new(usize::MAX - (ALIGN - 1), FULL)],
        ] {
            reset(Fixture::default());
            let mut e = PinEvidence::new(Mode::MappedPinned, ranges);
            assert!(unsafe { PinGuard::<Mock>::acquire(&mut e) }.is_err());
            FIXTURE.with(|f| assert!(f.borrow().calls.is_empty()));
        }
        let mut e = evidence(Mode::MappedPinned, 1);
        e.ranges[0].pointer_mod_4096 = 1;
        assert!(validate_ranges(&e.ranges).is_err());
    }
    #[test]
    fn hma1fb_checked_multiplication_and_active_addition_overflow() {
        assert!(pin_bytes(usize::MAX).is_err());
        let mut e = success(Mode::MappedPinned, 8);
        e.process.pre.as_mut().unwrap().vmpin_bytes = u64::MAX;
        assert!(e.validate_active().unwrap_err().contains("overflow"));
    }
    #[test]
    fn hma1fb_exact_nonzero_pre_transition_no_tolerance() {
        let e = success(Mode::MappedPinned, 2);
        assert_eq!(e.process.pre.as_ref().unwrap().vmpin_bytes, 4096);
        assert_eq!(
            e.process.active.as_ref().unwrap().vmpin_bytes,
            4096 + 2 * FULL as u64
        );
        for phase in [0, 1, 2] {
            let mut bad = e.clone();
            let snapshot = match phase {
                0 => bad.process.pre.as_mut(),
                1 => bad.process.active.as_mut(),
                _ => bad.process.after.as_mut(),
            }
            .unwrap();
            snapshot.vmpin_bytes += 1;
            assert!(bad.validate(Mode::MappedPinned, 2).is_err());
        }
    }
    #[test]
    fn hma1fb_vmlck_invariant_every_phase() {
        for mode in ORDER {
            let e = success(mode, 2);
            for phase in [0, 1, 2] {
                let mut bad = e.clone();
                let snapshot = match phase {
                    0 => bad.process.pre.as_mut(),
                    1 => bad.process.active.as_mut(),
                    _ => bad.process.after.as_mut(),
                }
                .unwrap();
                snapshot.vmlck_bytes += 1;
                assert!(bad.validate(mode, 2).is_err());
            }
        }
    }
    #[test]
    fn hma1fb_ring_creation_error_retains_errno_and_prevents_registration() {
        reset(Fixture {
            create_error: true,
            ..Default::default()
        });
        let mut e = evidence(Mode::MappedPinned, 2);
        assert!(unsafe { PinGuard::<Mock>::acquire(&mut e) }.is_err());
        assert_eq!(e.ring.create.attempts, 1);
        assert_eq!(e.ring.create.failures, 1);
        assert_eq!(e.ring.create.raw_os_error, Some(libc::EPERM));
        FIXTURE.with(|f| assert_eq!(f.borrow().calls, ["create"]));
    }
    #[test]
    fn hma1fb_registration_error_has_no_unregister_and_drops_ring() {
        reset(Fixture {
            register_error: true,
            ..Default::default()
        });
        let mut e = evidence(Mode::MappedPinned, 2);
        assert!(unsafe { PinGuard::<Mock>::acquire(&mut e) }.is_err());
        assert_eq!(e.registration.attempts, 1);
        assert_eq!(e.registration.failures, 1);
        assert_eq!(e.registration.raw_os_error, Some(libc::ENOMEM));
        assert!(e.registration.error_kind.is_some());
        assert!(e.registration.error_message.is_some());
        assert_eq!(e.unregister.attempts, 0);
        assert!(e.ring_dropped);
        FIXTURE.with(|f| {
            assert_eq!(
                f.borrow().calls,
                ["create", "status", "register", "status", "drop"]
            )
        });
    }
    #[test]
    fn hma1fb_unregister_error_is_non_authoritative_and_ring_closes() {
        reset(Fixture {
            unregister_error: true,
            ..Default::default()
        });
        let mut e = evidence(Mode::MappedPinned, 2);
        {
            let mut g = unsafe { PinGuard::<Mock>::acquire(&mut e) }.unwrap();
            assert!(g.finish().is_err());
        }
        assert_eq!(e.unregister.attempts, 1);
        assert_eq!(e.unregister.failures, 1);
        assert_eq!(e.unregister.raw_os_error, Some(libc::EIO));
        assert!(e.validate(Mode::MappedPinned, 2).is_err());
        assert!(e.ring_dropped);
        FIXTURE.with(|f| {
            let f = f.borrow();
            assert!(!f.registered);
            assert_eq!(f.calls.last(), Some(&"drop"));
        });
    }
    #[test]
    fn hma1fb_proc_error_each_phase_still_cleans_up() {
        for phase in [0, 1, 2] {
            reset(Fixture {
                status_error_at: Some(phase),
                ..Default::default()
            });
            let mut e = evidence(Mode::MappedPinned, 2);
            if phase < 2 {
                assert!(unsafe { PinGuard::<Mock>::acquire(&mut e) }.is_err());
            } else {
                let mut g = unsafe { PinGuard::<Mock>::acquire(&mut e) }.unwrap();
                assert!(g.finish().is_err());
            }
            assert_eq!(e.unregister.successes, u64::from(phase > 0));
            assert!(e.ring_dropped);
            assert!(e.validate(Mode::MappedPinned, 2).is_err());
        }
    }
    #[test]
    fn hma1fb_bad_active_state_prevents_read_and_unregisters() {
        for mode in ORDER {
            for bad_lock in [false, true] {
                reset(Fixture {
                    bad_active: !bad_lock,
                    bad_lock,
                    ..Default::default()
                });
                let mut e = evidence(mode, 2);
                assert!(unsafe { PinGuard::<Mock>::acquire(&mut e) }.is_err());
                assert_eq!(
                    e.unregister.successes,
                    u64::from(mode == Mode::MappedPinned)
                );
                assert!(e.ring_dropped);
            }
        }
    }
    #[test]
    fn hma1fb_after_cleanup_mismatch_is_failure() {
        reset(Fixture {
            bad_after: true,
            ..Default::default()
        });
        let mut e = evidence(Mode::MappedPinned, 2);
        {
            let mut g = unsafe { PinGuard::<Mock>::acquire(&mut e) }.unwrap();
            assert!(g.finish().is_err());
        }
        assert_eq!(e.unregister.successes, 1);
        assert!(e.validate(Mode::MappedPinned, 2).is_err());
    }
    #[test]
    fn hma1fb_read_failure_still_explicitly_unregisters_before_propagation() {
        reset(Fixture::default());
        let mut e = evidence(Mode::MappedPinned, 2);
        let result = (|| -> Result<(), String> {
            let mut g = unsafe { PinGuard::<Mock>::acquire(&mut e) }?;
            let read: Result<(), String> = Err("injected source failure".into());
            let cleanup = g.finish();
            drop(g);
            cleanup?;
            read
        })();
        assert!(result.is_err());
        assert_eq!(e.unregister.attempts, 1);
        assert_eq!(e.unregister.successes, 1);
        assert!(e.ring_dropped);
    }
    #[test]
    fn hma1fb_unwind_fallback_unregisters_before_view_scope_ends() {
        reset(Fixture::default());
        let mut e = evidence(Mode::MappedPinned, 2);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = unsafe { PinGuard::<Mock>::acquire(&mut e) }.unwrap();
            panic!("fixture unwind");
        }));
        assert!(panic.is_err());
        assert_eq!(e.unregister.successes, 1);
        assert!(e.ring_dropped);
    }
    #[test]
    fn hma1fb_status_parser_strict_missing_duplicate_units_values_overflow() {
        assert_eq!(
            ProcessMemory::parse("VmPin:\t4 kB\nVmLck: 8 kB\n").unwrap(),
            ProcessMemory {
                vmpin_bytes: 4096,
                vmlck_bytes: 8192
            }
        );
        for text in [
            "",
            "VmPin: 0 kB",
            "VmPin: 0 kB\nVmLck: 0 kB\nVmPin: 0 kB",
            "VmPin: 0 B\nVmLck: 0 kB",
            "VmPin: -1 kB\nVmLck: 0 kB",
            "VmPin: 18446744073709551615 kB\nVmLck: 0 kB",
            "VmPin: 0 kB extra\nVmLck: 0 kB",
            "VmPin: 0 kB\nVmLck: bad kB",
        ] {
            assert!(ProcessMemory::parse(text).is_err());
        }
    }
    #[test]
    fn hma1fb_ring_evidence_mutations_cannot_authorize() {
        let original = success(Mode::MappedPinned, 2);
        for i in 0..8 {
            let mut e = original.clone();
            match i {
                0 => e.sqes_submitted = 1,
                1 => e.iovec_count = 1,
                2 => e.requested_pin_bytes -= 1,
                3 => e.ring.queue_depth = 4,
                4 => e.ring.create.attempts = 2,
                5 => e.registration.successes = 0,
                6 => e.unregister.attempts = 2,
                _ => e.ring_dropped = false,
            }
            assert!(e.validate(Mode::MappedPinned, 2).is_err());
        }
    }
    #[test]
    fn hma1fb_observer_preserves_request_identity_and_failed_closed_state() {
        let o = Observer::new(Mode::MappedPinned);
        o.begin_request(true, 2);
        o.record(
            &[9, 7],
            0,
            0,
            None,
            evidence(o.mode, 2),
            Some("read failed".into()),
        );
        o.begin_request(false, 0);
        assert!(o.failed());
        let r = &o.records()[0];
        assert_eq!(r.ids, [9, 7]);
        assert_eq!(r.width, 2);
        assert!(r.measured);
        assert_eq!(r.request_index, 2);
        assert_eq!(r.source_index, 0);
    }
}
