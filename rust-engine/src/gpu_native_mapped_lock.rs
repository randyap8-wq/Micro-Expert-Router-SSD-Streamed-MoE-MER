//! HMA-1F-only mapped destination residency intervention and boundary observations.
//! Ordinary constructors never install an Observer. Workers only receive RawRead slots.
use crate::gpu_native_source_upload::{ALIGN, FULL};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

pub(crate) const WIDTH: usize = 8;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Mode {
    MappedBaseline,
    MappedLocked,
}
pub(crate) const ORDER: [Mode; 4] = [
    Mode::MappedBaseline,
    Mode::MappedLocked,
    Mode::MappedLocked,
    Mode::MappedBaseline,
];

/// Unique, preallocated mutable slot. No attempt timing or retry state.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawRead {
    pub(crate) start: Option<Instant>,
    pub(crate) end: Option<Instant>,
    pub(crate) success: bool,
}
#[derive(Clone, Debug, Default)]
pub(crate) struct RawBatch {
    pub(crate) reads: [RawRead; WIDTH],
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Read {
    pub(crate) wrapper_start_ns: u64,
    pub(crate) wrapper_end_ns: u64,
    pub(crate) wrapper_wall_ns: u64,
    pub(crate) success: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Timing {
    pub(crate) reads: Vec<Read>,
    pub(crate) read_critical_span_ns: u64,
    pub(crate) batch_max_read_wall_ns: u64,
    pub(crate) batch_sum_read_wall_ns: u64,
}
fn ns(start: Instant, end: Instant) -> Result<u64, String> {
    u64::try_from(
        end.checked_duration_since(start)
            .ok_or("timestamp order")?
            .as_nanos(),
    )
    .map_err(|e| e.to_string())
}
impl RawBatch {
    /// Called only after the source helper returns and all workers join.
    pub(crate) fn reconstruct(
        &self,
        width: usize,
        start: Instant,
        end: Instant,
    ) -> Result<Timing, String> {
        if !(1..=WIDTH).contains(&width) {
            return Err("source width".into());
        }
        let caller = ns(start, end)?;
        let reads = self.reads[..width]
            .iter()
            .map(|r| {
                let a = r.start.ok_or("missing wrapper start")?;
                let b = r.end.ok_or("missing wrapper end")?;
                let read = Read {
                    wrapper_start_ns: ns(start, a)?,
                    wrapper_end_ns: ns(start, b)?,
                    wrapper_wall_ns: ns(a, b)?,
                    success: r.success,
                };
                if read.wrapper_end_ns > caller {
                    return Err("wrapper outside helper".into());
                }
                Ok(read)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Timing::from_reads(reads)
    }
}
impl Timing {
    pub(crate) fn from_reads(reads: Vec<Read>) -> Result<Self, String> {
        if !(1..=WIDTH).contains(&reads.len()) {
            return Err("source width".into());
        }
        let mut sum = 0u64;
        for r in &reads {
            if r.wrapper_end_ns.checked_sub(r.wrapper_start_ns) != Some(r.wrapper_wall_ns) {
                return Err("wrapper interval mismatch".into());
            }
            sum = sum
                .checked_add(r.wrapper_wall_ns)
                .ok_or("wrapper sum overflow")?;
        }
        Ok(Self {
            read_critical_span_ns: reads
                .iter()
                .map(|r| r.wrapper_end_ns)
                .max()
                .unwrap()
                .checked_sub(reads.iter().map(|r| r.wrapper_start_ns).min().unwrap())
                .ok_or("critical span underflow")?,
            batch_max_read_wall_ns: reads.iter().map(|r| r.wrapper_wall_ns).max().unwrap(),
            batch_sum_read_wall_ns: sum,
            reads,
        })
    }
    pub(crate) fn validate(&self, width: usize, caller_ns: u64) -> Result<(), String> {
        if self.reads.len() != width
            || self
                .reads
                .iter()
                .any(|r| !r.success || r.wrapper_end_ns > caller_ns)
            || Self::from_reads(self.reads.clone())? != *self
        {
            return Err("wrapper evidence mismatch".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Range {
    pub(crate) pointer: usize,
    pub(crate) length: usize,
}
impl Range {
    pub(crate) fn validate(self) -> Result<(), String> {
        if self.pointer == 0
            || self.pointer % ALIGN != 0
            || self.length != FULL
            || self.length % ALIGN != 0
            || self.pointer.checked_add(self.length).is_none()
        {
            return Err("mapped lock destination must be a nonzero 4096-aligned FULL range".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LockEvidence {
    pub(crate) destinations: Vec<Range>,
    pub(crate) lock_ranges_requested: u64,
    pub(crate) lock_attempts: u64,
    pub(crate) lock_successes: u64,
    pub(crate) lock_failures: u64,
    pub(crate) requested_locked_bytes: u64,
    pub(crate) mlock_wall_ns: u64,
    pub(crate) unlock_attempts: u64,
    pub(crate) unlock_successes: u64,
    pub(crate) unlock_failures: u64,
    pub(crate) munlock_wall_ns: u64,
    pub(crate) errno_histogram: BTreeMap<i32, u64>,
    pub(crate) pointer_mod_4096: Vec<usize>,
    pub(crate) lengths: Vec<usize>,
}
impl LockEvidence {
    pub(crate) fn new(mode: Mode, destinations: Vec<Range>) -> Self {
        let n = if mode == Mode::MappedLocked {
            destinations.len() as u64
        } else {
            0
        };
        Self {
            lock_ranges_requested: n,
            requested_locked_bytes: n * FULL as u64,
            pointer_mod_4096: destinations.iter().map(|r| r.pointer % ALIGN).collect(),
            lengths: destinations.iter().map(|r| r.length).collect(),
            destinations,
            ..Self::default()
        }
    }
    pub(crate) fn validate(&self, mode: Mode, width: usize) -> Result<(), String> {
        if self.destinations.len() != width || !(1..=WIDTH).contains(&width) {
            return Err("lock destination count".into());
        }
        for (i, r) in self.destinations.iter().enumerate() {
            r.validate()?;
            if self.destinations[..i]
                .iter()
                .any(|a| a.pointer < r.pointer + r.length && r.pointer < a.pointer + a.length)
            {
                return Err("overlapping lock destinations".into());
            }
        }
        let n = if mode == Mode::MappedLocked {
            width as u64
        } else {
            0
        };
        if self.lock_ranges_requested != n
            || self.requested_locked_bytes != n * FULL as u64
            || self.lock_attempts != n
            || self.lock_successes != n
            || self.lock_failures != 0
            || self.unlock_attempts != n
            || self.unlock_successes != n
            || self.unlock_failures != 0
            || !self.errno_histogram.is_empty()
            || self.pointer_mod_4096 != vec![0; width]
            || self.lengths != vec![FULL; width]
            || (n == 0 && (self.mlock_wall_ns != 0 || self.munlock_wall_ns != 0))
        {
            return Err("lock/unlock authority failed".into());
        }
        Ok(())
    }
}
/// Only the system implementation dereferences the range through the OS.
/// Test implementations record calls without touching memory or hardware.
pub(crate) trait LockSyscalls {
    unsafe fn lock(&self, range: Range) -> Result<(), i32>;
    unsafe fn unlock(&self, range: Range) -> Result<(), i32>;
}
pub(crate) struct SystemLocks;
impl LockSyscalls for SystemLocks {
    unsafe fn lock(&self, range: Range) -> Result<(), i32> {
        #[cfg(target_os = "linux")]
        {
            if libc::mlock(range.pointer as *const libc::c_void, range.length) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error()
                    .raw_os_error()
                    .unwrap_or(libc::EIO))
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = range;
            Err(libc::ENOSYS)
        }
    }
    unsafe fn unlock(&self, range: Range) -> Result<(), i32> {
        #[cfg(target_os = "linux")]
        {
            if libc::munlock(range.pointer as *const libc::c_void, range.length) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error()
                    .raw_os_error()
                    .unwrap_or(libc::EIO))
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = range;
            Err(libc::ENOSYS)
        }
    }
}
/// The caller must keep every mapped view alive until this guard drops.
/// Declare after the views and before the source timer. Release after helper
/// completion, before materialization/unmap. Drop also covers early unwinding.
pub(crate) struct LockGuard<'a, S: LockSyscalls> {
    sys: &'a S,
    evidence: &'a mut LockEvidence,
    locked: Vec<Range>,
}
impl<'a, S: LockSyscalls> LockGuard<'a, S> {
    /// SAFETY: destinations are exact live mapped ranges until guard destruction.
    pub(crate) unsafe fn acquire(
        mode: Mode,
        sys: &'a S,
        evidence: &'a mut LockEvidence,
    ) -> Result<Self, String> {
        for (i, r) in evidence.destinations.iter().enumerate() {
            r.validate()?;
            if evidence.destinations[..i]
                .iter()
                .any(|a| a.pointer < r.pointer + r.length && r.pointer < a.pointer + a.length)
            {
                return Err("overlapping lock destinations".into());
            }
        }
        let mut guard = Self {
            sys,
            locked: Vec::with_capacity(evidence.destinations.len()),
            evidence,
        };
        if mode == Mode::MappedLocked {
            for i in 0..guard.evidence.destinations.len() {
                let range = guard.evidence.destinations[i];
                guard.evidence.lock_attempts += 1;
                let start = Instant::now();
                let result = sys.lock(range);
                guard.evidence.mlock_wall_ns += start.elapsed().as_nanos() as u64;
                match result {
                    Ok(()) => {
                        guard.evidence.lock_successes += 1;
                        guard.locked.push(range);
                    }
                    Err(errno) => {
                        guard.evidence.lock_failures += 1;
                        *guard.evidence.errno_histogram.entry(errno).or_default() += 1;
                        return Err(format!("mlock failed at source index {i}: errno {errno}"));
                    }
                }
            }
        }
        Ok(guard)
    }
    pub(crate) fn release(&mut self) -> Result<(), String> {
        let mut remaining = Vec::new();
        for range in self.locked.drain(..) {
            self.evidence.unlock_attempts += 1;
            let start = Instant::now();
            let result = unsafe { self.sys.unlock(range) };
            self.evidence.munlock_wall_ns += start.elapsed().as_nanos() as u64;
            match result {
                Ok(()) => self.evidence.unlock_successes += 1,
                Err(errno) => {
                    self.evidence.unlock_failures += 1;
                    *self.evidence.errno_histogram.entry(errno).or_default() += 1;
                    remaining.push(range);
                }
            }
        }
        self.locked = remaining;
        if self.evidence.unlock_failures != 0 {
            Err("munlock failed; arm is non-authoritative".into())
        } else {
            Ok(())
        }
    }
}
impl<S: LockSyscalls> Drop for LockGuard<'_, S> {
    fn drop(&mut self) {
        let _ = self.release();
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProcessMemory {
    pub(crate) rlimit_memlock_soft: u64,
    pub(crate) rlimit_memlock_hard: u64,
    pub(crate) vmlck_bytes: u64,
}
impl ProcessMemory {
    pub(crate) fn capture() -> Result<Self, String> {
        #[cfg(target_os = "linux")]
        {
            let mut limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) } != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            let status = std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?;
            Ok(Self {
                rlimit_memlock_soft: limit.rlim_cur,
                rlimit_memlock_hard: limit.rlim_max,
                vmlck_bytes: parse_vmlck(&status)?,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err("HMA-1F hardware execution requires Linux; portable tests use fixtures".into())
        }
    }
    pub(crate) fn capacity(&self, bytes: u64) -> Result<(), String> {
        if self.rlimit_memlock_soft > self.rlimit_memlock_hard
            || self
                .vmlck_bytes
                .checked_add(bytes)
                .is_none_or(|n| n > self.rlimit_memlock_soft)
        {
            return Err("insufficient RLIMIT_MEMLOCK".into());
        }
        Ok(())
    }
    /// Isolated process, synchronous unmap/shutdown, page-aligned full ranges:
    /// exact process VmLck equality is required. No global Mlocked tolerance.
    pub(crate) fn cleaned(&self, after: &Self) -> Result<(), String> {
        if self != after {
            Err("arm RLIMIT_MEMLOCK/VmLck cleanup did not return exactly to baseline".into())
        } else {
            Ok(())
        }
    }
}
pub(crate) fn parse_vmlck(status: &str) -> Result<u64, String> {
    let rows: Vec<_> = status.lines().filter(|l| l.starts_with("VmLck:")).collect();
    if rows.len() != 1 {
        return Err("missing/duplicate VmLck".into());
    }
    let fields: Vec<_> = rows[0].split_whitespace().collect();
    if fields.len() != 3 || fields[2] != "kB" {
        return Err("VmLck units".into());
    }
    fields[1]
        .parse::<u64>()
        .ok()
        .and_then(|v| v.checked_mul(1024))
        .ok_or("VmLck value overflow".into())
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    pub(crate) measured: bool,
    pub(crate) request_index: usize,
    pub(crate) source_index: usize,
    pub(crate) ids: Vec<u32>,
    pub(crate) bytes: usize,
    pub(crate) caller_ns: u64,
    pub(crate) timing: Option<Timing>,
    pub(crate) locks: LockEvidence,
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
        locks: LockEvidence,
        failure: Option<String>,
    ) {
        let mut s = self.store.lock();
        let r = Record {
            measured: s.measured,
            request_index: s.request_index,
            source_index: s.source_index,
            ids: ids.to_vec(),
            bytes,
            caller_ns,
            timing,
            locks,
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

/// Hardware capability gate only. This function has no runtime, model, expert
/// storage, inference, source helper, queue submission or qualifier dependency.
pub(crate) fn probe_command(
    report_out: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::gpu_native_physical_install_staging::source_to_upload_production::mapped_lock::write_new;
    let mut report = serde_json::json!({"schema":"mer.gpu-native-mapped-destination-lock-probe.v1","complete":false,"capability_pass":false,"failure":null});
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        if !cfg!(target_os = "linux") {
            return Err("probe requires Linux NVIDIA L4/Vulkan".into());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .find(|a| {
                let i = a.get_info();
                i.name == "NVIDIA L4"
                    && i.backend == wgpu::Backend::Vulkan
                    && i.device_type == wgpu::DeviceType::DiscreteGpu
                    && i.vendor == 0x10de
            })
            .ok_or("exact NVIDIA L4 Vulkan adapter unavailable")?;
        let info = adapter.get_info();
        report["hardware"] = serde_json::json!({"name":info.name,"backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type),"vendor":info.vendor,"device":info.device,"driver":info.driver,"driver_info":info.driver_info});
        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("HMA-1F mapped lock capability"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HMA-1F single upload capability buffer"),
            size: crate::gpu_native_source_upload::UPLOAD_BYTES as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Write, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv_timeout(std::time::Duration::from_secs(30))??;
        let mut view = buffer.slice(..).get_mapped_range_mut();
        let offset =
            crate::gpu_native_source_upload::aligned_offset(view.as_ptr() as usize, view.len())?;
        let range = Range {
            pointer: view[offset..offset + FULL].as_mut_ptr() as usize,
            length: FULL,
        };
        range.validate()?;
        let before = ProcessMemory::capture()?;
        report["before"] = serde_json::to_value(&before)?;
        before.capacity(FULL as u64)?;
        let mut evidence = LockEvidence::new(Mode::MappedLocked, vec![range]);
        let lock_result = (|| -> Result<(), String> {
            // SAFETY: view lives until after guard and all cleanup paths.
            let mut guard =
                unsafe { LockGuard::acquire(Mode::MappedLocked, &SystemLocks, &mut evidence) }?;
            let locked = ProcessMemory::capture();
            let unlock = guard.release();
            drop(guard);
            let locked = locked?;
            report["locked"] = serde_json::to_value(&locked).map_err(|e| e.to_string())?;
            if locked.rlimit_memlock_soft != before.rlimit_memlock_soft
                || locked.rlimit_memlock_hard != before.rlimit_memlock_hard
                || before.vmlck_bytes.checked_add(FULL as u64) != Some(locked.vmlck_bytes)
            {
                return Err("probe VmLck increase must equal exact FULL bytes".into());
            }
            unlock?;
            evidence.validate(Mode::MappedLocked, 1)
        })();
        report["locks"] = serde_json::to_value(&evidence)?;
        let after = ProcessMemory::capture()?;
        report["after"] = serde_json::to_value(&after)?;
        drop(view);
        buffer.unmap();
        device.poll(wgpu::Maintain::Wait);
        let gpu_error = pollster::block_on(device.pop_error_scope());
        lock_result?;
        before.cleaned(&after)?;
        if let Some(e) = gpu_error {
            return Err(e.to_string().into());
        }
        Ok(())
    })();
    report["complete"] = serde_json::json!(result.is_ok());
    report["capability_pass"] = serde_json::json!(result.is_ok());
    report["failure"] = serde_json::json!(result.as_ref().err().map(ToString::to_string));
    write_new(report_out, &serde_json::to_vec_pretty(&report)?)?;
    result
}

#[cfg(test)]
mod hma1f_tests {
    use super::*;
    use std::cell::RefCell;
    #[derive(Default)]
    struct Mock {
        calls: RefCell<Vec<(bool, Range)>>,
        fail_lock: Option<usize>,
        fail_unlock: bool,
    }
    impl LockSyscalls for Mock {
        unsafe fn lock(&self, r: Range) -> Result<(), i32> {
            let mut c = self.calls.borrow_mut();
            let n = c.iter().filter(|(l, _)| *l).count();
            c.push((true, r));
            if self.fail_lock == Some(n) {
                Err(12)
            } else {
                Ok(())
            }
        }
        unsafe fn unlock(&self, r: Range) -> Result<(), i32> {
            self.calls.borrow_mut().push((false, r));
            if self.fail_unlock {
                Err(22)
            } else {
                Ok(())
            }
        }
    }
    fn ranges() -> Vec<Range> {
        (1..=3)
            .map(|i| Range {
                pointer: i * (FULL + ALIGN),
                length: FULL,
            })
            .collect()
    }
    #[test]
    fn baseline_zero_lock_and_unlock_calls() {
        let m = Mock::default();
        let mut e = LockEvidence::new(Mode::MappedBaseline, ranges());
        {
            let mut g = unsafe { LockGuard::acquire(Mode::MappedBaseline, &m, &mut e) }.unwrap();
            g.release().unwrap();
        }
        assert!(m.calls.borrow().is_empty());
        e.validate(Mode::MappedBaseline, 3).unwrap();
    }
    #[test]
    fn locked_every_exact_aligned_full_range() {
        let m = Mock::default();
        let mut e = LockEvidence::new(Mode::MappedLocked, ranges());
        {
            let mut g = unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }.unwrap();
            assert_eq!(
                m.calls.borrow().as_slice(),
                ranges().iter().map(|r| (true, *r)).collect::<Vec<_>>()
            );
            g.release().unwrap();
        }
        e.validate(Mode::MappedLocked, 3).unwrap();
    }
    fn error_cleanup(stage: &str) {
        let m = Mock::default();
        let mut e = LockEvidence::new(Mode::MappedLocked, ranges());
        let mut source_entered = false;
        let result = (|| -> Result<(), String> {
            let mut g = unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }?;
            source_entered = true;
            let source = if stage == "source" {
                Err("source read failed")
            } else {
                Ok(())
            };
            g.release()?;
            drop(g);
            source?;
            if stage != "success" {
                return Err(stage.into());
            }
            Ok(())
        })();
        assert!(source_entered);
        assert_eq!(result.is_ok(), stage == "success");
        e.validate(Mode::MappedLocked, 3).unwrap();
        assert_eq!(m.calls.borrow().len(), 6);
    }
    #[test]
    fn successful_source_cleanup() {
        error_cleanup("success");
    }
    #[test]
    fn source_error_cleanup() {
        error_cleanup("source");
    }
    #[test]
    fn short_read_cleanup() {
        error_cleanup("short read");
    }
    #[test]
    fn materialization_error_cleanup() {
        error_cleanup("materialization");
    }
    #[test]
    fn later_error_before_unmap_cleanup() {
        error_cleanup("later error");
    }
    #[test]
    fn partial_lock_failure_unwinds_and_prevents_source() {
        for n in 0..3 {
            let m = Mock {
                fail_lock: Some(n),
                ..Default::default()
            };
            let mut e = LockEvidence::new(Mode::MappedLocked, ranges());
            let mut source = false;
            {
                if let Ok(_g) = unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) } {
                    source = true;
                }
            }
            assert!(!source);
            assert_eq!(e.lock_successes, n as u64);
            assert_eq!(e.unlock_successes, n as u64);
            assert_eq!(e.lock_failures, 1);
            assert_eq!(e.errno_histogram.get(&12), Some(&1));
        }
    }
    #[test]
    fn drop_unwinds_successful_locks_on_panic() {
        let m = Mock::default();
        let mut e = LockEvidence::new(Mode::MappedLocked, ranges());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }.unwrap();
            panic!("fault after locking");
        }));
        assert!(panic.is_err());
        e.validate(Mode::MappedLocked, 3).unwrap();
    }
    #[test]
    fn unlock_failure_fails_authority_and_attempts_all_ranges() {
        let m = Mock {
            fail_unlock: true,
            ..Default::default()
        };
        let mut e = LockEvidence::new(Mode::MappedLocked, ranges());
        {
            let mut g = unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }.unwrap();
            assert!(g.release().is_err());
        }
        assert_eq!(e.unlock_failures, 6);
        assert_eq!(e.unlock_successes, 0);
        assert!(e.validate(Mode::MappedLocked, 3).is_err());
    }
    #[test]
    fn misaligned_pointer_rejects_before_any_lock() {
        let m = Mock::default();
        let mut r = ranges();
        r[1].pointer += 1;
        let mut e = LockEvidence::new(Mode::MappedLocked, r);
        assert!(unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }.is_err());
        assert!(m.calls.borrow().is_empty());
    }
    #[test]
    fn wrong_full_length_rejects_before_any_lock() {
        let m = Mock::default();
        let mut r = ranges();
        r[1].length -= ALIGN;
        let mut e = LockEvidence::new(Mode::MappedLocked, r);
        assert!(unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }.is_err());
        assert!(m.calls.borrow().is_empty());
    }
    #[test]
    fn overlapping_ranges_reject_before_lock() {
        let m = Mock::default();
        let mut r = ranges();
        r[1] = r[0];
        let mut e = LockEvidence::new(Mode::MappedLocked, r);
        assert!(unsafe { LockGuard::acquire(Mode::MappedLocked, &m, &mut e) }.is_err());
        assert!(m.calls.borrow().is_empty());
    }
    #[test]
    fn insufficient_memlock_probe_rejects() {
        let p = ProcessMemory {
            rlimit_memlock_soft: FULL as u64 - 1,
            rlimit_memlock_hard: FULL as u64,
            vmlck_bytes: 0,
        };
        assert!(p.capacity(FULL as u64).is_err());
    }
    #[test]
    fn vmlck_cleanup_is_exact_with_no_tolerance() {
        let p = ProcessMemory {
            rlimit_memlock_soft: u64::MAX,
            rlimit_memlock_hard: u64::MAX,
            vmlck_bytes: 0,
        };
        p.cleaned(&p).unwrap();
        let mut after = p.clone();
        after.vmlck_bytes = 4096;
        assert!(p.cleaned(&after).is_err());
        after = p.clone();
        after.rlimit_memlock_soft -= 1;
        assert!(p.cleaned(&after).is_err());
    }
    #[test]
    fn vmlck_parse_rejects_ambiguity_units_and_overflow() {
        assert_eq!(parse_vmlck("Name: mer\nVmLck:\t12 kB\n").unwrap(), 12288);
        for text in [
            "",
            "VmLck: 0 B",
            "VmLck: 0 kB\nVmLck: 0 kB",
            "VmLck: 18446744073709551615 kB",
        ] {
            assert!(parse_vmlck(text).is_err());
        }
    }
    #[test]
    fn primary_reconstruction_excludes_setup_and_cleanup() {
        let base = Instant::now();
        let mut raw = RawBatch::default();
        raw.reads[0] = RawRead {
            start: Some(base + std::time::Duration::from_nanos(100)),
            end: Some(base + std::time::Duration::from_nanos(130)),
            success: true,
        };
        raw.reads[1] = RawRead {
            start: Some(base + std::time::Duration::from_nanos(110)),
            end: Some(base + std::time::Duration::from_nanos(160)),
            success: true,
        };
        let t = raw
            .reconstruct(2, base, base + std::time::Duration::from_nanos(1000))
            .unwrap();
        assert_eq!(t.read_critical_span_ns, 60);
        assert_eq!(t.batch_max_read_wall_ns, 50);
        assert_eq!(t.batch_sum_read_wall_ns, 80);
        t.validate(2, 1000).unwrap();
    }
    #[test]
    fn corrupt_wrapper_and_overflow_reject() {
        let r = Read {
            wrapper_start_ns: 10,
            wrapper_end_ns: 9,
            wrapper_wall_ns: 1,
            success: true,
        };
        assert!(Timing::from_reads(vec![r]).is_err());
        let r = Read {
            wrapper_start_ns: 0,
            wrapper_end_ns: u64::MAX,
            wrapper_wall_ns: u64::MAX,
            success: true,
        };
        assert!(Timing::from_reads(vec![r.clone(), r]).is_err());
    }
    #[test]
    fn observation_has_no_attempt_or_retry_fields() {
        let s = include_str!("gpu_native_mapped_lock.rs");
        let body = s
            .split("pub(crate) struct RawRead {")
            .nth(1)
            .unwrap()
            .split('}')
            .next()
            .unwrap();
        assert!(!body.contains("attempt"));
        assert!(!body.contains("retry"));
        assert!(!body.contains("Vec"));
    }
    #[test]
    fn actual_integration_order_locks_timer_unlock_views_unmap() {
        let s = include_str!("gpu_native_source_upload.rs");
        let body = s
            .split("pub(crate) async fn read_source(")
            .nth(1)
            .unwrap()
            .split("fn materialize_source_payload(")
            .next()
            .unwrap();
        let labels = [
            "let mut views",
            "let mut destinations",
            "LockGuard::acquire",
            "let started = Instant::now()",
            ".read_experts_batch_into_aligned_slices",
            "let stopped",
            "g.release()",
            "drop(guard)",
            "drop(destinations)",
            "self.materialize_source_payload",
            "drop(views)",
            "lease.unmap()",
        ];
        let mut last = 0;
        for l in labels {
            let i = body.find(l).unwrap_or_else(|| panic!("missing {l}"));
            assert!(i >= last, "ordering {l}");
            last = i;
        }
        assert_eq!(
            body.matches(".read_experts_batch_into_aligned_slices")
                .count(),
            1
        );
    }
    #[test]
    fn ordinary_constructor_does_not_enable_locking() {
        let s = crate::gpu_native_source_upload::State::cpu_test_production_state();
        assert!(s.snapshot().production_owned);
        let src = include_str!("gpu_native_source_upload.rs");
        let ctor = src
            .split("pub(crate) fn new_production(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(ctor.contains("mapped_lock_observer: std::sync::OnceLock::new()"));
        assert!(!ctor.contains("enable_mapped_lock_observer"));
        assert!(!ctor.contains("Mode::MappedLocked"));
    }
    #[test]
    fn probe_no_model_storage_inference_or_source_work() {
        let s = include_str!("gpu_native_mapped_lock.rs");
        let p = s
            .split("pub(crate) fn probe_command(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "NvmeStorage",
            "load_real",
            "construct_runtime",
            "run_worker",
            "read_experts",
            "execute_request",
            ".submit(",
        ] {
            assert!(!p.contains(forbidden), "{forbidden}");
        }
        assert_eq!(p.matches("device.create_buffer(").count(), 1);
        assert!(p.contains("before.capacity(FULL as u64)"));
    }
}
