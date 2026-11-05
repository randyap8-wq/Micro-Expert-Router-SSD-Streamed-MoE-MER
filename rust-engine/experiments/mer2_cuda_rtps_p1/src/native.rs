//! Independent fixed-Q4 diagnostic, not a P6B/model owner or execution grant.
//! SOURCE ONLY: no native compilation or execution in the Mac R1 task.
//! All unsafe operations are confined here. Raw handles have no implicit Drop:
//! successful explicit release is distinguished from quarantine and no terminal.
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{expected, fixture::Fixture, D, DFF, PROJECTION_BYTES, Q4_BYTES,
    STAGE_LENGTHS, STAGE_NAMES};
use cudarc::driver::{result as cu, sys};
use std::{convert::Infallible, ffi::{c_void, CString}, io::{self, Write},
    marker::PhantomData, panic::{catch_unwind, AssertUnwindSafe}, ptr, rc::Rc,
    sync::atomic::{AtomicBool, Ordering}};

/// Intentionally uninhabited. There is NO constructor, deserializer, Boolean,
/// hash, environment variable, feature, or unsafe public bypass that mints this.
/// A future reviewed process/host-gate patch must supply actual authority.
pub(super) struct ReviewedHostGate {
    _unbound: Infallible,
    _single_thread: PhantomData<Rc<()>>,
}
static INVOCATION_SPENT: AtomicBool = AtomicBool::new(false);

pub(super) const SYMBOLS: [&str; 3] = [
    "dequantize_mul_mat_vec_q4_0_cuda", "usilu_f32", "bmul_f32",
];
pub(super) const SOURCE_SHA256: [&str; 3] = [
    "510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972",
    "bea116bebb9b84106661dd425ed56ceddabbff337762edcee05f6e9134fb7e5b",
    "dc9961a0013fe24972adf27a37a986df5d78081912a6de21684a0433ba8d1b17",
];
pub(super) const BUFFER_BYTES: [usize; 8] = [
    Q4_BYTES, D * 4, 3 * 8, DFF * 4, DFF * 4, DFF * 4, DFF * 4, D * 4,
];
const MAX_PTX_BYTES: usize = 16 * 1024 * 1024; // each of three embedded images
const MAX_REPORT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Launch {
    pub module: usize,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}
pub(super) const LAUNCHES: [Launch; 5] = [
    Launch { module: 0, grid: (768, 1, 1), block: (32, 1, 1) },
    Launch { module: 0, grid: (768, 1, 1), block: (32, 1, 1) },
    Launch { module: 1, grid: (3, 1, 1), block: (256, 1, 1) },
    Launch { module: 2, grid: (3, 1, 1), block: (256, 1, 1) },
    Launch { module: 0, grid: (2048, 1, 1), block: (32, 1, 1) },
];

/// These exact build-embedded bytes are the ONLY module-load input. The later
/// host gate must SHA-256 these bytes and bind them to source/build/binary logs.
/// No caller-supplied PTX, file loading, NVRTC or runtime package resolution.
pub(super) fn embedded_ptx() -> [&'static str; 3] {
    [candle_kernels::QUANTIZED.ptx(), candle_kernels::UNARY.ptx(),
        candle_kernels::BINARY.ptx()]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Fault {
    pub at: &'static str,
    // Numeric CUresult only: formatting DriverError can itself call the driver.
    pub cuda: Option<u32>,
}
impl Fault {
    fn local(at: &'static str) -> Self { Self { at, cuda: None } }
}
type NativeResult<T> = std::result::Result<T, Fault>;
fn checked<T>(at: &'static str, result: std::result::Result<T, cu::DriverError>) -> NativeResult<T> {
    result.map_err(|error| Fault { at, cuda: Some(error.0 as u32) })
}

#[derive(Default)]
pub(super) struct Failures {
    pub first: Option<Fault>,
    pub secondary: [Option<Fault>; 8],
    pub count: usize,
    pub overflow: bool,
}
impl Failures {
    pub fn latch(&mut self, fault: Fault) {
        if self.first.is_none() { self.first = Some(fault); }
        else if self.count < self.secondary.len() {
            self.secondary[self.count] = Some(fault);
            self.count += 1;
        } else { self.overflow = true; }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cleanup {
    NotAttempted, NoContextAttempted, ExplicitCallsReturnedSuccess,
    QuarantinedNeedsExternalReaper,
}

pub(super) struct NativeReport {
    pub failures: Failures,
    pub initialization_attempts: u8,
    pub symbols_resolved: [bool; 3],
    pub launch_attempted: [bool; 5],
    pub launch_returned: [bool; 5],
    pub fence_returned: [bool; 5],
    pub copied: [bool; 5],
    pub readback_complete: bool,
    pub exact: bool,
    pub cleanup: Cleanup,
    pub cleanup_attempts: u8,
    pub terminal: bool,
    // 0 ctx, 1 stream, 2..5 modules, 5..13 allocations; receipt only.
    pub quarantined_handles: [u64; 13],
    bits: [Vec<u32>; 5],
}
impl NativeReport {
    fn new() -> Self {
        Self { failures: Failures::default(), initialization_attempts: 0,
            symbols_resolved: [false; 3], launch_attempted: [false; 5],
            launch_returned: [false; 5], fence_returned: [false; 5],
            copied: [false; 5], readback_complete: false, exact: false,
            cleanup: Cleanup::NotAttempted, cleanup_attempts: 0, terminal: false,
            quarantined_handles: [0; 13],
            bits: STAGE_LENGTHS.map(|length| vec![0x7fc0_d1a6; length]) }
    }

    /// Reports every copied stage element, including failures; no hash-only
    /// receipt can populate bits or become complete readback. A killed/hung
    /// process may never return/write this object: terminal must not be inferred.
    pub(super) fn write<W: Write>(&self, output: &mut W) -> io::Result<()> {
        let mut out = LimitedWriter { output, left: MAX_REPORT_BYTES };
        writeln!(out, "kind=SYNTHETIC_NATIVE_Q4_DIAGNOSTIC\nrun_authority=EXTERNAL_GATE_REQUIRED\ncpu_fallback=false")?;
        writeln!(out, "terminal={}\nmissing_terminal_possible=true\nreadback_complete={}\nexact_bits={}\ninitialization_attempts={}",
            self.terminal, self.readback_complete, self.exact, self.initialization_attempts)?;
        let complete = self.terminal && self.readback_complete && self.exact
            && self.failures.first.is_none() && self.cleanup == Cleanup::ExplicitCallsReturnedSuccess;
        writeln!(out, "diagnostic_complete={}\nqualification=NONE", complete)?;
        writeln!(out, "first_failure={:?}\nsecondary={:?}\nsecondary_overflow={}",
            self.failures.first, &self.failures.secondary[..self.failures.count], self.failures.overflow)?;
        for (index, ptx) in embedded_ptx().iter().enumerate() {
            writeln!(out, "module={} source_sha256={} symbol={} resolved={} embedded_ptx_bytes={} ptx_sha256=EXTERNAL_SEAL_REQUIRED",
                index, SOURCE_SHA256[index], SYMBOLS[index], self.symbols_resolved[index], ptx.len())?;
        }
        for (stage, launch) in LAUNCHES.iter().enumerate() {
            writeln!(out, "stage={} symbol={} grid={:?} block={:?} shared=0 launch_attempted={} launch_returned={} fence_returned={} copied={}",
                STAGE_NAMES[stage], SYMBOLS[launch.module], launch.grid, launch.block,
                self.launch_attempted[stage], self.launch_returned[stage], self.fence_returned[stage], self.copied[stage])?;
            if !self.copied[stage] { continue; }
            for (index, bits) in self.bits[stage].iter().copied().enumerate() {
                let expected_bits = expected::bits(stage, index)
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "oracle index"))?;
                let actual = f32::from_bits(bits);
                let delta = actual as f64 - f32::from_bits(expected_bits) as f64;
                writeln!(out, "stage={} index={} expected=0x{:08x} actual=0x{:08x} finite={} signed_delta={:.17e} abs_delta={:.17e} exact={}",
                    STAGE_NAMES[stage], index, expected_bits, bits, actual.is_finite(),
                    delta, delta.abs(), actual.is_finite() && bits == expected_bits)?;
            }
        }
        writeln!(out, "cleanup={:?}\ncleanup_attempts={}\nquarantined_handles={:?}\nCudaContext_Drop=NOT_USED\nresource_leak_freedom=NOT_ASSERTED\nP6B_loaded_owner=NOT_TESTED",
            self.cleanup, self.cleanup_attempts, self.quarantined_handles)
    }
}

struct LimitedWriter<'a, W> { output: &'a mut W, left: usize }
impl<W: Write> Write for LimitedWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.left { return Err(io::Error::other("native report cap")); }
        let written = self.output.write(bytes)?;
        self.left -= written;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> { self.output.flush() }
}

/// A single raw owner, never Clone/Send/Sync, with no implicit driver destructor.
/// Only explicit cleanup below touches retained resources after an error.
struct Owner {
    device: Option<sys::CUdevice>,
    context: Option<sys::CUcontext>,
    stream: Option<sys::CUstream>,
    modules: [Option<sys::CUmodule>; 3],
    functions: [Option<sys::CUfunction>; 3],
    buffers: [Option<sys::CUdeviceptr>; 8],
    context_attempted: bool,
    bound: bool,
    ambiguous: bool,
    cleanup_spent: bool,
    _single_thread: PhantomData<Rc<()>>,
}
impl Owner {
    fn empty() -> Self {
        Self { device: None, context: None, stream: None, modules: [None; 3],
            functions: [None; 3], buffers: [None; 8], context_attempted: false,
            bound: false, ambiguous: false, cleanup_spent: false,
            _single_thread: PhantomData }
    }
    fn buffer(&self, slot: usize) -> NativeResult<sys::CUdeviceptr> {
        self.buffers.get(slot).copied().flatten().ok_or(Fault::local("missing_buffer"))
    }
    fn quarantine(&self, report: &mut NativeReport) {
        report.cleanup = Cleanup::QuarantinedNeedsExternalReaper;
        report.quarantined_handles[0] = self.context.map_or(0, |x| x as usize as u64);
        report.quarantined_handles[1] = self.stream.map_or(0, |x| x as usize as u64);
        for (i, handle) in self.modules.iter().enumerate() {
            report.quarantined_handles[2+i] = handle.map_or(0, |x| x as usize as u64);
        }
        for (i, handle) in self.buffers.iter().enumerate() {
            report.quarantined_handles[5+i] = handle.unwrap_or(0);
        }
        // Raw handles deliberately remain unreleased. No retry, reset, new owner,
        // implicit Drop, or guessed success. The external reaper owns resolution.
    }

    fn execute(&mut self, fixture: &Fixture, images: &[CString; 3], report: &mut NativeReport) -> NativeResult<()> {
        report.initialization_attempts = 1;
        checked("driver_init", cu::init())?;
        if checked("current_context", cu::ctx::get_current())?.is_some() {
            return Err(Fault::local("existing_context_rejected"));
        }
        let device = checked("device_ordinal_zero", cu::device::get(0))?;
        self.device = Some(device);
        self.context_attempted = true;
        // SAFETY: sole invocation, valid device, external gate must establish a
        // fresh single-threaded process with no shared primary-context users.
        let context = checked("primary_context_retain", unsafe { cu::primary_ctx::retain(device) })?;
        self.context = Some(context);
        // SAFETY: retained live primary context, same thread for all operations.
        checked("bind_context", unsafe { cu::ctx::set_current(context) })?;
        self.bound = true;
        self.stream = Some(checked("stream_create", cu::stream::create(cu::stream::StreamKind::NonBlocking))?);
        for index in 0..3 {
            // SAFETY: bounded build-embedded NUL-terminated PTX retained by caller.
            let module = checked("module_load", unsafe { cu::module::load_data(images[index].as_ptr().cast()) })?;
            self.modules[index] = Some(module);
            let name = CString::new(SYMBOLS[index]).map_err(|_| Fault::local("symbol_nul"))?;
            // SAFETY: module retained until explicit cleanup; exact f32 symbol.
            self.functions[index] = Some(checked("symbol_resolve", unsafe { cu::module::get_function(module, name) })?);
            report.symbols_resolved[index] = true;
        }
        for (slot, bytes) in BUFFER_BYTES.into_iter().enumerate() {
            // SAFETY: eight fixed nonzero sizes; no host-controlled allocation.
            self.buffers[slot] = Some(checked("allocate", unsafe { cu::malloc_sync(bytes) })?);
        }
        // Metadata is real device storage, avoiding null-pointer arithmetic in
        // vendor macros: dims[0]=768, lhs stride=1, rhs stride=1, size_t=u64.
        let metadata: [u64; 3] = [DFF as u64, 1, 1];
        // SAFETY: allocations exactly match slice sizes; synchronous copies keep
        // borrowed host data alive through completion even on early return.
        checked("upload_q4", unsafe { cu::memcpy_htod_sync(self.buffer(0)?, fixture.bytes()) })?;
        checked("upload_input", unsafe { cu::memcpy_htod_sync(self.buffer(1)?, fixture.input()) })?;
        checked("upload_metadata", unsafe { cu::memcpy_htod_sync(self.buffer(2)?, &metadata) })?;
        for slot in 3..8 {
            // SAFETY: exact output allocation size. NaN poison detects unwritten
            // elements; it is never represented as completed GPU readback.
            checked("poison_output", unsafe { cu::memset_d8_sync(self.buffer(slot)?, 0xff, BUFFER_BYTES[slot]) })?;
        }
        checked("upload_fence", cu::ctx::synchronize())?;
        for stage in 0..5 { self.launch(stage, report)?; }
        for stage in 0..5 {
            // SAFETY: stage buffer has exactly len*4 bytes, all five launch
            // fences returned; u32 preserves the device's exact f32 bits.
            checked("readback", unsafe { cu::memcpy_dtoh_sync(&mut report.bits[stage], self.buffer(3+stage)?) })?;
            report.copied[stage] = true;
        }
        checked("readback_fence", cu::ctx::synchronize())?;
        report.readback_complete = true;
        report.exact = true;
        for stage in 0..5 {
            for (index, bits) in report.bits[stage].iter().copied().enumerate() {
                let wanted = expected::bits(stage, index).map_err(|_| Fault::local("oracle_index"))?;
                if !f32::from_bits(bits).is_finite() || bits != wanted { report.exact = false; }
            }
        }
        if !report.exact { return Err(Fault::local("native_exact_bits_mismatch")); }
        Ok(())
    }

    fn launch(&self, stage: usize, report: &mut NativeReport) -> NativeResult<()> {
        let spec = *LAUNCHES.get(stage).ok_or(Fault::local("stage_index"))?;
        let stream = self.stream.ok_or(Fault::local("missing_stream"))?;
        let function = self.functions[spec.module].ok_or(Fault::local("missing_function"))?;
        let mut matrix = self.buffer(0)? + match stage { 1 => PROJECTION_BYTES as u64, 4 => (2*PROJECTION_BYTES) as u64, _ => 0 };
        let input_slot = match stage { 0 | 1 => 1, 2 => 3, 3 => 5, 4 => 6,
            _ => return Err(Fault::local("stage_index")) };
        let mut input = self.buffer(input_slot)?;
        let mut output = self.buffer(3+stage)?;
        let mut up = self.buffer(4)?;
        let mut metadata = self.buffer(2)?;
        let mut count: u64 = DFF as u64;
        let mut dims: u64 = 1;
        let mut cols: i32 = if stage == 4 { DFF as i32 } else { D as i32 };
        let mut rows: i32 = STAGE_LENGTHS[stage] as i32;
        let mut args: [*mut c_void; 6] = [ptr::null_mut(); 6];
        let arity = if stage == 0 || stage == 1 || stage == 4 {
            args[..5].copy_from_slice(&[arg(&mut matrix), arg(&mut input), arg(&mut output), arg(&mut cols), arg(&mut rows)]);
            5
        } else if stage == 2 {
            args[..5].copy_from_slice(&[arg(&mut count), arg(&mut dims), arg(&mut metadata), arg(&mut input), arg(&mut output)]);
            5
        } else {
            args.copy_from_slice(&[arg(&mut count), arg(&mut dims), arg(&mut metadata), arg(&mut input), arg(&mut up), arg(&mut output)]);
            6
        };
        report.launch_attempted[stage] = true; // latch before the driver call
        // SAFETY: exact pinned symbol ABI; u64 device pointers/size_t and i32
        // dimensions; disjoint outputs; all allocations and modules retained.
        // Kernel parameter storage lives until the enqueue returns. No CPU data
        // populates an output; no alias to input or prior-stage output is writable.
        let launch_at = ["gate_launch", "up_launch", "silu_launch", "product_launch", "down_launch"][stage];
        checked(launch_at, unsafe { cu::launch_kernel(function, spec.grid, spec.block, 0, stream, &mut args[..arity]) })?;
        report.launch_returned[stage] = true;
        // SAFETY: owned live stream. Every stage must complete before the next.
        let fence_at = ["gate_fence", "up_fence", "silu_fence", "product_fence", "down_fence"][stage];
        checked(fence_at, unsafe { cu::stream::synchronize(stream) })?;
        report.fence_returned[stage] = true;
        Ok(())
    }

    fn cleanup(&mut self, report: &mut NativeReport) -> NativeResult<()> {
        if self.cleanup_spent { return Err(Fault::local("cleanup_repeated")); }
        self.cleanup_spent = true;
        report.cleanup_attempts = 1;
        if !self.context_attempted {
            report.cleanup = Cleanup::NoContextAttempted;
            return Ok(());
        }
        if !self.bound { self.quarantine(report); return Ok(()); }
        // One checked drain attempt. It can hang: a process-local timeout cannot
        // prove completion, so the separately reviewed external reaper is needed.
        if let Err(error) = checked("cleanup_drain", cu::ctx::synchronize()) {
            self.quarantine(report);
            return Err(error);
        }
        if self.ambiguous { self.quarantine(report); return Ok(()); }
        for slot in (0..8).rev() {
            if let Some(handle) = self.buffers[slot] {
                // SAFETY: known allocation, drain succeeded, no later use.
                if let Err(error) = checked("cleanup_free", unsafe { cu::free_sync(handle) }) {
                    self.quarantine(report); return Err(error);
                }
                self.buffers[slot] = None;
            }
        }
        self.functions = [None; 3];
        for slot in (0..3).rev() {
            if let Some(handle) = self.modules[slot] {
                // SAFETY: no pending work and no remaining function use.
                if let Err(error) = checked("cleanup_module", unsafe { cu::module::unload(handle) }) {
                    self.quarantine(report); return Err(error);
                }
                self.modules[slot] = None;
            }
        }
        if let Some(stream) = self.stream {
            // SAFETY: drained owned stream, destroyed at most once.
            if let Err(error) = checked("cleanup_stream", unsafe { cu::stream::destroy(stream) }) {
                self.quarantine(report); return Err(error);
            }
            self.stream = None;
        }
        // SAFETY: no current resources remain in use; detach on this same thread.
        if let Err(error) = checked("cleanup_unbind", unsafe { cu::ctx::set_current(ptr::null_mut()) }) {
            self.quarantine(report); return Err(error);
        }
        self.bound = false;
        if let Some(device) = self.device {
            // SAFETY: balance exactly the one primary retain. This return code
            // is not proof of final destruction, system quiescence, or no leaks.
            if let Err(error) = checked("cleanup_primary_release", unsafe { cu::primary_ctx::release(device) }) {
                self.quarantine(report); return Err(error);
            }
            self.context = None;
        }
        report.cleanup = Cleanup::ExplicitCallsReturnedSuccess;
        Ok(())
    }
}

fn arg<T>(value: &mut T) -> *mut c_void { (value as *mut T).cast() }

/// Staged internal seam; no caller can construct its required gate in this
/// patch. Feature enablement is not run authority. No public CLI calls this.
pub(super) fn staged_once(_gate: ReviewedHostGate) -> NativeReport {
    let mut report = NativeReport::new();
    if INVOCATION_SPENT.swap(true, Ordering::SeqCst) {
        report.failures.latch(Fault::local("invocation_already_spent"));
        report.cleanup = Cleanup::NoContextAttempted;
        report.terminal = true;
        return report;
    }
    let mut owner = Owner::empty();
    let run = catch_unwind(AssertUnwindSafe(|| -> NativeResult<()> {
        let fixture = Fixture::build();
        fixture.validate().map_err(|_| Fault::local("fixed_fixture_validation"))?;
        let ptx = embedded_ptx();
        for image in ptx {
            if image.is_empty() || image.len() > MAX_PTX_BYTES {
                return Err(Fault::local("embedded_ptx_size"));
            }
        }
        let images = [
            CString::new(ptx[0]).map_err(|_| Fault::local("embedded_ptx_nul"))?,
            CString::new(ptx[1]).map_err(|_| Fault::local("embedded_ptx_nul"))?,
            CString::new(ptx[2]).map_err(|_| Fault::local("embedded_ptx_nul"))?,
        ];
        owner.execute(&fixture, &images, &mut report)
    }));
    match run {
        Ok(Ok(())) => {},
        Ok(Err(error)) => {
            owner.ambiguous = error.cuda.is_some();
            report.failures.latch(error);
        },
        Err(payload) => {
            // Arbitrary panic-payload Drop could panic again; retain it for the
            // terminating process. This is a single bounded invocation.
            std::mem::forget(payload);
            owner.ambiguous = true;
            report.failures.latch(Fault::local("native_panic"));
        },
    }
    match catch_unwind(AssertUnwindSafe(|| owner.cleanup(&mut report))) {
        Ok(Ok(())) => {},
        Ok(Err(error)) => report.failures.latch(error),
        Err(payload) => {
            std::mem::forget(payload);
            owner.quarantine(&mut report);
            report.failures.latch(Fault::local("cleanup_panic"));
        },
    }
    // Only reached when calls return. Abort, OOM, driver hang, external kill,
    // and output loss remain a distinct MISSING_TERMINAL external observation.
    report.terminal = true;
    report
}
