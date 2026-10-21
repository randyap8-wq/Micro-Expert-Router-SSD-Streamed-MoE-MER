//! SOURCE REVIEW ONLY under #296: never compiled or executed in Stage S.
use crate::{
    state::{Adapter, Kind, Op, Receipt, Runner, Token},
    verify_output, Data, Prepared,
};
use cudarc::{
    driver::{
        sys, CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr,
        DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits,
    },
    nvrtc::Ptx,
};
use mer2_cuda_q4_s2::{sha256, KERNEL, MUL, SILU};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

enum Host {
    Bytes(PinnedHostSlice<u8>),
    Layout(PinnedHostSlice<usize>),
    Floats(PinnedHostSlice<f32>),
}
enum Device {
    Bytes(CudaSlice<u8>),
    Layout(CudaSlice<usize>),
    Floats(CudaSlice<f32>),
}
impl Device {
    fn bytes(&self) -> usize {
        match self {
            Self::Bytes(v) => v.len(),
            Self::Layout(v) => v.len() * 8,
            Self::Floats(v) => v.len() * 4,
        }
    }
    fn address(&self, stream: &Arc<CudaStream>) -> u64 {
        match self {
            Self::Bytes(v) => v.device_ptr(stream).0,
            Self::Layout(v) => v.device_ptr(stream).0,
            Self::Floats(v) => v.device_ptr(stream).0,
        }
    }
    fn floats(&self) -> Result<&CudaSlice<f32>, String> {
        if let Self::Floats(v) = self {
            Ok(v)
        } else {
            Err("device f32 type".into())
        }
    }
    fn floats_mut(&mut self) -> Result<&mut CudaSlice<f32>, String> {
        if let Self::Floats(v) = self {
            Ok(v)
        } else {
            Err("device f32 type".into())
        }
    }
}

#[derive(Default)]
struct Batch {
    staging_ms: f64,
    enqueue_start: Option<Instant>,
    enqueue_ms: Option<f64>,
    wait_ms: Option<f64>,
}

/// The sole concrete, non-cloneable run owner. Every successfully returned
/// CUDA owner is inserted here before another fallible operation. On unknown
/// drain Runner quarantines this entire object, including host destinations.
struct Owner {
    id: u64,
    ctx: Option<Arc<CudaContext>>,
    stream: Option<Arc<CudaStream>>,
    modules: [Option<Arc<CudaModule>>; 3],
    functions: [Option<CudaFunction>; 3],
    host: [Option<Host>; 6],
    initialized: [bool; 6],
    device: [Option<Device>; 10],
    events: [Option<CudaEvent>; 8],
    recorded: [Option<Token>; 8],
    completed: [bool; 8],
    addresses: BTreeSet<u64>,
    output: [Box<[f32; 128]>; 2],
    batches: [Batch; 3],
}
impl Owner {
    fn new(id: u64) -> Self {
        Self {
            id,
            ctx: None,
            stream: None,
            modules: std::array::from_fn(|_| None),
            functions: std::array::from_fn(|_| None),
            host: std::array::from_fn(|_| None),
            initialized: [false; 6],
            device: std::array::from_fn(|_| None),
            events: std::array::from_fn(|_| None),
            recorded: [None; 8],
            completed: [false; 8],
            addresses: BTreeSet::new(),
            output: [
                Box::new([f32::from_bits(0x7fc00000); 128]),
                Box::new([f32::from_bits(0x7fc00000); 128]),
            ],
            batches: std::array::from_fn(|_| Batch::default()),
        }
    }
    fn context(&self) -> Result<&Arc<CudaContext>, String> {
        self.ctx.as_ref().ok_or_else(|| "missing context".into())
    }
    fn stream(&self) -> Result<&Arc<CudaStream>, String> {
        self.stream.as_ref().ok_or_else(|| "missing stream".into())
    }
    fn create(&mut self, kind: Kind, id: u8, data: &Data) -> Result<Receipt, String> {
        let i = usize::from(id);
        let mut r = Receipt::default();
        match kind {
            Kind::Context => {
                self.ctx = Some(CudaContext::new(0).map_err(err)?);
                let ctx = self.context()?;
                r.detail = json!({"context_pointer":ctx.cu_ctx() as usize,"ordinal":ctx.ordinal(),"async_allocator":ctx.has_async_alloc(),
                    "library_event_tracking":ctx.is_event_tracking(),"observation":"CUDA_API_AT_CONTEXT_CREATION"});
            }
            Kind::Stream => {
                self.stream = Some(self.context()?.default_stream());
                r.detail = json!({"stream":"single_default_stream"});
            }
            Kind::Module => {
                let ptx = match id {
                    0 => candle_kernels::QUANTIZED.ptx(),
                    1 => candle_kernels::UNARY.ptx(),
                    2 => candle_kernels::BINARY.ptx(),
                    _ => return Err("module id".into()),
                };
                self.modules[i] = Some(
                    self.context()?
                        .load_module(Ptx::from_src(ptx))
                        .map_err(err)?,
                );
                r.detail = json!({"ptx_sha256":sha256(ptx.as_bytes())});
            }
            Kind::Function => {
                self.functions[i] = Some(
                    self.modules[i]
                        .as_ref()
                        .ok_or("missing module")?
                        .load_function([KERNEL, SILU, MUL][i])
                        .map_err(err)?,
                );
            }
            Kind::Pinned => {
                let ctx = self.context()?.clone();
                // SAFETY: bounded nonzero primitive lengths, checked 64-bit
                // host. Uninitialized allocation is immediately retained here;
                // no typed slice is formed until every element has been written.
                self.host[i] = Some(unsafe {
                    match id {
                        0..=2 => Host::Bytes(ctx.alloc_pinned::<u8>(9216).map_err(err)?),
                        3 => Host::Layout(ctx.alloc_pinned::<usize>(3).map_err(err)?),
                        4..=5 => Host::Floats(ctx.alloc_pinned::<f32>(128).map_err(err)?),
                        _ => return Err("pinned id".into()),
                    }
                });
                let start = Instant::now();
                match self.host[i].as_mut().ok_or("missing pinned owner")? {
                    Host::Bytes(v) => {
                        initialize(
                            v,
                            [
                                &data.fixture.gate.packed,
                                &data.fixture.up.packed,
                                &data.fixture.down.packed,
                            ][i],
                        )?;
                        r.bytes = v.num_bytes() as u64;
                    }
                    Host::Layout(v) => {
                        initialize(v, &[128usize, 1, 1])?;
                        r.bytes = v.num_bytes() as u64;
                    }
                    Host::Floats(v) => {
                        if id == 4 {
                            initialize(v, &data.inputs[0])?
                        } else {
                            initialize(v, &[f32::from_bits(0x7fc00000); 128])?
                        }
                        r.bytes = v.num_bytes() as u64;
                    }
                }
                self.batches[0].staging_ms += start.elapsed().as_secs_f64() * 1000.0;
                self.initialized[i] = true;
                r.destination = Some(100 + u64::from(id));
                r.detail = json!({"api":"CudaContext::alloc_pinned","flags":"WRITECOMBINED","all_elements_initialized":true});
            }
            Kind::Device => {
                let stream = self.stream()?.clone();
                // SAFETY: bounded primitive payloads. No device read is
                // scheduled until the engine has confirmed all input/reset H2D.
                self.device[i] = Some(unsafe {
                    match id {
                        0..=2 => Device::Bytes(stream.alloc::<u8>(9216).map_err(err)?),
                        3 => Device::Layout(stream.alloc::<usize>(3).map_err(err)?),
                        4..=9 => Device::Floats(stream.alloc::<f32>(128).map_err(err)?),
                        _ => return Err("device id".into()),
                    }
                });
                let owner = self.device[i].as_ref().ok_or("missing device owner")?;
                r.bytes = owner.bytes() as u64;
                r.destination = Some(200 + u64::from(id));
                let address = owner.address(&stream);
                if !self.addresses.insert(address) {
                    return Err("STOP_OWNERSHIP: duplicate live device allocation".into());
                }
                r.detail = json!({"device_pointer":address,"retained_whole_slice":true});
            }
            Kind::Event => {
                let flags = if id < 6 {
                    sys::CUevent_flags::CU_EVENT_DEFAULT
                } else {
                    sys::CUevent_flags::CU_EVENT_DISABLE_TIMING
                };
                self.events[i] = Some(self.context()?.new_event(Some(flags)).map_err(err)?);
                r.destination = Some(400 + u64::from(id));
                r.detail = json!({"timing_enabled":id<6,"application_event":true});
            }
        }
        Ok(r)
    }
    fn h2d(&mut self, src: u8, dst: u8, expected: u64) -> Result<Receipt, String> {
        if !self.initialized[usize::from(src)] {
            return Err("uninitialized staging".into());
        }
        let stream = self.stream()?.clone();
        let source = self.host[usize::from(src)]
            .as_ref()
            .ok_or("missing pinned")?;
        let target = self.device[usize::from(dst)]
            .as_mut()
            .ok_or("missing device")?;
        let bytes = match (source, target) {
            (Host::Bytes(s), Device::Bytes(d)) => copy(&stream, s, d, expected)?,
            (Host::Layout(s), Device::Layout(d)) => copy(&stream, s, d, expected)?,
            (Host::Floats(s), Device::Floats(d)) => copy(&stream, s, d, expected)?,
            _ => return Err("H2D type mismatch".into()),
        };
        Ok(Receipt {
            bytes,
            source: Some(100 + u64::from(src)),
            destination: Some(200 + u64::from(dst)),
            detail: json!({"api":"CudaStream::memcpy_htod","source_type":"PinnedHostSlice","physical_bus_traffic":"NOT_MEASURED"}),
            ..Receipt::default()
        })
    }
    fn kernel(&mut self, k: u8) -> Result<(), String> {
        let stream = self.stream()?.clone();
        let output = match k {
            0 => 5,
            1 => 6,
            2 => 7,
            3 => 8,
            4 => 9,
            _ => return Err("kernel id".into()),
        };
        let (prefix, suffix) = self.device.split_at_mut(output);
        let dst = suffix[0].as_mut().ok_or("missing output")?.floats_mut()?;
        let get = |id: usize| {
            prefix[id]
                .as_ref()
                .ok_or_else(|| "missing input".to_string())
        };
        if dst.len() != 128 {
            return Err("output geometry".into());
        }
        if matches!(k, 0 | 1 | 4) {
            let weight = match get(if k == 4 { 2 } else { usize::from(k) })? {
                Device::Bytes(v) => v,
                _ => return Err("Q4 type".into()),
            };
            let input = get(if k == 4 { 8 } else { 4 })?.floats()?;
            if weight.len() != 9216 || input.len() != 128 {
                return Err("Q4 geometry".into());
            }
            let (cols, rows) = (128i32, 128i32);
            // SAFETY: unchanged S2 ABI (void*,float*,float*,int,int), one
            // 32-thread warp/row. Disjoint whole slices, Q4 rows 4*18 bytes,
            // initialized 128-float input, live module/context and final owner.
            unsafe {
                stream
                    .launch_builder(self.functions[0].as_ref().ok_or("Q4 function")?)
                    .arg(weight)
                    .arg(input)
                    .arg(dst)
                    .arg(&cols)
                    .arg(&rows)
                    .launch(LaunchConfig {
                        grid_dim: (128, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(err)?;
        } else {
            let layout = match get(3)? {
                Device::Layout(v) => v,
                _ => return Err("layout type".into()),
            };
            if layout.len() != 3 {
                return Err("layout length".into());
            }
            let (numel, ndims) = (128usize, 1usize);
            let config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            if k == 2 {
                let gate = get(5)?.floats()?;
                // SAFETY: usilu_f32(size_t,size_t,size_t*,float*,float*).
                // [128,1,1] metadata and all disjoint lengths are fixed.
                unsafe {
                    stream
                        .launch_builder(self.functions[1].as_ref().ok_or("SiLU function")?)
                        .arg(&numel)
                        .arg(&ndims)
                        .arg(layout)
                        .arg(gate)
                        .arg(dst)
                        .launch(config)
                }
                .map_err(err)?;
            } else {
                let activated = get(7)?.floats()?;
                let up = get(6)?.floats()?;
                // SAFETY: bmul_f32 adds rhs before output; the immutable
                // prefix and exclusive suffix prove no input/output alias.
                unsafe {
                    stream
                        .launch_builder(self.functions[2].as_ref().ok_or("multiply function")?)
                        .arg(&numel)
                        .arg(&ndims)
                        .arg(layout)
                        .arg(activated)
                        .arg(up)
                        .arg(dst)
                        .launch(config)
                }
                .map_err(err)?;
            }
        }
        Ok(())
    }
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn initialize<T: DeviceRepr + ValidAsZeroBits + Copy>(
    dst: &mut PinnedHostSlice<T>,
    src: &[T],
) -> Result<(), String> {
    if dst.len() != src.len() {
        return Err("pinned initialization exact length".into());
    }
    let pointer = dst.as_mut_ptr().map_err(err)?;
    // SAFETY: new/previously completed pinned owner, matching checked lengths,
    // nonoverlapping initialized source, primitive T. Write every element before
    // creating a typed borrowed slice or passing the pinned object to cudarc.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), pointer, src.len()) };
    Ok(())
}
fn copy<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    src: &PinnedHostSlice<T>,
    dst: &mut CudaSlice<T>,
    expected: u64,
) -> Result<u64, String> {
    let bytes = src
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or("H2D length overflow")? as u64;
    if src.len() != dst.len() || bytes != expected || !Arc::ptr_eq(src.context(), dst.context()) {
        return Err("H2D exact length/context".into());
    }
    // Passing the actual pinned owner preserves HostSlice event tracking.
    stream.memcpy_htod(src, dst).map_err(err)?;
    Ok(bytes)
}

impl Adapter for Owner {
    fn execute(&mut self, op: &Op, data: &Data) -> Result<Receipt, String> {
        let mut r = Receipt::default();
        match op {
            Op::Create(kind, id) => return self.create(*kind, *id, data),
            Op::H2d { src, dst, bytes } => return self.h2d(*src, *dst, *bytes),
            Op::Install | Op::Begin(_) | Op::Retire => {}
            Op::Stage(seq) => {
                if *seq == 1 && !self.completed[6] {
                    return Err("input still in flight".into());
                }
                let start = Instant::now();
                let pinned = match self.host[4].as_mut().ok_or("missing input staging")? {
                    Host::Floats(v) => v,
                    _ => return Err("input type".into()),
                };
                initialize(pinned, &data.inputs[usize::from(*seq)])?;
                self.batches[usize::from(*seq) + 1].staging_ms =
                    start.elapsed().as_secs_f64() * 1000.0;
                r.detail = json!({"input_sha256":data.input_ids[usize::from(*seq)]});
            }
            Op::Record(t) => {
                if t.context != self.id || self.recorded[t.event as usize].is_some() {
                    return Err("event context/duplicate".into());
                }
                let i = usize::from(t.event);
                if i < 6 && i % 2 == 0 {
                    self.batches[i / 2].enqueue_start = Some(Instant::now());
                }
                self.events[i]
                    .as_ref()
                    .ok_or("missing event")?
                    .record(self.stream()?)
                    .map_err(err)?;
                self.recorded[i] = Some(*t);
                r.token = Some(*t);
                if i < 6 && i % 2 == 1 {
                    self.batches[i / 2].enqueue_ms = Some(
                        self.batches[i / 2]
                            .enqueue_start
                            .ok_or("missing enqueue start")?
                            .elapsed()
                            .as_secs_f64()
                            * 1000.0,
                    );
                }
            }
            Op::Fence(t) => {
                let i = usize::from(t.event);
                if self.recorded[i] != Some(*t) || self.completed[i] {
                    return Err("unrecorded/stale completion".into());
                }
                let start = Instant::now();
                // One of the preserved seven explicit successful-path stream
                // waits (H2D + five kernels + D2H) per request. Event identity
                // and a checked event wait additionally guard ownership.
                self.stream()?.synchronize().map_err(err)?;
                self.events[i]
                    .as_ref()
                    .ok_or("missing event")?
                    .synchronize()
                    .map_err(err)?;
                self.context()?.check_err().map_err(err)?;
                self.completed[i] = true;
                r.token = Some(*t);
                if i < 6 {
                    self.batches[i / 2].wait_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            Op::Measure(batch) => {
                let b = usize::from(*batch);
                let end = 2 * b + 1;
                if !self.completed[end] {
                    return Err("timing before checked completion".into());
                }
                let ms = self.events[2 * b]
                    .as_ref()
                    .ok_or("missing timing start")?
                    .elapsed_ms(self.events[end].as_ref().ok_or("missing timing end")?)
                    .map_err(err)?;
                if !ms.is_finite() || ms < 0.0 {
                    return Err("invalid CUDA interval".into());
                }
                let t = &self.batches[b];
                r.detail = json!({"status":"MEASURED_CUDA_API_INTERVAL","batch":batch,"logical_bytes":if b==0{27672}else{3072},
                    "host_staging_ms":t.staging_ms,"enqueue_ms":t.enqueue_ms,"host_wait_ms":t.wait_ms,"cuda_event_ms":ms,
                    "timing_flags":"CU_EVENT_DEFAULT","scope":"three batch boundaries; request includes five sentinel resets",
                    "resolution":"finite; zero may be resolution limited; includes intervening work; no bandwidth inference",
                    "initial_staging":"sum of six initialized host payload writes; allocation excluded"});
            }
            Op::Kernel(_, k) => self.kernel(*k)?,
            Op::KernelWait(_, _) => self.stream()?.synchronize().map_err(err)?,
            Op::D2h(seq) => {
                let stream = self.stream()?.clone();
                let y = self.device[9].as_ref().ok_or("missing y")?.floats()?;
                let out = &mut self.output[usize::from(*seq)];
                if y.len() != out.len() {
                    return Err("D2H length".into());
                }
                r.bytes = (y.len() * 4) as u64;
                r.source = Some(209);
                r.destination = Some(300 + u64::from(*seq));
                // Stable initialized Box allocation remains in this Owner even
                // on submit/record/wait failure; no temporary host Vec escapes.
                stream.memcpy_dtoh(y, &mut out[..]).map_err(err)?;
            }
            Op::Read(seq) => {
                if !self.completed[6 + usize::from(*seq)] {
                    return Err("pending host output".into());
                }
                r.output = Some(self.output[usize::from(*seq)].to_vec());
            }
        }
        Ok(r)
    }
    fn check(&mut self) -> Result<(), String> {
        if let Some(ctx) = &self.ctx {
            ctx.check_err().map_err(err)?
        }
        Ok(())
    }
    fn drain(&mut self) -> Result<(), String> {
        if let Some(ctx) = &self.ctx {
            ctx.synchronize().map_err(err)?;
            ctx.check_err().map_err(err)?
        }
        Ok(())
    }
    fn release(&mut self) -> Result<Value, String> {
        let Some(ctx) = self.ctx.as_ref().cloned() else {
            return Ok(json!({"device":0,"pinned":0,"events":0}));
        };
        let (mut devices, mut pinned, mut events) = (0, 0, 0);
        // Check each individual destructor's deferred error before dropping the
        // next resource. No claimed recovery if cudarc's Drop itself fails.
        for v in &mut self.device {
            if let Some(v) = v.take() {
                drop(v);
                devices += 1;
                ctx.synchronize().map_err(err)?;
                ctx.check_err().map_err(err)?
            }
        }
        for v in &mut self.host {
            if let Some(v) = v.take() {
                drop(v);
                pinned += 1;
                ctx.check_err().map_err(err)?
            }
        }
        for v in &mut self.events {
            if let Some(v) = v.take() {
                drop(v);
                events += 1;
                ctx.check_err().map_err(err)?
            }
        }
        for v in &mut self.functions {
            drop(v.take());
            ctx.check_err().map_err(err)?
        }
        for v in &mut self.modules {
            drop(v.take());
            ctx.check_err().map_err(err)?
        }
        drop(self.stream.take());
        ctx.synchronize().map_err(err)?;
        ctx.check_err().map_err(err)?;
        Ok(
            json!({"device":devices,"pinned":pinned,"events":events,"context_destruction":"UNOBSERVABLE_AFTER_LAST_OWNER_DROP"}),
        )
    }
    fn simulated(&self) -> bool {
        false
    }
}

static NEXT_CONTEXT: AtomicU64 = AtomicU64::new(1);
pub(crate) fn run(prepared: Prepared, report: &mut Value) -> Result<(), String> {
    let id = NEXT_CONTEXT
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_add(1))
        .map_err(|_| "context identity overflow")?;
    report["GPU_INVOCATIONS"] = json!(1);
    report["CUDA_GPU_TEST"] = json!("STARTED");
    let mut run = Runner::new(Owner::new(id), id, &prepared.data);
    let mut result = run.execute(&prepared.data);
    if result.is_ok() {
        for seq in 0..2 {
            let actual = run.core.outputs[seq]
                .as_ref()
                .ok_or("missing final output")?;
            let bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
            if let Err(e) = verify_output(&prepared.references[seq], actual, &bits) {
                run.core.fail(e.clone());
                result = Err(e);
                break;
            }
        }
    }
    run.finish();
    report["execution"] = run.report(false);
    report["outputs"]=json!(run.core.outputs.iter().enumerate().map(|(seq,o)|json!({"sequence":seq,
        "input_sha256":prepared.data.input_ids[seq],"values":o,"f32_bits":o.as_ref().map(|v|v.iter().map(|x|x.to_bits()).collect::<Vec<_>>())})).collect::<Vec<_>>());
    if let Some(error) = &run.core.first {
        return Err(error.clone());
    }
    if run.cleanup.contains("UNCONFIRMED") {
        return Err(run.cleanup.clone());
    }
    result?;
    report["CUDA_GPU_TEST"] = json!("PASS_REQUIRES_EXTERNAL_CHECKER");
    Ok(())
}
