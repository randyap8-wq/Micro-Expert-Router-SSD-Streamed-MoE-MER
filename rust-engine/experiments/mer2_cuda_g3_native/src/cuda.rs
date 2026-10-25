//! Shared native ownership decisions; CPU injection never proves CUDA destruction.
pub(crate) mod ownership {
    use mer2_cuda_g3_s1::{require, Result, Stop};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[derive(Default)]
    pub(crate) struct State {
        pub initialization_attempted: bool,
        pub ownership_uncertain: bool,
        // A constructor has started but has not stored a returned typed owner.
        // This is possible inaccessible dependency ownership, not an observed leak.
        pub unreturned_owner_possible: bool,
        pub drained: bool,
        pub released: bool,
        pub first_error: Option<Stop>,
    }
    impl State {
        pub fn initialize_allowed(&self, context: u64) -> Result<()> {
            require(
                !self.initialization_attempted && !self.drained && !self.released && context != 0,
                "STOP_OWNERSHIP",
                "single native initialization",
            )
        }
    }

    pub(crate) trait Owned {
        fn state(&self) -> &State;
        fn state_mut(&mut self) -> &mut State;
        fn note_error(&mut self, error: &Stop, first: bool);
    }

    pub(crate) fn call<O: Owned, T>(
        owner: &mut O,
        cleanup: bool,
        f: impl FnOnce(&mut O) -> Result<T>,
    ) -> Result<T> {
        if !cleanup {
            if let Some(error) = &owner.state().first_error {
                return Err(error.clone());
            }
        }
        let result = catch_unwind(AssertUnwindSafe(|| f(owner)))
            .unwrap_or_else(|_| Err(Stop::new("STOP_PANIC", "native adapter operation panicked")));
        if let Err(error) = &result {
            let first = owner.state().first_error.is_none();
            if first {
                owner.state_mut().first_error = Some(error.clone());
            }
            owner.note_error(error, first);
        }
        result
    }

    /// Store the returned typed owner BEFORE checking a deferred error. The store
    /// closure must not perform any fallible work after storing that owner.
    pub(crate) fn retain<O: Owned>(
        owner: &mut O,
        create_and_store: impl FnOnce(&mut O) -> Result<()>,
        check: impl FnOnce(&O) -> Result<()>,
    ) -> Result<()> {
        owner.state_mut().ownership_uncertain = true;
        owner.state_mut().unreturned_owner_possible = true;
        create_and_store(owner)?;
        owner.state_mut().unreturned_owner_possible = false;
        check(owner)?;
        owner.state_mut().ownership_uncertain = false;
        Ok(())
    }

    pub(crate) fn drain<O: Owned>(
        owner: &mut O,
        matching_context: bool,
        synchronize_and_check: impl FnOnce(&mut O) -> Result<()>,
    ) -> Result<()> {
        require(
            !owner.state().drained && !owner.state().released && matching_context,
            "STOP_DRAIN_UNCONFIRMED",
            "single drain of matching native context",
        )?;
        synchronize_and_check(owner)?;
        require(
            !owner.state().ownership_uncertain,
            "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED",
            "Cudarc resource creation may have allocated without returning an owner",
        )?;
        owner.state_mut().drained = true;
        Ok(())
    }

    pub(crate) fn release<O: Owned>(
        owner: &mut O,
        matching_context: bool,
        no_pending_work: bool,
        release_and_check: impl FnOnce(&mut O) -> Result<()>,
    ) -> Result<()> {
        require(
            owner.state().drained
                && !owner.state().released
                && !owner.state().ownership_uncertain
                && matching_context
                && no_pending_work,
            "STOP_RELEASE_UNCONFIRMED",
            "checked drain before single native release",
        )?;
        release_and_check(owner)?;
        owner.state_mut().released = true;
        Ok(())
    }

    /// An error or panic stops at the first attempted destructor/check. Remaining
    /// slots stay owned. The caller's guarded Runner settles and quarantines them.
    pub(crate) fn release_slots<T>(
        slots: &mut [Option<T>],
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        for slot in slots {
            drop(slot.take());
            check()?;
        }
        Ok(())
    }

    pub(crate) fn dispose<O: Owned>(owner: Box<O>, quarantine: impl FnOnce(&O)) {
        if owner.state().released || !owner.state().initialization_attempted {
            drop(owner);
        } else {
            quarantine(&owner);
            std::mem::forget(owner);
        }
    }
}

#[cfg(feature = "cuda")]
pub use native::{NativeCudaAdapter, NativeEvidence};

#[cfg(feature = "cuda")]
mod native {
use super::ownership::{self, Owned, State};
// The only unsafe boundary. All CUDA resources stay in Owner through checked cleanup.
use crate::KIND;
use cudarc::{
    driver::{
        CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
        PushKernelArg,
    },
    nvrtc::Ptx,
};
use mer2_cuda_g3_s1::{
    abi::{
        Adapter, CleanupReceipt, Completion, ExpertWeights, FfnStages, InitReceipt, Lease,
        Operation, OperationKind, Resident, Submission, Vector, OPERATIONS,
    },
    host_token::StepOutput,
    identity::sha256,
    require,
    state::{Phase, Report},
    Result, Stop, D, DFF, EXPERTS, LAYERS, Q4_BYTES,
};
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::BTreeSet,
    rc::Rc,
    sync::Arc,
};

const SYMBOLS: [&str; 3] = ["dequantize_mul_mat_vec_q4_0_cuda", "usilu_f32", "bmul_f32"];
const RESIDENTS: usize = LAYERS * EXPERTS;

#[derive(Default)]
struct Journal {
    context: u64,
    initialization: Option<Value>,
    attempted: Vec<Operation>,
    submitted: Vec<Operation>,
    completed: Vec<Completion>,
    fences: Vec<Value>,
    kernel_launches: usize,
    first_error: Option<Stop>,
    secondary_errors: Vec<Stop>,
    cleanup: &'static str,
}

/// Read-only native evidence handle. It owns no CUDA resource and cannot forge a completion.
#[derive(Clone)]
pub struct NativeEvidence(Rc<RefCell<Journal>>);
impl NativeEvidence {
    pub fn json(&self) -> Value {
        let j = self.0.borrow();
        let operation = |op: &Operation| {
            json!({"lease":op.lease.json(),
            "sequence":op.sequence,"kind":format!("{:?}",op.kind)})
        };
        json!({"kind":KIND,"initialization":j.initialization,
            "attempted":j.attempted.iter().map(&operation).collect::<Vec<_>>(),
            "submitted":j.submitted.iter().map(operation).collect::<Vec<_>>(),
            "fenced_operations":j.fences,"kernel_launches_submitted":j.kernel_launches,
            "first_error":j.first_error.as_ref().map(ToString::to_string),
            "secondary_errors":j.secondary_errors.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "cleanup":j.cleanup,
            "context_destruction":"UNOBSERVABLE_AFTER_LAST_OWNER_DROP",
            "qualification":"NONE; native API observations require independent validation"})
    }

    pub fn verify_step(&self, output: &StepOutput) -> Result<()> {
        let j = self.0.borrow();
        require(
            j.initialization.is_some() && output.completions.len() == LAYERS * 2,
            "STOP_NATIVE_PROVENANCE",
            "initialized native owner and four FFNs",
        )?;
        for ffn in &output.completions {
            require(
                ffn.lease().ticket == output.ticket
                    && ffn.lease().context_identity == j.context
                    && ffn.completions().len() == 7,
                "STOP_NATIVE_PROVENANCE",
                "native token/context binding",
            )?;
            for completion in ffn.completions() {
                require(
                    j.completed.iter().filter(|c| *c == completion).count() == 1,
                    "STOP_NATIVE_PROVENANCE",
                    "exact unique native completion including stage bits",
                )?;
            }
            require(
                ffn.completions()
                    .last()
                    .and_then(|c| c.output_sha256.as_ref())
                    == Some(&ffn.stages().digest()),
                "STOP_NATIVE_PROVENANCE",
                "native fenced stage digest",
            )?;
        }
        Ok(())
    }

    pub fn verify_chain(&self, report: &Report) -> Result<()> {
        if let Some(error) = &report.first_error {
            return Err(error.clone());
        }
        let j = self.0.borrow();
        if let Some(error) = &j.first_error {
            return Err(error.clone());
        }
        require(
            report.phase == Phase::Released
                && report.context_identity == j.context
                && report.counts.consumed_ffns == 24
                && report.counts.attempted == 168
                && report.counts.submitted == 168
                && report.counts.completed == 168
                && j.attempted.len() == 168
                && j.submitted.len() == 168
                && j.completed.len() == 168
                && j.kernel_launches == 120
                && j.cleanup == "CHECKED_NATIVE_DRAIN_AND_RESOURCE_RELEASE",
            "STOP_NATIVE_PROVENANCE",
            "six tokens, 24 FFNs, 120 launches, 168 fenced operations and cleanup",
        )
    }
}

struct Pending {
    operation: Operation,
    recorded: bool,
}

struct Owner {
    ordinal: usize,
    context_id: u64,
    ctx: Option<Arc<CudaContext>>,
    stream: Option<Arc<CudaStream>>,
    modules: [Option<Arc<CudaModule>>; 3],
    functions: [Option<CudaFunction>; 3],
    events: [Option<CudaEvent>; 8], // initialization, then the seven operation boundaries
    residents: Vec<Resident>,
    weights: Vec<ExpertWeights>, // stable owned H2D sources, including on an error
    device_weights: [Option<CudaSlice<u8>>; RESIDENTS * 3],
    layout_host: Box<[usize; 3]>,
    layout: Option<CudaSlice<usize>>,
    input_host: Box<[f32; D]>,
    sentinel: Box<[f32; D]>,
    scratch: [Option<CudaSlice<f32>>; 6], // input, gate, up, SiLU, product, down
    readback: [Box<[f32; D]>; 5],
    active: Option<Lease>,
    pending: Option<Pending>,
    next_sequence: usize,
    used_events: BTreeSet<u64>,
    initialized: bool,
    ownership: State,
    journal: Rc<RefCell<Journal>>,
}

/// Construct without touching the CUDA driver. Use only through S1's guarded Runner.
pub struct NativeCudaAdapter {
    owner: Option<Box<Owner>>,
}
impl NativeCudaAdapter {
    pub fn new(ordinal: usize) -> (Self, NativeEvidence) {
        let journal = Rc::new(RefCell::new(Journal {
            cleanup: "NOT_ATTEMPTED",
            ..Journal::default()
        }));
        let owner = Owner {
            ordinal,
            context_id: 0,
            ctx: None,
            stream: None,
            modules: std::array::from_fn(|_| None),
            functions: std::array::from_fn(|_| None),
            events: std::array::from_fn(|_| None),
            residents: Vec::new(),
            weights: Vec::new(),
            device_weights: std::array::from_fn(|_| None),
            layout_host: Box::new([DFF, 1, 1]),
            layout: None,
            input_host: Box::new([0.0; D]),
            sentinel: Box::new([f32::NAN; D]),
            scratch: std::array::from_fn(|_| None),
            readback: std::array::from_fn(|_| Box::new([f32::NAN; D])),
            active: None,
            pending: None,
            next_sequence: 0,
            used_events: BTreeSet::new(),
            initialized: false,
            ownership: State::default(),
            journal: journal.clone(),
        };
        (
            Self {
                owner: Some(Box::new(owner)),
            },
            NativeEvidence(journal),
        )
    }

    fn call<T>(&mut self, cleanup: bool, f: impl FnOnce(&mut Owner) -> Result<T>) -> Result<T> {
        let o = self
            .owner
            .as_mut()
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "missing native owner"))?;
        ownership::call(o.as_mut(), cleanup, f)
    }
}

impl Drop for NativeCudaAdapter {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            ownership::dispose(owner, |owner| {
                // Runner normally forgets us on quarantine; direct Drop is conservative too.
                owner.journal.borrow_mut().cleanup = "QUARANTINED_UNCHECKED_OWNER_DROP";
            });
        }
    }
}

fn cuda_error(error: impl std::fmt::Display) -> Stop {
    Stop::new("STOP_CUDA", error.to_string())
}
fn present<T>(value: &Option<T>) -> Result<&T> {
    value
        .as_ref()
        .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "missing retained CUDA resource"))
}
fn present_mut<T>(value: &mut Option<T>) -> Result<&mut T> {
    value
        .as_mut()
        .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "missing retained CUDA resource"))
}

impl Owned for Owner {
    fn state(&self) -> &State { &self.ownership }
    fn state_mut(&mut self) -> &mut State { &mut self.ownership }
    fn note_error(&mut self, error: &Stop, first: bool) {
        let mut journal = self.journal.borrow_mut();
        if first {
            journal.first_error = Some(error.clone());
        } else {
            journal.secondary_errors.push(error.clone());
        }
    }
}

impl Owner {
    fn context(&self) -> Result<&Arc<CudaContext>> {
        present(&self.ctx)
    }
    fn stream(&self) -> Result<&Arc<CudaStream>> {
        present(&self.stream)
    }
    fn check(&self) -> Result<()> {
        if let Some(ctx) = &self.ctx {
            ctx.check_err().map_err(cuda_error)?;
        }
        Ok(())
    }

    fn initialize(
        &mut self,
        context: u64,
        residents: &[Resident],
        weights: &[ExpertWeights],
    ) -> Result<InitReceipt> {
        self.ownership.initialize_allowed(context)?;
        require(
            cfg!(target_endian = "little")
                && usize::BITS == 64
                && D == 64
                && DFF == 64
                && Q4_BYTES == 2304
                && RESIDENTS == 8,
            "STOP_ABI",
            "64-bit little-endian host; frozen 64x64 geometry",
        )?;
        require(
            residents.len() == RESIDENTS && weights.len() == RESIDENTS,
            "STOP_OWNERSHIP",
            "all eight experts",
        )?;
        for (slot, (r, w)) in residents.iter().zip(weights).enumerate() {
            require(
                r.slot == slot
                    && r.generation != 0
                    && r.key == *w.key()
                    && r.key.layer == slot / EXPERTS
                    && r.key.expert as usize == slot % EXPERTS
                    && w.matrices().iter().all(|m| m.bytes().len() == Q4_BYTES),
                "STOP_EXPERT",
                "resident layer/expert/digest/generation and Q4 payload",
            )?;
        }
        self.context_id = context;
        self.residents = residents.to_vec();
        self.weights = weights.to_vec();
        self.ownership.initialization_attempted = true;
        self.journal.borrow_mut().context = context;
        // A Cudarc constructor may allocate before returning Err or panicking.
        // Keep uncertainty latched until the returned owner AND deferred check succeed.
        macro_rules! retain {
            ($owner:ident => $place:expr, $create:expr) => {{
                ownership::retain(self, |$owner| {
                    $place = Some($create.map_err(cuda_error)?);
                    Ok(())
                }, Owner::check)?;
            }};
        }
        retain!(o => o.ctx, CudaContext::new(o.ordinal));
        let ctx = self.context()?.clone();
        ownership::retain(self, |o| {
            o.stream = Some(ctx.default_stream());
            Ok(())
        }, Owner::check)?;
        let stream = self.stream()?.clone();
        let ptx = [
            candle_kernels::QUANTIZED.ptx(),
            candle_kernels::UNARY.ptx(),
            candle_kernels::BINARY.ptx(),
        ];
        for i in 0..3 {
            retain!(o => o.modules[i], ctx.load_module(Ptx::from_src(ptx[i])));
            retain!(
                o => o.functions[i],
                present(&o.modules[i])?.load_function(SYMBOLS[i])
            );
        }
        for i in 0..8 {
            retain!(o => o.events[i], ctx.new_event(None));
        }
        for i in 0..RESIDENTS * 3 {
            // SAFETY: bounded primitive byte allocation, retained before any H2D.
            // Every byte is written before kernels can read it.
            retain!(o => o.device_weights[i], unsafe {
                stream.alloc::<u8>(Q4_BYTES)
            });
            let source = self.weights[i / 3].matrices()[i % 3].bytes();
            stream
                .memcpy_htod(source, present_mut(&mut self.device_weights[i])?)
                .map_err(cuda_error)?;
            self.check()?;
        }
        // SAFETY: three usize elements initialized by the owned layout H2D below.
        retain!(o => o.layout, unsafe { stream.alloc::<usize>(3) });
        stream
            .memcpy_htod(&self.layout_host[..], present_mut(&mut self.layout)?)
            .map_err(cuda_error)?;
        self.check()?;
        for i in 0..6 {
            // SAFETY: no device read occurs until UploadInput writes input and all sentinels.
            retain!(o => o.scratch[i], unsafe { stream.alloc::<f32>(D) });
        }
        present(&self.events[0])?
            .record(&stream)
            .map_err(cuda_error)?;
        self.check()?;
        stream.synchronize().map_err(cuda_error)?;
        present(&self.events[0])?
            .synchronize()
            .map_err(cuda_error)?;
        self.check()?;
        self.initialized = true;
        self.journal.borrow_mut().initialization = Some(json!({
            "context_identity":context,"cuda_context_pointer":ctx.cu_ctx() as usize,
            "device_ordinal":ctx.ordinal(),"residency_fence_confirmed":true,
            "resident_weight_payload_bytes":RESIDENTS * 3 * Q4_BYTES,
            "physical_vram_measurement":null,
            "residents":self.residents.iter().zip(&self.weights).map(|(r,w)|json!({
                "layer":r.key.layer,"expert":r.key.expert,"weight_digest":r.key.weight_digest,
                "matrix_digests":w.matrix_digests(),"slot":r.slot,"generation":r.generation})).collect::<Vec<_>>(),
            "ptx_sha256":ptx.map(|p| sha256(p.as_bytes())),"kernel_symbols":SYMBOLS,
            "geometry":{"d_model":D,"d_ff":DFF,"q4_matrix_bytes":Q4_BYTES},
            "host_transfers":"retained ordinary host slices; no pinned-transfer or PCIe performance claim"}));
        Ok(InitReceipt {
            context_identity: context,
            residents: self.residents.clone(),
            confirmed: true,
        })
    }

    fn submit(&mut self, op: &Operation, input: &Vector) -> Result<Submission> {
        require(
            self.initialized && !self.ownership.drained && !self.ownership.released && self.pending.is_none(),
            "STOP_OWNERSHIP",
            "initialized owner with no in-flight operation",
        )?;
        let l = &op.lease;
        let r = self
            .residents
            .get(l.slot)
            .ok_or_else(|| Stop::new("STOP_EXPERT", "native resident slot"))?;
        let w = self
            .weights
            .get(l.slot)
            .ok_or_else(|| Stop::new("STOP_EXPERT", "native resident weights"))?;
        let weight = f32::from_bits(l.selected_weight_bits);
        require(
            l.context_identity == self.context_id
                && l.event_identity != 0
                && l.slot == r.slot
                && l.generation == r.generation
                && l.layer == r.key.layer
                && l.expert == r.key.expert
                && l.weight_digest == r.key.weight_digest
                && l.weight_sha256 == w.matrix_digests()
                && l.rank < 2
                && weight.is_finite()
                && (0.0..=1.0).contains(&weight)
                && input.digest() == l.input_sha256,
            "STOP_OWNERSHIP",
            "exact native lease/input/resident identity",
        )?;
        require(
            OPERATIONS.get(op.sequence) == Some(&op.kind),
            "STOP_OWNERSHIP",
            "native operation sequence/kind",
        )?;
        if op.sequence == 0 {
            require(
                self.active.is_none() && self.used_events.insert(l.event_identity),
                "STOP_OWNERSHIP",
                "scratch still leased or stale FFN event identity",
            )?;
            self.active = Some(l.clone());
            self.next_sequence = 0;
            self.input_host.copy_from_slice(&input.values());
            for destination in &mut self.readback {
                destination.fill(f32::NAN);
            }
        }
        require(
            self.active.as_ref() == Some(l) && self.next_sequence == op.sequence,
            "STOP_OWNERSHIP",
            "same full lease and strictly ordered completions",
        )?;
        // Record pending ownership BEFORE the first fallible enqueue; Err never releases buffers.
        self.pending = Some(Pending {
            operation: op.clone(),
            recorded: false,
        });
        self.journal.borrow_mut().attempted.push(op.clone());
        let stream = self.stream()?.clone();
        match op.kind {
            OperationKind::UploadInput => {
                stream
                    .memcpy_htod(&self.input_host[..], present_mut(&mut self.scratch[0])?)
                    .map_err(cuda_error)?;
                self.check()?;
                for i in 1..6 {
                    stream
                        .memcpy_htod(&self.sentinel[..], present_mut(&mut self.scratch[i])?)
                        .map_err(cuda_error)?;
                    self.check()?;
                }
            }
            OperationKind::Readback => {
                // S1 requires all five raw stages, so read all five only after Down is fenced.
                for i in 0..5 {
                    stream
                        .memcpy_dtoh(present(&self.scratch[i + 1])?, &mut self.readback[i][..])
                        .map_err(cuda_error)?;
                    self.check()?;
                }
            }
            _ => {
                self.kernel(op)?;
                self.journal.borrow_mut().kernel_launches += 1;
                self.check()?;
            }
        }
        present(&self.events[op.sequence + 1])?
            .record(&stream)
            .map_err(cuda_error)?;
        self.check()?;
        self.pending
            .as_mut()
            .ok_or_else(|| Stop::new("STOP_OWNERSHIP", "pending lost"))?
            .recorded = true;
        self.journal.borrow_mut().submitted.push(op.clone());
        Ok(Submission {
            operation: op.clone(),
        })
    }

    fn kernel(&mut self, op: &Operation) -> Result<()> {
        let stream = self.stream()?.clone();
        let output = op.sequence; // Gate..Down are scratch indices 1..5.
        require((1..=5).contains(&output), "STOP_ABI", "five kernel stages")?;
        let (prefix, suffix) = self.scratch.split_at_mut(output);
        let dst = present_mut(&mut suffix[0])?;
        require(dst.len() == D, "STOP_SHAPE", "64-wide device output")?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DFF as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        match op.kind {
            OperationKind::Gate | OperationKind::Up | OperationKind::Down => {
                let matrix = match op.kind {
                    OperationKind::Gate => 0,
                    OperationKind::Up => 1,
                    _ => 2,
                };
                let weights = present(&self.device_weights[op.lease.slot * 3 + matrix])?;
                let input = present(&prefix[if op.kind == OperationKind::Down { 4 } else { 0 }])?;
                require(
                    weights.len() == Q4_BYTES && input.len() == D && Q4_BYTES == D * (D / 32) * 18,
                    "STOP_SHAPE",
                    "Q4 row geometry",
                )?;
                let (cols, rows) = (D as i32, D as i32);
                // SAFETY: pinned vendor ABI (void*,float*,float*,int,int), f32 dfloat.
                // Allocator-aligned whole Q4 buffer (64 rows * two 18-byte blocks),
                // one 32-lane warp per row, 64 cols divisible by the kernel's 64 stride.
                // Exclusive suffix proves output disjoint from input; Owner retains all.
                unsafe {
                    stream
                        .launch_builder(present(&self.functions[0])?)
                        .arg(weights)
                        .arg(input)
                        .arg(dst)
                        .arg(&cols)
                        .arg(&rows)
                        .launch(LaunchConfig {
                            grid_dim: (D as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                }
                .map_err(cuda_error)?;
            }
            OperationKind::Silu => {
                let (numel, ndims) = (DFF, 1usize);
                // SAFETY: size_t,size_t,size_t*,float*,float*. Non-null [64,1,1]
                // layout, 64 f32 elements, disjoint gate/output, same context/stream.
                unsafe {
                    stream
                        .launch_builder(present(&self.functions[1])?)
                        .arg(&numel)
                        .arg(&ndims)
                        .arg(present(&self.layout)?)
                        .arg(present(&prefix[1])?)
                        .arg(dst)
                        .launch(config)
                }
                .map_err(cuda_error)?;
            }
            OperationKind::Multiply => {
                let (numel, ndims) = (DFF, 1usize);
                // SAFETY: binary ABI adds rhs before output; [64,1,1] gives both
                // contiguous strides. SiLU/up/output are separate retained slices.
                unsafe {
                    stream
                        .launch_builder(present(&self.functions[2])?)
                        .arg(&numel)
                        .arg(&ndims)
                        .arg(present(&self.layout)?)
                        .arg(present(&prefix[3])?)
                        .arg(present(&prefix[2])?)
                        .arg(dst)
                        .launch(config)
                }
                .map_err(cuda_error)?;
            }
            _ => return Err(Stop::new("STOP_ABI", "not a kernel operation")),
        }
        Ok(())
    }

    fn complete(&mut self, op: &Operation) -> Result<Completion> {
        require(
            !self.ownership.drained
                && !self.ownership.released
                && self
                    .pending
                    .as_ref()
                    .is_some_and(|p| p.operation == *op && p.recorded),
            "STOP_RECEIPT",
            "exact submitted operation with recorded native event",
        )?;
        let event = present(&self.events[op.sequence + 1])?;
        self.stream()?.synchronize().map_err(cuda_error)?;
        event.synchronize().map_err(cuda_error)?;
        self.check()?;
        let event_pointer = event.cu_event() as usize;
        let stages = if op.kind == OperationKind::Readback {
            let values = self
                .readback
                .iter()
                .map(|v| Vector::new(&v[..]))
                .collect::<Result<Vec<_>>>()?;
            Some(FfnStages(values.try_into().map_err(|_| {
                Stop::new("STOP_OUTPUT", "five finite 64-wide native stages")
            })?))
        } else {
            None
        };
        let output_sha256 = stages.as_ref().map(FfnStages::digest);
        let completion = Completion {
            operation: op.clone(),
            fence_confirmed: true,
            stages,
            output_sha256,
        };
        {
            let mut j = self.journal.borrow_mut();
            j.fences
                .push(json!({"lease":op.lease.json(),"sequence":op.sequence,
                "operation":format!("{:?}",op.kind),"cuda_event_pointer":event_pointer,
                "stream_synchronize":"OK","event_synchronize":"OK","deferred_check":"OK",
                "stage_digest":completion.output_sha256}));
            j.completed.push(completion.clone());
        }
        self.pending = None;
        self.next_sequence += 1;
        if op.kind == OperationKind::Readback {
            self.active = None;
        }
        Ok(completion)
    }

    fn drain(&mut self, context: u64) -> Result<CleanupReceipt> {
        // Mark quarantine first: a panic/Err cannot leave an optimistic cleanup label.
        self.journal.borrow_mut().cleanup = "QUARANTINED_DRAIN_UNCONFIRMED";
        let matching_context = self.context_id == 0 || self.context_id == context;
        ownership::drain(self, matching_context, |o| {
            if let Some(ctx) = &o.ctx {
                ctx.synchronize().map_err(cuda_error)?;
            }
            o.check()
        })?;
        self.context_id = context;
        // A drain does not manufacture missing operation completions or readable outputs.
        self.pending = None;
        self.active = None;
        self.journal.borrow_mut().cleanup = "CHECKED_NATIVE_DRAIN";
        Ok(CleanupReceipt {
            context_identity: context,
            all_work_complete: true,
            ownership_confirmed: true,
        })
    }

    fn release(&mut self, context: u64) -> Result<CleanupReceipt> {
        self.journal.borrow_mut().cleanup = "QUARANTINED_RELEASE_UNCONFIRMED";
        let matching_context = context == self.context_id;
        let no_pending_work = self.pending.is_none() && self.active.is_none();
        ownership::release(self, matching_context, no_pending_work, |o| {
            if let Some(ctx) = o.ctx.as_ref().cloned() {
                // Stop at the first destructor/deferred error. Remaining owners stay quarantined.
                ownership::release_slots(&mut o.device_weights, || {
                    ctx.synchronize().map_err(cuda_error)?;
                    ctx.check_err().map_err(cuda_error)
                })?;
                ownership::release_slots(&mut o.scratch, || {
                    ctx.synchronize().map_err(cuda_error)?;
                    ctx.check_err().map_err(cuda_error)
                })?;
                ownership::release_slots(std::slice::from_mut(&mut o.layout), || {
                    ctx.synchronize().map_err(cuda_error)?;
                    ctx.check_err().map_err(cuda_error)
                })?;
                ownership::release_slots(&mut o.events, || ctx.check_err().map_err(cuda_error))?;
                ownership::release_slots(&mut o.functions, || ctx.check_err().map_err(cuda_error))?;
                ownership::release_slots(&mut o.modules, || ctx.check_err().map_err(cuda_error))?;
                ownership::release_slots(std::slice::from_mut(&mut o.stream), || {
                    ctx.synchronize().map_err(cuda_error)?;
                    ctx.check_err().map_err(cuda_error)
                })?;
            }
            o.weights.clear();
            o.residents.clear();
            // Retain ctx for Runner's post-release check. Last context drop is unobservable.
            Ok(())
        })?;
        self.journal.borrow_mut().cleanup = "CHECKED_NATIVE_DRAIN_AND_RESOURCE_RELEASE";
        Ok(CleanupReceipt {
            context_identity: context,
            all_work_complete: true,
            ownership_confirmed: true,
        })
    }
}

impl Adapter for NativeCudaAdapter {
    fn initialize(
        &mut self,
        context: u64,
        residents: &[Resident],
        weights: &[ExpertWeights],
    ) -> Result<InitReceipt> {
        self.call(false, |o| o.initialize(context, residents, weights))
    }
    fn submit(&mut self, operation: &Operation, input: &Vector) -> Result<Submission> {
        self.call(false, |o| o.submit(operation, input))
    }
    fn complete(&mut self, operation: &Operation) -> Result<Completion> {
        self.call(false, |o| o.complete(operation))
    }
    fn check(&mut self) -> Result<()> {
        self.call(true, |o| o.check())
    }
    fn drain(&mut self, context: u64) -> Result<CleanupReceipt> {
        self.call(true, |o| o.drain(context))
    }
    fn release(&mut self, context: u64) -> Result<CleanupReceipt> {
        self.call(true, |o| o.release(context))
    }
}
}
