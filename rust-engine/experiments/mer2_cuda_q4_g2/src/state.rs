//! Private, statically dispatched execution engine shared by CUDA and fake CPU.
#![cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
use crate::{Data, BASE};
use serde_json::{json, Value};
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Context,
    Stream,
    Module,
    Function,
    Pinned,
    Device,
    Event,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Token {
    pub context: u64,
    pub slot: u8,
    pub generation: u64,
    pub sequence: u8,
    pub event: u8,
    pub recorded: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Create(Kind, u8),
    Install,
    Begin(u8),
    Stage(u8),
    Record(Token),
    H2d { src: u8, dst: u8, bytes: u64 },
    Fence(Token),
    Measure(u8),
    Kernel(u8, u8),
    KernelWait(u8, u8),
    D2h(u8),
    Read(u8),
    Retire,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Empty,
    Uploading,
    Resident,
    InFlight(u8),
    Retiring,
    Retired,
    Poisoned,
}

pub(crate) fn token(context: u64, event: u8) -> Token {
    Token {
        context,
        slot: 0,
        generation: 1,
        sequence: match event {
            0 | 1 => 2,
            2 | 3 | 6 => 0,
            _ => 1,
        },
        event,
        recorded: true,
    }
}

pub(crate) fn payload(kind: Kind, id: u8) -> Option<u64> {
    match (kind, id) {
        (Kind::Pinned | Kind::Device, 0..=2) => Some(9216),
        (Kind::Pinned | Kind::Device, 3) => Some(24),
        (Kind::Pinned, 4..=5) | (Kind::Device, 4..=9) => Some(512),
        (Kind::Event, 0..=7) => Some(0),
        (Kind::Module | Kind::Function, 0..=2) | (Kind::Context | Kind::Stream, 0) => Some(0),
        _ => None,
    }
}

pub(crate) fn expected() -> Value {
    json!({"h2d_bytes":33816,"d2h_bytes":1024,
    "h2d_calls":16,"d2h_calls":2,"kernels":10,"device_allocs":10,"device_bytes":30744,
    "pinned_allocs":6,"pinned_bytes":28696,"events":8,"intermediate_d2h":0,
    "between_a_b_allocs_frees":0,"between_a_b_weight_layout_copies":0})
}

pub(crate) fn plan(context: u64) -> Vec<Op> {
    let mut p = vec![Op::Create(Kind::Context, 0), Op::Create(Kind::Stream, 0)];
    for (kind, n) in [
        (Kind::Module, 3),
        (Kind::Function, 3),
        (Kind::Pinned, 6),
        (Kind::Device, 10),
        (Kind::Event, 8),
    ] {
        for i in 0..n {
            p.push(Op::Create(kind, i));
        }
    }
    p.push(Op::Install);
    p.push(Op::Record(token(context, 0)));
    for i in 0..4 {
        p.push(Op::H2d {
            src: i,
            dst: i,
            bytes: payload(Kind::Device, i).unwrap(),
        });
    }
    p.push(Op::Record(token(context, 1)));
    p.push(Op::Fence(token(context, 1)));
    p.push(Op::Measure(0));
    for seq in 0..2 {
        p.push(Op::Begin(seq));
        p.push(Op::Stage(seq));
        p.push(Op::Record(token(context, 2 + seq * 2)));
        p.push(Op::H2d {
            src: 4,
            dst: 4,
            bytes: 512,
        });
        for dst in 5..10 {
            p.push(Op::H2d {
                src: 5,
                dst,
                bytes: 512,
            });
        }
        p.push(Op::Record(token(context, 3 + seq * 2)));
        p.push(Op::Fence(token(context, 3 + seq * 2)));
        p.push(Op::Measure(seq + 1));
        for k in 0..5 {
            p.push(Op::Kernel(seq, k));
            p.push(Op::KernelWait(seq, k));
        }
        p.push(Op::D2h(seq));
        p.push(Op::Record(token(context, 6 + seq)));
        p.push(Op::Fence(token(context, 6 + seq)));
        p.push(Op::Read(seq));
    }
    p.push(Op::Retire);
    p
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Counts {
    pub h2d: u64,
    pub d2h: u64,
    pub h2d_calls: u64,
    pub d2h_calls: u64,
    pub kernels: u64,
    pub device: u64,
    pub pinned: u64,
    pub device_bytes: u64,
    pub pinned_bytes: u64,
    pub events: u64,
}
impl Counts {
    fn json(&self) -> Value {
        json!({"h2d_bytes":self.h2d,"d2h_bytes":self.d2h,
    "h2d_calls":self.h2d_calls,"d2h_calls":self.d2h_calls,"kernels":self.kernels,
    "device_allocs":self.device,"pinned_allocs":self.pinned,"device_bytes":self.device_bytes,
    "pinned_bytes":self.pinned_bytes,"events":self.events})
    }
}
fn add(dst: &mut u64, n: u64) -> Result<(), String> {
    *dst = dst
        .checked_add(n)
        .ok_or("STOP_OWNERSHIP: counter overflow")?;
    Ok(())
}

#[derive(Debug, Default)]
pub(crate) struct Receipt {
    pub bytes: u64,
    pub source: Option<u64>,
    pub destination: Option<u64>,
    pub token: Option<Token>,
    pub output: Option<Vec<f32>>,
    pub detail: Value,
}
pub(crate) trait Adapter {
    fn execute(&mut self, op: &Op, data: &Data) -> Result<Receipt, String>;
    fn check(&mut self) -> Result<(), String>;
    fn drain(&mut self) -> Result<(), String>;
    fn release(&mut self) -> Result<Value, String>;
    fn simulated(&self) -> bool;
}

pub(crate) struct Engine {
    pub phase: Phase,
    pub cursor: usize,
    pub schedule: Vec<Op>,
    pub submitted: Counts,
    pub completed: Counts,
    pub first: Option<String>,
    pub secondary: Vec<String>,
    pub trace: Vec<Value>,
    pub outputs: [Option<Vec<f32>>; 2],
    pub recorded: [bool; 8],
    pub consumed: [bool; 8],
    pub pending_h2d: u64,
    pub pending_h2d_calls: u64,
    pub pending_d2h: u64,
    pub next_sequence: u8,
    pub context: u64,
    pub expert: String,
    pub inputs: [String; 2],
}
impl Engine {
    fn new(context: u64, data: &Data) -> Self {
        Self {
            phase: Phase::Empty,
            cursor: 0,
            schedule: plan(context),
            submitted: Counts::default(),
            completed: Counts::default(),
            first: None,
            secondary: vec![],
            trace: vec![],
            outputs: [None, None],
            recorded: [false; 8],
            consumed: [false; 8],
            pending_h2d: 0,
            pending_h2d_calls: 0,
            pending_d2h: 0,
            next_sequence: 0,
            context,
            expert: data.expert.clone(),
            inputs: data.input_ids.clone(),
        }
    }
    pub fn fail(&mut self, e: String) {
        if self.first.is_none() {
            self.first = Some(e)
        } else {
            self.secondary.push(e)
        };
        self.phase = Phase::Poisoned;
    }
    fn guard(&self, op: &Op, data: &Data) -> Result<(), String> {
        if matches!(self.phase, Phase::Poisoned | Phase::Retired)
            || self.schedule.get(self.cursor) != Some(op)
            || self.expert != data.expert
            || self.inputs != data.input_ids
        {
            return Err("STOP_OWNERSHIP: terminal, out-of-order or identity mismatch".into());
        }
        match op {
            Op::Install if self.phase != Phase::Empty => {
                return Err("STOP_OWNERSHIP: busy install".into())
            }
            Op::Begin(seq) if self.phase != Phase::Resident || *seq != self.next_sequence => {
                return Err("STOP_OWNERSHIP: busy input refill".into())
            }
            Op::Record(t) | Op::Fence(t) => {
                if t.event >= 8
                    || *t != token(self.context, t.event)
                    || self.consumed[t.event as usize]
                {
                    return Err("STOP_OWNERSHIP: stale completion identity".into());
                }
                if matches!(op, Op::Record(_)) == self.recorded[t.event as usize] {
                    return Err("STOP_OWNERSHIP: duplicate/unrecorded event".into());
                }
            }
            Op::Read(seq)
                if self.phase != Phase::Resident
                    || self.next_sequence != *seq + 1
                    || self.pending_d2h != 0 =>
            {
                return Err("STOP_OWNERSHIP: pending host output".into())
            }
            Op::Retire
                if self.phase != Phase::Resident
                    || self.next_sequence != 2
                    || self.pending_h2d != 0
                    || self.pending_d2h != 0 =>
            {
                return Err("STOP_OWNERSHIP: busy retirement".into())
            }
            _ => {}
        }
        Ok(())
    }
    /// Submission counters come from the adapter receipt BEFORE the deferred
    /// error check. Completion counters advance only AFTER that check succeeds.
    fn submitted(&mut self, op: &Op, r: &Receipt) -> Result<(), String> {
        match op {
            Op::Create(kind, id) => {
                if Some(r.bytes) != payload(*kind, *id) {
                    return Err("STOP_OWNERSHIP: allocation payload receipt".into());
                }
                let expected_id = match kind {
                    Kind::Pinned => 100 + u64::from(*id),
                    Kind::Device => 200 + u64::from(*id),
                    Kind::Event => 400 + u64::from(*id),
                    _ => 0,
                };
                if expected_id != 0 && r.destination != Some(expected_id) {
                    return Err("STOP_OWNERSHIP: allocation identity".into());
                }
                match kind {
                    Kind::Pinned => {
                        add(&mut self.submitted.pinned, 1)?;
                        add(&mut self.submitted.pinned_bytes, r.bytes)?
                    }
                    Kind::Device => {
                        add(&mut self.submitted.device, 1)?;
                        add(&mut self.submitted.device_bytes, r.bytes)?
                    }
                    Kind::Event => add(&mut self.submitted.events, 1)?,
                    _ => {}
                }
            }
            Op::H2d { src, dst, bytes } => {
                if r.bytes != *bytes
                    || r.source != Some(100 + u64::from(*src))
                    || r.destination != Some(200 + u64::from(*dst))
                {
                    return Err("STOP_OWNERSHIP: H2D allocation/length receipt".into());
                }
                add(&mut self.submitted.h2d, r.bytes)?;
                add(&mut self.submitted.h2d_calls, 1)?;
                add(&mut self.pending_h2d, r.bytes)?;
                add(&mut self.pending_h2d_calls, 1)?;
            }
            Op::D2h(seq) => {
                if r.bytes != 512
                    || r.source != Some(209)
                    || r.destination != Some(300 + u64::from(*seq))
                {
                    return Err("STOP_OWNERSHIP: final-only D2H receipt".into());
                }
                add(&mut self.submitted.d2h, r.bytes)?;
                add(&mut self.submitted.d2h_calls, 1)?;
                add(&mut self.pending_d2h, r.bytes)?;
            }
            Op::Kernel(_, _) => add(&mut self.submitted.kernels, 1)?,
            Op::Record(t) | Op::Fence(t) if r.token != Some(*t) => {
                return Err("STOP_OWNERSHIP: unconfirmed completion token".into())
            }
            _ => {}
        }
        Ok(())
    }
    fn commit(&mut self, op: &Op, r: Receipt) -> Result<(), String> {
        match op {
            Op::Install => self.phase = Phase::Uploading,
            Op::Begin(seq) => self.phase = Phase::InFlight(*seq),
            Op::Record(t) => self.recorded[t.event as usize] = true,
            Op::Fence(t) => {
                self.consumed[t.event as usize] = true;
                if t.event < 6 {
                    add(&mut self.completed.h2d, self.pending_h2d)?;
                    add(&mut self.completed.h2d_calls, self.pending_h2d_calls)?;
                    self.pending_h2d = 0;
                    self.pending_h2d_calls = 0;
                    if t.event == 1 {
                        self.phase = Phase::Resident;
                    }
                } else {
                    add(&mut self.completed.d2h, self.pending_d2h)?;
                    add(&mut self.completed.d2h_calls, 1)?;
                    self.pending_d2h = 0;
                    self.next_sequence = self
                        .next_sequence
                        .checked_add(1)
                        .ok_or("STOP_OWNERSHIP: sequence overflow")?;
                    self.phase = Phase::Resident;
                }
            }
            Op::KernelWait(_, _) => add(&mut self.completed.kernels, 1)?,
            Op::Read(seq) => {
                let output = r.output.ok_or("STOP_OWNERSHIP: missing final output")?;
                if output.len() != 128 || output.iter().any(|x| !x.is_finite()) {
                    return Err("STOP_OWNERSHIP: invalid final output".into());
                }
                self.outputs[*seq as usize] = Some(output);
            }
            Op::Retire => self.phase = Phase::Retiring,
            _ => {}
        }
        self.trace.push(json!({"operation":format!("{op:?}"),"bytes":r.bytes,"source_allocation":r.source,
            "destination_allocation":r.destination,"detail":r.detail,"phase":format!("{:?}",self.phase),
            "submitted":self.submitted.json(),"completed":self.completed.json()}));
        self.cursor = self
            .cursor
            .checked_add(1)
            .ok_or("STOP_OWNERSHIP: cursor overflow")?;
        Ok(())
    }
}

/// No Clone, no exposed adapter. Its ManuallyDrop is the quarantine boundary:
/// unconfirmed drain must not invoke cudarc's unconditional resource Drops.
pub(crate) struct Runner<A: Adapter> {
    pub core: Engine,
    adapter: ManuallyDrop<A>,
    settled: bool,
    pub drain_attempts: u8,
    pub cleanup: String,
    pub dependency_leak_possible: bool,
    pub releases: Value,
}
impl<A: Adapter> Runner<A> {
    pub fn new(adapter: A, context: u64, data: &Data) -> Self {
        Self {
            core: Engine::new(context, data),
            adapter: ManuallyDrop::new(adapter),
            settled: false,
            drain_attempts: 0,
            cleanup: "NOT_ATTEMPTED".into(),
            dependency_leak_possible: false,
            releases: Value::Null,
        }
    }
    pub fn step(&mut self, op: Op, data: &Data) -> Result<(), String> {
        let mut adapter_called = false;
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.core.guard(&op, data)?;
            adapter_called = true;
            let receipt = self.adapter.execute(&op, data)?;
            self.core.submitted(&op, &receipt)?;
            self.adapter.check()?;
            self.core.commit(&op, receipt)
        }))
        .unwrap_or_else(|_| Err("STOP_OWNERSHIP: panic during operation".into()));
        if let Err(e) = &result {
            self.core.trace.push(json!({"operation":format!("{op:?}"),"status":"FAILED",
                "error":e,"submitted":self.core.submitted.json(),"completed":self.core.completed.json()}));
            if adapter_called
                && matches!(
                    op,
                    Op::Create(Kind::Context | Kind::Pinned | Kind::Device, _)
                )
            {
                // cudarc retains a context before a fallible attribute query,
                // and allocates before creating internal tracking events;
                // an Err can hide a resource with no returned Rust owner.
                self.dependency_leak_possible = true;
            }
            self.core.fail(e.clone());
        }
        result
    }
    pub fn execute(&mut self, data: &Data) -> Result<(), String> {
        while let Some(op) = self.core.schedule.get(self.core.cursor).cloned() {
            self.step(op, data)?;
        }
        Ok(())
    }
    pub fn cancel(&mut self, reason: &str) {
        self.core.fail(reason.into());
        self.finish();
    }
    pub fn finish(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        self.drain_attempts += 1;
        if self.core.first.is_none() && self.core.phase != Phase::Retiring {
            self.core
                .fail("STOP_OWNERSHIP: cancellation before retirement".into());
        }
        let drained = catch_unwind(AssertUnwindSafe(|| {
            self.adapter.drain()?;
            self.adapter.check()
        }))
        .unwrap_or_else(|_| Err("panic in drain".into()));
        if let Err(e) = drained {
            self.core.fail(e);
            self.cleanup = "STOP_DRAIN_UNCONFIRMED".into();
            return;
        }
        if self.dependency_leak_possible {
            self.cleanup = "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED".into();
            return;
        }
        let release = catch_unwind(AssertUnwindSafe(|| {
            let released = self.adapter.release()?;
            self.adapter.check()?;
            Ok::<Value, String>(released)
        }))
        .unwrap_or_else(|_| Err("panic in release".into()));
        match release {
            Err(e) => {
                self.core.fail(e);
                self.cleanup = "STOP_RELEASE_UNCONFIRMED".into();
            }
            Ok(r) => {
                self.releases = r;
                self.cleanup =
                    "DRAIN_AND_RELEASE_CONFIRMED_CONTEXT_DESTRUCTION_UNOBSERVABLE".into();
                if self.core.first.is_none() {
                    self.core.phase = Phase::Retired;
                }
                // SAFETY: all owned payloads/modules/events were released only
                // after checked drain. The remaining adapter has no queued work.
                unsafe { ManuallyDrop::drop(&mut self.adapter) };
            }
        }
    }
    pub fn report(&self, simulated: bool) -> Value {
        json!({"kind":if simulated{"SIMULATED"}else{"CUDA_API_LOGICAL_PAYLOAD"},
            "source_base":BASE,"context_identity":self.core.context,"slot":0,"generation":1,
            "expert_sha256":self.core.expert,"input_sha256":self.core.inputs,
            "expected":expected(),"submitted":self.core.submitted.json(),"completed":self.core.completed.json(),
            "phase":format!("{:?}",self.core.phase),"first_error":self.core.first,"secondary_errors":self.core.secondary,
            "drain_attempts":self.drain_attempts,"cleanup":self.cleanup,"releases":self.releases,
            "dependency_inaccessible_allocation_possible":self.dependency_leak_possible,
            "trace":self.core.trace,"timing_status":if simulated{"NOT_MEASURED"}else{"SEE_ADAPTER_BATCH_RECEIPTS"},
            "vram":{"status":"NOT_MEASURED","free_bytes":null,"total_bytes":null}})
    }
}
impl<A: Adapter> Drop for Runner<A> {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };
    struct Fake {
        call: usize,
        fail: Option<usize>,
        panic_at: Option<usize>,
        deferred_at: Option<usize>,
        drain_fail: bool,
        release_fail: bool,
        pinned: BTreeMap<u8, Vec<u8>>,
        device: BTreeMap<u8, Vec<u8>>,
        events: [bool; 8],
        created_events: [bool; 8],
        output: [Vec<f32>; 2],
        drops: Arc<AtomicUsize>,
        released: bool,
    }
    impl Fake {
        fn new() -> Self {
            Self {
                call: 0,
                fail: None,
                panic_at: None,
                deferred_at: None,
                drain_fail: false,
                release_fail: false,
                pinned: BTreeMap::new(),
                device: BTreeMap::new(),
                events: [false; 8],
                created_events: [false; 8],
                output: [vec![], vec![]],
                drops: Arc::new(AtomicUsize::new(0)),
                released: false,
            }
        }
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            assert!(self.released, "premature host/device destruction");
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl Adapter for Fake {
        fn execute(&mut self, op: &Op, data: &Data) -> Result<Receipt, String> {
            let index = self.call;
            self.call += 1;
            assert_ne!(self.panic_at, Some(index), "injected panic {index}");
            if self.fail == Some(index) {
                return Err(format!("direct:{index}"));
            }
            let mut r = Receipt::default();
            match op {
                Op::Create(kind, id) => {
                    r.bytes = payload(*kind, *id).unwrap();
                    match kind {
                        Kind::Pinned => {
                            self.pinned.insert(*id, vec![0; r.bytes as usize]);
                            r.destination = Some(100 + u64::from(*id))
                        }
                        Kind::Device => {
                            self.device.insert(*id, vec![0; r.bytes as usize]);
                            r.destination = Some(200 + u64::from(*id))
                        }
                        Kind::Event => {
                            assert!(!self.created_events[*id as usize]);
                            self.created_events[*id as usize] = true;
                            r.destination = Some(400 + u64::from(*id));
                        }
                        _ => {}
                    }
                }
                Op::H2d { src, dst, .. } => {
                    let source = self.pinned.get(src).unwrap();
                    let target = self.device.get_mut(dst).unwrap();
                    if source.len() != target.len() {
                        return Err("fake length".into());
                    }
                    target.copy_from_slice(source);
                    r.bytes = source.len() as u64;
                    r.source = Some(100 + u64::from(*src));
                    r.destination = Some(200 + u64::from(*dst));
                }
                Op::Record(t) => {
                    self.events[t.event as usize] = true;
                    r.token = Some(*t)
                }
                Op::Fence(t) => {
                    if !self.events[t.event as usize] {
                        return Err("fake unrecorded".into());
                    }
                    r.token = Some(*t)
                }
                Op::Kernel(seq, 4) => {
                    let mut f = data.fixture.clone();
                    f.input = data.inputs[*seq as usize].clone();
                    self.output[*seq as usize] = f.reference()?;
                }
                Op::D2h(seq) => {
                    r.bytes = self.device[&9].len() as u64;
                    r.source = Some(209);
                    r.destination = Some(300 + u64::from(*seq));
                }
                Op::Read(seq) => r.output = Some(self.output[*seq as usize].clone()),
                Op::Measure(_) => {
                    r.detail = json!({"status":"NOT_MEASURED","host_staging_ms":null,"enqueue_ms":null,"host_wait_ms":null,"cuda_event_ms":null})
                }
                _ => {}
            }
            Ok(r)
        }
        fn check(&mut self) -> Result<(), String> {
            if self.deferred_at.is_some() && self.deferred_at == self.call.checked_sub(1) {
                Err(format!("deferred:{}", self.call - 1))
            } else {
                Ok(())
            }
        }
        fn drain(&mut self) -> Result<(), String> {
            if self.drain_fail {
                Err("drain failure".into())
            } else {
                self.deferred_at = None;
                Ok(())
            }
        }
        fn release(&mut self) -> Result<Value, String> {
            if self.release_fail {
                return Err("release failure".into());
            }
            let r = json!({"pinned":self.pinned.len(),"device":self.device.len(),"events":self.created_events.iter().filter(|v|**v).count()});
            self.released = true;
            self.pinned.clear();
            self.device.clear();
            Ok(r)
        }
        fn simulated(&self) -> bool {
            true
        }
    }
    fn runner() -> (crate::Prepared, Runner<Fake>) {
        let p = crate::prepare().unwrap();
        let r = Runner::new(Fake::new(), 17, &p.data);
        (p, r)
    }
    fn resource_create_failure_is_quarantined(f: Fake, op: Op, expected: &str) {
        let p = crate::prepare().unwrap();
        let drops = f.drops.clone();
        let mut r = Runner::new(f, 17, &p.data);
        let index = r.core.schedule.iter().position(|next| *next == op).unwrap();
        for next in plan(17).into_iter().take(index) {
            r.step(next, &p.data).unwrap();
        }
        assert_eq!(r.step(op.clone(), &p.data), Err(expected.into()));
        assert_eq!(r.core.first.as_deref(), Some(expected));
        assert_eq!(r.core.phase, Phase::Poisoned);
        assert_eq!(r.core.cursor, index);
        assert_eq!(r.adapter.call, index + 1);
        assert!(r.dependency_leak_possible);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        if r.adapter.deferred_at == Some(index) {
            match op {
                Op::Create(Kind::Pinned, id) => {
                    assert!(r.adapter.pinned.contains_key(&id));
                    assert_eq!(r.core.submitted.pinned, u64::from(id) + 1);
                }
                Op::Create(Kind::Device, id) => {
                    assert!(r.adapter.device.contains_key(&id));
                    assert_eq!(r.core.submitted.device, u64::from(id) + 1);
                }
                _ => {}
            }
        }
        assert!(r.step(Op::Install, &p.data).is_err());
        assert_eq!(r.adapter.call, index + 1);
        assert_eq!(r.core.first.as_deref(), Some(expected));
        r.finish();
        r.finish();
        assert_eq!(r.drain_attempts, 1);
        assert_eq!(r.core.phase, Phase::Poisoned);
        assert_eq!(r.core.first.as_deref(), Some(expected));
        assert_eq!(r.cleanup, "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED");
        assert!(r.releases.is_null());
        assert!(!r.adapter.released);
        assert_eq!(r.adapter.call, index + 1);
        let report = r.report(true);
        assert_eq!(report["dependency_inaccessible_allocation_possible"], true);
        assert_eq!(report["cleanup"], "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED");
        assert_eq!(report["first_error"], expected);
        assert!(report["releases"].is_null());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        println!("RESOURCE_CREATE_FAILURE={op:?}:{expected}; REPORT={report}");
        drop(r);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn context_create_direct_failure_is_quarantined() {
        let mut f = Fake::new();
        f.fail = Some(0);
        resource_create_failure_is_quarantined(f, Op::Create(Kind::Context, 0), "direct:0");
    }
    #[test]
    fn context_create_deferred_failure_is_quarantined() {
        let mut f = Fake::new();
        f.deferred_at = Some(0);
        resource_create_failure_is_quarantined(f, Op::Create(Kind::Context, 0), "deferred:0");
    }
    #[test]
    fn context_create_panic_is_quarantined() {
        let mut f = Fake::new();
        f.panic_at = Some(0);
        resource_create_failure_is_quarantined(
            f,
            Op::Create(Kind::Context, 0),
            "STOP_OWNERSHIP: panic during operation",
        );
    }
    #[test]
    fn pinned_and_device_create_direct_deferred_and_panic_failures_are_quarantined() {
        let mut tested = 0;
        for (index, op) in plan(17).into_iter().enumerate() {
            if !matches!(op, Op::Create(Kind::Pinned | Kind::Device, _)) {
                continue;
            }
            for failure in 0..3 {
                let mut f = Fake::new();
                let expected = match failure {
                    0 => {
                        f.fail = Some(index);
                        format!("direct:{index}")
                    }
                    1 => {
                        f.deferred_at = Some(index);
                        format!("deferred:{index}")
                    }
                    _ => {
                        f.panic_at = Some(index);
                        "STOP_OWNERSHIP: panic during operation".into()
                    }
                };
                resource_create_failure_is_quarantined(f, op.clone(), &expected);
                tested += 1;
            }
        }
        assert_eq!(tested, 48);
    }
    #[test]
    fn guard_rejected_context_after_non_resource_failure_releases_owners() {
        for deferred in [false, true] {
            let (p, mut r) = runner();
            while r.core.schedule[r.core.cursor] != Op::Install {
                let op = r.core.schedule[r.core.cursor].clone();
                r.step(op, &p.data).unwrap();
            }
            let index = r.adapter.call;
            let expected = if deferred {
                r.adapter.deferred_at = Some(index);
                format!("deferred:{index}")
            } else {
                r.adapter.fail = Some(index);
                format!("direct:{index}")
            };
            let drops = r.adapter.drops.clone();
            assert_eq!(r.step(Op::Install, &p.data), Err(expected.clone()));
            assert!(!r.dependency_leak_possible);
            assert_eq!(
                r.step(Op::Create(Kind::Context, 0), &p.data),
                Err("STOP_OWNERSHIP: terminal, out-of-order or identity mismatch".into())
            );
            assert_eq!(r.adapter.call, index + 1);
            assert!(!r.dependency_leak_possible);
            assert_eq!(r.core.first.as_deref(), Some(expected.as_str()));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            r.finish();
            r.finish();
            assert_eq!(r.drain_attempts, 1);
            assert_eq!(r.core.phase, Phase::Poisoned);
            assert_eq!(r.core.first.as_deref(), Some(expected.as_str()));
            assert_eq!(
                r.cleanup,
                "DRAIN_AND_RELEASE_CONFIRMED_CONTEXT_DESTRUCTION_UNOBSERVABLE"
            );
            assert_eq!(r.releases, json!({"pinned":6,"device":10,"events":8}));
            assert_eq!(
                r.report(true)["dependency_inaccessible_allocation_possible"],
                false
            );
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            drop(r);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }
    #[test]
    fn guard_rejected_out_of_order_pinned_and_device_creates_do_not_quarantine() {
        for kind in [Kind::Pinned, Kind::Device] {
            let (p, mut r) = runner();
            let drops = r.adapter.drops.clone();
            let expected = "STOP_OWNERSHIP: terminal, out-of-order or identity mismatch";
            assert_eq!(r.step(Op::Create(kind, 0), &p.data), Err(expected.into()));
            assert_eq!(r.adapter.call, 0);
            assert_eq!(r.core.cursor, 0);
            assert_eq!(r.core.first.as_deref(), Some(expected));
            assert!(!r.dependency_leak_possible);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            r.finish();
            r.finish();
            assert_eq!(r.drain_attempts, 1);
            assert_eq!(r.core.phase, Phase::Poisoned);
            assert_eq!(r.core.first.as_deref(), Some(expected));
            assert_eq!(
                r.cleanup,
                "DRAIN_AND_RELEASE_CONFIRMED_CONTEXT_DESTRUCTION_UNOBSERVABLE"
            );
            assert_eq!(r.releases, json!({"pinned":0,"device":0,"events":0}));
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            drop(r);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }
    #[test]
    fn empty_cancellation_without_context_attempt_releases_normally() {
        let (_, mut r) = runner();
        let drops = r.adapter.drops.clone();
        assert_eq!(r.adapter.call, 0);
        r.cancel("cancelled before context create");
        r.finish();
        assert_eq!(r.core.first.as_deref(), Some("cancelled before context create"));
        assert_eq!(r.core.phase, Phase::Poisoned);
        assert_eq!(r.core.cursor, 0);
        assert!(r.core.trace.is_empty());
        assert!(!r.dependency_leak_possible);
        assert_eq!(r.drain_attempts, 1);
        assert_eq!(r.cleanup, "DRAIN_AND_RELEASE_CONFIRMED_CONTEXT_DESTRUCTION_UNOBSERVABLE");
        assert_eq!(r.releases, json!({"pinned":0,"device":0,"events":0}));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(r);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn complete_shared_schedule_and_observed_counters() {
        let (p, mut r) = runner();
        r.execute(&p.data).unwrap();
        assert!(r.adapter.simulated());
        assert_eq!(
            (
                r.core.submitted.h2d,
                r.core.completed.h2d,
                r.core.completed.d2h
            ),
            (33816, 33816, 1024)
        );
        assert_eq!(
            (
                r.core.submitted.h2d_calls,
                r.core.submitted.d2h_calls,
                r.core.submitted.kernels
            ),
            (16, 2, 10)
        );
        assert_eq!(
            (
                r.core.submitted.device,
                r.core.submitted.pinned,
                r.core.submitted.events
            ),
            (10, 6, 8)
        );
        assert_eq!(
            (r.core.submitted.device_bytes, r.core.submitted.pinned_bytes),
            (30744, 28696)
        );
        for seq in 0..2 {
            assert_eq!(r.core.outputs[seq], Some(p.references[seq].clone()));
        }
        let drops = r.adapter.drops.clone();
        r.finish();
        assert_eq!(r.core.phase, Phase::Retired);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let report = r.report(true);
        let roundtrip: Value = serde_json::from_str(&report.to_string()).unwrap();
        assert_eq!(report, roundtrip);
        println!("G2_SIMULATED_TRACE={report}");
    }
    #[test]
    fn every_operation_direct_and_deferred_failure_is_terminal() {
        let p = crate::prepare().unwrap();
        let schedule = plan(17);
        let mut tested = 0;
        for i in 0..schedule.len() {
            for deferred in [false, true] {
                let mut f = Fake::new();
                if deferred {
                    f.deferred_at = Some(i)
                } else {
                    f.fail = Some(i)
                }
                let mut r = Runner::new(f, 17, &p.data);
                assert!(r.execute(&p.data).is_err());
                let first = r.core.first.clone();
                assert_eq!(
                    first,
                    Some(format!(
                        "{}:{i}",
                        if deferred { "deferred" } else { "direct" }
                    ))
                );
                let resource_create = matches!(
                    schedule[i],
                    Op::Create(Kind::Context | Kind::Pinned | Kind::Device, _)
                );
                assert_eq!(r.dependency_leak_possible, resource_create);
                let calls = r.adapter.call;
                let drops = r.adapter.drops.clone();
                assert!(r.step(schedule[0].clone(), &p.data).is_err());
                assert_eq!(r.adapter.call, calls);
                assert_eq!(r.core.first, first);
                assert_eq!(r.dependency_leak_possible, resource_create);
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                assert!(r.core.completed.h2d <= r.core.submitted.h2d);
                assert!(r.core.completed.d2h <= r.core.submitted.d2h);
                if matches!(schedule[i], Op::H2d { .. } | Op::D2h(_) | Op::Fence(_)) {
                    assert!(r.core.completed.h2d + r.core.completed.d2h < 34840);
                }
                r.finish();
                r.finish();
                assert_eq!(r.drain_attempts, 1);
                assert_eq!(r.core.phase, Phase::Poisoned);
                assert_eq!(r.core.first, first);
                if resource_create {
                    assert_eq!(r.cleanup, "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED");
                    assert!(r.releases.is_null());
                    assert!(!r.adapter.released);
                } else {
                    assert_eq!(
                        r.cleanup,
                        "DRAIN_AND_RELEASE_CONFIRMED_CONTEXT_DESTRUCTION_UNOBSERVABLE"
                    );
                    assert!(r.releases.is_object());
                }
                assert_eq!(drops.load(Ordering::SeqCst), usize::from(!resource_create));
                drop(r);
                assert_eq!(drops.load(Ordering::SeqCst), usize::from(!resource_create));
                tested += 1;
            }
        }
        println!("FAULT_COVERAGE direct+deferred={tested}; all 6 pinned,10 device,8 event,3 module,3 function creation;16 H2D;3 H2D fences;10 launches+10 waits;2 D2H submit/record/wait");
    }
    #[test]
    fn wrong_missing_duplicate_extra_operations_stop_before_adapter() {
        let p = crate::prepare().unwrap();
        let schedule = plan(17);
        for i in 0..schedule.len() {
            let mut r = Runner::new(Fake::new(), 17, &p.data);
            for op in schedule.iter().take(i) {
                r.step(op.clone(), &p.data).unwrap()
            }
            let calls = r.adapter.call;
            let bad = schedule.get(i + 1).cloned().unwrap_or(Op::Install);
            assert!(r.step(bad, &p.data).is_err());
            assert_eq!(r.adapter.call, calls);
        }
    }
    #[test]
    fn all_completion_token_fields_unrecorded_stale_and_duplicate_rejected() {
        for event in [1, 3, 5, 6, 7] {
            for field in 0..7 {
                let (p, mut r) = runner();
                let target = Op::Fence(token(17, event));
                while r.core.schedule[r.core.cursor] != target {
                    let op = r.core.schedule[r.core.cursor].clone();
                    r.step(op, &p.data).unwrap()
                }
                let mut t = token(17, event);
                match field {
                    0 => t.context += 1,
                    1 => t.slot = 1,
                    2 => t.generation += 1,
                    3 => t.sequence = 9,
                    4 => t.event = 0,
                    5 => t.recorded = false,
                    _ => r.core.recorded[event as usize] = false,
                }
                assert!(r.step(Op::Fence(t), &p.data).is_err());
            }
        }
        let (p, mut r) = runner();
        r.execute(&p.data).unwrap();
        r.finish();
        assert!(r.step(Op::Retire, &p.data).is_err());
    }
    #[test]
    fn pending_upload_and_readback_forbid_refill_read_retire() {
        for event in [1, 6, 7] {
            for bad in [
                Op::Install,
                Op::Begin(1),
                Op::Stage(1),
                Op::Read(0),
                Op::Retire,
            ] {
                let (p, mut r) = runner();
                let fence = Op::Fence(token(17, event));
                while r.core.schedule[r.core.cursor] != fence {
                    let op = r.core.schedule[r.core.cursor].clone();
                    r.step(op, &p.data).unwrap()
                }
                if event == 1 {
                    assert_eq!(r.core.phase, Phase::Uploading);
                    assert_eq!(r.core.completed.h2d, 0)
                } else {
                    assert!(matches!(r.core.phase, Phase::InFlight(_)));
                    assert!(r.core.pending_d2h > 0)
                }
                assert!(r.step(bad, &p.data).is_err());
            }
        }
    }
    #[test]
    fn cancel_and_panic_retain_owners_until_one_drain() {
        for phase in [
            Phase::Empty,
            Phase::Uploading,
            Phase::Resident,
            Phase::InFlight(0),
        ] {
            for panic in [false, true] {
                let (p, mut r) = runner();
                while r.core.phase != phase {
                    let op = r.core.schedule[r.core.cursor].clone();
                    r.step(op, &p.data).unwrap()
                }
                let drops = r.adapter.drops.clone();
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                if panic {
                    r.adapter.panic_at = Some(r.adapter.call);
                    let op = r.core.schedule[r.core.cursor].clone();
                    assert!(r.step(op, &p.data).is_err());
                    r.finish()
                } else {
                    r.cancel("cancelled")
                }
                r.finish();
                assert_eq!(r.drain_attempts, 1);
                assert_eq!(r.core.phase, Phase::Poisoned);
                if panic && phase == Phase::Empty {
                    assert!(r.dependency_leak_possible);
                    assert_eq!(r.cleanup, "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED");
                    assert!(r.releases.is_null());
                    assert_eq!(drops.load(Ordering::SeqCst), 0);
                } else {
                    assert_eq!(drops.load(Ordering::SeqCst), 1);
                }
            }
        }
    }
    #[test]
    fn first_error_survives_failed_drain_and_release_and_dependency_limits() {
        for release in [false, true] {
            let (p, mut r) = runner();
            r.step(r.core.schedule[0].clone(), &p.data).unwrap();
            if release {
                r.adapter.release_fail = true
            } else {
                r.adapter.drain_fail = true
            }
            let drops = r.adapter.drops.clone();
            r.cancel("FIRST");
            r.finish();
            assert_eq!(r.core.first.as_deref(), Some("FIRST"));
            assert_eq!(r.drain_attempts, 1);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert!(r.cleanup.ends_with("UNCONFIRMED"));
        }
        let (p, mut r) = runner();
        r.adapter.fail = plan(17)
            .iter()
            .position(|op| *op == Op::Create(Kind::Pinned, 3));
        assert!(r.execute(&p.data).is_err());
        r.finish();
        assert!(r.dependency_leak_possible);
        assert_eq!(r.cleanup, "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED");
    }
    #[test]
    fn overflow_and_forged_receipts_cannot_be_reported_as_completion() {
        let (p, mut r) = runner();
        r.core.submitted.h2d = u64::MAX;
        while !matches!(r.core.schedule[r.core.cursor], Op::H2d { .. }) {
            let op = r.core.schedule[r.core.cursor].clone();
            r.step(op, &p.data).unwrap()
        }
        let op = r.core.schedule[r.core.cursor].clone();
        assert!(r.step(op, &p.data).is_err());
        assert_eq!(r.core.completed.h2d, 0);
        for forged in [
            Receipt {
                bytes: 9216,
                source: Some(999),
                destination: Some(200),
                ..Receipt::default()
            },
            Receipt {
                bytes: 33816,
                source: Some(100),
                destination: Some(200),
                ..Receipt::default()
            },
        ] {
            let mut engine = Engine::new(17, &p.data);
            assert!(engine
                .submitted(
                    &Op::H2d {
                        src: 0,
                        dst: 0,
                        bytes: 9216
                    },
                    &forged
                )
                .is_err());
            assert_eq!(engine.completed.h2d, 0);
        }
        let mut n = u64::MAX;
        assert!(add(&mut n, 1).is_err());
    }
}
