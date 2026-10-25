//! CPU-only injected driver for the ownership helpers used by native cuda.rs.
//! Counters observe Rust test handles, never CUDA allocations or physical cleanup.
use crate::cuda::ownership::{self, Owned, State};
use mer2_cuda_g3_s1::{
    abi::{
        Adapter, CleanupReceipt, Completion, ExpertWeights, FfnProvider, InitReceipt, Operation,
        Resident, Submission, Vector,
    },
    host_token::Session,
    identity::{source_files, Model, Variant, VerifiedSources},
    state::{Phase, Runner},
    Result, Stop,
};
use std::{cell::RefCell, path::Path, rc::Rc};

// Constructor order in frozen native initialize(): context, stream, interleaved
// modules/functions, 8 events, 24 weights, layout, 6 scratch buffers.
const OWNERS: usize = 47;
fn kind(i: usize) -> &'static str {
    match i {
        0 => "context",
        1 => "stream",
        2 | 4 | 6 => "module",
        3 | 5 | 7 => "function",
        8..=15 => "event",
        16..=39 => "device_weight",
        40 => "layout",
        41..=46 => "scratch",
        _ => panic!("invalid test owner"),
    }
}
fn release_order() -> Vec<usize> {
    (16..40)
        .chain(41..47)
        .chain([40])
        .chain(8..16)
        .chain([3, 5, 7, 2, 4, 6, 1])
        .collect()
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Point {
    Create(usize),
    AfterReturn(usize),
    H2d,
    Enqueue,
    StreamFence,
    EventFence,
    Drain,
    DrainDeferred,
    Destroy(usize),
    ReleaseSync(usize),
    ReleaseDeferred(usize),
    PostDrain,
    PostRelease,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Error,
    Panic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fault(Point, Mode);
#[derive(Default)]
struct Audit {
    created: Vec<usize>,
    drop_attempts: Vec<usize>,
    possible_unreturned: usize,
    calls: Vec<Point>,
    drains: usize,
    releases: usize,
    owner_drops: usize,
    direct_quarantines: usize,
    first: Option<Stop>,
    secondary: Vec<Stop>,
    deferred_destroy: Option<Stop>,
}
#[derive(Clone)]
struct Driver {
    audit: Rc<RefCell<Audit>>,
    fault: Option<Fault>,
}
fn injected(point: Point) -> Stop {
    Stop::new("INJECTED_DRIVER_ERROR", format!("{point:?}"))
}
fn expected(fault: Fault) -> Stop {
    match fault.1 {
        Mode::Error => injected(fault.0),
        Mode::Panic => Stop::new("STOP_PANIC", "native adapter operation panicked"),
    }
}
impl Driver {
    fn hit(&self, point: Point) -> Result<()> {
        self.audit.borrow_mut().calls.push(point);
        match self.fault {
            Some(Fault(p, Mode::Error)) if p == point => Err(injected(point)),
            Some(Fault(p, Mode::Panic)) if p == point => panic!("injected driver panic: {point:?}"),
            _ => Ok(()),
        }
    }
    fn create(&self, index: usize) -> Result<Handle> {
        if self.fault.is_some_and(|f| f.0 == Point::Create(index)) {
            // Models F-001's side effect BEFORE returning any typed owner. No
            // raw CUDA handle exists here and no recovery release is attempted.
            self.audit.borrow_mut().possible_unreturned += 1;
        }
        self.hit(Point::Create(index))?;
        self.audit.borrow_mut().created.push(index);
        Ok(Handle {
            index,
            driver: self.clone(),
        })
    }
    fn release_check(&self, index: usize) -> Result<()> {
        // Native synchronizes allocations and stream; events/functions/modules
        // only check the context's deferred error slot.
        if index == 1 || index >= 16 {
            self.hit(Point::ReleaseSync(index))?;
        }
        if let Some(error) = self.audit.borrow_mut().deferred_destroy.take() {
            return Err(error);
        }
        self.hit(Point::ReleaseDeferred(index))
    }
}
struct Handle {
    index: usize,
    driver: Driver,
}
impl Drop for Handle {
    fn drop(&mut self) {
        self.driver
            .audit
            .borrow_mut()
            .drop_attempts
            .push(self.index);
        if let Err(error) = self.driver.hit(Point::Destroy(self.index)) {
            self.driver.audit.borrow_mut().deferred_destroy = Some(error);
        }
    }
}
// Deliberately typed: successful context return is distinct from the inaccessible
// dependency side effect modeled by Driver::create's failing path.
struct ContextOwner {
    _handle: Handle,
}
struct CpuOwner {
    state: State,
    driver: Driver,
    context_id: u64,
    context: Option<ContextOwner>,
    slots: [Option<Handle>; OWNERS],
    pending: bool,
}
impl Owned for CpuOwner {
    fn state(&self) -> &State {
        &self.state
    }
    fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }
    fn note_error(&mut self, error: &Stop, first: bool) {
        let mut a = self.driver.audit.borrow_mut();
        if first {
            a.first = Some(error.clone());
        } else {
            a.secondary.push(error.clone());
        }
    }
}
impl Drop for CpuOwner {
    fn drop(&mut self) {
        self.driver.audit.borrow_mut().owner_drops += 1;
    }
}
impl CpuOwner {
    fn initialize(&mut self, context: u64) -> Result<()> {
        self.state.initialize_allowed(context)?;
        self.context_id = context;
        self.state.initialization_attempted = true;
        for index in 0..OWNERS {
            ownership::retain(
                self,
                |o| {
                    let handle = o.driver.create(index)?;
                    if index == 0 {
                        o.context = Some(ContextOwner { _handle: handle });
                    } else {
                        o.slots[index] = Some(handle);
                    }
                    Ok(())
                },
                |o| o.driver.hit(Point::AfterReturn(index)),
            )?;
        }
        Ok(())
    }
    fn operation(&mut self, point: Point) -> Result<()> {
        self.pending = true;
        self.driver.hit(point)
    }
    fn drain(&mut self, context: u64) -> Result<CleanupReceipt> {
        let matching = self.context_id == 0 || self.context_id == context;
        ownership::drain(self, matching, |o| {
            o.driver.audit.borrow_mut().drains += 1;
            o.driver.hit(Point::Drain)?;
            o.driver.hit(Point::DrainDeferred)
        })?;
        self.context_id = context;
        self.pending = false;
        Ok(receipt(context))
    }
    fn release(&mut self, context: u64) -> Result<CleanupReceipt> {
        let matching = context == self.context_id;
        let no_pending = !self.pending;
        ownership::release(self, matching, no_pending, |o| {
            o.driver.audit.borrow_mut().releases += 1;
            let driver = o.driver.clone();
            for index in release_order() {
                ownership::release_slots(std::slice::from_mut(&mut o.slots[index]), || {
                    driver.release_check(index)
                })?;
            }
            Ok(())
        })?;
        Ok(receipt(context))
    }
}
fn receipt(context: u64) -> CleanupReceipt {
    CleanupReceipt {
        context_identity: context,
        all_work_complete: true,
        ownership_confirmed: true,
    }
}
struct CpuAdapter {
    owner: Option<Box<CpuOwner>>,
}
impl CpuAdapter {
    fn new(fault: Option<Fault>) -> (Self, Rc<RefCell<Audit>>) {
        let audit = Rc::new(RefCell::new(Audit::default()));
        let owner = CpuOwner {
            state: State::default(),
            driver: Driver {
                audit: audit.clone(),
                fault,
            },
            context_id: 0,
            context: None,
            slots: std::array::from_fn(|_| None),
            pending: false,
        };
        (
            Self {
                owner: Some(Box::new(owner)),
            },
            audit,
        )
    }
    fn call<T>(&mut self, cleanup: bool, f: impl FnOnce(&mut CpuOwner) -> Result<T>) -> Result<T> {
        ownership::call(self.owner.as_mut().unwrap().as_mut(), cleanup, f)
    }
    fn state(&self) -> &State {
        &self.owner.as_ref().unwrap().state
    }
    fn initialize_direct(&mut self) -> Result<()> {
        self.call(false, |o| o.initialize(1))
    }
}
impl Drop for CpuAdapter {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            ownership::dispose(owner, |o| {
                o.driver.audit.borrow_mut().direct_quarantines += 1
            });
        }
    }
}
impl Adapter for CpuAdapter {
    fn initialize(
        &mut self,
        context: u64,
        residents: &[Resident],
        _: &[ExpertWeights],
    ) -> Result<InitReceipt> {
        self.call(false, |o| o.initialize(context))?;
        Ok(InitReceipt {
            context_identity: context,
            residents: residents.to_vec(),
            confirmed: true,
        })
    }
    fn submit(&mut self, _: &Operation, _: &Vector) -> Result<Submission> {
        panic!("numerical execution is outside this ownership test")
    }
    fn complete(&mut self, _: &Operation) -> Result<Completion> {
        panic!("numerical execution is outside this ownership test")
    }
    fn check(&mut self) -> Result<()> {
        self.call(true, |o| {
            if o.state.released {
                o.driver.hit(Point::PostRelease)
            } else if o.state.drained {
                o.driver.hit(Point::PostDrain)
            } else {
                Ok(())
            }
        })
    }
    fn drain(&mut self, context: u64) -> Result<CleanupReceipt> {
        self.call(true, |o| o.drain(context))
    }
    fn release(&mut self, context: u64) -> Result<CleanupReceipt> {
        self.call(true, |o| o.release(context))
    }
}
fn initialized(fault: Option<Fault>) -> (CpuAdapter, Rc<RefCell<Audit>>) {
    let (mut adapter, audit) = CpuAdapter::new(fault);
    adapter.initialize_direct().unwrap();
    (adapter, audit)
}
fn assert_first(adapter: &CpuAdapter, audit: &Rc<RefCell<Audit>>, first: &Stop) {
    assert_eq!(adapter.state().first_error.as_ref(), Some(first));
    assert_eq!(audit.borrow().first.as_ref(), Some(first));
}
fn trace(case: &str) {
    println!("CPU_OWNERSHIP_CASE={case}; PHYSICAL_CUDA_PROOF=NONE");
}

#[test]
fn cold_drop_is_safe_without_any_driver_call() {
    let (adapter, audit) = CpuAdapter::new(None);
    drop(adapter);
    let a = audit.borrow();
    assert_eq!(a.owner_drops, 1);
    assert_eq!(a.direct_quarantines, 0);
    assert!(a.calls.is_empty() && a.created.is_empty() && a.drop_attempts.is_empty());
    trace("cold_drop");
}
#[test]
fn typed_context_success_checked_drain_single_release_then_context_drop() {
    let (mut adapter, audit) = initialized(None);
    assert!(adapter.owner.as_ref().unwrap().context.is_some());
    assert!(!adapter.state().ownership_uncertain && !adapter.state().unreturned_owner_possible);
    assert_eq!(audit.borrow().created, (0..OWNERS).collect::<Vec<_>>());
    adapter.drain(1).unwrap();
    adapter.release(1).unwrap();
    assert!(adapter.state().drained && adapter.state().released);
    assert_eq!(audit.borrow().drop_attempts, release_order());
    assert!(!audit.borrow().drop_attempts.contains(&0)); // last context Drop is later
    assert_eq!(adapter.drain(1).unwrap_err().code, "STOP_DRAIN_UNCONFIRMED");
    assert_eq!(
        adapter.release(1).unwrap_err().code,
        "STOP_RELEASE_UNCONFIRMED"
    );
    assert_eq!((audit.borrow().drains, audit.borrow().releases), (1, 1));
    drop(adapter);
    let a = audit.borrow();
    assert_eq!(a.drop_attempts.last(), Some(&0));
    assert_eq!(a.drop_attempts.len(), OWNERS);
    assert_eq!((a.owner_drops, a.direct_quarantines), (1, 0));
    trace("owned_resources_released_context_drop_observed_only_in_cpu_model");
}
#[test]
fn every_constructor_error_and_panic_quarantines_possible_inaccessible_owner() {
    for index in 0..OWNERS {
        for mode in [Mode::Error, Mode::Panic] {
            let fault = Fault(Point::Create(index), mode);
            let (mut adapter, audit) = CpuAdapter::new(Some(fault));
            let first = adapter.initialize_direct().unwrap_err();
            assert_eq!(first, expected(fault));
            assert!(
                adapter.state().ownership_uncertain && adapter.state().unreturned_owner_possible
            );
            assert_eq!(audit.borrow().created.len(), index);
            assert_eq!(audit.borrow().possible_unreturned, 1);
            let calls = audit.borrow().calls.len();
            assert_eq!(
                adapter
                    .call::<()>(false, |_| panic!("terminal operation executed"))
                    .unwrap_err(),
                first
            );
            assert_eq!(audit.borrow().calls.len(), calls);
            assert_eq!(
                adapter.drain(1).unwrap_err().code,
                "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED"
            );
            assert_eq!(
                adapter.release(1).unwrap_err().code,
                "STOP_RELEASE_UNCONFIRMED"
            );
            assert_first(&adapter, &audit, &first);
            assert!(!adapter.state().drained && !adapter.state().released);
            drop(adapter);
            let a = audit.borrow();
            assert_eq!(
                (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
                (1, 0, 0, 1)
            );
            assert!(a.drop_attempts.is_empty());
            trace(&format!(
                "constructor_{index}_{}_{mode:?}_unknown_inaccessible_quarantined",
                kind(index)
            ));
        }
    }
}
#[test]
fn every_returned_owner_deferred_error_and_panic_stays_owned_and_quarantined() {
    for index in 0..OWNERS {
        for mode in [Mode::Error, Mode::Panic] {
            let fault = Fault(Point::AfterReturn(index), mode);
            let (mut adapter, audit) = CpuAdapter::new(Some(fault));
            let first = adapter.initialize_direct().unwrap_err();
            assert_eq!(first, expected(fault));
            assert!(adapter.state().ownership_uncertain);
            assert!(!adapter.state().unreturned_owner_possible);
            assert_eq!(audit.borrow().created.len(), index + 1);
            assert_eq!(audit.borrow().possible_unreturned, 0);
            assert!(adapter.owner.as_ref().unwrap().context.is_some());
            assert_eq!(
                adapter.drain(1).unwrap_err().code,
                "STOP_DEPENDENCY_CLEANUP_UNCONFIRMED"
            );
            assert_first(&adapter, &audit, &first);
            drop(adapter);
            let a = audit.borrow();
            assert_eq!(
                (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
                (1, 0, 0, 1)
            );
            assert!(a.drop_attempts.is_empty());
            trace(&format!(
                "returned_{index}_{}_{mode:?}_known_owned_quarantined",
                kind(index)
            ));
        }
    }
}
#[test]
fn h2d_enqueue_and_fences_preserve_first_error_and_require_checked_cleanup() {
    for point in [
        Point::H2d,
        Point::Enqueue,
        Point::StreamFence,
        Point::EventFence,
    ] {
        for mode in [Mode::Error, Mode::Panic] {
            let fault = Fault(point, mode);
            let (mut adapter, audit) = initialized(Some(fault));
            let first = adapter.call(false, |o| o.operation(point)).unwrap_err();
            assert_eq!(first, expected(fault));
            assert!(adapter.owner.as_ref().unwrap().pending);
            assert!(audit.borrow().drop_attempts.is_empty());
            assert!(!adapter.state().ownership_uncertain);
            assert_eq!(
                adapter.release(1).unwrap_err().code,
                "STOP_RELEASE_UNCONFIRMED"
            );
            assert_eq!(
                adapter
                    .call::<()>(false, |_| panic!("enqueue after terminal error"))
                    .unwrap_err(),
                first
            );
            adapter.drain(1).unwrap();
            adapter.release(1).unwrap();
            assert_first(&adapter, &audit, &first);
            drop(adapter);
            assert_eq!(audit.borrow().drop_attempts.len(), OWNERS);
            trace(&format!(
                "{point:?}_{mode:?}_terminal_error_checked_cleanup_only"
            ));
        }
    }
}
#[test]
fn premature_drop_with_partial_or_inflight_resources_never_runs_destructors() {
    for point in [
        None,
        Some(Point::H2d),
        Some(Point::Enqueue),
        Some(Point::EventFence),
    ] {
        let fault = point.map(|p| Fault(p, Mode::Error));
        let (mut adapter, audit) = initialized(fault);
        if let Some(p) = point {
            assert!(adapter.call(false, |o| o.operation(p)).is_err());
        }
        drop(adapter);
        let a = audit.borrow();
        assert_eq!(
            (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
            (0, 0, 0, 1)
        );
        assert!(a.drop_attempts.is_empty());
        trace(&format!(
            "direct_drop_{point:?}_intentionally_quarantined_until_process_exit"
        ));
    }
}
#[test]
fn drain_error_and_panic_keep_owned_resources_and_first_error() {
    for point in [Point::Drain, Point::DrainDeferred] {
        for mode in [Mode::Error, Mode::Panic] {
            let (mut adapter, audit) = initialized(Some(Fault(point, mode)));
            let primary = Stop::new("FIRST_OPERATION_ERROR", "work may be in flight");
            let first = adapter
                .call::<()>(false, |o| {
                    o.pending = true;
                    Err(primary.clone())
                })
                .unwrap_err();
            assert_eq!(adapter.drain(1).unwrap_err(), expected(Fault(point, mode)));
            assert!(!adapter.state().drained && !adapter.state().released);
            assert_first(&adapter, &audit, &first);
            drop(adapter);
            let a = audit.borrow();
            assert_eq!(
                (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
                (1, 0, 0, 1)
            );
            assert!(a.drop_attempts.is_empty());
            trace(&format!(
                "{point:?}_{mode:?}_first_error_preserved_no_release"
            ));
        }
    }
}
#[test]
fn every_release_destructor_and_check_fault_stops_before_remaining_owners() {
    for (position, index) in release_order().into_iter().enumerate() {
        let mut points = vec![Point::Destroy(index), Point::ReleaseDeferred(index)];
        if index == 1 || index >= 16 {
            points.push(Point::ReleaseSync(index));
        }
        for point in points {
            for mode in [Mode::Error, Mode::Panic] {
                let fault = Fault(point, mode);
                let (mut adapter, audit) = initialized(Some(fault));
                adapter.drain(1).unwrap();
                let first = adapter.release(1).unwrap_err();
                assert_eq!(first, expected(fault));
                assert!(!adapter.state().released);
                assert!(!adapter.state().unreturned_owner_possible);
                assert_eq!(audit.borrow().drop_attempts, release_order()[..=position]);
                assert_first(&adapter, &audit, &first);
                drop(adapter);
                let a = audit.borrow();
                assert_eq!(a.drop_attempts.len(), position + 1);
                assert!(!a.drop_attempts.contains(&0));
                assert_eq!(
                    (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
                    (1, 1, 0, 1)
                );
                trace(&format!(
                    "release_{index}_{}_{point:?}_{mode:?}_remaining_owners_quarantined",
                    kind(index)
                ));
            }
        }
    }
}
#[test]
fn wrong_context_pending_work_and_release_before_drain_are_rejected() {
    let (mut adapter, audit) = initialized(None);
    assert_eq!(
        adapter.release(1).unwrap_err().code,
        "STOP_RELEASE_UNCONFIRMED"
    );
    assert_eq!(adapter.drain(2).unwrap_err().code, "STOP_DRAIN_UNCONFIRMED");
    assert_eq!((audit.borrow().drains, audit.borrow().releases), (0, 0));
    adapter.drain(1).unwrap();
    assert_eq!(
        adapter.release(2).unwrap_err().code,
        "STOP_RELEASE_UNCONFIRMED"
    );
    adapter.owner.as_mut().unwrap().pending = true;
    assert_eq!(
        adapter.release(1).unwrap_err().code,
        "STOP_RELEASE_UNCONFIRMED"
    );
    assert_eq!(audit.borrow().releases, 0);
    drop(adapter);
    assert!(audit.borrow().drop_attempts.is_empty());
    trace("wrong_context_pending_work_release_without_drain");
}

fn runner(fault: Option<Fault>) -> (Runner<CpuAdapter>, Rc<RefCell<Audit>>, Result<()>) {
    let model = Model::generate(Variant::Primary).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let sources = VerifiedSources::verify(root, &source_files()).unwrap();
    let (adapter, audit) = CpuAdapter::new(fault);
    let mut runner = Runner::new(&model, adapter).unwrap();
    let mut session = Session::new(model, sources).unwrap();
    let request = session.request("ownership-only", 3).unwrap();
    // Only initialize the provider. No token decoder, numerical FFN, or GPU runs.
    let result = runner.begin_token(&request.ticket());
    (runner, audit, result)
}
#[test]
fn actual_s1_runner_settles_constructor_failures_once_without_drop() {
    for point in [
        Point::Create(0),
        Point::Create(16),
        Point::AfterReturn(0),
        Point::AfterReturn(46),
    ] {
        for mode in [Mode::Error, Mode::Panic] {
            let fault = Fault(point, mode);
            let (mut runner, audit, result) = runner(Some(fault));
            let first = result.unwrap_err();
            assert_eq!(first, expected(fault));
            assert_eq!(runner.report().first_error.as_ref(), Some(&first));
            assert_eq!(runner.report().phase, Phase::Quarantined);
            let settled = runner.report().clone();
            runner.finish();
            runner.finish();
            assert_eq!(runner.report(), &settled);
            assert_eq!(runner.report().drain_attempts, 1);
            drop(runner);
            let a = audit.borrow();
            assert_eq!(
                (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
                (1, 0, 0, 0)
            );
            assert!(a.drop_attempts.is_empty()); // S1 forgot adapter, its Drop never ran
            trace(&format!("actual_S1_finish_{point:?}_{mode:?}_settled_once"));
        }
    }
}
#[test]
fn actual_s1_runner_settles_cleanup_failures_panics_and_postchecks_once() {
    for point in [
        Point::Drain,
        Point::DrainDeferred,
        Point::Destroy(16),
        Point::ReleaseSync(16),
        Point::ReleaseDeferred(16),
        Point::PostDrain,
        Point::PostRelease,
    ] {
        for mode in [Mode::Error, Mode::Panic] {
            let (mut runner, audit, result) = runner(Some(Fault(point, mode)));
            result.unwrap();
            let first = Stop::new("FIRST_OPERATION_ERROR", "injected before cleanup");
            runner.poison(first.clone());
            assert_eq!(runner.report().first_error.as_ref(), Some(&first));
            assert_eq!(runner.report().phase, Phase::Quarantined);
            assert_eq!(runner.report().drain_attempts, 1);
            assert!(!runner.report().secondary_errors.is_empty());
            let settled = runner.report().clone();
            let before = audit.borrow().drop_attempts.clone();
            runner.finish();
            runner.finish();
            assert_eq!(runner.report(), &settled);
            drop(runner);
            let a = audit.borrow();
            let released = !matches!(
                point,
                Point::Drain | Point::DrainDeferred | Point::PostDrain
            );
            assert_eq!((a.drains, a.releases), (1, usize::from(released)));
            assert_eq!((a.owner_drops, a.direct_quarantines), (0, 0));
            assert_eq!(a.drop_attempts, before);
            assert!(!a.drop_attempts.contains(&0));
            trace(&format!(
                "actual_S1_finish_{point:?}_{mode:?}_quarantine_first_error_preserved"
            ));
        }
    }
}
#[test]
fn actual_s1_runner_repeated_finish_after_confirmed_cleanup_does_not_repeat() {
    let (mut runner, audit, result) = runner(None);
    result.unwrap();
    // Ending an unfinished token preserves cancellation as the first error but
    // can still safely settle known ownership through checked drain and release.
    runner.finish();
    assert_eq!(
        runner.report().first_error.as_ref().unwrap().code,
        "STOP_CANCELLED"
    );
    let settled = runner.report().clone();
    runner.finish();
    runner.finish();
    assert_eq!(runner.report(), &settled);
    drop(runner);
    let a = audit.borrow();
    assert_eq!(
        (a.drains, a.releases, a.owner_drops, a.direct_quarantines),
        (1, 1, 1, 0)
    );
    assert_eq!(a.drop_attempts.len(), OWNERS);
    trace("actual_S1_finish_success_after_cancellation_settled_once");
}
#[test]
fn pinned_owner_is_only_a_generic_helper_contract_not_a_native_allocation_path() {
    for mode in [Mode::Error, Mode::Panic] {
        let (mut adapter, audit) = CpuAdapter::new(Some(Fault(Point::Create(0), mode)));
        // Native G3 uses ordinary retained host slices. This models a hypothetical
        // pinned constructor through the SAME retain helper, without adding one.
        adapter
            .owner
            .as_mut()
            .unwrap()
            .state
            .initialization_attempted = true;
        let result = adapter.call(false, |o| {
            ownership::retain(
                o,
                |o| {
                    o.slots[0] = Some(o.driver.create(0)?);
                    Ok(())
                },
                |_| Ok(()),
            )
        });
        assert!(result.is_err());
        assert!(adapter.state().ownership_uncertain && adapter.state().unreturned_owner_possible);
        drop(adapter);
        assert!(audit.borrow().drop_attempts.is_empty());
        assert_eq!(audit.borrow().direct_quarantines, 1);
        trace(&format!(
            "hypothetical_pinned_helper_only_{mode:?}_NATIVE_PINNED_PATH_ABSENT"
        ));
    }
}
