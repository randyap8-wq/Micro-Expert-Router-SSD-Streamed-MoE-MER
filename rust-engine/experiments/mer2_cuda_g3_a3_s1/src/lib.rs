//! CPU-only A3-S1 contract and synchronous Unix process supervisor. No GPU API.
#![forbid(unsafe_code)]
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
pub type Result<T> = std::result::Result<T, &'static str>;
pub const SESSION_LIMIT: Duration = Duration::from_secs(2);
pub const REAP_LIMIT: Duration = Duration::from_secs(2);
const POLL: Duration = Duration::from_millis(5);
pub const MAX_FRAME: usize = 2048;
pub const FAKE_OUTPUT: &[u8] = b"abc";
pub const FAKE_DIGEST: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub source: String,
    pub model: String,
    pub kv: String,
}
impl Binding {
    fn valid(&self) -> bool {
        [&self.source, &self.model, &self.kv]
            .iter()
            .all(|s| digest(s))
    }
}
fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub pid: u32,
    pub generation: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ticket {
    pub request: String,
    pub generation: u64,
    pub nonce: u64,
    pub binding: Binding,
}
#[derive(Default, Clone)]
pub struct Ledger(Arc<Mutex<BTreeSet<String>>>);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Initializing,
    Ready,
    InFlight,
    Draining,
    Quarantined,
    Terminating,
    Exited,
    ExitUnconfirmed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    NotAttempted,
    UnreturnedPossible,
    ReturnedUnchecked,
    Confirmed,
    Uncertain,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    None,
    SubmissionUncertain,
    Submitted,
    Fenced,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cleanup {
    NotAttempted,
    DrainChecked,
    ReleaseCheckedScoped,
    Unconfirmed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    None,
    Prepared,
    SentUnacknowledged,
    Acknowledged,
    Ambiguous,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    Unknown,
    SyntheticBaselineObserved,
}
#[derive(Clone, Debug)]
pub struct Receipt {
    pub identity: Identity,
    pub phase: Phase,
    pub owner: Owner,
    pub work: Work,
    pub cleanup: Cleanup,
    pub delivery: Delivery,
    pub first_error: Option<&'static str>,
    pub secondary_errors: Vec<&'static str>,
    pub unknown_ownership: bool,
    pub drain_attempts: u8,
    pub release_attempts: u8,
    pub process_exit_observed: bool,
    pub post_exit_baseline: Observation,
    pub last_cuda_context_drop: &'static str,
    pub leak_free: &'static str,
    pub automatic_retry: &'static str,
}
/// Private mutable state; receipts are snapshots, never writable authority.
pub struct Contract {
    r: Receipt,
    binding: Binding,
    ledger: Ledger,
    current: Option<Ticket>,
    output: Option<String>,
}
impl Contract {
    pub fn new(identity: Identity, binding: Binding, ledger: Ledger) -> Result<Self> {
        if identity.pid == 0 || identity.generation == 0 || !binding.valid() {
            return Err("STOP_INITIAL_IDENTITY");
        }
        Ok(Self {
            binding,
            ledger,
            current: None,
            output: None,
            r: Receipt {
                identity,
                phase: Phase::Initializing,
                owner: Owner::NotAttempted,
                work: Work::None,
                cleanup: Cleanup::NotAttempted,
                delivery: Delivery::None,
                first_error: None,
                secondary_errors: vec![],
                unknown_ownership: false,
                drain_attempts: 0,
                release_attempts: 0,
                process_exit_observed: false,
                post_exit_baseline: Observation::Unknown,
                last_cuda_context_drop: "UNOBSERVABLE",
                leak_free: "NOT_CLAIMED",
                automatic_retry: "FORBIDDEN",
            },
        })
    }
    pub fn receipt(&self) -> Receipt {
        self.r.clone()
    }
    pub fn fault(&mut self, code: &'static str) -> Result<()> {
        let code = if code.starts_with("STOP_") {
            code
        } else {
            "STOP_INVALID_FAULT_LABEL"
        };
        if self.r.first_error.is_none() {
            self.r.first_error = Some(code);
        } else {
            self.r.secondary_errors.push(code);
        }
        if !matches!(self.r.phase, Phase::Exited | Phase::ExitUnconfirmed) {
            self.r.phase = Phase::Quarantined;
        }
        if matches!(
            self.r.owner,
            Owner::UnreturnedPossible | Owner::ReturnedUnchecked
        ) {
            self.r.owner = Owner::Uncertain;
            self.r.unknown_ownership = true;
        }
        if self.r.cleanup != Cleanup::ReleaseCheckedScoped {
            self.r.cleanup = Cleanup::Unconfirmed;
        }
        if self.current.is_some() && self.r.delivery != Delivery::Acknowledged {
            self.r.delivery = Delivery::Ambiguous;
        }
        Err(code)
    }
    fn require(&mut self, ok: bool, code: &'static str) -> Result<()> {
        if ok {
            Ok(())
        } else {
            self.fault(code)
        }
    }
    pub fn constructor_entered(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Initializing && self.r.owner == Owner::NotAttempted,
            "STOP_CONSTRUCTOR_ORDER",
        )?;
        self.r.owner = Owner::UnreturnedPossible;
        self.r.unknown_ownership = true;
        Ok(())
    }
    pub fn typed_owner_returned(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Initializing && self.r.owner == Owner::UnreturnedPossible,
            "STOP_CONTEXT_OWNER_ORDER",
        )?;
        self.r.owner = Owner::ReturnedUnchecked;
        Ok(())
    }
    pub fn deferred_check(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Initializing && self.r.owner == Owner::ReturnedUnchecked,
            "STOP_CONTEXT_CHECK_ORDER",
        )?;
        self.r.owner = Owner::Confirmed;
        self.r.unknown_ownership = false;
        Ok(())
    }
    pub fn ready(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Initializing
                && self.r.owner == Owner::Confirmed
                && !self.r.unknown_ownership
                && self.r.first_error.is_none(),
            "STOP_READY_WITH_UNCERTAIN_OWNER",
        )?;
        self.r.phase = Phase::Ready;
        Ok(())
    }
    pub fn admit(&mut self, t: &Ticket) -> Result<()> {
        self.require(
            self.r.phase == Phase::Ready
                && self.current.is_none()
                && self.r.first_error.is_none()
                && name(&t.request)
                && t.generation == self.r.identity.generation
                && t.nonce > 0
                && t.binding == self.binding,
            "STOP_ADMISSION_REPLAY_OR_IDENTITY",
        )?;
        let inserted = match self.ledger.0.lock() {
            Ok(mut l) => l.insert(t.request.clone()),
            Err(_) => false,
        };
        self.require(inserted, "STOP_ADMISSION_REPLAY_OR_IDENTITY")?;
        self.current = Some(t.clone());
        self.r.phase = Phase::InFlight;
        Ok(())
    }
    fn active(&self, t: &Ticket) -> bool {
        self.r.phase == Phase::InFlight && self.current.as_ref() == Some(t)
    }
    pub fn submission_entered(&mut self, t: &Ticket) -> Result<()> {
        self.require(
            self.active(t) && self.r.work == Work::None,
            "STOP_SUBMISSION_ORDER",
        )?;
        self.r.work = Work::SubmissionUncertain;
        Ok(())
    }
    pub fn submission_confirmed(&mut self, t: &Ticket) -> Result<()> {
        self.require(
            self.active(t) && self.r.work == Work::SubmissionUncertain,
            "STOP_SUBMISSION_CONFIRMATION",
        )?;
        self.r.work = Work::Submitted;
        Ok(())
    }
    pub fn fence(&mut self, t: &Ticket) -> Result<()> {
        self.require(
            self.active(t) && self.r.work == Work::Submitted,
            "STOP_FENCE_CONFIRMATION",
        )?;
        self.r.work = Work::Fenced;
        Ok(())
    }
    pub fn prepare(&mut self, t: &Ticket, hash: &str) -> Result<()> {
        self.require(
            self.active(t)
                && self.r.work == Work::Fenced
                && self.r.delivery == Delivery::None
                && digest(hash),
            "STOP_OUTPUT_UNCOMMITTED",
        )?;
        self.output = Some(hash.to_owned());
        self.r.delivery = Delivery::Prepared;
        Ok(())
    }
    pub fn begin_delivery(&mut self, t: &Ticket, hash: &str) -> Result<()> {
        self.require(
            self.active(t)
                && self.r.delivery == Delivery::Prepared
                && self.output.as_deref() == Some(hash),
            "STOP_DELIVERY_COMMITMENT",
        )?;
        self.r.delivery = Delivery::SentUnacknowledged;
        Ok(())
    }
    pub fn acknowledge(&mut self, t: &Ticket, hash: &str) -> Result<()> {
        self.require(
            self.active(t)
                && self.r.delivery == Delivery::SentUnacknowledged
                && self.output.as_deref() == Some(hash),
            "STOP_DELIVERY_ACK",
        )?;
        self.r.delivery = Delivery::Acknowledged;
        self.r.phase = Phase::Draining;
        Ok(())
    }
    pub fn begin_drain(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Draining
                && self.r.owner == Owner::Confirmed
                && !self.r.unknown_ownership
                && self.r.work == Work::Fenced
                && self.r.delivery == Delivery::Acknowledged
                && self.r.cleanup == Cleanup::NotAttempted
                && self.r.drain_attempts == 0,
            "STOP_DRAIN_UNCONFIRMED",
        )?;
        self.r.drain_attempts = 1;
        Ok(())
    }
    pub fn confirm_drain(&mut self, success: bool) -> Result<()> {
        self.require(
            self.r.phase == Phase::Draining
                && self.r.drain_attempts == 1
                && self.r.cleanup == Cleanup::NotAttempted,
            "STOP_DRAIN_UNCONFIRMED",
        )?;
        self.require(success, "STOP_DRAIN_FAILED")?;
        self.r.cleanup = Cleanup::DrainChecked;
        Ok(())
    }
    pub fn drain(&mut self, success: bool) -> Result<()> {
        self.begin_drain()?;
        self.confirm_drain(success)
    }
    pub fn begin_release(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Draining
                && self.r.cleanup == Cleanup::DrainChecked
                && self.r.owner == Owner::Confirmed
                && self.r.release_attempts == 0,
            "STOP_RELEASE_UNCONFIRMED",
        )?;
        self.r.release_attempts = 1;
        Ok(())
    }
    pub fn confirm_release(&mut self, success: bool) -> Result<()> {
        self.require(
            self.r.phase == Phase::Draining
                && self.r.release_attempts == 1
                && self.r.cleanup == Cleanup::DrainChecked,
            "STOP_RELEASE_UNCONFIRMED",
        )?;
        self.require(success, "STOP_DESTRUCTOR_FAILED")?;
        self.r.cleanup = Cleanup::ReleaseCheckedScoped;
        self.r.phase = Phase::Terminating;
        Ok(())
    }
    pub fn release(&mut self, success: bool) -> Result<()> {
        self.begin_release()?;
        self.confirm_release(success)
    }
    pub fn terminate_quarantined(&mut self) -> Result<()> {
        self.require(self.r.phase == Phase::Quarantined, "STOP_TERMINATION_ORDER")?;
        self.r.phase = Phase::Terminating;
        Ok(())
    }
    pub fn observe_exit(&mut self, id: Identity) -> Result<()> {
        self.require(
            self.r.phase == Phase::Terminating && id == self.r.identity,
            "STOP_EXIT_IDENTITY",
        )?;
        self.r.process_exit_observed = true;
        self.r.phase = Phase::Exited;
        Ok(())
    }
    pub fn exit_unconfirmed(&mut self) -> Result<()> {
        self.require(
            self.r.phase == Phase::Terminating,
            "STOP_EXIT_UNCONFIRMED_ORDER",
        )?;
        self.r.phase = Phase::ExitUnconfirmed;
        Ok(())
    }
    /// Synthetic supplied observations only. Never a physical GPU attestation.
    pub fn synthetic_baseline(&mut self, samples: &[(u64, u32)], baseline: u64) -> Result<()> {
        self.require(
            self.r.phase == Phase::Exited
                && self.r.process_exit_observed
                && self.r.post_exit_baseline == Observation::Unknown
                && samples.len() == 8
                && samples.iter().all(|s| s.1 == 0)
                && samples.iter().skip(4).all(|s| s.0.abs_diff(baseline) <= 2),
            "STOP_POST_EXIT_OBSERVATION",
        )?;
        self.r.post_exit_baseline = Observation::SyntheticBaselineObserved;
        Ok(())
    }
}
/// Every frame binds version, child PID, generation, source/model/KV. Request
/// events also echo the request ID and nonce. This is not adversarial IPC auth.
pub fn frame(id: Identity, b: &Binding, body: &str) -> String {
    format!(
        "A3S1 {} {} {} {} {} {}\n",
        id.pid, id.generation, b.source, b.model, b.kv, body
    )
}
pub fn parse_frame(line: &str, id: Identity, b: &Binding) -> Result<String> {
    let prefix = frame(id, b, "");
    if line.len() > MAX_FRAME
        || !line.ends_with('\n')
        || line[..line.len() - 1].contains(['\n', '\r'])
    {
        return Err("STOP_MALFORMED_IPC");
    }
    let body = line
        .strip_prefix(prefix.trim_end_matches('\n'))
        .ok_or("STOP_PROTOCOL_IDENTITY")?;
    let body = body.trim_end_matches('\n');
    if body.is_empty() || !body.is_ascii() {
        return Err("STOP_MALFORMED_IPC");
    }
    Ok(body.to_owned())
}
fn check(deadline: Instant, cancel: &AtomicBool) -> Result<()> {
    check_with_clock(deadline, cancel, Instant::now)
}
fn check_with_clock(
    deadline: Instant,
    cancel: &AtomicBool,
    now: impl FnOnce() -> Instant,
) -> Result<()> {
    if cancel.load(Ordering::SeqCst) {
        Err("STOP_CANCELLED")
    } else if now() >= deadline {
        Err("STOP_TIMEOUT")
    } else {
        Ok(())
    }
}
fn poll_wait(deadline: Instant, cancel: &AtomicBool) -> Result<()> {
    check(deadline, cancel)?;
    thread::sleep(POLL.min(deadline.saturating_duration_since(Instant::now())));
    Ok(())
}
/// Read one bounded frame for the blocking fake-worker caller, restoring its
/// blocking writes on return. The supervisor uses its already nonblocking socket
/// directly through read_line_poll, without changing socket options per frame.
pub fn read_line(
    stream: &mut UnixStream,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<String> {
    check(deadline, cancel)?;
    stream.set_nonblocking(true).map_err(|_| "STOP_IPC_IO")?;
    let result = read_line_poll(stream, deadline, cancel);
    let restore = stream.set_nonblocking(false).map_err(|_| "STOP_IPC_IO");
    // A mode-restoration error must not replace the original read failure.
    result.and_then(|line| restore.map(|()| line))
}
// The generic I/O boundary permits deterministic error/progress regression tests
// against these same loops; production callers supply a nonblocking UnixStream.
pub(crate) fn read_line_poll(
    stream: &mut impl Read,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<String> {
    let mut data = Vec::new();
    loop {
        check(deadline, cancel)?;
        let mut byte = [0];
        match stream.read(&mut byte) {
            Ok(0) => {
                return Err(if data.is_empty() {
                    "STOP_MISSING_RECEIPT"
                } else {
                    "STOP_PARTIAL_IPC"
                })
            }
            Ok(_) => {
                check(deadline, cancel)?;
                data.push(byte[0]);
                if data.len() > MAX_FRAME {
                    return Err("STOP_MALFORMED_IPC");
                }
                if byte[0] == b'\n' {
                    return String::from_utf8(data).map_err(|_| "STOP_MALFORMED_IPC");
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) =>
            {
                poll_wait(deadline, cancel)?;
            }
            Err(_) => return Err("STOP_IPC_IO"),
        }
    }
}
pub(crate) fn send(
    stream: &mut impl Write,
    data: &str,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<()> {
    if data.len() > MAX_FRAME {
        return Err("STOP_MALFORMED_IPC");
    }
    let mut left = data.as_bytes();
    while !left.is_empty() {
        check(deadline, cancel)?;
        match stream.write(left) {
            Ok(0) => return Err("STOP_IPC_IO"),
            Ok(n) => {
                check(deadline, cancel)?;
                left = &left[n..];
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) =>
            {
                poll_wait(deadline, cancel)?;
            }
            Err(_) => return Err("STOP_IPC_IO"),
        }
    }
    Ok(())
}
#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    Normal,
    BeforeRetain,
    AfterRetain,
    AfterOwner,
    ForgedReleased,
    WrongProtocol,
    WrongIdentity,
    WrongGeneration,
    Hang,
    Blocked,
    AbnormalExit,
    MissingFinal,
    Malformed,
    Partial,
    Oversized,
    BeforeEnqueue,
    AfterEnqueue,
    FenceFailure,
    MutatedOutput,
    WrongTicket,
    DrainAbsent,
    DrainFailure,
    DestructorFailure,
}
impl Scenario {
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::BeforeRetain => "before-retain",
            Self::AfterRetain => "after-retain",
            Self::AfterOwner => "after-owner",
            Self::ForgedReleased => "forged-released",
            Self::WrongProtocol => "wrong-protocol",
            Self::WrongIdentity => "wrong-identity",
            Self::WrongGeneration => "wrong-generation",
            Self::Hang => "hang",
            Self::Blocked => "blocked",
            Self::AbnormalExit => "abnormal-exit",
            Self::MissingFinal => "missing-final",
            Self::Malformed => "malformed",
            Self::Partial => "partial",
            Self::Oversized => "oversized",
            Self::BeforeEnqueue => "before-enqueue",
            Self::AfterEnqueue => "after-enqueue",
            Self::FenceFailure => "fence-failure",
            Self::MutatedOutput => "mutated-output",
            Self::WrongTicket => "wrong-ticket",
            Self::DrainAbsent => "drain-absent",
            Self::DrainFailure => "drain-failure",
            Self::DestructorFailure => "destructor-failure",
        }
    }
}
#[derive(Clone, Debug)]
pub struct ClientChunk {
    pub ticket: Ticket,
    pub digest: String,
    pub bytes: Vec<u8>,
    pub final_chunk: bool,
}
#[derive(Clone, Debug)]
pub struct ClientAck {
    pub ticket: Ticket,
    pub digest: String,
}
#[derive(Debug)]
pub struct ProcessReport {
    pub receipt: Receipt,
    pub exit_status: Option<String>,
    pub kill_attempted: bool,
    pub reaped: bool,
    pub elapsed_ms: u128,
    pub containment_ms: u128,
}
// Internal observation/clock seam for deterministic success-boundary tests.
// Both entry points run the same supervisor body and real channel/Child calls.
// Production uses the wall clock and no-op observers; no public API is added.
pub(crate) trait SuccessBoundary {
    fn now(&mut self, _deadline: Instant) -> Instant {
        Instant::now()
    }
    fn ack_wait_returned(
        &mut self,
        _result: &std::result::Result<ClientAck, RecvTimeoutError>,
        _wait: Duration,
        _delivery: Delivery,
        _deadline: Instant,
    ) {
    }
    fn exit_wait_returned(
        &mut self,
        _result: &std::io::Result<Option<ExitStatus>>,
        _deadline: Instant,
    ) {
    }
    fn sending(&mut self, _body: &str) {}
}
struct LiveBoundary;
impl SuccessBoundary for LiveBoundary {}
struct SocketDir(PathBuf);
static SOCKET_COUNTER: AtomicU64 = AtomicU64::new(1);
impl SocketDir {
    fn create() -> Result<Self> {
        for _ in 0..32 {
            // Short Darwin sockaddr_un path, private atomic directory, no reuse.
            let path = PathBuf::from(format!(
                "/tmp/mer2-a3-{}-{}",
                std::process::id(),
                SOCKET_COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err("STOP_SOCKET_CREATE"),
            }
        }
        Err("STOP_SOCKET_CREATE")
    }
    fn socket(&self) -> PathBuf {
        self.0.join("ipc")
    }
}
impl Drop for SocketDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.socket());
        let _ = std::fs::remove_dir(&self.0);
    }
}
struct OwnedChild {
    child: Child,
    reaped: bool,
    containment_attempted: bool,
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        // Panic/early-return safety only. Normal containment is bounded below and
        // is never silently retried or relabeled after an unconfirmed reap.
        if !self.reaped && !self.containment_attempted {
            let _ = self.child.kill();
            let end = Instant::now() + REAP_LIMIT;
            while Instant::now() < end {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    self.reaped = true;
                    break;
                }
                thread::sleep(POLL);
            }
        }
    }
}
/// In-memory tombstones survive worker replacement, not supervisor restart.
/// No replay operation. Synchronous execution polls cancellation and deadlines.
#[derive(Default)]
pub struct Supervisor {
    ledger: Ledger,
    generation: u64,
    unconfirmed: Vec<OwnedChild>,
}
impl Supervisor {
    pub fn unconfirmed_children(&self) -> usize {
        self.unconfirmed.len()
    }
    pub fn next_generation(&self) -> Option<u64> {
        self.generation.checked_add(1)
    }
    pub fn execute(
        &mut self,
        executable: &Path,
        binding: Binding,
        scenario: Scenario,
        ticket: Ticket,
        output: SyncSender<ClientChunk>,
        ack: Receiver<ClientAck>,
        cancel: Arc<AtomicBool>,
    ) -> Result<ProcessReport> {
        self.execute_with_boundary(
            executable,
            binding,
            scenario,
            ticket,
            output,
            ack,
            cancel,
            &mut LiveBoundary,
        )
    }
    pub(crate) fn execute_with_boundary(
        &mut self,
        executable: &Path,
        binding: Binding,
        scenario: Scenario,
        ticket: Ticket,
        output: SyncSender<ClientChunk>,
        ack: Receiver<ClientAck>,
        cancel: Arc<AtomicBool>,
        boundary: &mut impl SuccessBoundary,
    ) -> Result<ProcessReport> {
        if !binding.valid() {
            return Err("STOP_INITIAL_IDENTITY");
        }
        if !self.unconfirmed.is_empty() {
            return Err("STOP_PREVIOUS_EXIT_UNCONFIRMED");
        }
        self.generation = self.next_generation().ok_or("STOP_GENERATION_EXHAUSTED")?;
        let started = Instant::now();
        let deadline = started + SESSION_LIMIT;
        let dir = SocketDir::create()?;
        let listener = UnixListener::bind(dir.socket()).map_err(|_| "STOP_SOCKET_BIND")?;
        listener
            .set_nonblocking(true)
            .map_err(|_| "STOP_SOCKET_BIND")?;
        let child = Command::new(executable)
            .arg(dir.socket())
            .arg(self.generation.to_string())
            .arg(scenario.label())
            .arg(&binding.source)
            .arg(&binding.model)
            .arg(&binding.kv)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| "STOP_SPAWN")?;
        let id = Identity {
            pid: child.id(),
            generation: self.generation,
        };
        let mut owned = OwnedChild {
            child,
            reaped: false,
            containment_attempted: false,
        };
        let mut c = Contract::new(id, binding.clone(), self.ledger.clone())?;
        let outcome = (|| {
            let mut stream = loop {
                check(deadline, &cancel)?;
                match listener.accept() {
                    Ok((stream, _)) => {
                        stream.set_nonblocking(true).map_err(|_| "STOP_IPC_IO")?;
                        break stream;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        poll_wait(deadline, &cancel)?;
                    }
                    Err(_) => return Err("STOP_IPC_IO"),
                }
            };
            let mut p = Protocol {
                stream: &mut stream,
                id,
                binding: &binding,
                deadline,
                cancel: &cancel,
                boundary,
            };
            p.expect("HELLO")?;
            p.send("INIT")?;
            p.expect("ENTER")?;
            // Before the command that permits the fallible fake constructor.
            c.constructor_entered()?;
            p.send("CONSTRUCT")?;
            p.expect("OWNER")?;
            c.typed_owner_returned()?;
            p.expect("CHECKED")?;
            c.deferred_check()?;
            p.expect("READY")?;
            c.ready()?;
            c.admit(&ticket)?;
            let tag = format!("{} {}", ticket.request, ticket.nonce);
            p.send(&format!("ADMIT {tag}"))?;
            p.expect(&format!("ACCEPT {tag}"))?;
            c.submission_entered(&ticket)?;
            p.send(&format!("SUBMIT {tag}"))?;
            p.expect(&format!("ENQUEUED {tag}"))?;
            c.submission_confirmed(&ticket)?;
            p.expect(&format!("FENCED {tag}"))?;
            c.fence(&ticket)?;
            p.expect(&format!("PREPARED {tag} {FAKE_DIGEST}"))?;
            c.prepare(&ticket, FAKE_DIGEST)?;
            let data = p.receive()?;
            let prefix = format!("DATA {tag} ");
            let bytes = data
                .strip_prefix(&prefix)
                .ok_or("STOP_TICKET_IDENTITY")?
                .as_bytes();
            if sha256(bytes) != FAKE_DIGEST {
                return Err("STOP_OUTPUT_DIGEST");
            }
            check(deadline, &cancel)?;
            c.begin_delivery(&ticket, FAKE_DIGEST)?;
            for (index, part) in [bytes[..1].to_vec(), bytes[1..].to_vec()]
                .into_iter()
                .enumerate()
            {
                let mut chunk = ClientChunk {
                    ticket: ticket.clone(),
                    digest: FAKE_DIGEST.into(),
                    bytes: part,
                    final_chunk: index == 1,
                };
                loop {
                    check(deadline, &cancel)?;
                    match output.try_send(chunk) {
                        Ok(()) => break,
                        Err(TrySendError::Full(v)) => {
                            chunk = v;
                            thread::sleep(POLL);
                        }
                        Err(TrySendError::Disconnected(_)) => return Err("STOP_CLIENT_DISCONNECT"),
                    }
                }
            }
            let client_ack = loop {
                check_with_clock(deadline, &cancel, || p.boundary.now(deadline))?;
                let wait = POLL.min(deadline.saturating_duration_since(p.boundary.now(deadline)));
                let received = ack.recv_timeout(wait);
                p.boundary
                    .ack_wait_returned(&received, wait, c.r.delivery, deadline);
                match received {
                    Ok(v) => {
                        check_with_clock(deadline, &cancel, || p.boundary.now(deadline))?;
                        break v;
                    }
                    Err(RecvTimeoutError::Timeout) => (),
                    Err(RecvTimeoutError::Disconnected) => return Err("STOP_CLIENT_DISCONNECT"),
                }
            };
            c.acknowledge(&client_ack.ticket, &client_ack.digest)?;
            p.send(&format!("ACK {tag} {FAKE_DIGEST}"))?;
            c.begin_drain()?;
            p.send("DRAIN")?;
            let drain = p.receive()?;
            c.confirm_drain(drain == "DRAINED")?;
            c.begin_release()?;
            p.send("RELEASE")?;
            let release = p.receive()?;
            c.confirm_release(release == "RELEASED")?;
            p.send("EXIT")?;
            p.expect("BYE")?;
            Ok(())
        })();
        if let Err(e) = outcome {
            if c.r.first_error.is_none() {
                let _ = c.fault(e);
            }
        }
        let containment = Instant::now();
        let mut kill_attempted = false;
        let mut status = None;
        if outcome.is_ok() {
            loop {
                let waited = owned.child.try_wait();
                boundary.exit_wait_returned(&waited, deadline);
                match waited {
                    Ok(Some(s)) => {
                        status = Some(s);
                        owned.reaped = true;
                        // Physical reap survives a late/cancelled logical STOP.
                        if let Err(e) =
                            check_with_clock(deadline, &cancel, || boundary.now(deadline))
                        {
                            let _ = c.fault(e);
                        }
                        break;
                    }
                    Ok(None) => {
                        if let Err(e) = check(deadline, &cancel) {
                            let _ = c.fault(e);
                            break;
                        }
                    }
                    Err(_) => {
                        let _ = c.fault("STOP_WAIT_IO");
                        break;
                    }
                }
                thread::sleep(POLL);
            }
        }
        if !owned.reaped {
            owned.containment_attempted = true;
            if c.r.phase == Phase::Quarantined {
                let _ = c.terminate_quarantined();
            }
            match owned.child.try_wait() {
                Ok(Some(s)) => {
                    status = Some(s);
                    owned.reaped = true;
                }
                Ok(None) => {
                    kill_attempted = true;
                    if owned.child.kill().is_err() {
                        let _ = c.fault("STOP_KILL_IO");
                    }
                }
                Err(_) => {
                    let _ = c.fault("STOP_WAIT_IO");
                }
            }
            let reap_end = Instant::now() + REAP_LIMIT;
            while !owned.reaped && Instant::now() < reap_end {
                match owned.child.try_wait() {
                    Ok(Some(s)) => {
                        status = Some(s);
                        owned.reaped = true;
                    }
                    Ok(None) => thread::sleep(POLL),
                    Err(_) => {
                        let _ = c.fault("STOP_REAP_IO");
                        break;
                    }
                }
            }
        }
        if let Some(s) = &status {
            if !s.success() {
                let _ = c.fault("STOP_CHILD_ABNORMAL_EXIT");
            }
            if c.r.phase == Phase::Quarantined {
                let _ = c.terminate_quarantined();
            }
            c.observe_exit(id)?;
        } else {
            let _ = c.fault("STOP_REAP_UNCONFIRMED");
            let _ = c.terminate_quarantined();
            c.exit_unconfirmed()?;
        }
        let report = ProcessReport {
            receipt: c.receipt(),
            exit_status: status.map(|s| s.to_string()),
            kill_attempted,
            reaped: owned.reaped,
            elapsed_ms: started.elapsed().as_millis(),
            containment_ms: containment.elapsed().as_millis(),
        };
        if !owned.reaped {
            self.unconfirmed.push(owned);
        }
        Ok(report)
    }
}
struct Protocol<'a, B> {
    stream: &'a mut UnixStream,
    id: Identity,
    binding: &'a Binding,
    deadline: Instant,
    cancel: &'a AtomicBool,
    boundary: &'a mut B,
}
impl<B: SuccessBoundary> Protocol<'_, B> {
    fn send(&mut self, body: &str) -> Result<()> {
        self.boundary.sending(body);
        send(
            self.stream,
            &frame(self.id, self.binding, body),
            self.deadline,
            self.cancel,
        )
    }
    fn receive(&mut self) -> Result<String> {
        let line = read_line_poll(self.stream, self.deadline, self.cancel)?;
        let body = parse_frame(&line, self.id, self.binding)?;
        match body.as_str() {
            "FAIL_PRE_RETAIN" => Err("STOP_BEFORE_RETAIN"),
            "FAIL_RETAIN" => Err("STOP_UNRETURNED_RETAIN"),
            "FAIL_DEFERRED" => Err("STOP_DEFERRED_CHECK"),
            "FAIL_PRE_ENQUEUE" => Err("STOP_BEFORE_ENQUEUE"),
            "FAIL_POST_ENQUEUE" => Err("STOP_ENQUEUE_AMBIGUOUS"),
            "FAIL_FENCE" => Err("STOP_FENCE_UNCONFIRMED"),
            _ => Ok(body),
        }
    }
    fn expect(&mut self, expected: &str) -> Result<()> {
        if self.receive()? == expected {
            Ok(())
        } else {
            Err("STOP_PROTOCOL_ORDER_OR_TICKET")
        }
    }
}
/// SHA-256 of transported bytes, standard algorithm; known-answer tested.
pub fn sha256(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut data = input.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bits.to_be_bytes());
    for block in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, b) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(b.try_into().unwrap());
        }
        for i in 16..64 {
            let x = w[i - 15];
            let y = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10));
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut z] = h;
        for i in 0..64 {
            let t = z
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let u = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            z = g;
            g = f;
            f = e;
            e = d.wrapping_add(t);
            d = c;
            c = b;
            b = a;
            a = t.wrapping_add(u);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, z]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}
