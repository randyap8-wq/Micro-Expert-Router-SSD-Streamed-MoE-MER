//! CPU-only parent-owned process contract for exactly one mocked FFN request.
//! R4's real state machine remains the authoritative ownership/delivery guard.
use crate::wire::{self, Codec, Kind};
use crate::{Result, D, VECTOR_BYTES, MOCK_KIND};
use mer2_g3_a3_s1::{Binding, Cleanup, ClientAck, ClientChunk, Contract, Delivery, Identity, Ledger, Owner, Phase, Receipt, Ticket, Work};
use mer2_g3_a3_s1::sha256;
use std::fs::{DirBuilder, File};
use std::io::Read;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_millis(1);
const REAP_LIMIT: Duration = Duration::from_secs(2);
const MAX_SESSION: Duration = Duration::from_secs(30);
pub const SCHEMA_DOMAIN: &[u8] = b"A3N1-MOCK-WIRE-v0.1-2026-10-27";

fn digest_valid(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}
fn push_field(dst: &mut Vec<u8>, content: &[u8]) -> Result<()> {
    let len: u32 = content.len().try_into().map_err(|_| "STOP_WORK_FIELD_LENGTH")?;
    dst.extend_from_slice(&len.to_le_bytes());
    dst.extend_from_slice(content);
    Ok(())
}
fn push_u64(dst: &mut Vec<u8>, value: u64) { dst.extend_from_slice(&value.to_le_bytes()); }
fn push_u32(dst: &mut Vec<u8>, value: u32) { dst.extend_from_slice(&value.to_le_bytes()); }

/// Mirror of the SOURCE-VERIFIED public S1 Ticket fields. This is explicitly
/// **not** S1's private Ticket::encode and has no S1 runtime provider authority.
#[derive(Clone, Debug)]
pub struct S1WireTicket {
    pub session: u64,
    pub contract_sha256: String,
    pub request: String,
    pub nonce: u64,
    pub attempt: u64,
    pub position: u64,
    pub input_id: u32,
    pub kv_generation: u64,
    pub kv_sha256: String,
    pub source_sha256: String,
    pub model_sha256: String,
}
#[derive(Clone, Debug)]
pub struct MockWork {
    pub r4: Ticket,
    pub s1: S1WireTicket,
    pub layer: u32,
    pub rank: u32,
    pub expert: u32,
    pub input_sha256: String,
    pub weight_commitment_sha256: String,
}
impl MockWork {
    pub fn validate(&self, binding: &Binding, input: &[u8]) -> Result<()> {
        if self.r4.binding != *binding || self.r4.generation == 0 || self.r4.nonce == 0
            || self.r4.request.is_empty() || self.r4.request.len() > 128
            || !self.r4.request.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            || self.s1.request != self.r4.request || self.s1.nonce != self.r4.nonce
            || self.s1.session == 0 || self.s1.attempt == 0
            || self.s1.position >= 4 || self.s1.input_id >= 32
            || self.s1.kv_generation == 0
            || self.s1.source_sha256 != binding.source || self.s1.model_sha256 != binding.model
            || self.s1.kv_sha256 != binding.kv
            || self.layer >= 2 || self.rank >= 2 || self.expert >= 4
            || !digest_valid(&self.s1.contract_sha256)
            || !digest_valid(&self.weight_commitment_sha256)
            || !digest_valid(&self.input_sha256) {
            return Err("STOP_WORK_BINDING");
        }
        if input.len() != VECTOR_BYTES || sha256(input) != self.input_sha256 {
            return Err("STOP_WORK_INPUT_DIGEST");
        }
        MockVector::decode(input)?;
        Ok(())
    }
    /// Canonical A3N1 wire-request encoding. Independent of S1's private encoder.
    pub fn canonical(&self) -> Result<Vec<u8>> {
        let mut raw = b"MER2-A3N1-WORK-v1\0".to_vec();
        push_field(&mut raw, self.r4.request.as_bytes())?;
        push_u64(&mut raw, self.r4.generation);
        push_u64(&mut raw, self.r4.nonce);
        push_field(&mut raw, self.r4.binding.source.as_bytes())?;
        push_field(&mut raw, self.r4.binding.model.as_bytes())?;
        push_field(&mut raw, self.r4.binding.kv.as_bytes())?;
        push_u64(&mut raw, self.s1.session);
        push_field(&mut raw, self.s1.contract_sha256.as_bytes())?;
        push_field(&mut raw, self.s1.request.as_bytes())?;
        push_u64(&mut raw, self.s1.nonce);
        push_u64(&mut raw, self.s1.attempt);
        push_u64(&mut raw, self.s1.position);
        push_u32(&mut raw, self.s1.input_id);
        push_u64(&mut raw, self.s1.kv_generation);
        push_field(&mut raw, self.s1.kv_sha256.as_bytes())?;
        push_field(&mut raw, self.s1.source_sha256.as_bytes())?;
        push_field(&mut raw, self.s1.model_sha256.as_bytes())?;
        push_u32(&mut raw, self.layer);
        push_u32(&mut raw, self.rank);
        push_u32(&mut raw, self.expert);
        push_field(&mut raw, self.input_sha256.as_bytes())?;
        push_field(&mut raw, self.weight_commitment_sha256.as_bytes())?;
        if raw.len() > wire::MAX_CONTROL { return Err("STOP_WORK_SIZE"); }
        Ok(raw)
    }
    pub fn commitment(&self) -> Result<String> { Ok(sha256(&self.canonical()?)) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MockVector([f32; D]);
impl MockVector {
    pub fn checked(values: [f32; D]) -> Result<Self> {
        if values.iter().any(|f| !f.is_finite()) { return Err("STOP_TENSOR_NONFINITE"); }
        Ok(Self(values))
    }
    pub fn bytes(&self) -> [u8; VECTOR_BYTES] {
        let mut result = [0u8; VECTOR_BYTES];
        for (i, v) in self.0.iter().enumerate() {
            result[i * 4..i * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        result
    }
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() != VECTOR_BYTES { return Err("STOP_TENSOR_LENGTH"); }
        let mut values = [0.0f32; D];
        for (i, value) in values.iter_mut().enumerate() {
            let raw: [u8; 4] = data[i * 4..i * 4 + 4].try_into().map_err(|_| "STOP_TENSOR_LENGTH")?;
            *value = f32::from_bits(u32::from_le_bytes(raw));
            if !value.is_finite() { return Err("STOP_TENSOR_NONFINITE"); }
        }
        Ok(Self(values))
    }
    pub fn sha256(&self) -> String { sha256(&self.bytes()) }
}

/// V1 mock descriptor is exactly 104 bytes: [dtype,rank,layout,flags],
/// four dims, four strides, byte length, payload SHA256, work SHA256.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub payload_sha256: [u8; 32],
    pub work_sha256: [u8; 32],
}
impl TensorDescriptor {
    pub const LEN: usize = 104;
    pub fn for_mock(payload: &[u8], work_sha256: [u8; 32]) -> Result<Self> {
        MockVector::decode(payload)?;
        Ok(Self { payload_sha256: wire::digest_bytes(payload), work_sha256 })
    }
    pub fn encode(&self) -> [u8; Self::LEN] {
        let mut raw = [0u8; Self::LEN];
        raw[0] = 1; // f32-le
        raw[1] = 1; // rank one
        raw[2] = 1; // contiguous layout ID
        // flags[3] = 0
        raw[4..8].copy_from_slice(&(D as u32).to_le_bytes());
        raw[20..24].copy_from_slice(&4u32.to_le_bytes());
        raw[36..40].copy_from_slice(&(VECTOR_BYTES as u32).to_le_bytes());
        raw[40..72].copy_from_slice(&self.payload_sha256);
        raw[72..104].copy_from_slice(&self.work_sha256);
        raw
    }
    pub fn decode(raw: &[u8]) -> Result<Self> {
        if raw.len() != Self::LEN { return Err("STOP_TENSOR_DESCRIPTOR_LENGTH"); }
        let mut canonical = [0u8; Self::LEN];
        canonical[0] = 1;
        canonical[1] = 1;
        canonical[2] = 1;
        canonical[4..8].copy_from_slice(&(D as u32).to_le_bytes());
        canonical[20..24].copy_from_slice(&4u32.to_le_bytes());
        canonical[36..40].copy_from_slice(&(VECTOR_BYTES as u32).to_le_bytes());
        if raw[..40] != canonical[..40] { return Err("STOP_TENSOR_DESCRIPTOR_LAYOUT"); }
        Ok(Self {
            payload_sha256: raw[40..72].try_into().map_err(|_| "STOP_TENSOR_DESCRIPTOR_LENGTH")?,
            work_sha256: raw[72..104].try_into().map_err(|_| "STOP_TENSOR_DESCRIPTOR_LENGTH")?,
        })
    }
    pub fn verify(&self, payload: &[u8], expected_work: &[u8; 32]) -> Result<()> {
        MockVector::decode(payload)?;
        if self.work_sha256 != *expected_work || self.payload_sha256 != wire::digest_bytes(payload) {
            return Err("STOP_TENSOR_DESCRIPTOR_BINDING");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockScenario {
    Normal,
    WrongPid,
    MutatedOutput,
    PostSubmitExit,
    HangAfterSubmit,
    DrainFailure,
    ReleaseFailure,
    BeforeRetain,
    UnreturnedRetain,
    TrailingByte,
    DuplicateBye,
    TrailingFrame,
    DelayedTrailingByte,
    SelfConsistentWrongOutput,
    MissingBye,
    AbnormalAfterBye,
}
impl MockScenario {
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::WrongPid => "wrong-pid",
            Self::MutatedOutput => "mutated-output",
            Self::PostSubmitExit => "post-submit-exit",
            Self::HangAfterSubmit => "hang-after-submit",
            Self::DrainFailure => "drain-failure",
            Self::ReleaseFailure => "release-failure",
            Self::BeforeRetain => "before-retain",
            Self::UnreturnedRetain => "unreturned-retain",
            Self::TrailingByte => "trailing-byte",
            Self::DuplicateBye => "duplicate-bye",
            Self::TrailingFrame => "trailing-frame",
            Self::DelayedTrailingByte => "delayed-trailing-byte",
            Self::SelfConsistentWrongOutput => "self-consistent-wrong-output",
            Self::MissingBye => "missing-bye",
            Self::AbnormalAfterBye => "abnormal-after-bye",
        }
    }
}

#[derive(Debug)]
pub struct MockReport {
    pub receipt: Receipt,
    pub process_pid: u32,
    pub exit_status: Option<String>,
    pub process_exit_success: bool,
    pub reaped: bool,
    pub kill_attempted: bool,
    pub kill_attempt_count: u8,
    pub containment_errors: Vec<&'static str>,
    pub terminal_clean_eof: bool,
    pub mock_output_verified: bool,
    pub observed_stages: Vec<&'static str>,
    pub elapsed_ms: u128,
    pub kind: &'static str,
}
impl MockReport {
    /// The ONLY accepted success predicate for the bounded, trusted-parent CPU
    /// mock. Ok(MockReport), Exited, and reaped are *not* success predicates.
    pub fn is_success(&self) -> bool {
        let r = &self.receipt;
        self.kind == MOCK_KIND && self.reaped && self.process_exit_success
            && !self.kill_attempted && self.kill_attempt_count == 0
            && self.containment_errors.is_empty()
            && self.terminal_clean_eof && self.mock_output_verified
            && r.phase == Phase::Exited && r.owner == Owner::Confirmed
            && !r.unknown_ownership && r.work == Work::Fenced
            && r.delivery == Delivery::Acknowledged
            && r.cleanup == Cleanup::ReleaseCheckedScoped
            && r.drain_attempts == 1 && r.release_attempts == 1
            && r.process_exit_observed
            && r.first_error.is_none() && r.secondary_errors.is_empty()
            && r.last_cuda_context_drop == "UNOBSERVABLE"
            && r.leak_free == "NOT_CLAIMED" && r.automatic_retry == "FORBIDDEN"
    }
}

struct SocketDir(PathBuf);
impl SocketDir {
    fn new() -> Result<Self> {
        let mut nonce = [0u8; 16];
        File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut nonce))
            .map_err(|_| "STOP_SESSION_RANDOM")?;
        let hex: String = nonce.iter().map(|n| format!("{n:02x}")).collect();
        let dir = std::env::temp_dir().join(format!("a3n1-{}-{hex}", std::process::id()));
        DirBuilder::new().mode(0o700).create(&dir).map_err(|_| "STOP_SOCKET_DIRECTORY")?;
        Ok(Self(dir))
    }
    fn socket(&self) -> PathBuf { self.0.join("socket") }
}
impl Drop for SocketDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.socket());
        let _ = std::fs::remove_dir(&self.0);
    }
}
fn session_id() -> Result<[u8; 16]> {
    let mut nonce = [0u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut nonce))
        .map_err(|_| "STOP_SESSION_RANDOM")?;
    if nonce == [0u8; 16] { return Err("STOP_SESSION_RANDOM"); }
    Ok(nonce)
}
fn hex_session(session: &[u8; 16]) -> String {
    session.iter().map(|x| format!("{x:02x}")).collect()
}
fn check_hello(payload: &[u8], child_pid: u32, binary_sha: &str, schema_sha: &str) -> Result<()> {
    if payload.len() != 4 + 64 + 64 { return Err("STOP_HELLO_LENGTH"); }
    if payload[..4] != child_pid.to_le_bytes() || &payload[4..68] != binary_sha.as_bytes()
        || &payload[68..132] != schema_sha.as_bytes() {
        return Err("STOP_HELLO_IDENTITY");
    }
    Ok(())
}
fn init_binding(binding: &Binding, binary_sha: &str, schema_sha: &str) -> Vec<u8> {
    let mut raw = Vec::with_capacity(320);
    for x in [&binding.source, &binding.model, &binding.kv, binary_sha, schema_sha] {
        raw.extend_from_slice(x.as_bytes());
    }
    raw
}
fn accept(listener: &UnixListener, deadline: Instant, cancel: &AtomicBool) -> Result<UnixStream> {
    loop {
        wire::check(deadline, cancel)?;
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(true).map_err(|_| "STOP_SOCKET_IO")?;
                wire::check(deadline, cancel)?;
                return Ok(stream);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(POLL),
            Err(_) => return Err("STOP_SOCKET_IO"),
        }
    }
}
fn ack_within(
    ack: &Receiver<ClientAck>, deadline: Instant, cancel: &AtomicBool,
) -> Result<ClientAck> {
    loop {
        wire::check(deadline, cancel)?;
        let wait = POLL.min(deadline.saturating_duration_since(Instant::now()));
        let response = ack.recv_timeout(wait);
        if let Ok(value) = response {
            wire::check(deadline, cancel)?;
            return Ok(value);
        }
        match response {
            Err(RecvTimeoutError::Timeout) => (),
            Err(RecvTimeoutError::Disconnected) => return Err("STOP_CLIENT_DISCONNECT"),
            Ok(_) => unreachable!(),
        }
    }
}
fn send_to_client(
    output: &SyncSender<ClientChunk>, ticket: &Ticket, data: &[u8], digest: &str,
    deadline: Instant, cancel: &AtomicBool,
) -> Result<()> {
    let mut chunk = ClientChunk {
        ticket: ticket.clone(), digest: digest.to_owned(), bytes: data.to_vec(), final_chunk: true,
    };
    loop {
        wire::check(deadline, cancel)?;
        match output.try_send(chunk) {
            Ok(()) => { wire::check(deadline, cancel)?; return Ok(()); }
            Err(TrySendError::Full(v)) => { chunk = v; thread::sleep(POLL); }
            Err(TrySendError::Disconnected(_)) => return Err("STOP_CLIENT_DISCONNECT"),
        }
    }
}
/// Collect every observed containment error in occurrence order; never
/// replace the first error with a later wait/kill/reap error, and never retry
/// a failed kill just to manufacture a complete-looking report.
fn reap_child(
    child: &mut Child, successful_protocol: bool, deadline: Instant, cancel: &AtomicBool,
) -> (Option<std::process::ExitStatus>, bool, bool, Vec<&'static str>) {
    let mut status = None;
    let mut kill_attempted = false;
    let mut errors: Vec<&'static str> = Vec::with_capacity(4);
    if successful_protocol {
        loop {
            match child.try_wait() {
                Ok(Some(s)) => {
                    status = Some(s);
                    if let Err(e) = wire::check(deadline, cancel) { errors.push(e); }
                    break;
                }
                Ok(None) => {
                    if let Err(e) = wire::check(deadline, cancel) { errors.push(e); break; }
                }
                Err(_) => { errors.push("STOP_WAIT_IO"); break; }
            }
            thread::sleep(POLL);
        }
    }
    if status.is_none() {
        match child.try_wait() {
            Ok(Some(s)) => status = Some(s),
            Ok(None) => {
                kill_attempted = true;
                if child.kill().is_err() { errors.push("STOP_KILL_IO"); }
            }
            Err(_) => errors.push("STOP_WAIT_IO"),
        }
        let until = Instant::now() + REAP_LIMIT;
        while status.is_none() && Instant::now() < until {
            match child.try_wait() {
                Ok(Some(s)) => { status = Some(s); break; }
                Ok(None) => thread::sleep(POLL),
                Err(_) => { errors.push("STOP_REAP_IO"); break; }
            }
        }
    }
    let reaped = status.is_some();
    (status, reaped, kill_attempted, errors)
}

/// Parent is one-shot per instance and has no across-restart durable journal.
/// Test code must not manufacture a success receipt without Contract approval.
#[derive(Default)]
pub struct MockSupervisor { attempted: bool, ledger: Ledger, unconfirmed: Vec<Child> }
impl MockSupervisor {
    pub fn unconfirmed_children(&self) -> usize { self.unconfirmed.len() }
    /// Explicitly transfer unresolved direct children to a designated external
    /// process owner; handles are NOT silently killed/retried in Drop.
    pub fn take_unconfirmed_children(&mut self) -> Vec<Child> {
        std::mem::take(&mut self.unconfirmed)
    }
    pub fn execute_once(
        &mut self, executable: &Path, binding: Binding, work: MockWork, input: MockVector,
        output: SyncSender<ClientChunk>, ack: Receiver<ClientAck>, cancel: Arc<AtomicBool>,
        session_limit: Duration, scenario: MockScenario,
    ) -> Result<MockReport> {
        if self.attempted { return Err("STOP_REPLAY_FORBIDDEN"); }
        self.attempted = true; // Even pre-spawn ambiguity is not silently retried.
        // A pre-cancelled request must not hash an executable or spawn a child.
        if cancel.load(std::sync::atomic::Ordering::SeqCst) { return Err("STOP_CANCELLED"); }
        if session_limit.is_zero() || session_limit > MAX_SESSION { return Err("STOP_SESSION_LIMIT"); }
        let input_bytes = input.bytes();
        work.validate(&binding, &input_bytes)?;
        // Prove constructor input validity before spawning a child.
        let _ = Contract::new(Identity { pid: 1, generation: work.r4.generation }, binding.clone(), self.ledger.clone())?;
        let request_bytes = work.canonical()?;
        let work_hash = sha256(&request_bytes);
        let work_hash_bin = wire::digest_bytes(&request_bytes);
        let source = std::fs::read(executable).map_err(|_| "STOP_EXECUTABLE_READ")?;
        let binary_sha = sha256(&source);
        let schema_sha = sha256(SCHEMA_DOMAIN);
        let session = session_id()?;
        let dir = SocketDir::new()?;
        let listener = UnixListener::bind(dir.socket()).map_err(|_| "STOP_SOCKET_BIND")?;
        listener.set_nonblocking(true).map_err(|_| "STOP_SOCKET_BIND")?;
        let started = Instant::now();
        let deadline = started.checked_add(session_limit).ok_or("STOP_SESSION_LIMIT")?;
        wire::check(deadline, &cancel)?; // cancellation and budget before spawn
        let mut child = Command::new(executable)
            .arg(dir.socket())
            .arg(work.r4.generation.to_string())
            .arg(hex_session(&session))
            .arg(scenario.label())
            .arg(&binding.source)
            .arg(&binding.model)
            .arg(&binding.kv)
            .arg(&binary_sha)
            .arg(&schema_sha)
            .arg(&work_hash)
            .arg(&work.input_sha256)
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
            .spawn().map_err(|_| "STOP_SPAWN")?;
        let pid = child.id();
        let id = Identity { pid, generation: work.r4.generation };
        let mut c = Contract::new(id, binding.clone(), self.ledger.clone())?;
        let mut codec = Codec::new(id.generation, session)?;
        let mut observed_stages: Vec<&'static str> = vec!["SPAWNED"];
        let mut terminal_clean_eof = false;
        let mut mock_output_verified = false;
        let result = (|| {
            wire::check(deadline, &cancel)?; // owned child already exists: fail via reap
            let mut stream = accept(&listener, deadline, &cancel)?;
            let hello = codec.recv(&mut stream, deadline, &cancel)?;
            if hello.header.kind != Kind::Hello { return Err("STOP_HELLO_KIND"); }
            check_hello(&hello.payload, pid, &binary_sha, &schema_sha)?;
            observed_stages.push("HELLO");
            codec.send(&mut stream, Kind::Init, &init_binding(&binding, &binary_sha, &schema_sha), deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Enter, b"", deadline, &cancel)?;
            c.constructor_entered()?;
            codec.send(&mut stream, Kind::Construct, b"", deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Owner, b"", deadline, &cancel)?;
            c.typed_owner_returned()?;
            codec.expect(&mut stream, Kind::Checked, b"", deadline, &cancel)?;
            c.deferred_check()?;
            codec.expect(&mut stream, Kind::Ready, b"", deadline, &cancel)?;
            c.ready()?;
            observed_stages.push("READY");
            c.admit(&work.r4)?;
            codec.send(&mut stream, Kind::Admit, &request_bytes, deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Accept, work_hash.as_bytes(), deadline, &cancel)?;
            codec.send(&mut stream, Kind::DataInput, &input_bytes, deadline, &cancel)?;
            c.submission_entered(&work.r4)?;
            observed_stages.push("SUBMIT_ENTERED");
            codec.send(&mut stream, Kind::Submit, work_hash.as_bytes(), deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Enqueued, work_hash.as_bytes(), deadline, &cancel)?;
            c.submission_confirmed(&work.r4)?;
            observed_stages.push("ENQUEUED");
            codec.expect(&mut stream, Kind::Fenced, work_hash.as_bytes(), deadline, &cancel)?;
            c.fence(&work.r4)?;
            observed_stages.push("FENCED");
            let descriptor_frame = codec.recv(&mut stream, deadline, &cancel)?;
            if descriptor_frame.header.kind != Kind::ResultDescriptor { return Err("STOP_OUTPUT_DESCRIPTOR_KIND"); }
            let descriptor = TensorDescriptor::decode(&descriptor_frame.payload)?;
            let result = codec.recv(&mut stream, deadline, &cancel)?;
            if result.header.kind != Kind::DataOutput { return Err("STOP_OUTPUT_KIND"); }
            descriptor.verify(&result.payload, &work_hash_bin)?;
            // A self-consistent checksum cannot prove the declared mock identity
            // operation. Compare against the independent parent input oracle.
            if result.payload.as_slice() != input_bytes.as_slice() {
                return Err("STOP_MOCK_IDENTITY_MISMATCH");
            }
            mock_output_verified = true;
            observed_stages.push("OUTPUT_VERIFIED");
            let result_hash = sha256(&result.payload);
            c.prepare(&work.r4, &result_hash)?;
            wire::check(deadline, &cancel)?;
            c.begin_delivery(&work.r4, &result_hash)?;
            send_to_client(&output, &work.r4, &result.payload, &result_hash, deadline, &cancel)?;
            observed_stages.push("CLIENT_OUTPUT_SENT");
            let response = ack_within(&ack, deadline, &cancel)?;
            c.acknowledge(&response.ticket, &response.digest)?;
            observed_stages.push("ACKNOWLEDGED");
            codec.send(&mut stream, Kind::ClientAck, result_hash.as_bytes(), deadline, &cancel)?;
            c.begin_drain()?;
            codec.send(&mut stream, Kind::Drain, b"", deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Drained, b"", deadline, &cancel)?;
            c.confirm_drain(true)?;
            observed_stages.push("DRAINED");
            c.begin_release()?;
            codec.send(&mut stream, Kind::Release, b"", deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Released, b"", deadline, &cancel)?;
            c.confirm_release(true)?;
            observed_stages.push("RELEASED");
            codec.send(&mut stream, Kind::Exit, b"", deadline, &cancel)?;
            codec.expect(&mut stream, Kind::Bye, b"", deadline, &cancel)?;
            observed_stages.push("BYE");
            codec.expect_clean_eof(&mut stream, deadline, &cancel)?;
            terminal_clean_eof = true;
            observed_stages.push("CLEAN_EOF");
            Ok(())
        })();
        if let Err(e) = result { if c.receipt().first_error.is_none() { let _ = c.fault(e); } }
        let (status, reaped, kill_attempted, containment_errors) =
            reap_child(&mut child, result.is_ok(), deadline, &cancel);
        for &error in &containment_errors { let _ = c.fault(error); }
        if let Some(s) = &status {
            if !s.success() { let _ = c.fault("STOP_CHILD_ABNORMAL_EXIT"); }
            if c.receipt().phase == Phase::Quarantined { let _ = c.terminate_quarantined(); }
            if c.observe_exit(id).is_err() { let _ = c.fault("STOP_EXIT_ORDER"); }
            observed_stages.push("REAPED");
        } else {
            let _ = c.fault("STOP_REAP_UNCONFIRMED");
            if c.receipt().phase == Phase::Quarantined { let _ = c.terminate_quarantined(); }
            let _ = c.exit_unconfirmed();
        }
        if !reaped { self.unconfirmed.push(child); }
        let process_exit_success = status.as_ref().is_some_and(|s| s.success());
        Ok(MockReport {
            receipt: c.receipt(), process_pid: pid, exit_status: status.map(|s| s.to_string()),
            process_exit_success, reaped, kill_attempted,
            kill_attempt_count: u8::from(kill_attempted), containment_errors,
            terminal_clean_eof, mock_output_verified, observed_stages,
            elapsed_ms: started.elapsed().as_millis(), kind: MOCK_KIND,
        })
    }
}
