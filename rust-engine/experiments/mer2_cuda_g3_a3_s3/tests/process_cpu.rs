//! Source-only corrective test draft. NONE of these tests have been executed.
//! The process trace and exact first-error assertions reject unrelated startup
//! failures; parent exit and client observations are separate evidence.
#![forbid(unsafe_code)]
use mer2_g3_a3_s3_p1a::contract::{
    MockReport, MockScenario, MockSupervisor, MockVector, MockWork, S1WireTicket,
};
use mer2_g3_a3_s3_p1a::MOCK_KIND;
use mer2_g3_a3_s1::{
    sha256, Binding, Cleanup, ClientAck, ClientChunk, Delivery, Owner, Phase, Ticket, Work,
};
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn worker_binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_cpu_mock_worker"))
}
fn fixture() -> (Binding, MockWork, MockVector) {
    let binding = Binding {
        source: sha256(b"R4_source"), model: sha256(b"synthetic_model"), kv: sha256(b"synthetic_kv"),
    };
    let vector = MockVector::checked([0.25f32; 64]).unwrap();
    let ticket = Ticket { request: "request_A".to_owned(), generation: 1, nonce: 7, binding: binding.clone() };
    let s1 = S1WireTicket {
        session: 1, contract_sha256: sha256(b"S1_contract"), request: ticket.request.clone(), nonce: 7,
        attempt: 1, position: 0, input_id: 1, kv_generation: 1,
        kv_sha256: binding.kv.clone(), source_sha256: binding.source.clone(), model_sha256: binding.model.clone(),
    };
    let work = MockWork {
        r4: ticket, s1, layer: 0, rank: 0, expert: 0,
        input_sha256: vector.sha256(), weight_commitment_sha256: sha256(b"synthetic_mock_weight"),
    };
    (binding, work, vector)
}
#[derive(Clone, Copy)]
enum AckMode { Correct, WrongDigest, Cancel, Disconnect }
struct CaseReport {
    report: MockReport,
    observed_client: Option<ClientChunk>,
    expected_ticket: Ticket,
    expected_input: [u8; 256],
}
impl Deref for CaseReport {
    type Target = MockReport;
    fn deref(&self) -> &Self::Target { &self.report }
}
impl CaseReport {
    fn assert_stage(&self, name: &str) {
        assert!(self.report.observed_stages.contains(&name), "missing stage {name}: {:?}", self.report);
    }
    fn assert_first(&self, expected: &'static str) {
        assert_eq!(self.report.receipt.first_error, Some(expected), "{:?}", self.report);
    }
    fn assert_client_payload(&self) {
        let chunk = self.observed_client.as_ref().expect("required client chunk not observed");
        assert_eq!(chunk.bytes.as_slice(), self.expected_input.as_slice());
        assert_eq!(chunk.digest, sha256(&self.expected_input));
        assert_eq!(chunk.ticket, self.expected_ticket);
        assert!(chunk.final_chunk);
    }
    fn assert_client_output(&self) {
        self.assert_client_payload();
        self.assert_stage("CLIENT_OUTPUT_SENT");
    }
}
fn run_case(name: &'static str, scenario: MockScenario, mode: AckMode, limit: Duration) -> CaseReport {
    let (binding, work, input) = fixture();
    let expected_ticket = work.r4.clone();
    let expected_input = input.bytes();
    let (tx_out, rx_out) = sync_channel::<ClientChunk>(1);
    let (tx_ack, rx_ack) = channel::<ClientAck>();
    let cancel = Arc::new(AtomicBool::new(false));
    let client_cancel = cancel.clone();
    let client = thread::spawn(move || -> Option<ClientChunk> {
        let chunk = rx_out.recv_timeout(limit + Duration::from_secs(1)).ok()?;
        if matches!(mode, AckMode::Cancel) { client_cancel.store(true, Ordering::SeqCst); }
        if !matches!(mode, AckMode::Disconnect) {
            let digest = if matches!(mode, AckMode::WrongDigest) { sha256(b"wrong") }
                         else { chunk.digest.clone() };
            let _ = tx_ack.send(ClientAck { ticket: chunk.ticket.clone(), digest });
        }
        Some(chunk)
    });
    let mut supervisor = MockSupervisor::default();
    let report = supervisor.execute_once(
        worker_binary(), binding, work, input, tx_out, rx_ack,
        cancel, limit, scenario,
    ).expect("supervisor must return a report even when the transaction STOPs");
    let observed_client = client.join().expect("client observer panicked");
    // These tests deliberately cover only direct children that were successfully
    // reaped; an unconfirmed direct child requires a separate host owner/gate.
    assert_eq!(supervisor.unconfirmed_children(), 0, "STOP_UNCONFIRMED_CHILD_REQUIRES_OUTER_OWNER");
    eprintln!("A3N1_CPU_CASE name={name} scenario={scenario:?} pid={} report={report:?} client_seen={}",
        report.process_pid, observed_client.is_some());
    CaseReport { report, observed_client, expected_ticket, expected_input }
}
fn assert_success(r: &CaseReport) {
    assert!(r.report.is_success(), "not an independently successful mock transaction: {:?}", r.report);
    r.assert_client_output();
    r.assert_stage("CLEAN_EOF");
    r.assert_stage("REAPED");
    assert!(r.report.containment_errors.is_empty());
    assert_eq!(r.receipt.secondary_errors.len(), 0);
    assert_eq!(r.receipt.owner, Owner::Confirmed);
    assert_eq!(r.receipt.work, Work::Fenced);
    assert_eq!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
    assert_eq!(r.receipt.delivery, Delivery::Acknowledged);
    assert_eq!(r.receipt.phase, Phase::Exited);
    assert!(r.receipt.process_exit_observed);
    assert_eq!(r.receipt.drain_attempts, 1);
    assert_eq!(r.receipt.release_attempts, 1);
    assert_eq!(r.kind, MOCK_KIND);
}
#[test]
fn p01_normal_single_request_direct_child_reaped() {
    let r = run_case("P01", MockScenario::Normal, AckMode::Correct, Duration::from_secs(3));
    assert_success(&r);
    assert!(!r.kill_attempted);
    assert_eq!(r.kill_attempt_count, 0);
    assert_eq!(r.receipt.last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(r.receipt.leak_free, "NOT_CLAIMED");
    assert_eq!(r.receipt.automatic_retry, "FORBIDDEN");
}
#[test]
fn p02_wrong_worker_pid_rejected() {
    let r = run_case("P02", MockScenario::WrongPid, AckMode::Correct, Duration::from_secs(2));
    r.assert_first("STOP_HELLO_IDENTITY");
    assert_eq!(r.receipt.owner, Owner::NotAttempted);
    assert_eq!(r.receipt.work, Work::None);
    assert_eq!(r.receipt.delivery, Delivery::None);
    assert!(!r.report.is_success());
    assert!(r.reaped && r.receipt.process_exit_observed);
    assert!(r.observed_client.is_none());
}
#[test]
fn p03_post_submit_exit_is_ambiguous_and_never_replayed() {
    let r = run_case("P03", MockScenario::PostSubmitExit, AckMode::Correct, Duration::from_secs(2));
    r.assert_stage("ENQUEUED");
    r.assert_first("STOP_WIRE_EOF");
    assert_eq!(r.receipt.work, Work::Submitted);
    assert_ne!(r.receipt.work, Work::Fenced);
    assert!(!r.process_exit_success);
    assert!(r.reaped && r.receipt.process_exit_observed);
    assert!(!r.report.is_success());
    assert_eq!(r.receipt.automatic_retry, "FORBIDDEN");
}
#[test]
fn p04_bad_output_descriptor_detected_before_client_ack() {
    let r = run_case("P04", MockScenario::MutatedOutput, AckMode::Correct, Duration::from_secs(2));
    r.assert_stage("FENCED");
    r.assert_first("STOP_TENSOR_DESCRIPTOR_BINDING");
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
    assert_eq!(r.receipt.drain_attempts, 0);
    assert_eq!(r.receipt.release_attempts, 0);
    assert!(r.observed_client.is_none());
}
#[test]
fn p05_wrong_client_ack_is_ambiguous() {
    let r = run_case("P05", MockScenario::Normal, AckMode::WrongDigest, Duration::from_secs(3));
    r.assert_client_output();
    r.assert_first("STOP_DELIVERY_ACK");
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
    assert_eq!(r.receipt.drain_attempts, 0);
    assert_eq!(r.receipt.release_attempts, 0);
}
#[test]
fn p06_cancel_at_ack_boundary_stops_cleanup_success() {
    let r = run_case("P06", MockScenario::Normal, AckMode::Cancel, Duration::from_secs(3));
    // Cancellation can win the post-send check even if the client saw bytes.
    r.assert_client_payload();
    r.assert_first("STOP_CANCELLED");
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
    assert_eq!(r.receipt.drain_attempts, 0);
    assert_eq!(r.receipt.release_attempts, 0);
    assert!(!r.report.is_success());
}
#[test]
fn p07_client_ack_disconnect_is_terminal_ambiguous() {
    let r = run_case("P07", MockScenario::Normal, AckMode::Disconnect, Duration::from_secs(3));
    r.assert_client_output();
    r.assert_first("STOP_CLIENT_DISCONNECT");
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
    assert_eq!(r.receipt.drain_attempts, 0);
}
#[test]
fn p08_drain_failure_keeps_first_error_and_no_release() {
    let r = run_case("P08", MockScenario::DrainFailure, AckMode::Correct, Duration::from_secs(3));
    r.assert_stage("ACKNOWLEDGED");
    r.assert_first("STOP_DRAIN_FAILED");
    assert_eq!(r.receipt.drain_attempts, 1);
    assert_eq!(r.receipt.release_attempts, 0);
    assert_eq!(r.receipt.cleanup, Cleanup::Unconfirmed);
    assert!(!r.report.is_success());
}
#[test]
fn p09_release_failure_is_not_last_cuda_drop_proof() {
    let r = run_case("P09", MockScenario::ReleaseFailure, AckMode::Correct, Duration::from_secs(3));
    r.assert_stage("DRAINED");
    r.assert_first("STOP_DESTRUCTOR_FAILED");
    assert_eq!(r.receipt.drain_attempts, 1);
    assert_eq!(r.receipt.release_attempts, 1);
    assert_eq!(r.receipt.cleanup, Cleanup::Unconfirmed);
    assert_eq!(r.receipt.last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(r.receipt.leak_free, "NOT_CLAIMED");
}
#[test]
fn p10_unreturned_retain_stays_unknown() {
    let r = run_case("P10", MockScenario::UnreturnedRetain, AckMode::Correct, Duration::from_secs(2));
    r.assert_first("STOP_UNRETURNED_RETAIN");
    assert!(r.reaped && r.receipt.unknown_ownership);
    assert_eq!(r.receipt.owner, Owner::Uncertain);
    assert_eq!(r.receipt.work, Work::None);
    assert_eq!(r.receipt.drain_attempts, 0);
    assert_eq!(r.receipt.release_attempts, 0);
}
#[test]
fn p11_before_retain_does_not_claim_owner() {
    let r = run_case("P11", MockScenario::BeforeRetain, AckMode::Correct, Duration::from_secs(2));
    r.assert_first("STOP_BEFORE_RETAIN");
    assert_eq!(r.receipt.owner, Owner::NotAttempted);
    assert!(!r.receipt.unknown_ownership);
    assert_eq!(r.receipt.work, Work::None);
    assert_eq!(r.receipt.drain_attempts, 0);
    assert_eq!(r.receipt.release_attempts, 0);
}
#[test]
fn p12_hung_worker_must_be_contained_once() {
    // Stage evidence must prove the child reached ENQUEUED; a startup timeout
    // or unrelated failure is not a successful negative test for this scenario.
    let r = run_case("P12", MockScenario::HangAfterSubmit, AckMode::Correct, Duration::from_secs(3));
    r.assert_stage("ENQUEUED");
    r.assert_first("STOP_DEADLINE");
    assert!(r.reaped, "direct child must be reaped or STOP_PENDING_REVIEW");
    assert!(r.kill_attempted);
    assert_eq!(r.kill_attempt_count, 1);
    assert_eq!(r.receipt.work, Work::Submitted);
    assert_eq!(r.receipt.automatic_retry, "FORBIDDEN");
}
#[test]
fn p13_invalid_work_refused_without_spawning() {
    let (binding, mut work, input) = fixture();
    work.s1.source_sha256 = sha256(b"foreign");
    let (tx_out, _rx_out) = sync_channel::<ClientChunk>(1);
    let (_tx_ack, rx_ack) = channel::<ClientAck>();
    let mut supervisor = MockSupervisor::default();
    let result = supervisor.execute_once(
        worker_binary(), binding, work, input, tx_out, rx_ack,
        Arc::new(AtomicBool::new(false)), Duration::from_secs(3), MockScenario::Normal,
    );
    assert!(matches!(result, Err("STOP_WORK_BINDING")));
    assert_eq!(supervisor.unconfirmed_children(), 0);
}
#[test]
fn p14_second_use_of_same_supervisor_is_refused_even_after_success() {
    let (binding, work, input) = fixture();
    let bytes = input.bytes();
    let (tx_out, rx_out) = sync_channel::<ClientChunk>(1);
    let (tx_ack, rx_ack) = channel::<ClientAck>();
    let client = thread::spawn(move || -> Option<ClientChunk> {
        let chunk = rx_out.recv_timeout(Duration::from_secs(4)).ok()?;
        let _ = tx_ack.send(ClientAck { ticket: chunk.ticket.clone(), digest: chunk.digest.clone() });
        Some(chunk)
    });
    let mut supervisor = MockSupervisor::default();
    let result = supervisor.execute_once(
        worker_binary(), binding.clone(), work.clone(), input.clone(), tx_out, rx_ack,
        Arc::new(AtomicBool::new(false)), Duration::from_secs(3), MockScenario::Normal,
    ).unwrap();
    let delivered = client.join().unwrap().expect("client output must be observed");
    assert!(result.is_success(), "first transaction must be genuinely successful: {result:?}");
    assert_eq!(delivered.bytes.as_slice(), bytes.as_slice());
    assert_eq!(delivered.digest, sha256(&bytes));
    assert_eq!(delivered.ticket, work.r4);
    assert!(delivered.final_chunk);
    let (tx_new, _rx_new) = sync_channel::<ClientChunk>(1);
    let (_ack_sender, ack_new) = channel::<ClientAck>();
    assert!(matches!(supervisor.execute_once(
        worker_binary(), binding, work, input, tx_new, ack_new,
        Arc::new(AtomicBool::new(false)), Duration::from_secs(3), MockScenario::Normal,
    ), Err("STOP_REPLAY_FORBIDDEN")));
    eprintln!("A3N1_CPU_CASE name=P14 pid={} first_transaction_success=true replay_refused=true", result.process_pid);
}
#[test]
fn p15_trailing_bytes_after_bye_are_terminal() {
    for (case, scenario) in [
        ("P15A", MockScenario::TrailingByte),
        ("P15B", MockScenario::DuplicateBye),
        ("P15C", MockScenario::TrailingFrame),
        ("P15D", MockScenario::DelayedTrailingByte),
    ] {
        let r = run_case(case, scenario, AckMode::Correct, Duration::from_secs(3));
        r.assert_stage("BYE");
        r.assert_first("STOP_WIRE_TRAILING_AFTER_BYE");
        assert!(!r.terminal_clean_eof);
        assert!(r.reaped && r.receipt.process_exit_observed);
        assert!(!r.report.is_success());
        // It is possible for scoped RELEASED to have been observed before the
        // bad terminal byte; that must never turn an invalid session into PASS.
    }
    let clean = run_case("P15_NORMAL", MockScenario::Normal, AckMode::Correct, Duration::from_secs(3));
    assert_success(&clean);
}
#[test]
fn p16_self_consistent_wrong_output_is_rejected() {
    let r = run_case("P16", MockScenario::SelfConsistentWrongOutput, AckMode::Correct, Duration::from_secs(3));
    r.assert_stage("FENCED");
    r.assert_first("STOP_MOCK_IDENTITY_MISMATCH");
    assert!(r.observed_client.is_none());
    assert_eq!(r.receipt.drain_attempts, 0);
    assert_eq!(r.receipt.release_attempts, 0);
    assert!(!r.mock_output_verified);
}
#[test]
fn p17_cancel_before_spawn_creates_no_child() {
    let (binding, work, input) = fixture();
    let (tx_out, _rx_out) = sync_channel::<ClientChunk>(1);
    let (_tx_ack, rx_ack) = channel::<ClientAck>();
    let cancel = Arc::new(AtomicBool::new(true));
    let mut supervisor = MockSupervisor::default();
    // A deliberately nonexistent executable exposes wrong preflight ordering.
    let result = supervisor.execute_once(
        Path::new("/deliberately/nonexistent/s3-p1a-cancelled-worker"),
        binding, work, input, tx_out, rx_ack, cancel,
        Duration::from_secs(3), MockScenario::Normal,
    );
    assert!(matches!(result, Err("STOP_CANCELLED")));
    assert_eq!(supervisor.unconfirmed_children(), 0);
}
#[test]
fn p21_missing_final_receipt_is_terminal() {
    let r = run_case("P21A", MockScenario::MissingBye, AckMode::Correct, Duration::from_secs(3));
    r.assert_stage("RELEASED");
    r.assert_first("STOP_WIRE_EOF");
    assert!(!r.terminal_clean_eof);
    assert!(!r.report.is_success());
    let r = run_case("P21B", MockScenario::AbnormalAfterBye, AckMode::Correct, Duration::from_secs(3));
    r.assert_stage("BYE");
    r.assert_first("STOP_CHILD_ABNORMAL_EXIT");
    assert!(r.terminal_clean_eof);
    assert!(!r.process_exit_success);
    assert!(r.reaped);
    assert!(!r.report.is_success());
}
