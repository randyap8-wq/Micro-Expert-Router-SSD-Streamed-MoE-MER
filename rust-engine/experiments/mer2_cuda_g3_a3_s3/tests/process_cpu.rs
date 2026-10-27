//! Draft only. Must be reviewed and separately authorized before executing.
#![forbid(unsafe_code)]
use mer2_g3_a3_s3_p1a::contract::{MockScenario, MockSupervisor, MockVector, MockWork, S1WireTicket};
use mer2_g3_a3_s3_p1a::MOCK_KIND;
use mer2_g3_a3_s1::{sha256, Binding, Cleanup, ClientAck, Delivery, Phase, Ticket, Work};
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
fn run_case(
    scenario: MockScenario, mode: AckMode, limit: Duration,
) -> mer2_g3_a3_s3_p1a::contract::MockReport {
    let (binding, work, input) = fixture();
    let (tx_out, rx_out) = sync_channel(1);
    let (tx_ack, rx_ack) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let client_cancel = cancel.clone();
    let client = thread::spawn(move || {
        if let Ok(chunk) = rx_out.recv_timeout(limit + Duration::from_secs(1)) {
            if matches!(mode, AckMode::Cancel) { client_cancel.store(true, Ordering::SeqCst); }
            if !matches!(mode, AckMode::Disconnect) {
                let digest = if matches!(mode, AckMode::WrongDigest) { sha256(b"wrong") } else { chunk.digest };
                let _ = tx_ack.send(ClientAck { ticket: chunk.ticket, digest });
            }
        }
    });
    let report = MockSupervisor::default().execute_once(
        worker_binary(), binding, work, input, tx_out, rx_ack,
        cancel, limit, scenario,
    ).unwrap();
    client.join().unwrap();
    report
}
#[test]
fn p01_normal_single_request_direct_child_reaped() {
    let r = run_case(MockScenario::Normal, AckMode::Correct, Duration::from_secs(3));
    assert!(r.reaped);
    assert!(!r.kill_attempted);
    assert!(r.receipt.first_error.is_none(), "{:?}", r.receipt.first_error);
    assert_eq!(r.receipt.phase, Phase::Exited);
    assert_eq!(r.receipt.delivery, Delivery::Acknowledged);
    assert_eq!(r.receipt.work, Work::Fenced);
    assert_eq!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
    assert_eq!(r.kind, MOCK_KIND);
    assert_eq!(r.receipt.last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(r.receipt.leak_free, "NOT_CLAIMED");
    assert_eq!(r.receipt.automatic_retry, "FORBIDDEN");
}
#[test]
fn p02_wrong_worker_pid_rejected() {
    let r = run_case(MockScenario::WrongPid, AckMode::Correct, Duration::from_secs(2));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_ne!(r.receipt.delivery, Delivery::Acknowledged);
}
#[test]
fn p03_post_submit_exit_is_ambiguous_and_never_replayed() {
    let r = run_case(MockScenario::PostSubmitExit, AckMode::Correct, Duration::from_secs(2));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_ne!(r.receipt.work, Work::Fenced);
    assert_eq!(r.receipt.automatic_retry, "FORBIDDEN");
}
#[test]
fn p04_bad_output_descriptor_detected_before_client_ack() {
    let r = run_case(MockScenario::MutatedOutput, AckMode::Correct, Duration::from_secs(2));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_ne!(r.receipt.delivery, Delivery::Acknowledged);
}
#[test]
fn p05_wrong_client_ack_is_ambiguous() {
    let r = run_case(MockScenario::Normal, AckMode::WrongDigest, Duration::from_secs(3));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
}
#[test]
fn p06_cancel_at_ack_boundary_stops_cleanup_success() {
    let r = run_case(MockScenario::Normal, AckMode::Cancel, Duration::from_secs(3));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_ne!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
}
#[test]
fn p07_client_ack_disconnect_is_terminal_ambiguous() {
    let r = run_case(MockScenario::Normal, AckMode::Disconnect, Duration::from_secs(3));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
}
#[test]
fn p08_drain_failure_keeps_first_error_and_no_release() {
    let r = run_case(MockScenario::DrainFailure, AckMode::Correct, Duration::from_secs(3));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_ne!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
}
#[test]
fn p09_release_failure_is_not_last_cuda_drop_proof() {
    let r = run_case(MockScenario::ReleaseFailure, AckMode::Correct, Duration::from_secs(3));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_eq!(r.receipt.last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(r.receipt.leak_free, "NOT_CLAIMED");
}
#[test]
fn p10_unreturned_retain_stays_unknown() {
    let r = run_case(MockScenario::UnreturnedRetain, AckMode::Correct, Duration::from_secs(2));
    assert!(r.reaped);
    assert!(r.receipt.unknown_ownership);
    assert!(r.receipt.first_error.is_some());
}
#[test]
fn p11_before_retain_does_not_claim_owner() {
    let r = run_case(MockScenario::BeforeRetain, AckMode::Correct, Duration::from_secs(2));
    assert!(r.reaped);
    assert!(r.receipt.first_error.is_some());
    assert_ne!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
}
#[test]
fn p12_hung_worker_must_be_contained_once() {
    let r = run_case(MockScenario::HangAfterSubmit, AckMode::Correct, Duration::from_millis(200));
    assert!(r.reaped, "direct child must be reaped or the source is STOP_PENDING_REVIEW");
    assert!(r.kill_attempted);
    assert!(r.receipt.first_error.is_some());
}
#[test]
fn p13_invalid_work_refused_without_spawning() {
    let (binding, mut work, input) = fixture();
    work.s1.source_sha256 = sha256(b"foreign");
    let (tx_out, _rx_out) = sync_channel(1);
    let (_tx_ack, rx_ack) = channel();
    let result = MockSupervisor::default().execute_once(
        worker_binary(), binding, work, input, tx_out, rx_ack,
        Arc::new(AtomicBool::new(false)), Duration::from_secs(3), MockScenario::Normal,
    );
    assert!(matches!(result, Err("STOP_WORK_BINDING")));
}
#[test]
fn p14_second_use_of_same_supervisor_is_refused_even_after_success() {
    let (binding, work, input) = fixture();
    let (tx_out, rx_out) = sync_channel(1);
    let (tx_ack, rx_ack) = channel();
    let client = thread::spawn(move || {
        if let Ok(chunk) = rx_out.recv_timeout(Duration::from_secs(4)) {
            let _ = tx_ack.send(ClientAck { ticket: chunk.ticket, digest: chunk.digest });
        }
    });
    let mut supervisor = MockSupervisor::default();
    let result = supervisor.execute_once(
        worker_binary(), binding.clone(), work.clone(), input.clone(), tx_out, rx_ack,
        Arc::new(AtomicBool::new(false)), Duration::from_secs(3), MockScenario::Normal,
    ).unwrap();
    client.join().unwrap();
    assert!(result.reaped);
    let (tx_new, _rx_new) = sync_channel(1);
    let (_ack_sender, ack_new) = channel();
    assert!(matches!(supervisor.execute_once(
        worker_binary(), binding, work, input, tx_new, ack_new,
        Arc::new(AtomicBool::new(false)), Duration::from_secs(3), MockScenario::Normal,
    ), Err("STOP_REPLAY_FORBIDDEN")));
}
