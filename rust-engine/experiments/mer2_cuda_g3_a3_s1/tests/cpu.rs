use mer2_g3_a3_s1::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;
fn binding() -> Binding {
    Binding {
        source: "a".repeat(64),
        model: "b".repeat(64),
        kv: "c".repeat(64),
    }
}
fn id(g: u64) -> Identity {
    Identity {
        pid: 1234,
        generation: g,
    }
}
fn ticket(g: u64) -> Ticket {
    Ticket {
        request: "request-1".into(),
        generation: g,
        nonce: 10,
        binding: binding(),
    }
}
fn initial() -> Contract {
    Contract::new(id(1), binding(), Ledger::default()).unwrap()
}
fn ready_with(g: u64, l: Ledger) -> Contract {
    let mut c = Contract::new(id(g), binding(), l).unwrap();
    c.constructor_entered().unwrap();
    c.typed_owner_returned().unwrap();
    c.deferred_check().unwrap();
    c.ready().unwrap();
    c
}
fn ready() -> Contract {
    ready_with(1, Ledger::default())
}
fn flight() -> (Contract, Ticket) {
    let mut c = ready();
    let t = ticket(1);
    c.admit(&t).unwrap();
    (c, t)
}
fn fenced() -> (Contract, Ticket) {
    let (mut c, t) = flight();
    c.submission_entered(&t).unwrap();
    c.submission_confirmed(&t).unwrap();
    c.fence(&t).unwrap();
    (c, t)
}
fn delivered() -> (Contract, Ticket) {
    let (mut c, t) = fenced();
    c.prepare(&t, FAKE_DIGEST).unwrap();
    c.begin_delivery(&t, FAKE_DIGEST).unwrap();
    c.acknowledge(&t, FAKE_DIGEST).unwrap();
    (c, t)
}
fn exited() -> Contract {
    let (mut c, _) = delivered();
    c.drain(true).unwrap();
    c.release(true).unwrap();
    c.observe_exit(id(1)).unwrap();
    c
}
macro_rules! reject {
    ($c:ident,$code:literal,$op:expr) => {{
        let before = $c.receipt();
        assert_eq!($op, Err($code));
        let after = $c.receipt();
        assert_eq!(after.first_error, before.first_error.or(Some($code)));
        assert_eq!(
            after.phase,
            if matches!(before.phase, Phase::Exited | Phase::ExitUnconfirmed) {
                before.phase
            } else {
                Phase::Quarantined
            }
        );
        assert_eq!(after.leak_free, "NOT_CLAIMED");
    }};
}
#[test]
fn c01_clean_retirement_and_separate_observations() {
    let mut c = exited();
    assert_eq!(c.receipt().post_exit_baseline, Observation::Unknown);
    c.synthetic_baseline(&[(0, 0); 8], 0).unwrap();
    assert_eq!(
        c.receipt().post_exit_baseline,
        Observation::SyntheticBaselineObserved
    );
}
#[test]
fn c02_ready_without_owner_rejected() {
    let mut c = initial();
    reject!(c, "STOP_READY_WITH_UNCERTAIN_OWNER", c.ready());
}
#[test]
fn c03_unreturned_retain_quarantined() {
    let mut c = initial();
    c.constructor_entered().unwrap();
    assert!(c.receipt().unknown_ownership);
    reject!(c, "STOP_CONSTRUCTOR", c.fault("STOP_CONSTRUCTOR"));
    assert_eq!(c.receipt().owner, Owner::Uncertain);
}
#[test]
fn c04_returned_unchecked_failure() {
    let mut c = initial();
    c.constructor_entered().unwrap();
    c.typed_owner_returned().unwrap();
    reject!(c, "STOP_DEFERRED", c.fault("STOP_DEFERRED"));
    assert_eq!(c.receipt().owner, Owner::Uncertain);
}
#[test]
fn c05_duplicate_constructor() {
    let mut c = initial();
    c.constructor_entered().unwrap();
    reject!(c, "STOP_CONSTRUCTOR_ORDER", c.constructor_entered());
}
#[test]
fn c06_check_without_owner() {
    let mut c = initial();
    c.constructor_entered().unwrap();
    reject!(c, "STOP_CONTEXT_CHECK_ORDER", c.deferred_check());
}
#[test]
fn c07_owner_return_after_ready() {
    let mut c = ready();
    reject!(c, "STOP_CONTEXT_OWNER_ORDER", c.typed_owner_returned());
}
#[test]
fn c08_stale_generation() {
    let mut c = ready_with(2, Ledger::default());
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&ticket(1)));
}
#[test]
fn c09_foreign_source() {
    let mut c = ready();
    let mut t = ticket(1);
    t.binding.source = "e".repeat(64);
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&t));
}
#[test]
fn c10_foreign_model() {
    let mut c = ready();
    let mut t = ticket(1);
    t.binding.model = "e".repeat(64);
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&t));
}
#[test]
fn c11_bad_kv() {
    let mut c = ready();
    let mut t = ticket(1);
    t.binding.kv = "notsha".into();
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&t));
}
#[test]
fn c12_invalid_nonce() {
    let mut c = ready();
    let mut t = ticket(1);
    t.nonce = 0;
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&t));
}
#[test]
fn c13_concurrent_admission() {
    let (mut c, _) = flight();
    let mut t = ticket(1);
    t.request = "second".into();
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&t));
}
#[test]
fn c14_duplicate_across_generations() {
    let l = Ledger::default();
    let mut a = ready_with(1, l.clone());
    a.admit(&ticket(1)).unwrap();
    let mut c = ready_with(2, l);
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&ticket(2)));
}
#[test]
fn c15_invalid_worker_identity() {
    assert!(Contract::new(id(0), binding(), Ledger::default()).is_err());
    assert!(Contract::new(
        Identity {
            pid: 0,
            generation: 1
        },
        binding(),
        Ledger::default()
    )
    .is_err());
    let mut b = binding();
    b.source.pop();
    assert!(Contract::new(id(1), b, Ledger::default()).is_err());
}
#[test]
fn c16_submit_without_request() {
    let mut c = ready();
    reject!(c, "STOP_SUBMISSION_ORDER", c.submission_entered(&ticket(1)));
}
#[test]
fn c17_unfenced_output() {
    let (mut c, t) = flight();
    c.submission_entered(&t).unwrap();
    c.submission_confirmed(&t).unwrap();
    reject!(c, "STOP_OUTPUT_UNCOMMITTED", c.prepare(&t, FAKE_DIGEST));
}
#[test]
fn c18_post_enqueue_uncertainty() {
    let (mut c, t) = flight();
    c.submission_entered(&t).unwrap();
    reject!(
        c,
        "STOP_ENQUEUE_AMBIGUOUS",
        c.fault("STOP_ENQUEUE_AMBIGUOUS")
    );
    assert_eq!(c.receipt().work, Work::SubmissionUncertain);
    assert_eq!(c.receipt().delivery, Delivery::Ambiguous);
}
#[test]
fn c19_foreign_fence_nonce() {
    let (mut c, mut t) = flight();
    c.submission_entered(&t).unwrap();
    c.submission_confirmed(&t).unwrap();
    t.nonce += 1;
    reject!(c, "STOP_FENCE_CONFIRMATION", c.fence(&t));
}
#[test]
fn c20_duplicate_fence() {
    let (mut c, t) = fenced();
    reject!(c, "STOP_FENCE_CONFIRMATION", c.fence(&t));
}
#[test]
fn c21_corrupt_digest() {
    let (mut c, t) = fenced();
    reject!(c, "STOP_OUTPUT_UNCOMMITTED", c.prepare(&t, "wrong"));
}
#[test]
fn c22_mutated_prepared_digest() {
    let (mut c, t) = fenced();
    c.prepare(&t, FAKE_DIGEST).unwrap();
    reject!(
        c,
        "STOP_DELIVERY_COMMITMENT",
        c.begin_delivery(&t, &"e".repeat(64))
    );
}
#[test]
fn c23_ack_before_delivery() {
    let (mut c, t) = fenced();
    c.prepare(&t, FAKE_DIGEST).unwrap();
    reject!(c, "STOP_DELIVERY_ACK", c.acknowledge(&t, FAKE_DIGEST));
}
#[test]
fn c24_sent_unacknowledged_ambiguous() {
    let (mut c, t) = fenced();
    c.prepare(&t, FAKE_DIGEST).unwrap();
    c.begin_delivery(&t, FAKE_DIGEST).unwrap();
    reject!(
        c,
        "STOP_CLIENT_DISCONNECT",
        c.fault("STOP_CLIENT_DISCONNECT")
    );
    assert_eq!(c.receipt().delivery, Delivery::Ambiguous);
}
#[test]
fn c25_ack_only_draining() {
    let (c, _) = delivered();
    assert_eq!(c.receipt().phase, Phase::Draining);
    assert!(!c.receipt().process_exit_observed);
    assert_eq!(c.receipt().cleanup, Cleanup::NotAttempted);
}
#[test]
fn c26_first_error_precedence() {
    let (mut c, _) = flight();
    reject!(c, "STOP_FIRST", c.fault("STOP_FIRST"));
    reject!(c, "STOP_SECOND", c.fault("STOP_SECOND"));
    assert_eq!(c.receipt().secondary_errors, vec!["STOP_SECOND"]);
}
#[test]
fn c27_no_drain_without_ack() {
    let (mut c, _) = fenced();
    reject!(c, "STOP_DRAIN_UNCONFIRMED", c.drain(true));
    assert_eq!(c.receipt().drain_attempts, 0);
}
#[test]
fn c28_release_before_drain() {
    let (mut c, _) = delivered();
    reject!(c, "STOP_RELEASE_UNCONFIRMED", c.release(true));
}
#[test]
fn c29_exactly_one_drain_release() {
    let (mut c, _) = delivered();
    c.drain(true).unwrap();
    c.release(true).unwrap();
    reject!(c, "STOP_RELEASE_UNCONFIRMED", c.release(true));
    assert_eq!(
        (c.receipt().drain_attempts, c.receipt().release_attempts),
        (1, 1)
    );
}
#[test]
fn c30_no_second_drain() {
    let (mut c, _) = delivered();
    c.drain(true).unwrap();
    reject!(c, "STOP_DRAIN_UNCONFIRMED", c.drain(true));
    assert_eq!(c.receipt().drain_attempts, 1);
}
#[test]
fn c31_quarantine_never_readmitted() {
    let (mut c, _) = flight();
    let _ = c.fault("STOP_UNKNOWN");
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&ticket(1)));
    reject!(c, "STOP_READY_WITH_UNCERTAIN_OWNER", c.ready());
}
#[test]
fn c32_quarantine_exit_keeps_error() {
    let mut c = ready();
    let _ = c.fault("STOP_RESOURCE");
    c.terminate_quarantined().unwrap();
    c.observe_exit(id(1)).unwrap();
    assert_eq!(c.receipt().first_error, Some("STOP_RESOURCE"));
    assert_eq!(c.receipt().cleanup, Cleanup::Unconfirmed);
}
#[test]
fn c33_direct_ready_termination_rejected() {
    let mut c = ready();
    reject!(c, "STOP_TERMINATION_ORDER", c.terminate_quarantined());
}
#[test]
fn c34_pid_reuse_generation_fenced() {
    let mut c = ready_with(3, Ledger::default());
    let _ = c.fault("STOP_CANCEL");
    c.terminate_quarantined().unwrap();
    reject!(c, "STOP_EXIT_IDENTITY", c.observe_exit(id(2)));
}
#[test]
fn c35_wrong_exit_pid() {
    let (mut c, _) = delivered();
    c.drain(true).unwrap();
    c.release(true).unwrap();
    reject!(
        c,
        "STOP_EXIT_IDENTITY",
        c.observe_exit(Identity {
            pid: 7777,
            generation: 1
        })
    );
}
#[test]
fn c36_exit_unconfirmed_is_unknown() {
    let (mut c, _) = delivered();
    c.drain(true).unwrap();
    c.release(true).unwrap();
    c.exit_unconfirmed().unwrap();
    assert!(!c.receipt().process_exit_observed);
    assert_eq!(c.receipt().phase, Phase::ExitUnconfirmed);
    assert_eq!(c.receipt().post_exit_baseline, Observation::Unknown);
}
#[test]
fn c37_baseline_before_exit() {
    let mut c = ready();
    reject!(
        c,
        "STOP_POST_EXIT_OBSERVATION",
        c.synthetic_baseline(&[(0, 0); 8], 0)
    );
}
#[test]
fn c38_foreign_process_baseline() {
    let mut c = exited();
    reject!(
        c,
        "STOP_POST_EXIT_OBSERVATION",
        c.synthetic_baseline(&[(0, 1); 8], 0)
    );
}
#[test]
fn c39_memory_outside_tolerance() {
    let mut c = exited();
    reject!(
        c,
        "STOP_POST_EXIT_OBSERVATION",
        c.synthetic_baseline(&[(4, 0); 8], 0)
    );
}
#[test]
fn c40_baseline_tolerance_without_drop_claim() {
    let mut c = exited();
    c.synthetic_baseline(&[(2, 0); 8], 0).unwrap();
    assert_eq!(c.receipt().last_cuda_context_drop, "UNOBSERVABLE");
}
#[test]
fn c40a_requires_eight_samples() {
    let mut c = exited();
    reject!(
        c,
        "STOP_POST_EXIT_OBSERVATION",
        c.synthetic_baseline(&[(0, 0); 7], 0)
    );
}
#[test]
fn c40b_early_settling_allowed() {
    let mut c = exited();
    let mut s = [(20, 0); 8];
    s[4..].fill((0, 0));
    c.synthetic_baseline(&s, 0).unwrap();
}
#[test]
fn c40c_no_second_baseline() {
    let mut c = exited();
    c.synthetic_baseline(&[(0, 0); 8], 0).unwrap();
    reject!(
        c,
        "STOP_POST_EXIT_OBSERVATION",
        c.synthetic_baseline(&[(0, 0); 8], 0)
    );
}
#[test]
fn c41_invalid_fault_label() {
    let mut c = ready();
    reject!(c, "STOP_INVALID_FAULT_LABEL", c.fault("good news"));
}
#[test]
fn c42_zero_pid_rejected() {
    assert!(Contract::new(
        Identity {
            pid: 0,
            generation: 1
        },
        binding(),
        Ledger::default()
    )
    .is_err());
}
#[test]
fn c43_crash_no_silent_replay() {
    let l = Ledger::default();
    let mut a = ready_with(1, l.clone());
    let t = ticket(1);
    a.admit(&t).unwrap();
    a.submission_entered(&t).unwrap();
    a.submission_confirmed(&t).unwrap();
    a.fence(&t).unwrap();
    a.prepare(&t, FAKE_DIGEST).unwrap();
    let _ = a.fault("STOP_CRASH");
    let mut c = ready_with(2, l);
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&ticket(2)));
}
#[test]
fn c44_never_global_leak_claim() {
    let c = ready();
    assert_eq!(c.receipt().leak_free, "NOT_CLAIMED");
    assert_eq!(c.receipt().last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(c.receipt().automatic_retry, "FORBIDDEN");
}
#[test]
fn e01_before_retain_distinct() {
    let mut c = initial();
    let _ = c.fault("STOP_BEFORE_RETAIN");
    assert_eq!(c.receipt().owner, Owner::NotAttempted);
    assert!(!c.receipt().unknown_ownership);
    assert_eq!(c.receipt().cleanup, Cleanup::Unconfirmed);
}
#[test]
fn e02_known_good_owner_only_after_check() {
    let mut c = initial();
    c.constructor_entered().unwrap();
    c.typed_owner_returned().unwrap();
    assert!(c.receipt().unknown_ownership);
    c.deferred_check().unwrap();
    c.ready().unwrap();
    assert_eq!(c.receipt().owner, Owner::Confirmed);
}
#[test]
fn e03_uncertain_return_cannot_ready() {
    let mut c = initial();
    c.constructor_entered().unwrap();
    c.typed_owner_returned().unwrap();
    reject!(c, "STOP_READY_WITH_UNCERTAIN_OWNER", c.ready());
}
#[test]
fn e04_before_enqueue_work_none() {
    let (mut c, _) = flight();
    let _ = c.fault("STOP_BEFORE_ENQUEUE");
    assert_eq!(c.receipt().work, Work::None);
}
#[test]
fn e05_failed_drain_never_retried() {
    let (mut c, _) = delivered();
    reject!(c, "STOP_DRAIN_FAILED", c.drain(false));
    reject!(c, "STOP_DRAIN_UNCONFIRMED", c.drain(true));
    reject!(c, "STOP_RELEASE_UNCONFIRMED", c.release(true));
    assert_eq!(
        (c.receipt().drain_attempts, c.receipt().release_attempts),
        (1, 0)
    );
}
#[test]
fn e06_destructor_failure_never_retried() {
    let (mut c, _) = delivered();
    c.drain(true).unwrap();
    reject!(c, "STOP_DESTRUCTOR_FAILED", c.release(false));
    reject!(c, "STOP_RELEASE_UNCONFIRMED", c.release(true));
    assert_eq!(c.receipt().release_attempts, 1);
}
#[test]
fn e07_valid_foreign_kv_rejected() {
    let mut c = ready();
    let mut t = ticket(1);
    t.binding.kv = "e".repeat(64);
    reject!(c, "STOP_ADMISSION_REPLAY_OR_IDENTITY", c.admit(&t));
}
#[test]
fn e08_sha256_known_answers() {
    assert_eq!(sha256(b"abc"), FAKE_DIGEST);
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
#[test]
fn e09_frame_strict_binding_and_bound() {
    assert_eq!(
        parse_frame(&frame(id(1), &binding(), "HELLO"), id(1), &binding()),
        Ok("HELLO".into())
    );
    assert!(parse_frame("", id(1), &binding()).is_err());
    assert!(parse_frame(&frame(id(1), &binding(), "HELLO"), id(2), &binding()).is_err());
    assert!(parse_frame(&frame(id(1), &binding(), "HELLO\nBYE"), id(1), &binding()).is_err());
}

#[derive(Clone, Copy)]
enum Client {
    Ack,
    NoAck,
    PartialCancel,
    WrongAck,
}
fn run(
    s: &mut Supervisor,
    scenario: Scenario,
    t: Ticket,
    client: Client,
    cancel: Arc<AtomicBool>,
) -> ProcessReport {
    let (out_tx, out_rx) = mpsc::sync_channel::<ClientChunk>(1);
    let (ack_tx, ack_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let client_cancel = cancel.clone();
    let client_thread = thread::spawn(move || {
        let mut bytes = Vec::new();
        while let Ok(chunk) = out_rx.recv_timeout(Duration::from_secs(6)) {
            bytes.extend_from_slice(&chunk.bytes);
            if matches!(client, Client::PartialCancel) {
                assert_eq!(bytes, b"a");
                client_cancel.store(true, Ordering::SeqCst);
                let _ = done_rx.recv_timeout(Duration::from_secs(6));
                return;
            }
            if chunk.final_chunk {
                assert_eq!(bytes, FAKE_OUTPUT);
                assert_eq!(sha256(&bytes), chunk.digest);
                if matches!(client, Client::NoAck) {
                    let _ = done_rx.recv_timeout(Duration::from_secs(6));
                    return;
                }
                let hash = if matches!(client, Client::WrongAck) {
                    "e".repeat(64)
                } else {
                    chunk.digest
                };
                let _ = ack_tx.send(ClientAck {
                    ticket: chunk.ticket,
                    digest: hash,
                });
                return;
            }
        }
    });
    let r = s
        .execute(
            Path::new(env!("CARGO_BIN_EXE_fake_worker")),
            binding(),
            scenario,
            t,
            out_tx,
            ack_rx,
            cancel,
        )
        .expect("OS fixture environment");
    let _ = done_tx.send(());
    client_thread.join().unwrap();
    eprintln!("OS_EVIDENCE scenario={} report={r:?}", scenario.label());
    assert_ne!(r.receipt.identity.pid, std::process::id());
    assert!(r.reaped && r.receipt.process_exit_observed);
    assert_eq!(r.receipt.phase, Phase::Exited);
    assert!(r.exit_status.is_some());
    assert!(r.elapsed_ms < 5000, "frozen 5s end-to-end ceiling");
    assert_eq!(r.receipt.post_exit_baseline, Observation::Unknown);
    assert_eq!(r.receipt.last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(r.receipt.leak_free, "NOT_CLAIMED");
    r
}
fn once(scenario: Scenario) -> ProcessReport {
    run(
        &mut Supervisor::default(),
        scenario,
        ticket(1),
        Client::Ack,
        Arc::new(AtomicBool::new(false)),
    )
}
fn failure(scenario: Scenario, error: &'static str) -> ProcessReport {
    let r = once(scenario);
    assert_eq!(r.receipt.first_error, Some(error));
    r
}
#[test]
fn os01_normal_ready_ack_exit() {
    let r = once(Scenario::Normal);
    assert_eq!(r.receipt.first_error, None);
    assert_eq!(r.receipt.delivery, Delivery::Acknowledged);
    assert_eq!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
    assert_eq!(
        (r.receipt.drain_attempts, r.receipt.release_attempts),
        (1, 1)
    );
    assert!(!r.kill_attempted);
}
#[test]
fn os02_before_retain() {
    let r = failure(Scenario::BeforeRetain, "STOP_BEFORE_RETAIN");
    assert_eq!(r.receipt.owner, Owner::NotAttempted);
}
#[test]
fn os03_unreturned_retain() {
    let r = failure(Scenario::AfterRetain, "STOP_UNRETURNED_RETAIN");
    assert!(r.receipt.unknown_ownership);
    assert_eq!(r.receipt.owner, Owner::Uncertain);
}
#[test]
fn os04_deferred_owner_failure() {
    let r = failure(Scenario::AfterOwner, "STOP_DEFERRED_CHECK");
    assert_eq!(r.receipt.owner, Owner::Uncertain);
}
#[test]
fn os05_released_cannot_make_ready() {
    let r = failure(Scenario::ForgedReleased, "STOP_PROTOCOL_ORDER_OR_TICKET");
    assert_eq!(r.receipt.owner, Owner::NotAttempted);
    assert_eq!(r.receipt.cleanup, Cleanup::Unconfirmed);
}
#[test]
fn os06_wrong_protocol() {
    failure(Scenario::WrongProtocol, "STOP_PROTOCOL_IDENTITY");
}
#[test]
fn os07_wrong_pid() {
    failure(Scenario::WrongIdentity, "STOP_PROTOCOL_IDENTITY");
}
#[test]
fn os08_wrong_generation() {
    failure(Scenario::WrongGeneration, "STOP_PROTOCOL_IDENTITY");
}
#[test]
fn os09_hang_timeout_kill_reap() {
    let r = failure(Scenario::Hang, "STOP_TIMEOUT");
    assert!(r.kill_attempted);
    assert!(r.elapsed_ms >= SESSION_LIMIT.as_millis());
    assert!(r.containment_ms < REAP_LIMIT.as_millis() + 100);
}
#[test]
fn os10_blocked_read_timeout_kill_reap() {
    let r = failure(Scenario::Blocked, "STOP_TIMEOUT");
    assert!(r.kill_attempted);
    assert!(r.containment_ms < REAP_LIMIT.as_millis() + 100);
}
#[test]
fn os11_abnormal_exit_preserves_first() {
    let r = once(Scenario::AbnormalExit);
    assert!(matches!(
        r.receipt.first_error,
        Some("STOP_MISSING_RECEIPT" | "STOP_IPC_IO")
    ));
    assert!(r
        .receipt
        .secondary_errors
        .contains(&"STOP_CHILD_ABNORMAL_EXIT"));
}
#[test]
fn os12_exit_without_final_receipt() {
    let r = failure(Scenario::MissingFinal, "STOP_MISSING_RECEIPT");
    assert_eq!(r.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
}
#[test]
fn os13_malformed_ipc() {
    failure(Scenario::Malformed, "STOP_MALFORMED_IPC");
}
#[test]
fn os14_partial_ipc() {
    failure(Scenario::Partial, "STOP_PARTIAL_IPC");
}
#[test]
fn os15_oversized_ipc_bounded() {
    failure(Scenario::Oversized, "STOP_MALFORMED_IPC");
}
#[test]
fn os16_before_enqueue() {
    let r = failure(Scenario::BeforeEnqueue, "STOP_BEFORE_ENQUEUE");
    assert_eq!(r.receipt.work, Work::None);
}
#[test]
fn os17_post_enqueue_ambiguous() {
    let r = failure(Scenario::AfterEnqueue, "STOP_ENQUEUE_AMBIGUOUS");
    assert_eq!(r.receipt.work, Work::SubmissionUncertain);
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
}
#[test]
fn os18_unconfirmed_fence() {
    let r = failure(Scenario::FenceFailure, "STOP_FENCE_UNCONFIRMED");
    assert_eq!(r.receipt.work, Work::Submitted);
}
#[test]
fn os19_mutated_prepared_bytes() {
    let r = failure(Scenario::MutatedOutput, "STOP_OUTPUT_DIGEST");
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
}
#[test]
fn os20_foreign_fence_ticket() {
    failure(Scenario::WrongTicket, "STOP_PROTOCOL_ORDER_OR_TICKET");
}
#[test]
fn os21_drain_absent() {
    let r = failure(Scenario::DrainAbsent, "STOP_MISSING_RECEIPT");
    assert_eq!(
        (r.receipt.drain_attempts, r.receipt.release_attempts),
        (1, 0)
    );
}
#[test]
fn os22_drain_failed() {
    let r = failure(Scenario::DrainFailure, "STOP_DRAIN_FAILED");
    assert_eq!(
        (r.receipt.drain_attempts, r.receipt.release_attempts),
        (1, 0)
    );
}
#[test]
fn os23_destructor_failed() {
    let r = failure(Scenario::DestructorFailure, "STOP_DESTRUCTOR_FAILED");
    assert_eq!(
        (r.receipt.drain_attempts, r.receipt.release_attempts),
        (1, 1)
    );
}
#[test]
fn os24_cancel_hung_child_kill_reap() {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let task = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        flag.store(true, Ordering::SeqCst);
    });
    let r = run(
        &mut Supervisor::default(),
        Scenario::Hang,
        ticket(1),
        Client::Ack,
        cancel,
    );
    task.join().unwrap();
    assert_eq!(r.receipt.first_error, Some("STOP_CANCELLED"));
    assert!(r.kill_attempted);
    assert!(r.elapsed_ms < 1000);
}
#[test]
fn os25_streamed_without_ack_is_ambiguous() {
    let r = run(
        &mut Supervisor::default(),
        Scenario::Normal,
        ticket(1),
        Client::NoAck,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(r.receipt.first_error, Some("STOP_TIMEOUT"));
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
    assert_eq!(r.receipt.drain_attempts, 0);
}
#[test]
fn os26_partial_stream_cancel_no_replay() {
    let mut s = Supervisor::default();
    let r = run(
        &mut s,
        Scenario::Normal,
        ticket(1),
        Client::PartialCancel,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(r.receipt.first_error, Some("STOP_CANCELLED"));
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
    let r = run(
        &mut s,
        Scenario::Normal,
        ticket(2),
        Client::Ack,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(
        r.receipt.first_error,
        Some("STOP_ADMISSION_REPLAY_OR_IDENTITY")
    );
}
#[test]
fn os27_bad_client_ack() {
    let r = run(
        &mut Supervisor::default(),
        Scenario::Normal,
        ticket(1),
        Client::WrongAck,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(r.receipt.first_error, Some("STOP_DELIVERY_ACK"));
    assert_eq!(r.receipt.delivery, Delivery::Ambiguous);
}
#[test]
fn os28_no_cross_generation_ticket_reuse() {
    let mut s = Supervisor::default();
    let a = run(
        &mut s,
        Scenario::Normal,
        ticket(1),
        Client::Ack,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(a.receipt.first_error, None);
    let b = run(
        &mut s,
        Scenario::Normal,
        ticket(1),
        Client::Ack,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(b.receipt.identity.generation, 2);
    assert_eq!(
        b.receipt.first_error,
        Some("STOP_ADMISSION_REPLAY_OR_IDENTITY")
    );
}
#[test]
fn os29_duplicate_id_in_replacement_generation() {
    let mut s = Supervisor::default();
    let _ = run(
        &mut s,
        Scenario::Normal,
        ticket(1),
        Client::Ack,
        Arc::new(AtomicBool::new(false)),
    );
    let b = run(
        &mut s,
        Scenario::Normal,
        ticket(2),
        Client::Ack,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(
        b.receipt.first_error,
        Some("STOP_ADMISSION_REPLAY_OR_IDENTITY")
    );
}
