//! Exercise the full production supervisor body, including real ACK receives,
//! real Child::try_wait success, containment and receipt finalization. Only the
//! acceptance clock/observations are injected; no fabricated wait/ACK results.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../src/lib.rs"]
mod subject;

use std::path::Path;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use subject::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    Ack,
    Exit,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    Timely,
    Late,
    Cancelled,
    Both,
    OneMillisecondRemaining,
    AlreadyExpired,
}
struct Probe {
    boundary: Boundary,
    condition: Condition,
    cancel: Arc<AtomicBool>,
    clock: Option<Instant>,
    ack_successes: usize,
    ack_before: Vec<Delivery>,
    waits: Vec<Duration>,
    exit_successes: usize,
    actual_exit: Option<String>,
    commands: Vec<String>,
}
impl Probe {
    fn observe_success(&mut self, boundary: Boundary, deadline: Instant) {
        if boundary != self.boundary {
            return;
        }
        // The syscall/channel operation already succeeded. Deterministically
        // model descheduling and/or cancellation before logical acceptance.
        assert!(!self.cancel.load(Ordering::SeqCst));
        if matches!(self.condition, Condition::Late | Condition::Both) {
            self.clock = Some(deadline);
        }
        if matches!(self.condition, Condition::Cancelled | Condition::Both) {
            self.cancel.store(true, Ordering::SeqCst);
        }
    }
}
impl SuccessBoundary for Probe {
    fn now(&mut self, deadline: Instant) -> Instant {
        match self.condition {
            Condition::OneMillisecondRemaining => deadline - Duration::from_millis(1),
            Condition::AlreadyExpired => deadline,
            _ => self.clock.unwrap_or_else(Instant::now),
        }
    }
    fn ack_wait_returned(
        &mut self,
        result: &std::result::Result<ClientAck, RecvTimeoutError>,
        wait: Duration,
        delivery: Delivery,
        deadline: Instant,
    ) {
        assert!(wait <= Duration::from_millis(5));
        self.waits.push(wait);
        if result.is_ok() {
            self.ack_successes += 1;
            self.ack_before.push(delivery);
            assert_eq!(delivery, Delivery::SentUnacknowledged);
            self.observe_success(Boundary::Ack, deadline);
        }
    }
    fn exit_wait_returned(
        &mut self,
        result: &std::io::Result<Option<ExitStatus>>,
        deadline: Instant,
    ) {
        if let Ok(Some(status)) = result {
            assert!(status.success());
            self.exit_successes += 1;
            self.actual_exit = Some(status.to_string());
            self.observe_success(Boundary::Exit, deadline);
        }
    }
    fn sending(&mut self, body: &str) {
        self.commands.push(body.to_owned());
    }
}

#[derive(Clone, Copy)]
enum AckIdentity {
    Matching,
    WrongTicket,
    WrongDigest,
}
fn run(boundary: Boundary, condition: Condition, identity: AckIdentity) -> (ProcessReport, Probe) {
    let binding = Binding {
        source: "a".repeat(64),
        model: "b".repeat(64),
        kv: "c".repeat(64),
    };
    let ticket = Ticket {
        request: "success-boundary".into(),
        generation: 1,
        nonce: 17,
        binding: binding.clone(),
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let mut probe = Probe {
        boundary,
        condition,
        cancel: cancel.clone(),
        clock: None,
        ack_successes: 0,
        ack_before: vec![],
        waits: vec![],
        exit_successes: 0,
        actual_exit: None,
        commands: vec![],
    };
    let (output, chunks) = mpsc::sync_channel::<ClientChunk>(2);
    let (acknowledge, ack) = mpsc::channel();
    let client = thread::spawn(move || {
        let mut bytes = Vec::new();
        loop {
            let chunk = chunks.recv_timeout(Duration::from_secs(6)).unwrap();
            bytes.extend_from_slice(&chunk.bytes);
            if chunk.final_chunk {
                assert_eq!(bytes, FAKE_OUTPUT);
                let mut reply = ClientAck {
                    ticket: chunk.ticket,
                    digest: chunk.digest,
                };
                match identity {
                    AckIdentity::Matching => (),
                    AckIdentity::WrongTicket => reply.ticket.nonce += 1,
                    AckIdentity::WrongDigest => reply.digest = "e".repeat(64),
                }
                // An already-expired precheck may close the receiver first.
                let _ = acknowledge.send(reply);
                return;
            }
        }
    });
    let mut supervisor = Supervisor::default();
    let report = supervisor
        .execute_with_boundary(
            Path::new(env!("CARGO_BIN_EXE_fake_worker")),
            binding,
            Scenario::Normal,
            ticket,
            output,
            ack,
            cancel,
            &mut probe,
        )
        .unwrap();
    client.join().unwrap();
    eprintln!(
        "BOUNDARY_EVIDENCE boundary={boundary:?} condition={condition:?} ack_successes={} ack_before={:?} ack_after={:?} waits_ns={:?} exit_successes={} actual_exit={:?} commands={:?} report={report:?}",
        probe.ack_successes,
        probe.ack_before,
        report.receipt.delivery,
        probe.waits.iter().map(Duration::as_nanos).collect::<Vec<_>>(),
        probe.exit_successes,
        probe.actual_exit,
        probe.commands,
    );
    assert!(report.reaped && report.receipt.process_exit_observed);
    assert_eq!(report.receipt.phase, Phase::Exited);
    assert_ne!(report.receipt.identity.pid, std::process::id());
    assert_eq!(report.receipt.identity.generation, 1);
    assert_eq!(supervisor.unconfirmed_children(), 0);
    assert!(report.exit_status.is_some());
    assert_eq!(report.receipt.post_exit_baseline, Observation::Unknown);
    assert_eq!(report.receipt.last_cuda_context_drop, "UNOBSERVABLE");
    assert_eq!(report.receipt.leak_free, "NOT_CLAIMED");
    (report, probe)
}
fn exit_case(condition: Condition, expected: Option<&'static str>) {
    let (report, probe) = run(Boundary::Exit, condition, AckIdentity::Matching);
    assert_eq!(probe.exit_successes, 1, "actual try_wait Some arm required");
    assert_eq!(report.exit_status, probe.actual_exit);
    assert_eq!(report.exit_status.as_deref(), Some("exit status: 0"));
    assert!(
        !report.kill_attempted,
        "an already-reaped child must not be killed"
    );
    assert_eq!(report.receipt.first_error, expected);
    assert!(report.receipt.secondary_errors.is_empty());
    assert_eq!(report.receipt.delivery, Delivery::Acknowledged);
    assert_eq!(report.receipt.cleanup, Cleanup::ReleaseCheckedScoped);
    assert_eq!(
        (
            report.receipt.drain_attempts,
            report.receipt.release_attempts
        ),
        (1, 1)
    );
}
fn ack_stop(condition: Condition, identity: AckIdentity, error: &'static str) {
    let (report, probe) = run(Boundary::Ack, condition, identity);
    if condition == Condition::AlreadyExpired {
        assert_eq!(probe.ack_successes, 0);
        assert!(probe.waits.is_empty(), "expired precheck must not receive");
    } else {
        assert_eq!(
            probe.ack_successes, 1,
            "actual recv_timeout Ok arm required"
        );
        assert_eq!(probe.ack_before, [Delivery::SentUnacknowledged]);
    }
    assert_eq!(report.receipt.first_error, Some(error));
    assert_eq!(report.receipt.delivery, Delivery::Ambiguous);
    assert_eq!(report.receipt.cleanup, Cleanup::Unconfirmed);
    assert_eq!(
        (
            report.receipt.drain_attempts,
            report.receipt.release_attempts
        ),
        (0, 0)
    );
    assert_eq!(
        probe.exit_successes, 0,
        "normal-exit path must be skipped after STOP"
    );
    assert!(
        probe.commands.iter().all(|command| {
            !command.starts_with("ACK ")
                && !matches!(command.as_str(), "DRAIN" | "RELEASE" | "EXIT")
        }),
        "no downstream worker command may be attempted after rejected ACK"
    );
    assert!(report.receipt.secondary_errors.iter().all(|e| *e != error));
}
fn timely_ack(condition: Condition) {
    let (report, probe) = run(Boundary::Ack, condition, AckIdentity::Matching);
    assert_eq!(probe.ack_successes, 1);
    assert_eq!(probe.ack_before, [Delivery::SentUnacknowledged]);
    assert_eq!(report.receipt.delivery, Delivery::Acknowledged);
    assert_eq!(report.receipt.first_error, None);
    assert!(report.receipt.secondary_errors.is_empty());
    assert!(!report.kill_attempted);
    assert_eq!(report.exit_status.as_deref(), Some("exit status: 0"));
    assert_eq!(
        (
            report.receipt.drain_attempts,
            report.receipt.release_attempts
        ),
        (1, 1)
    );
    for command in ["DRAIN", "RELEASE", "EXIT"] {
        assert_eq!(
            probe
                .commands
                .iter()
                .filter(|v| v.as_str() == command)
                .count(),
            1
        );
    }
    assert_eq!(
        probe
            .commands
            .iter()
            .filter(|v| v.starts_with("ACK "))
            .count(),
        1
    );
    assert!(!probe.waits.is_empty());
    if condition == Condition::OneMillisecondRemaining {
        assert!(probe
            .waits
            .iter()
            .all(|wait| *wait == Duration::from_millis(1)));
    }
}

#[test]
fn b01_late_exit0_is_reaped_timeout_without_kill() {
    exit_case(Condition::Late, Some("STOP_TIMEOUT"));
}
#[test]
fn b02_cancel_at_successful_exit_wait_is_reaped_without_kill() {
    exit_case(Condition::Cancelled, Some("STOP_CANCELLED"));
}
#[test]
fn b03_simultaneous_exit_cancel_and_timeout_prefers_cancel() {
    exit_case(Condition::Both, Some("STOP_CANCELLED"));
}
#[test]
fn b04_timely_exit0_is_clean_and_never_killed() {
    exit_case(Condition::Timely, None);
}
#[test]
fn b05_late_matching_ack_is_ambiguous_without_downstream_commands() {
    ack_stop(Condition::Late, AckIdentity::Matching, "STOP_TIMEOUT");
}
#[test]
fn b06_cancel_at_successful_ack_wait_is_ambiguous_without_downstream_commands() {
    ack_stop(
        Condition::Cancelled,
        AckIdentity::Matching,
        "STOP_CANCELLED",
    );
}
#[test]
fn b07_simultaneous_ack_cancel_and_timeout_prefers_cancel() {
    ack_stop(Condition::Both, AckIdentity::Matching, "STOP_CANCELLED");
}
#[test]
fn b08_timely_matching_ack_retires_once() {
    timely_ack(Condition::Timely);
}
#[test]
fn b09_wrong_ack_ticket_remains_rejected_and_ambiguous() {
    ack_stop(
        Condition::Timely,
        AckIdentity::WrongTicket,
        "STOP_DELIVERY_ACK",
    );
}
#[test]
fn b10_wrong_ack_digest_remains_rejected_and_ambiguous() {
    ack_stop(
        Condition::Timely,
        AckIdentity::WrongDigest,
        "STOP_DELIVERY_ACK",
    );
}
#[test]
fn b11_ack_wait_is_capped_to_one_millisecond_remaining() {
    timely_ack(Condition::OneMillisecondRemaining);
}
#[test]
fn b12_expired_ack_deadline_never_receives_or_sends_downstream() {
    ack_stop(
        Condition::AlreadyExpired,
        AckIdentity::Matching,
        "STOP_TIMEOUT",
    );
}
