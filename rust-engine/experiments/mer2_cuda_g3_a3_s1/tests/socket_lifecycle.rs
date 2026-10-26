//! R3 additive socket regressions. Include the exact library source so private
//! crate-visible polling loops can be fault-injected without a public test API.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../src/lib.rs"]
mod subject;

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use subject::*;

fn binding() -> Binding {
    Binding {
        source: "a".repeat(64),
        model: "b".repeat(64),
        kv: "c".repeat(64),
    }
}
fn identity() -> Identity {
    Identity {
        pid: std::process::id(),
        generation: 1,
    }
}
fn pair() -> (UnixStream, UnixStream) {
    let (host, peer) = UnixStream::pair().unwrap();
    host.set_nonblocking(true).unwrap();
    (host, peer)
}
fn buffered(bytes: &[u8]) -> UnixStream {
    let (host, mut peer) = pair();
    peer.write_all(bytes).unwrap();
    drop(peer); // The entire frame is buffered and peer closed before any read.
    host
}
fn receive(host: &mut UnixStream) -> subject::Result<String> {
    read_line_poll(
        host,
        Instant::now() + SESSION_LIMIT,
        &AtomicBool::new(false),
    )
}
struct Counted<T> {
    io: T,
    calls: usize,
}
impl<T: Read> Read for Counted<T> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.calls += 1;
        self.io.read(buf)
    }
}
impl<T: Write> Write for Counted<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        self.io.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}
fn fill(host: &mut UnixStream) -> usize {
    let started = Instant::now();
    let mut total = 0;
    loop {
        assert!(total < 8 * 1024 * 1024 && started.elapsed() < Duration::from_secs(1));
        match host.write(&[b'x'; 4096]) {
            Ok(0) => panic!("zero write while filling socket"),
            Ok(n) => total += n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return total,
            Err(e) => panic!("fill socket: {e}"),
        }
    }
}
fn delayed_drain(mut peer: UnixStream, filled: usize, suffix: Vec<u8>) -> thread::JoinHandle<()> {
    peer.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        let mut bytes = vec![0; filled + suffix.len()];
        peer.read_exact(&mut bytes).unwrap();
        assert!(bytes[..filled].iter().all(|&b| b == b'x'));
        assert_eq!(&bytes[filled..], suffix);
    })
}

#[test]
fn r01_closed_peer_buffered_bye_parsed_before_eof() {
    let expected = frame(identity(), &binding(), "BYE");
    let mut host = buffered(expected.as_bytes());
    let line = receive(&mut host).unwrap();
    assert_eq!(line, expected);
    assert_eq!(parse_frame(&line, identity(), &binding()), Ok("BYE".into()));
    assert_eq!(receive(&mut host), Err("STOP_MISSING_RECEIPT"));
    eprintln!(
        "SOCKET_EVIDENCE buffered_bye_bytes={} parsed=BYE next=STOP_MISSING_RECEIPT",
        line.len()
    );
}
#[test]
fn r02_closed_peer_without_bye_is_missing() {
    assert_eq!(receive(&mut buffered(b"")), Err("STOP_MISSING_RECEIPT"));
}
#[test]
fn r03_partial_line_before_eof_is_partial() {
    assert_eq!(
        receive(&mut buffered(b"A3S1 partial")),
        Err("STOP_PARTIAL_IPC")
    );
}
#[test]
fn r04_malformed_final_frame_stays_rejected() {
    assert_eq!(receive(&mut buffered(b"\xff\n")), Err("STOP_MALFORMED_IPC"));
    let line = frame(identity(), &binding(), "BYE\r");
    let line = receive(&mut buffered(line.as_bytes())).unwrap();
    assert_eq!(
        parse_frame(&line, identity(), &binding()),
        Err("STOP_MALFORMED_IPC")
    );
}
#[test]
fn r05_forged_final_identity_stays_rejected() {
    let forged = Identity {
        generation: 2,
        ..identity()
    };
    let line = frame(forged, &binding(), "BYE");
    let line = receive(&mut buffered(line.as_bytes())).unwrap();
    assert_eq!(
        parse_frame(&line, identity(), &binding()),
        Err("STOP_PROTOCOL_IDENTITY")
    );
}
#[test]
fn r06_frame_bound_is_unchanged() {
    assert_eq!(
        receive(&mut buffered(&vec![b'x'; MAX_FRAME + 1])),
        Err("STOP_MALFORMED_IPC")
    );
}
#[test]
fn r07_silent_read_has_absolute_two_second_deadline_and_bounded_polls() {
    let (host, _peer) = pair();
    let mut host = Counted { io: host, calls: 0 };
    let start = Instant::now();
    let result = read_line_poll(&mut host, start + SESSION_LIMIT, &AtomicBool::new(false));
    let elapsed = start.elapsed();
    eprintln!(
        "SOCKET_EVIDENCE silent_read elapsed_us={} read_calls={} result={result:?}",
        elapsed.as_micros(),
        host.calls
    );
    assert_eq!(result, Err("STOP_TIMEOUT"));
    assert!(elapsed >= SESSION_LIMIT && elapsed < SESSION_LIMIT + Duration::from_millis(500));
    assert!(
        host.calls > 0 && host.calls <= 450,
        "WouldBlock must sleep, not spin"
    );
}
struct LateRead;
impl Read for LateRead {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        // Models descheduling across a successful nonblocking read.
        thread::sleep(Duration::from_millis(30));
        bytes[0] = b'\n';
        Ok(1)
    }
}
struct LateWrite;
impl Write for LateWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        thread::sleep(Duration::from_millis(30));
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn r08_successful_io_cannot_accept_after_deadline() {
    assert_eq!(
        read_line_poll(
            &mut LateRead,
            Instant::now() + Duration::from_millis(10),
            &AtomicBool::new(false)
        ),
        Err("STOP_TIMEOUT")
    );
    assert_eq!(
        subject::send(
            &mut LateWrite,
            "EXIT\n",
            Instant::now() + Duration::from_millis(10),
            &AtomicBool::new(false)
        ),
        Err("STOP_TIMEOUT")
    );
}
#[test]
fn r09_cancel_during_read_is_bounded() {
    let (host, _peer) = pair();
    let mut host = Counted { io: host, calls: 0 };
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let task = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        flag.store(true, Ordering::SeqCst);
    });
    let start = Instant::now();
    let result = read_line_poll(&mut host, start + SESSION_LIMIT, &cancel);
    task.join().unwrap();
    eprintln!(
        "SOCKET_EVIDENCE read_cancel elapsed_us={} read_calls={} result={result:?}",
        start.elapsed().as_micros(),
        host.calls
    );
    assert_eq!(result, Err("STOP_CANCELLED"));
    assert!(start.elapsed() < Duration::from_millis(500));
    assert!(host.calls <= 100);
}
struct Fault(i32);
impl Read for Fault {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::from_raw_os_error(self.0))
    }
}
impl Write for Fault {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::from_raw_os_error(self.0))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn r10_nontransient_read_and_write_errors_are_stop() {
    for errno in [5, 22, 32] {
        let mut io = Counted {
            io: Fault(errno),
            calls: 0,
        };
        assert_eq!(
            read_line_poll(
                &mut io,
                Instant::now() + SESSION_LIMIT,
                &AtomicBool::new(false)
            ),
            Err("STOP_IPC_IO")
        );
        assert_eq!(io.calls, 1);
        assert_eq!(
            subject::send(
                &mut io,
                "EXIT\n",
                Instant::now() + SESSION_LIMIT,
                &AtomicBool::new(false)
            ),
            Err("STOP_IPC_IO")
        );
        assert_eq!(io.calls, 2);
        eprintln!("SOCKET_EVIDENCE injected_errno={errno} read=STOP_IPC_IO write=STOP_IPC_IO");
    }
}
#[test]
fn r11_real_closed_peer_write_is_stop() {
    let (mut host, peer) = pair();
    drop(peer);
    let result = subject::send(
        &mut host,
        "EXIT\n",
        Instant::now() + SESSION_LIMIT,
        &AtomicBool::new(false),
    );
    eprintln!("SOCKET_EVIDENCE real_closed_peer_write={result:?}");
    assert_eq!(result, Err("STOP_IPC_IO"));
}
#[test]
fn r12_write_backpressure_times_out_without_spinning() {
    let (mut host, _peer) = pair();
    let filled = fill(&mut host);
    let mut host = Counted { io: host, calls: 0 };
    let start = Instant::now();
    let bound = Duration::from_millis(100);
    let result = subject::send(&mut host, "EXIT\n", start + bound, &AtomicBool::new(false));
    eprintln!("SOCKET_EVIDENCE write_pressure filled={filled} elapsed_us={} write_calls={} result={result:?}", start.elapsed().as_micros(), host.calls);
    assert_eq!(result, Err("STOP_TIMEOUT"));
    assert!(start.elapsed() >= bound && start.elapsed() < Duration::from_millis(500));
    assert!(host.calls > 0 && host.calls <= 30);
}
#[test]
fn r13_write_backpressure_recovers_without_lost_or_duplicate_bytes() {
    let (mut host, peer) = pair();
    let filled = fill(&mut host);
    let message = frame(identity(), &binding(), "EXIT");
    let task = delayed_drain(peer, filled, message.as_bytes().to_vec());
    let mut host = Counted { io: host, calls: 0 };
    let result = subject::send(
        &mut host,
        &message,
        Instant::now() + SESSION_LIMIT,
        &AtomicBool::new(false),
    );
    task.join().unwrap();
    eprintln!(
        "SOCKET_EVIDENCE write_recovery filled={filled} write_calls={} result={result:?}",
        host.calls
    );
    assert_eq!(result, Ok(()));
    assert!(host.calls > 1 && host.calls <= 200);
}
#[derive(Default)]
struct PartialWriter {
    calls: usize,
    bytes: Vec<u8>,
}
impl Write for PartialWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.calls == 2 {
            return Err(io::ErrorKind::WouldBlock.into());
        }
        if self.calls == 3 {
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.calls == 4 {
            return Err(io::ErrorKind::TimedOut.into());
        }
        let n = bytes.len().min(2);
        self.bytes.extend_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn r14_partial_writes_and_transients_preserve_bytes() {
    let mut writer = PartialWriter::default();
    assert_eq!(
        subject::send(
            &mut writer,
            "EXIT\n",
            Instant::now() + SESSION_LIMIT,
            &AtomicBool::new(false)
        ),
        Ok(())
    );
    assert_eq!(writer.bytes, b"EXIT\n");
    assert_eq!(writer.calls, 6);
}
#[test]
fn r15_io_error_keeps_first_error_and_terminal_quarantine() {
    let mut contract = Contract::new(identity(), binding(), Ledger::default()).unwrap();
    assert_eq!(contract.fault("STOP_FIRST"), Err("STOP_FIRST"));
    let err = subject::send(
        &mut Fault(32),
        "EXIT\n",
        Instant::now() + SESSION_LIMIT,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    let _ = contract.fault(err);
    let receipt = contract.receipt();
    assert_eq!(receipt.first_error, Some("STOP_FIRST"));
    assert_eq!(receipt.secondary_errors, ["STOP_IPC_IO"]);
    assert_eq!(receipt.phase, Phase::Quarantined);
    assert!(contract.ready().is_err());
    assert_eq!(contract.receipt().first_error, Some("STOP_FIRST"));
}
#[test]
fn r16_expired_or_cancelled_before_io_makes_no_call() {
    let mut io = Counted {
        io: Fault(22),
        calls: 0,
    };
    let past = Instant::now() - Duration::from_millis(1);
    assert_eq!(
        read_line_poll(&mut io, past, &AtomicBool::new(false)),
        Err("STOP_TIMEOUT")
    );
    assert_eq!(
        subject::send(&mut io, "EXIT\n", past, &AtomicBool::new(false)),
        Err("STOP_TIMEOUT")
    );
    assert_eq!(
        read_line_poll(&mut io, past, &AtomicBool::new(true)),
        Err("STOP_CANCELLED")
    );
    assert_eq!(
        subject::send(&mut io, "EXIT\n", past, &AtomicBool::new(true)),
        Err("STOP_CANCELLED")
    );
    assert_eq!(io.calls, 0);
}
#[test]
fn r17_cancel_during_write_backpressure_is_bounded() {
    let (mut host, _peer) = pair();
    fill(&mut host);
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let task = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        flag.store(true, Ordering::SeqCst);
    });
    let start = Instant::now();
    let result = subject::send(&mut host, "EXIT\n", start + SESSION_LIMIT, &cancel);
    task.join().unwrap();
    assert_eq!(result, Err("STOP_CANCELLED"));
    assert!(start.elapsed() < Duration::from_millis(500));
}
#[test]
fn r18_blocking_caller_write_mode_restored_after_success_and_error() {
    for success in [true, false] {
        let (mut host, mut peer) = pair();
        let filled = fill(&mut host);
        host.set_nonblocking(false).unwrap();
        host.set_write_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        if success {
            peer.write_all(b"BYE\n").unwrap();
        }
        let result = read_line(
            &mut host,
            Instant::now() + Duration::from_millis(30),
            &AtomicBool::new(false),
        );
        if success {
            assert_eq!(result, Ok("BYE\n".into()));
        } else {
            assert_eq!(result, Err("STOP_TIMEOUT"));
        }
        let task = delayed_drain(peer, filled, b"ACK\n".to_vec());
        let write = host.write_all(b"ACK\n");
        task.join().unwrap();
        assert!(
            write.is_ok(),
            "blocking caller must retain working write_all: {write:?}"
        );
    }
}
