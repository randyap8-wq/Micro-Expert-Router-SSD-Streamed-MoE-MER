//! Deliberately faultable CPU fixture, never a CUDA/model worker.
#![forbid(unsafe_code)]
use mer2_g3_a3_s1::{frame, parse_frame, read_line, Binding, Identity, FAKE_DIGEST};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

struct FakeOwner {
    drained: bool,
}
fn construct(mode: &str) -> Option<FakeOwner> {
    // Represents a fake retain before a fallible owner return. No device call.
    let _retained_cpu_marker = true;
    if mode == "after-retain" {
        None
    } else {
        Some(FakeOwner { drained: false })
    }
}
fn send(s: &mut UnixStream, id: Identity, b: &Binding, body: &str) -> std::io::Result<()> {
    s.write_all(frame(id, b, body).as_bytes())
}
fn receive(s: &mut UnixStream, id: Identity, b: &Binding) -> Result<String, &'static str> {
    let line = read_line(
        s,
        Instant::now() + Duration::from_secs(10),
        &AtomicBool::new(false),
    )?;
    parse_frame(&line, id, b)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err("expected socket generation scenario source model kv".into());
    }
    let id = Identity {
        pid: std::process::id(),
        generation: args[2].parse()?,
    };
    let mode = args[3].as_str();
    let b = Binding {
        source: args[4].clone(),
        model: args[5].clone(),
        kv: args[6].clone(),
    };
    let mut s = UnixStream::connect(&args[1])?;
    s.set_write_timeout(Some(Duration::from_secs(10)))?;
    match mode {
        "hang" => {
            std::thread::sleep(Duration::from_secs(10));
            return Ok(());
        }
        "blocked" => {
            // A real blocking read on a private connected pair. No grandchildren.
            let (mut a, _held_peer) = UnixStream::pair()?;
            let _ = a.read(&mut [0]);
            return Ok(());
        }
        "wrong-protocol" => {
            s.write_all(
                frame(id, &b, "HELLO")
                    .replacen("A3S1", "A3S9", 1)
                    .as_bytes(),
            )?;
            return Ok(());
        }
        "wrong-identity" => {
            send(
                &mut s,
                Identity {
                    pid: id.pid + 1,
                    ..id
                },
                &b,
                "HELLO",
            )?;
            return Ok(());
        }
        "wrong-generation" => {
            send(
                &mut s,
                Identity {
                    generation: id.generation + 1,
                    ..id
                },
                &b,
                "HELLO",
            )?;
            return Ok(());
        }
        "malformed" => {
            s.write_all(b"\xff\n")?;
            return Ok(());
        }
        "partial" => {
            s.write_all(b"A3S1 partial")?;
            return Ok(());
        }
        "oversized" => {
            s.write_all(&vec![b'x'; 2049])?;
            return Ok(());
        }
        _ => (),
    }
    send(&mut s, id, &b, "HELLO")?;
    if mode == "abnormal-exit" {
        std::process::exit(23);
    }
    if receive(&mut s, id, &b)? != "INIT" {
        return Err("expected INIT".into());
    }
    if mode == "before-retain" {
        send(&mut s, id, &b, "FAIL_PRE_RETAIN")?;
        return Ok(());
    }
    if mode == "forged-released" {
        send(&mut s, id, &b, "RELEASED")?;
        return Ok(());
    }
    send(&mut s, id, &b, "ENTER")?;
    if receive(&mut s, id, &b)? != "CONSTRUCT" {
        return Err("expected CONSTRUCT".into());
    }
    let Some(mut owner) = construct(mode) else {
        send(&mut s, id, &b, "FAIL_RETAIN")?;
        return Ok(());
    };
    send(&mut s, id, &b, "OWNER")?;
    if mode == "after-owner" {
        send(&mut s, id, &b, "FAIL_DEFERRED")?;
        return Ok(());
    }
    send(&mut s, id, &b, "CHECKED")?;
    send(&mut s, id, &b, "READY")?;
    let admit = receive(&mut s, id, &b)?;
    let tag = admit.strip_prefix("ADMIT ").ok_or("expected ADMIT")?;
    if mode == "before-enqueue" {
        send(&mut s, id, &b, "FAIL_PRE_ENQUEUE")?;
        return Ok(());
    }
    send(&mut s, id, &b, &format!("ACCEPT {tag}"))?;
    if receive(&mut s, id, &b)? != format!("SUBMIT {tag}") {
        return Err("wrong SUBMIT ticket".into());
    }
    if mode == "after-enqueue" {
        send(&mut s, id, &b, "FAIL_POST_ENQUEUE")?;
        return Ok(());
    }
    send(&mut s, id, &b, &format!("ENQUEUED {tag}"))?;
    if mode == "fence-failure" {
        send(&mut s, id, &b, "FAIL_FENCE")?;
        return Ok(());
    }
    let fence_tag = if mode == "wrong-ticket" {
        "foreign 999"
    } else {
        tag
    };
    send(&mut s, id, &b, &format!("FENCED {fence_tag}"))?;
    send(&mut s, id, &b, &format!("PREPARED {tag} {FAKE_DIGEST}"))?;
    let payload = if mode == "mutated-output" {
        "abd"
    } else {
        "abc"
    };
    send(&mut s, id, &b, &format!("DATA {tag} {payload}"))?;
    if receive(&mut s, id, &b)? != format!("ACK {tag} {FAKE_DIGEST}") {
        return Err("wrong ACK".into());
    }
    if receive(&mut s, id, &b)? != "DRAIN" {
        return Err("expected DRAIN".into());
    }
    if mode == "drain-absent" {
        return Ok(());
    }
    if mode == "drain-failure" {
        send(&mut s, id, &b, "DRAIN_FAILED")?;
        return Ok(());
    }
    owner.drained = true;
    send(&mut s, id, &b, "DRAINED")?;
    if receive(&mut s, id, &b)? != "RELEASE" || !owner.drained {
        return Err("release before drain".into());
    }
    if mode == "destructor-failure" {
        send(&mut s, id, &b, "RELEASE_FAILED")?;
        return Ok(());
    }
    send(&mut s, id, &b, "RELEASED")?;
    if receive(&mut s, id, &b)? != "EXIT" {
        return Err("expected EXIT".into());
    }
    if mode != "missing-final" {
        send(&mut s, id, &b, "BYE")?;
    }
    Ok(())
}
fn main() {
    if run().is_err() {
        std::process::exit(24);
    }
}
