//! One-process, one-request CPU MOCK. No CUDA or native GPU code is reachable.
#![forbid(unsafe_code)]
use mer2_g3_a3_s3_p1a::contract::{MockVector, TensorDescriptor, SCHEMA_DOMAIN};
use mer2_g3_a3_s3_p1a::wire::{Codec, Kind};
use mer2_g3_a3_s3_p1a::Result;
use mer2_g3_a3_s1::sha256;
use std::os::unix::net::UnixStream;
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use std::thread;

fn hex_nibble(v: u8) -> Result<u8> {
    match v {
        b'0'..=b'9' => Ok(v - b'0'),
        b'a'..=b'f' => Ok(v - b'a' + 10),
        _ => Err("STOP_WORKER_SESSION_HEX"),
    }
}
fn session_hex(s: &str) -> Result<[u8; 16]> {
    if s.len() != 32 { return Err("STOP_WORKER_SESSION_HEX"); }
    let mut output = [0u8; 16];
    for (i, val) in output.iter_mut().enumerate() {
        let b = s.as_bytes();
        *val = (hex_nibble(b[2 * i])? << 4) | hex_nibble(b[2 * i + 1])?;
    }
    if output == [0u8; 16] { return Err("STOP_WORKER_SESSION_HEX"); }
    Ok(output)
}
fn expected_frame(codec: &mut Codec, stream: &mut UnixStream, kind: Kind, expected: &[u8], end: Instant, cancel: &AtomicBool) -> Result<()> {
    codec.expect(stream, kind, expected, end, cancel)
}
fn emit(codec: &mut Codec, stream: &mut UnixStream, kind: Kind, payload: &[u8], end: Instant, cancel: &AtomicBool) -> Result<()> {
    codec.send(stream, kind, payload, end, cancel)
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 12 { return Err("STOP_WORKER_ARGS"); }
    let generation = args[2].parse::<u64>().map_err(|_| "STOP_WORKER_ARGS")?;
    let session = session_hex(&args[3])?;
    let scenario = args[4].as_str();
    let binding = [&args[5], &args[6], &args[7]];
    let binary_sha = &args[8];
    let schema_sha = &args[9];
    let work_sha = &args[10];
    let input_sha = &args[11];
    for hash in binding.into_iter().chain([binary_sha, schema_sha, work_sha, input_sha]) {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err("STOP_WORKER_HASH_ARGS");
        }
    }
    if schema_sha != &sha256(SCHEMA_DOMAIN) { return Err("STOP_WORKER_SCHEMA"); }
    let mut codec = Codec::new(generation, session)?;
    let mut stream = UnixStream::connect(&args[1]).map_err(|_| "STOP_WORKER_CONNECT")?;
    stream.set_nonblocking(true).map_err(|_| "STOP_WORKER_CONNECT")?;
    let end = Instant::now() + Duration::from_secs(30);
    let cancel = AtomicBool::new(false);
    let mut hello = Vec::with_capacity(132);
    let pid = std::process::id();
    let claimed_pid = if scenario == "wrong-pid" { pid.checked_add(1).ok_or("STOP_WORKER_PID")? } else { pid };
    hello.extend_from_slice(&claimed_pid.to_le_bytes());
    hello.extend_from_slice(binary_sha.as_bytes());
    hello.extend_from_slice(schema_sha.as_bytes());
    emit(&mut codec, &mut stream, Kind::Hello, &hello, end, &cancel)?;
    let mut init = Vec::with_capacity(320);
    for x in [binding[0], binding[1], binding[2], binary_sha, schema_sha] { init.extend_from_slice(x.as_bytes()); }
    expected_frame(&mut codec, &mut stream, Kind::Init, &init, end, &cancel)?;
    if scenario == "before-retain" {
        // Fail before ENTER: no fallible construction has been authorized.
        emit(&mut codec, &mut stream, Kind::Fail, b"BEFORE_RETAIN", end, &cancel)?;
        return Ok(());
    }
    emit(&mut codec, &mut stream, Kind::Enter, b"", end, &cancel)?;
    expected_frame(&mut codec, &mut stream, Kind::Construct, b"", end, &cancel)?;
    if scenario == "unreturned-retain" {
        emit(&mut codec, &mut stream, Kind::Fail, b"UNRETURNED_RETAIN", end, &cancel)?;
        return Ok(());
    }
    // OWNER/CHECKED mean CPU MOCK allocation only, never CUDA context ownership.
    emit(&mut codec, &mut stream, Kind::Owner, b"", end, &cancel)?;
    emit(&mut codec, &mut stream, Kind::Checked, b"", end, &cancel)?;
    emit(&mut codec, &mut stream, Kind::Ready, b"", end, &cancel)?;
    let admit = codec.recv(&mut stream, end, &cancel)?;
    if admit.header.kind != Kind::Admit || sha256(&admit.payload).as_str() != work_sha.as_str() {
        return Err("STOP_WORKER_WORK_BINDING");
    }
    emit(&mut codec, &mut stream, Kind::Accept, work_sha.as_bytes(), end, &cancel)?;
    let input = codec.recv(&mut stream, end, &cancel)?;
    if input.header.kind != Kind::DataInput || sha256(&input.payload).as_str() != input_sha.as_str() {
        return Err("STOP_WORKER_INPUT");
    }
    MockVector::decode(&input.payload)?;
    expected_frame(&mut codec, &mut stream, Kind::Submit, work_sha.as_bytes(), end, &cancel)?;
    emit(&mut codec, &mut stream, Kind::Enqueued, work_sha.as_bytes(), end, &cancel)?;
    if scenario == "post-submit-exit" { std::process::exit(23); }
    if scenario == "hang-after-submit" {
        thread::sleep(Duration::from_secs(31));
        return Err("STOP_WORKER_TIMEOUT");
    }
    emit(&mut codec, &mut stream, Kind::Fenced, work_sha.as_bytes(), end, &cancel)?;
    let mut output = input.payload;
    let work_hash_raw = mer2_g3_a3_s3_p1a::wire::digest_bytes(&admit.payload);
    // Corrupt-before-descriptor is deliberately self-consistent: only the
    // parent's independent identity oracle can catch it.
    if scenario == "self-consistent-wrong-output" { output[0] ^= 1; }
    let descriptor = TensorDescriptor::for_mock(&output, work_hash_raw)?;
    emit(&mut codec, &mut stream, Kind::ResultDescriptor, &descriptor.encode(), end, &cancel)?;
    if scenario == "mutated-output" { output[0] ^= 1; }
    emit(&mut codec, &mut stream, Kind::DataOutput, &output, end, &cancel)?;
    expected_frame(&mut codec, &mut stream, Kind::ClientAck, sha256(&output).as_bytes(), end, &cancel)?;
    expected_frame(&mut codec, &mut stream, Kind::Drain, b"", end, &cancel)?;
    if scenario == "drain-failure" {
        emit(&mut codec, &mut stream, Kind::Fail, b"DRAIN_FAILED", end, &cancel)?;
        return Ok(());
    }
    emit(&mut codec, &mut stream, Kind::Drained, b"", end, &cancel)?;
    expected_frame(&mut codec, &mut stream, Kind::Release, b"", end, &cancel)?;
    if scenario == "release-failure" {
        emit(&mut codec, &mut stream, Kind::Fail, b"RELEASE_FAILED", end, &cancel)?;
        return Ok(());
    }
    emit(&mut codec, &mut stream, Kind::Released, b"", end, &cancel)?;
    expected_frame(&mut codec, &mut stream, Kind::Exit, b"", end, &cancel)?;
    if scenario == "missing-bye" { return Ok(()); }
    emit(&mut codec, &mut stream, Kind::Bye, b"", end, &cancel)?;
    if scenario == "abnormal-after-bye" { std::process::exit(23); }
    if scenario == "duplicate-bye" {
        emit(&mut codec, &mut stream, Kind::Bye, b"", end, &cancel)?;
    } else if scenario == "trailing-frame" {
        emit(&mut codec, &mut stream, Kind::Ready, b"", end, &cancel)?;
    } else if scenario == "trailing-byte" || scenario == "delayed-trailing-byte" {
        if scenario == "delayed-trailing-byte" { thread::sleep(Duration::from_millis(25)); }
        stream.write_all(b"!").map_err(|_| "STOP_WORKER_TRAILING_IO")?;
    }
    Ok(())
}
fn main() {
    if let Err(code) = run() {
        eprintln!("A3N1_CPU_MOCK_STOP={code}");
        std::process::exit(23);
    }
}
