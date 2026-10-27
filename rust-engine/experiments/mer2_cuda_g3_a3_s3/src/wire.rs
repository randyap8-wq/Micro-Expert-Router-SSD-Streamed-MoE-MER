//! Fixed-size authenticated-by-contract (not cryptographically authenticated)
//! binary envelope; peer identity and on-host trust are separate concerns.
use crate::Result;
use mer2_g3_a3_s1::sha256;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub const HEADER_LEN: usize = 80;
pub const MAX_CONTROL: usize = 4096;
pub const MAX_TENSOR: usize = 1024 * 1024;
pub const MAGIC: [u8; 4] = *b"A3N1";
pub const MAJOR: u8 = 1;
pub const MINOR: u8 = 0;
const POLL: Duration = Duration::from_millis(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Hello = 1,
    Init = 2,
    Enter = 3,
    Construct = 4,
    Owner = 5,
    Checked = 6,
    Ready = 7,
    Admit = 8,
    Accept = 9,
    DataInput = 10,
    Submit = 11,
    Enqueued = 12,
    Fenced = 13,
    ResultDescriptor = 14,
    DataOutput = 15,
    ClientAck = 16,
    Drain = 17,
    Drained = 18,
    Release = 19,
    Released = 20,
    Exit = 21,
    Bye = 22,
    Fail = 23,
}
impl Kind {
    pub fn from_byte(v: u8) -> Result<Self> {
        use Kind::*;
        Ok(match v {
            1 => Hello, 2 => Init, 3 => Enter, 4 => Construct,
            5 => Owner, 6 => Checked, 7 => Ready, 8 => Admit,
            9 => Accept, 10 => DataInput, 11 => Submit, 12 => Enqueued,
            13 => Fenced, 14 => ResultDescriptor, 15 => DataOutput,
            16 => ClientAck, 17 => Drain, 18 => Drained,
            19 => Release, 20 => Released, 21 => Exit, 22 => Bye,
            23 => Fail, _ => return Err("STOP_WIRE_KIND"),
        })
    }
    pub fn limit(self) -> usize {
        match self { Self::DataInput | Self::DataOutput => MAX_TENSOR, _ => MAX_CONTROL }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub generation: u64,
    pub sequence: u64,
    pub session: [u8; 16],
    pub payload_len: u32,
    pub payload_sha256: [u8; 32],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub header: Header,
    pub payload: Vec<u8>,
}

pub fn check(deadline: Instant, cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::SeqCst) { return Err("STOP_CANCELLED"); }
    if Instant::now() >= deadline { return Err("STOP_DEADLINE"); }
    Ok(())
}

fn hex_nibble(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err("STOP_WIRE_DIGEST_ENCODING"),
    }
}
pub fn digest_bytes(payload: &[u8]) -> [u8; 32] {
    // The frozen R4 SHA256 implementation was exercised by its own accepted
    // CPU test suite. This new dependency/usage has not been compiled or run.
    let hex = sha256(payload);
    let bytes = hex.as_bytes();
    let mut digest = [0u8; 32];
    for (i, out) in digest.iter_mut().enumerate() {
        // Hex characters are guaranteed by the imported SHA256 formatting.
        let hi = hex_nibble(bytes[2 * i]).expect("SHA256 hex formatting");
        let lo = hex_nibble(bytes[2 * i + 1]).expect("SHA256 hex formatting");
        *out = (hi << 4) | lo;
    }
    digest
}
fn max_payload(kind: Kind, len: usize) -> Result<()> {
    if len > kind.limit() { return Err("STOP_WIRE_OVERSIZED"); }
    Ok(())
}

impl Header {
    pub fn encode(&self) -> Result<[u8; HEADER_LEN]> {
        if self.generation == 0 || self.sequence == 0 || self.session == [0u8; 16] {
            return Err("STOP_WIRE_IDENTITY");
        }
        max_payload(self.kind, self.payload_len as usize)?;
        let mut raw = [0u8; HEADER_LEN];
        raw[0..4].copy_from_slice(&MAGIC);
        raw[4] = MAJOR;
        raw[5] = MINOR;
        raw[6] = self.kind as u8;
        // flags byte = 0
        raw[8..16].copy_from_slice(&self.generation.to_le_bytes());
        raw[16..24].copy_from_slice(&self.sequence.to_le_bytes());
        raw[24..40].copy_from_slice(&self.session);
        raw[40..44].copy_from_slice(&self.payload_len.to_le_bytes());
        raw[44..76].copy_from_slice(&self.payload_sha256);
        // reserved[76..80] = 0
        Ok(raw)
    }
    pub fn decode(raw: &[u8]) -> Result<Self> {
        if raw.len() != HEADER_LEN { return Err("STOP_WIRE_HEADER_SIZE"); }
        if raw[0..4] != MAGIC[..] || raw[4] != MAJOR || raw[5] != MINOR {
            return Err("STOP_WIRE_VERSION");
        }
        if raw[7] != 0 || raw[76..80] != [0u8; 4] {
            return Err("STOP_WIRE_RESERVED");
        }
        let kind = Kind::from_byte(raw[6])?;
        let generation = u64::from_le_bytes(raw[8..16].try_into().map_err(|_| "STOP_WIRE_HEADER_SIZE")?);
        let sequence = u64::from_le_bytes(raw[16..24].try_into().map_err(|_| "STOP_WIRE_HEADER_SIZE")?);
        let session: [u8; 16] = raw[24..40].try_into().map_err(|_| "STOP_WIRE_HEADER_SIZE")?;
        let payload_len = u32::from_le_bytes(raw[40..44].try_into().map_err(|_| "STOP_WIRE_HEADER_SIZE")?);
        let payload_sha256 = raw[44..76].try_into().map_err(|_| "STOP_WIRE_HEADER_SIZE")?;
        if generation == 0 || sequence == 0 || session == [0u8; 16] {
            return Err("STOP_WIRE_IDENTITY");
        }
        max_payload(kind, payload_len as usize)?;
        Ok(Self { kind, generation, sequence, session, payload_len, payload_sha256 })
    }
}

pub fn encode_frame(header: &Header, payload: &[u8]) -> Result<Vec<u8>> {
    max_payload(header.kind, payload.len())?;
    if header.payload_len as usize != payload.len() || header.payload_sha256 != digest_bytes(payload) {
        return Err("STOP_WIRE_DIGEST");
    }
    let mut raw = Vec::with_capacity(HEADER_LEN + payload.len());
    raw.extend_from_slice(&header.encode()?);
    raw.extend_from_slice(payload);
    Ok(raw)
}
pub fn decode_frame_exact(raw: &[u8]) -> Result<Frame> {
    if raw.len() < HEADER_LEN { return Err("STOP_WIRE_HEADER_SIZE"); }
    let header = Header::decode(&raw[..HEADER_LEN])?;
    let expected = HEADER_LEN.checked_add(header.payload_len as usize).ok_or("STOP_WIRE_OVERFLOW")?;
    if raw.len() != expected { return Err("STOP_WIRE_TRUNCATED_OR_TRAILING"); }
    let payload = raw[HEADER_LEN..].to_vec();
    if digest_bytes(&payload) != header.payload_sha256 { return Err("STOP_WIRE_DIGEST"); }
    Ok(Frame { header, payload })
}

fn exact_read(stream: &mut UnixStream, buf: &mut [u8], deadline: Instant, cancel: &AtomicBool) -> Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        check(deadline, cancel)?;
        match stream.read(&mut buf[offset..]) {
            Ok(0) => return Err("STOP_WIRE_EOF"),
            Ok(n) => { offset += n; check(deadline, cancel)?; }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(POLL),
            Err(_) => return Err("STOP_WIRE_READ_IO"),
        }
    }
    Ok(())
}
fn exact_write(stream: &mut UnixStream, buf: &[u8], deadline: Instant, cancel: &AtomicBool) -> Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        check(deadline, cancel)?;
        match stream.write(&buf[offset..]) {
            Ok(0) => return Err("STOP_WIRE_WRITE_ZERO"),
            Ok(n) => { offset += n; check(deadline, cancel)?; }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(POLL),
            Err(_) => return Err("STOP_WIRE_WRITE_IO"),
        }
    }
    Ok(())
}

pub struct Codec {
    pub generation: u64,
    pub session: [u8; 16],
    next_send: u64,
    next_recv: u64,
    poisoned: bool,
}
impl Codec {
    pub fn new(generation: u64, session: [u8; 16]) -> Result<Self> {
        if generation == 0 || session == [0u8; 16] { return Err("STOP_WIRE_IDENTITY"); }
        Ok(Self { generation, session, next_send: 1, next_recv: 1, poisoned: false })
    }
    fn active(&self) -> Result<()> {
        if self.poisoned { return Err("STOP_WIRE_POISONED"); }
        Ok(())
    }
    pub fn send(&mut self, stream: &mut UnixStream, kind: Kind, payload: &[u8], deadline: Instant, cancel: &AtomicBool) -> Result<()> {
        self.active()?;
        let action = (|| {
            check(deadline, cancel)?;
            max_payload(kind, payload.len())?;
            let header = Header {
                kind, generation: self.generation, sequence: self.next_send,
                session: self.session, payload_len: payload.len() as u32,
                payload_sha256: digest_bytes(payload),
            };
            let raw = encode_frame(&header, payload)?;
            exact_write(stream, &raw, deadline, cancel)?;
            self.next_send = self.next_send.checked_add(1).ok_or("STOP_WIRE_SEQUENCE_OVERFLOW")?;
            Ok(())
        })();
        if action.is_err() { self.poisoned = true; }
        action
    }
    pub fn recv(&mut self, stream: &mut UnixStream, deadline: Instant, cancel: &AtomicBool) -> Result<Frame> {
        self.active()?;
        let action = (|| {
            let mut prefix = [0u8; HEADER_LEN];
            exact_read(stream, &mut prefix, deadline, cancel)?;
            let header = Header::decode(&prefix)?;
            if header.generation != self.generation || header.session != self.session {
                return Err("STOP_WIRE_IDENTITY");
            }
            if header.sequence != self.next_recv { return Err("STOP_WIRE_SEQUENCE"); }
            let mut payload = vec![0u8; header.payload_len as usize];
            exact_read(stream, &mut payload, deadline, cancel)?;
            if digest_bytes(&payload) != header.payload_sha256 { return Err("STOP_WIRE_DIGEST"); }
            check(deadline, cancel)?;
            self.next_recv = self.next_recv.checked_add(1).ok_or("STOP_WIRE_SEQUENCE_OVERFLOW")?;
            Ok(Frame { header, payload })
        })();
        if action.is_err() { self.poisoned = true; }
        action
    }
    pub fn expect(&mut self, stream: &mut UnixStream, kind: Kind, payload: &[u8], deadline: Instant, cancel: &AtomicBool) -> Result<()> {
        let frame = self.recv(stream, deadline, cancel)?;
        if frame.header.kind == Kind::Fail {
            self.poisoned = true;
            let code = match frame.payload.as_slice() {
                b"BEFORE_RETAIN" => "STOP_BEFORE_RETAIN",
                b"UNRETURNED_RETAIN" => "STOP_UNRETURNED_RETAIN",
                b"DRAIN_FAILED" => "STOP_DRAIN_FAILED",
                b"RELEASE_FAILED" => "STOP_DESTRUCTOR_FAILED",
                _ => "STOP_WORKER_REPORTED_FAIL",
            };
            return Err(code);
        }
        if frame.header.kind != kind || frame.payload != payload {
            self.poisoned = true;
            return Err("STOP_WIRE_ORDER_OR_PAYLOAD");
        }
        Ok(())
    }
}
