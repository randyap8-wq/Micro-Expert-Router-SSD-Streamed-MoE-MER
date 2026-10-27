//! Unexecuted static CPU-only test draft. Not evidence of a PASS.
#![forbid(unsafe_code)]
use mer2_g3_a3_s3_p1a::contract::{MockVector, TensorDescriptor};
use mer2_g3_a3_s3_p1a::wire::{self, Codec, Frame, Header, Kind, HEADER_LEN, MAX_CONTROL, MAX_TENSOR};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

fn header(kind: Kind, data: &[u8]) -> Header {
    Header {
        kind, generation: 9, sequence: 1, session: [3u8; 16],
        payload_len: data.len() as u32, payload_sha256: wire::digest_bytes(data),
    }
}
fn encoded(kind: Kind, data: &[u8]) -> Vec<u8> {
    wire::encode_frame(&header(kind, data), data).unwrap()
}

#[test]
fn w01_valid_fixed_80_byte_header_and_binary_payload() {
    let payload = vec![0x01, 0, 0x7f, 0xff, 0x55];
    let raw = encoded(Kind::DataInput, &payload);
    assert_eq!(raw.len(), HEADER_LEN + payload.len());
    let out = wire::decode_frame_exact(&raw).unwrap();
    assert_eq!(out.payload, payload);
    assert_eq!(out.header, header(Kind::DataInput, &payload));
}
#[test]
fn w02_magic_and_major_version_rejected() {
    let mut raw = encoded(Kind::Hello, b"abc");
    raw[0] = b'X';
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_VERSION");
    raw[0] = b'A'; raw[4] = 2;
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_VERSION");
}
#[test]
fn w03_minor_version_rejected() {
    let mut raw = encoded(Kind::Init, b"abc"); raw[5] = 1;
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_VERSION");
}
#[test]
fn w04_unknown_kind_rejected() {
    let mut raw = encoded(Kind::Enter, b""); raw[6] = 254;
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_KIND");
}
#[test]
fn w05_flags_and_reserved_must_be_zero() {
    let mut raw = encoded(Kind::Construct, b""); raw[7] = 1;
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_RESERVED");
    raw[7] = 0; raw[79] = 1;
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_RESERVED");
}
#[test]
fn w06_truncated_and_trailing_payload_rejected() {
    let raw = encoded(Kind::Admit, b"hello");
    assert_eq!(wire::decode_frame_exact(&raw[..79]).unwrap_err(), "STOP_WIRE_HEADER_SIZE");
    assert_eq!(wire::decode_frame_exact(&raw[..raw.len() - 1]).unwrap_err(), "STOP_WIRE_TRUNCATED_OR_TRAILING");
    let mut extra = raw; extra.push(1);
    assert_eq!(wire::decode_frame_exact(&extra).unwrap_err(), "STOP_WIRE_TRUNCATED_OR_TRAILING");
}
#[test]
fn w07_zero_generation_or_sequence_or_session_rejected() {
    let mut raw = encoded(Kind::Ready, b"");
    raw[8..16].fill(0);
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_IDENTITY");
    raw = encoded(Kind::Ready, b""); raw[16..24].fill(0);
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_IDENTITY");
    raw = encoded(Kind::Ready, b""); raw[24..40].fill(0);
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_IDENTITY");
}
#[test]
fn b01_control_and_tensor_caps_are_distinct() {
    assert!(wire::encode_frame(&header(Kind::Init, &vec![0u8; MAX_CONTROL]), &vec![0u8; MAX_CONTROL]).is_ok());
    assert_eq!(wire::encode_frame(&header(Kind::Init, &vec![0u8; MAX_CONTROL + 1]), &vec![0u8; MAX_CONTROL + 1]).unwrap_err(), "STOP_WIRE_OVERSIZED");
    assert!(wire::encode_frame(&header(Kind::DataInput, &vec![0u8; MAX_TENSOR]), &vec![0u8; MAX_TENSOR]).is_ok());
    assert_eq!(wire::encode_frame(&header(Kind::DataInput, &vec![0u8; MAX_TENSOR + 1]), &vec![0u8; MAX_TENSOR + 1]).unwrap_err(), "STOP_WIRE_OVERSIZED");
}
#[test]
fn b02_header_length_checked_before_payload_allocation() {
    let mut raw = encoded(Kind::DataOutput, b"");
    raw[40..44].copy_from_slice(&(MAX_TENSOR as u32 + 1).to_le_bytes());
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_OVERSIZED");
    raw[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_OVERSIZED");
}
#[test]
fn b03_wrong_endian_payload_length_rejected() {
    let mut raw = encoded(Kind::Init, b"abc");
    raw[40..44].copy_from_slice(&3u32.to_be_bytes());
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_OVERSIZED");
}
#[test]
fn b04_digest_mismatch_rejected() {
    let mut raw = encoded(Kind::DataInput, b"abc");
    *raw.last_mut().unwrap() ^= 1;
    assert_eq!(wire::decode_frame_exact(&raw).unwrap_err(), "STOP_WIRE_DIGEST");
}
#[test]
fn b05_generation_and_sequence_checked_before_state_advance() {
    let (mut sender, mut receiver) = UnixStream::pair().unwrap();
    sender.set_nonblocking(true).unwrap(); receiver.set_nonblocking(true).unwrap();
    let raw = encoded(Kind::Hello, b"");
    sender.write_all(&raw).unwrap();
    let mut codec = Codec::new(9, [3u8; 16]).unwrap();
    let cancel = AtomicBool::new(false);
    let deadline = Instant::now() + Duration::from_secs(2);
    let ok = codec.recv(&mut receiver, deadline, &cancel).unwrap();
    assert_eq!(ok.header.sequence, 1);
    sender.write_all(&raw).unwrap();
    assert_eq!(codec.recv(&mut receiver, deadline, &cancel).unwrap_err(), "STOP_WIRE_SEQUENCE");
    assert_eq!(codec.recv(&mut receiver, deadline, &cancel).unwrap_err(), "STOP_WIRE_POISONED");
}
#[test]
fn b06_cancel_before_send_no_bytes_written() {
    let (mut sender, mut receiver) = UnixStream::pair().unwrap();
    sender.set_nonblocking(true).unwrap(); receiver.set_nonblocking(true).unwrap();
    let mut codec = Codec::new(1, [1u8; 16]).unwrap();
    let cancel = AtomicBool::new(true);
    assert_eq!(codec.send(&mut sender, Kind::Init, b"abc", Instant::now() + Duration::from_secs(1), &cancel).unwrap_err(), "STOP_CANCELLED");
    let mut b = [0u8; 1];
    use std::io::Read;
    assert_eq!(receiver.read(&mut b).unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}
#[test]
fn t01_d64_f32_finite_roundtrip() {
    let vec = MockVector::checked([0.25f32; 64]).unwrap();
    let bytes = vec.bytes();
    assert_eq!(bytes.len(), 256);
    assert_eq!(MockVector::decode(&bytes).unwrap(), vec);
    let work_sha = wire::digest_bytes(b"work");
    let descriptor = TensorDescriptor::for_mock(&bytes, work_sha).unwrap();
    assert_eq!(TensorDescriptor::decode(&descriptor.encode()).unwrap(), descriptor);
    assert!(descriptor.verify(&bytes, &work_sha).is_ok());
}
#[test]
fn t02_nonfinite_input_and_wrong_shape_stop() {
    assert_eq!(MockVector::checked([f32::NAN; 64]).unwrap_err(), "STOP_TENSOR_NONFINITE");
    assert_eq!(MockVector::decode(&[0u8; 255]).unwrap_err(), "STOP_TENSOR_LENGTH");
    assert_eq!(MockVector::decode(&[0xffu8; 256]).unwrap_err(), "STOP_TENSOR_NONFINITE");
}
#[test]
fn t03_tensor_layout_flags_and_work_hash_bound() {
    let bytes = MockVector::checked([1.0f32; 64]).unwrap().bytes();
    let work = wire::digest_bytes(b"wire-request");
    let descriptor = TensorDescriptor::for_mock(&bytes, work).unwrap();
    let mut raw = descriptor.encode(); raw[2] = 2;
    assert_eq!(TensorDescriptor::decode(&raw).unwrap_err(), "STOP_TENSOR_DESCRIPTOR_LAYOUT");
    assert_eq!(descriptor.verify(&bytes, &wire::digest_bytes(b"foreign")).unwrap_err(), "STOP_TENSOR_DESCRIPTOR_BINDING");
}
#[test]
fn t04_mutated_payload_digest_stop() {
    let mut bytes = MockVector::checked([1.0f32; 64]).unwrap().bytes();
    let work = wire::digest_bytes(b"request");
    let descriptor = TensorDescriptor::for_mock(&bytes, work).unwrap();
    bytes[0] ^= 1;
    assert_eq!(descriptor.verify(&bytes, &work).unwrap_err(), "STOP_TENSOR_DESCRIPTOR_BINDING");
}
#[test]
fn t05_cuda_is_an_explicit_refusal() {
    assert_eq!(mer2_g3_a3_s3_p1a::native_cuda_unavailable().unwrap_err(), "STOP_CUDA_NOT_AUTHORIZED");
}

// Corrective review additions: all SOURCE ONLY, never executed in this gate.
#[test]
fn w08_golden_header_field_offsets() {
    // Known SHA-256("abc") digest, encoded independently of Header::encode.
    let digest = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    ];
    let hdr = Header {
        kind: Kind::DataInput,
        generation: 0x1122334455667788,
        sequence: 0x8877665544332211,
        session: [0xa5; 16],
        payload_len: 3,
        payload_sha256: digest,
    };
    let mut expected = [0u8; HEADER_LEN];
    expected[0..4].copy_from_slice(b"A3N1");
    expected[4] = 1;
    expected[6] = 10;
    expected[8..16].copy_from_slice(&[0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
    expected[16..24].copy_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    expected[24..40].fill(0xa5);
    expected[40] = 3;
    expected[44..76].copy_from_slice(&digest);
    assert_eq!(hdr.encode().unwrap(), expected);
    let mut complete = expected.to_vec();
    complete.extend_from_slice(b"abc");
    assert_eq!(wire::decode_frame_exact(&complete).unwrap().header, hdr);
    assert_eq!(wire::digest_bytes(b"abc"), digest);
}
#[test]
fn b07_sequence_identity_and_poison_matrix() {
    for (which, generation, session, sequence, expected) in [
        ("generation", 10u64, [3u8;16], 1u64, "STOP_WIRE_IDENTITY"),
        ("session", 9u64, [4u8;16], 1u64, "STOP_WIRE_IDENTITY"),
        ("sequence-gap", 9u64, [3u8;16], 2u64, "STOP_WIRE_SEQUENCE"),
    ] {
        let (mut sender, mut receiver) = UnixStream::pair().unwrap();
        // A3N1 Codec is now responsible for enabling nonblocking mode.
        let payload = b"a";
        let hdr = Header {
            kind: Kind::Hello, generation, sequence, session,
            payload_len: payload.len() as u32, payload_sha256: wire::digest_bytes(payload),
        };
        sender.write_all(&wire::encode_frame(&hdr, payload).unwrap()).unwrap();
        let mut codec = Codec::new(9, [3u8; 16]).unwrap();
        let cancel = AtomicBool::new(false);
        let limit = Instant::now() + Duration::from_secs(2);
        assert_eq!(codec.recv(&mut receiver, limit, &cancel).unwrap_err(), expected, "{which}");
        assert_eq!(codec.recv(&mut receiver, limit, &cancel).unwrap_err(), "STOP_WIRE_POISONED");
    }
}
#[test]
fn b08_partial_io_and_success_boundary_matrix() {
    let (mut sender, mut receiver) = UnixStream::pair().unwrap();
    let frame = encoded(Kind::Bye, b"");
    let writer = std::thread::spawn(move || {
        // Fragment the 80-byte header deliberately; no partial header can be
        // mistaken for a completed frame.
        for chunk in frame.chunks(7) {
            sender.write_all(chunk).unwrap();
        }
        // Closing the sending end supplies the required clean terminal EOF.
    });
    let mut codec = Codec::new(9, [3u8; 16]).unwrap();
    let cancel = AtomicBool::new(false);
    let deadline = Instant::now() + Duration::from_secs(3);
    codec.expect(&mut receiver, Kind::Bye, b"", deadline, &cancel).unwrap();
    assert!(codec.expect_clean_eof(&mut receiver, deadline, &cancel).is_ok());
    assert_eq!(codec.recv(&mut receiver, deadline, &cancel).unwrap_err(), "STOP_WIRE_POISONED");
    writer.join().unwrap();

    let (mut sender, mut receiver) = UnixStream::pair().unwrap();
    sender.write_all(&encoded(Kind::Bye, b"")).unwrap();
    sender.write_all(b"!").unwrap();
    drop(sender);
    let mut codec = Codec::new(9, [3u8; 16]).unwrap();
    let second_deadline = Instant::now() + Duration::from_secs(3);
    codec.expect(&mut receiver, Kind::Bye, b"", second_deadline, &cancel).unwrap();
    assert_eq!(codec.expect_clean_eof(&mut receiver, second_deadline, &cancel).unwrap_err(),
               "STOP_WIRE_TRAILING_AFTER_BYE");
    assert_eq!(codec.recv(&mut receiver, deadline, &cancel).unwrap_err(), "STOP_WIRE_POISONED");
}
#[test]
fn t06_descriptor_metadata_matrix() {
    let bytes = MockVector::checked([0.5f32; 64]).unwrap().bytes();
    let work = wire::digest_bytes(b"request");
    let raw = TensorDescriptor::for_mock(&bytes, work).unwrap().encode();
    for offset in 0..40 {
        let mut mutated = raw;
        mutated[offset] ^= 0x01;
        assert_eq!(TensorDescriptor::decode(&mutated).unwrap_err(),
                   "STOP_TENSOR_DESCRIPTOR_LAYOUT", "offset={offset}");
    }
    assert_eq!(TensorDescriptor::decode(&raw).unwrap().encode(), raw);
}
