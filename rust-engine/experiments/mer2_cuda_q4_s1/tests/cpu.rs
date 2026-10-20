use mer2_cuda_q4_s1::{
    compare, dequantize, fixtures, preflight, sha256, Fixture, ATOL, KERNEL, KERNEL_SHA256,
};

fn block(bits: u16, packed: u8) -> Vec<u8> {
    let mut out = bits.to_le_bytes().to_vec();
    out.extend([packed; 16]);
    out
}

fn known(rows: usize, cols: usize) -> Fixture {
    Fixture {
        name: "known",
        rows,
        cols,
        weights: block(0x3800, 0xf0).repeat(rows * cols / 32),
        input: (1..=cols).map(|x| x as f32).collect(),
    }
}

#[test]
fn packed_bytes_and_split_nibbles() {
    let bytes = block(0x3800, 0xf0);
    assert_eq!(&bytes[..2], &[0x00, 0x38]);
    assert_eq!(&bytes[2..], &[0xf0; 16]);
    assert_eq!(bytes.len(), 18);
    let values = dequantize(&bytes).unwrap();
    assert_eq!(&values[..16], &[-4.0; 16]);
    assert_eq!(&values[16..], &[3.5; 16]);
}

#[test]
fn every_nibble_maps_to_its_position() {
    let mut bytes = vec![0x00, 0x3c]; // +1.0, little endian
    bytes.extend((0..16u8).map(|j| ((15 - j) << 4) | j));
    let out = dequantize(&bytes).unwrap();
    for j in 0..16 {
        assert_eq!(out[j], j as f32 - 8.0);
        assert_eq!(out[j + 16], 7.0 - j as f32);
    }
}

#[test]
fn signed_scales_and_little_endian_half() {
    assert_eq!(dequantize(&block(0x3c00, 0x99)).unwrap(), [1.0; 32]);
    assert_eq!(dequantize(&block(0xbc00, 0x99)).unwrap(), [-1.0; 32]);
    assert_eq!(
        dequantize(&block(0x0001, 0x99)).unwrap(),
        [2f32.powi(-24); 32]
    );
    assert_eq!(
        dequantize(&block(0x8000, 0x99)).unwrap()[0].to_bits(),
        0x80000000
    );
    assert_ne!(dequantize(&block(0x003c, 0x99)).unwrap(), [1.0; 32]);
}

#[test]
fn zero_scales_and_zero_quants() {
    assert_eq!(dequantize(&block(0, 0xf0)).unwrap(), [0.0; 32]);
    assert_eq!(dequantize(&block(0x3c00, 0x88)).unwrap(), [0.0; 32]);
    assert_eq!(fixtures()[0].reference().unwrap(), [0.0]);
}

#[test]
fn malformed_blocks_and_nonfinite_scales_fail() {
    assert!(dequantize(&[0; 17]).is_err());
    assert!(dequantize(&[0; 19]).is_err());
    for bits in [0x7c00, 0xfc00, 0x7e00] {
        assert!(dequantize(&block(bits, 0x88)).is_err());
    }
}

#[test]
fn independently_calculated_projection_values() {
    // -4 * sum(1..=16) + 3.5 * sum(17..=32) = 828.
    assert_eq!(known(1, 32).reference().unwrap(), [828.0]);
    // Additional blocks contribute 572 and 316, respectively.
    assert_eq!(known(3, 64).reference().unwrap(), [1400.0; 3]);
    assert_eq!(known(2, 96).reference().unwrap(), [1716.0; 2]);
    let mut negative = known(1, 32);
    negative.weights = block(0xb800, 0xf0);
    assert_eq!(negative.reference().unwrap(), [-828.0]);
}

#[test]
fn row_and_block_scale_indexing() {
    let mut f = known(2, 64);
    f.weights = [
        block(0x3800, 0xf0),
        block(0xb800, 0xf0),
        block(0xbc00, 0xf0),
        block(0x3c00, 0xf0),
    ]
    .concat();
    assert_eq!(f.reference().unwrap(), [256.0, -512.0]);
}

#[test]
fn shape_and_length_guards() {
    for (rows, cols) in [(0, 32), (1, 0), (33, 32), (1, 31), (1, 4097)] {
        assert!(known(rows, cols).validate().is_err());
    }
    let mut f = known(1, 32);
    f.weights.push(0);
    assert!(f.validate().is_err());
    f.weights.pop();
    f.input.pop();
    assert!(f.validate().is_err());
}

#[test]
fn nonfinite_input_and_accumulation_fail() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut f = known(1, 32);
        f.input[0] = value;
        assert!(f.reference().is_err());
    }
    let mut f = known(1, 32);
    f.input.fill(f32::MAX);
    assert!(f.reference().is_err());
}

#[test]
fn cuda_tail_and_c_int_bounds_are_checked_without_cuda() {
    assert!(known(1, 32).cuda_dimensions().is_err());
    assert!(known(1, 96).cuda_dimensions().is_err());
    for f in fixtures() {
        assert_eq!(f.cuda_dimensions().unwrap(), (f.cols as i32, f.rows as i32));
    }
    let mut f = known(1, 64);
    f.rows = usize::MAX;
    assert!(f.cuda_dimensions().is_err());
}

#[test]
fn numerical_acceptance_is_finite_and_shape_strict() {
    assert!(compare(&[], &[]).is_err());
    assert!(compare(&[1.0], &[1.0, 2.0]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(compare(&[0.0], &[bad]).is_err());
        assert!(compare(&[bad], &[0.0]).is_err());
    }
    assert!(compare(&[0.0], &[(ATOL * 0.5) as f32]).is_ok());
    assert!(compare(&[0.0], &[(ATOL * 2.0) as f32]).is_err());
    assert!(compare(&[1000.0], &[1000.005]).is_ok());
    assert!(compare(&[1000.0], &[1000.02]).is_err());
    assert!(compare(&[-1000.0], &[-1000.005]).is_ok());
}

#[test]
fn fixture_bytes_match_independent_python_sha256_receipts() {
    let expected = [
        "9d457eb950d411095c5f1fcf64024f04e67ebf1ce9ceb33d77e2060527675acd",
        "6a5d5c4d51e9ca0d3f265fde6791fed9cbe7640ec1529a024094c2bd0677267b",
        "87e9c02bc57028eff67e459627b775a06a6d181268432c68924edf2085306526",
        "eb053516b2208b1e7077202405ae5ee288db89509add7aa7338db620e3bc24f1",
    ];
    for (f, hash) in fixtures().into_iter().zip(expected) {
        assert_eq!(f.fingerprint(), hash);
        assert_eq!(f.reference().unwrap().len(), f.rows);
    }
}

#[test]
fn identity_changes_when_shape_weight_or_input_changes() {
    let f = known(2, 64);
    let hash = f.fingerprint();
    let mut changed = f.clone();
    changed.weights[2] ^= 1;
    assert_ne!(hash, changed.fingerprint());
    changed = f.clone();
    changed.input[0] += 1.0;
    assert_ne!(hash, changed.fingerprint());
    changed = f;
    changed.rows = 1;
    changed.cols = 128;
    assert_ne!(hash, changed.fingerprint());
}

#[test]
fn preflight_is_machine_readable_and_never_gpu_pass() {
    let report = preflight().unwrap();
    assert_eq!(report["CUDA_GPU_TEST"], "NOT_RUN");
    assert_eq!(report["source"]["kernel"], KERNEL);
    assert_eq!(report["source"]["kernel_source_sha256"], KERNEL_SHA256);
    assert_eq!(report["fixtures"].as_array().unwrap().len(), 4);
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_binary_reports_not_run_and_disabled_cuda_fails_explicitly() {
    use std::process::Command;
    let binary = env!("CARGO_BIN_EXE_mer2-cuda-q4-s1");
    let cpu = Command::new(binary).arg("--cpu").output().unwrap();
    assert!(cpu.status.success());
    let report: serde_json::Value = serde_json::from_slice(&cpu.stdout).unwrap();
    assert_eq!(report["CUDA_GPU_TEST"], "NOT_RUN");
    // This binary was compiled WITHOUT CUDA. This tests refusal, not GPU access.
    let refused = Command::new(binary).arg("--cuda").output().unwrap();
    assert!(!refused.status.success());
    let last = String::from_utf8(refused.stdout).unwrap();
    let report: serde_json::Value = serde_json::from_str(last.lines().last().unwrap()).unwrap();
    assert_eq!(report["CUDA_GPU_TEST"], "FAIL");
    assert_eq!(report["error"], "CUDA feature not compiled; no fallback");
    assert!(!Command::new(binary)
        .arg("--unknown")
        .output()
        .unwrap()
        .status
        .success());
}
