use mer2_cuda_q4_g2::{
    contract, expert_digest, input_digest, preflight, prepare, verify_output, MIXED,
};
use mer2_cuda_q4_s2::{dequantize, fixtures, pack_block, Fixture, Matrix, ATOL, RTOL};
use serde_json::{json, Value};

#[test]
fn frozen_fixture_all_256_rust_reference_bits_and_discriminating_witness() {
    prepare().unwrap();
    let c = contract().unwrap();
    let mut f = fixtures().remove(3);
    assert_eq!(f.fingerprint(), MIXED);
    assert_eq!(expert_digest(&f).unwrap(), c["expert_sha256"]);
    let mut refs = vec![];
    for seq in 0..2 {
        if seq == 1 {
            for x in &mut f.input {
                *x = -*x;
            }
        }
        assert_eq!(
            input_digest(&f.input).unwrap(),
            c["requests"][seq]["input_sha256"]
        );
        let reference = f.reference().unwrap();
        assert_eq!(reference.len(), 128);
        let bits: Vec<u32> = reference.iter().map(|x| x.to_bits()).collect();
        assert_eq!(json!(bits), c["requests"][seq]["reference_f32_bits"]);
        let stored: Vec<f32> =
            serde_json::from_value(c["requests"][seq]["reference"].clone()).unwrap();
        assert_eq!(bits, stored.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        refs.push(reference);
    }
    let k = c["witness"]["index"].as_u64().unwrap() as usize;
    let (a, b) = (f64::from(refs[0][k]), f64::from(refs[1][k]));
    assert!((a - b).abs() > 2.0 * ATOL + RTOL * (a.abs() + b.abs()));
    let a_bits: Vec<u32> = refs[0].iter().map(|x| x.to_bits()).collect();
    assert!(verify_output(&refs[1], &refs[0], &a_bits).is_err());
    println!("FROZEN_G2_FIXTURE={c}");
}

#[test]
fn independent_identity_serialization_and_signed_zero() {
    let f = fixtures().remove(3);
    let mut raw = b"MER2-CUDA-G2-expert-v1\0".to_vec();
    for m in [&f.gate, &f.up, &f.down] {
        raw.extend(128u32.to_le_bytes());
        raw.extend(128u32.to_le_bytes());
        raw.extend(&m.packed);
    }
    let original = expert_digest(&f).unwrap();
    assert_eq!(original, mer2_cuda_q4_s2::sha256(&raw));
    let mut b = f.clone();
    b.input.iter_mut().for_each(|x| *x = -*x);
    assert_eq!(expert_digest(&b).unwrap(), original);
    assert_ne!(
        input_digest(&b.input).unwrap(),
        input_digest(&f.input).unwrap()
    );
    let mut zeros = 0;
    for (&a, &b) in f.input.iter().zip(&b.input) {
        assert_eq!(a.to_bits() ^ 0x80000000, b.to_bits());
        if a == 0.0 {
            zeros += 1;
            assert_eq!(b.to_bits(), 0x80000000)
        }
    }
    assert!(zeros > 0);
    let mut raw = b"MER2-CUDA-G2-input-v1\0".to_vec();
    raw.extend(128u32.to_le_bytes());
    for v in &b.input {
        raw.extend(v.to_le_bytes())
    }
    assert_eq!(
        input_digest(&b.input).unwrap(),
        mer2_cuda_q4_s2::sha256(&raw)
    );
    for matrix in 0..3 {
        let mut changed = f.clone();
        match matrix {
            0 => changed.gate.packed[2] ^= 1,
            1 => changed.up.packed[2] ^= 1,
            _ => changed.down.packed[2] ^= 1,
        };
        assert_ne!(expert_digest(&changed).unwrap(), original);
    }
}

#[test]
fn q4_scale_f16_split_nibble_and_known_projection() {
    let mut q = [0i8; 32];
    for j in 0..16 {
        q[j] = j as i8 - 8;
        q[j + 16] = 7 - j as i8;
    }
    let bytes = pack_block(0x3800, &q).unwrap();
    assert_eq!(&bytes[..2], &[0, 0x38]);
    for j in 0..16 {
        assert_eq!(bytes[2 + j], j as u8 | ((15 - j as u8) << 4))
    }
    let decoded = dequantize(&bytes).unwrap();
    for j in 0..32 {
        assert_eq!(decoded[j], q[j] as f32 * 0.5)
    }
    for (s, v) in [(1, 2f32.powi(-24)), (0x3c00, 1.0), (0xbc00, -1.0)] {
        assert_eq!(
            dequantize(&pack_block(s, &[1; 32]).unwrap()).unwrap(),
            [v; 32]
        );
    }
    assert_eq!(
        dequantize(&pack_block(0x8000, &[1; 32]).unwrap()).unwrap()[0].to_bits(),
        0x80000000
    );
    for s in [0x7c00, 0xfc00, 0x7e00] {
        assert!(pack_block(s, &[1; 32]).is_err());
    }
    for v in [-9, 8] {
        assert!(pack_block(0, &[v; 32]).is_err());
    }
    assert!(dequantize(&[0; 17]).is_err());
    let m = Matrix {
        rows: 128,
        cols: 128,
        packed: pack_block(0x3c00, &[1; 32]).unwrap().repeat(512),
    };
    assert_eq!(m.reference(&[0.5; 128]).unwrap(), vec![64.0; 128]);
}

#[test]
fn independent_diagonal_swiglu_and_invalid_geometry() {
    let diagonal = |scale| {
        let mut packed = vec![];
        for row in 0..128 {
            for block in 0..4 {
                let mut levels = [0; 32];
                if row / 32 == block {
                    levels[row % 32] = 1
                }
                packed.extend(pack_block(scale, &levels).unwrap());
            }
        }
        Matrix {
            rows: 128,
            cols: 128,
            packed,
        }
    };
    let f = Fixture {
        name: "diagonal",
        gate: diagonal(0x3c00),
        up: diagonal(0x4000),
        down: diagonal(0x3c00),
        input: (0..128).map(|i| (i % 5) as f32 - 2.0).collect(),
    };
    let expected: Vec<f32> = f
        .input
        .iter()
        .map(|&x| {
            let v = f64::from(x);
            (v / (1.0 + (-v).exp()) * (2.0 * v)) as f32
        })
        .collect();
    mer2_cuda_q4_s2::compare(&expected, &f.reference().unwrap()).unwrap();
    for bad in [0, 64, 129, usize::MAX] {
        let mut f = fixtures().remove(3);
        f.gate.rows = bad;
        assert!(expert_digest(&f).is_err());
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut f = fixtures().remove(3);
        f.input[0] = bad;
        assert!(input_digest(&f.input).is_err());
        assert!(f.reference().is_err());
    }
    let mut f = fixtures().remove(3);
    f.down.packed.pop();
    assert!(expert_digest(&f).is_err());
    assert!(input_digest(&[]).is_err());
}

#[test]
fn output_shape_finiteness_bits_and_tolerance_are_independently_checked() {
    assert_eq!((ATOL, RTOL), (1e-4, 1e-5));
    let reference = fixtures().remove(3).reference().unwrap();
    let correct: Vec<u32> = reference.iter().map(|v| v.to_bits()).collect();
    verify_output(&reference, &reference, &correct).unwrap();
    let mut badbits = correct.clone();
    badbits[0] ^= 1;
    assert!(verify_output(&reference, &reference, &badbits).is_err());
    for bad in [f32::NAN, f32::INFINITY, 42.0] {
        let mut actual = reference.clone();
        actual[0] = bad;
        let bits = actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert!(verify_output(&reference, &actual, &bits).is_err());
    }
    assert!(verify_output(&reference, &reference[..127], &correct[..127]).is_err());
}

#[test]
fn expected_schema_roundtrip_has_null_measurements_and_exact_source_identities() {
    let p = preflight().unwrap();
    let copy: Value = serde_json::from_str(&p.to_string()).unwrap();
    assert_eq!(p, copy);
    assert_eq!(p["counters"]["kind"], "EXPECTED");
    assert!(p["counters"]["submitted"].is_null());
    assert!(p["counters"]["completed"].is_null());
    assert_eq!(p["timing"]["status"], "NOT_MEASURED");
    assert!(p["timing"]["batches"].is_null());
    assert_eq!(p["vram"]["status"], "NOT_MEASURED");
    assert!(p["vram"]["free_bytes"].is_null());
    assert_eq!(p["source"]["base"], mer2_cuda_q4_g2::BASE);
    assert_eq!(
        p["source"]["pinned_source_sha256"]
            .as_object()
            .unwrap()
            .len(),
        45
    );
    assert_eq!(
        p["source"]["g2_source_sha256"].as_object().unwrap().len(),
        10
    );
    assert_eq!(p["fixture"]["expected"]["h2d_bytes"], 33816);
    assert_eq!(p["fixture"]["expected"]["d2h_bytes"], 1024);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn single_cpu_cli_preflight_and_cuda_disabled_refusal() {
    use std::process::Command;
    let exe = env!("CARGO_BIN_EXE_mer2-cuda-q4-g2");
    let cpu = Command::new(exe).arg("--cpu").output().unwrap();
    assert!(cpu.status.success());
    let p: Value = serde_json::from_slice(&cpu.stdout).unwrap();
    assert_eq!(p["GPU_INVOCATIONS"], 0);
    assert_eq!(p["CUDA_GPU_TEST"], "NOT_RUN");
    println!(
        "G2_CPU_PREFLIGHT_JSON={}",
        String::from_utf8(cpu.stdout).unwrap().trim()
    );
    let refusal = Command::new(exe).arg("--cuda").output().unwrap();
    assert!(!refusal.status.success());
    let stdout = String::from_utf8(refusal.stdout).unwrap();
    let p: Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
    assert_eq!(p["error"], "CUDA feature not compiled; no fallback");
    assert_eq!(p["GPU_INVOCATIONS"], 0);
    assert_eq!(p["CUDA_GPU_TEST"], "NOT_RUN");
    assert!(p.get("outputs").is_none());
    assert!(!Command::new(exe)
        .arg("--unknown")
        .output()
        .unwrap()
        .status
        .success());
    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("default = []"));
    assert!(manifest
        .contains("mer2-cuda-q4-s2 = { path = \"../mer2_cuda_q4_s2\", default-features = false }"));
}
