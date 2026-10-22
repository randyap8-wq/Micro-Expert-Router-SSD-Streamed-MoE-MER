use mer2_cuda_q4_s2::{
    compare, dequantize, fixtures, pack_block, preflight, swiglu, Fixture, Matrix, ATOL, RTOL,
};
use serde_json::Value;

fn constant(rows: usize, cols: usize, scale: u16, q: i8) -> Matrix {
    Matrix {
        rows,
        cols,
        packed: pack_block(scale, &[q; 32])
            .unwrap()
            .repeat(rows * cols / 32),
    }
}

fn diagonal(rows: usize, cols: usize, scale: u16) -> Matrix {
    let mut packed = Vec::new();
    for row in 0..rows {
        for b in 0..cols / 32 {
            let mut q = [0; 32];
            if row % cols / 32 == b {
                q[row % 32] = 1;
            }
            packed.extend(pack_block(scale, &q).unwrap());
        }
    }
    Matrix { rows, cols, packed }
}

#[test]
fn exact_q4_packing_and_split_nibbles() {
    let mut levels = [0i8; 32];
    for j in 0..16 {
        levels[j] = j as i8 - 8;
        levels[j + 16] = 7 - j as i8;
    }
    let bytes = pack_block(0x3800, &levels).unwrap();
    assert_eq!(&bytes[..2], &[0, 0x38]);
    assert_eq!(
        &bytes[2..],
        &(0..16u8).map(|j| j | ((15 - j) << 4)).collect::<Vec<_>>()
    );
    let decoded = dequantize(&bytes).unwrap();
    for j in 0..32 {
        assert_eq!(decoded[j], levels[j] as f32 * 0.5);
    }
}

#[test]
fn signed_zero_subnormal_and_finite_half_scales() {
    for (bits, expected) in [(0x3c00, 1.0), (0xbc00, -1.0), (1, 2f32.powi(-24))] {
        assert_eq!(
            dequantize(&pack_block(bits, &[1; 32]).unwrap()).unwrap(),
            [expected; 32]
        );
    }
    assert_eq!(
        dequantize(&pack_block(0x8000, &[1; 32]).unwrap()).unwrap()[0].to_bits(),
        0x80000000
    );
    assert_eq!(
        dequantize(&pack_block(0x3c00, &[0; 32]).unwrap()).unwrap(),
        [0.0; 32]
    );
}

#[test]
fn invalid_quants_blocks_and_scales_are_rejected() {
    for bad in [-9, 8, i8::MIN, i8::MAX] {
        assert!(pack_block(0, &[bad; 32]).is_err());
    }
    for bad in [0x7c00, 0xfc00, 0x7e00] {
        assert!(pack_block(bad, &[0; 32]).is_err());
    }
    assert!(dequantize(&[0; 17]).is_err());
    assert!(dequantize(&[0; 19]).is_err());
}

#[test]
fn independent_matvec_and_row_block_addressing() {
    let mut m = constant(64, 64, 0x3c00, 1);
    // row 0: +sum(1..32) - sum(33..64) = -1024; other rows = 2080.
    m.packed[18..36].copy_from_slice(&pack_block(0xbc00, &[1; 32]).unwrap());
    let x: Vec<f32> = (1..=64).map(|v| v as f32).collect();
    let result = m.reference(&x).unwrap();
    assert_eq!(result[0], -1024.0);
    assert_eq!(&result[1..], &[2080.0; 63]);
}

#[test]
fn elementwise_known_values_and_signs() {
    let actual = swiglu(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[3.0, -2.0, 9.0, 2.0, -3.0]).unwrap();
    let expected = [-0.71521753, 0.53788286, 0.0, 1.4621172, -5.2847824];
    compare(&expected, &actual).unwrap();
    assert_eq!(swiglu(&[0.0], &[7.0]).unwrap(), [0.0]);
}

#[test]
fn activation_shape_nonfinite_and_overflow_errors() {
    assert!(swiglu(&[], &[]).is_err());
    assert!(swiglu(&[1.0], &[]).is_err());
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(swiglu(&[v], &[1.0]).is_err());
        assert!(swiglu(&[1.0], &[v]).is_err());
    }
    assert!(swiglu(&[f32::MAX], &[2.0]).is_err());
    assert_eq!(swiglu(&[-1000.0], &[1.0]).unwrap()[0], 0.0);
}

#[test]
fn complete_expert_matches_independent_diagonal_reference() {
    let input: Vec<f32> = (0..64).map(|i| (i % 5) as f32 - 2.0).collect();
    let f = Fixture {
        name: "diagonal",
        gate: diagonal(64, 64, 0x3c00),
        up: diagonal(64, 64, 0x4000),
        down: diagonal(64, 64, 0x3c00),
        input,
    };
    let expected: Vec<f32> = f
        .input
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            (x / (1.0 + (-x).exp()) * (2.0 * x)) as f32
        })
        .collect();
    compare(&expected, &f.reference().unwrap()).unwrap();
}

#[test]
fn rectangular_expert_geometry_has_independent_reference() {
    for (dm, df) in [(128, 64), (64, 128)] {
        let f = Fixture {
            name: "rectangular",
            gate: diagonal(df, dm, 0x3c00),
            up: diagonal(df, dm, 0x3c00),
            down: diagonal(dm, df, 0x3c00),
            input: (0..dm).map(|i| (i % 7) as f32 / 8.0).collect(),
        };
        let expected: Vec<f32> = (0..dm)
            .map(|i| {
                let x = f64::from(f.input[(i % df) % dm]);
                (x * x / (1.0 + (-x).exp())) as f32
            })
            .collect();
        compare(&expected, &f.reference().unwrap()).unwrap();
    }
}

#[test]
fn zero_expert_has_exact_zero_output() {
    let f = fixtures().remove(0);
    assert_eq!(f.reference().unwrap(), vec![0.0; 64]);
}

#[test]
fn dimensions_tail_and_integer_overflow_rejected_before_arithmetic() {
    for bad in [0, 1, 32, 63, 65, 96, 129, usize::MAX] {
        let mut f = fixtures().remove(0);
        f.gate.rows = bad;
        assert!(f.cuda_dimensions().is_err());
        let mut f = fixtures().remove(0);
        f.down.cols = bad;
        assert!(f.cuda_dimensions().is_err());
    }
}

#[test]
fn all_three_shapes_and_packed_lengths_checked() {
    let f = fixtures().remove(0);
    let mut changed = f.clone();
    changed.up = constant(128, 64, 0, 0);
    assert!(changed.validate().is_err());
    changed = f.clone();
    changed.down = constant(64, 128, 0, 0);
    assert!(changed.validate().is_err());
    changed = f.clone();
    changed.down = constant(128, 64, 0, 0);
    assert!(changed.validate().is_err());
    changed = f.clone();
    changed.input.pop();
    assert!(changed.validate().is_err());
    for index in 0..3 {
        changed = f.clone();
        let m = match index {
            0 => &mut changed.gate,
            1 => &mut changed.up,
            _ => &mut changed.down,
        };
        m.packed.pop();
        assert!(changed.validate().is_err());
    }
}

#[test]
fn nonfinite_inputs_and_every_matrix_scale_are_rejected() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut f = fixtures().remove(0);
        f.input[0] = bad;
        assert!(f.reference().is_err());
    }
    for index in 0..3 {
        let mut f = fixtures().remove(0);
        let m = match index {
            0 => &mut f.gate,
            1 => &mut f.up,
            _ => &mut f.down,
        };
        m.packed[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
        assert!(f.validate().is_err());
    }
}

#[test]
fn projection_and_expert_overflow_fail_closed() {
    assert!(constant(64, 64, 0x3c00, 7)
        .reference(&[f32::MAX; 64])
        .is_err());
    let mut f = fixtures().remove(1);
    f.input.fill(f32::MAX);
    assert!(f.validate().is_err());
    assert!(f.reference().is_err());
}

#[test]
fn fixture_fingerprints_match_independent_byte_serialization() {
    let contract: Value = serde_json::from_str(include_str!("../fixture-contract.json")).unwrap();
    assert_eq!(contract["atol"], ATOL);
    assert_eq!(contract["rtol"], RTOL);
    let frozen = contract["fixtures"].as_array().unwrap();
    let generated = fixtures();
    assert_eq!(frozen.len(), generated.len());
    for (f, expected) in generated.iter().zip(frozen) {
        assert_eq!(expected["name"], f.name);
        assert_eq!(
            f.fingerprint(),
            expected["fixture_sha256"].as_str().unwrap()
        );
        assert_eq!(expected["d_model"], f.gate.cols);
        assert_eq!(expected["d_ff"], f.gate.rows);
        f.cuda_dimensions().unwrap();
    }
}

#[test]
fn fingerprints_cover_all_matrices_geometry_and_input_bits() {
    let f = fixtures().remove(3);
    let original = f.fingerprint();
    for index in 0..3 {
        let mut changed = f.clone();
        let m = match index {
            0 => &mut changed.gate,
            1 => &mut changed.up,
            _ => &mut changed.down,
        };
        m.packed[2] ^= 1;
        assert_ne!(changed.fingerprint(), original);
    }
    let mut changed = f.clone();
    changed.input[0] = -0.0;
    assert_ne!(changed.fingerprint(), original);
    changed = f;
    changed.down.rows = 64;
    assert_ne!(changed.fingerprint(), original);
}

#[test]
fn references_and_preflight_are_deterministic() {
    for (a, b) in fixtures().iter().zip(fixtures()) {
        assert_eq!(a.fingerprint(), b.fingerprint());
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(a.reference().unwrap()), bits(b.reference().unwrap()));
    }
    assert_eq!(preflight().unwrap(), preflight().unwrap());
}

#[test]
fn logical_copy_trace_has_no_intermediate_readback() {
    for f in fixtures() {
        let t = f.trace().unwrap();
        let dm = f.gate.cols;
        let df = f.gate.rows;
        let weights = 3 * dm * df / 32 * 18;
        assert_eq!(t["h2d"]["total"], weights + dm * 8 + df * 16 + 24);
        assert_eq!(t["device_temporaries"]["total"], df * 16);
        assert_eq!(t["d2h"]["final_y_f32"], dm * 4);
        assert_eq!(t["d2h"]["intermediates"], 0);
        assert_eq!(t["launches"].as_array().unwrap().len(), 5);
        assert_eq!(t["timing"], "NOT_MEASURED");
    }
}

#[test]
fn fixed_acceptance_rejects_bad_shape_nonfinite_and_excess_error() {
    assert_eq!((ATOL, RTOL), (1e-4, 1e-5));
    assert!(compare(&[], &[]).is_err());
    assert!(compare(&[0.0], &[0.0, 0.0]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(compare(&[0.0], &[bad]).is_err());
        assert!(compare(&[bad], &[0.0]).is_err());
    }
    assert!(compare(&[0.0], &[0.00005]).is_ok());
    assert!(compare(&[0.0], &[0.0002]).is_err());
    assert!(compare(&[1000.0], &[1000.005]).is_ok());
    assert!(compare(&[1000.0], &[1000.02]).is_err());
}

#[test]
fn source_preflight_pins_vendor_entries_and_is_unqualified() {
    let p = preflight().unwrap();
    assert_eq!(p["CUDA_GPU_TEST"], "NOT_RUN");
    assert_eq!(p["CUDA_S2_COMPILATION"], "NOT_RUN");
    assert_eq!(p["source"]["entries"]["UNARY"], "usilu_f32");
    assert_eq!(p["source"]["entries"]["BINARY"], "bmul_f32");
    assert_eq!(p["fixtures"].as_array().unwrap().len(), 4);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_cli_emits_preflight_and_disabled_cuda_refuses_without_fallback() {
    use std::process::Command;
    let exe = env!("CARGO_BIN_EXE_mer2-cuda-q4-s2");
    let cpu = Command::new(exe).arg("--cpu").output().unwrap();
    assert!(cpu.status.success());
    let p: Value = serde_json::from_slice(&cpu.stdout).unwrap();
    assert_eq!(p, {
        let mut p = preflight().unwrap();
        p["phase"] = serde_json::json!("preflight");
        p
    });
    // Retained in the single test attempt log; no later executable run needed.
    println!(
        "S2_PREFLIGHT_JSON={}",
        String::from_utf8(cpu.stdout).unwrap().trim()
    );
    let refused = Command::new(exe).arg("--cuda").output().unwrap();
    assert!(!refused.status.success());
    let stdout = String::from_utf8(refused.stdout).unwrap();
    let p: Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
    assert_eq!(p["error"], "CUDA feature not compiled; no fallback");
    assert_eq!(p["gpu_test_invocations"], 0);
    assert_eq!(p["CUDA_GPU_TEST"], "NOT_RUN");
    assert!(p.get("results").is_none());
    assert!(!Command::new(exe)
        .arg("--unknown")
        .output()
        .unwrap()
        .status
        .success());
}
