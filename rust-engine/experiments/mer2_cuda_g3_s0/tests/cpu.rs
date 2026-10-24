//! Authored under D1, NOT executed. Analytical witnesses are independent of
//! full-token outputs. Lifecycle/replay comparisons below are structural tests,
//! not numerical goldens. D2 must freeze its command/count before running once.
use mer2_cuda_g3_s0::identity::{self, Model, Variant};
use mer2_cuda_g3_s0::reference::*;
use mer2_cuda_g3_s0::{bits, cuda_unavailable, Result, Stop, D, KIND};

fn model() -> Model {
    Model::generate(Variant::Primary).unwrap()
}
fn setup() -> (Model, Request) {
    let m = model();
    let mut s = Session::new(m.clone());
    let r = s.request("A", 3).unwrap();
    (m, r)
}
fn run(r: &mut Request, m: &Model) -> StepOutput {
    let ticket = r.ticket();
    r.step_token(m, ticket).unwrap()
}
fn floats(r: &StageReceipt) -> Vec<f32> {
    r.f32_bits.iter().map(|&b| f32::from_bits(b)).collect()
}
fn stage<'a>(o: &'a StepOutput, l: Option<usize>, name: &str) -> &'a StageReceipt {
    o.stages
        .iter()
        .find(|r| r.layer == l && r.stage == name)
        .unwrap()
}
fn near(actual: f32, expected: f64, atol: f64) {
    assert!(actual.is_finite());
    assert!(
        (f64::from(actual) - expected).abs() <= atol,
        "{actual} vs {expected}"
    );
}
fn assert_rollback(r: &Request, bytes: &[u8], next: u32, len: usize) {
    assert_eq!(r.committed_bytes(), bytes);
    assert_eq!(r.next_id(), next);
    assert_eq!(r.len(), len);
    assert!(r.first_error().is_some());
    for l in 0..2 {
        assert_eq!(r.kv(l).unwrap().len(), len);
    }
}
fn injected(mut change: impl FnMut(&mut StageReceipt)) {
    let (m, mut r) = setup();
    let before = r.committed_bytes();
    let mut touched = false;
    let ticket = r.ticket();
    let error = r
        .step_with_hook(&m, ticket, &mut |receipt| {
            if !touched {
                change(receipt);
                touched = true;
            }
            Ok(())
        })
        .unwrap_err();
    assert!(touched);
    assert_eq!(error.code, "STOP_RECEIPT");
    assert_rollback(&r, &before, 3, 0);
}

#[test]
fn sha256_has_independent_standard_vector() {
    assert_eq!(
        identity::sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
#[test]
fn r1_manifest_has_exact_41_tensors_and_independent_input_commitments() {
    let m = model();
    assert_eq!(m.tensors().len(), 41);
    assert_eq!(
        m.hash(),
        "6927d4063b79e0c45ee711e35c3184edffd70088f97cf2b484cd396e028751a1"
    );
    let packed: usize = m
        .tensors()
        .values()
        .filter(|t| t.dtype == 2)
        .map(|t| t.raw.len())
        .sum();
    assert_eq!(packed, 55296);
    assert_eq!(
        identity::specifications().keys().collect::<Vec<_>>(),
        m.tensors().keys().collect::<Vec<_>>()
    );
}
#[test]
fn tie_fixture_changes_only_l0_router_and_has_distinct_digest() {
    let a = model();
    let b = Model::generate(Variant::TieV1).unwrap();
    assert_eq!(
        b.hash(),
        "b8ef837a457f07156176713fb2d79c89a78fc75e7570abd05dd3766c0c3fbecf"
    );
    assert_ne!(a.hash(), b.hash());
    for (name, t) in a.tensors() {
        if name == "L0.router" {
            assert_eq!(b.tensor(name).unwrap().raw, vec![0; 4 * 64 * 4]);
        } else {
            assert_eq!(t, b.tensor(name).unwrap());
        }
    }
}
#[test]
fn gguf_q4_low_high_nibbles_and_signed_half_scales_are_analytical() {
    for (scale_bits, scale) in [
        (0x3400u16, 0.25f32),
        (0x3800, 0.5),
        (0xb400, -0.25),
        (0xb800, -0.5),
    ] {
        let mut block = scale_bits.to_le_bytes().to_vec();
        block.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let decoded = dequantize(&block).unwrap();
        for j in 0..16 {
            assert_eq!(
                decoded[j].to_bits(),
                (scale * (j as i32 - 8) as f32).to_bits()
            );
            assert_eq!(
                decoded[j + 16].to_bits(),
                (scale * (7 - j as i32) as f32).to_bits()
            );
        }
    }
}
#[test]
fn q4_matvec_column_order_and_row_separation_have_analytical_witness() {
    let mut matrix = Vec::new();
    for row in 0..64 {
        for b in 0..2 {
            matrix.extend(0x3800u16.to_le_bytes());
            // Row r: low halves +0.5, high halves -0.5; second block +1/-1.
            let byte = if b == 0 { 0x79 } else { 0x6a };
            matrix.extend([if row % 2 == 0 { byte } else { 0x88 }; 16]);
        }
    }
    let input: Vec<f32> = (0..64).map(|j| j as f32).collect();
    let out = q4_matvec(&matrix, &input).unwrap();
    // sum(j=0..15)(j-(j+16))*0.5 plus second block at scale 1 = -384.
    for row in 0..64 {
        assert_eq!(out[row], if row % 2 == 0 { -384.0 } else { 0.0 });
    }
}
#[test]
fn scalar_accumulation_order_is_left_to_right() {
    assert_eq!(dot(&[1.0, 1.0, 1.0], &[1e20, 1.0, -1e20]).unwrap(), 0.0);
    let mut weights = vec![0.0; 64 * 64];
    for r in 0..64 {
        weights[r * 64 + r] = 2.0;
    }
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 8.0).collect();
    assert_eq!(
        dense(&weights, 64, &input).unwrap(),
        input.iter().map(|x| 2.0 * x).collect::<Vec<_>>()
    );
}
#[test]
fn rms_norm_zero_constant_and_gamma_have_independent_math() {
    assert_eq!(rms_norm(&[0.0; 64], &[1.0; 64]).unwrap(), vec![0.0; 64]);
    let got = rms_norm(&[2.0; 64], &[1.5; 64]).unwrap();
    let expected = 3.0f64 / (4.0 + f64::from(1e-5f32)).sqrt();
    for x in got {
        near(x, expected, 2e-7);
    }
}
#[test]
fn stable_softmax_and_scalar_router_have_independent_math() {
    assert_eq!(softmax(&[1000.0; 4]).unwrap(), vec![0.25; 4]);
    let r = route(&[0.0, 2.0f32.ln(), 0.0, 0.0]).unwrap();
    assert_eq!(r.ranked, [1, 0, 2, 3]);
    for (i, b) in r.probability_bits.iter().enumerate() {
        near(f32::from_bits(*b), if i == 1 { 0.4 } else { 0.2 }, 1e-7);
    }
    near(f32::from_bits(r.selected_weight_bits[0]), 2.0 / 3.0, 1e-7);
    near(f32::from_bits(r.selected_weight_bits[1]), 1.0 / 3.0, 1e-7);
}
#[test]
fn exact_router_ties_rank_lower_ids_and_renormalize_top_two() {
    let r = route(&[0.0; 4]).unwrap();
    assert_eq!(r.ranked, [0, 1, 2, 3]);
    assert_eq!(r.selected_weight_bits, [0x3f000000; 2]);
    assert!(r.observed);
    assert_eq!(route(&[-1.0, 2.0, 2.0, -1.0]).unwrap().ranked, [1, 2, 0, 3]);
}
#[test]
fn rope_interleaves_each_head_and_preserves_v_input() {
    let mut x = vec![0.0; 64];
    for h in 0..4 {
        x[h * 16] = 2.0;
        x[h * 16 + 1] = 3.0;
    }
    let before = x.clone();
    let out = rope(&x, 1).unwrap();
    assert_eq!(x, before);
    for h in 0..4 {
        near(out[h * 16], 2.0 * 1.0f64.cos() - 3.0 * 1.0f64.sin(), 5e-7);
        near(
            out[h * 16 + 1],
            2.0 * 1.0f64.sin() + 3.0 * 1.0f64.cos(),
            5e-7,
        );
    }
    assert_eq!(rope(&[1.0; 64], 0).unwrap(), vec![1.0; 64]);
}
#[test]
fn causal_attention_includes_provisional_input_and_denies_future() {
    let empty = LayerKv::default();
    let a = attention(&[0.0; 64], &empty, 0, &[0.0; 64], &[7.0; 64]).unwrap();
    assert_eq!(a.probabilities, vec![1.0; 4]);
    assert_eq!(a.context, vec![7.0; 64]);
    assert!(empty.at(0).is_err());
    let cache = empty.appended(0, &[0.0; 64], &[2.0; 64]).unwrap();
    let b = attention(&[0.0; 64], &cache, 1, &[0.0; 64], &[6.0; 64]).unwrap();
    assert_eq!(b.context, vec![4.0; 64]);
    assert_eq!(b.probabilities, vec![0.5; 8]);
    assert!(cache.at(1).is_err());
    assert!(attention(&[0.0; 64], &cache, 0, &[0.0; 64], &[99.0; 64]).is_err());
    assert!(cache.appended(2, &[0.0; 64], &[0.0; 64]).is_err());
}
#[test]
fn attention_head_storage_has_independent_analytical_isolation() {
    let v: Vec<f32> = (0..64).map(|j| (j / 16 + 1) as f32).collect();
    let cache = LayerKv::default().appended(0, &[0.0; 64], &v).unwrap();
    let out = attention(&[0.0; 64], &cache, 1, &[0.0; 64], &[0.0; 64]).unwrap();
    for j in 0..64 {
        assert_eq!(out.context[j], (j / 16 + 1) as f32 / 2.0);
    }
    assert_eq!(cache.at(0).unwrap().1, v);
}
#[test]
fn silu_and_finite_greedy_have_independent_spots_and_tie_rule() {
    let a = silu(&[0.0, 1.0, -1.0]).unwrap();
    assert_eq!(a[0], 0.0);
    near(a[1], 1.0 / (1.0 + (-1.0f64).exp()), 1e-7);
    near(a[2], -1.0 / (1.0 + 1.0f64.exp()), 1e-7);
    assert_eq!(greedy(&[0.0; 32]).unwrap(), 0);
    let mut logits = [-1.0; 32];
    logits[7] = 2.0;
    logits[19] = 2.0;
    assert_eq!(greedy(&logits).unwrap(), 7);
}
#[test]
fn independent_constant_q4_swiglu_chain_has_known_scalar_result() {
    fn constant(scale: u16, byte: u8) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..128 {
            out.extend(scale.to_le_bytes());
            out.extend([byte; 16]);
        }
        out
    }
    let mut x = [0.0f32; 64];
    x[0] = 2.0;
    let gate = q4_matvec(&constant(0x3800, 0x99), &x).unwrap();
    assert_eq!(gate, vec![1.0; 64]);
    let up = q4_matvec(&constant(0x3800, 0xaa), &x).unwrap();
    assert_eq!(up, vec![2.0; 64]);
    let hidden: Vec<f32> = silu(&gate)
        .unwrap()
        .iter()
        .zip(&up)
        .map(|(g, u)| g * u)
        .collect();
    let y = q4_matvec(&constant(0x3400, 0x99), &hidden).unwrap();
    // 64 * 0.25 * (1/(1+exp(-1))) * 2, independently evaluated in f64.
    for v in y {
        near(v, 32.0 / (1.0 + (-1.0f64).exp()), 5e-5);
    }
    let zero = q4_matvec(&constant(0x3400, 0x88), &x).unwrap();
    assert_eq!(
        q4_matvec(&constant(0x3400, 0x99), &silu(&zero).unwrap()).unwrap(),
        vec![0.0; 64]
    );
}
#[test]
fn nonfinite_and_invalid_shapes_fail_closed() {
    assert!(dot(&[f32::INFINITY], &[1.0]).is_err());
    assert!(dot(&[f32::MAX], &[2.0]).is_err());
    assert!(softmax(&[f32::NAN]).is_err());
    assert!(route(&[0.0, 0.0, 0.0, f32::INFINITY]).is_err());
    let mut logits = [0.0; 32];
    logits[31] = f32::NAN;
    assert!(greedy(&logits).is_err());
    assert!(rms_norm(&[f32::MAX; 64], &[1.0; 64]).is_err());
    assert!(rope(&[f32::NAN; 64], 0).is_err());
    assert!(silu(&[-1000.0]).is_err());
    assert!(dense(&[], 1, &[0.0; 64]).is_err());
    assert!(q4_matvec(&[0; 2303], &[0.0; 64]).is_err());
    assert!(dequantize(&[0; 17]).is_err());
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(dequantize(&block).is_err());
    assert!(attention(
        &[0.0; 64],
        &LayerKv::default(),
        0,
        &[0.0; 64],
        &[f32::NAN; 64]
    )
    .is_err());
}
#[test]
fn mutated_tensor_source_or_manifest_identity_is_rejected() {
    let m = model();
    let mut tensors = m.tensors().clone();
    tensors.get_mut("L1.E3.down").unwrap().raw[2] ^= 1;
    assert_eq!(
        Model::from_tensors(Variant::Primary, tensors)
            .unwrap_err()
            .code,
        "STOP_IDENTITY"
    );
    let mut sources = identity::source_files();
    sources.insert("src/reference.rs".into(), "00".repeat(32));
    assert_eq!(
        identity::verify_source(&sources).unwrap_err().code,
        "STOP_SOURCE_IDENTITY"
    );
    let mut tensors = m.tensors().clone();
    tensors.remove("L1.Wq");
    assert!(Model::from_tensors(Variant::Primary, tensors).is_err());
}
#[test]
fn nonfinite_weight_payload_cannot_enter_model() {
    let mut tensors = model().tensors().clone();
    tensors.get_mut("embedding").unwrap().raw[..4]
        .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert_eq!(
        Model::from_tensors(Variant::Primary, tensors)
            .unwrap_err()
            .code,
        "STOP_NONFINITE"
    );
}
#[test]
fn full_tie_fixture_has_analytical_l0_observed_route() {
    let m = Model::generate(Variant::TieV1).unwrap();
    let mut s = Session::new(m.clone());
    let mut r = s.request("TIE_V1", 3).unwrap();
    let out = run(&mut r, &m);
    let route = stage(&out, Some(0), "router_probs").route.as_ref().unwrap();
    assert_eq!(route.ranked, [0, 1, 2, 3]);
    assert_eq!(route.selected_weight_bits, [0x3f000000; 2]);
}
#[test]
fn two_requests_three_tokens_are_disjoint_and_discriminating() {
    let m = model();
    let mut s = Session::new(m.clone());
    let mut a = s.request("A", 3).unwrap();
    let mut b = s.request("B", 17).unwrap();
    assert_ne!(a.ticket().nonce, b.ticket().nonce);
    let mut all = Vec::new();
    let mut routes_differ = false;
    for p in 0..3 {
        let bb = b.committed_bytes();
        let ao = run(&mut a, &m);
        assert_eq!(bb, b.committed_bytes());
        let aa = a.committed_bytes();
        let bo = run(&mut b, &m);
        assert_eq!(aa, a.committed_bytes());
        assert_eq!(ao.ticket.position, p);
        assert_eq!(bo.ticket.position, p);
        println!("G3_A_THREE_TOKEN_TRACE={}", ao.json());
        println!("G3_B_THREE_TOKEN_TRACE={}", bo.json());
        assert_ne!(ao.tokens_sha256, bo.tokens_sha256);
        assert!(b.verify_output(&ao).is_err());
        for l in 0..2 {
            routes_differ |= stage(&ao, Some(l), "router_probs")
                .route
                .as_ref()
                .unwrap()
                .ranked[..2]
                != stage(&bo, Some(l), "router_probs")
                    .route
                    .as_ref()
                    .unwrap()
                    .ranked[..2];
        }
        all.push(ao.logits_bits);
        all.push(bo.logits_bits);
    }
    for i in 0..all.len() {
        for j in 0..i {
            assert_ne!(
                all[i], all[j],
                "STOP_FIXTURE_NONDISCRIMINATING: logits {i}/{j}"
            );
        }
    }
    assert!(
        routes_differ,
        "STOP_FIXTURE_NONDISCRIMINATING: A/B selected routes"
    );
    assert_eq!((a.len(), b.len()), (3, 3));
}
#[test]
fn same_session_sequence_replays_bits_only_on_same_binary_target() {
    let m = model();
    let mut a = Session::new(m.clone());
    let mut b = Session::new(m.clone());
    let mut ra = a.request("REPLAY", 3).unwrap();
    let mut rb = b.request("REPLAY", 3).unwrap();
    for _ in 0..3 {
        assert_eq!(run(&mut ra, &m), run(&mut rb, &m));
    }
    assert_eq!(ra.committed_bytes(), rb.committed_bytes());
}
#[test]
fn fourth_position_is_legal_and_fifth_rolls_back() {
    let (m, mut r) = setup();
    for p in 0..4 {
        assert_eq!(run(&mut r, &m).ticket.position, p);
    }
    let before = r.committed_bytes();
    let next = r.next_id();
    let ticket = r.ticket();
    assert_eq!(r.step_token(&m, ticket).unwrap_err().code, "STOP_INPUT");
    assert_rollback(&r, &before, next, 4);
}
#[test]
fn committed_kv_belongs_to_input_not_sampled_next_token_and_layers_differ() {
    let (m, mut r) = setup();
    let o = run(&mut r, &m);
    assert_eq!(o.ticket.input_id, 3);
    for l in 0..2 {
        let (k, v) = r.kv(l).unwrap().at(0).unwrap();
        assert_eq!(bits(&k), stage(&o, Some(l), "k_rope").f32_bits);
        assert_eq!(bits(&v), stage(&o, Some(l), "v").f32_bits);
        assert!(r.kv(l).unwrap().at(1).is_err());
    }
    assert_ne!(
        stage(&o, Some(0), "k_rope").f32_bits,
        stage(&o, Some(1), "k_rope").f32_bits,
        "STOP_FIXTURE_NONDISCRIMINATING: layers"
    );
    let previous = r.kv(0).unwrap().at(0).unwrap();
    let n = run(&mut r, &m);
    assert_eq!(n.ticket.input_id, o.next_id);
    assert_eq!(n.ticket.position, 1);
    assert_eq!(r.kv(0).unwrap().at(0).unwrap(), previous);
}
#[test]
fn raw_stage_trace_is_complete_bound_and_json_serializable() {
    let (m, mut r) = setup();
    let out = run(&mut r, &m);
    assert_eq!(out.logits_bits.len(), 32);
    assert_eq!(out.stages.first().unwrap().stage, "begin");
    assert_eq!(out.stages.last().unwrap().stage, "before_commit");
    for (i, s) in out.stages.iter().enumerate() {
        assert_eq!(s.kind, KIND);
        assert_eq!(s.ticket, out.ticket);
        assert_eq!(s.sequence, i);
        assert_eq!(s.sha256, s.commitment());
        assert!(floats(s).iter().all(|v| v.is_finite()));
    }
    for l in 0..2 {
        for name in [
            "attention_pre_norm",
            "q",
            "k",
            "v",
            "q_rope",
            "k_rope",
            "provisional_kv",
            "attention_scores",
            "attention_softmax",
            "context",
            "attention_projection",
            "h_att",
            "moe_pre_norm",
            "router_logits",
            "router_probs",
            "topk_weights",
            "agg",
            "h_out",
        ] {
            stage(&out, Some(l), name);
        }
        for name in [
            "expert_gate",
            "expert_up",
            "expert_silu",
            "expert_activated",
            "expert_down",
            "expert_weighted",
        ] {
            assert_eq!(
                out.stages
                    .iter()
                    .filter(|s| s.layer == Some(l) && s.stage == name)
                    .count(),
                2
            );
        }
    }
    assert_eq!(out.stages.len(), 66);
    let json = out.json();
    let mut reparsed: serde_json::Value =
        serde_json::from_str(&json.to_string()).unwrap();
    let mut original = json.clone();
    // The canonical stage payload is f32_bits. JSON min/max are derived f32
    // extrema, and their f64 Value representation can differ by a ULP after
    // stringify/reparse. Verify every extremum via its exact f32 bits while
    // keeping strict JSON equality for all other fields (including f32_bits).
    {
        let original_stages = original["stages"].as_array_mut().unwrap();
        let parsed_stages = reparsed["stages"].as_array_mut().unwrap();
        assert_eq!(original_stages.len(), out.stages.len());
        assert_eq!(parsed_stages.len(), out.stages.len());
        for (i, receipt) in out.stages.iter().enumerate() {
            let extrema = [
                (
                    "min",
                    receipt
                        .f32_bits
                        .iter()
                        .copied()
                        .map(f32::from_bits)
                        .reduce(f32::min),
                ),
                (
                    "max",
                    receipt
                        .f32_bits
                        .iter()
                        .copied()
                        .map(f32::from_bits)
                        .reduce(f32::max),
                ),
            ];
            for (field, expected) in extrema {
                for stage in [&mut original_stages[i], &mut parsed_stages[i]] {
                    let value = stage
                        .as_object_mut()
                        .unwrap()
                        .remove(field)
                        .expect("stage extrema field must exist");
                    match expected {
                        Some(extremum) => {
                            assert!(extremum.is_finite());
                            let number = value.as_f64().expect("finite JSON extrema number");
                            assert!(number.is_finite());
                            assert_eq!(
                                (number as f32).to_bits(),
                                extremum.to_bits(),
                                "stage {i} {field} changed canonical f32 bits"
                            );
                        }
                        None => assert!(value.is_null(), "empty stage {i} {field}"),
                    }
                }
            }
        }
    }
    assert_eq!(reparsed, original);
    r.verify_output(&out).unwrap();
    println!("G3_UNQUALIFIED_CPU_TRACE={json}");
}
#[test]
fn stale_or_foreign_tickets_and_wrong_input_cannot_mutate_kv() {
    for field in 0..10 {
        let (m, mut r) = setup();
        run(&mut r, &m);
        let before = r.committed_bytes();
        let next = r.next_id();
        let mut t = r.ticket();
        match field {
            0 => t.position = 0,
            1 => t.input_id = (next + 1) % 32,
            2 => t.nonce += 1,
            3 => t.attempt += 1,
            4 => t.kv_generation += 1,
            5 => t.kv_sha256 = "00".repeat(32),
            6 => t.source_sha256 = "00".repeat(32),
            7 => t.model_sha256 = "00".repeat(32),
            8 => t.request = "B".into(),
            _ => t.input_id = 32,
        }
        assert_eq!(r.step_token(&m, t).unwrap_err().code, "STOP_TICKET");
        assert_rollback(&r, &before, next, 1);
    }
}
#[test]
fn wrong_model_and_duplicate_request_are_rejected() {
    let (m, mut r) = setup();
    let wrong = Model::generate(Variant::TieV1).unwrap();
    let ticket = r.ticket();
    let before = r.committed_bytes();
    assert_eq!(
        r.step_token(&wrong, ticket).unwrap_err().code,
        "STOP_IDENTITY"
    );
    assert_rollback(&r, &before, 3, 0);
    let mut s = Session::new(m);
    s.request("A", 3).unwrap();
    assert!(s.request("A", 17).is_err());
    assert!(s.request("C", 32).is_err());
}
#[test]
fn malformed_stage_identity_shape_bits_and_digest_are_rejected() {
    for field in 0..8 {
        injected(|r| match field {
            0 => r.kind = "CUDA".into(),
            1 => r.sequence += 1,
            2 => r.layer = Some(1),
            3 => r.stage = "foreign".into(),
            4 => r.dtype = 2,
            5 => r.shape = vec![1],
            6 => r.f32_bits = vec![f32::NAN.to_bits()],
            _ => r.sha256 = "00".repeat(32),
        });
    }
}
#[test]
fn swapped_expert_layer_slot_generation_and_weights_are_rejected() {
    for field in 0..6 {
        let (m, mut r) = setup();
        let before = r.committed_bytes();
        let ticket = r.ticket();
        let mut touched = false;
        let error = r
            .step_with_hook(&m, ticket, &mut |s| {
                if let Some(e) = s.expert.as_mut() {
                    if !touched {
                        match field {
                            0 => e.expert = (e.expert + 1) % 4,
                            1 => e.layer = 1 - e.layer,
                            2 => e.rank = 1 - e.rank,
                            3 => e.slot += 1,
                            4 => e.generation += 1,
                            _ => e.weight_sha256[0] = "00".repeat(32),
                        }
                        s.sha256 = s.commitment();
                        touched = true;
                    }
                }
                Ok(())
            })
            .unwrap_err();
        assert!(touched);
        assert_eq!(error.code, "STOP_RECEIPT");
        assert_rollback(&r, &before, 3, 0);
    }
}
#[test]
fn prediction_substitution_wrong_rank_tie_and_weights_are_rejected() {
    for field in 0..4 {
        let m = Model::generate(Variant::TieV1).unwrap();
        let mut session = Session::new(m.clone());
        let mut r = session.request("TIE_V1_BAD_ROUTE", 3).unwrap();
        let before = r.committed_bytes();
        let ticket = r.ticket();
        let mut touched = false;
        let error = r
            .step_with_hook(&m, ticket, &mut |s| {
                if let Some(route) = s.route.as_mut() {
                    if !touched {
                        match field {
                            0 => route.observed = false,
                            1 => route.ranked.swap(0, 1),
                            2 => route.ranked = [3, 2, 1, 0],
                            _ => route.selected_weight_bits = [1.0f32.to_bits(), 0],
                        }
                        s.sha256 = s.commitment();
                        touched = true;
                    }
                }
                Ok(())
            })
            .unwrap_err();
        assert!(touched);
        assert_eq!(error.code, "STOP_RECEIPT");
        assert_rollback(&r, &before, 3, 0);
    }
}
#[test]
fn every_stage_cancellation_and_error_preserve_both_layers_atomically() {
    let (m, mut baseline) = setup();
    run(&mut baseline, &m);
    let success = run(&mut baseline, &m);
    // Enumerate actual boundaries, also freeze trace count separately above.
    for target in 0..success.stages.len() {
        for code in ["STOP_CANCELLED", "STOP_INJECTED_ERROR"] {
            let mut session = Session::new(m.clone());
            let mut r = session.request("A", 3).unwrap();
            run(&mut r, &m);
            let before = r.committed_bytes();
            let next = r.next_id();
            let ticket = r.ticket();
            let mut reached = false;
            let error = r
                .step_with_hook(&m, ticket, &mut |s| {
                    if s.sequence == target {
                        reached = true;
                        return Err(Stop::new(code, format!("boundary {target}")));
                    }
                    Ok(())
                })
                .unwrap_err();
            assert!(reached);
            assert_eq!(error.code, code);
            assert_rollback(&r, &before, next, 1);
            println!(
                "G3_FAULT_RECEIPT={}",
                serde_json::json!({"scenario":code,"boundary":target,
                "first_error":error.to_string(),"before_kv_sha256":identity::sha256(&before),
                "after_kv_sha256":identity::sha256(&r.committed_bytes()),"next_id":r.next_id(),"committed_length":r.len()})
            );
            let retry = r.ticket();
            assert_eq!(r.step_token(&m, retry).unwrap_err(), error);
            assert_eq!(r.first_error(), Some(&error));
        }
    }
}
#[test]
fn stale_output_and_cross_request_receipts_fail_even_if_rehashed() {
    let m = model();
    let mut s = Session::new(m.clone());
    let mut a = s.request("A", 3).unwrap();
    let mut b = s.request("B", 17).unwrap();
    let old = run(&mut a, &m);
    let newer = run(&mut a, &m);
    assert!(a.verify_output(&old).is_err());
    a.verify_output(&newer).unwrap();
    let mut bad = newer.clone();
    bad.logits_bits[0] ^= 1;
    assert!(a.verify_output(&bad).is_err());
    bad = newer.clone();
    bad.stages.remove(3);
    assert!(a.verify_output(&bad).is_err());
    bad = newer.clone();
    bad.stages.push(bad.stages[2].clone());
    assert!(a.verify_output(&bad).is_err());
    let before = b.committed_bytes();
    let ticket = b.ticket();
    let foreign = old.stages[0].clone();
    let error = b
        .step_with_hook(&m, ticket, &mut |r| {
            *r = foreign.clone();
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error.code, "STOP_RECEIPT");
    assert_rollback(&b, &before, 17, 0);
}
#[test]
fn changing_output_greedy_or_kv_identity_invalidates_commitment() {
    let (m, mut r) = setup();
    let original = run(&mut r, &m);
    for field in 0..4 {
        let mut bad = original.clone();
        match field {
            0 => bad.next_id = (bad.next_id + 1) % 32,
            1 => bad.committed_kv_sha256 = "00".repeat(32),
            2 => bad.tokens_sha256 = "00".repeat(32),
            _ => bad.ticket.kv_generation += 1,
        }
        assert!(r.verify_output(&bad).is_err());
    }
}
#[test]
fn panic_at_final_boundary_is_terminal_and_rolls_back() {
    let (m, mut r) = setup();
    run(&mut r, &m);
    let before = r.committed_bytes();
    let next = r.next_id();
    let ticket = r.ticket();
    let error = r
        .step_with_hook(&m, ticket, &mut |s| {
            if s.stage == "before_commit" {
                panic!("deliberate CPU hook panic");
            }
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error.code, "STOP_PANIC");
    assert_rollback(&r, &before, next, 1);
}
#[test]
fn nonfinite_expert_result_receipt_is_rejected_before_commit() {
    let (m, mut r) = setup();
    let before = r.committed_bytes();
    let ticket = r.ticket();
    let mut reached = false;
    let error = r
        .step_with_hook(&m, ticket, &mut |s| {
            if s.stage == "expert_down" {
                s.f32_bits[0] = f32::NAN.to_bits();
                s.sha256 = s.commitment();
                reached = true;
            }
            Ok(())
        })
        .unwrap_err();
    assert!(reached);
    assert_eq!(error.code, "STOP_RECEIPT");
    assert_rollback(&r, &before, 3, 0);
}
#[test]
fn tensor_domain_shape_dtype_and_request_sequences_are_unambiguous() {
    let m = model();
    let original = m.tensor("L0.router").unwrap();
    let mut changed = original.clone();
    changed.name = "L1.router".into();
    assert_ne!(original.hash().unwrap(), changed.hash().unwrap());
    changed = original.clone();
    changed.shape = vec![64, 4];
    assert_ne!(original.hash().unwrap(), changed.hash().unwrap());
    assert_ne!(
        identity::tokens_hash("A", 1, &[3, 17]),
        identity::tokens_hash("A", 1, &[17, 3])
    );
    assert_ne!(
        identity::tokens_hash("A", 1, &[3]),
        identity::tokens_hash("B", 1, &[3])
    );
    assert_ne!(
        identity::tokens_hash("A", 1, &[3]),
        identity::tokens_hash("A", 2, &[3])
    );
}
#[test]
fn cuda_mode_explicitly_refuses_and_has_no_feature_or_dependencies() {
    assert_eq!(
        cuda_unavailable().unwrap_err().code,
        "STOP_CUDA_UNAVAILABLE"
    );
    let manifest = include_str!("../Cargo.toml");
    assert!(!manifest.contains("cudarc"));
    assert!(!manifest.contains("candle"));
    assert!(!manifest.contains("cuda ="));
    assert!(manifest.contains("default = []"));
    let _type_check_only: fn(
        &mut Request,
        &Model,
        mer2_cuda_g3_s0::reference::Ticket,
    ) -> Result<StepOutput> = Request::step_token;
    assert_eq!(D, 64);
}
