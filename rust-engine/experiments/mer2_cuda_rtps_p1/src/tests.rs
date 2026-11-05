//! Authored negative/independent-oracle declarations. ZERO tests executed in P1.
use crate::{contract::{Cleanup, Operation, OwnerTag, TraceContract, ORDER},
    expected, fixture::{self, Fixture, Layout, Orientation, Projection},
    reference, report, Reject, D, DFF, NATIVE_STOP, PROJECTION_BYTES, Q4_BYTES};

fn tag() -> OwnerTag { OwnerTag { context: 1, request: 1, expert: 0 } }
fn started() -> TraceContract {
    let mut state = TraceContract::new(tag()).unwrap();
    state.begin(tag()).unwrap();
    state
}
fn prefix(state: &mut TraceContract, count: usize) {
    for operation in ORDER.into_iter().take(count) {
        state.enqueue(tag(), operation).unwrap();
        state.fenced(tag(), operation, true).unwrap();
    }
}

#[test]
fn independent_scalar_reference_matches_all_predeclared_bits() {
    let fixture = Fixture::build();
    fixture.validate().unwrap();
    let stages = reference::evaluate(fixture.bytes(), fixture.input()).unwrap();
    expected::check_reference(&stages).unwrap();
    assert_eq!(stages.values.iter().map(Vec::len).sum::<usize>(), 5120);
}

#[test]
fn low_high_nibbles_are_sixteen_columns_apart_including_final_block() {
    let fixture = Fixture::build();
    let gate = fixture::matrix_span(fixture.bytes(), Layout::exact(Projection::Gate)).unwrap();
    assert_eq!(reference::scalar(gate, 768, 2048, 0, 15).unwrap().to_bits(), 0x3f00_0000);
    assert_eq!(reference::scalar(gate, 768, 2048, 0, 16).unwrap().to_bits(), 0xbf00_0000);
    assert_eq!(reference::scalar(gate, 768, 2048, 767, 2047).unwrap().to_bits(), 0xc000_0000);
}

#[test]
fn exact_rectangular_bytes_have_no_uth_header() {
    let fixture = Fixture::build();
    assert_eq!(DFF * (D / 32) * 18, PROJECTION_BYTES);
    assert_eq!(D * (DFF / 32) * 18, PROJECTION_BYTES);
    assert_eq!(fixture.bytes().len(), Q4_BYTES);
    assert_eq!(&fixture.bytes()[..2], &[0x00, 0x38]);
    let mut padded = vec![0u8; 4096];
    padded.extend_from_slice(fixture.bytes());
    assert_eq!(fixture::validate_payload(&padded), Err(Reject::Tail));
}

#[test]
fn rejects_wrong_shape_transpose_stride_and_projection_offset() {
    let mut l = Layout::exact(Projection::Gate);
    l.rows = 2048; l.cols = 768;
    assert_eq!(l.validate(Q4_BYTES), Err(Reject::Shape));
    l = Layout::exact(Projection::Gate); l.cols = 2047;
    assert_eq!(l.validate(Q4_BYTES), Err(Reject::Shape));
    l = Layout::exact(Projection::Gate); l.orientation = Orientation::Transposed;
    assert_eq!(l.validate(Q4_BYTES), Err(Reject::Orientation));
    l = Layout::exact(Projection::Gate); l.row_stride_bytes += 18;
    assert_eq!(l.validate(Q4_BYTES), Err(Reject::Stride));
    l = Layout::exact(Projection::Gate); l.offset_bytes = PROJECTION_BYTES;
    assert_eq!(l.validate(Q4_BYTES), Err(Reject::Orientation));
}

#[test]
fn rejects_tail_truncation_and_oob_before_indexing() {
    let fixture = Fixture::build();
    assert_eq!(fixture::validate_payload(&fixture.bytes()[..Q4_BYTES - 1]), Err(Reject::Tail));
    let mut bytes = fixture.bytes().to_vec(); bytes.push(0);
    assert_eq!(fixture::validate_payload(&bytes), Err(Reject::Tail));
    let matrix = &fixture.bytes()[..PROJECTION_BYTES];
    assert_eq!(reference::scalar(matrix, DFF, D, DFF, 0), Err(Reject::Bounds));
    assert_eq!(reference::scalar(matrix, DFF, D, 0, D), Err(Reject::Bounds));
    assert_eq!(reference::scalar(matrix, usize::MAX, D, 0, 0), Err(Reject::Shape));
}

#[test]
fn rejects_wrong_finite_zero_negative_nan_and_infinite_scales() {
    let fixture = Fixture::build();
    for scale in [0x0000u16, 0x3c00, 0xb800, 0x7c00, 0xfc00, 0x7e00, 0xffff] {
        let mut bytes = fixture.bytes().to_vec();
        bytes[..2].copy_from_slice(&scale.to_le_bytes());
        assert_eq!(fixture::validate_payload(&bytes), Err(Reject::Scale), "{scale:04x}");
    }
}

#[test]
fn independent_half_decoder_rejects_nonfinite_and_preserves_signed_zero() {
    for bits in [0x7c00, 0xfc00, 0x7e00, 0xffff] {
        assert_eq!(reference::half(bits), Err(Reject::Scale));
    }
    assert_eq!(reference::half(0x8000).unwrap().to_bits(), 0x8000_0000);
    assert_eq!(reference::half(0x0001).unwrap().to_bits(), 0x3380_0000);
    assert_eq!(reference::half(0x3800).unwrap().to_bits(), 0x3f00_0000);
}

#[test]
fn rejects_swapped_projections_nibble_orientation_and_last_byte_drift() {
    let fixture = Fixture::build();
    let mut bytes = fixture.bytes().to_vec();
    bytes[..PROJECTION_BYTES].copy_from_slice(&fixture.bytes()[PROJECTION_BYTES..2 * PROJECTION_BYTES]);
    assert_eq!(fixture::validate_payload(&bytes), Err(Reject::Scale));
    let mut bytes = fixture.bytes().to_vec(); bytes[2] = bytes[2].rotate_left(4);
    assert_eq!(fixture::validate_payload(&bytes), Err(Reject::FixtureBytes));
    let mut bytes = fixture.bytes().to_vec(); bytes[Q4_BYTES - 1] ^= 1;
    assert_eq!(fixture::validate_payload(&bytes), Err(Reject::FixtureBytes));
}

#[test]
fn rejects_activation_length_bits_nan_and_inf() {
    let fixture = Fixture::build();
    assert_eq!(fixture::validate_input(&fixture.input()[..D - 1]), Err(Reject::Shape));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = fixture.input().to_vec(); input[D - 1] = value;
        assert_eq!(fixture::validate_input(&input), Err(Reject::Nonfinite));
    }
    let mut input = fixture.input().to_vec(); input[0] = -input[0];
    assert_eq!(fixture::validate_input(&input), Err(Reject::InputBits));
}

#[test]
fn arithmetic_oracle_itself_rejects_nan_scale_and_input() {
    let fixture = Fixture::build();
    let mut bytes = fixture.bytes().to_vec(); bytes[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
    assert!(matches!(reference::evaluate(&bytes, fixture.input()), Err(Reject::Scale)));
    let mut input = fixture.input().to_vec(); input[0] = f32::NAN;
    assert!(matches!(reference::evaluate(fixture.bytes(), &input), Err(Reject::Nonfinite)));
}

#[test]
fn expected_oracle_rejects_unknown_stage_index_and_nonfinite_readback() {
    assert_eq!(expected::bits(5, 0), Err(Reject::StageOrder));
    assert_eq!(expected::bits(0, DFF), Err(Reject::Bounds));
    assert_eq!(expected::difference(0, 0, f32::NAN), Err(Reject::Nonfinite));
    assert_eq!(expected::difference(4, 0, f32::INFINITY), Err(Reject::Nonfinite));
}

#[test]
fn one_ulp_difference_is_reported_and_never_tolerated() {
    let actual = f32::from_bits(expected::GATE[0] + 1);
    let difference = expected::difference(0, 0, actual).unwrap();
    assert!(!difference.exact);
    assert_eq!(difference.expected_bits, 0x4200_0000);
    assert_eq!(difference.actual_bits, 0x4200_0001);
    assert!(difference.signed_delta > 0.0);
    assert_eq!(difference.signed_delta, difference.absolute_delta);
}

#[test]
fn default_native_endpoint_is_terminal_stop_without_cpu_fallback() {
    let result = crate::native_once();
    assert_eq!(result.first_failure, NATIVE_STOP);
    assert_eq!(result.attempted_invocations, 0);
    assert_eq!(result.native_initializations, 0);
    assert_eq!(result.kernels_submitted, 0);
    assert!(result.terminal);
    let mut bytes = Vec::new(); result.write(&mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains("gpu_output=ABSENT\ncpu_fallback=false"));
    assert!(!text.contains("stage=gate index="));
}

#[test]
fn rejects_missing_driver_and_resource_without_retry() {
    for reason in [Reject::MissingDriver, Reject::MissingResource] {
        let mut state = started();
        assert_eq!(state.unavailable(tag(), reason), Err(reason));
        assert_eq!(state.enqueue(tag(), Operation::Upload), Err(reason));
        assert_eq!(state.first_failure(), Some(reason));
    }
}

#[test]
fn rejects_invalid_owner_and_owner_change_before_enqueue_or_fence() {
    for owner in [OwnerTag { context: 0, ..tag() }, OwnerTag { request: 2, ..tag() },
        OwnerTag { expert: 1, ..tag() }] {
        assert!(matches!(TraceContract::new(owner), Err(Reject::Owner)));
    }
    let mut state = started();
    assert_eq!(state.enqueue(OwnerTag { context: 2, ..tag() }, Operation::Upload), Err(Reject::Owner));
    let mut state = started(); state.enqueue(tag(), Operation::Upload).unwrap();
    assert_eq!(state.fenced(OwnerTag { context: 2, ..tag() }, Operation::Upload, true), Err(Reject::Owner));
    assert_eq!(state.pending(), Some(Operation::Upload));
}

#[test]
fn rejects_second_invocation_even_without_submitted_work() {
    let mut state = started();
    assert_eq!(state.begin(tag()), Err(Reject::RepeatedInvocation));
    assert_eq!(state.enqueue(tag(), Operation::Upload), Err(Reject::RepeatedInvocation));
}

#[test]
fn rejects_every_out_of_order_stage_and_missing_fence() {
    for index in 0..ORDER.len() {
        let mut state = started(); prefix(&mut state, index);
        let wrong = ORDER[(index + 1) % ORDER.len()];
        assert_eq!(state.enqueue(tag(), wrong), Err(Reject::StageOrder));
        let mut state = started(); prefix(&mut state, index);
        state.enqueue(tag(), ORDER[index]).unwrap();
        assert_eq!(state.enqueue(tag(), ORDER[index]), Err(Reject::StageOrder));
        assert_eq!(state.pending(), Some(ORDER[index]));
    }
}

#[test]
fn enqueue_failure_at_every_stage_preserves_first_failure_through_cleanup() {
    for index in 0..ORDER.len() {
        let mut state = started(); prefix(&mut state, index);
        state.enqueue(tag(), ORDER[index]).unwrap();
        assert_eq!(state.enqueue_failed(tag()), Err(Reject::Enqueue));
        assert_eq!(state.finish(false), Err(Reject::CleanupAmbiguous));
        assert_eq!(state.first_failure(), Some(Reject::Enqueue));
        assert_eq!(state.secondary_failures(), &[Some(Reject::CleanupAmbiguous)]);
        assert_eq!(state.cleanup(), Cleanup::Quarantined);
        assert!(state.terminal());
    }
}

#[test]
fn fence_failure_at_every_stage_prevents_later_work() {
    for index in 0..ORDER.len() {
        let mut state = started(); prefix(&mut state, index);
        state.enqueue(tag(), ORDER[index]).unwrap();
        assert_eq!(state.fenced(tag(), ORDER[index], false), Err(Reject::Fence));
        assert_eq!(state.enqueue(tag(), Operation::Readback), Err(Reject::Fence));
        assert_eq!(state.pending(), Some(ORDER[index]));
    }
}

#[test]
fn readback_failure_never_becomes_completed_output() {
    let mut state = started(); prefix(&mut state, 6);
    state.enqueue(tag(), Operation::Readback).unwrap();
    assert_eq!(state.readback_failed(tag()), Err(Reject::Readback));
    assert_eq!(state.finish(true), Err(Reject::CleanupAmbiguous));
    assert_eq!(state.first_failure(), Some(Reject::Readback));
    assert_eq!(state.cleanup(), Cleanup::Quarantined);
}

#[test]
fn ambiguous_or_incomplete_cleanup_is_terminal_and_not_retried() {
    let mut state = started(); prefix(&mut state, 7);
    assert_eq!(state.finish(false), Err(Reject::CleanupAmbiguous));
    assert_eq!(state.finish(true), Err(Reject::CleanupRepeated));
    assert_eq!(state.first_failure(), Some(Reject::CleanupAmbiguous));
    assert_eq!(state.begin(tag()), Err(Reject::Terminal));
    let mut state = started(); prefix(&mut state, 3);
    assert_eq!(state.finish(true), Err(Reject::StageOrder));
    assert_eq!(state.cleanup(), Cleanup::Quarantined);
}

#[test]
fn synthetic_receipt_order_completes_without_claiming_context_drop() {
    let mut state = started(); prefix(&mut state, 7);
    state.finish(true).unwrap();
    assert_eq!(state.cleanup(), Cleanup::CheckedDrainAndRelease);
    assert_eq!(state.first_failure(), None);
    assert!(state.terminal());
    // This tests a pure state machine, not a native owner/resource lifecycle.
}

#[test]
fn secondary_evidence_is_bounded_and_overflow_is_explicit() {
    let mut state = started();
    state.unavailable(tag(), Reject::MissingDriver).unwrap_err();
    state.finish(false).unwrap_err();
    for _ in 0..12 { state.finish(false).unwrap_err(); }
    assert_eq!(state.first_failure(), Some(Reject::MissingDriver));
    assert_eq!(state.secondary_failures().len(), 8);
    assert!(state.secondary_overflow());
}

#[test]
fn source_manifest_is_retained_in_prospective_report() {
    assert!(report::SOURCE_MANIFEST.contains("src/reference.rs"));
    assert!(report::SOURCE_MANIFEST.contains("src/expected.rs"));
    let mut output = Vec::new(); crate::native_once().write(&mut output).unwrap();
    assert!(String::from_utf8(output).unwrap().contains(crate::W3_TREE));
}
