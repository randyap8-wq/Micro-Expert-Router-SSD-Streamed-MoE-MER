//! Four SOURCE-ONLY declarations; NOT compiled or executed in Mac R1.
//! Pure ABI/receipt checks only; no CUDA initializer, permit or hardware claim.
use crate::native::{Failures, Fault, LAUNCHES, BUFFER_BYTES, SYMBOLS};

#[test]
fn native_declaration_uses_actual_f32_silu_symbol() {
    assert_eq!(SYMBOLS[1], "usilu_f32");
    assert_eq!(LAUNCHES[2].module, 1);
    assert_eq!(SYMBOLS[2], "bmul_f32");
}

#[test]
fn native_declaration_has_one_warp_per_q4_row() {
    for (stage, rows) in [(0, 768), (1, 768), (4, 2048)] {
        assert_eq!(LAUNCHES[stage].grid, (rows, 1, 1));
        assert_eq!(LAUNCHES[stage].block, (32, 1, 1));
    }
    assert_eq!(LAUNCHES[2].grid.0 * LAUNCHES[2].block.0, 768);
    assert_eq!(LAUNCHES[3].grid.0 * LAUNCHES[3].block.0, 768);
}

#[test]
fn native_declaration_retains_all_five_separate_stage_buffers() {
    assert_eq!(&BUFFER_BYTES[3..], &[3072, 3072, 3072, 3072, 8192]);
    assert_eq!(BUFFER_BYTES.iter().sum::<usize>(), 2_682_904);
    assert_eq!(BUFFER_BYTES[2], 24); // real device metadata, not null+dimension
}

#[test]
fn native_failure_record_latches_first_and_bounds_secondary() {
    let mut failures = Failures::default();
    let first = Fault { at: "synthetic_launch_error", cuda: Some(1) };
    failures.latch(first);
    for _ in 0..10 { failures.latch(Fault { at: "synthetic_cleanup_error", cuda: Some(2) }); }
    assert_eq!(failures.first, Some(first));
    assert_eq!(failures.count, 8);
    assert!(failures.overflow);
}
