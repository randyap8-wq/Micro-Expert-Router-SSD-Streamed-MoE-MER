# RTPS-P1 R1 independent source and CPU review handoff

Disposition: original P1 CPU oracle verified; independent native adapter authored,
default-off, uncompiled and unexecuted. Review readiness is not a GPU, merge or
production PASS. Source history and exact hashes are in
[SOURCE-IDENTITIES.md](SOURCE-IDENTITIES.md); the final file inventory is
[SOURCE-PAYLOAD.sha256](SOURCE-PAYLOAD.sha256).

## Changes and frozen boundary

The R1 delta is exactly ten paths: Cargo.toml, src/lib.rs, new src/native.rs,
new src/native_tests.rs, README.md, NEXT-GATES.md, this handoff,
SOURCE-IDENTITIES.md, PATH-ALLOWLIST.txt and SOURCE-PAYLOAD.sha256.
Seven original source files are byte-identical: fixture.rs, reference.rs,
expected.rs, contract.rs, tests.rs, main.rs and report.rs. The old receipt-only
contract remains a receipt checker; it is not used to authorize native calls.

No W3 import or edit, P6B cuda.rs/mod.rs/identity/checkpoint/model change, MER
main change, vendor change, threshold change, toolchain install or root lock
change occurred. The 258-file W3 source tree remains frozen. The local exact
incremental patch is retained as
`outputs/MER2-RTPS-P1-R1/evidence/RTPS-P1-R1.patch`; PR #380's diff provides the
published review surface. The local Stage A lock/target/evidence are not PR files.

## Stage A observed results

The original 15-file source was verified against live head
`3a29f3cce5ea74ae0b1099c4081aa8d42618f7b6` and tree
`5a057f6f2298315573472899a6991fd074469fa3` before any test or edit.
Installed Mac toolchain: Homebrew Rust/Cargo 1.97.1, aarch64-apple-darwin.
Exactly these commands ran once each, with the absolute installed Cargo binary
`/opt/homebrew/Cellar/rust/1.97.1/bin/cargo`:

```text
cargo check --offline --manifest-path rust-engine/experiments/mer2_cuda_rtps_p1/Cargo.toml --all-targets --no-default-features
cargo test --offline --manifest-path rust-engine/experiments/mer2_cuda_rtps_p1/Cargo.toml --lib --no-default-features
```

Each command had a 120-second wall deadline and 1,048,576-byte combined log cap.
The wrapper would kill the command process group on a limit and stop without
retry. An explicitly constructed environment set CARGO_NET_OFFLINE=true,
CARGO_BUILD_JOBS=1, RUST_TEST_THREADS=1, CARGO_INCREMENTAL=0, exact RUSTC,
exact Apple Clang linker and local CARGO_HOME/TMPDIR/CARGO_TARGET_DIR beneath
`outputs/MER2-RTPS-P1-R1/`. No inherited Cargo config or external dependency
was admitted. Full commands/environment/tool hashes are sealed in the evidence.

Observed check: exit 0, about 0.399 seconds, 185 log bytes.
Observed test: exit 0, about 1.260 seconds, 2,229 log bytes.
First CPU failure: NONE. 24 passed, 0 failed, 0 ignored, 0 filtered out.
No expectations or fixture bytes were changed. The test assertion compares all
5,120 stage elements; it does not produce a separate 5,120-line actual-value dump.
No second Cargo check/test or CPU-reference CLI run was performed.

| Original test | Observed result |
| --- | --- |
| independent_scalar_reference_matches_all_predeclared_bits | PASS |
| low_high_nibbles_are_sixteen_columns_apart_including_final_block | PASS |
| exact_rectangular_bytes_have_no_uth_header | PASS |
| rejects_wrong_shape_transpose_stride_and_projection_offset | PASS |
| rejects_tail_truncation_and_oob_before_indexing | PASS |
| rejects_wrong_finite_zero_negative_nan_and_infinite_scales | PASS |
| independent_half_decoder_rejects_nonfinite_and_preserves_signed_zero | PASS |
| rejects_swapped_projections_nibble_orientation_and_last_byte_drift | PASS |
| rejects_activation_length_bits_nan_and_inf | PASS |
| arithmetic_oracle_itself_rejects_nan_scale_and_input | PASS |
| expected_oracle_rejects_unknown_stage_index_and_nonfinite_readback | PASS |
| one_ulp_difference_is_reported_and_never_tolerated | PASS |
| default_native_endpoint_is_terminal_stop_without_cpu_fallback | PASS |
| rejects_missing_driver_and_resource_without_retry | PASS |
| rejects_invalid_owner_and_owner_change_before_enqueue_or_fence | PASS |
| rejects_second_invocation_even_without_submitted_work | PASS |
| rejects_every_out_of_order_stage_and_missing_fence | PASS |
| enqueue_failure_at_every_stage_preserves_first_failure_through_cleanup | PASS |
| fence_failure_at_every_stage_prevents_later_work | PASS |
| readback_failure_never_becomes_completed_output | PASS |
| ambiguous_or_incomplete_cleanup_is_terminal_and_not_retried | PASS |
| synthetic_receipt_order_completes_without_claiming_context_drop | PASS |
| secondary_evidence_is_bounded_and_overflow_is_explicit | PASS |
| source_manifest_is_retained_in_prospective_report | PASS |

The last tests concern synthetic receipts, not real driver failure/cleanup.
The new four native-source declarations check symbol selection, launch geometry,
buffer coverage and bounded first-error bookkeeping. They are NOT COMPILED OR
EXECUTED. R1 Rust syntax parsing is not Rust type checking or Linux linking.

## Native source, tensor ABI and stage coverage

This is a new independent diagnostic using cudarc 0.19.7 public driver result
APIs and unchanged vendored candle-kernels 0.10.2. cudaforge 0.1.5 is the reviewed
build source. No P6B Native/Lease/permit code was transplanted; no include! or
#[path] bypass and no model-shaped initializer or forged UTH1 header is used.

One fixed expert/request, D=2048, DFF=768. Gate and Up are output-row-major
[768,2048], Down [2048,768]. Each raw Q4_0 block is 18 bytes for 32 weights:
little-endian binary16 scale followed by 16 bytes, low nibble for column j,
high nibble for j+16, decoded as (nibble-8)*scale. Gate/Up row stride 1152,
Down stride 432. Projection offsets are 0, 884736, 1769472; each span is
884736 bytes and total raw upload is 2654208 bytes. Input is exactly 2048 f32
values, +1/32 for each block's first 16 columns and -1/32 for its last 16.

The adapter allocates exactly eight application device buffers, total
2,682,904 bytes, excluding driver/module overhead: raw Q4 2,654,208; input
8,192; metadata 24; Gate/Up/SiLU/Product 3,072 each; Down 8,192. Output
allocations are separate, NaN-poisoned before launch, and never populated from
CPU reference or expected arrays. CPU readback storage is 5,120 u32 values
(20,480 bytes). Fixture storage is fixed. Three PTX CString copies are each
capped at 16 MiB; report output is capped at 4 MiB. Driver-internal overhead,
blocking calls, panic-hook output and total process resources require the
separate host gate's external limits.

| Stage | Exact function and argument ABI | Grid / block | Output |
| --- | --- | --- | --- |
| Gate | q4 symbol: device pointer, f32 input pointer, f32 output pointer, i32 ncols=2048, i32 nrows=768 | (768,1,1) / (32,1,1) | 768 f32 |
| Up | same q4 symbol, second matrix offset, same input, distinct output | (768,1,1) / (32,1,1) | 768 f32 |
| SiLU | usilu_f32: u64 numel=768, u64 num_dims=1, device metadata pointer, Gate pointer, SiLU pointer | (3,1,1) / (256,1,1) | 768 f32 |
| Product | bmul_f32: u64 numel=768, u64 num_dims=1, device metadata pointer, SiLU pointer, Up pointer, Product pointer | (3,1,1) / (256,1,1) | 768 f32 |
| Down | q4 symbol: third matrix offset, Product pointer, Down pointer, i32 ncols=768, i32 nrows=2048 | (2048,1,1) / (32,1,1) | 2048 f32 |

All device pointers and CUDA size_t are 64-bit; module compilation is restricted
to little-endian Linux x86_64. Metadata is a real device [u64;3] = [768,1,1],
covering dims and both strides. This avoids relying on vendor null-pointer
arithmetic. All launch shared-memory sizes are zero. The q4 symbol's full name
is `dequantize_mul_mat_vec_q4_0_cuda`.

Pinned quantized.cu lines 18–26 force f32 dfloat and DMMV_X=32; lines 291–298
define the 18-byte block; 655–673 decode low/high nibbles; 1160–1228 implement
64-column strides, one warp's row reduction and the exact exported symbol.
Both column counts are divisible by 64. unary.cu lines 6–29 define the five
arguments, 59–61 implement x/(1+exp(-x)), and line 226 instantiates usilu_f32.
binary_op_macros.cuh lines 3–18 define six arguments/contiguous strides;
binary.cu line 59 instantiates bmul_f32. The source hashes bind these facts.

## Source to PTX and actual readback

candle-kernels exposes QUANTIZED/UNARY/BINARY Module objects with .ptx().
The exact build-embedded strings, bounded and NUL-checked, are the only
load_data inputs. Exact f32 names are resolved from the corresponding retained
modules. No caller PTX path, generated replacement kernel, NVRTC compilation,
runtime download or CPU fallback exists. embedded_ptx() exposes the same
immutable strings for a future externally sealed PTX/binary inventory;
source hashes or strings alone do not authenticate a host/build.

Synchronous uploads and poisoning precede a checked context fence. The owned
nonblocking stream runs Gate → Up → SiLU → Product → Down, with an attempted
flag set before each call, a distinct return flag, and a checked stream fence
before advancing. Five separate synchronous device-to-host copies then retain
the exact output bits; each copied flag is set only after that copy returns.
A final checked context fence is required for readback_complete. Partial
copied vectors can be reported, but cannot become complete/verified readback.
Exact comparisons use the unchanged expected-bit API, never reference output.

The native report records every copied bit, expected bit, finite flag, signed
and absolute f64 delta, all launch geometries/symbols, first/secondary failure,
cleanup and terminal state. Hash-only receipts cannot supply these arrays.
No GPU report exists from this task. A later report's diagnostic_complete
requires complete readback, all bits exact, no first failure and successful
explicit cleanup calls.

SiLU fixture values are positive-saturated: identity substitution would pass
numerical equality. Actual usilu_f32 source/PTX, successful symbol resolution,
and this function's launch/fence evidence are therefore mandatory. The fixture
remains a narrow geometry/stage diagnostic, not nonlinear or P5D qualification.

## Authority, ownership and failure boundary

The public CLI and native_once() remain unconditional preinitialization STOP.
native-diagnostic is optional, default empty; the module is private and has no
public export. All unsafe FFI lives solely in native.rs. ReviewedHostGate has
an Infallible field with no constructor, so there is no valid safe caller for
staged_once in this patch. No caller boolean/hash/env/feature grants access.
An external host-gate implementation requires a later reviewed source change.

Once that future gate exists, the staged source spends a process-wide atomic
invocation latch before any driver lookup; one non-Clone/non-Send/non-Sync owner
retains one primary context, one stream, three modules and eight allocations.
It rejects a preexisting current context. This does not prove absence of other
threads or primary-context users: a fresh isolated single-threaded process and
physical-device binding remain explicit external admission requirements.

The first error is latched; eight secondary records and an overflow flag bound
further evidence. Numeric CUresult avoids driver calls during error formatting.
Any driver error or caught panic makes resource ownership ambiguous. There is
one cleanup attempt: if a context is known/current, attempt one context drain;
on ambiguity retain resources in quarantine even if drain returns success.
After a clean run or solely numerical mismatch, release allocations/modules/
stream in fixed order, detach and balance the single primary retain. Stop on
the first cleanup-call failure and quarantine remaining/uncertain handles;
do not retry, reset the device or construct another owner. A failed retain/bind
may yield quarantine with no complete handle inventory.

Raw handles have no implicit driver destructor. Unwinding panics are caught
where possible and recorded; abort/OOM/driver hang/kill or failed output can
still produce MISSING_TERMINAL. Quarantine is deliberate retention until the
external reaper resolves the process; it is not successful cleanup. Checked
release return codes do not prove final context destruction or leak freedom.
CudaContext::Drop is not used or asserted. No generic supervisor, host launcher
or in-process timeout claim is introduced.

## Review disposition and next gate

Static review covers the allowlist, original frozen bytes, vendor/archive
identity, manifest default-off wiring, raw API signatures, syntax parsing and
source dataflow. Native dependency resolution, type checking, linking, PTX
generation and native tests remain unexecuted. Cargo.lock is only the observed
original CPU resolution and is excluded from publication.

Use [NEXT-GATES.md](NEXT-GATES.md): first independently review source and CPU
evidence; then prepare a fresh one-use offline Linux native compile/link gate
with exact dependency/build-script/host input pins. A separate reviewed
host-permit patch and one-use GPU test authorization must follow. No VM
launcher is supplied, no AWS/SSH/EC2 action occurred, and no successor is granted.

P5D_STRICT_STAGE_STOP=PRESERVED.
CPU_Q2=HOLD_NO_AUTHORITY (119 separate Rust tests).
W3 R01/R02, O1/H14 and CPU-REC remain unchanged.
No actual Qwen, driver execution, GPU kernel, token, TPS or production qualification.

MER2_RTPS_P1_R1=READY_FOR_INDEPENDENT_SOURCE_AND_CPU_REVIEW
