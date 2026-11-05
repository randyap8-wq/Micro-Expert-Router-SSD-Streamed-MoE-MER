# RTPS-P1 source review handoff — 2026-11-04 Costa Rica

**Disposition: safe partial source, native adapter unbound. NO BUILDS / NO RUNS /
NO MERGE.** Review the actual Rust in `src/`, especially the independent scalar
reference, fixed bit patterns, rejection tests and explicitly noninitializing
native endpoint. No claim of a finished CUDA implementation or numeric PASS.

## Exact blocking interface and required future path changes

The common P6B native source is not a reusable standalone library. W3
`rust-engine/experiments/mer2_cuda_p6b/src/mod.rs:13–16` intentionally omits
`cuda.rs`. `Native::execute` in `cuda.rs:86–89,187–225` consumes `VerifiedExpert`
and a model-shaped `Lease`; it returns only Down plus five stage hashes. The
five retained stage vectors are private (`cuda.rs:48,218–225`). `Owner::kernel`
is private (`254–304`). `identity.rs:152–173` requires a full UTH1 expert image.
The selected kernels take raw projection pointers, so the file header is not a
kernel requirement. A synthetic input adapter must make that distinction.

Re-declaring this W3 module, replacing its identity module with lookalike types,
using `#[path]`/`include!` to smuggle it in, copying its unsafe constructors, or
calling its public initializer would cross the authorized boundary. No such
workaround was added. G3's existing standalone adapter is D=DFF=64, two layers;
it cannot serve as the rectangular adapter unchanged.

One concrete **future extraction route**, requiring new source authorization:

| Exact path(s) | Required change, not performed |
| --- | --- |
| `rust-engine/experiments/mer2_cuda_p6b/src/cuda.rs` | Extract the retained native owner/kernel-launch implementation into a reusable component; leave an adapter consuming the existing model-validated type. Do not reconnect W3 `mod.rs` or weaken `VerifiedExpert`. |
| `rust-engine/experiments/mer2_cuda_q4_native/Cargo.toml`, `Cargo.lock`, `src/lib.rs`, `src/owner.rs` (proposed new directory) | Own the extracted implementation once, with a sealed validated raw-Q4 input API, immutable retained inputs, five stage vectors, one initialization/invocation, first-failure evidence and checked cleanup/quarantine. Review every constructor/error/destructor route before use. These files do not exist in this patch. |
| `rust-engine/experiments/mer2_cuda_rtps_p1/Cargo.toml`, new `Cargo.lock`, new `src/native.rs`, `src/lib.rs`, `src/main.rs` | Bind the reviewed component and dependency closure, actual source/build/PTX identity checks, native-only outputs and a separately reviewed host-bound entrypoint. Replace the unconditional native STOP only in a new approved patch. |

No root Cargo.lock, production loader, W3 module declaration, vendor kernel or
G3 source change is proposed as an incidental repair. The table is an explicit
future path delta, not authorization. A different route needs its own review;
this package does not assert that every possible reuse design requires it.

Dependency changes still needed: a new standalone locked closure for exact
`cudarc = 0.19.7` with P6B's std/driver/nvrtc/dynamic-loading/build-system-version
features, unchanged vendored `candle-kernels = 0.10.2`, its `cudaforge = 0.1.5`
locked build dependency and all transitive packages. The existing W3 Cargo.lock
records these versions but is not the new crate's resolved closure; no dependency
solver was run, and no new CUDA manifest or invented lockfile was supplied.
The vendored build.rs compiles more than these three kernels, consults compute
capability and includes `rust-engine/cuda/glibc_cuda_compat.h`; exact build inputs,
host compiler/toolkit policy and any environment override need separate review.

## Source evidence and publication

All new local files are beneath `rust-engine/experiments/mer2_cuda_rtps_p1/`.
The source payload is enumerated by `PATH-ALLOWLIST.txt`; `SOURCE-PAYLOAD.sha256`
is embedded as identity evidence in the future CLI. It is not authentication of
a compiled binary, host or execution grant. The manifest excludes itself;
`evidence/PATH-HASH-INVENTORY.json` additionally records the patch and evidence
scope, and `evidence/ALL-FILES.sha256` covers every other final local file.

The exact addition patch is `evidence/RTPS-P1-SOURCE.patch`. Its Git parent is
the live-verified baseline `b55fd52d5043e7865a340c04f02433b0fa5dc974`, not W3's
258-file tree. W3 is read-only reference material; none of its changed files
are imported. The existing remote base branch
`fix/mer2-g3-p6a-a3-s3-p1a-r1-source-blockers` was verified at that baseline;
the complete live baseline tree has 248 blobs and no `.github` workflow files.
The intended draft head is `codex/mer2-rtps-p1-source-20261104-01a1a973`.
Only the new directory may differ. Publication receipts, exact head and PR URL
belong in local `evidence/PUBLICATION.json` after remote readback. Do not retarget
the draft to main or represent this partial source as a native CUDA artifact.

## Two future command proposals — NOT EXECUTED

After independent review and a separate CPU-only authorization, from this
experiment directory, one proposed offline compile/type-check command is:

```sh
CARGO_TARGET_DIR="$PWD/target-review" cargo check --offline --manifest-path Cargo.toml --all-targets --no-default-features --features diagnostic-cli
```

This checks only the partial std-only crate, not CUDA, not a linked executable
and not test execution. Cargo may create the new experiment's local lockfile;
there is no root lockfile mutation or network dependency. Toolchain/sysroot,
Cargo configuration, executable hashes, deadline and write paths must first be
bound. Later CPU execution of the authored reference/tests requires its own
explicit command allowlist and bounded output. No commands were run here.

After the native interface is implemented/reviewed, its immutable binary/PTX and
CPU oracle are independently sealed, and the existing Linux/L4 host/process
boundary is separately authorized, the diagnostic **command shape** is:

```sh
timeout --signal=TERM --kill-after=5s 30s /reviewed/absolute/path/mer2-rtps-p1 native-once
```

This path is deliberately a placeholder, not an installed binary or launcher.
With today's source, `native-once` can only STOP (prospective exit 78). GNU
`timeout` alone is **not** the required process boundary; there is no runnable
GPU command in this package. The founder operates the existing VM under the
standing issue #351 protocol; this handoff grants no AWS/SSH or execution scope.

## Resource, timeout, cleanup and report envelope

Proposed bounds for a later reviewed native implementation: one process/CPU
thread doing synchronous CUDA calls, one L4 device, one context, one stream,
one event, three modules/functions, one expert/request/invocation and no retry.
Three Q4 device buffers total 2,654,208 bytes; six f32 vectors
`[2048,768,768,768,768,2048]` total 28,672 bytes; a three-usize layout adds
24 bytes: **2,682,904 explicit device payload bytes** before guards, allocator,
module, context and driver overhead. These overheads are unknown and not
claimed bounded by that arithmetic. No actual device allocation exists here.

Proposal: host RSS ceiling 256 MiB for this no-model diagnostic, explicit device
allocation ceiling 4 MiB (including reviewed canaries), stdout+stderr 4 MiB,
30-second worker wall deadline plus at most 5-second escalation grace. These
are prospective limits, not measured sufficiency or an enforcement claim.
The host admission must select enforceable memory/output/process containment
and GPU accounting; reject if incompatible. A CUDA API/fence can block beyond
an in-process deadline. A separately controlled parent must own kill/reap,
missing-terminal evidence and device/context recovery. CPU-Q2 H14 observer
requirements are not imported or claimed satisfied by this simpler diagnostic.

Require one terminal cleanup attempt after every native outcome, latch the
first failure before secondary failures, retain uncertain owners and pending
operations, prohibit reuse, and quarantine on any ambiguous enqueue/fence/
readback/destructor result. Checked drain/resource-release receipts never prove
final `CudaContext::Drop` or full leak freedom. Allocation panic/OOM/abort or
output failure may prevent a terminal record: the external parent must detect
that absence, not interpret it as success. The pure `TraceContract` tests only
receipt ordering and rejection; it has no authority to certify native cleanup.

Prospective report fields: source/binary/PTX/toolchain/host identities; immutable
fixture/oracle identities; actual owner/request/expert IDs; attempted/submitted/
fenced operations; all five stage arrays' raw bits and scalar differences;
latched first failure and bounded secondary list/overflow; pending work at
cleanup; one-shot cleanup disposition; terminal status; explicitly unobservable
final context destruction. Actual GPU output remains ABSENT in this partial.

## Hashes and unresolved risks

`SOURCE-IDENTITIES.md` pins inspected native/vendor sources; the complete source
recheck evidence remains local. PTX hashes for QUANTIZED/UNARY/BINARY are
**UNBOUND**, as are build flags, generated symbols, cubin/JIT outcome, binary,
driver ABI, L4 identity, device memory and host containment. No PTX was generated.
Require the f32 ABI (`GGML_CUDA_F16` absent), DMMV geometry, exact symbols
`dequantize_mul_mat_vec_q4_0_cuda`, `usilu_f32`, `bmul_f32`, source/PTX relation,
64-bit size_t/layout and all bounds before any native call.

The saturated SiLU/repeating-pattern limitations in README are deliberate and
must constrain future interpretation. The manifest and receipt checker are
not trusted authority. Rust source has not been compiled, formatted or executed;
borrow/type errors, platform libm behavior and native integration remain
unqualified. No source review here is independent acceptance. CPU_Q2 remains
HOLD_NO_AUTHORITY; P5D, W3 R01/R02, O1/H14 and CPU-REC STOPs remain unchanged.
No successor is scheduled or authorized.

MER2_RTPS_P1_SOURCE=STOP_NATIVE_SYNTHETIC_STAGE_ADAPTER_UNBOUND
