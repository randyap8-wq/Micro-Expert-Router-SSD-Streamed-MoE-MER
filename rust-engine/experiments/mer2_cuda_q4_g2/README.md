# MER2 CUDA G2 S0: one resident synthetic expert

Authority: [issue 296](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/296),
governed by [267 comment 6305294537](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/267#issuecomment-6305294537).
This standalone package adds exactly ten files above frozen S2 commit
`3d3aef41799b65b88f07feedc495707afaaad20c`, tree
`d3ab4c1120726970356ac5e2dc4c64583b3f3de8`, sole parent
`be19b4217ba5e6b5bb963663bf3c3cf88566006f`.
S1, S2, production, vendor sources and existing lockfiles are immutable.

## Scope and fixture

One S2 mixed GGUF Q4_0 SwiGLU expert, 128x128, three 9216-byte matrices.
Install once in slot 0, generation 1. Request sequence 0 is S2 input A;
sequence 1 negates every f32 input bit pattern, including positive zero becoming
negative zero. Weights/layout stay resident through both requests. Retire once.
No concurrency, second stream, eviction policy, model/storage access, fallback,
warmup, intermediate D2H, custom CUDA kernels or production integration.

`fixture-contract.json` freezes both domain-separated input digests, the
input-independent expert digest, all 256 CPU output scalars and bits, original
f64 acceptance (`1e-4 + 1e-5*abs(reference)`), and coordinate 0 as the
nonoverlapping output witness. The source authoring calculation is independently
constructed; acceptance requires exact comparison to the actual unchanged Rust
S2 oracle during the single CPU attempt. A stale A result cannot pass B.

Expert SHA256: `18ac217e178045c2187089e4b9f122e301d833ee41ab5205ef1737c3724733cd`.
A input: `eb7358180e4f3c6a093bbfe1fd19f095624d7fb3994167072e93563cd7bba45a`.
B input: `62a35275c35723f93e1d9f42e8bb6fea967058321e25e70b5c94efca4c074635`.
S2 mixed: `1a774f651b0ab77f17d881a104c490841fa72c4c03faf85a162d15b95bf7b363`.

## Logical payload contract

| Operation | H2D bytes/calls | D2H bytes/calls | Kernels |
| --- | --- | --- | --- |
| Install | 27672 / 4 | 0 / 0 | 0 |
| A | 3072 / 6 | 512 / 1 | 5 |
| B | 3072 / 6 | 512 / 1 | 5 |
| Total | **33816 / 16** | **1024 / 2** | **10** |

Each request resets gate/up/activated/hidden/y from the same initialized
512-byte pinned NaN vector (bits `0x7fc00000`), then copies its 512-byte input.
Actual order is input first, then the five resets. Device allocation IDs
200..209 and pinned IDs 100..105 are retained for the whole run. Device IDs
200/201/202 are the immutable gate/up/down weights; 203 is layout, 204 input,
205..209 gate/up/activated/hidden/y. Host IDs 100..103 stage weights/layout,
104 input, 105 sentinel. Stable boxed host outputs are IDs 300/301.
Exactly 10 device slices (30744 payload bytes), six pinned allocations
(28696 payload bytes), eight application events (IDs 400..407).
There are no allocation/free or weight/layout copy operations between A and B.
Driver/library allocations, internal events, page rounding, pageable fixture
generation, report memory and modules are outside application payload totals.
These counts are not physical PCIe traffic, allocator reservation or performance.

## Shared engine and ownership source review

`state::Runner<A>` statically dispatches the private `Adapter` seam. Both the
concrete CUDA owner and CPU unit-test fake execute this same ordered operation
schedule, state transitions, token checks and receipt accounting. Integration
tests exercise fixture/oracle, serialization and CLI. Expected counters are
separate from successful adapter submission receipts; checked fences alone
advance completion-confirmed bytes. Failed/deferred operations are preserved in
the trace. CPU fake output is SIMULATED; preflight output is EXPECTED, with null
submitted/completed observations and NOT_MEASURED/null timing and VRAM.

`Owner` is non-Clone; it retains context, one default stream, modules/functions,
all slices, pinned sources, events and output boxes. The safe API never exposes
the adapter or concurrent owner access. Event tokens bind run/context identity,
slot, generation, sequence, event index and recorded flag. Exact schedule guards
reject early refill, output reads, install, retirement, stale/duplicate tokens
and extra/missing/reordered operations before any adapter call.

Unsafe source review boundaries (not CUDA compilation or hardware proof):

* `alloc_pinned` is unsafe/uninitialized and WRITECOMBINED in cudarc 0.19.7.
  Allocation is inserted in the owner before fallible initialization. Only
  bounded u8/usize/f32 storage is used; a checked raw pointer receives every
  initialized element before a typed slice or submission. Input refills require
  prior completion. `memcpy_htod` receives the pinned object, with exact length
  and context checks. No reusable-path slice cloning, D2D or pageable fallback.
* Ten disjoint device allocations receive initialized H2D data before reading.
  Kernel output borrows come from a disjoint mutable suffix; weights/input/layout
  use the immutable prefix. Three Q4 calls keep S2's `(weights,input,output,
  ncols:i32,nrows:i32)`, 128 rows, 32 threads/row. `usilu_f32` and `bmul_f32`
  retain their size_t/layout/input/output order and `[128usize,1,1]` metadata.
* Install and request H2D events use explicit `CU_EVENT_DEFAULT` timing pairs.
  Each request preserves seven explicit stream waits: H2D, five kernels, D2H.
  Checked event synchronization and deferred errors additionally guard commits.
  No use of boolean event polling. Timing records distinguish host staging,
  enqueue, host wait and checked event intervals, include sentinel reset payloads,
  and warn about finite resolution/intervening work. VRAM remains NOT_MEASURED
  unless separately instrumented/qualified; no payload-to-VRAM equality claim.
* Direct errors, deferred errors and caught operation panics poison the engine.
  `Runner` preserves the first failure, stops normal calls and attempts one
  checked drain. An unconfirmed drain retains the entire ManuallyDrop owner,
  including host D2H storage. This is terminal quarantine, not recoverable
  rollback. Successful drain does not reclassify incomplete bytes as completed.
* cudarc may allocate memory before an internal event creation fails, leaving
  inaccessible driver resources. Any pinned/device creation error conservatively
  reports this cleanup limit even after drain. Release checks each resource
  destructor's deferred status and device free synchronization; a failure stops
  further release. Final context destruction cannot be observed after the last
  owner is gone. No claim of cleanup after arbitrary driver failure or of
  long-running leak freedom follows from CPU tests.

## Local validation boundary

Use preinstalled Rust/Cargo **1.97.1**, the recorded aarch64-apple-darwin host,
isolated Cargo home and target outside the checkout, one build job, offline
registry bytes. Default features are empty; S2 defaults are disabled. Optional
CUDA dependencies and features exactly match S2 (`cudarc =0.19.7`, vendored
`candle-kernels =0.10.2`, `serde_json =1.0.149`). Preserve all 56 S2 registry
lock entries. The new lock may receive one explicit `cargo update --workspace
--offline --manifest-path .../mer2_cuda_q4_g2/Cargo.toml` reconciliation.

Issue 296 permits at most one bounded CPU attempt (timeout <=300 seconds):

```text
cargo test --locked --offline --no-default-features --jobs 1 --manifest-path rust-engine/experiments/mer2_cuda_q4_g2/Cargo.toml -- --nocapture --test-threads=1
```

The integration test invokes the CPU binary once with `--cpu`, verifies
nonzero `--cuda` refusal with zero GPU invocations, and rejects unknown CLI
arguments. No second test/build attempt or fix/retest after failure is allowed.
Keep first output and draft unchanged on any failed gate. Source formatting and
static identity/graph/linkage inspection are separate from CUDA qualification.

Unit fault injection covers every operation with both a direct and a deferred
failure, including 6/10/8 creation stages, all copies/fences, module/function
creation, all ten kernel launch/wait pairs and both D2H record/waits. Additional
tests cover pending reads/refills, each token field, incorrect calls, overflow,
cancellation/panics in each live phase and failed drain/release. Fault-injection
errors are expected test witnesses, never actual CUDA execution.

## Remaining hardware gate

CUDA feature compilation and execution are **NOT_RUN** under this authority.
GPU_INVOCATIONS=0; GPU measurements NOT_MEASURED; G2 PARTIAL; G3 NOT_IMPLEMENTED.
The external source-gate evidence package records the actual one-shot outcome;
this README makes no claim of a passing invocation before that record exists.
No AWS/SSH/SSM, model, SSD, toolkit installation, download or dependency upgrade.
Historical S2 R1 GPU authority remains consumed.

Future founder-only hardware work requires a separate frozen one-attempt issue
and checksum-pinned operator after review. It must establish CUDA compilation,
real pinned allocation/lifetimes, stream/event completion, unchanged kernel ABI,
256 independently verified outputs/bits, actual allocation/copy receipts,
three timing intervals, appropriately labeled host/VRAM observations and cleanup.
The external checker must independently verify all scalar-bit pairs, hashes,
original f64 tolerance, exact schedule and fixed discrimination witness.
Retain original raw artifacts for off-host audit. No automatic next stage,
performance threshold, throughput claim, merge or production integration.
