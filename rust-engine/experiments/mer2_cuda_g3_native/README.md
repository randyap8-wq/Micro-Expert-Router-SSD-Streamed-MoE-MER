# MER2 G3 native CUDA adapter — P3 source draft

This standalone sibling crate implements the public G3-S1 `abi::Adapter` for
`state::Runner<NativeCudaAdapter>`. It is an uncompiled, unexecuted source draft
under issue #340. It carries no CUDA, numerical, model, performance or production
qualification. No existing S0/S1 file or root workspace is changed. No lockfile
is introduced in this source gate.

## Frozen boundary

- Base HEAD: `ec8eea2edd6ea7a577d2f6de3e99d074462b00e1`
- Base tree: `42d8c41a8187dbc104ff061b5dd2748baaf2f4d8`
- Local branch: `feat/mer2-cuda-g3-s1-host-seam-p1a`
- S0 aggregate: `77858567feeaf0b03736bdd913c954946b6ba6a7c480a79d891a635a3e40193b`
- S1 manifest: `b66d6bd2854299fe2379119ca557c301fef9291cf47679f299168b992235a34e`
- S1 lock: `3203426614a1fa78cd61fa2a201014d969b1e88b2475cd4b80e52f4471fa72d7`

The accepted P2-R3 19-test CPU result applies to the frozen S1 host implementation.
It does not qualify this native adapter. The sealed `FfnProvider` is implemented
only by S1's existing Runner; this crate neither implements nor bypasses it.

## Computation and identities

The CPU decoder observes top-2 routing. Each selected expert receives a full S1
lease: request/ticket, token position/attempt, layer, expert, rank, slot,
generation, context/event identities, input hash, selected weight bits, combined
weight digest and all three matrix digests. The adapter validates the immutable
resident binding and operation order, and returns exact cloned operation
identities. It cannot create S1's private `ConfirmedFfn` directly.

Initialization uploads all eight experts (2 layers x 4 experts), each with three
64x64 Q4_0 matrices. Each matrix is 64 x (64/32) x 18 = 2304 bytes; an expert is
6912 bytes; all resident weight payloads total 55296 bytes. These are payload
counts, not measured VRAM usage. Resident allocations remain live until cleanup;
there is no eviction, reload or prediction path.

Every expert uses exactly seven operations:

1. Upload the owned 64-f32 input and reset all five scratch outputs to NaN.
2. Gate Q4_0 matvec (`dequantize_mul_mat_vec_q4_0_cuda`).
3. Up Q4_0 matvec (the same symbol).
4. SiLU (`usilu_f32`).
5. Elementwise product (`bmul_f32`).
6. Down Q4_0 matvec.
7. Transfer all five 64-f32 stages to retained host destinations.

Matvec uses grid `(64,1,1)` and block `(32,1,1)`: one warp per row, 64-column
stride, two 18-byte Q4 blocks per row. Elementwise kernels use 64 threads and
non-null contiguous `[64,1,1]` metadata with one dimension. No 128x128 scheduling
assumptions are imported from G2. The five-stage readback supplies S1's required
`FfnStages`; CPU work never substitutes for any expert stage. Embedding,
attention, routing, KV updates, rank-weighted aggregation and logits stay in S1.

## Ownership, fences and unsafe boundary

Only `src/cuda.rs` permits unsafe code. Its unsafe blocks allocate uninitialized
device buffers or launch the three source-pinned vendor kernel ABIs. All lengths
are fixed and checked, outputs are disjoint whole slices, and source buffers are
initialized before use. Unsafe code does not alter the qualified host crate.

One explicit Owner retains context, stream, modules/functions, eight events,
24 device weight buffers, layout, six scratch buffers, and all host transfer
sources/destinations. Ordinary host slices have stable owned allocations and
remain alive through completion or quarantine, including on partial submission
failure. This does not assert pinned transfers, overlap or transfer performance.

Submission records pending ownership before a fallible enqueue. An application
event is recorded after all work for that operation. Completion checks the exact
pending operation and then checks stream synchronization, event synchronization
and Cudarc deferred errors. Output is consumed only after the final D2H fence;
all five stages must be finite and exactly 64-wide. NaN sentinels expose missing
writes. Scratch reuse requires the preceding final readback to be completed.
Event objects are reused only after checked completion; full leases and unique
logical event IDs distinguish uses. A failed drain creates no synthetic completion.

Errors are terminal for execution and the first error is retained separately
from cleanup errors. S1 retains its own first error, guarding every operation.
Resource-creation uncertainty is latched before entering Cudarc and cleared only
after retaining a returned owner and checking deferred errors. As in G2, a failed
constructor may hide an allocation: even a successful context synchronization
cannot then confirm ownership. S1 quarantines the complete adapter instead of
running destructors. Direct unchecked adapter drop is conservative too.

Release requires checked drain. Each resource destructor is followed by its
deferred-error check; device frees also require synchronization. The first failed
release stops further teardown and quarantines remaining resources. Context
ownership remains available for S1's post-release check. As in accepted G2,
destruction of the final Cudarc context owner is unobservable and is explicitly
excluded from the resource-release claim. Neither source inspection nor Cudarc
receipts prove recovery of an allocation hidden by a failed dependency call.

## Unexecuted harness and provenance

Default features are empty. The feature-disabled executable always exits with
`STOP_CUDA_UNAVAILABLE` before reading arguments or constructing a GPU owner.
With separately authorized compilation and execution, the CLI shape is:

```text
cuda-chain REPO_ROOT S1_SOURCE_SHA256_JSON NATIVE_SOURCE_SHA256_JSON DEVICE_ORDINAL
```

The JSON files are externally reviewed maps from crate-relative filenames to
SHA-256. S1's map must also match the accepted manifest hash. Native verification
compares exactly five compiled source hashes with both the supplied map and
regular physical files, and repeats before each token. S1's VerifiedSources
checks its frozen physical S0/S1 baseline and is rechecked by the existing host.
The evidence ZIP supplies both maps. Source verification is not execution authority.

The harness prepares Primary synthetic requests A (initial token 3) and B
(initial token 17), interleaved A/B for three tokens each. It uses the frozen
Session/Request decoder and a single Runner. A complete future run should have
24 FFNs, 120 kernel launches and 168 submitted/completed operations. These are
expected counts, not results of this source gate. There is no CPU fallback.

S1 hard-codes simulated labels in its JSON and cleanup report. This crate leaves
them intact as `inherited_s1_receipt`/`inherited_runner_*`; they are not CUDA
proof. Separate `MER2_G3_NATIVE_CUDA_UNQUALIFIED` evidence records PTX hashes,
resident identities, actual context/event observations, operation submissions,
checked fences and stage digests. Each host completion is matched to exactly one
native completion, including raw stage bits. Successful harness output still
requires independent numerical/driver review; it never prints a qualification pass.

## Source references and unresolved gates

- S1 `src/abi.rs`: Adapter, Lease, Operation, Completion, FfnStages, private
  ConfirmedFfn and sealed provider boundary.
- S1 `src/state.rs`: guarded Runner, immutable residents, seven operations,
  completion validation, checked drain/release and quarantine.
- S1 `src/host_token.rs`: injected provider execution, CPU aggregation, session
  isolation and transactional KV handling; `src/main.rs`: A/B workload.
- S2 `src/cuda.rs`: 64-compatible five-kernel argument layouts and launch geometry.
- G2 `src/cuda.rs` and `src/state.rs`: retain-before-fallible-call ownership,
  checked fences/destructors, dependency allocation uncertainty and quarantine.
- Vendored Candle `src/quantized.cu`, `src/unary.cu`,
  `src/binary_op_macros.cuh`, `src/binary.cu`: exact kernel ABI and indexing.

Optional Candle 0.10.2 uses the same vendored path as S2/G2; optional Cudarc
0.19.7 uses their exact feature configuration, with serde_json 1.0.149. The vendor
build script builds PTX and additional MoE objects, discovers CUDA architecture
and includes the MER glibc compatibility header. None of that has run here.

Separate review must authorize dependency/lock resolution and feature compilation
before any controlled NVIDIA L4 run. Open questions include Rust type/link
compatibility; CUDA toolkit, driver and host compiler compatibility; generated
PTX ISA/SM target, symbols and hashes; the vendor's auxiliary MoE build; behavior
of retained ordinary host transfers on the selected driver; error-injection and
cleanup behavior; and numeric/route/token differences from CUDA reduction order,
FMA and SiLU. Exact CPU bit identity is not assumed. Future validation needs an
independent tolerance/route/token acceptance policy and complete operation logs.
No hardware probe, compilation, test, inference or benchmark is part of P3.
