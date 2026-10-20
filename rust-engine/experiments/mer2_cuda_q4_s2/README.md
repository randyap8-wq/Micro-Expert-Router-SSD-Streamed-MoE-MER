# MER2-CUDA-S2: isolated synthetic Q4_0 SwiGLU expert

Authority: [issue #291](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/291),
read in full with all comments (empty), refreshed immediately before authoring.
Source and one bounded local offline CPU-only test attempt are authorized.
CUDA compilation and hardware execution require separate founder authority.

Exact base `be19b4217ba5e6b5bb963663bf3c3cf88566006f`, tree
`4c46e997d8bd7ae298ed659c909b73f7fbb243fa`, sole parent
`fec4d5f9e6304e0642990e9debe5d7449a8def38` were verified in a clean fresh
checkout and against GitHub's S1 branch. All 182 existing tracked files were
SHA256-hashed and compared with immutable Git object bytes before additions.
New branch: `feat/mer2-cuda-q4-swiglu-s2`. Only this S2 directory is added.
The stacked draft PR must target `feat/mer2-cuda-q4-reuse-s1` at that exact base.

## Computation and source/API review

Three row-major packed GGUF Q4_0 matrices implement:

```text
gate = W_gate @ x
up = W_up @ x
activated = gate / (1 + exp(-gate))
hidden = activated * up
y = W_down @ hidden
```

All five operations in `src/cuda.rs::expert` use device buffers and the same
default stream. The only computation-path D2H is final `y`. The separate CPU
oracle is used for comparison; its activations never feed the CUDA computation.
There is no intermediate diagnostic readback or fallback to CPU/WGPU.

The following existing entries were reviewed before choosing activation reuse.
All references are to the exact base; hashes are enforced by `src/identity.rs`
and emitted in the machine-readable preflight.

| PTX module / entry | Reviewed contract |
| --- | --- |
| `QUANTIZED` / `dequantize_mul_mat_vec_q4_0_cuda` | `quantized.cu:1160–1229`: `(const void*, const float*, float*, int ncols, int nrows)`; grid `(nrows,1,1)`, block `(32,1,1)`, zero shared memory; 64-column iterations, one warp per row |
| `UNARY` / `usilu_f32` | `unary.cu:6–30,59–61,226`: `(size_t numel, size_t num_dims, const size_t* info, const float* inp, float* out)`; `silu_fwd<float>` uses `x/(1+expg(-x))` |
| `BINARY` / `bmul_f32` | `binary.cu:59`, `binary_op_macros.cuh:3–82`: `(size_t numel, size_t num_dims, const size_t* info, const float* lhs, const float* rhs, float* out)`; multiplies two contiguous f32 arrays |

`cuda_utils.cuh:9–25` reviews contiguity; line 148 maps f32 `expg` to `expf`.
Elementwise launches use one 128-thread block, `numel=d_ff`, `num_dims=1`,
and a non-null device `size_t[3]` containing `[d_ff,1,1]` (dimension, lhs stride,
rhs stride). Unary reads the first two entries. This avoids the macros' null
pointer arithmetic. Both kernels guard `i<numel`, and cover each output once.
The reviewed ABI requires a little-endian 64-bit host and rejects other hosts.

`quantized.cu` explicitly undefines `GGML_CUDA_F16`; `dfloat` is float. Its
Q4_0 block is 18 bytes: f16 scale plus 16 bytes of split nibbles. The low nibble
is element j; high nibble is j+16; both decode as `(q-8)*scale`. Matrices have
64 or 128 rows/columns, guaranteeing kernel tails and C-int indexing are safe.
CUDA allocations supply base alignment; every block offset is a multiple of 2,
and f32/size_t buffers are typed, separate allocations with no subviews/aliasing.

Reuse is through Candle **0.10.2** `QUANTIZED.ptx()`, `UNARY.ptx()` and
`BINARY.ptx()` (`src/lib.rs` exports Module objects), loaded with cudarc
**0.19.7** `Ptx::from_src`, `load_module`, and `load_function`.
No new CUDA source, NVRTC compiler call, dependency, Q8_1 packing, or framework
is introduced. The unchanged Candle build still invokes cudaforge for its PTX
and wider MoE static library, with the existing glibc compatibility header,
CUDA runtime and C++ linker prerequisites. Those prerequisites are unqualified.

The reviewed cudarc 0.19.7 archive SHA256 is
`1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc`,
matching the lockfile. Cached review source bytes were compared with that
archive. `driver/safe/core.rs` supplies context, stream, allocation, copy,
module, function, deferred-error and synchronization APIs; `safe/launch.rs`
supplies `LaunchConfig`, `PushKernelArg`, and `launch_builder`/unsafe `launch`.
Plain host buffers remain live through explicit H2D synchronization; device
buffers/modules remain live through launches and final readback. Each launch
and final D2H has an explicit synchronization boundary. Device allocations are
owned by cudarc RAII; the final context synchronization checks recorded cleanup
errors. Loader panics become a nonzero failure, never a substitute result.

All intermediate/output buffers start with NaN sentinels. Unwritten entries
propagate through these arithmetic kernels to fail the final finite gate,
including zero-weight cases. Shape, exact byte length, finite scales/input,
and conservative absolute f64 bounds are checked before device allocation.
The bounds leave f32 headroom for every projection and the hidden product;
they are overflow guards, not a relaxed comparison tolerance. Final actual
values and their exact f32 bits are captured before comparison, including
nonfinite values (which JSON represents as null with their raw bits retained).

## Frozen fixtures and acceptance

The standalone manifest/lock isolates production MER. S1 is a path dependency
with default features disabled solely for Q4 decoding, SHA256, and comparison.
No S1 files or shared lockfile are edited. The four fixtures are deterministic,
synthetic, and have no model/filesystem/RNG dependency. `pack_block` encodes
exact signed Q4 levels and an already quantized f16 scale, without a new lossy
float quantizer. Fixture scales/input are exactly representable powers of two.

| Fixture | d_model | d_ff | SHA256 |
| --- | --- | --- | --- |
| zero | 64 | 64 | `774167646910b18b31ffca71a76fa2b93d15efa9abcbc0dd16a28206e921250b` |
| alternating | 128 | 64 | `d58ed6cd42fba55ec8f2a014d92553cbcf268b6706b2f8409cbb1612674d3e6c` |
| sign | 64 | 128 | `34fc7c663dbe980c39a0de65c25e8c4ab6da2503ac60d6906d36d87bce4eb3ea` |
| mixed | 128 | 128 | `1a774f651b0ab77f17d881a104c490841fa72c4c03faf85a162d15b95bf7b363` |

Fingerprint bytes: ASCII `MER2-CUDA-S2-fixture-v1` plus NUL; then for each of
gate/up/down: LE u32 rows, LE u32 cols, packed matrix bytes; finally LE u32 input
length and each LE f32 input. Names are labels, not part of the byte identity.
`fixture-contract.json` was serialized independently with Python struct/hashlib.
CPU tests check the Rust-generated identities against that frozen contract.

The CPU reference uses S1's actual f16/nibble decoding and scalar f32 multiply
and accumulation in column order, followed by f32 SiLU and multiply. Reference
bits are deterministic for the tested build/platform; `expf` and GPU reduction
rounding can vary across platforms. No cross-platform bitwise parity is claimed.

Every one of the **384 final outputs** must be finite, have the expected shape,
and satisfy `abs(actual-reference) <= 1e-4 + 1e-5*abs(reference)`, evaluated in
f64. There are no tolerance flags/environment overrides. A future numerical
failure requires stopping; a different bound needs separate acceptance review.

Preflight JSON contains fixed source and fixture hashes, entries, CPU reference
values/bits, acceptance and a deterministic logical byte trace per fixture.
The trace includes three weight uploads, x, all NaN initializers, 24 bytes of
layout metadata, four f32 device temporaries, y, and final-y-only D2H. Totals
are logical payload sizes, excluding allocator/module/driver overhead. They
are not measured VRAM, transfer timing, bandwidth or throughput.

## Local CPU validation

The authorized single bounded attempt is:

```text
cargo test --locked --offline --no-default-features --manifest-path rust-engine/experiments/mer2_cuda_q4_s2/Cargo.toml -- --nocapture --test-threads=1
```

Run from repository root so production `rust-engine/.cargo/config.toml` is not
loaded. This task uses an isolated Cargo home copied from preexisting local
cache bytes, an external target directory, one Cargo build job, and a 300-second
test timeout. No network fetch/toolchain installation is allowed. The offline
resolved CPU graph must exclude cudarc, Candle kernels, cudaforge and production
MER before the test process starts. Missing environment or test failure stops
publication and leaves source uncommitted; no retry is authorized.

Tests cover Q4 packing/decoding, signed/zero/subnormal scales, invalid values,
independent projection arithmetic, SwiGLU values/errors, diagonal and rectangular
three-matrix references, malformed geometry, finite/overflow guards, frozen
hashes, determinism, logical boundaries and numerical acceptance. A CPU-only
CLI test records the complete preflight and proves `--cuda` fails explicitly
with zero GPU invocations and `CUDA_GPU_TEST=NOT_RUN` when the feature is absent.
This refusal does not access CUDA. No later CLI execution is needed for receipts.

## Remaining founder-only validation

Review the draft source, published commit/tree and every file hash first.
Then obtain separate S2 hardware/evidence authority, verify host/device/driver,
Linux Rust/Cargo, toolkit/NVCC/compute architecture, dependency cache and Candle
build prerequisites. Build the CUDA feature only under that future authority.
Record complete build logs and identities; source review/CPU tests do not prove
CUDA compilation, linker compatibility, pointer correctness on hardware or parity.

A separately authorized future run must retain flushed preflight and final JSON,
exit status, all three loaded PTX hashes, exact fixture hashes and all 384 output
bits. Independently verify every output against the frozen tolerance and review
device residency/complete writes. Missing final JSON, nonzero exit, nonfinite
output or any numerical failure cannot be PASS. Status strings are selected by
Rust branches; there is no Bash status formatter or concatenated PASS/FAIL.
There is no benchmark, model restoration, Qwen/full-token or production claim.

The issue records the historical S1 L4 R3 attempt as consumed, operator-reported
PASS with a formatter defect. Its original evidence is immutable; independent
off-host raw-evidence audit remains pending. It must not be repeated.
No AWS/SSH/SSM/VM/GPU access by Codex and no automatic next stage.

```text
R3_HISTORICAL_GPU_ATTEMPT=CONSUMED_PASS_WITH_FORMAT_DEFECT
CUDA_S2_COMPILATION=NOT_RUN
CUDA_S2_GPU_TEST=NOT_RUN
S2_GPU_RUNS_BY_CODEX=0
NATIVE_CUDA_FULL_TOKEN=NOT_IMPLEMENTED
WGPU_PRODUCTION=UNCHANGED
PR283=OPEN_DRAFT_UNMERGED
PR265=OPEN_UNMERGED
AWS_VM_ACCESS_BY_CODEX=FORBIDDEN
```
