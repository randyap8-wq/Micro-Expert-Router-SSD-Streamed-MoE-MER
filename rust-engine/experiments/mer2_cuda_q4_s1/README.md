# MER2-CUDA-S1: synthetic Q4_0 matvec

Authority: [issue #282](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/282),
read in full with its complete comment collection (empty) before implementation.
This standalone experiment has its own workspace and lockfile. Production MER,
WGPU, the vendored kernels, the parent branch and PR #265 are unchanged.

Frozen parent: `diag/predictor-v2-p1x-multilayer-coverage`,
commit `fec4d5f9e6304e0642990e9debe5d7449a8def38`,
tree `951748c8440839d39d1fca2f173e705d6774ff40`.
Working branch: `feat/mer2-cuda-q4-reuse-s1`.

## Static source/API review

All references below are to the frozen parent, verified against Git object bytes
before edits. This is source evidence, not CUDA compilation or numerical parity.

- `rust-engine/vendor/candle-kernels/src/quantized.cu`: `dfloat` is `float`
  (line 25); `GGML_CUDA_F16` is explicitly undefined (line 18).
  `block_q4_0` has `half d` then 16 bytes of nibbles, with a size assertion
  of 18 bytes (lines 291–298). Dequantization (655–671) maps the low nibble
  to element j and the high nibble to j+16, both as `(q-8)*d`.
- The selected export (1227–1229) takes
  `(const void *weights, const float *input, float *output, int ncols, int nrows)`.
  The underlying loop (1160–1225) consumes 64 columns per iteration without a
  tail guard and reduces over 32 lanes. We launch `(32,1,1)` threads and
  `(nrows,1,1)` blocks, zero dynamic shared memory. CUDA widths must be multiples
  of 64; CPU decoding accepts multiples of 32. No padding or alternative kernel.
  Tiny shape limits (1–32 rows, 32–4096 columns) bound signed-int indexing.
- `mul_mat_vec_q4_0_q8_1_cuda1` (2701–2707) expects Q8_1 activations and seven
  arguments. It is deliberately not used.
- Candle 0.10.2 exports `QUANTIZED` as a **Module**, not a string:
  `candle_kernels::QUANTIZED.ptx()` is the reviewed API. `src/lib.rs` includes
  generated `$OUT_DIR/ptx.rs`; checked-in `src/ptx.rs` documents corresponding
  PTX constants but is not the included generated file.
- The unchanged Candle `build.rs` uses cudaforge to compile `.cu` sources to PTX,
  and also builds/links `libmoe.a`, `cudart` and, on non-MSVC, `stdc++`.
  Enabling this small probe therefore still requires Candle's wider existing
  CUDA build prerequisites. This slice does not change that build.
- `rust-engine/src/inference.rs:762–783` supplies the exact little-endian f16
  and split-nibble dequantization semantics reproduced here. The CPU oracle
  dequantizes first and accumulates scalar f32 products in column order.
  `gpu_native_q4_expert.wgsl:64–86` confirms row/block addressing and projection
  semantics, but factors scales and uses a different summation order.
  GPU warp reductions may differ numerically; bitwise parity is not required.
- The issue's `src/moe/...` references resolve inside the vendored Candle crate:
  `rust-engine/vendor/candle-kernels/src/moe/{moe_gguf,moe_wmma_gguf}.cu`.
  Their dtype dispatch handles Q8_0/Q4_K/etc., not Q4_0. Neither is called here.
  Root `cuda=["candle-core/cuda"]` remains the legacy F32 path.
  `scripts/setup_cuda_env.sh` was inspected as text and was not executed.
- cudarc is exactly **0.19.7**, not the coexisting 0.17.8 in MER's lockfile.
  Its cached archive SHA256 matches that lockfile:
  `1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc`.
  Reviewed `src/driver/safe/core.rs` methods: `CudaContext::new`,
  `default_stream`, `clone_htod`, `clone_dtoh`, `synchronize`, `load_module`,
  `load_function`; `safe/launch.rs`: `LaunchConfig`, `PushKernelArg`,
  `launch_builder` and unsafe `launch`; `nvrtc/safe.rs`: `Ptx::from_src`.
  `load_module` requires the `nvrtc` feature even for existing PTX.
  No NVRTC compilation API is called. Dynamic-loader panics are caught as
  explicit failures; CUDA toolkit version detection is deferred to the build.

Original source SHA256 (paths relative to repository root):

| Path | SHA256 |
| --- | --- |
| `rust-engine/vendor/candle-kernels/src/quantized.cu` | `510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972` |
| `rust-engine/vendor/candle-kernels/src/lib.rs` | `91992b48ff931befcaccc854f639a173de1d1f831c7f5b45185b6bd811a7d7ad` |
| `rust-engine/vendor/candle-kernels/src/ptx.rs` | `1815594dc23daf6354c343059621165cf1ee4bd3cfa71729d35668b22de6239f` |
| `rust-engine/vendor/candle-kernels/build.rs` | `d2683e67d2c82d7d3b2e03e937c625155e2218702fe21cf38a164a82b8bdeedf` |
| `rust-engine/vendor/candle-kernels/Cargo.toml` | `fc19f7428b8df705b699c386dac86f25f7b3ad02dfa624146b88d3b351c39805` |
| `rust-engine/vendor/candle-kernels/src/moe/moe_gguf.cu` | `71aa0a76621d222737d069d62c437f70061f2a5bab31a5c1b1dca218d13de06f` |
| `rust-engine/vendor/candle-kernels/src/moe/moe_wmma_gguf.cu` | `536205c9d568b5c67b5e5889d88ace36dadcc8517242fd569f43ba11ca236735` |
| `rust-engine/Cargo.toml` | `3bbe81d80e83619608bcec21641ccfae8e419d9f83b52490fd9da97bcd30aaff` |
| `rust-engine/Cargo.lock` | `857922a38b4317e1767890f9d9d1438581314ffc0af8191d2a671a91bf7df9c5` |
| `rust-engine/src/inference.rs` | `53bd19d68f512d36bd85fb417de698424b5b10831b88b657d4b493e5bfa2c6ad` |
| `rust-engine/src/backend/wgpu_shaders/gpu_native_q4_expert.wgsl` | `54bae352aad7f25920212f596c00b8e3bac2c0a14281b9664d219d1eea212306` |
| `scripts/setup_cuda_env.sh` | `6d1102d84d8e17803762a901199523e57be8332e8cf3abbec9e0bed255919bc3` |
| `rust-engine/cuda/glibc_cuda_compat.h` | `7f84797c9c440978534dd54d7ee476b83e37e6d591063582f48e8f8b1b964fee` |

## Fixtures and acceptance frozen before GPU execution

Four row-major synthetic projections use positive/negative f16 scales, zero
quants, alternating nibbles, mixed nibbles and nontrivial deterministic f32 input.
No model files or production GPU residency are accessed.

SHA256 serialization is ASCII `MER2-CUDA-S1-fixture-v1` followed by a NUL byte,
LE u32 rows, LE u32 columns, packed matrix bytes, then LE f32 input bytes.
Expected hashes were independently calculated with Python `struct`/`hashlib`
and are asserted by CPU tests.

| Fixture | Rows × columns | SHA256 |
| --- | --- | --- |
| zero | 1 × 64 | `9d457eb950d411095c5f1fcf64024f04e67ebf1ce9ceb33d77e2060527675acd` |
| alternating | 3 × 64 | `6a5d5c4d51e9ca0d3f265fde6791fed9cbe7640ec1529a024094c2bd0677267b` |
| mixed | 5 × 128 | `87e9c02bc57028eff67e459627b775a06a6d181268432c68924edf2085306526` |
| wide | 2 × 192 | `eb053516b2208b1e7077202405ae5ee288db89509add7aa7338db620e3bc24f1` |

Every output must be finite and have exactly `nrows` elements. The fixed gate is
`abs(actual-reference) <= 1e-4 + 1e-5*abs(reference)`, calculated in f64.
Thresholds have no runtime overrides and must not be tuned after GPU results.
CPU tests additionally cover 32/96-column reference cases, all nibble positions,
signed/zero/subnormal f16 scales, exact hand-calculated projections, invalid
shapes, nonfinite data, reference overflow, and tolerance acceptance/rejection.

CUDA output starts as NaN so unwritten rows cannot pass a zero reference.
All allocations/copies belong to this experiment, on one default stream, with
explicit synchronization after H2D, launch, D2H and completion. The executable
emits a preflight JSON line (flushed before CUDA access), then a result JSON line
for requested CUDA execution. These contain source and fixture hashes, fixed
tolerances, references, GPU-returned values and exact u32 f32 bit patterns.
The loaded PTX hash and device identity are included when available.
NaN/Infinity JSON values can be null; their exact bit patterns are retained and
the numerical gate fails. Any runtime error is nonzero, with no CPU fallback.
`PASS` requires all four actual GPU comparisons. A missing final summary,
process startup failure or abnormal termination must never be treated as PASS.

## Local CPU checks

Known local toolchain: Homebrew Rust/Cargo 1.97.1, `aarch64-apple-darwin`.
`rustc` commit `8bab26f4f68e0e26f0bb7960be334d5b520ea452`;
Cargo commit `c980f4866141969fab6254a680546a277789d6f0`.
No `nvcc` was found on PATH. No CUDA build, GPU execution, remote access or
cloud command was performed.

From this experiment directory, with a prepopulated offline Cargo cache:

```text
cargo test --locked --offline --no-default-features
cargo run --locked --offline --no-default-features -- --cpu
```

The test command also checks the CPU binary's JSON and the explicit refusal of
`--cuda` when the feature is absent. That refusal does not access CUDA.
The resolved CPU graph contains neither cudarc, candle-kernels, cudaforge nor
the production MER package. Optional dependencies are present in the separate
lockfile but are not compiled by CPU-only commands. No dependency was fetched.

Local receipt, 2026-10-20: the single authorized
`cargo test --locked --offline --no-default-features` attempt passed:
**15 passed, 0 failed, 0 ignored** (plus empty library/binary/doc-test suites).
The command used `--manifest-path` from the checkout root, a task-local Cargo
home populated from existing cached bytes, and an external target directory.
No environment retry or dependency download was needed. Scoped rustfmt and
staged `git diff --check` passed. A single subsequent invocation of the already
built executable with `--cpu` exited 0 and emitted all four frozen fixture hashes,
CPU reference bits and `CUDA_GPU_TEST=NOT_RUN`.
The expected disabled-feature refusal in the tests emits `FAIL`; it is a
negative CPU-only test and supplies no GPU execution evidence.

## Future manual operator validation — separate approval required

These are text instructions, not an approved VM script. Codex must not perform
VM access, SSH/SSM, remote execution, GPU runs, benchmarks, model restoration or
SSD changes. The founder will manually execute a separately reviewed future
script under separate authority.

Before any future build/run, verify the published commit and every experiment
file hash, the frozen parent/tree, the unchanged kernel hashes above, and the
fixture hashes/tolerances. Record exact host, NVIDIA device/driver, Rust/Cargo,
NVCC/toolkit, compute architecture and dependency-cache identities. A compatible
Linux NVIDIA host, toolkit/header/compiler stack, driver PTX support, CUDA runtime
linker paths and the unchanged Candle build prerequisites must be verified.
The vendored glibc compatibility header is still used by Candle. Do not assume
this source-only API review establishes host or toolchain compatibility.

Expected future commands, only after independent approval and cache preparation:

```text
cargo build --release --locked --offline --features cuda
cargo run --release --locked --offline --features cuda -- --cuda
```

Retain build logs, exit statuses and both JSON lines, including loaded PTX hash
and every output bit pattern. Compare the preflight fixtures to the frozen table,
then independently review all four result comparisons. The binary hardcodes
device ordinal 0; confirm its identity before execution. Failed compilation,
missing CUDA, wrong shapes, nonfinite results or tolerance failures are failures,
not permission to change the kernel, tolerance or parent.

No performance, full inference, pinned-H2D, stream/event ownership qualification,
GPUDirect Storage, top-k, complete SwiGLU or full-token claim is made.
Historical GPU2 `FAIL_CONSUMED` remains unchanged. No automatic next CUDA slice.

```text
MER2_CUDA_S1=SOURCE_IMPLEMENTED_CPU_TESTED_GPU_UNEXECUTED
CUDA_COMPILATION=NOT_RUN
CUDA_GPU_TEST=NOT_RUN
NATIVE_CUDA_FULL_TOKEN=NOT_IMPLEMENTED
WGPU_PRODUCTION=UNCHANGED
PR265=OPEN_UNMERGED
AWS_VM_ACCESS_BY_CODEX=FORBIDDEN
```
