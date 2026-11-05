# RTPS-P1 R1 — verified CPU oracle and isolated native adapter source

**CPU gate: 24/24 passed on the original P1 source. Native adapter: source only,
uncompiled, unexecuted, no run authority. NO MERGE.** One offline Cargo check
and one offline library-test attempt ran on macOS before any R1 source edit.
The independent CPU reference reproduced all 5,120 unchanged expected bits.
No subsequent Cargo command ran. These results do not validate the R1 native
module, Linux compilation, a driver, or GPU behavior.

`src/native.rs` stages a separate synthetic-only adapter for the exact vendored
Q4_0, f32 SiLU and f32 multiply kernels. It retains five output buffers, uses
actual device-to-host bit copies, and records symbol/launch/fence evidence.
Its required `ReviewedHostGate` is uninhabited with no constructor. The private
module has no CLI caller or public export. `native-once` still returns
`STOP_NATIVE_SYNTHETIC_STAGE_ADAPTER_UNBOUND` before fixture allocation or driver
lookup. No CPU fallback or P6B/model owner is connected.

The isolated workspace is `rtps-p1-20261104-01a1a973-source-only`; the standalone
`[workspace]`, empty default features and `publish=false` prevent implicit parent
workspace membership or default CLI inclusion. Enabling `diagnostic-cli` exposes
only `cpu-reference` and the noninitializing `native-once` STOP. The separate
`native-diagnostic` feature is default-off, Linux x86_64 little-endian only,
and cannot mint authority. Its dependency lock/PTX/host gate remain unresolved.

Review [the source handoff](RTPS-P1-SOURCE-REVIEW-HANDOFF.md),
[source identities](SOURCE-IDENTITIES.md), [allowlist](PATH-ALLOWLIST.txt), and
[next gates](NEXT-GATES.md). Original P1 evidence under `evidence/` describes
that earlier source-only package. R1 logs, exact incremental patch, admission
records and final inventory are retained separately at
`outputs/MER2-RTPS-P1-R1/evidence/` in the local MER workspace, excluded from the
PR. The Stage A CPU-only Cargo.lock is also local, not a native dependency lock.

## Fixed synthetic input

Exactly one expert, request 1, D=2048, DFF=768. Rows are output coordinates;
columns are input coordinates. Gate and Up are `[768,2048]`; Down is `[2048,768]`.
Each row has contiguous 18-byte Q4_0 blocks of 32 values. Bytes 0–1 are a
little-endian binary16 scale. Byte `2+j` stores column `j` in its low nibble and
column `j+16` in its high nibble; dequantization is `(nibble - 8) * scale`.

Both orientations have `2048 * 768 / 32 * 18 = 884,736` bytes per projection.
The three raw projections total **2,654,208** bytes, with offsets
0 / 884,736 / 1,769,472. Gate/Up row strides are 1152 bytes; Down is 432 bytes.
There is **no UTH1 header**: this writer and reference consume raw memory.
P6B's existing `VerifiedExpert::from_bytes` instead requires a 4096-byte UTH1
page. That API mismatch is a blocker, not permission to forge a model file.

Let `m = 1 + row % 4`. The activation is exactly +1/32 in columns 0–15 of each
32-value block and -1/32 in columns 16–31 (`0x3d000000` / `0xbd000000`).

| Matrix | Scale bits | Signed Q4 integer, before scaling |
| --- | --- | --- |
| Gate | `0x3800` = 1/2 | +m in the low half, -m in the high half |
| Up | `0x2800` = 1/32 | Gate integer times -1 for even rows, +1 for odd rows |
| Down | `0x3000` = 1/8 | +m if row+column is even, otherwise -m |

Every block and input column is used, including the final blocks. There is no
random seed, file input, GGUF, tokenizer, dense model or original Qwen weight.
Only tensor geometry matches Qwen3-Coder; this cannot establish real model
loading, R01/R02, a token or TPS.

## Independent expected results, fixed before device results

`reference.rs` implements a separate scalar binary16/Q4 decoder, ascending-column
f32 matvec, f32 SiLU, multiplication and Down. It imports neither the fixture
encoder nor vendor CUDA code. `expected.rs` supplies an independently derived
period-four exact-bit oracle for all **5120** stage elements:

| Stage | Values by index modulo four | Expected f32 bits |
| --- | --- | --- |
| Gate | 32, 64, 96, 128 | `42000000 42800000 42c00000 43000000` |
| Up | -2, 4, -6, 8 | `c0000000 40800000 c0c00000 41000000` |
| SiLU | 32, 64, 96, 128 | same as Gate |
| Product | -64, 256, -576, 1024 | `c2800000 43800000 c4100000 44800000` |
| Down | -46080, 92160, -138240, 184320 | `c7340000 47b40000 c8070000 48340000` |

Derivation: Gate = `2048*m/64 = 32m`; Up = `±2048*m/1024 = ±2m`.
For Gate >=32, rounding `1 + exp(-Gate)` to f32 gives 1, so SiLU has Gate bits.
One four-element product group has absolute sum 1920; Down has 192 groups,
scale 1/8 and signed row multiplier m, giving `±46080m`. All matvec sums here
are exactly representable dyadic values. These expectations were authored before
device results and remain byte-identical. The R1 Stage A CPU test independently
reproduced them; no GPU results exist and no expected bits were retuned.

Acceptance is exact bit equality, with actual bits, expected bits, signed f64
difference and absolute f64 difference printed for every element. No adjustable
epsilon or GPU-driven regeneration exists. The CPU result is never substituted
for a device result. The partial CLI cannot accept or claim GPU output.

Coverage limit: SiLU is on its positive saturated branch here; replacing SiLU
with identity would escape numerical detection. Row/block patterns also repeat.
This is a deliberately narrow geometry fixture, not comprehensive nonlinear,
random-matrix or reduction-drift qualification. A future adapter must prove the
actual SiLU symbol/launch order, and separate numerical coverage must be selected
before broader claims. P5D strict numerical STOP remains unchanged.

## Source rejection and evidence surface

`fixture.rs` rejects dimensions, orientation, stride, projection offsets, tail
bytes, scale drift (including finite wrong scales), NaN/Inf and fixture mutations.
`reference.rs` separately rejects malformed input and arithmetic nonfiniteness.
`contract.rs` authors one-invocation/owner/order/fence/readback/cleanup rejection
logic with a latched first error, bounded ordered secondary errors and overflow
flag. It is a **pure receipt checker**, not a native resource owner, independent
host observer or authorization API. Its mock receipt tests cannot establish
actual driver failure handling. `report.rs` retains source identities and never
asserts final `CudaContext::Drop` or full resource leak freedom.

All original 24 Rust tests passed in Stage A on commit
`3a29f3cce5ea74ae0b1099c4081aa8d42618f7b6`. Four additional native-source
test declarations were added afterward and remain uncompiled/unexecuted.
They check ABI constants and error bookkeeping, not hardware. The adapter is an
independent kernel/geometry diagnostic; authentic Qwen, W3 R01/R02, P5D,
CPU_Q2, model loading, tokens and TPS remain outside its claims.
