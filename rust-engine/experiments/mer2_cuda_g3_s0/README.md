# MER2 CUDA G3-S0 synthetic CPU source candidate

Authority: GitHub issue #322, accepted D0 plus R1 in #321 comment 6349142148.
Base `ec8eea2edd6ea7a577d2f6de3e99d074462b00e1`, tree
`42d8c41a8187dbc104ff061b5dd2748baaf2f4d8`; ordered parents
`551085db7bf30465c9a4eb76527d68b75560000e`,
`366824927efc69af8fd0c84d71f30b2e71d076a3`.

This is an uncommitted, uncompiled, unexecuted source draft. D1 does not qualify
CPU numerics, CUDA, a real model, throughput, storage, or production integration.
All actual output bits and stage digests remain
`UNRESOLVED_PENDING_D2_CPU_EXECUTION`. The input manifest was serialized using an
independent Python hashlib/struct implementation of the frozen R1 recipe; that
operation generated only input bytes/hashes and did not evaluate the decoder.
Rust reproduction of those input hashes is authored but unexecuted.

## Isolated scope and fixed data

Exactly nine new files; standalone `[workspace]`, empty default features, no CUDA
feature. Only std, half 2.7.1, sha2 0.10.9 and serde_json 1.0.149 are direct
dependencies. S1/S2/G2 are read-only semantic references, never imported.

There are 41 tensors: 17 f32 tensors and 24 packed Q4 matrices. Every name, shape,
dtype and canonical tensor SHA is listed in fixture-contract.json. Tensor and
model hashing follow R1 exactly, including ASCII sort and raw 32-byte tensor
digests in the model commitment. Neither fixture JSON nor source contains its
own literal checksum. External D1 hashes pin all nine files; the future binary
reports the actual compile-time source bytes via include_bytes. That self-report
alone does not validate source authorization: D2 must compare it with the frozen
external source manifest using verify_source and record actual toolchain/target,
features and binary hash before interpreting replay bits.

PRIMARY_V1 model commitment:
`6927d4063b79e0c45ee711e35c3184edffd70088f97cf2b484cd396e028751a1`.
TIE_V1 model commitment:
`b8ef837a457f07156176713fb2d79c89a78fc75e7570abd05dd3766c0c3fbecf`.
The sole TIE override is all 256 L0.router f32 values becoming positive zero.
No seed adjustment is permitted if D2 finds inadequate discrimination.

The 2-layer decoder has d_model=d_ff=64, four attention and KV heads of 16,
four experts per layer/top2, independent [32,64] embedding and LM head, no bias,
context 4, RMS epsilon f32(1e-5), interleaved RoPE base 10000 on Q/K only.
Each Q4 gate/up/down matrix is exactly 2304 bytes: 64 rows, two 18-byte blocks
per row. Low nibbles represent columns 0..15 of each block, high nibbles 16..31;
little-endian f16 scale multiplies nibble minus 8. Negative scales are preserved.
No EOS, tokenizer, hidden prefill, dense expert substitute or model loader exists.

## Scalar operation order and transactions

Each dot starts at +0.0f32 and separately rounds multiply then add, ascending
column order. RMSNorm sums x*x the same way, divides by 64, adds epsilon, takes
sqrt, then computes `(x / denom) * gamma` with a separate f32 division and
multiplication. This spells out the division/multiplication choice left implicit
in R1's output expression. Stable softmax visits logits in ascending index,
subtracts max, computes f32 exp, sums ascending, then divides. RoPE calculates
f32 exponent `(2*j)/16`, f32 powf, divides position by that result, then separate
sin and cos, four f32 products and two add/subtract operations. Attention scales
the complete ordered dot by f32(1/sqrt(16)). SiLU is x/(1+exp(-x)), then multiply
up, then down projection. Every intermediate is finite-checked, including exp
and accumulation overflow, not just final outputs. No mul_add, BLAS, Rayon or
intentional reassociation is used. Cross-platform transcendental bit parity is
not promised; D2 must record its exact compiler, target and flags.

Routing computes all four observed probabilities, ranks descending probability
with ascending expert ID on exact ties, normalizes only the selected pair, and
accumulates their weighted down results in that rank order before residual.
Prediction is deliberately absent from the computation interface. A receipt
claiming prediction in place of the observed route is rejected. CPU expert
leases bind layer, expert, rank, slot=layer*4+expert, immutable generation 1 and
three weight digests. These are CPU ownership labels, not CUDA residency/events.

A Session allocates unique request nonces and rejects duplicate request IDs.
Each Request owns its own [layer][head][position][dim] K/V arrays with private
length/generation; neither Session nor Request implements Clone. Separate replay
sessions intentionally restart identity numbering; their receipts are scoped to
that session/run and must not be used as cross-run authorization credentials.
The ticket binds request/nonce, attempt, input, position, source, fixture and
the full committed KV snapshot. Its input must be the request's initial ID or
the last returned greedy next ID. Both layers compute provisionally for this
input token. Entire token computation, stage hooks, logits and greedy selection
finish before one non-fallible state install advances both KVs, input ledger,
generation and next token. The sampled next token does not enter KV until its
next step. Failed requests retain the first error and become terminal; committed
bytes/generation/length, last output and next ID stay unchanged. No retry inside
the request is supported. Hooks never receive committed storage or live values.

Every stage emits raw f32 bits, shape, contiguous byte strides, layer, sequence,
ticket and SHA; routes carry all four ranked IDs/probabilities and selected
weights, expert stages carry CPU leases, and greedy carries the integer ID.
There are 66 stage boundaries per successful token. The hook sees a copy;
comparison to a sealed pre-hook receipt rejects mutations even if rehashed.
The committed output hash then rejects missing/duplicate/stale/foreign stages,
changed logits, next token or KV receipt. This is an integrity witness, not an
independent numerical oracle. Analytical tests separately cover arithmetic.

## Explicit receipt serialization choice requiring D2 review

R1 freezes domains but not every receipt field width. `G3-S0-RECEIPTS-v1` is the
implementation's versioned choice, not a claim of a previously approved golden
receipt hash. All encodings below use little endian and byte lengths.

`field(bytes)` means u64 byte length followed by exactly those bytes. Stage
preimage starts `MER2-G3-STAGE-v1` + NUL, field(version), field(kind), then Ticket:
field(request ASCII), field(u64 nonce), field(u64 attempt), field(u64 position),
field(u64 input ID), field(u64 KV generation), field(KV hash ASCII hex),
field(source hash ASCII hex), field(model hash ASCII hex). It continues with
field(u64 sequence), field(u64 layer; u64::MAX for no layer), field(stage ASCII),
field(u8 dtype=1), field(u64 ndim), each field(u32 dimension), and field(raw
u32LE f32 bits). Zero-length control stages have shape [0] and no f32 values.

Next comes u8 route-present; if present: u8 observed, four u32 ranked IDs, four
u32 probability bits, two u32 selected-weight bits. Then u8 expert-present;
if present: field(u64 layer), field(u64 expert), field(u64 rank), field(u64 slot),
field(u64 generation), three field(weight SHA ASCII hex) in gate/up/down order.
Finally u8 greedy-present and, if present, u32 greedy ID. Hash excludes its own
sha256 field. Hash context includes original input and committed-state binding.

Token-sequence preimage starts `MER2-G3-TOKENS-v1` + NUL, field(version),
field(request ASCII), field(u64 nonce), field(u64 count), then each field(u32
input ID). It includes processed inputs only, not unprocessed greedy IDs.
KV snapshot uses `MER2-G3-CPU-KV-v1` + NUL, field(request), field(u64 nonce),
field(model hash), field(source hash), u64 generation; each layer contributes u64
length then all K and V f32 bits in head/position/dimension order, including
zero-initialized unused positions. End: u64 input count followed by u32 IDs.
Output commitment uses `MER2-G3-TOKEN-OUTPUT-v1` + NUL, Ticket encoding, u64 stage
count, each field(recomputed stage hash hex) and field(recorded stage hash hex),
32 u32 logits bits, u32 next ID, field(token hash), field(committed KV hash).
Source commitment uses `MER2-G3-SOURCE-RECEIPT-v1` + NUL, u16-length base ASCII,
then ASCII-sorted pairs of u16-length filename and u16-length SHA hex.

No runtime output or receipt golden is frozen here. D2 must independently review
these serialization choices and source, reproduce input hashes, and capture
full raw bits for A starting 3 and B starting 17 for three steps at 0/1/2. p3
acceptance/p>=4 rejection is separate. The authored discrimination test demands
pairwise-distinct six full-logit vectors and at least one A/B selected-route
difference; wrong expert/layer/order scenarios also use metadata-bound fault
injection to ensure numerical coincidence cannot authorize a malformed receipt.

## Execution boundary

D1 allows one `cargo generate-lockfile --offline` only, direct rustfmt on the
five new Rust files, JSON syntax, source inspection and diff/scope checks.
No Cargo build/check/test/run/clippy or compiled executable is allowed in D1.
The authored CLI has exactly `preflight-cpu` and `cuda-unavailable` modes; the
latter refuses before fixture/model computation. The former will compute six
synthetic CPU token steps and emit full traces only when separately authorized.
It cannot confer independent qualification merely by returning a report.

The 35 integration tests are authored and unexecuted (static source count only).
Analytical constants cover signed Q4 and a constant-matrix SwiGLU chain,
ordered matvec/dot, RMSNorm, softmax/router ties, interleaved RoPE, causal/head
attention, SiLU and greedy. Structural tests cover request/layer isolation,
three-step replay/discrimination, fourth-position limit, input-vs-next placement,
source/tensor identity, every stage cancellation/error, stale KV/slots/output,
wrong prediction/rank/expert/layer, and mutated/missing/duplicate receipts.
Their full-token comparisons are replay/lifecycle checks, not invented goldens.
D2 requires new authority freezing an exact test count/command and one attempt
with locked/offline/no-default-features before any execution. No D2 action,
commit, push or PR is authorized by this source candidate.

The offline lock resolution completed once without compilation. The lock has 25
registry packages; 24 archive checksums are locally cached and match. The sole
uncached archive is crunchy 0.2.4, already pinned upstream and selected only for
half's `cfg(target_arch = "spirv")` dependency (not the prospective Apple arm64
CPU target). The archive manifest was read directly to establish this distinction;
no broad Cargo metadata/build probe or dependency fetch was run. D2 remains
responsible for its exact target dependency preflight under new authority.
