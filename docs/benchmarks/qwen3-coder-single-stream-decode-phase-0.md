# Qwen3-Coder single-stream decode: Phase 0 audit

## Status and boundary

Phase 0 was performed from `322aababb2f9ff513a91c783244b5feb7fbd27e7`
(the merge commit for PR #136) on branch
`perf/qwen3-coder-single-stream-decode`.

This is a macOS development audit, not a performance result. No Prompt 2
target-host baseline JSON was supplied or found in the available workspaces.
The historical July 11 Qwen report used boot-disk-backed storage and is context
only; it is not a qualifying before-result for the local-NVMe milestone.

No optimization workstream has been selected and no performance-sensitive
runtime code has been changed. Phase 1 must run on the specified Linux x86-64
`g2-standard-32` host before optimization begins.

## What exists at the post-PR-136 starting point

### Real-model timing and `bench-real`

`stage_timing.rs` defines timers for embedding, RMS norm, Q/K/V/O projections,
RoPE, attention score/value work, router, cache lookup, foreground expert I/O,
Q8 preparation, Q8 gate/up and down kernels, expert compute, weighted
combination, final norm, LM head, sampling, scheduler overhead, and total
prompt/decode wall time.

`bench-real` already emits a single JSON document containing:

* Git commit (short form), build features, worker count, and dense backend;
* prompt text, warmup/measured counts, reset policy, and greedy flag;
* prompt/completion counts and generated token IDs per run;
* prompt TPS, decode TPS, TTFT, total time, and p50/p95/p99/max decode latency;
* cache hits/misses/hit rate, SSD bytes, foreground SSD stall, and current RSS;
* detailed stage-timing snapshots; and
* within-suite generated-token parity.

The real benchmark fails closed if required weights remain seeded or if the
measured window uses the legacy non-finite attention fallback. Model startup
logs strict/loaded/required tensor status.

### Rayon selection and autotuning

The process-wide pool is initialized once, after affinity placement and before
runtime construction. The current resolver's implemented precedence is:

1. `RAYON_NUM_THREADS`;
2. global `--rayon-threads`;
3. `performance.rayon_threads`;
4. startup autotune selection;
5. reusable profile; and
6. the reserved-headroom default.

Prompt 2 requires reusable matching real-model profiles to precede a new
real-model autotune selection. That is a deliberate Phase 2 change; the audit
must not describe the current synthetic ordering as already compliant.

The current autotuner applies only to synthetic `run`. It already uses a
parent/isolated-child design with a recursion guard, coarse/fine candidates,
repeat observations, p50/p95/p99, confidence, rejection reasons, atomic
profile writes, stale key rejection, and low-confidence opt-in. Its fixed
80/120 ms slow thresholds and synthetic `sustained_tps` ranking are not valid
for real Qwen decode. The profile store and profile payload have no explicit
schema version. `bench-real` has neither real probes nor a real-model profile
identity.

### Dense tensors and Qwen attention

The real path keeps F32 or native Q8_0 dense tensors behind `DenseWeight`.
Runtime dense backend selection supports `matrixmultiply`, Rayon row-parallel,
and Rayon-chunked matrixmultiply. Q8 embeddings perform row lookup and Q8 LM
head greedy/top-k scans operate without materializing a vocab-sized F32
vector.

Qwen geometry is preserved: Q is 4096 wide, K/V are 512 wide, residual width
is 2048, and the implementation does not require `head_dim * num_heads ==
d_model`. The production CPU attention path reuses request scratch for Q/K/V,
scores and output; preserves QK-Norm then RoPE order; maps 32 query heads onto
4 KV heads; uses paged KV and checked strict softmax; and currently performs
separate Q, K, and V projection dispatches plus scalar QK and weighted-V inner
loops.

### LM head and direct-Q8 experts

The Q8 LM head uses deterministic per-worker greedy/top-k candidate reduction
with tie handling and reference tests.

Production Q8 experts execute directly from `ExpertResident::data()`. The path
has fused gate/up traversal, direct down projection, scalar fallback,
runtime-checked AVX2+FMA, optional runtime-checked AVX-512F+BW, and the
post-PR-136 conservative auto policy (AVX2+FMA when qualified, otherwise
scalar). AVX-512 is explicit rather than automatically preferred. Candle
QStorage is retained only behind `q8-candle-reference`. Memory telemetry
reports resident, primary/shadow pool, and prepared duplicate bytes; tests
require production prepared duplicate bytes to remain zero.

### Cache and I/O telemetry

The engine retains Linux O_DIRECT, a per-layer bounded cache, a primary/shadow
buffer pool, foreground SSD stall, total bytes read, prefetch
completion/use/drop counters, pre-gate statistics, governor statistics, and
degraded/read-failure counters. Current `bench-real` exports only the cache
totals, aggregate SSD bytes/stall, and RSS subset of the richer `EngineReport`.

The optional `io_uring` implementation and fixed-buffer tests exist, but the
production engine does not currently use it for expert reads. Even synthetic
`run --io-uring` constructs it only as a startup probe and logs that the
generate loop continues through portable `pread`; `bench-real` has no runtime
selector and calls `NvmeStorage::read_expert`. Building with the feature is
therefore not proof of active `io_uring`. This contradicts the milestone's
assumed existing behavior and must be resolved explicitly before Linux I/O
qualification; no baseline report may label its storage backend `io_uring`
until the actual read path proves it.

## Missing gates before a qualifying Phase 1 baseline

The present JSON is valid but does **not** satisfy the Prompt 2 artifact
contract. Before accepting a baseline, add and test the following
instrumentation without changing the math or scheduling policy:

* a versioned `bench-real` schema;
* full commit SHA and dirty-worktree state;
* build profile, OS/architecture, CPU vendor/model/features, requested and
  effective affinity, logical CPUs, actual Rayon pool size, and selection
  source;
* model/checkpoint and converted-manifest fingerprints, model dimensions,
  dense/expert dtype, storage/direct-I/O identity, and direct-Q8 selected
  backend;
* truthful active-I/O-backend reporting and an explicit decision on the
  currently unwired `io_uring` engine path;
* prompt identifier and checksum;
* strict loaded/required counts, seeded-fallback state, inference policy,
  degraded substitutions, read failures, Q8 scalar-layout fallbacks,
  prepared duplicate bytes, and truncated/non-finite counters;
* steady-state RSS semantics (rather than one unlabeled current sample) plus
  externally collected peak RSS;
* speculative, useful, wasted, and foreground SSD byte attribution;
* the richer cache/prefetch telemetry already present in `EngineReport`; and
* non-overlapping prompt/decode critical-path categories with an explicit
  unattributed residual and a coverage ratio.

The detailed stage totals are nested and sometimes cumulative across experts
or workers. They must remain diagnostic fields and must not be added together
as wall-time attribution. No optimization may be chosen until a separate
critical-path layer accounts for at least 95% of prompt and decode wall time
and identifies the dominant stages independently for 1,536 and 6,144 slots.

The existing matvec microbenchmark uses generated F32 weights, reports only
best/mean time, and omits the Q8 expert gate/up and down shapes. It therefore
does not satisfy the Prompt 2 native-Q8 Qwen-shape benchmark contract.

## Checked-in target inputs

Immutable request JSON fixtures are under
`benchmarks/qwen3-coder-single-stream/prompts`. `jq -j .prompt` yields the
exact prompt bytes passed by `bench-real`; the collector records SHA-256 for
those bytes. Both requests specify 128 output tokens.

The expected prompt hashes are:

* short: `5a53db2f1afb97647c4c0b14f2f6cc49bd1ba3fdac5e6e37e33eee601aedeb23`;
* medium: `243005bad7a1aacf9c318dbe917cf89592ae7bd83265d5de1780d239d9266d95`.

`qwen3-coder-q8.toml.in` fixes the CPU-only Qwen geometry, strict policies,
Q8 expert size, deterministic sampling, disabled experimental predictors,
O_DIRECT, and all settings shared by the 1,536- and 6,144-slot controls. The
collector substitutes only the model path, tokenizer path, and cache size.

For the current pool geometry (`expert_size=5,017,600`, fanout 2, pipeline
depth 3):

| cache slots | primary allocation | shadow allocation | total expert pool |
|---:|---:|---:|---:|
| 1,536 | 7,712,051,200 B (7.18 GiB) | 30,105,600 B (28.71 MiB) | 7,742,156,800 B (7.21 GiB) |
| 6,144 | 30,833,152,000 B (28.72 GiB) | 30,105,600 B (28.71 MiB) | 30,863,257,600 B (28.74 GiB) |

These are pool allocations, not whole-process RSS. The Linux preflight must
also record available memory and `/usr/bin/time -v` peak RSS. The 6,144-slot
configuration must not be silently reduced when it fits.

## Exact Linux baseline collection

From a clean commit containing only Phase 0/instrumentation changes:

```bash
cd /path/to/Micro-Expert-Router-SSD-Streamed-MoE-MER

unset RAYON_NUM_THREADS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

export MER_QWEN_CONVERTED_DIR=/mnt/localssd/models/qwen3-coder-30b-a3b-q8
export MER_QWEN_TOKENIZER="$MER_QWEN_CONVERTED_DIR/tokenizer.json"
export MER_EXPECTED_NVME_MOUNT=/mnt/localssd

scripts/collect_qwen3_coder_prompt2_baseline.sh \
  /mnt/localssd/benchmarks/prompt2-baseline-$(git rev-parse --short=12 HEAD)
```

The collector verifies the model is under the declared NVMe mount, records
`uname`, `lscpu`, `lsblk`, `findmnt`, `df -T`, memory, Git state, metadata and
prompt checksums, rendered configs, binary checksum, pool sizing, separate
JSON/stderr/time outputs, and `jq` syntax/parity checks for all four required
cache/prompt cases. It refuses non-Linux, non-x86-64, non-32-vCPU hosts and a
model filesystem mounted somewhere other than the declared local NVMe. It
passes global options before `bench-real` and fixes the requested placement at
30 Rayon workers on CPU mask `0-31`.

The collector's current-field validation is only a collection sanity check.
Do not call its output a qualifying Prompt 2 baseline until every missing JSON
and critical-path gate above is implemented and validated.

## Mac development checks

The following portable checks were run on the Phase 0 worktree:

* `cargo test --manifest-path rust-engine/Cargo.toml`: 782 passed, 0 failed,
  2 ignored;
* `cargo check --manifest-path rust-engine/Cargo.toml --all-targets`: passed
  with the repository's existing warning backlog;
* `bash -n scripts/collect_qwen3_coder_prompt2_baseline.sh`: passed;
* both request fixtures passed `jq` parsing and exact prompt SHA-256 checks;
* a rendered 1,536-slot config passed Python `tomllib` parsing; and
* `git diff --check`: passed.

The repository root has no `Cargo.toml`, so the brief's literal
`cargo fmt --all -- --check` cannot run there. The applicable command,
`cargo fmt --manifest-path rust-engine/Cargo.toml --all -- --check`, fails on
pre-existing rustfmt drift across many Rust files at the untouched PR #136
starting point. Phase 0 did not reformat or modify Rust source files.

## Phase gate

Work stops here on macOS. Required next input is the target-host preflight and
four valid baseline JSON artifacts produced from one clean instrumentation
commit on the specified local-NVMe Linux host. Only then may the 1,536- and
6,144-slot dominant critical paths select the smallest justified optimization
workstream.
