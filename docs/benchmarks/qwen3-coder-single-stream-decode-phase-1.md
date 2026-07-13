# Qwen3-Coder single-stream decode: Phase 1 observability

Phase 1 adds the qualification and critical-path evidence needed before any
Prompt 2 optimization is selected. It does not change inference math, routing,
cache or prefetch policy, CPU placement, storage configuration, prompts, or the
benchmark matrix. Mac results are correctness checks only; throughput evidence
must come from the specified Linux target.

## Raw pre-instrumentation target-host baseline

These measurements were collected from clean commit
`b0e6ba7efe0ca350c8f5ac5d55814ba3c8e47556` on a GCP `g2-standard-32` with 32
logical CPUs, Rayon 30, CPU mask `0-31`, and the converted Qwen3-Coder Q8 model
on `/dev/nvme0n1` mounted as ext4 at `/mnt/localssd` with
`rw,noatime,nodiratime`. Each case used strict greedy inference, one warmup,
five measured runs, 128 output tokens, and output-token parity.

| Case | Prompt tokens | Decode TPS | Prompt TPS | TTFT p50 | Mean decode | Mean total | Hit rate | Misses (5 runs) | SSD bytes (5 runs) | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,536 short | 14 | 1.2854 | 0.9214 | 15.250 s | 98.805 s | 114.037 s | 73.95% | 70,521 | 409.070 GB | 13.59 GiB |
| 1,536 medium | 65 | 1.4687 | 1.6963 | 38.337 s | 86.473 s | 124.831 s | 80.28% | 72,693 | 420.585 GB | 13.59 GiB |
| 6,144 short | 14 | 3.1244 | 3.8714 | 3.655 s | 40.648 s | 44.304 s | 100% | 0 | 0 measured | 35.14 GiB |
| 6,144 medium | 65 | 2.8859 | 3.6139 | 18.079 s | 44.008 s | 62.034 s | 100% | 0 | 0 measured | 35.14 GiB |

The 1,536-slot configuration is the primary realistic SSD-streaming baseline.
The 6,144-slot configuration is an all-resident-capacity control after warmup;
its approximately 3 TPS is neither SSD-streaming throughput nor an optimization
result. The older approximately 0.599 TPS boot-disk result is historical
context, not the same-machine comparison baseline. The raw primary geometric
mean is 1.3740 decode tokens/s; the resident-control geometric mean is about
3.0028 decode tokens/s. Fresh instrumented results supersede these raw values
for optimization deltas.

The source archive remains on the validation VM at
`/home/randyap8/benchmarks/prompt2-baseline-b0e6ba7-20260713T020959Z.tar.gz`.
Phase 1 does not depend on access to it.

## Versioned `bench-real` schema

JSON reports now carry:

```json
"schema": { "name": "mer-bench-real", "version": 2 }
```

Version 2 is the first qualifying Prompt 2 schema. In addition to the legacy
aggregate and run metrics, it defines these stable top-level groups:

- `execution`: full Git SHA, dirty state at execution, profile, features, host
  and CPU identity, instruction features, requested/effective CPU placement,
  requested/actual Rayon sizing and source, dense backend, and selected native
  direct-Q8 backend;
- `model` and `prompt_identity`: checkpoint/config/manifest hashes, resolved
  architecture and dimensions, dtypes, immutable fixture identity and exact
  prompt hash, and requested completion count;
- `storage`: the active expert-read backend, direct-I/O state, and separate
  compiled/active `io_uring` states;
- `strictness`, `predictive_policy`, and `memory_layout`: resolved fail-closed
  state, explicit experimental-policy switches, expert-pool allocation, and
  duplicate-representation state;
- per-run `correctness`, `cache_io`, `memory`, `critical_path`, and
  `diagnostic_stage_timings`.

The checked-in parser/serialization tests lock the schema name/version and the
local SHA-256 implementation against standard vectors. Schema evolution must
increment the version when field meaning or qualification semantics change.

`nonfinite_output_count` is currently `null`: no zero-overhead output-wide
non-finite scan exists in this path. The field is explicit so absence is not
mistaken for a measured zero. Internal RSS is a current sample taken after
completion decode and before report serialization. It is not peak RSS. Peak RSS
comes from the collector's `/usr/bin/time -v` process measurement and is stored
in each machine-readable case summary.

## Truthful expert storage and Q8 execution

The production expert reader is `pread` with `O_DIRECT`, reported as
`active_expert_io_backend = "pread-odirect"`. A build may compile the
`io_uring` feature while the real expert read path does not use it; therefore
`io_uring_compiled = true` and `io_uring_active_for_expert_reads = false` are
both expected. Phase 1 does not wire `io_uring` into production.

Q8 experts continue to execute directly over `ExpertResident::data()` through
the production native backend. The target collector requires the conservative
AVX2/FMA selection, positive direct-Q8 dispatch counts, zero scalar-layout
fallbacks, and zero persistent prepared-duplicate bytes. The retained Candle
reference is a build/test feature, not the production representation.

## Critical-path methodology

Prompt and decode have independent timer accumulators and independent wall
durations. Attribution uses only sequential request-path wall slices. Parallel
work is represented by the enclosing stage wall duration; worker-cumulative
times are never added. In particular, nested Q8 preparation/gate-up/down kernel
diagnostics remain in `diagnostic_stage_timings` and are excluded from additive
attribution.

The exclusive categories are embedding/input preparation, normalization
(including final normalization), Q/K/V projections, RoPE and attention,
attention output projection, routing, expert-cache coordination, foreground
expert-I/O await, expert compute, weighted combination, LM-head evaluation,
sampling, and scheduler/runtime overhead. Foreground I/O attribution starts
after cache lookup and miss-task submission; the older broader SSD-stall timer
remains diagnostic and is not added.

For each phase and run:

```text
attributed = sum(exclusive categories)
residual = wall - attributed
coverage = attributed / wall
```

The signed residual is never clamped. A material negative residual fails the
non-overlap invariant and exposes overlap. Qualification requires both that
invariant and at least 95% coverage for prompt and decode in every measured
run. Detailed legacy stage timers remain available but must not be summed as
wall-clock attribution.

Instrumentation adds timer reads and atomic counter updates. Fresh Phase 1
target-host runs are therefore the required comparison baseline for later code
changes; the raw table above is not silently mixed with instrumented results.

## Qualification contract

The Linux collector fails unless all of the following hold:

1. Linux x86-64, exactly 32 online logical CPUs, model below the declared local
   NVMe ext4 mount with `rw,noatime,nodiratime`, a machine-readable `findmnt`
   identity artifact, and separate stdout JSON, stderr, and GNU-time files.
2. Full report SHA equals the collected clean Git SHA; release profile;
   requested/effective mask `0-31`; requested/actual Rayon count 30 from CLI;
   required Cargo features; resolved dense backend; and native AVX2/FMA Q8.
3. Resolved Qwen3-MoE geometry and Q8 dtypes match the fixed target; report
   config/manifest hashes equal independently collected hashes.
4. Exact checked-in prompt SHA and token count (short 14, medium 65), requested
   and actual completion count 128, five equal token-output vectors, and
   aggregate parity.
5. Strict weights; nonzero required tensors with loaded equal required; no
   seeded fallback; every fail-open policy disabled; and zero degraded
   substitutions, read failures, truncated payload uses, non-finite attention
   fallbacks, Q8 scalar-layout fallbacks, and prepared duplicate bytes.
6. Actual `pread-odirect` expert reads; direct I/O enabled; `io_uring` compiled
   but inactive for expert reads; packed storage disabled.
7. Experimental locality, neural speculation, affinity, governor, cost-aware
   eviction, pre-gating, and static residency disabled. Baseline Markov
   prefetch fanout and pipeline depth remain unchanged and are reported.
8. Required cache, foreground I/O, prefetch, shadow-residency, pool-allocation,
   current-RSS, and sampling-point fields are present for all runs.
9. Every prompt and decode critical path passes the non-overlap invariant and
   at least 95% coverage.
10. External peak RSS parses to a positive byte count for every case, all four
    summaries exist, `qualification.json` says pass, and immutable artifacts
    have an SHA-256 manifest. The two still-open collector logs and the manifest
    itself are explicitly excluded from that manifest.

Passing produces `baseline-*.case-summary.json`, `four-case-summary.json`,
`qualification.json`, and `artifact-sha256.txt`. A missing field or failed gate
terminates collection before a pass manifest is created.

## Future optimization comparisons

The primary KPI is the geometric mean of short and medium decode TPS at 1,536
slots:

```text
GM = sqrt(short_decode_tps * medium_decode_tps)
case_delta_percent = 100 * (new_case_tps / baseline_case_tps - 1)
GM_delta_percent = 100 * (new_GM / baseline_GM - 1)
gap_closed_percent = 100 * (new_streaming_GM - baseline_streaming_GM)
                         / (baseline_resident_GM - baseline_streaming_GM)
```

Every future report must include absolute short/medium decode TPS, their
same-machine deltas, GM and GM delta, peak-RSS delta, cache misses/hit rate and
SSD-byte changes, TTFT changes, output parity, strictness/fallback status, and
streaming-to-resident gap closed. The original 1.5 TPS goal may be reported as
a minimum milestone, but it does not replace a reproducible same-machine
improvement. The resident control exceeding 3 TPS is never itself success.

Workstream selection happens only after qualified fresh reports show the
dominant stages: choose a small cache/I/O change only if foreground expert I/O
or cache coordination dominates 1,536 slots and disappears in the resident
control; choose a compute stage only if it dominates both; improve
instrumentation if evidence is mixed or coverage fails.

## Target VM collection

From the existing VM repository checkout:

```bash
cd /home/randyap8/Micro-Expert-Router-SSD-Streamed-MoE-MER
git fetch origin perf/qwen3-coder-single-stream-decode
git switch perf/qwen3-coder-single-stream-decode
git pull --ff-only origin perf/qwen3-coder-single-stream-decode
git status --short
git rev-parse HEAD

export MER_QWEN_CONVERTED_DIR=/mnt/localssd/models/qwen3-coder-30b-a3b-q8
export MER_EXPECTED_NVME_MOUNT=/mnt/localssd
export MER_QWEN_TOKENIZER="$MER_QWEN_CONVERTED_DIR/tokenizer.json"
ARTIFACT_DIR="/mnt/localssd/benchmarks/prompt2-phase1-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/collect_qwen3_coder_prompt2_baseline.sh "$ARTIFACT_DIR"
jq . "$ARTIFACT_DIR/qualification.json"
jq . "$ARTIFACT_DIR/four-case-summary.json"
sha256sum -c "$ARTIFACT_DIR/artifact-sha256.txt"
```

Adjust only `MER_QWEN_CONVERTED_DIR` if the existing local-NVMe checkpoint has
a different directory name. Do not change the mount, CPU, prompt, sampling,
cache, or run-count contract.
