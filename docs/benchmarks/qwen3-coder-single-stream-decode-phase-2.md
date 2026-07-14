# Qwen3-Coder single-stream decode: Phase 2 negative result

## Final status

Phase 2 activated an in-flight resident handoff mechanism intended to let a
demand request consume the immutable resident produced by an overlapping
foreground or speculative expert read. The Linux target-host run confirmed
that the mechanism was active: it avoided 454 duplicate physical reads in the
1,536-short case and 424 in 1,536-medium, or 878 in total.

That reduction did not produce a performance benefit. SSD traffic changed
negligibly, streaming throughput did not improve, and the all-resident control
regressed. The Phase 2B resident-path isolation attempt also failed its gate,
measuring 9.33% below the immediate Phase 1 resident reference. The handoff
runtime has therefore been removed. This document retains the experiment as a
negative result; no performance win is claimed.

The production runtime is restored to the qualified Phase 1 implementation at
commit `327d263193c48f9dde9f6a716562260ab49fa7ef` for every file changed by the
handoff experiment. This restores notify-only single-flight coordination and
the original counter/layout behavior. It also removes result publication,
direct resident handoff and demand-promotion state, handoff-specific retry
outcomes and ownership changes, separate handoff counters, registry guards,
runtime telemetry fields, and tests specific to that removed implementation.

The rollback preserves Phase 1 direct-Q8 zero-copy inference, observability,
strict inference, output parity, expert-cache behavior, primary/shadow buffer
ownership, and qualification semantics. No dead or disabled copy of the
handoff runtime remains behind a runtime conditional or feature.

## Evidence

The qualified Linux target-host A-B-A comparison was:

| Candidate | 1,536 short | 1,536 medium | Streaming GM | 6,144 short | 6,144 medium | Resident GM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 initial | 1.3000562561 | 1.5165972824 | 1.404158746356 | 3.7349088600 | 3.5123644733 | 3.621927827950 |
| Phase 2 handoff | 1.2877193867 | 1.4804354696 | 1.380719180341 | 3.1906553312 | 3.0339322511 | 3.111307138724 |
| Phase 1 immediate recheck | 1.2944549612 | 1.5184726960 | 1.401996617251 | 3.4921515988 | 3.5081547663 | 3.500144036461 |

Relative to the immediate Phase 1 recheck, Phase 2 was about 1.5% slower on
the streaming geometric mean and about 11% slower on the resident geometric
mean. The 878 avoided reads did not materially reduce bytes read from SSD, so
they did not improve the measured streaming rate. Both resident cases had zero
SSD bytes, cache misses, foreground reads, speculative reads, and handoff
joins; the resident regression was therefore not caused by I/O.

Phase 2B moved miss/handoff state away from the resident fast path but did not
recover the control:

| Phase 2B resident case | Decode TPS |
| --- | ---: |
| 6,144 short | 3.2589499509 |
| 6,144 medium | 3.0904253666 |
| Geometric mean | 3.173569220438 |

The Phase 2B geometric mean was `-9.330325%` relative to the Phase 1 reference
of `3.500144036461`. Output parity passed and the run recorded zero SSD bytes,
cache misses, foreground reads, speculative reads, and handoff joins, but the
performance gate failed. Further incremental adjustment of the handoff
implementation was rejected in favor of removing it.

## Collector behavior retained

The original four-case collector remains the qualification path and retains
its Phase 1 case validation and output. The `--resident-only` mode remains
available as a faster control gate and runs only 6,144-short and 6,144-medium.
It compares their decode-TPS geometric mean with the immediate Phase 1
reference using a +/-2% tolerance.

Because the production handoff runtime and its telemetry fields are gone, a
resident-only Phase 1 report has no handoff counter to sample. The resident
summary records `handoff_runtime_present: false`; zero handoff activity is by
construction. It still requires output parity, zero cache misses, zero SSD
bytes, zero foreground expert I/O, and zero speculative prefetch I/O.

After both cases complete, the collector always writes:

- `resident-control-summary.json`;
- `qualification.json`;
- `artifact-sha256.txt`;
- both raw reports and case summaries; and
- the normal build, configuration, provenance, storage, prompt, timing, and
  environment artifacts.

When the performance tolerance fails, both JSON finalization files contain
`qualification_passed: false` and an explicit failure reason. The collector
hashes the complete artifact and only then exits nonzero, so
`sha256sum -c artifact-sha256.txt` remains valid. A missing, empty, or invalid
case is instead reported as `INCOMPLETE`; it is not converted into a completed
gate failure and does not receive final summary or qualification files.

## Validation policy

macOS compilation and tests are correctness evidence only. Runtime and
performance decisions target Linux x86_64. Do not claim throughput recovery,
production-feature compilation, or Linux `io_uring` validation from a Mac
result.

Local correctness validation includes the library check, complete test suite,
the macOS-supported release build, shell syntax checks, resident finalization
tests for pass/fail/incomplete outcomes, and `git diff --check`. The resident
finalization test verifies that a complete tolerance failure returns nonzero
only after writing both JSON files and that the resulting checksum manifest
verifies successfully.

## Exact Linux resident-only validation

Run this on the 32-logical-CPU Linux x86_64 target VM. The model must be the
converted Qwen directory at `/mnt/localssd/data/qwen3-coder-q8`, and the
repository checkout must be `$HOME/mer-prompt2`.

```bash
cd "$HOME/mer-prompt2"
git fetch origin perf/qwen3-coder-single-stream-decode
git switch perf/qwen3-coder-single-stream-decode
git pull --ff-only origin perf/qwen3-coder-single-stream-decode
git status --short
git rev-parse HEAD

unset RAYON_NUM_THREADS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MER_QWEN_CONVERTED_DIR=/mnt/localssd/data/qwen3-coder-q8
export MER_QWEN_TOKENIZER=/mnt/localssd/data/qwen3-coder-q8/tokenizer.json
export MER_EXPECTED_NVME_MOUNT=/mnt/localssd

ARTIFACT_DIR="/mnt/localssd/benchmarks/prompt2-phase2c-resident-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
set +e
scripts/collect_qwen3_coder_prompt2_baseline.sh "$ARTIFACT_DIR" --resident-only
COLLECTOR_STATUS=$?
set -e

test -s "$ARTIFACT_DIR/baseline-6144-short.json"
test -s "$ARTIFACT_DIR/baseline-6144-medium.json"
test -s "$ARTIFACT_DIR/resident-control-summary.json"
test -s "$ARTIFACT_DIR/qualification.json"
test -s "$ARTIFACT_DIR/artifact-sha256.txt"
jq . "$ARTIFACT_DIR/resident-control-summary.json"
jq . "$ARTIFACT_DIR/qualification.json"
sha256sum -c "$ARTIFACT_DIR/artifact-sha256.txt"
exit "$COLLECTOR_STATUS"
```

A zero status means every resident gate passed. Status 1 with all five final
files present is an auditable completed gate failure; inspect
`failure_reasons`. Status 2 or missing case/final files is an incomplete run
and must not be interpreted as a performance result.

Only after the resident-only gate passes should the unchanged four-case
collector be run:

```bash
cd "$HOME/mer-prompt2"
FULL_ARTIFACT_DIR="/mnt/localssd/benchmarks/prompt2-phase2c-full-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/collect_qwen3_coder_prompt2_baseline.sh "$FULL_ARTIFACT_DIR"
jq . "$FULL_ARTIFACT_DIR/four-case-summary.json"
jq . "$FULL_ARTIFACT_DIR/qualification.json"
sha256sum -c "$FULL_ARTIFACT_DIR/artifact-sha256.txt"
```

Do not change the prompts, output-token count, warmup or measured-run counts,
cache sizes, CPU mask, Rayon worker count, storage layout, or predictive-policy
switches when comparing with the Phase 1 evidence.
