# Qwen3-Coder single-stream decode: Phase 4A prefetch ablation

## Purpose

Phase 4A is a benchmark-harness-only causal configuration ablation. It does
not add or change a runtime policy. All three collections use one exact commit
and the existing runtime behavior; only `predict_fanout` and `pipeline_depth`
change. The no-prefetch fixture uses the runtime's existing
`predict_fanout = 0` configuration to disable predictive prefetching fully.

The qualified performance target is Linux x86_64 on GCP `g2-standard-32`,
with exactly 32 online logical CPUs, CPU mask `0-31`, 30 Rayon workers, and
local ext4 NVMe mounted at `/mnt/localssd` with
`rw,noatime,nodiratime`. macOS fixtures validate artifact logic only and do not
support Linux or NVMe performance claims.

## A-B-A matrix

Run the collections sequentially on the same host and exact commit, without
rebuilding or changing the model, binary, prompts, mount, CPU identity, or
Cargo feature set between them.

| Order | Fixture | `MER_PROMPT2_PREDICT_FANOUT` | `MER_PROMPT2_PIPELINE_DEPTH` | Expected prefetch state |
| ---: | --- | ---: | ---: | --- |
| 1 | Control A1 | 2 | 3 | enabled |
| 2 | No-prefetch B | 0 | 1 | disabled; every prefetch counter and prompt/decode speculative-overlap count must be zero |
| 3 | Control A2 | 2 | 3 | enabled |

The collector defaults remain `2/3`, so an unset pair produces the same
qualified Prompt 2 configuration used before Phase 4A. An explicit run looks
like:

```bash
MER_PROMPT2_PREDICT_FANOUT=2 MER_PROMPT2_PIPELINE_DEPTH=3 \
  scripts/collect_qwen3_coder_prompt2_baseline.sh /path/to/control-a1 four-case

MER_PROMPT2_PREDICT_FANOUT=0 MER_PROMPT2_PIPELINE_DEPTH=1 \
  scripts/collect_qwen3_coder_prompt2_baseline.sh /path/to/no-prefetch-b four-case

MER_PROMPT2_PREDICT_FANOUT=2 MER_PROMPT2_PIPELINE_DEPTH=3 \
  scripts/collect_qwen3_coder_prompt2_baseline.sh /path/to/control-a2 four-case
```

Each directory contains `ablation-provenance.json`, which records the
configuration and expected prefetch state, exact commit, model and prompt
hashes, hostname, CPU identity, model-mount identity, Cargo features, and
binary hash. The ordinary strict loading, output parity, critical-path
coverage, O_DIRECT, CPU, checkpoint, prompt, storage-backend, and memory-layout
gates remain mandatory.

## Read-only comparison

Compare the three immutable collections in A-B-A order:

```bash
scripts/compare_qwen3_coder_prompt2_prefetch_ablation.sh \
  /path/to/control-a1 \
  /path/to/no-prefetch-b \
  /path/to/control-a2 \
  > /path/to/phase4a-prefetch-ablation.json
```

The comparison first requires matching hostname, exact commit, model hashes,
prompt hashes, binary hash, Cargo features, CPU identity, and qualified mount
identity. It also requires the exact `2/3`, `0/1`, `2/3` matrix and passing
streaming qualification and output-parity gates. A1-to-A2 short decode TPS,
medium decode TPS, and streaming geometric-mean TPS must each remain within
±2%.

B is compared with the arithmetic mean of A1 and A2 for every short and
medium metric. For streaming geometric-mean decode TPS, the control reference
is the arithmetic mean of the A1 and A2 geometric means. A latency family is
considered consistently improved only when its mean delta is negative in both
the short and medium fixtures.

## Decision rules

The explicit thresholds are:

- stable controls: absolute A2-versus-A1 delta no greater than `2%`;
- throughput-neutral lower bound: `-0.5%` for B streaming geometric-mean TPS
  versus the control mean; and
- material throughput regression: `-2%` or worse.

The result is classified as follows:

- `prefetch_net_negative_or_unnecessary`: controls, provenance,
  qualifications, parity, and no-prefetch invariants pass; B geometric-mean
  TPS is at least `-0.5%`; and mean demand-storage latency or mean layer-fetch
  critical path improves in both fixtures.
- `prefetch_helpful_but_contentious`: the same gates pass; B geometric-mean
  TPS regresses by at least `2%`; and mean demand-storage latency or mean
  layer-fetch critical path improves in both fixtures.
- `prefetch_not_primary_bottleneck`: the same gates pass; B geometric-mean TPS
  regresses by at least `2%`; and neither mean latency family improves in both
  fixtures.
- `inconclusive`: controls are unstable, a provenance/qualification/parity or
  no-prefetch invariant fails, or the measurements do not satisfy another
  category.

These classifications are specific to the Phase 4A configuration ablation and
do not reuse the rejected Phase 3B `policy_accepted` semantics.
