# Qwen3-Coder single-stream decode: Phase 4B routing trace and oracle

## Scope and goal

Phase 4B adds opt-in diagnostic evidence for Prompt 2. It does not optimize
the predictor or change runtime policy. The project objective remains correct,
reproducible single-stream decode while SSD is essential and no more than half
of expert weights are resident. The long-term north star is at least 10 TPS on
very large MoE models under that partial-residency constraint.

The Qwen3-Coder workload has 48 routed layers, 128 experts per layer, top-8
routing, and 6,144 layer-qualified experts. The intended SSD-streamed case has
1,536 expert slots (32 per layer, 25% of expert slots); the 6,144-slot resident
case is a diagnostic compute reference, not a valid deployment architecture.

Phase 4A found that the current fanout-2/depth-3 prefetch configuration was net
negative for this workload: the qualified no-prefetch treatment was about
10.33% faster, transferred about 12.5% fewer SSD bytes, reduced mean demand
storage latency by about 31%, and reduced mean layer-fetch critical-path
latency by about 15-16%. This establishes that the current implementation is
harmful; it does not establish that timely predictive prefetch is impossible.

The preceding audit found several plausible mechanisms: Markov prediction is
one layer ahead even when `pipeline_depth` is 3; depth sizes the shadow pool;
cold prior fill can produce low global IDs in the wrong layer; inference uses
the lowest-ranked `target.last()` context; filtering occurs after fanout
truncation without backfill; equal-score ordering can vary between processes;
history is engine-global; demand and speculation share synchronous positional
`pread` without demand priority; prefetched residents enter the normal
per-layer LRU; and all eight experts must be available before expert compute
begins. Phase 4B observes these mechanisms without correcting them.

## Enablement and bounds

Tracing is disabled by default. Enable one bounded JSONL trace for a direct
`bench-real` invocation with:

```bash
MER_PROMPT2_PHASE4B_TRACE_PATH=/path/to/trace.jsonl \
MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS=1000000 \
  rust-engine/target/release/micro-expert-router bench-real ...
```

`MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS` must be a positive integer and defaults
to 1,000,000 when a path is set. The writer reserves the last allowed record
for an explicit `trace_truncated` event. Later events are dropped and counted.
Detailed diagnostic state stops accepting new high-cardinality entries after
truncation, while already tracked in-flight lifecycles may finish so sampled
arithmetic remains reconcilable. No events, JSON values, writer lock, or
diagnostic maps exist when the trace path is unset; the engine performs only
an `Option` check at observation sites.

Use the collector's explicit `phase4b-diagnostic` mode with a trace path
template containing `{case}`:

```bash
MER_PROMPT2_PHASE4B_TRACE_PATH=/traces/{case}.jsonl \
MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS=2000000 \
  scripts/collect_qwen3_coder_prompt2_baseline.sh \
    /artifacts/phase4b-diagnostic phase4b-diagnostic
```

The mode initially collects only the 1,536-slot short and medium cases. Each
case gets a separate bounded trace. It does not collect the 6,144-slot resident
compute references because full residency is not the intended SSD-streamed
architecture.

Diagnostic qualification retains the ordinary provenance, platform, release
feature, model/tokenizer/configuration/prompt identity, strictness, correctness,
output parity, run/token count, partial-residency memory layout, O_DIRECT,
prefetch-policy, trace integrity, and lifecycle-reconciliation gates. It does
not require 95% production critical-path coverage or the existing per-stage
critical-path qualification flags to be true. Synchronous per-event JSONL
serialization and writing adds wall time outside the existing production stage
categories, so those gates do not establish diagnostic validity.

Diagnostic case and collection summaries explicitly report
`qualification_kind: "phase4b-diagnostic"`,
`diagnostic_qualification_passed`, and
`performance_qualification_applicable: false`. They retain the observed prompt
and decode coverage values, set the ordinary `qualification_passed` performance
field to `false`, and explain why performance qualification is not applicable.
A trace-enabled invocation in `four-case` or `resident-only` mode is rejected;
it cannot be mislabeled as a qualified performance baseline.

With tracing disabled, `four-case` and `resident-only` keep the unchanged
Phase 4A production qualification, including the 95% prompt/decode coverage
threshold and every existing critical-path, performance, correctness,
provenance, memory, and output-parity gate.

The first full 1,536-slot short diagnostic produced approximately 704 MB and
1.37 million events over roughly 22 minutes. Its trace integrity, lifecycle
reconciliation, replay, and output-parity gates passed, while observed
critical-path coverage fell as low as about 22% for prompt and 17% for decode.
Traced TPS and coverage are therefore not directly comparable with untraced
Phase 4A performance results.

Trace creation errors stop the diagnostic command before inference begins.
Write failures are latched, increase the dropped-event count, and do not alter
model output. The benchmark reports `trace_write_failed`; an enabled
diagnostic qualification requires a positive event count, zero dropped events,
no truncation, no writer failure, and reconciled top-level and per-run
lifecycles. It rejects a partial trace rather than treating it as complete
evidence. The trace contains fixture identifiers, numeric token/routing
identity, scores, lifecycle state, and timing. It never contains model tensor
contents or prompt text. Use a fixture ID or prompt hash rather than placing
prompt content in a trace path.

## JSONL schema

Every line has this versioned envelope:

```json
{
  "schema_name": "mer-prompt2-phase4b-routing-trace",
  "schema_version": 1,
  "event_id": 42,
  "event_type": "initial_demand_lookup",
  "monotonic_timestamp_ns": 123456789,
  "payload": {}
}
```

All timestamps come from the collector's single monotonic clock. Event IDs,
request IDs, lookup IDs, read IDs, and lifecycle IDs are stable within one
trace. A `trace_truncated` envelope carries its fields at top level because it
is the terminal boundedness marker.

Version 1 event types are:

- `request_begin`, `request_end`: fixture ID, repetition, measured/warmup
  status, request and stream identity.
- `routing`: token/layer identity, ordered local and global top-8 IDs, ordered
  gate weights, routing time, and the request identity attached to Markov
  history.
- `prediction_candidate`: source and expected target layer, context, arm,
  rank, score, decoded layer/local identity, current-target, residency, and
  global-in-flight facts.
- `prediction_candidate_list`: ordered arm candidates and equal-score
  diagnostic context.
- `lifecycle_transition`: filtering, admission, task/singleflight, publication,
  first-use, and eviction transitions.
- `physical_read_issued`, `physical_read_completed`,
  `physical_read_failed`: demand/speculative classification, expert and byte
  counts, service timestamps, and demand/speculative concurrency at issue.
- `initial_demand_lookup`: the exactly-one lookup class and any known
  speculative issue, completion, and publication times.
- `demand_lookup_timing`: the reconciled lookup, demand-start, availability,
  consumption, and layer-compute-start timestamps once the layer fetch barrier
  opens.
- `lookup_singleflight_follower`: a demand lookup joined the existing
  same-expert physical read.
- `layer_fetch_complete`: ordered routed experts, initial hit/miss sets,
  speculative joins, ordinary demand reads, per-expert availability,
  straggler source, lookup start, fetch completion, and compute start.
- `layer_compute_complete`: layer compute completion.
- `trace_truncated`: terminal indicator that the maximum event count was
  reached.

Predictor arms are `first_order_markov`, `second_order_markov`,
`fallback_prior_fill`, `affinity`, `neural_speculator`, `locality`, `combined`,
and `other`. The Prompt 2 Markov path reports first-order, second-order,
fallback, and the final combined list separately. Candidate order is never
normalized by the trace.

Initial demand lookup classes are mutually exclusive:

- `ordinary_resident`;
- `ready_prefetched_resident`;
- `speculative_read_in_flight`;
- `ordinary_miss`.

An in-flight join is not a ready hit. Later availability, consumption, and
compute-start timestamps are linked by lookup, request, token, layer, slot,
and expert identity.

## Lifecycle and reconciliation

A candidate can be filtered, rejected, admitted, find a cache race, follow an
existing singleflight leader, or become the leader that issues a speculative
physical read. A completed leader attempts publication. A published expert can
be first-used, evicted before first use, or remain resident and unused at the
sample. A completed read whose publication has not yet resolved is an explicit
sample state.

The report validates:

```text
physical_read_issued =
    physical_read_completed
  + physical_read_failed
  + physical_read_inflight_at_sample

physical_read_completed =
    published
  + publication_rejected
  + completion_not_yet_published_at_sample

published =
    first_use
  + evicted_before_first_use
  + still_resident_unused_at_sample
```

These are speculative lifecycle counters; demand reads are reported in the
read-overlap section. Impossible diagnostic arithmetic makes an
instrumented collector case fail qualification.

Terminal usefulness separates ready-before-lookup, in-flight completion
before unrelated misses, in-flight final straggler, in-flight completion after
another demand read, publication followed by eviction, never requested,
publication rejection, read failure, and still-in-flight-at-sample states.
The live runtime does not convert observed overlap into claimed saved latency.

## Benchmark report

The ordinary report always adds:

```json
{"phase4b_trace_enabled": false}
```

When enabled, top-level `phase4b_trace` and each run's
`phase4b_diagnostics` contain schema/path/boundedness information, lifecycle
arithmetic, initial lookup counts, terminal timeliness, layer critical-path
attribution, per-arm precision/recall and rank distributions, read overlap,
active speculative reads observed at demand issue, and request-boundary
contamination. Run snapshots are cumulative through that run because
speculative tasks can legally cross sampling boundaries; this preserves a
reconciled view rather than assigning a cross-boundary completion to only one
side.

Direct observations include routing, lookup class, lifecycle transition,
read issue/completion, bytes, concurrency at issue, cache eviction, expert
availability, and the observed final straggler. Precision/recall and
timeliness classes are derived from those linked observations. The controlled
replay-only counter for a prefetch shortening the final straggler remains zero
in the live runtime.

## Offline replay

The first read-only replay/validator writes JSON only to stdout:

```bash
python3 scripts/replay_qwen3_coder_prompt2_phase4b.py \
  /path/to/trace.jsonl \
  --per-layer-slots 32 \
  --global-slots 1536 \
  --pretty > /tmp/phase4b-replay.json
```

It validates the envelope and stable event IDs, reconstructs current observed
lookup classes, simulates demand-only current per-layer LRU and pooled global
LRU, simulates perfect next-layer fanout 1/2/4/8 without crossing request
boundaries, reports an ideal zero-latency oracle, and summarizes recorded
demand/speculative service latency. It does not modify the input or benchmark
artifact directory.

Belady replacement, optimized per-layer allocation, measured lookahead
2/3/6/12, alternate queue scheduling, probationary cache, cache-pollution
oracles, and incremental expert execution remain explicit future replay work.
The trace's ordered routes, read service intervals, lifecycle identities, and
layer barriers are the inputs needed to add them offline.

## Causal limits and policy statement

Production overlap counters are observations. They may show that a demand read
was issued while speculation was active, but they do not prove contention or
quantify counterfactual delay. Causal delay or saved-latency claims are allowed
only in barrier-controlled deterministic fake-I/O tests or in an offline
replay whose scheduling assumptions are stated.

Phase 4B makes no predictor-policy change. Candidate ranking, `target.last()`
context, fanout defaults, pipeline-depth semantics, shadow sizing,
post-truncation filtering and lack of backfill, cache admission/eviction,
demand-versus-prefetch scheduling, physical I/O, governor policy, and the
all-eight execution barrier remain as they were.
