# Qwen3-Coder single-stream decode: Phase 2 in-flight resident handoff

## Status and scope

This phase implements one narrowly scoped foreground-I/O optimization:
demand-aware publication of an in-flight expert read's immutable resident.
It does not change routing, prediction, cache capacity, model math, storage
layout, prompts, token counts, sampling, CPU placement, Rayon sizing, or any
experimental predictive policy. Mac results are correctness evidence only.
Candidate throughput is intentionally absent until the unchanged four-case
collector runs on the Linux target host.

The `bench-real` schema remains `mer-bench-real` version 2. The new fields are
additive and do not change Phase 1 qualification semantics.

## Qualified Phase 1 evidence

The qualified target-host Phase 1 run passed schema/provenance, output parity,
strict inference, native AVX2/FMA direct-Q8, zero prepared duplicate bytes,
truthful `pread-odirect` expert-I/O, inactive production `io_uring`, critical
path coverage above 97%, and artifact checksum gates.

| Case | Decode TPS | Decode wall |
| --- | ---: | ---: |
| 1,536 short | 1.2802183838270493 | 99.203 s |
| 1,536 medium | 1.4685114039759062 | 86.483 s |
| 6,144 short | 3.181157734961981 | 39.923 s |
| 6,144 medium | 3.027596566141206 | 41.948 s |

The primary 1,536-slot geometric mean is `1.3711364980298737` decode
tokens/s. The 6,144-slot all-resident control geometric mean is
`3.103427497900415` decode tokens/s; it is not SSD-streaming throughput.

Foreground expert-I/O wait accounts for 59.38% of 1,536-short decode and
50.92% of 1,536-medium decode, while it is zero in both resident controls.
Major compute-stage absolute durations are nearly identical between streaming
and resident cases. Expert-cache coordination is only about 0.05--0.06% of
decode wall time, so cache-map and lock bookkeeping were not selected.

Across five measured runs, the 1,536 cases performed 70,217 and 72,234
foreground reads, while only 4,743 of 11,624 completed short prefetches and
4,557 of 11,285 completed medium prefetches became useful. More than 99% of
misses still reached foreground reads. This selected the foreground-I/O and
prefetch lifecycle for audit.

## Lifecycle audit before implementation

### Prediction and submission

`PredictiveLoader` learns first- and second-order Markov transitions from
routed expert sets. `Engine::union_prefetch` combines Markov candidates with
the disabled-by-baseline locality/speculator arms, canonicalizes aliases,
drops current targets and residents, ranks the remaining candidates, and
truncates to the shadow-slot budget. `spawn_prefetch` rejects residents and
known in-flight identities, applies the disabled-by-baseline governor, and
requires a bounded prefetch-semaphore permit.

There is no explicit queued-prefetch registry between `tokio::spawn` and the
task's first poll. If demand arrives in that window, demand installs the
foreground in-flight entry and the later prefetch task observes it and exits.
The result is one foreground read, not duplicate I/O. Once the prefetch task
installs its entry, subsequent demand joins it.

### Buffer allocation and storage read

An admitted prefetch first tries a shadow (`Buffer B`) slot. If all shadow
slots are parked in cache residents, it evicts the least-recently-used
unpinned shadow-backed resident and retries once; otherwise it records a
pool-starvation drop. It never waits for or consumes a primary slot when the
split pool is configured. A later demand for a dropped prediction follows the
ordinary foreground path.

Foreground loads evict an LRU resident when necessary and await a primary
buffer. Neither the cache nor in-flight registry lock is held across storage
I/O. Production expert reads call `NvmeStorage::read_expert`, which uses
positional `pread` with `O_DIRECT` on the qualified target.

### Completion, residency, and consumption

A successful prefetch constructs one `Arc<ExpertResident>` over the exact
shadow buffer and inserts it into the ordinary CPU expert cache. It
deliberately leaves the buffer shadow-tagged so eventual eviction returns the
slot to Buffer B. Demand can execute directly over that resident; no byte copy
or prepared expert representation is created, and direct-Q8 continues to read
`ExpertResident::data()`.

Before this phase, the in-flight map stored only `Arc<Notify>`. A demand or
second foreground caller joined the active operation, waited for a
notification, and then looked in the cache. The leader inserted before
notifying, so the common case worked, but the operation did not publish its
result to its waiters.

That notify-only handoff created two concrete races:

1. another cache insertion or shadow-pool recycle could evict the completed
   resident before a notified demand task reacquired the cache; and
2. a cache full of pinned entries could reject the insert even though the
   leader still held a perfectly usable resident.

In both cases an already-joined demand caller saw an empty cache after wakeup,
won a new foreground election, and issued another physical `pread`. Multiple
foreground callers had the same sequential-duplicate risk. The physical
reads were not simultaneous, but completion was not reliably coalesced.

### Exact pre-change state answers

- Primary-cache resident: demand cloned and used the cached `Arc`.
- Shadow-backed resident in the cache: demand cloned the same `Arc`; the first
  routing consumption credited `prefetch_used`.
- Prefetch task queued but not started: demand could become the foreground
  leader; the later prefetch task exited on the in-flight check.
- Prefetch physical read in progress: demand waited on its `Notify`, then
  re-read the cache.
- Prefetch complete but not yet published: insertion was synchronous, but
  demand still had no owning reference until its later cache lookup.
- Concurrent duplicate demand: one active read leader, but followers depended
  on post-notify cache residency and could issue sequential duplicate reads.
- Evicted unused prefetch: the cache dropped its last `Arc` and the buffer
  returned to its original shadow free list.
- Pool-starved prediction: the prefetch was dropped and later demand used the
  foreground path.
- Prefetch-use accounting: the higher-level routing paths credited both a
  normal shadow-resident cache hit and a successful miss task that returned a
  shadow resident. The in-flight registry itself did not distinguish
  foreground and speculative leaders, so it could not report joins or
  promotions precisely.

## Chosen implementation

The bounded per-expert map now stores an `InFlightEntry` containing:

- immutable origin (`foreground` or `prefetch`);
- a one-way demand-promotion flag;
- one terminal outcome (`Arc<ExpertResident>`, final foreground error, or
  `Retry` for a failed/abandoned speculative leader); and
- a race-safe `Notify`.

The one physical-read leader publishes its outcome before removing the map
entry and notifying waiters. Each already-joined demand waiter holds an
`Arc<InFlightEntry>`, so the published resident and its pooled buffer stay
alive even if the LRU rejects or evicts the resident. All successful waiters
receive clones of the exact same immutable `Arc<ExpertResident>`.

A foreground leader publishes its final retry-loop error to every waiter, so
they fail consistently without a retry herd. A speculative read failure still
publishes `Retry`: this preserves the previous contract that best-effort
prefetch failure is not itself a demand failure. Demand callers re-contend,
exactly one becomes the foreground retry leader, and the others join it.

## Concurrency and ownership invariants

1. At most one registry entry and one active physical operation exist for an
   expert identity.
2. Different identities use different entries and may load concurrently.
3. No cache or registry shard lock is held during buffer waits, storage I/O,
   retry backoff, or waiter suspension.
4. Publication precedes bounded-map removal and notification.
5. Joined waiters own the entry; its published resident pins the pooled buffer
   until each waiter has cloned or dropped the outcome.
6. Completed, failed, cancelled, and panicking leader paths remove the map
   entry. The guard publishes `Retry` if normal completion did not publish.
7. Dropping a demand waiter does not cancel or strand the shared operation.
8. Shadow buffers remain shadow-tagged; direct handoff changes ownership, not
   bytes or pool capacity.
9. Final foreground errors remain fail-closed and are identical for all
   waiters.

The global map is bounded by active operations and removes every terminal
entry immediately. Only waiters that already joined may retain a completed
entry temporarily; no completed entry remains discoverable from the map.

## Telemetry semantics

The following additive per-run fields are exposed under `runs[].cache_io`:

- `demand_requests_joined_inflight_prefetch`: demand callers that found and
  joined a speculative entry;
- `demand_requests_joined_inflight_foreground`: demand callers that joined a
  foreground entry;
- `speculative_loads_promoted_to_demand`: speculative operations marked
  demand-critical, once per operation on the first join;
- `duplicate_physical_reads_avoided`: joined demand callers that did not issue
  a competing physical read, regardless of the leader outcome; it equals the
  two demand-join counters;
- `completed_prefetch_direct_handoffs`: successful speculative operations that
  published their resident to an entry with at least one demand join, counted
  once per operation; a caller cancelled after joining is not subtracted, so
  confirmed routing consumption remains the existing `prefetch_used` metric;
- `foreground_read_operations_issued`: foreground storage calls issued,
  including failures; the existing `foreground_read_operations` retains its
  Phase 1 meaning of successful foreground reads;
- `speculative_read_operations_issued`: speculative storage calls issued after
  admission and buffer acquisition;
- `in_flight_registry_peak_size`: lifetime high-water mark sampled in each
  run; it is a gauge, not a per-run delta;
- `in_flight_registry_size_at_sample`: current map occupancy at the run sample;
- `in_flight_entries_removed`: all terminal entry removals; and
- `in_flight_failed_or_abandoned_entries_removed`: the subset removed with a
  failed, cancelled, or abandoned outcome.

The collector validates the fields and adds aggregate totals to each case
summary. Schema version 2 remains appropriate because existing fields and
qualification gates retain their meaning.

## Tests

Deterministic engine tests use a test-only storage read gate with per-expert
physical-read counters and explicit completion permits. They cover:

- 32 simultaneous demand requests returning the same resident after one read;
- demand joining an in-flight prefetch after one speculative read;
- demand reusing a completed shadow-backed prefetch without copy or foreground
  I/O;
- one final foreground error published to every waiter and registry cleanup;
- a dropped demand waiter and an abandoned leader guard not stranding state;
- a published resident surviving LRU eviction until the waiter owns its `Arc`;
- unrelated expert reads progressing concurrently; and
- bounded primary/shadow pool progress without deadlock.

Existing strict-mode, buffer-pool, expert-cache, direct-Q8, prepared-duplicate,
`bench-real` schema, prompt fixture, and output-parity tests remain applicable.

## Target-host validation

Push the committed branch from a foreground terminal on the development host:

```bash
cd /path/to/Micro-Expert-Router-SSD-Streamed-MoE-MER
git status --short
git push origin perf/qwen3-coder-single-stream-decode
```

Then run the unchanged four-case collector in a foreground VM terminal:

```bash
cd /home/randyap8/Micro-Expert-Router-SSD-Streamed-MoE-MER
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
export MER_QWEN_CONVERTED_DIR=/mnt/localssd/models/qwen3-coder-30b-a3b-q8
export MER_QWEN_TOKENIZER="$MER_QWEN_CONVERTED_DIR/tokenizer.json"
export MER_EXPECTED_NVME_MOUNT=/mnt/localssd

ARTIFACT_DIR="/mnt/localssd/benchmarks/prompt2-phase2-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/collect_qwen3_coder_prompt2_baseline.sh "$ARTIFACT_DIR"
jq . "$ARTIFACT_DIR/qualification.json"
jq . "$ARTIFACT_DIR/four-case-summary.json"
sha256sum -c "$ARTIFACT_DIR/artifact-sha256.txt"
```

Do not change the prompts, output tokens, run counts, cache sizes, sampling,
CPU mask, Rayon worker count, storage layout, or predictive-policy switches.

### Exact candidate comparison

Select the newest artifact for the exact qualified Phase 1 commit and retain
`CANDIDATE_DIR` from the collection above. The commit assertion below prevents
an identically named diagnostic or another Phase 1 revision from being used:

```bash
BASELINE_DIR=$(find /mnt/localssd/benchmarks -maxdepth 1 -type d -name 'prompt2-phase1-327d263193c4-*' -print | sort | tail -n 1)
CANDIDATE_DIR="$ARTIFACT_DIR"
test -n "$BASELINE_DIR"
test -f "$BASELINE_DIR/qualification.json"
test -f "$CANDIDATE_DIR/qualification.json"
jq -e --arg commit 327d263193c48f9dde9f6a716562260ab49fa7ef \
  '.qualification_passed == true and .git_commit_full == $commit' \
  "$BASELINE_DIR/qualification.json"
jq -e '.qualification_passed == true' "$CANDIDATE_DIR/qualification.json"
```

Calculate per-case TPS, TTFT and peak-RSS deltas; streaming/control geometric
means; the streaming geometric-mean delta; and percentage of the
streaming-to-resident gap closed:

```bash
jq -n \
  --slurpfile bs "$BASELINE_DIR/baseline-1536-short.case-summary.json" \
  --slurpfile bm "$BASELINE_DIR/baseline-1536-medium.case-summary.json" \
  --slurpfile brs "$BASELINE_DIR/baseline-6144-short.case-summary.json" \
  --slurpfile brm "$BASELINE_DIR/baseline-6144-medium.case-summary.json" \
  --slurpfile cs "$CANDIDATE_DIR/baseline-1536-short.case-summary.json" \
  --slurpfile cm "$CANDIDATE_DIR/baseline-1536-medium.case-summary.json" \
  --slurpfile crs "$CANDIDATE_DIR/baseline-6144-short.case-summary.json" \
  --slurpfile crm "$CANDIDATE_DIR/baseline-6144-medium.case-summary.json" '
  def delta($b; $c): 100 * ($c / $b - 1);
  def row($b; $c): {
    baseline_decode_tps: $b.decode_tps_mean,
    candidate_decode_tps: $c.decode_tps_mean,
    decode_tps_delta_percent: delta($b.decode_tps_mean; $c.decode_tps_mean),
    ttft_delta_seconds: ($c.time_to_first_token_p50_seconds - $b.time_to_first_token_p50_seconds),
    external_peak_rss_delta_bytes: ($c.external_peak_rss_bytes - $b.external_peak_rss_bytes),
    output_parity: $c.output_token_parity,
    qualification_passed: $c.qualification_passed
  };
  ($bs[0]) as $bs | ($bm[0]) as $bm |
  ($brs[0]) as $brs | ($brm[0]) as $brm |
  ($cs[0]) as $cs | ($cm[0]) as $cm |
  ($crs[0]) as $crs | ($crm[0]) as $crm |
  (($bs.decode_tps_mean * $bm.decode_tps_mean) | sqrt) as $bgm |
  (($cs.decode_tps_mean * $cm.decode_tps_mean) | sqrt) as $cgm |
  (($brs.decode_tps_mean * $brm.decode_tps_mean) | sqrt) as $brgm |
  (($crs.decode_tps_mean * $crm.decode_tps_mean) | sqrt) as $crgm |
  {
    streaming_short: row($bs; $cs),
    streaming_medium: row($bm; $cm),
    streaming_geometric_mean: {
      baseline: $bgm,
      candidate: $cgm,
      delta_percent: delta($bgm; $cgm),
      gap_closed_percent: (100 * ($cgm - 1.3711364980298737) /
        (3.103427497900415 - 1.3711364980298737)),
      minimum_1_50_passed: ($cgm >= 1.50)
    },
    resident_short: row($brs; $crs),
    resident_medium: row($brm; $crm),
    resident_geometric_mean: {
      baseline: $brgm,
      candidate: $crgm,
      delta_percent: delta($brgm; $crgm)
    }
  }'
```

Calculate foreground/cache/prefetch deltas and all new mechanism counters from
the raw reports. The same command checks output parity, strictness, fallback,
read-failure, and prepared-duplicate status:

```bash
jq -n \
  --slurpfile bs "$BASELINE_DIR/baseline-1536-short.json" \
  --slurpfile bm "$BASELINE_DIR/baseline-1536-medium.json" \
  --slurpfile brs "$BASELINE_DIR/baseline-6144-short.json" \
  --slurpfile brm "$BASELINE_DIR/baseline-6144-medium.json" \
  --slurpfile cs "$CANDIDATE_DIR/baseline-1536-short.json" \
  --slurpfile cm "$CANDIDATE_DIR/baseline-1536-medium.json" \
  --slurpfile crs "$CANDIDATE_DIR/baseline-6144-short.json" \
  --slurpfile crm "$CANDIDATE_DIR/baseline-6144-medium.json" '
  def total($r; $field): ([$r.runs[].cache_io[$field]] | add);
  def change($b; $c; $field): {
    baseline: total($b; $field),
    candidate: total($c; $field),
    delta: (total($c; $field) - total($b; $field))
  };
  def case($b; $c): {
    foreground_read_operations: change($b; $c; "foreground_read_operations"),
    foreground_read_operations_issued: total($c; "foreground_read_operations_issued"),
    foreground_expert_bytes: change($b; $c; "foreground_expert_bytes"),
    foreground_expert_io_wait_seconds: change($b; $c; "foreground_expert_io_wait_seconds"),
    cache_hits: change($b; $c; "cache_hits"),
    cache_misses: change($b; $c; "cache_misses"),
    cache_evictions: change($b; $c; "cache_evictions"),
    prefetch_useful_bytes: change($b; $c; "useful_prefetch_bytes"),
    prefetch_unused_bytes_at_sample: change($b; $c; "unused_prefetch_bytes_at_sample"),
    new_mechanism_totals: {
      speculative_read_operations_issued: total($c; "speculative_read_operations_issued"),
      demand_joined_prefetch: total($c; "demand_requests_joined_inflight_prefetch"),
      demand_joined_foreground: total($c; "demand_requests_joined_inflight_foreground"),
      speculative_promoted_to_demand: total($c; "speculative_loads_promoted_to_demand"),
      duplicate_physical_reads_avoided: total($c; "duplicate_physical_reads_avoided"),
      completed_prefetch_direct_handoffs: total($c; "completed_prefetch_direct_handoffs"),
      in_flight_registry_peak_size: ([$c.runs[].cache_io.in_flight_registry_peak_size] | max),
      in_flight_entries_removed: total($c; "in_flight_entries_removed"),
      failed_or_abandoned_entries_removed: total($c; "in_flight_failed_or_abandoned_entries_removed")
    },
    output_parity: $c.aggregate.output_token_parity,
    strict_fail_closed: (
      $c.strictness.strict_weights and
      ($c.strictness.seeded_fallback_remained | not) and
      ([$c.runs[].correctness] | all(
        .degraded_expert_substitutions == 0 and
        .expert_read_failures == 0 and
        .truncated_expert_payload_uses == 0 and
        .nonfinite_attention_fallbacks == 0 and
        .q8_scalar_layout_fallbacks == 0 and
        .prepared_duplicate_expert_bytes == 0 and
        (.inference_policy | all(. == false))))
    )
  };
  {
    streaming_short: case($bs[0]; $cs[0]),
    streaming_medium: case($bm[0]; $cm[0]),
    resident_short: case($brs[0]; $crs[0]),
    resident_medium: case($brm[0]; $crm[0])
  }'
```

## Acceptance and rollback

The candidate must retain every Phase 1 qualification gate, output parity,
strict fail-closed inference, zero degraded substitutions/read failures,
native AVX2/FMA direct-Q8, zero prepared duplicate bytes, and comparable peak
RSS. The new telemetry must show whether demand joined speculative or
foreground operations. For the 1,536 cases, successful foreground operations
or foreground-I/O wait must decrease without a material regression in either
case. Both 6,144 resident controls must remain within normal run variance.

The minimum milestone is a 1,536-slot geometric mean of at least 1.50 decode
tokens/s (about 9.4% above Phase 1). Report the result even below the threshold.
Retain a sub-threshold candidate only if telemetry proves less foreground I/O
with no meaningful throughput or memory regression. Otherwise revert this
single candidate before selecting another workstream.

Use:

```text
gap_closed_percent = 100 * (candidate_streaming_GM - 1.3711364980298737)
                           / (3.103427497900415 - 1.3711364980298737)
```

## Limitations and later work

- The registry begins when the spawned prefetch task is polled, not at
  `spawn_prefetch` submission. Demand in the short queued window already wins
  foreground leadership, so it does not duplicate I/O, but it is not counted
  as a speculative promotion.
- The portable `pread` provider has no device-level priority API. Promotion
  makes lifecycle ownership demand-safe; it cannot reprioritize a syscall
  already executing in the kernel.
- This change cannot improve incorrect predictions that never overlap demand.
  Prefetch admission quality, speculative/foreground device scheduling, and
  production `io_uring` remain separate later workstreams and must not be
  combined with this candidate before target-host review.
