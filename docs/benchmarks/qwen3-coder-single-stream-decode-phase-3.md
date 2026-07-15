# Qwen3-Coder single-stream decode: Prompt 2 Phase 3

## Scope and status

Phase 3A, commit `f59ba97afb7829a2cbf4bea50a959181ec95de2e`,
instrumented foreground demand-miss fanout without changing runtime policy. Its
qualified Linux four-case collection ran on the required GCP
`g2-standard-32`: Linux x86_64, 32 online logical CPUs, CPU mask `0-31`, 30
Rayon workers, about 128 GB RAM, and local ext4 NVMe at `/mnt/localssd` mounted
`rw,noatime,nodiratime` with O_DIRECT expert reads.

That evidence established:

- top-8 demand misses are already logically concurrent;
- multiple physical reads overlap and peak foreground physical-read
  concurrency reaches eight;
- primary Buffer-A wait and demand-task issue spread are negligible;
- expert compute begins almost immediately after the final required expert is
  available;
- storage-service stragglers dominate the layer-fetch critical path; and
- about one third of demand reads were issued while speculative physical reads
  were active.

Those findings reject Buffer-A capacity, generic Tokio fanout, routed-order
handle consumption, and expert-compute pipelining as the first Phase 3B target.
They also leave packed storage, io_uring, cache policy, routing, model math, and
native direct-Q8 execution outside this phase.

Phase 3B tests one narrower hypothesis: a newly starting speculative read can
contribute to foreground storage-service tails by sharing `NvmeStorage`, Tokio
blocking capacity, the kernel/device queue, and the NVMe device with an active
demand burst. Overlap remains exposure, not proof of causal delay.

No Linux performance result is included in the implementation commit. macOS
results below are correctness evidence only.

## Unchanged production control flow

For each sparse transformer layer, `moe_step_inner` routes top-k experts,
submits the existing prediction arms, probes the resident cache in routed
order, and spawns every initial demand miss before awaiting any miss handle.
`fetch_with_retry` retains its cache rechecks, per-identity DashMap
singleflight, primary Buffer-A acquisition, retries, strict errors, and cache
insertion. A demand leader still calls the same O_DIRECT positional `pread`
through `NvmeStorage::read_expert_observed`. There is no foreground semaphore
and demand physical-read concurrency is unchanged.

Speculation retains the existing prediction identities, bounded concurrency
semaphore, separate shadow Buffer-B pool, singleflight identity, cache policy,
and insertion/error behavior. Already-running speculative storage calls are
never canceled.

## Phase 3B arbitration semantics

The miss-only `LayerFetchTracker` owns a demand-burst guard. It is created only
after the layer observes its first initial cache miss, before that layer's
demand tasks can enter storage, and is finished after every required miss
handle has been drained. Dropping the tracker is a failure-safe release path.

One engine-scoped `AtomicU64` packs two gauges:

- high 32 bits: active foreground demand-burst guards;
- low 32 bits: speculative reads admitted into physical storage service.

Demand entry performs one non-waiting atomic increment. Speculative admission
uses a compare/exchange that can increment the low half only when the high half
is zero. The successful speculative CAS is executed immediately before
the positional `pread`, after fd resolution, Buffer-B acquisition, and
singleflight election. This gives demand entry and speculative physical
submission one total atomic order:

- if speculative admission linearizes first, it is already running when the
  demand burst begins and is allowed to finish;
- if demand entry linearizes first, speculative admission fails before storage
  submission and the request defers.

When pressure is already visible, a speculative task waits without claiming a
singleflight identity or Buffer-B slot. If pressure races the final admission
check, the task releases its Buffer-B slot and singleflight guard before
waiting. This prevents a foreground request for the same expert from waiting
on a deferred speculative leader. Demand never waits for a Phase 3B permit,
semaphore, condition, queue slot, or speculative completion.

Only speculative tasks wait on the pressure-cleared notification. The final
overlapping demand guard wakes them when the packed demand count reaches zero.
They retry through the existing bounded speculative semaphore, cache recheck,
governor, singleflight election, and Buffer-B acquisition. Cache hits and
duplicates retain their existing best-effort exit; deferred instances receive
an additional accounting counter rather than a storage or strictness failure.

No arbitration lock is held across I/O. The policy itself has no mutex. The
existing fd-cache, cache, singleflight, pool, and telemetry-concurrency locks
remain confined to their prior non-I/O ownership boundaries.

### Race and starvation analysis

- A speculative request cannot pass a stale `pressure == 0` observation: the
  final packed-state CAS is authoritative.
- Demand entry cannot cancel a read represented in the packed speculative
  count; it only prevents later speculative admissions.
- Multiple demand bursts are unioned. Releasing any guard except the last
  leaves pressure active.
- Notification registration occurs before the speculative task rechecks the
  demand count, avoiding a lost wakeup at the zero transition.
- Deferred tasks remain bounded by the existing speculative permit count.
  When demand pressure clears they are woken together and must reacquire the
  same bounded machinery, so there is no unbounded deferred queue.
- Sustained foreground traffic can postpone speculation by design. Once the
  union becomes empty, notification plus the existing retry path makes the
  deferred work eligible again; stale work is explicitly accounted instead of
  being reported as I/O failure.

## Resident-path protection

The per-layer tracker, demand guard, timestamp, and arbitration atomic update
are created only inside the first-miss branch. A zero-miss layer executes no
new Phase 3B timestamp read, allocation, lock/condition access, registry or
queue operation, histogram update, formatting, trace event, or shared atomic
update. Speculative arbitration is inspected only by a speculative request
that has survived the existing cache/dedup/admission checks and is attempting a
physical read. The 6,144 resident control requires every Phase 3B counter and
histogram to stay zero in addition to the prior zero-I/O gates.

## Telemetry definitions

`bench-real` preserves every Phase 3A field and adds the following bounded
fields independently under `demand_miss_fanout.prompt` and `.decode`:

- `foreground_demand_bursts_entered`: miss-only layer guards created;
- `foreground_demand_pressure_active_seconds`: union time with at least one
  guard active;
- `speculative_physical_reads_admitted_without_demand_pressure`: successful
  final admission CAS operations;
- `speculative_physical_reads_deferred_for_demand_pressure`: logical
  speculative operations that first encountered pressure;
- `deferred_speculative_physical_reads_resumed`: deferred operations later
  admitted into physical service;
- `deferred_speculative_physical_reads_dropped_stale_duplicate_or_cache_hit`:
  deferred operations made unnecessary by the existing cache/singleflight
  rechecks;
- `speculative_physical_reads_active_when_demand_burst_began`: running
  speculative reads observed at demand-guard entry, summed across bursts;
- `demand_reads_issued_while_speculative_reads_active`: preserved Phase 3A
  overlap count;
- `demand_physical_read_service_{without,with}_speculation_{operations,seconds,mean_seconds,max_seconds,histogram}`:
  demand storage service categorized by speculative physical activity at issue;
- `demand_layers_final_straggler_issued_while_speculative_reads_active`: layers
  whose last-required expert's final physical issue observed speculation; and
- `demand_critical_reads_delayed_by_speculative_activity: null`: retained to
  make clear that overlap alone is not a causal-delay claim.

Both service histograms reuse the fixed 16 Phase 3A latency buckets. Per-layer
state remains exactly top-k slots plus one bounded worst-layer sample; no
per-token or per-expert history is retained.

Case summaries preserve `phase3a_decode` and add `phase3b_prompt` and
`phase3b_decode`. The latter aggregate all Phase 3B counters and both bounded
service histograms across the five measured runs.

## Deterministic and macOS correctness evidence

The arbitration tests use atomic schedules, Tokio notifications, and oneshot
channels rather than arbitrary sleeps. They cover idle speculative admission,
non-waiting demand entry, deferral before submission, no cancellation,
overlapping demand guards, wake/resume, stale/duplicate/cache-hit accounting,
and a blocked injected-storage operation proving that arbitration holds no
lock. The all-resident engine test additionally asserts every Phase 3B field
and histogram is inactive.

Run the complete macOS correctness set (performance conclusions are forbidden):

```bash
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router demand_fetch_telemetry::tests
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router phase3a_all_resident_layer_stays_miss_telemetry_inactive
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router moe_step_weighted_into_matches_per_expert_combiner
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router direct_q8_same_resident_is_concurrent
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router --features q8-candle-reference direct_q8_matches_candle_tiny_medium_and_qwen_shapes
bash scripts/tests/test_qwen3_coder_prompt2_resident_finalization.sh
bash scripts/tests/test_qwen3_coder_prompt2_same_host_compare.sh
```

On macOS, the implementation commit passed the default binary suite (`748`
passed, `2` ignored, `0` failed), all 14 Phase 3 telemetry/arbitration tests,
the resident-inactivity test, weighted combiner test, concurrent-resident
direct-Q8 test, Candle-reference direct-Q8 parity test, resident-finalizer
fixture test, and immutable same-host comparison fixture test. These are
correctness and artifact-pipeline results only; they provide no Linux or NVMe
performance evidence.

Also run `bash -n` on every changed shell file, compile the jq comparison
program with the synthetic helper test, run rustfmt on the changed Rust files,
and finish with `git diff --check`.

## Exact qualified Linux collection

Use the same qualified host for both commits. Keep the Phase 3A and Phase 3B
streaming/resident directories below one bundle directory per phase so the
read-only comparison helper can discover them.

```bash
cd /path/to/Micro-Expert-Router-SSD-Streamed-MoE-MER

export MER_QWEN_CONVERTED_DIR=/mnt/localssd/qwen3-coder-30b-a3b-instruct-q8_0
export MER_QWEN_TOKENIZER="$MER_QWEN_CONVERTED_DIR/tokenizer.json"
export MER_EXPECTED_NVME_MOUNT=/mnt/localssd

# Collect this once from the exact qualified Phase 3A commit.
git switch --detach f59ba97afb7829a2cbf4bea50a959181ec95de2e
PHASE3A_ROOT=/mnt/localssd/benchmarks/prompt2-phase3a-same-host
scripts/collect_qwen3_coder_prompt2_baseline.sh "$PHASE3A_ROOT/streaming" four-case
scripts/collect_qwen3_coder_prompt2_baseline.sh "$PHASE3A_ROOT/resident" --resident-only || true
sha256sum -c "$PHASE3A_ROOT/streaming/artifact-sha256.txt"
sha256sum -c "$PHASE3A_ROOT/resident/artifact-sha256.txt"

# Return to the committed Phase 3B candidate and collect without changing any fixture.
git switch perf/qwen3-coder-single-stream-decode
PHASE3B_ROOT=/mnt/localssd/benchmarks/prompt2-phase3b-same-host
scripts/collect_qwen3_coder_prompt2_baseline.sh "$PHASE3B_ROOT/streaming" four-case
scripts/collect_qwen3_coder_prompt2_baseline.sh "$PHASE3B_ROOT/resident" --resident-only || true
sha256sum -c "$PHASE3B_ROOT/streaming/artifact-sha256.txt"
sha256sum -c "$PHASE3B_ROOT/resident/artifact-sha256.txt"

# Writes only the new destination; neither source artifact is modified.
scripts/compare_qwen3_coder_prompt2_same_host.sh \
  "$PHASE3A_ROOT" "$PHASE3B_ROOT" \
  > /mnt/localssd/benchmarks/prompt2-phase3a-vs-phase3b-same-host.json
jq . /mnt/localssd/benchmarks/prompt2-phase3a-vs-phase3b-same-host.json
```

The resident collector intentionally preserves the historical reference
`3.500144036461` decode TPS geometric mean from commit
`327d263193c48f9dde9f6a716562260ab49fa7ef`. The replacement VM is known to
fall below that cross-VM gate. Keep its `qualification_passed: false` and
failure reason truthful. The comparison helper separately reports the
contemporaneous Phase 3A-to-3B resident delta and never rewrites either source.

## Success and rollback criteria

Phase 3B is successful only when the same-host Linux report shows all of the
following:

- streaming and resident correctness, strictness, provenance, prompt identity,
  critical-path coverage, checksums, and exact output parity pass;
- every resident Phase 3B counter/histogram remains zero and the same-host
  resident geometric mean stays within ±2% of Phase 3A;
- neither 1,536 short nor medium streaming case materially regresses;
- streaming geometric mean improves beyond likely noise, preferably at least
  2%;
- demand storage tail or layer fetch critical path decreases consistently;
- cache-miss and SSD-byte changes are reported, including increases; and
- every deferred speculative operation is accounted as resumed, made stale by
  cache/singleflight state, or dropped by an existing bounded-mechanism counter.

If those criteria fail, retain and checksum the Phase 3B artifacts, record the
negative result, and revert the runtime arbitration commit (or remove the
policy while retaining additive telemetry). Do not reinterpret overlap as
causality, silently update the historical resident reference, or claim Linux
performance from macOS correctness checks.
