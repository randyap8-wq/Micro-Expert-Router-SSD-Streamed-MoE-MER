# Qwen3-Coder single-stream decode: Prompt 2 Phase 3A

## Scope and status

Phase 3A audits and instruments foreground demand-miss fanout. It does not
change fetching, scheduling, cache, prediction, eviction, storage, compute, or
concurrency policy. The Phase 2 in-flight resident handoff remains removed.

No performance result is claimed by this change. macOS results are correctness
evidence only. Throughput, concurrency, and device-service conclusions require
the qualified Linux x86_64 target-host collection below.

## Exact production control flow

For each sparse transformer layer:

1. `RealModel::forward_token_hidden_with_timing` runs attention, then
   `TransformerLayer::moe_pre_into_with_timing` performs the router gate and
   top-k selection. For Qwen3-Coder, `top_k = 8`.
2. Layer-local expert ids are converted to the global layer-qualified namespace
   (`layer * 128 + local`), preserving routed order and weights.
3. `Engine::moe_step_weighted_into_with_timing` invokes `moe_step_inner`.
   Existing Markov/look-ahead speculation is submitted before the demand cache
   loop. Current-layer target ids are excluded from new union-prefetch requests,
   although a prior layer can already be fetching the same identity.
4. The demand loop probes the resident `MultiLayerExpertCache` in routed order.
   Hits retain their `Arc<ExpertResident>`. Every initial miss increments the
   existing miss counter and immediately spawns a Tokio task for
   `fetch_with_retry`; no miss handle is awaited inside this lookup loop.
5. `fetch_with_retry` rechecks cache, then enters the per-identity DashMap
   singleflight election. A follower waits on that identity's `Notify`. A leader
   alone executes `fetch_once` and its retry loop.
6. `fetch_once` may evict an unpinned LRU resident, then awaits a primary
   Buffer-A slot. It does not acquire a foreground semaphore. The optional
   prefetch governor guard is accounting/admission feedback for future
   speculative requests, not foreground admission.
7. After primary-buffer acquisition, `NvmeStorage::read_expert_observed` resolves
   the already-warmed fd (or packed slot), calls `block_in_place`, and performs
   the O_DIRECT positional `pread`. Independent ids use independent tasks and
   positional reads. The fd-LRU mutex is held only for lookup/clone or insert;
   opening occurs outside it and no fd-cache lock spans storage I/O.
8. The leader inserts the completed resident into the cache before dropping its
   singleflight guard. Dropping the guard removes the identity and notifies
   followers.
9. Only after all miss tasks have been spawned does `moe_step_inner` await their
   handles. Handles are consumed in routed order. This can create measurable
   completion-to-consumption delay when a later routed slot completes before an
   earlier slot.
10. The layer does not stream expert compute as individual reads finish. It
    drains every required miss handle first. Strict mode returns on any fetch
    failure only after all handles have been drained.
11. Qwen Q8_0 `expert_execution_policy = auto` resolves to parallel experts with
    single-thread inner kernels for top-8 on the 30-worker target. Weighted
    partial outputs are reduced into the routed weighted accumulation only after
    all required residents are available. The transformer then performs the
    MoE residual combination and completes the layer.

## Concurrency and ownership audit

All missing expert fetch tasks are submitted before the first await. Logical
fanout is therefore concurrent. Physical read concurrency is mixed/dynamic,
not guaranteed to equal the miss count:

- there is no foreground semaphore, reactor queue, or configured foreground
  concurrency cap on this production path;
- each leader must first obtain a primary Buffer-A slot;
- the production pool has `cache_slots + 1` primary slots. Live cache residents
  retain primary buffers, so free capacity and the separate `len >= capacity`
  / LRU-eviction race can partially serialize a top-8 burst;
- speculative requests use the separate six-slot Buffer-B pool and therefore
  cannot consume a primary slot in the qualified configuration;
- speculative and demand reads still share the same fd cache, Tokio runtime,
  `NvmeStorage`, NVMe device, and kernel/device queues;
- each synchronous `pread` donates its Tokio worker with `block_in_place`.
  Effective physical fanout is consequently bounded by ready primary buffers,
  scheduled blocking work, the number of distinct singleflight leaders, and the
  device/kernel—not by a foreground permit configured in MER.

No lock is held across storage I/O. The cache, singleflight shard, primary pool,
fd LRU, and telemetry concurrency locks are acquired and released at separate
ownership boundaries before storage service begins.

Speculative overlap is observable, but causal speculative *delay* is not. The
collector therefore reports `demand_reads_issued_while_speculative_reads_active`
and leaves `demand_critical_reads_delayed_by_speculative_activity` as `null`.
It does not relabel overlap as proof of delay.

## Phase 3A telemetry semantics

Every `bench-real` run contains `demand_miss_fanout.prompt` and
`demand_miss_fanout.decode`. Prompt and decode use separate reset epochs so a
warmup or prompt peak cannot contaminate decode peak concurrency or the bounded
worst-layer sample.

The resident path is protected as follows:

- a per-layer tracker is allocated only after the first initial miss;
- no Phase 3A timestamp, allocation, mutex, registry access, histogram update,
  string construction, tracing event, or shared atomic is added to a zero-miss
  layer;
- routed layers, routed activations, resident hits, and initial misses reuse the
  Phase 1 counters;
- missing-count buckets 1 through top-k are miss-only atomics; bucket zero is
  derived as `routed_layers - sum(nonzero buckets)` in the benchmark report.

The report separates these intervals:

- cache lookup: existing benchmark-scoped `expert_cache_lookup` stage;
- primary-buffer acquisition wait: only `BufferPool::acquire` elapsed time;
- foreground admission wait: explicitly zero, with admission control reported
  as `none`;
- storage service: callbacks after fd resolution and immediately around the
  `NvmeStorage` positional-read service, including its internal retry service;
- completion-to-consumption delay: expert-task availability until the routed
  handle is consumed;
- singleflight wait: demand follower wait on an already-active identity.

Bounded telemetry includes the 0..8 miss histogram, multiple-miss layers,
serial/overlapping read layers, physical-read latency buckets, active-time and
concurrency integral, peak and time-weighted foreground QD, first/last issue and
availability timing, completion spread, layer fetch critical path, compute-start
ordering, final-straggler routed-slot histogram, and one worst-layer sample with
the final expert identity. It records no unbounded per-token trace.

The first Phase 3 optimization decision is gated on target data:

- if top-8 layers commonly have multiple misses but peak/average foreground QD
  stays near one and primary-buffer wait dominates, investigate the existing
  primary-buffer acquisition/eviction boundary;
- if primary-buffer wait is small but issue spread is large, investigate Tokio
  task dispatch before storage submission;
- if reads overlap and one issue-to-completion service time dominates the final
  availability window, investigate storage stragglers/layout;
- if completion-to-consumption delay is material, investigate routed-order join
  handling;
- if misses are usually zero or one, fanout is not the first optimization target.

No policy change should be selected before these measurements are collected.

## Deterministic tests

The Phase 3A unit tests inject exact nanosecond event schedules rather than use
wall-clock sleeps. They cover zero, one, and multiple misses; serial and
overlapping reads; exact time-weighted QD and peak; buffer/admission/storage
separation; a slow final straggler; concurrent unrelated identities; compute
ordering; and reset behavior. The engine test loads a layer once, resets the
collector, and proves the all-resident output is identical while every miss-only
counter remains inactive. Existing native-Q8 parity/concurrency tests remain the
direct-Q8 guard.

```bash
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router \
  demand_fetch_telemetry::tests -- --nocapture
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router \
  phase3a_all_resident_layer_stays_miss_telemetry_inactive -- --nocapture
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router \
  moe_step_weighted_into_matches_per_expert_combiner -- --nocapture
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router \
  direct_q8_same_resident_is_concurrent -- --nocapture
cargo test --manifest-path rust-engine/Cargo.toml --bin micro-expert-router \
  --features q8-candle-reference direct_q8_matches_candle_tiny_medium_and_qwen_shapes \
  -- --nocapture
bash scripts/tests/test_qwen3_coder_prompt2_resident_finalization.sh
```

## Exact qualified Linux collection

Run only on the specified Linux x86_64 host with exactly 32 online logical CPUs,
30 Rayon workers pinned inside CPUs 0-31, the converted Qwen model on local ext4
NVMe, and O_DIRECT enabled.

```bash
cd /path/to/Micro-Expert-Router-SSD-Streamed-MoE-MER

export MER_QWEN_CONVERTED_DIR=/mnt/localssd/qwen3-coder-30b-a3b-instruct-q8_0
export MER_QWEN_TOKENIZER="$MER_QWEN_CONVERTED_DIR/tokenizer.json"
export MER_EXPECTED_NVME_MOUNT=/mnt/localssd

ARTIFACT_DIR="/mnt/localssd/benchmarks/prompt2-phase3a-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/collect_qwen3_coder_prompt2_baseline.sh "$ARTIFACT_DIR" four-case

sha256sum -c "$ARTIFACT_DIR/artifact-sha256.txt"
jq '.cases[] | {case, decode_tps_mean, phase3a_decode}' \
  "$ARTIFACT_DIR/four-case-summary.json"
```

Run the resident control independently when auditing the ±2% gate:

```bash
RESIDENT_DIR="/mnt/localssd/benchmarks/prompt2-phase3a-resident-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/collect_qwen3_coder_prompt2_baseline.sh "$RESIDENT_DIR" --resident-only
jq '{resident_gates, performance_gate, qualification_passed}' \
  "$RESIDENT_DIR/resident-control-summary.json"
```

Do not treat a dirty-tree diagnostic, a macOS run, a non-ext4 filesystem, a
buffered-I/O run, a different CPU/Rayon placement, or a run with failed parity,
strictness, resident, or qualification gates as a performance result.
