# Qwen3-Coder single-stream decode: Prompt 2 Phase 3

## Disposition

Phase 3A, commit `f59ba97afb7829a2cbf4bea50a959181ec95de2e`,
added bounded, miss-only foreground demand-fetch telemetry without changing
runtime policy. Phase 3B, tested commit
`ca7ca5389bbcd86f199c45244e99a48483de249c`, added a demand-priority
speculative-read arbitration policy.

The qualified same-host Linux x86_64 A-B-A benchmark rejected that policy. The
small throughput improvement appears real, but it did not reach the preferred
threshold and the latency, resident, miss, fetch-fraction, and accounting gates
did not support production use. The follow-up disposition therefore removes
the Phase 3B arbitration and restores the Phase 3A demand and speculative
physical-read paths. The Phase 3A miss telemetry remains enabled.

Phase 3B is disabled and no Phase 3B production performance claim is made.
macOS results are correctness evidence only and are not Linux or NVMe
performance evidence.

## Qualified Linux A-B-A result

All three collections ran on the same qualified GCP `g2-standard-32`: Linux
x86_64, 32 online logical CPUs, CPU mask `0-31`, 30 Rayon workers, about
128 GB RAM, and local ext4 NVMe at `/mnt/localssd` mounted
`rw,noatime,nodiratime` with O_DIRECT expert reads. The Phase 3A before and
after measurements bracketed the tested Phase 3B commit.

| Measurement | Phase 3A before | Phase 3B | Phase 3A after |
| --- | ---: | ---: | ---: |
| Short decode TPS | 1.293438199065394 | 1.305594662395702 | 1.297117606206983 |
| Medium decode TPS | 1.5008307858944783 | 1.5249795388009992 | 1.494644032414944 |
| Streaming geometric-mean TPS | 1.3932809726717916 | 1.4110298175875817 | 1.3923825226774516 |
| Resident geometric-mean TPS | 3.2307087765796116 | 3.532783148464943 | 3.2437048410808447 |

The two Phase 3A streaming geometric means differed by
`-0.06448448029956477%`, about 0.06%, so the A measurements were stable. Phase
3B improved streaming geometric-mean throughput by `+1.3392%` versus the
fresh Phase 3A after measurement. It was `+0.6535%` on the short fixture and
`+2.0296%` on the medium fixture.

The detailed Phase 3B versus fresh Phase 3A changes were:

| Metric | Short | Medium | Disposition |
| --- | ---: | ---: | --- |
| Decode TPS | +0.6535% | +2.0296% | Positive, but geometric mean remained below +2% |
| Mean demand-storage latency | +3.6406% | -0.2809% | Inconsistent |
| Mean layer-fetch critical path | +5.3349% | +4.0320% | Worse in both fixtures |
| Decode-wall fetch fraction | +6.0315% | +6.3846% | Worse in both fixtures |
| Cache misses | +1.6641% | +1.2174% | Increased in both fixtures |
| SSD bytes | -0.5781% | -0.8048% | Decreased slightly in both fixtures |

Maximum demand-storage latency and maximum layer-fetch critical-path changes
were inconsistent between the fixtures; the retained, checksummed comparison
artifacts remain the canonical exact record for those per-case values. Phase
3B resident geometric-mean TPS was `+9.350095993645734%` above the Phase 3A
before measurement and `+8.9119794046296%` above the Phase 3A after
measurement. Arbitration should be inactive during resident inference, so that
approximately 9% separation is not accepted as evidence of a resident-path
improvement.

The tested Phase 3B service split also showed that demand reads issued while
speculative physical reads were active remained substantially slower than
demand reads issued without speculative activity. Arbitration did not close
that gap. This keeps the broader shared-device-tail hypothesis open, but does
not validate the tested policy or establish speculation as the cause.

The policy failed acceptance because:

1. streaming geometric-mean improvement was `+1.3392%`, below the preferred
   `+2%`;
2. mean layer-fetch critical path worsened in both fixtures;
3. decode-wall fetch fraction worsened in both fixtures;
4. maximum-latency improvement was inconsistent between fixtures;
5. cache misses increased in both fixtures;
6. resident performance differed from both surrounding Phase 3A resident
   measurements by about 9%; and
7. the deferred/resumed terminal accounting was incomplete at snapshot time.

## Active runtime after disposition

The active runtime is the Phase 3A runtime:

- demand misses retain the existing cache rechecks, per-identity DashMap
  singleflight, primary Buffer-A acquisition, retry, strict-error, cache
  insertion, and O_DIRECT positional-read behavior;
- all initial routed misses are spawned before any miss handle is awaited, with
  no foreground admission semaphore;
- speculative requests retain their existing prediction identities, governor,
  bounded concurrency semaphore, cache/singleflight logic, separate Buffer-B
  pool, insertion behavior, and ordinary `NvmeStorage::read_expert` call;
- demand and speculative reads are no longer ordered by a Phase 3B packed
  arbitration state, demand-burst guard, notification, or deferred retry;
- the Phase 3B conditional storage API and arbitration-only tests/state are
  removed; and
- model math, routing, cache and eviction policy, strictness, output order,
  native direct-Q8 execution, storage backend, Buffer-A semantics, and Buffer-B
  semantics are unchanged.

Phase 3A telemetry remains passive and miss-only. A `LayerFetchTracker` is
allocated only after a routed layer observes its first initial cache miss.
It records cache lookup, Buffer-A wait, storage service, singleflight wait,
completion-to-consumption delay, bounded miss histograms, read concurrency,
issue/availability spreads, layer fetch critical path, final-straggler slot,
and one bounded worst-layer sample. The existing speculative-active gauge only
classifies overlap observed when a demand physical read begins; it does not
defer, cancel, admit, or reorder either read class. An all-resident layer does
not allocate a tracker or update miss telemetry.

Overlap remains exposure rather than proof of causal delay.
`demand_critical_reads_delayed_by_speculative_activity` therefore remains
`null`.

## Phase 3B deferral-accounting audit

The historical Phase 3B counters did not share one terminal unit:

- `speculative_physical_reads_deferred_for_demand_pressure` incremented once
  when one logical speculative operation first encountered demand pressure.
  The `was_deferred` flag prevented the same operation from incrementing it
  again on subsequent pressure encounters.
- `deferred_speculative_physical_reads_resumed` incremented when a previously
  deferred operation later passed the final admission compare/exchange and
  entered physical storage service. It counted admission, not successful read
  completion.
- `deferred_speculative_physical_reads_dropped_stale_duplicate_or_cache_hit`
  incremented only when a later cache or singleflight recheck made the
  deferred operation unnecessary.

A deferred retry re-entered the complete existing speculative path. It could
therefore also exit through the pre-existing governor, concurrency-ceiling, or
Buffer-B/pool-starvation counters. Those outcomes did not increment the named
Phase 3B stale/duplicate/cache-hit counter. A storage error happened only after
admission, so a deferred operation reaching that point was already counted as
resumed.

Speculative tasks were detached `tokio::spawn` tasks; their join handles were
not retained. The benchmark took the prompt and decode snapshots immediately
after foreground generation and did not drain speculative tasks. The Phase 3B
telemetry reset rejected active demand bursts and foreground reads, but did not
wait for deferred speculative tasks. Consequently, a snapshot could contain
deferred work that was still waiting or retrying, and such a task could cross a
prompt/decode reset boundary. There was no separate deferred queue or
retry-shutdown terminal counter; runtime shutdown could cancel remaining
detached tasks.

The captured aggregate values were:

| Epoch | Deferred once | Later admitted (“resumed”) | Stale/duplicate/cache-hit drop | Not classified by those terminal counters at snapshot |
| --- | ---: | ---: | ---: | ---: |
| Prompt | 5,002 | 4,424 | 0 | 578 |
| Decode | 16,360 | 15,051 | 0 | 1,309 |

The final column is a snapshot difference, not a count of leaked or lost work.
The retained artifacts cannot distinguish still-pending detached work from
exits through the existing bounded-mechanism counters, especially across reset
boundaries. No leak or loss claim is supported.

Because arbitration and its retry state are now removed, no ambiguous
Phase 3B counter remains in production. The comparison helper preserves the
historical fields but now reports
`classified_by_phase3b_terminal_counters`,
`not_classified_by_phase3b_terminal_counters_at_snapshot`, and
`fully_classified_at_snapshot` with the exact semantics above.

## Comparison helper and acceptance report

`scripts/qwen3_coder_prompt2_same_host_compare.jq` defines a portable
`absolute` helper instead of jq's unavailable `abs` filter, so it executes
with jq 1.6-compatible syntax. The shell fixture runs the complete helper,
including a negative resident delta that exercises `absolute`.

Comparison schema version 2 adds a real `acceptance` object. It reports:

- same-host identity;
- streaming and resident qualifications and output parity;
- contemporaneous resident percent delta and ±2% status;
- short and medium throughput-regression status;
- streaming geometric-mean percent delta and the preferred +2% threshold;
- per-fixture mean and maximum demand-storage and layer-fetch critical-path
  changes;
- decode-wall fetch-fraction, cache-miss, and SSD-byte changes;
- prompt and decode deferred/resumed/drop accounting; and
- `policy_accepted` plus explicit rejection reasons.

The helper does not mutate or rewrite either source artifact. For the captured
Phase 3B result, `acceptance.policy_accepted` is `false`.

## Correctness checks

Run the complete macOS correctness set below. These checks validate restored
runtime behavior and artifact processing; they do not establish Linux
performance:

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

Also run `bash -n` on every changed shell file, compile and execute the jq
comparison fixture, run rustfmt on changed Rust files, and finish with
`git diff --check`.

The disposition commit passed the full macOS binary suite (`740` passed, `2`
ignored, `0` failed), all seven Phase 3A demand-fetch telemetry tests, the
resident-inactivity test, weighted combiner test, concurrent-resident direct-Q8
test, Candle-reference direct-Q8 parity test, resident-finalizer fixture, and
same-host comparison fixture. Rustfmt and shell syntax checks also passed.
These results are correctness and artifact-pipeline evidence only.

## Historical artifact handling

The qualified Phase 3A and Phase 3B streaming/resident directories and their
checksums must be retained unchanged. Reproduce comparisons read-only with:

```bash
scripts/compare_qwen3_coder_prompt2_same_host.sh \
  /path/to/prompt2-phase3a-same-host \
  /path/to/prompt2-phase3b-same-host \
  > /path/to/prompt2-phase3a-vs-phase3b-same-host.json
jq . /path/to/prompt2-phase3a-vs-phase3b-same-host.json
```

The resident collector intentionally preserves the historical reference
`3.500144036461` decode TPS geometric mean from commit
`327d263193c48f9dde9f6a716562260ab49fa7ef`. The replacement VM is known to
fall below that cross-VM gate. Keep `qualification_passed: false` and its
failure reason truthful. The comparison helper separately reports the
contemporaneous Phase 3A-to-3B resident delta and never rewrites either source.

The failed Phase 3B implementation commit and all captured benchmark artifacts
remain experimental evidence only. Production behavior is Phase 3A.
