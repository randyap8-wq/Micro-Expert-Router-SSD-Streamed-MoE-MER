# Mixtral 8x7B CPU Cache Scaling, 2026-06-27

This note records the verified CPU-only `run` expert-streaming benchmark
results for Mixtral 8x7B expert streaming collected on 2026-06-27. These
are not full autoregressive decoder-inference numbers. The benchmark
performs real Q4_0 SwiGLU expert FFN execution while exercising routing,
RAM cache lookup, SSD reads, prefetching, and run-summary telemetry.
`sustained_tps` means benchmark iterations per second, not generated
language tokens per second.

The defensible central claim from this run is that MER retained most of
the measured expert-FFN benchmark throughput while caching only a small
fraction of the expert namespace.

## Hardware And Model Configuration

| Field | Value |
|---|---|
| Date | 2026-06-27 |
| Cloud | GCP |
| Machine | `g2-standard-32` |
| CPU allocation | 32 vCPUs / 16 physical cores |
| RAM | 128 GB |
| Storage | GCP local NVMe SSD |
| GPU | NVIDIA L4 attached, not used by these CPU benchmark runs |
| Model file | `mixtral-8x7b-instruct-v0.1.Q4_0.gguf` |
| Extracted format | Native Q4_0 expert blobs produced by `gguf-convert` |
| Expert namespace | 256 layer-qualified experts |
| Experts per layer | 8 |
| Layers | 32 |
| Routing | top-2 |
| `d_model` | 4096 |
| `d_ff` | 14336 |
| Expert size | 99,090,432 bytes, approximately 94.5 MiB |
| Primary run length | 10,000 benchmark iterations |
| Workload | skewed |
| Zipf parameter | 1.2 |
| Workload correlation | 0.7 |
| Seed | 42 |
| Locality | enabled |
| Affinity | enabled |
| Adaptive prefetch governor | enabled |
| Neural speculator | disabled |
| I/O | `io_uring` |
| SSD-backed execution | forced with `--force-ssd` |

## Exact Command And Configuration

The common primary command, parameterized by cache size, was:

```bash
RUST_LOG=info \
./target/release/micro-expert-router run \
  --data-dir /mnt/localssd/data/mixtral-q4 \
  --cache-slots <16|32|64|124> \
  --tokens 10000 \
  --predict-fanout 4 \
  --pipeline-depth 4 \
  --io-uring \
  --locality \
  --affinity \
  --affinity-neighbors-k 2 \
  --affinity-decay-epoch 4096 \
  --num-layers 32 \
  --num-experts-per-layer 8 \
  --workload skewed \
  --zipf-s 1.2 \
  --workload-correlation 0.7 \
  --prefetch-governor \
  --seed 42 \
  --force-ssd
```

`--speculator` was omitted and therefore disabled.

The short boundary and control tests used 2,000 iterations and
explicitly set:

```bash
RAYON_NUM_THREADS=30
```

The primary 10,000-iteration logs do not establish that
`RAYON_NUM_THREADS` was explicitly set for those runs.

## Primary 10,000-Iteration Results

| Expert cache | Namespace cached | Approx. expert-cache payload | Iterations | Sustained benchmark iterations/s | Hit rate | Avg. compute/iteration | Avg. I/O wait/iteration | I/O share |
| -----------: | ---------------: | ---------------------------: | ---------: | -------------------------------: | -------: | ---------------------: | ----------------------: | --------: |
|     16 slots |            6.25% |                     1.48 GiB |     10,000 |                11.78248398546161 |  86.775% |              56.089 ms |               28.578 ms |    33.74% |
|     32 slots |           12.50% |                     2.95 GiB |     10,000 |               13.502514435722702 |  91.625% |              55.486 ms |               18.369 ms |    24.86% |
|     64 slots |           25.00% |                     5.91 GiB |     10,000 |                8.865952914887854 |  94.615% |              96.334 ms |               11.965 ms |    11.04% |
|    124 slots |           48.44% |                    11.44 GiB |     10,000 |               15.080930209605933 |  96.675% |              58.358 ms |                7.519 ms |    11.41% |

The best observed primary run was 15.080930209605933 benchmark
iterations per second with 124 cache slots.

## Low-Memory Efficiency Analysis

The most important low-memory result is the 16-slot run:

| Metric | Value |
|---|---:|
| Namespace cached | 6.25% of the 256-expert namespace |
| Approx. expert-cache payload | 1.48 GiB |
| Sustained benchmark iterations/s | 11.78248398546161 |
| Cache hit rate | 86.775% |

Relative to the 124-slot run, the 16-slot configuration used 12.9% as
much cache capacity and retained 78.1% of the measured throughput. The
32-slot configuration used 25.8% as much cache capacity as 124 slots and
retained 89.5% of the measured throughput.

This supports the intended MER design goal: useful expert execution
without keeping most of the expert namespace in RAM. High routing
locality allows a small resident working set to serve most lookups, and
SSD reads handle the remaining cold expert activations.

## Historical June 25 Comparison

The June 25 and June 27 configurations were not identical, so this is a
cautious historical comparison rather than an isolated causal claim.

| Metric | June 25 16 slots | June 27 16 slots |
|---|---:|---:|
| `sustained_tps` | 2.9101667108676104 | 11.78248398546161 |
| Hit rate | 79.635% | 86.775% |
| Average I/O wait | 251.5833 ms | 28.5783 ms |
| Completed prefetches | 21,353 | 30 |
| Speculative I/O behavior | speculator enabled | speculator disabled; adaptive governor enabled |

The June 27 configuration produced approximately 4.05x the historical
16-slot throughput while dramatically reducing speculative SSD traffic
and foreground I/O wait. Attribute the difference to the updated
benchmark configuration and much stricter speculative-I/O behavior, not
to one isolated code change.

The June 25 report remains preserved as
[`mixtral-8x7b-cpu-cache-scaling-2026-06-25.md`](mixtral-8x7b-cpu-cache-scaling-2026-06-25.md).

## Known Bimodal FFN Compute Anomaly

Do not interpret the primary table as a normal monotonic cache-scaling
curve. The 64-slot cache achieved a healthy 94.615% hit rate and average
foreground I/O wait was only 11.9648 ms. Its lower
8.865952914887854 benchmark iterations per second result was caused by
average FFN compute time rising to 96.3341 ms.

Normal fast-state FFN compute in the 16-, 32-, and mostly the 124-slot
runs was approximately 55-58 ms. Repeated short tests between 60 and 68
slots showed compute p50 values near 98-100 ms. A 32-slot control
remained fully in the fast state. A 124-slot control was bimodal: p50
remained fast while p95 approached 99 ms.

The issue is under investigation. Possible memory placement, buffer
identity, worker scheduling, NUMA locality, or kernel behavior remain
hypotheses, not established conclusions. The 64-slot result is included
for transparency and should not be interpreted as proof that larger
caches inherently reduce performance.

## Boundary And Control Tests

### Boundary Tests

| Cache slots | Iterations |     TPS | Hit rate | Compute p50 | Avg. compute | Avg. I/O wait |
| ----------: | ---------: | ------: | -------: | ----------: | -----------: | ------------: |
|          60 |      2,000 | 10.3950 |   94.05% |  100.415 ms |    80.779 ms |     12.825 ms |
|          63 |      2,000 |  9.2282 |   94.15% |   98.047 ms |    91.571 ms |     12.683 ms |
|          64 |      2,000 |  9.2109 |   94.18% |   98.943 ms |    91.510 ms |     12.622 ms |
|          65 |      2,000 |  9.3160 |   94.23% |   97.663 ms |    90.273 ms |     13.118 ms |
|          68 |      2,000 |  9.3353 |   94.38% |   98.303 ms |    90.634 ms |     12.581 ms |

This rules out a cliff occurring only at exactly 64 slots. The slow
state covers at least the tested 60-68 range.

### Interleaved Controls

| Cache slots | Iterations |     TPS | Hit rate | Compute p50 | Compute p95 | Compute p99 | Avg. compute | Avg. I/O wait |
| ----------: | ---------: | ------: | -------: | ----------: | ----------: | ----------: | -----------: | ------------: |
|          32 |      2,000 | 12.5221 |   90.30% |   55.359 ms |   57.951 ms |   59.615 ms |    55.923 ms |     23.700 ms |
|         124 |      2,000 | 12.8629 |   95.65% |   56.447 ms |   99.007 ms |  100.351 ms |    66.500 ms |      9.977 ms |

The 32-slot control remained entirely in the fast compute state. The
124-slot control was bimodal: fast median, slow p95/p99. This makes a
global persistent VM slowdown unlikely, but the mechanism remains
unresolved. The approximate slow-event share may be inferred from the
average and percentile values, but it was not measured directly here and
should not be quoted as a measured fact.

## Metric Semantics

These definitions are based on the current Rust implementation:

| Metric | Meaning |
|---|---|
| `sustained_tps` | `tokens_processed / wall_time` for the `run` benchmark. It is benchmark iterations per second, not generated language token production. |
| Hit rate | `hits / (hits + misses)` across routed expert activations. |
| I/O reads | Foreground critical-path read samples recorded in the I/O latency histogram. The `reads` field printed in the run summary comes from `r.io_count`. |
| Bytes read | Engine-wide `bytes_read`, including foreground bytes and speculative prefetch bytes. |
| Foreground I/O latency | Histogram samples for SSD reads that block routed work, recorded around `fetch_once`. |
| Average I/O wait | Aggregate critical-path wait for routed misses divided by processed benchmark iterations. |
| Average compute | Average time spent executing the real Q4_0 SwiGLU expert FFN work per benchmark iteration. |
| I/O share | `total_io_wait_us / total_cycle_us`, the percentage of benchmark cycle time spent waiting on foreground SSD reads. |
| Governor precision | EWMA of recent `prefetch_used / prefetch_completed`, folded by the prefetch governor over measurement windows, not a lifetime ratio. |
| Prefetch completed | Speculative prefetch reads admitted and completed by the engine. |
| Prefetch used | Completed speculative prefetches that were later consumed by a routed activation. |

High `avg_throughput_mibps` is not automatically better because it can
reflect more cache misses or unnecessary speculative traffic rather than
higher useful benchmark throughput.

## Full Raw Summaries

### 16 Cache Slots

```text
wall_s=848.717470131
sustained_tps=11.78248398546161
avg_throughput_mibps=297.85877882274593
hit_rate_pct=86.775
hits=17355
misses=2645
prefetches_completed=30
predictor_observations=0
reads=2645
bytes=252797.95 MiB
io_p50=116735us
io_p95=242175us
io_p99=259327us
compute_p50=55935us
compute_p95=57695us
compute_p99=61823us
cycle_p50=56159us
cycle_p95=183039us
cycle_p99=304127us
cycle_max=844799us
avg_io_wait=28578.3us
avg_compute=56089.1us
io_share=33.74%
pinned=15
locality_hit_rate=82.14%
ssd_stall=285782.8ms
governor_precision=4.29%
prefetch_used=0/30
governor_admitted=30
governor_throttled=159115
prefetch_dropped_governor=159115
```

### 32 Cache Slots

```text
wall_s=740.602800138
sustained_tps=13.502514435722702
avg_throughput_mibps=223.68987482275355
hit_rate_pct=91.625
hits=18325
misses=1675
prefetches_completed=80
predictor_observations=0
reads=1673
bytes=165665.35 MiB
singleflight_dedup_followers=2
io_p50=116927us
io_p95=239231us
io_p99=258559us
compute_p50=55327us
compute_p95=56767us
compute_p99=57919us
cycle_p50=55391us
cycle_p95=176255us
cycle_p99=285183us
cycle_max=1200127us
avg_io_wait=18369.4us
avg_compute=55486.0us
io_share=24.86%
pinned=31
locality_hit_rate=82.14%
ssd_stall=183693.9ms
governor_precision=2.39%
prefetch_used=2/80
governor_admitted=80
governor_throttled=146456
prefetch_dropped_governor=146456
```

### 64 Cache Slots

```text
wall_s=1127.910343761
sustained_tps=8.865952914887854
avg_throughput_mibps=90.23829561830577
hit_rate_pct=94.615
hits=18923
misses=1077
prefetches_completed=0
predictor_observations=0
reads=1077
bytes=101780.71 MiB
io_p50=117247us
io_p95=230399us
io_p99=249599us
compute_p50=99967us
compute_p95=102143us
compute_p99=103295us
cycle_p50=100031us
cycle_p95=218879us
cycle_p99=227839us
cycle_max=382975us
avg_io_wait=11964.8us
avg_compute=96334.1us
io_share=11.04%
pinned=45
locality_hit_rate=82.14%
ssd_stall=119647.6ms
governor_precision=50.00%
prefetch_used=0/0
governor_admitted=0
governor_throttled=32125
prefetch_dropped_governor=32125
```

### 124 Cache Slots

```text
wall_s=663.089070834
sustained_tps=15.080930209605933
avg_throughput_mibps=94.77625317697756
hit_rate_pct=96.675
hits=19335
misses=665
prefetches_completed=0
predictor_observations=0
reads=665
bytes=62845.10 MiB
io_p50=117055us
io_p95=229375us
io_p99=242815us
compute_p50=56159us
compute_p95=61823us
compute_p99=101631us
cycle_p50=56191us
cycle_p95=173055us
cycle_p99=187007us
cycle_max=385791us
avg_io_wait=7518.9us
avg_compute=58358.4us
io_share=11.41%
pinned=45
locality_hit_rate=82.14%
ssd_stall=75189.3ms
governor_precision=50.00%
prefetch_used=0/0
governor_admitted=0
governor_throttled=21457
prefetch_dropped_governor=21457
```

## Current Conclusions

MER retained most of the measured expert-FFN benchmark throughput while
caching only a small fraction of the expert namespace. The 16-slot
configuration is the headline low-memory result: 11.78248398546161
benchmark iterations per second with approximately 1.48 GiB of cached
expert payload and only 6.25% of the namespace resident. The best
observed primary run was the 124-slot run at 15.080930209605933
benchmark iterations per second.

The 64-slot result is transparent evidence of the current bimodal FFN
compute anomaly, not evidence that larger caches inherently reduce
performance. Until the anomaly is explained, the primary table should be
read as observed benchmark evidence rather than as a smooth cache-size
scaling curve.

## Reproducibility Limitations And Next Investigation Steps

The primary runs share the same documented high-level command,
configuration, seed, workload shape, model blobs, and machine class, but
they remain VM benchmark observations. Storage virtualization, scheduler
state, memory placement, kernel behavior, and worker-pool state can
affect measurements.

Next investigation steps:

| Area | Follow-up |
|---|---|
| Bimodal compute | Instrument buffer addresses, allocation identity, thread placement, worker ids, and per-worker compute histograms. |
| Cache boundary | Repeat 48-80 slot sweeps with multiple seeds and fresh processes. |
| Controls | Repeat 16-, 32-, and 124-slot controls around every slow-state run. |
| Host locality | Capture NUMA placement, CPU affinity maps, and kernel scheduler counters. |
| Benchmark reporting | Keep reporting benchmark iterations per second separately from full model inference throughput. |
