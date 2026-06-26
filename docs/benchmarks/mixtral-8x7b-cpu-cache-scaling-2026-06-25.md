# Mixtral 8x7B CPU Cache Scaling, 2026-06-25

This note preserves the latest observed CPU-only `run` benchmark results
for Mixtral 8x7B expert streaming. These are not full autoregressive
decoder-inference numbers. The `run` benchmark executes real Q4_0
SwiGLU expert FFNs while exercising routing, RAM cache lookup, SSD
reads, prefetching, and run-summary telemetry. Here, `sustained_tps`
means benchmark iterations per second, not generated language tokens per
second.

## Hardware And Model

| Field | Value |
|---|---|
| Date | 2026-06-25 |
| Host | GCP `g2-standard-32` |
| CPU | 32 vCPUs / 16 physical cores |
| RAM | 128 GB |
| Storage | GCP local SSD |
| GPU | NVIDIA L4 attached, not used by these CPU-path runs |
| Model source | `mixtral-8x7b-instruct-v0.1.Q4_0.gguf` |
| Extracted format | Native Q4_0 expert blobs |
| Expert namespace | 256 layer-qualified expert files |
| Routing | top-2 |
| Expert size | 99,090,432 bytes, approximately 94.5 MiB |

## Results

| Expert cache | Namespace cached | Approx. expert-cache payload | Iterations | Sustained benchmark TPS | Hit rate | Avg. I/O wait/token | I/O share |
| -----------: | ---------------: | ---------------------------: | ---------: | ----------------------: | -------: | ------------------: | --------: |
|     16 slots |            6.25% |                     1.48 GiB |     10,000 |      2.9101667108676104 |   79.64% |            251.6 ms |    73.86% |
|     64 slots |              25% |                     5.91 GiB |     30,000 |       7.780785593717826 |   94.71% |             16.0 ms |    12.99% |
|    128 slots |              50% |                    11.81 GiB |     10,000 |       9.445934926474694 |   96.83% |              7.7 ms |     7.55% |

The 64-slot run used 30,000 iterations, while the 16- and 128-slot
runs used 10,000. Treat these as latest observed runs, not a perfectly
controlled benchmark suite. A formal comparison should rerun every cache
size with the same commit, seed, flags, workload trace, warm-up policy,
and iteration count.

The synthetic benchmark router uses Markov-style routing labels that are
not derived from `synth_hidden_state`. The neural speculator top-1
accuracy staying near random for a 256-expert namespace is therefore
expected and should not be interpreted as evidence that the speculator
caused the observed hit rates.

## Metric Semantics

These definitions are based on the current Rust implementation:

| Metric | Meaning |
|---|---|
| Sustained TPS | `tokens_processed / wall_time` for the `run` benchmark. It is benchmark iterations per second, not generated-token throughput. |
| Hit rate | `hits / (hits + misses)` across routed expert activations. |
| I/O reads | Foreground critical-path read samples recorded in the I/O latency histogram. The `reads` field printed in the run summary comes from `r.io_count`. |
| Bytes | Engine-wide `bytes_read`, including foreground bytes and speculative prefetch bytes. Do not treat `avg_throughput_mibps` or bytes alone as a higher-is-better score because speculative traffic can raise bytes while lowering TPS. |
| Foreground I/O latency | Histogram samples for SSD reads that block routed work, recorded around `fetch_once`. |
| Avg. I/O wait/token | Aggregate critical-path wait for routed misses divided by processed benchmark iterations. |
| I/O share | `total_io_wait_us / total_cycle_us`, the percentage of benchmark cycle time spent waiting on foreground SSD reads. |
| Governor precision | EWMA of recent `prefetch_used / prefetch_completed`, folded by the prefetch governor over measurement windows, not a lifetime ratio. |

## Raw Summaries

### 64 cache slots, CPU only, 30,000 iterations

```text
sustained_tps=7.780785593717826
hit_rate_pct=94.71000000000001
hits=56826
misses=3174
prefetches completed=6579
i/o reads=3162
i/o p50=119871us p95=358911us p99=671743us
compute p50=101823us p95=105471us p99=106623us
cycle p50=107391us p95=227967us p99=358655us
avg io_wait=16018.0us
avg compute=101913.6us
I/O share=12.99%
speculator top-1 accuracy approximately 0.28%
governor prefetch_used=185/6579
```

### 16 cache slots, CPU only, 10,000 iterations

```text
sustained_tps=2.9101667108676104
hit_rate_pct=79.635
hits=15927
misses=4073
prefetches completed=21353
i/o reads=3886
i/o p50=634367us p95=1278975us p99=2068479us
compute p50=99391us p95=102655us p99=104127us
cycle p50=105791us p95=1090559us p99=1524735us
avg io_wait=251583.3us
avg compute=83544.1us
I/O share=73.86%
speculator top-1 accuracy approximately 0.29%
governor prefetch_used=198/21353
```

### 128 cache slots, CPU only, 10,000 iterations

```text
sustained_tps=9.445934926474694
hit_rate_pct=96.83
hits=19366
misses=634
prefetches completed=170
i/o reads=634
i/o p50=117055us p95=233599us p99=561151us
compute p50=99519us p95=101695us p99=102655us
cycle p50=105023us p95=179711us p99=226687us
avg io_wait=7699.6us
avg compute=88898.7us
I/O share=7.55%
speculator top-1 accuracy approximately 0.27%
governor prefetch_used=43/170
```
