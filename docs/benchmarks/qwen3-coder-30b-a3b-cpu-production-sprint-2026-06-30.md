# Qwen3-Coder 30B A3B CPU Production Sprint, 2026-06-30

This report records the implementation status for the CPU-production
optimization sprint on branch `cpu-production-sprint`.

No throughput gain is claimed here. The mandatory real
`Qwen3-Coder-30B-A3B-Instruct Q8_0` checkpoint benchmark is blocked in
this workspace because the full checkpoint or converted dense/expert
directory is not present locally. The only Qwen-related local path found
was `/Users/r/model-configs/qwen3-moe`, which contains `config.json` and
`model.safetensors.index.json` metadata stubs, not weight shards or
converted MER expert blobs.

## Implementation Status

Implemented and committed:

| Area | Status |
|---|---|
| Strict real checkpoint loading | Complete. Production strict mode aggregates missing, malformed, unsupported, and shape-mismatched tensors instead of silently retaining seeded fallback weights. |
| Prompt/decode semantics | Complete. Real-model prompt ingestion evaluates the model `P + C - 1` times and evaluates the LM head only for completion tokens. |
| Real benchmark harness | Complete. `bench-real` reports prompt/decode TPS, forward counts, cache stats, SSD bytes/stall, stage timing snapshots, RSS, thread count, build features, and git commit. |
| Stage timing metrics | Complete. HTTP serving now keeps detailed stage timing opt-in, while `bench-real` continues to collect detailed stage timings for benchmark runs. |
| Scheduler request lifetime | Complete. Real requests register scheduler KV state once, release on completion/error/drop, and skip unhelpful singleton pre-pass work. |
| CPU hot-path cleanup | Complete for the scoped sprint items: runtime dense matvec backend selection, shared RoPE cache, request scratch buffers, router top-k cleanup, and allocation-count smoke benchmark. |
| Native quantized dense backbone | Complete for Q8_0/F32. GGUF conversion preserves native Q8_0 resident dense tensors through `dense_manifest.json`; legacy F32 converted directories remain supported. |
| Quantized embedding and LM head | Complete for Q8_0. Embeddings dequantize only the requested row. Greedy and top-k LM-head paths scan quantized rows directly and merge deterministic per-thread candidates without a vocab-sized logits allocation. |
| Prepared Q8 expert execution | Complete. Q8_0 expert QMatMul preparation is cached once per resident, preparation latency is timed, malformed preparation is cached as an error, expert execution policy is explicit, and weighted MoE accumulation avoids materializing all top-k outputs before combining. |
| Optional real prefill | Deferred. The sprint brief says not to begin prefill until optimized single-stream decode is benchmarked. |
| Acceptance benchmarks | Blocked by missing local Qwen checkpoint/converted weights. |

Validation commit for this corrective pass:

```text
a93317cb184569fbd2172b0ff5b69d349fd2c96e fix: gate stage timing and protect q8 gpu fallback
```

This is the code/test commit on which the local validation commands
below were run. The documentation-only commit that updates this report is
not itself claimed as a code validation run.

## Validation Run Locally

The following checks were run against validation commit
`a93317cb184569fbd2172b0ff5b69d349fd2c96e` in this workspace:

```bash
rustfmt --edition 2021 src/config.rs src/server.rs src/transformer.rs
cargo test dense_tensor -- --nocapture
cargo test stage_timing -- --nocapture
cargo test config -- --nocapture
cargo test
cargo build --release
cargo build --release --features "avx512,blas,tokenizer"
git diff --check
```

Notes:

- Release builds still report the repository's existing warning backlog.
  No new warning from this corrective pass was left in the build output.
- A repository-wide formatting rewrite was intentionally not applied; only
  the Rust files touched by the corrective pass were formatted.
- CUDA, Linux `io_uring`, and real checkpoint benchmark commands were not
  run in this macOS workspace.
- The real Qwen checkpoint benchmark remains blocked, so this document is
  not a performance result and claims no TPS improvement.

## Mandatory Benchmark Blocker

`bench-real` intentionally refuses seeded fallback production
measurements:

```text
bench-real requires real_transformer.weights_dir; seeded fallback benchmarks are not production measurements
```

To complete Milestone 6, provide or mount one of:

- a converted MER directory containing `config.json`, `metadata.json`,
  layer-qualified `expert_<layer>_<expert>.bin` or packed expert files,
  `dense_manifest.json`, and any canonical dense tensor payloads; or
- the full original checkpoint directory with `config.json`,
  tokenizer files, and all safetensors/GGUF shards needed by the loader
  and converter.

## Reproduction Commands Once Weights Exist

Use a CPU-only config with the real Qwen paths filled in:

```toml
[server]
bind = "127.0.0.1:0"
max_tokens = 256
session_ttl_secs = 0
max_concurrent_requests = 0
admission_min_free_blocks = 0

[sampling]
temperature = 0.0
top_p = 1.0
top_k = 0
seed = 0

[model]
data_dir = "/path/to/converted/qwen3-coder-30b-a3b-q8"
num_experts = 128
top_k = 8
d_model = 2048
d_ff = 768
expert_size = 0 # replace with converted metadata value
num_layers = 48
dtype = "q8_0"

[storage]
cache_slots = 1536 # rerun with 6144 for all-resident isolation
block_align = 4096
no_direct = false
predict_fanout = 2
pipeline_depth = 3
predict_min_prob = 0.05
partial_load_fraction = 1.0
pin_after_observations = 0

[tokenizer]
path = "/path/to/qwen3-coder-30b-a3b/tokenizer.json"

[real_transformer]
enabled = true
weights_dir = "/path/to/converted/qwen3-coder-30b-a3b-q8"
strict_weights = true
architecture = "qwen3_moe"
vocab_size = 151936
num_heads = 32
num_kv_heads = 4
head_dim = 128
rope_base = 10000000.0
rms_eps = 1e-6
max_batch_size = 1
batch_timeout_ms = 0
compute_offload = "cpu"
dense_matvec_backend = "rayon-matrixmultiply"
expert_execution_policy = "auto"

[predictive]
locality_enabled = false
speculator_enabled = false
affinity_enabled = false
prefetch_governor = false
cost_aware_eviction = false
pregate_enabled = false
static_residency_fraction = 0.0
static_residency_warmup_tokens = 0

[gpu_cache]
enabled = false
vram_capacity_mb = 0
promote_after_hits = 0
vram_anchor_ratio = 0.5
dtype = "q8_0"
```

Then run the required CPU-only benchmark matrix:

```bash
# Build current feature set.
cargo build --release --features "cuda,avx512,blas,tokenizer,io_uring"

# Build no-blas CPU dispatch.
cargo build --release --no-default-features --features "cuda,avx512,tokenizer,io_uring"

# cache_slots=1536, realistic out-of-core run.
./target/release/micro-expert-router bench-real \
  --config /path/to/qwen-cpu-1536.toml \
  --prompt "Write a small Rust function that checks whether a string is a palindrome." \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measured-runs 5 \
  --cache-reset keep \
  --greedy \
  --format json \
  > ../docs/benchmarks/qwen3-coder-30b-a3b-cpu-1536-after.json

# cache_slots=6144, all-resident compute isolation run.
./target/release/micro-expert-router bench-real \
  --config /path/to/qwen-cpu-6144.toml \
  --prompt "Write a small Rust function that checks whether a string is a palindrome." \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measured-runs 5 \
  --cache-reset keep \
  --greedy \
  --format json \
  > ../docs/benchmarks/qwen3-coder-30b-a3b-cpu-6144-after.json
```

For before/after comparison, rerun the same two `bench-real` commands at
baseline commit `af13e6647b625fca7b07a8f4ebe0c6aeea158569` if that
commit can load the converted checkpoint, or at the earliest sprint
commit that can run `bench-real` without seeded fallback. Record that
choice in the JSON filenames and report.

## Decision Gate

The sprint decision is intentionally left open until the real checkpoint
benchmarks exist. Per the brief:

- continue CPU work if native dense Q8 plus prepared Q8 experts reach at
  least 1.5 decode TPS on the target VM or at least 3x total decode
  speedup over the original sustained baseline; and
- pivot primary performance work to GPU if decode remains below 1 TPS
  or below 3x the original sustained baseline after these CPU changes.
