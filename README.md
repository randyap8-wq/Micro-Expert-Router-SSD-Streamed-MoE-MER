# MER — Micro-Expert Router

**Constrained-memory Mixture-of-Experts inference in Rust.**

MER is an experimental inference runtime for sparse Mixture-of-Experts (MoE) models that treats **NVMe/SSD, system RAM, and GPU VRAM as one managed expert-residency hierarchy**.

Instead of requiring every routed expert to remain permanently in VRAM, MER keeps expert weights in lower-cost storage tiers and materializes the experts needed by the active route onto the execution tier. The current GPU path uses **WGPU/Vulkan** for routed-expert execution while dense transformer work, attention, KV state, routing, and orchestration can remain on the host.

The project started as a CPU/SSD-streaming engine. That work is still part of MER, but it is now the storage and fallback foundation for a broader constrained-memory inference architecture.

> **Project status:** pre-release systems research. MER has real hardware qualification and full-model execution paths, but not every feature or model family is production-qualified. Performance claims are only made where a controlled benchmark explicitly authorizes them.

<p align="center">
  <a href="https://amalgafy.com">
    <img src="https://raw.githubusercontent.com/randyap8-wq/Clean-Shot/main/public/amalgafy-icon.svg" alt="Amalgafy" width="160" />
  </a>
</p>

---

## Why MER exists

Large sparse MoE models have a useful property: only a small subset of experts is active for each token.

The conventional deployment answer is still often to provision enough accelerator memory for the whole expert set—or to accept expensive transfers and framework-level offload when it does not fit.

MER explores a different execution model:

\`\`\`text
                    ┌───────────────────────────┐
                    │  Dense / attention plane  │
                    │ CPU today; GPU work may   │
                    │ expand independently      │
                    └─────────────┬─────────────┘
                                  │
                           router selects
                                  │
                                  ▼
NVMe / SSD  ──▶  Host expert cache  ──▶  Physical GPU residency  ──▶  Q4 routed experts
 backing          RAM / generations       bounded VRAM                WGPU / Vulkan
 store                  │                        │
                        └──── recovery / install ┴──── constrained-memory execution
\`\`\`

The goal is not “CPU inference instead of GPU inference.” The goal is to make **expert residency itself virtualized and explicit**, so useful GPU execution remains possible when the routed-expert working set is larger than the VRAM budget.

---

## Current architecture

MER is a Rust runtime with separate storage, routing, residency, execution, and observability planes.

### Expert storage and host residency

- Expert weights can live outside accelerator memory and be brought in on demand.
- Linux storage paths support page-aligned direct I/O and optional \`io_uring\`.
- A host expert cache tracks payloads, generations, residency, and eviction independently from physical GPU state.
- Packed expert storage, cache policies, predictive prefetching, and historical CPU streaming paths remain available for storage-oriented experiments.

### GPU-native routed experts

- WGPU is compiled into the standard runtime and can use Vulkan-backed hardware adapters.
- Routed experts can execute on the GPU while attention, router, dense projections, KV, embeddings, and LM head remain on the CPU.
- MER tracks **logical admission** separately from **physical device residency**.
- Generation checks prevent stale logical admissions from being treated as valid GPU-resident weights.
- The GPU expert registry is bounded by a configured expert-weight budget; workspaces are tracked separately.
- Explicit GPU modes fail closed when the required device or execution contract cannot be satisfied.

### Quantized expert execution

The current hardware-qualified routed-expert path includes canonical **GGML Q4_0** execution. MER has qualified the production WGPU/Vulkan Q4_0 path against an authoritative CPU reference using an NVIDIA L4.

See:

- [Qwen3-Coder 30B-A3B PR6 Q4_0 numerical parity](docs/benchmarks/qwen3-coder-30b-a3b-pr6-q4-parity-2026-08-11.md)

That qualification established raw shader parity, complete expert parity, physical residency reuse, GPU I/O completion, and strict no-fallback behavior for the qualified checkpoint and adapter. It was a **correctness qualification, not a TPS claim**.

### Recovery under constrained VRAM

MER does not assume a routed expert will already be physically current on the GPU.

The runtime can checkpoint execution at the failed expert boundary, service the missing residency, and resume from the exact checkpoint rather than replaying already-completed work. This recovery model is important in the small-VRAM regime where many expert demands are expected to miss ordinary physical residency.

---

## Current hardware target

The primary current qualification target is:

| Field | Current target |
|---|---|
| Model | Qwen3-Coder-30B-A3B-Instruct |
| Architecture | \`qwen3_moe\` |
| Routed expert dtype | Q4_0 |
| Transformer layers | 48 |
| Experts per layer | 128 |
| Global routed experts | 6,144 |
| Top-K | 8 |
| Hidden dimension | 2,048 |
| Expert FFN dimension | 768 |
| Qualified GPU | NVIDIA L4 |
| GPU API | WGPU / Vulkan |
| Current constrained-VRAM research point | 2 GiB expert budget |

A release-grade model-quality claim is separate from runtime qualification. Some historical Q4 artifacts were requantized for systems validation; the README does not treat those artifacts as a model-quality benchmark.

---

## Predictor-v2: predictive physical expert movement

MER's current development lane goes beyond reacting to physical misses.

**Predictor-v2** observes committed routing truth and predicts a future routed expert for a later position. When the prediction has a valid host-backed payload and the ordinary GPU-residency path does not already satisfy it, MER can place that expert into an **isolated sidecar GPU residency slot** before matching future demand reaches it.

This is different from the older generic SSD prefetch logic:

- it predicts a concrete future routed expert;
- it operates on committed route observations;
- it uses exact expert identity and generation;
- movement has its own accounting lifecycle;
- ordinary demand/residency remains authoritative;
- a prediction receives credit only when real matching demand consumes the moved residency.

### Latest certified Predictor-v2 result

Predictor-v2 is currently a **validated development branch**, not yet part of frozen \`main\`.

The P1O certification on an NVIDIA L4 completed successfully with:

| Certification field | Result |
|---|---:|
| Planned positions | 143 |
| Generated tokens | 128 |
| Generated-output exact parity | PASS |
| Semantic route parity | PASS |
| Evidence-structure parity | PASS |
| Predictor-v2 movement lifecycles emitted | 13 |
| Consumed by matching future route | 9 |
| Evicted unused | 4 |
| Direct matching demand credits | 9 |
| Source / install / accounting failures | 0 |
| Ordinary runtime invariants | PASS |
| Controlled shutdown | PASS |

The control and treatment generated the same token sequence and preserved the same predictor/route semantics. Mechanical cache state was allowed to diverge because successful movement can legitimately change later ordinary residency behavior.

Authoritative development record:

- [Issue #197 — Predictor-v2 P1O semantic-route parity qualifier](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/197)

### What is *not* claimed yet

The P1O certification deliberately did **not** authorize a performance verdict because its control and treatment did not have identical GPU resource footprints.

The next performance lane is building a matched-resource control and calibrating runtime noise before an A/B throughput claim is allowed.

- [Issue #198 — matched-resource performance experiment design](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/198)
- [Issue #199 — matched-resource performance harness](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/199)

Until that work closes, Predictor-v2 should be described as **correctness/mechanism certified**, not “faster.”

---

## What is on \`main\` vs. current development work?

MER is developed with strict qualification branches, so the newest validated research can be ahead of the frozen production branch.

### Frozen \`main\`

The mainline runtime contains the core MER execution architecture, including:

- Rust orchestration and full-transformer paths;
- SSD/RAM expert streaming and cache management;
- WGPU/Vulkan GPU backend;
- bounded physical GPU expert residency;
- GPU-native Q4_0 routed-expert execution;
- strict hardware qualification seams;
- checkpointed physical-residency recovery;
- OpenAI-style serving infrastructure and telemetry;
- optional \`io_uring\`, distributed expert sharding, and historical CPU execution paths.

### Qualified development work

Predictor-v2 sidecar movement and its current certification/performance harness live on dedicated development branches until the associated gates are complete.

This distinction is intentional: MER does not merge a mechanism merely because it compiles or produces a promising benchmark.

---

## Execution hierarchy

At a high level, MER treats expert execution as a hierarchy rather than a single cache:

\`\`\`text
                   route demand
                       │
                       ▼
              ┌──────────────────┐
              │ ordinary GPU     │
              │ physical expert  │
              │ residency        │
              └────────┬─────────┘
                       │ miss
                       ▼
              ┌──────────────────┐
              │ host expert      │
              │ payload / cache  │
              └────────┬─────────┘
                       │ miss
                       ▼
              ┌──────────────────┐
              │ NVMe / SSD       │
              │ backing store    │
              └──────────────────┘

Predictor-v2 development path:

committed route p
      │
      ▼
future expert prediction
      │
      ├── already current ──▶ no movement
      │
      └── host-backed + absent
                 │
                 ▼
        isolated GPU sidecar
                 │
                 ▼
       matching future demand
\`\`\`

Logical host admission and physical GPU residency are intentionally separate concepts. Metrics that report logical GPU admission should not be interpreted as proof that an expert is physically resident on the device.

---

## Model support

MER has architecture-aware loading and execution code for multiple model families, but **implemented parsing is not the same thing as hardware qualification**.

| Status | Model families / scope |
|---|---|
| Current primary qualification target | Qwen3-Coder-30B-A3B / \`qwen3_moe\` |
| Implemented MoE architecture paths | Mixtral/Llama-style MoE, Qwen3-MoE, DeepSeek-V3/V3.1, MiMo-V2-Flash, GPT-OSS paths |
| Dense paths | Qwen3 dense, Mistral Small 3, Phi-family dense loading/execution paths |
| Historical benchmark target | Mixtral 8x7B |
| Not implied | Arbitrary GGUF recipes, arbitrary sparse MoEs, or every checkpoint in a supported family |

Use model-specific qualification evidence before treating a checkpoint as supported for production use.

---

## Build

The engine lives in [\`rust-engine/\`](rust-engine/).

A current portable release build is:

\`\`\`bash
cd rust-engine
cargo build --release --features tokenizer
\`\`\`

Run the software test suite with:

\`\`\`bash
cargo test --locked --features tokenizer
\`\`\`

Linux deployments that intentionally use the direct-I/O reactor can add the optional \`io_uring\` feature. CPU kernel features such as \`avx512\` are opt-in and runtime-gated.

WGPU support is part of the normal runtime; the legacy \`gpu\` Cargo feature is retained only for backward compatibility and is a no-op.

The separate \`cuda\` feature enables the Candle CUDA path and is distinct from the WGPU/Vulkan routed-expert path.

---

## Configuration

The annotated root [\`config.toml\`](config.toml) documents the runtime configuration.

Important concepts:

- \`[real_transformer]\` enables real decoder execution instead of the legacy benchmark generator.
- \`[gpu_cache]\` controls logical GPU admission and the bounded physical routed-expert budget.
- \`[storage]\` controls the host expert cache and backing-store behavior.
- \`[sampling]\` controls deterministic or stochastic token sampling.
- \`[performance]\` contains host-side placement and worker controls.
- \`[distributed]\` configures optional expert partitioning across nodes.

For deterministic qualification, MER generally uses greedy sampling and explicit fail-closed hardware contracts.

For deployment and API details, see [\`docs/production.md\`](docs/production.md).

---

## Serving and observability

MER includes an OpenAI-style HTTP serving path and native telemetry.

The runtime exposes observability for:

- host expert-cache activity;
- logical GPU admission;
- physical GPU expert residency;
- promotions and evictions;
- source and install activity;
- routed-expert execution;
- request/runtime health.

The native terminal monitor can consume the health and metrics endpoints for a live view of storage and residency activity.

Production deployment guidance, authentication, rate limiting, health endpoints, and operational caveats live in:

- [\`docs/production.md\`](docs/production.md)

---

## CPU and SSD-streaming work still matters

MER's CPU-first history has not been discarded.

Those paths established much of the machinery the current runtime still uses:

- direct expert storage;
- page-aligned reads;
- host residency and cache policy;
- quantized expert formats;
- deterministic routing experiments;
- recovery and telemetry;
- full-transformer loading;
- architecture support;
- synthetic and real cache-pressure benchmarks.

What changed is the role of that work.

CPU-only execution is now **one execution/fallback mode and an important historical baseline**, not the project's primary identity.

Historical CPU benchmark reports remain available under [\`docs/benchmarks/\`](docs/benchmarks/), including:

- [Qwen3-Coder 30B-A3B Q8 CPU full-transformer validation](docs/benchmarks/qwen3-coder-30b-a3b-q8-cpu-2026-07-11.md)
- [Mixtral 8x7B CPU cache-scaling study](docs/benchmarks/mixtral-8x7b-cpu-cache-scaling-2026-06-27.md)

---

## Validation philosophy

MER separates four kinds of claims that are easy to conflate:

1. **Software validation** — tests, static contracts, build/scope checks.
2. **Numerical correctness** — GPU/CPU parity for a qualified kernel or expert path.
3. **Runtime mechanism correctness** — real model execution, resource accounting, recovery, route/output parity.
4. **Performance** — only after control/treatment resource footprints and runtime noise are explicitly matched.

A mechanism can therefore be considered correct without being described as faster.

That distinction is intentional and is reflected throughout the qualification issues and benchmark reports.

---

## Repository map

| Path | Purpose |
|---|---|
| [\`rust-engine/\`](rust-engine/) | Core Rust inference runtime |
| [\`rust-engine/src/backend/\`](rust-engine/src/backend/) | CPU/GPU execution backends |
| [\`config.toml\`](config.toml) | Annotated runtime configuration |
| [\`docs/production.md\`](docs/production.md) | Serving and deployment guidance |
| [\`docs/audit-findings.md\`](docs/audit-findings.md) | Runtime/model-loading audit notes |
| [\`docs/distributed.md\`](docs/distributed.md) | Distributed expert-sharding design and transport |
| [\`docs/benchmarks/\`](docs/benchmarks/) | Current and historical qualification/benchmark evidence |
| [\`scripts/\`](scripts/) | Model conversion, extraction, and support tooling |

---

## Current roadmap

Near-term work is deliberately narrow:

1. complete the matched-resource Predictor-v2 performance harness;
2. calibrate A/A runtime noise with fresh isolated runtimes;
3. run Predictor-v2 A/B only after the noise threshold is frozen;
4. continue hardware/model qualification without weakening exact parity and accounting gates;
5. only then consider broader activation in serving paths.

Longer-term work includes broader constrained-VRAM qualification, additional model families, expanded execution offload, and edge/hardware-specific deployment profiles.

---

## Historical benchmark index

The top-level README intentionally no longer reproduces every benchmark table. The detailed evidence remains in the repository.

Useful starting points:

- [Qwen3-Coder Q4_0 NVIDIA L4 numerical qualification](docs/benchmarks/qwen3-coder-30b-a3b-pr6-q4-parity-2026-08-11.md)
- [Qwen3-Coder Q8 CPU full-transformer benchmark](docs/benchmarks/qwen3-coder-30b-a3b-q8-cpu-2026-07-11.md)
- [Mixtral CPU cache-scaling study](docs/benchmarks/mixtral-8x7b-cpu-cache-scaling-2026-06-27.md)
- [All benchmark and qualification notes](docs/benchmarks/)

For the newest Predictor-v2 development evidence, use the linked GitHub qualification issues above; those results are intentionally not rewritten as \`main\` benchmark documents until that development lane is merged.

---

## License

MER is distributed under the **Business Source License 1.1** with the additional-use terms and Change Date defined in [\`LICENSE\`](LICENSE).

Non-commercial research, evaluation, and other uses are governed by the license text. Commercial use requires a separate commercial license.

For commercial licensing inquiries: **sales@amalgafy.com**

---

<p align="center">
  Built by <a href="https://amalgafy.com"><strong>Amalgafy</strong></a>.
</p>
