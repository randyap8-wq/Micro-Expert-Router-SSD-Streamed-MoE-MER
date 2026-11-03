# MER — Micro-Expert Router

**Storage-backed, constrained-memory Mixture-of-Experts inference in Rust.**

MER (Micro-Expert Router) explores how to run sparse MoE models when their expert weights exceed the memory that can be kept hot on an accelerator. Its core design treats **NVMe/SSD, host RAM, and GPU VRAM as explicitly managed storage and residency tiers**, rather than requiring the entire expert population to remain in VRAM.

The repository contains a working **WGPU/Vulkan GPU-native inference path**, CPU and hybrid implementations, and an active **native NVIDIA CUDA development program**. These paths have different levels of qualification; experimental CUDA work is **not** a released CUDA backend on `main`.

MER is pre-release systems software, not a production-ready general-purpose model server. Benchmarks, numerical tests, model support, and backend readiness must be interpreted within the exact source, model, hardware, and workload that was tested.

<p align="center">
  <a href="https://amalgafy.com">
    <img src="docs/assets/amalgafy-brand-lockup.png" alt="Amalgafy" width="194" />
  </a>
</p>

## Why MER exists

Sparse MoE models may activate only a few experts per token, but **all of their expert weights still occupy storage**. The amount of compute needed for an active route and the amount of VRAM needed to hold every possible route are different problems.

MER is designed to manage that mismatch:

```text
                  sparse MoE token execution
                             |
                             v
               router selects expert IDs
                             |
                             v
              exact physical VRAM residency?
                   /                 \
                 hit                 miss
                  |                    |
                  |           resolve exact expert
                  |            from RAM or NVMe
                  |                    |
                  |           bounded GPU transfer
                  |                    |
                  +---------+----------+
                            |
                            v
                  execute routed experts
                            |
                            v
                    continue token loop
```

The storage hierarchy applies primarily to **routed-expert weights**, not to streaming every transformer operation from disk. The GPU-owned transformer loop and the expert-residency system are distinct parts of the architecture.

The long-term product goal is to let customers **choose the model and supported weight precision that their workload requires**, while MER manages limited hot memory. **Q4_0 is the current development and numerical-qualification baseline, not an intended product-wide precision restriction.** The ability to retain a selected precision under memory constraints must be demonstrated independently for each format and checkpoint; it is not yet a general released capability.

## Engineering status — November 2, 2026

| Area | What is established | Important limit |
| --- | --- | --- |
| **WGPU/Vulkan mainline** | GPU-owned transformer/token execution, Q4_0 routed-expert compute, host-backed expert residency, CPU/hybrid paths, and model-serving infrastructure exist on `main`. | A code path or historical pass does not qualify every model, format, GPU, or production environment. |
| **Q4_0 WGPU evidence** | An NVIDIA L4 [strict Hybrid Q4_0 expert-parity run](docs/benchmarks/qwen3-coder-30b-a3b-pr6-q4-parity-2026-08-11.md) passed in its pinned August scope. | That specific run used CPU attention/router and GPU routed experts; it was **not** a full-model quality or throughput result. |
| **Native CUDA experiments** | Separate G2/G3 Rust/Cudarc/Candle development vehicles have exercised synthetic expert computation on an NVIDIA L4. | Native CUDA real-Qwen full-token correctness, complete resource lifecycle and product performance are **not qualified**. [Qualification disposition](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351). |
| **R02-W3 checkpoint-owner qualification** | The earlier W3-S1 CPU-Q1 run passed **107/107 synthetic/metadata Rust tests**. A separate W3-S2-S1 **258-file candidate Git tree** (`59e545e2d394c9eb159c3547cc826e41a5df9b97`, not a commit) contains 12 new private tests for a complete 435-role synthetic checkpoint-owner transaction. | The 12 new tests are **authored, not executed**; **119-test CPU-Q2** has not run. This candidate is separately archived, **not merged to `main`**. [Current gate](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351). |
| **CPU-Q2 execution supervisor** | An isolated Python R0 supervisor source draft was independently integrity-reviewed; 80 adversarial tests were authored. | **Launch HOLD**: a persistent monitor failure after child spawn can prevent verified termination. The 80 tests remain **unexecuted**, and no one-use CPU-Q2 execution authority has been issued. [R0 containment finding](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351#issuecomment-6595585393). |
| **CUDA numerics and cleanup** | The synthetic [P5D stage comparison](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/347) identified strict differences; [P5E](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/348) isolated upstream input differences in the failing Down results. External process-exit GPU-memory observation was also recorded. | **P5D STOP remains** (8/60 eligible stage vectors); in-process final CUDA context cleanup and fault paths remain unproven. [Resource-lifetime scope](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/358). |
| **CUDA worker isolation** | The A3-S3 corrected CPU-only source passed a [completed independent static rereview with limitations](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/375) (historical review issue now closed). | 40 tests were **authored, not executed**. The [next Linux CPU gate](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/376) is design-only; no test or CUDA execution is authorized by it. |
| **Precision-first CUDA architecture** | [Completed decision #377](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/377) records the **owner-adopted Cudarc/Candle-primary backend decision** and a MER-owned, format-aware architecture; its issue closure did not implement the backend. | NVIDIA CUDA Rust/cuTile is **optional parked research, not a release dependency**. Additional formats and runtime migration are **not completed or authorized implementations**. |

This is a **dated snapshot**, not a live gate or a substitute for the issues' complete comments. An accepted synthetic CPU test, successful CUDA kernel invocation, or GPU process exit must not be described as a qualified full-model CUDA release. Historical STOPs remain part of the evidence even when a later, separately authorized revision passes.

### The AWS L4 qualification environment

AWS `g6.8xlarge` / NVIDIA L4 bring-up and exact Qwen artifact restoration progressed past the earlier regional-quota problem. See [PRE-1](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/255), [RESTORE-1](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/256), and [AWS GPU-2](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/257).

Those records cover **separate environment, recovery, WGPU semantic, and native CUDA synthetic evidence**. They are not a blanket AWS deployment pass or an interchangeable performance comparison. The WGPU strict semantic-v2 failure is preserved as a consumed result, and later native CUDA synthetic tests do not resolve it retroactively.

## Architecture

MER separates **model decoding**, **routing**, **expert storage**, **residency policy**, **execution backends**, and **observability**.

### NVMe / RAM / VRAM residency

- An SSD/NVMe-backed expert set can be larger than the selected GPU hot-memory budget.
- Host cache entries and GPU-resident allocations are separate states. A valid **host admission is not proof of physical device residency**.
- Expert IDs, generations, completion events, eviction and slot reuse must remain consistent across transfers and compute.
- On the WGPU path, a physical miss can be serviced from a valid host source and the token loop resumed from the corresponding recovery boundary.
- Linux supports direct-I/O-oriented storage paths and optional `io_uring`. The supported path depends on OS, filesystem, alignment and configuration.
- CPU/Hybrid execution paths remain important for verification and for execution profiles that are explicitly supported.

### WGPU/Vulkan inference on `main`

The mainline GPU-native path contains embedding, dense/attention and KV operations, routing/top-k, routed-expert compute and combine, normalization, LM head, and token selection. The host manages source retrieval, device residency and recovery.

The **legacy `cuda` Cargo feature is not this native GPU-owned CUDA engine**: in `rust-engine/Cargo.toml` it enables Candle's separate per-expert CUDA compute path. Do not enable it expecting the newer G3 native CUDA experiment to be present on `main`. WGPU is compiled into the ordinary engine, and the legacy `gpu` feature is retained as a no-op compatibility flag.

### Native CUDA under development

The G2/G3 research path uses Rust with Cudarc and pinned/vendored Candle CUDA kernel components. It is being built and qualified through separate source, CPU, CUDA, numerical and resource-ownership gates, without silently replacing the WGPU baseline.

The [owner-adopted CUDA backend decision](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/377) keeps **Cudarc and Candle as MER's primary native CUDA implementation**. NVIDIA's newer CUDA Rust/cuTile work remains **optional, parked research**, not a prerequisite for GPU qualification, multi-precision support, or commercial release. If an isolated, version-pinned cuTile kernel experiment is later justified by a measured bottleneck, it must independently prove actual context identity, liveness, synchronization, numerical equivalence, performance benefit and checked cleanup. CUDA Graphs, GPU virtual-memory redesign, `cuda-oxide` kernels and a wholesale `cuda-core` / `cuda-async` swap are not accepted changes merely because upstream examples exist.

### Private CUDA alpha path (target: December 31, 2026)

The near-term goal is a **private, single-host NVIDIA CUDA alpha** for a pinned **Qwen3-Coder-30B-A3B-Instruct Q4_0** checkpoint by **December 31, 2026**. This is a target, **not** an execution permit or a claim that full-token native CUDA inference works today.

The critical path is to (1) independently qualify the 435-role loaded-owner transaction and its bounded CPU supervisor; (2) bind the original GGUF, tokenizer, 6,144 SSD experts and generation identity to independent R01/R02 provenance and CPU-REC custody; (3) resolve P5D strict numerical and native CUDA lifecycle STOPs; then (4) execute real prefill/autoregressive decode and measure cold/warm end-to-end tokens/s on pinned hardware. **The first native CUDA Q4_0 TPS measurement has not happened.**

[Internal gated alpha plan](https://app.notion.com/p/40d422bc7eca81c3b684f13228ed2d18) · [Internal W3 evidence index](https://app.notion.com/p/40d422bc7eca8148b57bddff61ea0749) (Notion access may be required) · [GitHub authorization and STOP records](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351).

## Model and precision policy

The current main development checkpoint is **Qwen3-Coder-30B-A3B-Instruct**, in the frozen GGUF-derived **Q4_0** configuration used for strict qualification:

| Checkpoint characteristic | Research baseline |
| --- | --- |
| Transformer layers | 48 |
| Experts per layer | 128 |
| Routed experts in total | 6,144 |
| Experts chosen per layer (top-k) | 8 |
| Hidden dimension | 2,048 |
| Expert FFN dimension | 768 |
| Hardware used for key GPU qualifications | NVIDIA L4 |
| Mainline GPU API | WGPU / Vulkan |
| Native CUDA work | Separate experimental/qualification branches |

**Format support is not a single yes/no property.** Loading a model, decoding one tensor format, executing a CPU expert, running a whole GPU token, and streaming a large checkpoint with reliable output quality are different levels of support.

| Format family | Current interpretation |
| --- | --- |
| **GGML/GGUF Q4_0** | Current pinned routed-expert correctness and native CUDA experiment baseline; not the intended precision ceiling. |
| **Q8_0** | Existing [CPU full-transformer validation evidence](docs/benchmarks/qwen3-coder-30b-a3b-q8-cpu-2026-07-11.md); this does **not** imply equivalent native CUDA/out-of-core qualification. |
| **FP16 / BF16** | Higher-precision product targets requiring separate format-preserving loading, kernel, residency, numerical and model tests. |
| **Q4_K / Q5_K / Q6_K and mixed GGUF tensors** | Planned compatibility targets; each tensor packing and scale layout requires its own evidence. |
| **FP8 / MXFP8 / FP4 variants** | Optional hardware-dependent tracks; support must be declared by the actual model, device and backend. |

The prospective backend design is **per-tensor format-aware**: physical bytes, quantization metadata, scales, alignment, temporary buffers and GPU workspace must be accounted for without hard-coding Q4_0 block geometry. A customer-selected format must not be **silently downgraded or requantized**. Lossless repacking and widened compute accumulation need to be distinguished from changing stored precision.

No universal accuracy or speed relationship among quantization formats is claimed. Quality must be assessed for the selected checkpoint and workload.

### Other model families

The repository includes architecture-aware model loading and experimental execution paths beyond the primary Qwen target, including Mixtral-style MoE, DeepSeek variants, GPT-OSS, MiMo, and some dense model families. **A parser or implemented code branch is not proof of whole-model, hardware, or customer-facing support.**

Historical baseline: [Mixtral 8x7B CPU expert cache scaling](docs/benchmarks/mixtral-8x7b-cpu-cache-scaling-2026-06-27.md).

## Predictor-v2 and measured performance

Predictor-v2 investigates **routing-aware expert movement before future demand**, using committed route evidence and exact expert/generation identities. Predictions cannot replace ordinary demand and should receive credit only when matching real demand consumes the proposed residency.

The [P1Q3 NVIDIA L4 matched-resource experiment](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/204) validated the movement mechanism but reported an **INCONCLUSIVE** request-time improvement at the frozen 2% detection floor. Later [P1R](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/208) and [P1V6AC](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/252) work continued the causal investigation. **P1R is no longer the current next engineering step**, and none of this constitutes a general production speedup claim.

Throughput is workload-, model-, precision-, cache-state-, RAM/VRAM-budget- and hardware-dependent. Do not compare warm all-resident throughput with cold SSD-backed out-of-core throughput as if they were the same benchmark. Historical benchmark reports live under [`docs/benchmarks/`](docs/benchmarks/).

### Related out-of-core MoE work

Storage-backed expert inference is **not exclusive to MER**. In llama.cpp, [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) merged a GPU cache for experts held in host RAM; [PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469) was **open and experimental as of November 2, 2026**, exploring direct NVMe I/O, VRAM expert slots and reduced synchronization. Its reported speedups are not matched, independently reproduced MER benchmarks. MER's case must rest on measured model correctness, explicitly controlled memory residency, provenance, resource ownership and comparable end-to-end performance—not an exclusive claim to SSD streaming.

## Build and local smoke test

The Rust crate lives in [`rust-engine/`](rust-engine/). An ordinary build with the tokenizer feature is:

```bash
cd rust-engine
cargo build --release --features tokenizer
```

An example software test command is:

```bash
cargo test --locked --features tokenizer
```

These are **documented commands, not tests executed as part of this README update**. Cargo features and platform/toolchain requirements are defined in [`rust-engine/Cargo.toml`](rust-engine/Cargo.toml). Additional features such as `io_uring` and `avx512` are opt-in, and some features require specific Linux/CPU/CUDA environments.

From the repository root, the synthetic-data bring-up path is:

```bash
./scripts/quickstart.sh
```

It generates/reuses synthetic expert data and starts a local service; it is **not** a real Qwen benchmark, native CUDA test, or production installation qualification. Use [`config.toml`](config.toml) and [deployment/serving guidance](docs/production.md) for configuration and HTTP API concepts. Deployment documentation is guidance, **not** a claim of a supported commercial release.

## Validation and evidence

MER's engineering process deliberately distinguishes:

1. **Source/static validation** — API and ownership design, reviewed source identity and authoring.
2. **CPU/software validation** — compiled tests and bounded mock-process evidence.
3. **Numerical correctness** — same-input expert/stage parity and independently defined acceptance tolerances.
4. **Physical GPU and resource correctness** — executed device operations, leases, fences, memory lifetimes and failure containment.
5. **Real-model and product quality** — full-token semantics, selected format/checkpoint fidelity, output quality, out-of-core behavior, and practical reliability.
6. **Performance** — matched hardware/model/precision/workload, cold/warm behavior, actual SSD/host transfers, end-to-end latency, peak memory, throughput and cost.

Passing one level does **not** automatically pass the later levels. An original STOP remains an original STOP; separately reviewed subsequent attempts have their own authority and evidence.

### Where to follow current development

- [CUDA-first MER2 strategy / parent epic #266](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/266)
- [Native CUDA qualification and numerical disposition #351](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351)
- [Completed A3 static review #375](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/375), [still-open CPU-E1 design #376](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/376), and [unresolved CPU-REC descriptor custody #378](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/378)
- [Adopted, completed Cudarc/Candle backend decision #377](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/377)
- [Internal W3 qualification evidence index](https://app.notion.com/p/40d422bc7eca8148b57bddff61ea0749) and [2026 CUDA-alpha plan](https://app.notion.com/p/40d422bc7eca81c3b684f13228ed2d18) (Notion access may be required)
- [Post-CUDA evidence-preserving cleanup #370](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/370)
- [Open GitHub issues](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues) — inspect the relevant issue **and its latest comments** before treating a proposed gate as authorized.

**Adopted sequencing:** finish the relevant independently authorized CUDA/CPU qualification gates on the **existing Cudarc/Candle path**, while designing MER-owned format/precision, ownership and byte-accounting contracts independently of NVIDIA CUDA Rust; qualify native CUDA on the real model; freeze the accepted baseline; perform the surgical post-CUDA cleanup and matched post-clean validation; then expand precision formats and evaluate DeepSeek/FreeToken and other research optimizations based on measured bottlenecks. **cuTile interoperability is an optional, off-critical-path experiment, not a prerequisite at any phase.** The research shortlist is parked, not a collection of already merged features. Wider AMD ROCm/HIP support and additional GPU/platform profiles remain demand- and qualification-dependent.

The README is an overview, **not an execution authority or live status ledger**. Exact source commits, model hashes, one-use execution gates, test receipts and STOP/PASS decisions live in their scoped GitHub records.

## Repository map

| Path | Purpose |
| --- | --- |
| [`rust-engine/`](rust-engine/) | Main Rust inference crate, CPU/hybrid and WGPU paths |
| [`rust-engine/src/backend/`](rust-engine/src/backend/) | Execution and GPU backend code |
| [`config.toml`](config.toml) | Example/annotated configuration |
| [`docs/production.md`](docs/production.md) | Service operation and deployment considerations |
| [`docs/audit-findings.md`](docs/audit-findings.md) | Audit notes and limitations |
| [`docs/distributed.md`](docs/distributed.md) | Distributed design and transport documentation |
| [`docs/benchmarks/`](docs/benchmarks/) | Historical benchmark and qualification reports |
| [`scripts/`](scripts/) | Model conversion, extraction and support tools |
| [GitHub MER2 issues](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues) | Current engineering approvals, experiments, code reviews and evidence |

## License and commercial use

MER is public source distributed under the **Business Source License 1.1**, **not** an unrestricted open-source license. The exact permissions, Additional Use Grant and Change Date are in [`LICENSE`](LICENSE). Commercial-purpose use as defined by that license requires a separate commercial license.

Commercial licensing: **sales@amalgafy.com**.

<p align="center">
  Built by <a href="https://amalgafy.com"><strong>Amalgafy</strong></a>.
</p>
