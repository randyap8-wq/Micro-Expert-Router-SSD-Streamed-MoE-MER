# Model-loading & runtime-dispatch audit — findings F1–F12

This document records the disposition of the twelve QA-audit findings that
hardened real-model loading, tokenizer/vocabulary handling, GPU fail-closed
behaviour, manifest/tensor validation, and the latent GPU asymmetric-V path.
All twelve are addressed on the same branch and land in a single PR.

The document has three parts:

1. [Supported feature-build matrix](#supported-feature-build-matrix) — which
   cargo features the engine ships and how each is exercised.
2. [Runtime-dispatch summary](#runtime-dispatch-summary) — how the backend and
   attention path are chosen at runtime (not build time).
3. [F1–F12 disposition table](#f1f12-disposition-table) — per-finding status,
   the code that satisfies it, and its regression tests.

---

## Supported feature-build matrix

The default build (`cargo build --release`) is CPU-only plus the `tui`
dashboard. Every math kernel and the GPU compute plane compile
unconditionally; hardware selection is a **runtime** decision (see below).
Cargo features gate only optional dependencies and opt-in kernels.

| Cargo feature | Default | Purpose | Notes |
| --- | --- | --- | --- |
| `tui` | **on** | Native `ratatui`/`crossterm` `monitor` dashboard | Pure-Rust deps; disable with `--no-default-features`. |
| `tokenizer` | off | Real HuggingFace `tokenizers` (`tokenizer.json`) | Pulls `onig` (C build); byte-level fallback tokenizer otherwise. |
| `io_uring` | off | Linux `io_uring` fixed-buffer SSD backend | Linux-only; `pread` + `block_in_place` path is the portable default. |
| `blas` | off | `matrixmultiply` SGEMV dense selector | `matrixmultiply` compiled unconditionally; feature retained for the `auto` selector. |
| `avx512` | off | `#[target_feature]` AVX-512F/BW/VNNI kernels | Runtime-probed; single binary still runs on non-AVX-512 hosts. |
| `amx` | off | AMX skeleton + tile-hint plumbing | Executor falls through to AVX-512/scalar until tile intrinsics stabilise. |
| `nightly-amx` | off | Real Intel AMX tile intrinsics | Implies `amx`; **requires nightly**; falls back to AVX-512 when the probe fails. |
| `gpu` | off | **No-op**, retained for back-compat | `wgpu` + `GpuBackend` compile unconditionally; select at runtime via `compute_offload = "gpu"`. |
| `cuda` | off | candle-core CUDA per-expert SwiGLU | Requires a CUDA toolkit; runtime falls back to CPU when no device. |
| `grpc` | off | Real `tonic`/`prost` gRPC expert sharding | Committed stubs (`grpc_gen.rs`) — no build-time `protoc`. |
| `simd` | off | **No-op**, retained for back-compat | Row-parallel matmul is always compiled. |
| `alloc-count` | off | Process-wide counting allocator for microbenchmarks | Off in production to avoid atomic counter traffic. |

Recommended verification matrix:

```sh
# Default (CPU + tui)
cargo build --release
cargo test  --release

# Optional dependency features (compile check)
cargo build --release --features tokenizer
cargo build --release --features io_uring   # Linux only
cargo build --release --features grpc
cargo build --release --no-default-features # minimal embedded build
```

`avx512`, `amx`, `cuda`, and `nightly-amx` require the corresponding
toolchain/hardware; enable them only on hosts that provide it.

---

## Runtime-dispatch summary

* **Math backend.** `backend::install_default()` installs `CandleBackend`
  (CPU) by default. When `[real_transformer].compute_offload = "gpu"` or
  `"auto"`, `GpuBackend::try_new()` runs first and, on success, is installed
  ahead of the CPU backend in a single `OnceLock`. A GPU initialization *or*
  backend-installation failure is handled per the `compute_offload` mode:

  | `compute_offload` | Semantics |
  | --- | --- |
  | `gpu` | **Fail closed** — GPU init *or* `set_backend`/installation failure is fatal; the run aborts and never silently continues on CPU. |
  | `auto` | **Observable GPU→CPU fallback** — on failure fall back to CPU and record it in logs, metrics, and runtime metadata. |
  | `cpu` | **CPU only** — no GPU initialization is attempted. |

  Explicit GPU does **not** transparently fall back to CPU (Finding 5 / gist
  finding 3); the run-command GPU installer (`install_run_gpu_backend`),
  serving installer, and any other explicit GPU path all enforce this.
* **CPU kernels.** `kernels::detect()` probes the host once at startup and
  auto-escalates AVX-512(+VNNI) → AVX2+FMA → scalar. The selected backend is
  logged on one startup line. No cargo feature is needed for AVX2.
* **Attention path.** Dense attention runs through the wgpu shader when the
  GPU backend is active **and** the architecture is GPU-eligible.
  Asymmetric-V architectures (`v_head_dim != head_dim`) retain an explicit
  eligibility guard in `MultiHeadSelfAttention` that forces the CPU attention
  path (Finding 12). The latent GPU KV-cache/attention shader carries the
  corrected asymmetric value width so the guard can be lifted once end-to-end
  asymmetric GPU execution is proven.

---

## F1–F12 disposition table

| # | Finding | Status | Where | Regression tests |
| --- | --- | --- | --- | --- |
| F1 | Strict real-model loading + explicit seeded-development fallback + required-vs-optional load accounting | Complete | `model.rs` (strict loader / seeded fallback, `WeightLoadStatus.optional_probed`/`optional_loaded`, gated shared-expert probe) | model-loading unit tests, `complete_strict_mixtral_converted_dir_reports_no_seeded_fallback`, `complete_strict_qwen3_moe_converted_dir_reports_no_seeded_fallback`, `declared_shared_expert_missing_required_weights_fails_strict`, `absent_optional_shared_sidecar_does_not_affect_required_counts` |
| F2 | Required-vs-optional conversion inventory + BF16 dense decode + architecture-aware allowlist / tied output | Complete | `gguf_loader.rs` (`emit_dense_manifest_tensor_req`, `resolve_conversion_profile` allowlist, `emit_required_dense_tensor_checked`, tied-output-by-contract, `ExtractOptions::arch_override`), `inference.rs` (`dequantize_bf16_to_f32`) | `required_dense_tensor_absence_is_fatal`, `required_dense_tensor_unsupported_quant_is_fatal`, `convert_unknown_architecture_fails_closed`, `convert_unknown_architecture_rescued_by_override`, `convert_fused_qkv_architecture_is_unsupported`, `convert_qwen3_moe_ties_output_by_contract`, `convert_untied_architecture_missing_output_fails`, `convert_untied_architecture_ties_by_metadata_flag` |
| F3 | Logical tensor shape/orientation validation (strict rank-2 matrices) | Complete | `model.rs` (`safetensor_orientation_ok` — exact rank-2 only, `require_2d`/`maybe2d!`, manifest `dense_dims_match_orientation`) | `safetensor_orientation_accepts_exact_rank2_only`, `safetensor_orientation_rejects_transposed_and_higher_rank`, `manifest_orientation_rejects_transposed_nonsquare` |
| F4 | Real tokenizer requirement + vocabulary compatibility | Complete | `tokenizer.rs`, config vocab checks | tokenizer/vocab unit tests |
| F5 | Explicit GPU fail-closed (init **and** install stages) + observable Auto fallback | Complete | `backend/mod.rs` (`BackendBox::init`), `main.rs` (`install_run_gpu_backend` returns `Result`, serving installer), startup logging | backend fallback tests, `resolve_backend_selection` init-failure tests |
| F6 | Strict manifest integrity + structural validation | Complete | `dense_tensor.rs` (`DenseTensorManifest` integrity: unsupported `format_version`, unsafe/traversal paths, duplicate name/alias/file, strict missing-checksum, byte-length mismatch, degenerate dims), wired in `model.rs` (`dense manifest integrity problem`) | `manifest_integrity_accepts_well_formed`, `manifest_integrity_rejects_unsupported_version`, `manifest_integrity_rejects_duplicate_alias`, `manifest_integrity_rejects_path_traversal`, `manifest_integrity_requires_checksum_only_under_strict`, `manifest_integrity_rejects_bytelen_mismatch`, `manifest_integrity_rejects_degenerate_dims` |
| F7 | Post-reconciliation architecture-aware resolved validation, shared by serving + bench-real; routed-cache working set excludes resident shared experts | Complete | `main.rs` (`reconcile_real_model_config` + `validate_resolved_real_model_config`, called by both `build_bench_real_runtime` and `cmd_serve`; cache-slot check uses `top_k` only) | 6 F7 tests (accepts reconciled checkpoint, rejects top_k>num_experts, rejects GQA indivisibility, allows asymmetric v_head_dim, rejects undersized cache, rejects O_DIRECT-misaligned expert_size), `serve_and_bench_real_share_reconcile_and_validation`, `explicit_architecture_still_reconciles_config_json`, `explicit_architecture_mismatch_is_rejected` |
| F8 | Exact gate-sidecar sizing + canonical fallback | Complete | `main.rs` (`load_gate_weights`), `model.rs` gate sizing | gate-sidecar sizing tests |
| F9 | Non-finite attention-softmax fallback observability | Complete | `transformer.rs` softmax guard + observability | softmax fallback tests |
| F10 | Actual generated-token metrics | Complete | metrics/generation counters | generated-token metric tests |
| F11 | Feature/runtime-dispatch documentation accuracy | Complete | `README.md` feature section + runtime-dispatch caveat, this document | n/a (docs) |
| F12 | Correct latent GPU asymmetric-V shape planning (guard retained) | Complete | `backend/mod.rs` (`GpuKvCache` k_dim/v_dim, `kv_offset_elems`, `AttentionPushConstants.v_head_dim`, `try_new` v_head_dim), `backend/wgpu_shaders/attention.wgsl` (asymmetric K/V strides) | `kv_offset_symmetric_matches_shared_stride`, `kv_offset_asymmetric_v_uses_independent_strides`, `kv_offset_bytes_are_elems_times_four` |

Findings F1, F4, F5, F8, F9, F10 predate this branch and were reviewed only
for integration regressions. F2, F3, F6, F7, F11, and F12 were implemented on
this branch.

A follow-up hardening pass on this branch further tightened several findings so
production paths fail closed: F1 (explicit required-vs-optional load accounting
so an architecture with no shared experts no longer trips
`seeded_fallback_remained`), F2 (architecture-aware conversion allowlist,
explicit `Unsupported` for fused-QKV/MLA families, and Qwen3-MoE tied-output by
GGUF contract), F3 (strict exact rank-2 matrix validation — flattened rank-1 and
rank>2 rejected), F5 (explicit GPU fail-closed at both init **and** install
stages via `install_run_gpu_backend` returning `Result`), and F7 (serving and
bench-real now share `reconcile_real_model_config` + `validate_resolved_real_model_config`,
and the routed-cache working set no longer counts resident shared experts).
