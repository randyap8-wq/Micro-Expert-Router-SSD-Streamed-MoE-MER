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
  (CPU) by default. When `[real_transformer].compute_offload = "gpu"`,
  `GpuBackend::try_new()` runs first and, on success, is installed ahead of
  the CPU backend in a single `OnceLock`. If no `wgpu` device can be
  acquired, `BackendBox::init` logs `GPU init failed — activating CPU
  fallback` and the CPU backend is used (Finding 5, fail-closed + observable).
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
| F1 | Strict real-model loading + explicit seeded-development fallback | Complete | `model.rs` (strict loader / seeded fallback) | model-loading unit tests |
| F2 | Required-vs-optional conversion inventory + BF16 dense decode | Complete | `gguf_loader.rs` (`emit_dense_manifest_tensor_req`, required `dense_specs`), `inference.rs` (`dequantize_bf16_to_f32`) | `required_dense_tensor_absence_is_fatal`, `required_dense_tensor_unsupported_quant_is_fatal` |
| F3 | Logical tensor shape/orientation validation | Complete | `model.rs` (`safetensor_orientation_ok`, `require_2d`/`maybe2d!`, manifest `dense_dims_match_orientation`) | `safetensor_orientation_accepts_exact_and_vectors`, `safetensor_orientation_rejects_transposed_nonsquare`, `manifest_orientation_rejects_transposed_nonsquare` |
| F4 | Real tokenizer requirement + vocabulary compatibility | Complete | `tokenizer.rs`, config vocab checks | tokenizer/vocab unit tests |
| F5 | Explicit GPU fail-closed + observable Auto fallback | Complete | `backend/mod.rs` (`BackendBox::init`), startup logging | backend fallback tests |
| F6 | Strict manifest integrity + structural validation | Complete | `dense_tensor.rs` (`DenseTensorManifest` integrity: unsupported `format_version`, unsafe/traversal paths, duplicate name/alias/file, strict missing-checksum, byte-length mismatch, degenerate dims), wired in `model.rs` (`dense manifest integrity problem`) | `manifest_integrity_accepts_well_formed`, `manifest_integrity_rejects_unsupported_version`, `manifest_integrity_rejects_duplicate_alias`, `manifest_integrity_rejects_path_traversal`, `manifest_integrity_requires_checksum_only_under_strict`, `manifest_integrity_rejects_bytelen_mismatch`, `manifest_integrity_rejects_degenerate_dims` |
| F7 | Post-reconciliation architecture-aware resolved validation | Complete | `main.rs` (`validate_resolved_bench_real_config`, called after `reconcile_bench_real_config`) | 6 F7 tests (accepts reconciled checkpoint, rejects top_k>num_experts, rejects GQA indivisibility, allows asymmetric v_head_dim, rejects undersized cache, rejects O_DIRECT-misaligned expert_size) |
| F8 | Exact gate-sidecar sizing + canonical fallback | Complete | `main.rs` (`load_gate_weights`), `model.rs` gate sizing | gate-sidecar sizing tests |
| F9 | Non-finite attention-softmax fallback observability | Complete | `transformer.rs` softmax guard + observability | softmax fallback tests |
| F10 | Actual generated-token metrics | Complete | metrics/generation counters | generated-token metric tests |
| F11 | Feature/runtime-dispatch documentation accuracy | Complete | `README.md` feature section + runtime-dispatch caveat, this document | n/a (docs) |
| F12 | Correct latent GPU asymmetric-V shape planning (guard retained) | Complete | `backend/mod.rs` (`GpuKvCache` k_dim/v_dim, `kv_offset_elems`, `AttentionPushConstants.v_head_dim`, `try_new` v_head_dim), `backend/wgpu_shaders/attention.wgsl` (asymmetric K/V strides) | `kv_offset_symmetric_matches_shared_stride`, `kv_offset_asymmetric_v_uses_independent_strides`, `kv_offset_bytes_are_elems_times_four` |

Findings F1, F4, F5, F8, F9, F10 predate this branch and were reviewed only
for integration regressions. F2, F3, F6, F7, F11, and F12 were implemented on
this branch.
