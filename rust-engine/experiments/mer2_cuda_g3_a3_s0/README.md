# MER2-G3-P6A-A3-S0 — CPU-only supervisor contract reference

**Status:** stand-alone Python standard-library behavioral model, not a Rust CUDA backend or deployable process supervisor. It has **no** OS child creation, CUDA calls, GPU telemetry, Cargo, Qwen, model inference, network, or storage operations. Tests simulate external events. Do **not** treat a test PASS as physical CUDA cleanup or production readiness.

## Authority and frozen source

- Design: [A3 ADR, issue #359](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/359#issuecomment-6407303851).
- Frozen native G3 base: commit `6150f4bb047b18679ddd48e468b819d7f39ad2ac`, tree `704a4b11908d5b233f40fdce1180642a3d176210`, parent `1b72de2fdb72fd10c8125d766a56b8876eb081af`.
- This model is isolated to this new directory. Existing `mer2_cuda_g3_native`, `mer2_cuda_g3_s1`, source fixtures, `Cargo.lock`, and PR #354 are unchanged.
- Historical A2 one-use GPU token consumed; P5D strict numerical STOP preserved.

## What is actually modeled

A supervisor-owned, single-session process generation with identity-bound ticket admission, uncertain CUDA primary-context retain before typed owner return, deferred-check confirmation, one in-flight request, unacknowledged delivery ambiguity, one checked drain and release, terminal quarantine/termination, PID+generation exit attribution, and an independent post-exit memory observation. A supervisor `Ledger` tombstones every admitted request ID across new worker generations. There is intentionally no retry operation.

`last_cuda_context_drop` always remains `UNOBSERVABLE`; `leak_free` is always `NOT_CLAIMED`. Model states are *claims about the protocol*, not statements about real GPU resources. A malicious or unexpected transition latches the first error, moves to quarantine (except already exited states), and cannot return to `READY`.

## Run tests (offline, CPU only)

```bash
python3 -B -m unittest -v test_supervisor_contract.py
```

Must be executed from this directory with Python 3.10+; no dependencies. No GPU, AWS or model data required. A test failure is a STOP for this bounded source attempt; do not relax assertions to convert a failure to PASS.

## Acceptance and follow-up

A3-S0 passes only when all unit tests and static allowlist checks pass. This is **not** Rust ABI compatibility, process isolation, CUDA context release, real-model inference, GPU performance, or production correctness. Next software milestone (separate authority): implement a minimal Rust supervisor/process protocol with equivalent negative tests, source-bound generation/ticket/KV identity and a real non-CUDA fake worker for kill/reap timeout behavior. No hardware failure injection is implied.
