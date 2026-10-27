# MER2 A3-S3-P1a: CPU mock binary worker bridge — SOURCE DRAFT ONLY

**UNBUILT, UNTESTED.** No Cargo, rustc, rustfmt, tests, CUDA, driver operations, VM start, model restoration, PR or merge are authorized by this draft. Do **not** use in production.

- Base: R4 exact commit `2236db82a5f22c0d9793084e89de18647d7ebdbd`, tree `89a5937c5e5e5e2bd72d32fd5696a061603c2e01`.
- New isolated source root: `rust-engine/experiments/mer2_cuda_g3_a3_s3/`; nine-file allowlist in issue #374.
- Control/transport: A3N1 v1.0 **proposed** 80-byte envelope; strict fixed fields and SHA256 of raw payloads. Local Unix domain only; hash checks are **not** peer authentication.
- CPU data plane: exactly one 64-element, little-endian finite-f32 vector (256 bytes) per direct child, mock result identity transformation (not a CUDA or model result). Control frames <=4096 bytes and data frames <=1 MiB, further shape-constrained to 256 bytes.
- Intended contract: R4's exact `Binding`, `Ticket`, `Contract` ownership/work/delivery states; a separately encoded S1-shaped ticket identity, **not** a type cast to S1's sealed `FfnProvider` or unpublished byte encoder.
- Ownership: parent controls one child, one request, one generation; one attempt to terminate/reap after failure; unknown state is quarantined. Direct child exit is not descendant containment; a parent restart has **no persistent replay guarantee**.
- This version does **not** link `mer2-cuda-g3-native`, `cudarc`, model weights, GPU contexts, device memory, or Qwen; no `ConfirmedFfn` is forged.
- Last CUDA context destruction `UNOBSERVABLE`, leak freedom `NOT_CLAIMED`, automatic replay `FORBIDDEN`; strict historic P5D stage numerical `STOP` preserved.
- `Cargo.lock` is a **manually prepared source draft** from the R4 local-only package graph; **it has not been generated or verified by Cargo**. A separately approved locked/offline build gate must prove it or STOP_DEPENDENCY.

## Important design limitations for independent review

1. A3N1 is a design candidate, not an accepted durable wire standard. There is no SO_PEERCRED check yet; the private mode-0700 socket directory and parent-owned child provide only a limited local threat boundary. Future hardening must independently verify OS peer credentials and symlink/exec identity.
2. The original R4 `Ledger` is in-memory only. Do not claim exactly once across process restarts or replay safety after a parent crash. There is **no automatic retry** of ambiguous work.
3. The mock has no GPU error provenance; `OWNER/CHECKED` describe mock lifecycle only. `RELEASED` is scoped and does not attest to a CUDA destructor.
4. New test sources are **drafts only**. No test count or command has been authorized and no tests were executed.

Design: [GitHub #373](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/373); source authority: [GitHub #374](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/374).
