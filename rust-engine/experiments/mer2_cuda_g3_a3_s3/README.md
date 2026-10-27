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

## R1 corrective source draft — not compiled or executed

The independent S3-P1a source review at commit
`9c48b707d026fdacbc51d2149865a8864dc9e64e` returned
`STOP_SOURCE_BLOCKERS` ([GitHub #374](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/374)).
This R1 source-only corrective draft addresses F001–F006 and the local
nonblocking Codec precondition from F007, and documents an explicit bounded
CPU profile for F008. **These corrections are not independently reviewed, built,
or tested.** F009/F010 remain documented trust/ownership limits, not PASSes.

### Explicit accepted-candidate *CPU mock* profile — not the full A3N1 RFC

- The transport uses the proposed A3N1 **80-byte LE** frame envelope, with
  control payloads <=4096 bytes and binary data payloads <=1 MiB. In this
  profile DATA_INPUT/DATA_OUTPUT **must** hold precisely 256 bytes representing
  64 contiguous finite `f32` values. The only allowed calculation is identity.
- Local **trusted-parent** and one directly supervised CPU child, one request,
  one generation, one bounded session. No peer credentials (`SO_PEERCRED`),
  immutable executable-image pinning, adversarial same-UID authentication,
  descendant/process-group containment, or durable crash-restart tombstone.
- INIT carries three binding digests (R4 source/model/KV) and two synthetic
  binary/schema commitments. The fixture is implicitly D64 identity; parent is
  producer of DATA_INPUT and consumer of DATA_OUTPUT. The 104-byte output-only
  descriptor has implicit producer/consumer roles, one fixed layout and work
  hash. These assumptions are profile limitations, not full RFC feature support.
- The S1-shaped public ticket is a **separately encoded mock mirror**; the
  source digest is deliberately constrained equal to the R4 binding digest in
  this CPU fixture. This is **not** a general equivalence of S1 and R4 hash
  domains. The child validates an opaque ADMIT hash against the trusted
  parent's supplied commitment; it does **not** decode every ticket field or
  independently authenticate model/weight provenance. Full validation is
  mandatory before a real CUDA worker or any shared service.
- SHA256 authenticates neither the parent nor the worker. The parent hashes
  an executable path before `Command::spawn`, which does not prove the actual
  executed file was the same object (TOCTOU). Limit this experiment to a
  private, controlled host with immutable staging handled out of band.
- The parent owns an **absolute monotonic session budget starting after**
  executable preflight, socket creation and listener setup. Thus elapsed_ms
  does **not** bound total function latency. Cancel is checked **before** the
  executable read and immediately before and after successful spawn. Any
  cancellation/error after spawn takes one bounded direct-child containment
  path, with no automatic second kill. Real-model startup budgets are not
  specified by this CPU fixture.
- The public Codec's I/O methods idempotently enforce nonblocking stream
  mode. Valid BYE is **not terminal acceptance** until bounded clean EOF, with
  zero extra bytes, has also been observed. Any extra byte, duplicate frame,
  delayed trailing bytes, deadline or cancellation is a STOP; physical child
  exit/reap is reported separately.
- A returned `Ok(MockReport)` means **report produced**, not successful work.
  Only `MockReport::is_success()` evaluates the full bounded CPU mock outcome:
  error-free confirmed owner/fenced work/ACK/checked drain and release, exactly
  one attempt of each, independently verified identity output, clean terminal
  EOF and observed zero-exit direct child, with no kill or containment errors.
  The client must independently verify all output bytes, digest, ticket and
  final chunk marker. This predicate is **not** GPU numerical or leak proof.
- The parent preserves the original first error and an ordered list of later
  containment errors; failure does not cause retransmission. Scoped RELEASED
  is **not** evidence of the last CUDA context Drop. Unconfirmed direct child
  handles can be transferred by `take_unconfirmed_children()` to a named
  external owner; this mock does not establish panic/unwind process-tree
  cleanup, and test hosts require an outer watchdog before running deliberate
  unconfirmed-reap scenarios.

### Test-source coverage and explicit remaining gaps

At this source draft there are **40 authored test functions** (22 wire/tensor
and 18 process). **0 have run**, and the hand-prepared Cargo.lock remains
unverified. P01–P14 now assert exact reached stages and/or first errors and
client output observations; new `w08`, `b07`, `b08`, `t06`, `p15`, `p16`,
`p17`, `p21` cover selected previously missing proof obligations. The
independent reviewer proposed additional `p18`, `p19`, `p20`, `p22` tests for
**deterministic late ACK, late exit, multiple process-operation failures and
unconfirmed-reap ownership**. Those need reviewed injectable clock/Child
seams and **are NOT authored here**. The reviewer must not treat a count of 40
as complete coverage or a frozen execution authority. Test function count is
not an executed child-process count; `p15`/`p21` each contain multiple cases.

The future workflow is: independent static rereview of exact corrected source
and test inventory; resolve residual blockers; separately approve one new
locked/offline Linux CPU build-and-test gate with a pinned toolchain, explicit
outer process owner, and zero retries. Historical S2 one-use authority remains
consumed; P5D strict numerical STOP remains unchanged. No CUDA, model, AWS,
PR or merge authority is implied by the R1 source draft.
