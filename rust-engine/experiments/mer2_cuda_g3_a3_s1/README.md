# MER2-G3-P6A-A3-S1: CPU process supervisor prototype

Authority: [#362](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/362), implementing baseline A from [A3 ADR #359](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/359#issuecomment-6407303851). Frozen parent [draft #361](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/pull/361): commit `79310a5d3aab9bae6f327fb268a40e4b4610fcaf`, tree `7841a792cb482297496c1a194bb108acae22c998`.

This isolated `[workspace]` has no dependencies. Its six new files are Cargo.toml, Cargo-generated Cargo.lock, src/lib.rs, src/bin/fake_worker.rs, tests/cpu.rs, and this README. It requires Unix (tested host recorded separately), not Windows. It neither links nor executes an inherited CUDA/model crate. No existing source, lock, numerical tolerance, CI, or production code is changed.

## Implementation and trust boundary

`Contract` has private mutable state and read-only receipt snapshots. It models one admitted bounded session; an error permanently closes admission. `Supervisor` owns monotonically allocated generations, the cross-generation request-ID tombstone ledger, actual `std::process::Child` handles, cancellation, a deadline, and kill/reap. `execute` synchronously drives the complete lifetime. There is no worker reuse, retry, raw-PID kill, raw driver release, or real KV transfer API.

Every IPC line binds `A3S1`, actual child PID, generation, source/model/KV hashes; request events also bind request ID and nonce. Unlike S0's hash-shape-only initial KV check, admission checks equality to the supervisor's expected KV digest. The private, atomically created mode-0700 Unix socket directory uses a bounded 2048-byte line protocol. It is removed after execution. All reads, writes, client-channel backpressure, acknowledgment waiting, and normal exit waiting check the same deadline and cancellation flag. There are no subprocess trees or shell commands in the library/worker.

The parent marks retain uncertainty **before** sending CONSTRUCT. The fake constructor can fail before retain, after the fake retain but before returning `FakeOwner`, or after owner return before its deferred check. OWNER, CHECKED and READY must occur in order. A forged RELEASED cannot make a worker ready. All resource messages remain fake scoped protocol evidence.

ADMIT/ACCEPT occurs before SUBMIT. A fault before SUBMIT leaves work NONE; SUBMIT latches SUBMISSION_UNCERTAIN before sending. ENQUEUED and FENCED are separate confirmations. The bounded fake output is the UTF-8 payload `abc`, SHA-256 `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`. A PREPARED digest is checked against the later actual payload bytes using a std-only SHA-256 routine. The parent marks delivery before exposing either of two chunks (`a`, `bc`) through the client channel. A separately supplied client acknowledgment must match the entire ticket and prepared digest. Timeout, partial streaming, cancellation or client disconnection before acknowledgment yields AMBIGUOUS and no automatic replay. Acknowledgment authorizes retirement, never a return to READY.

Drain and release each latch an attempt before their IPC command. Release requires successful checked drain. Failed/missing responses cause terminal quarantine; no second cleanup is attempted. The original error is latched and kill/wait/abnormal-exit errors are secondary. The report separately records actual wait status, successful reap, kill attempted, process-exit observation and post-exit baseline UNKNOWN. Synthetic baseline validation is exposed only on the CPU contract and labeled `SyntheticBaselineObserved`; process execution never invokes it.

## Prospective frozen gate

The acceptance source, fixture, toolchain, manifest and test-name inventory must be hashed before running the final gate. Any failed frozen test is `STOP_TEST`; preserve its first log and do not repair/retry or weaken a threshold under this attempt. Missing local Rust is `STOP_TEST_ENV`. Publication requires exact parent/branch identity, unchanged inherited files, the six-file addition allowlist, all Rust tests and the separate unchanged S0 reference run to pass.

Frozen time bounds: session **2000 ms**, kill/reap budget **2000 ms**, poll **5 ms**, frame **2048 bytes**. Integration assertions require total elapsed **<5000 ms** and timeout containment **<2100 ms** (100 ms scheduling allowance over the reap budget). The cancellation scenario requires **<1000 ms** after a cancellation scheduled at 100 ms. These are observed local acceptance bounds, not real-time kernel guarantees. The child hang has a 10-second fallback; the blocked-child scenario performs a real blocking read on a private socket pair until killed. Every process report must show an owned PID distinct from the test process, a wait status and completed reap.

From repository root, using external Cargo target/home directories if desired:

```sh
cargo test --manifest-path rust-engine/experiments/mer2_cuda_g3_a3_s1/Cargo.toml --locked --offline
```

Freeze `RUST_TEST_THREADS=1` and `RUST_TEST_NOCAPTURE=1` to preserve ordered per-child `OS_EVIDENCE` lines in the test log. This suite contains **85 named integration tests**, no ignored tests; lib/bin unit and doc-test targets have zero tests. Of the 85, 29 are real-child scenarios; the replacement tests each execute a second worker. Record the exact observed counts and statuses rather than inferring PASS from this prospective inventory.

Run the inherited reference separately, unchanged:

```sh
cd rust-engine/experiments/mer2_cuda_g3_a3_s0
python3 -B -m unittest -v test_supervisor_contract.py
```

Its expected count is **47**. It is independent Python behavior-model evidence, not a substitute for Rust/process results.

## Explicit acceptance mapping

The `c01` through `c44` tests (including `c40a`, `c40b`, `c40c`) preserve the 47 S0 test identities, in order; Rust's integer type replaces the Python boolean-PID exclusion with zero-PID rejection in c42. State assertions count actual cleanup starts, not rejected API calls. Exited physical state is not undone by later invalid evidence. Additional coverage tightens KV equality and verifies real OS effects.

| #362 requirement | Named Rust tests (prefix identifies unique full name in log) |
| --- | --- |
| Bad source/model/KV; nonce | c09, c10, c11, c12, e07 |
| Stale generation; duplicate request across generations | c08, c14, c43, os26, os28, os29 |
| Concurrent admission; PID reuse with new generation | c13, c34, c35; os07, os08 validate wrong process identities |
| Before retain, unreturned retain, owner-before-check | c03, c04, e01, e03, os02, os03, os04 |
| Known-good owner; cannot READY while uncertain | c02, c05, c06, c07, e02, os01, os05 |
| Before vs after enqueue ambiguity | c16, c18, e04, os16, os17 |
| Unconfirmed/foreign/duplicate fence | c17, c19, c20, os18, os20 |
| Mutated prepared output; invalid commitment | c21, c22, e08, os19 |
| Ack boundary, unacknowledged/partial streaming, no replay | c23, c24, c25, c43, os25, os26, os27 |
| Drain absent/failed, release-before-drain, one cleanup | c27, c28, c29, c30, e05, os21, os22 |
| Destructor failure, quarantine never readmitted | c31, e06, os23, os05 |
| First error versus secondary faults | c26, c32, c41, os11; each negative process scenario asserts its original error |
| Normal child ready/ack/exit; absent final receipt | os01, os12 |
| Hung/blocked child timeout and owned kill/reap | os09, os10 |
| Malformed, partial, oversized IPC; wrong version/identity | e09, os06, os07, os08, os13, os14, os15 |
| Cancellation and cross-generation rejection | os24, os26, os28, os29 |
| Exit versus baseline observation; UNKNOWN distinct | c01, c33, c34, c35, c36, c37, c38, c39, c40, c40a, c40b, c40c; every OS test |
| Never CUDA destructor/global leak claim | c44; every OS test |
| Invalid initial immutable identity | c15, c42 |

## Limits and remaining unproven guarantees

- This is a bounded local CPU protocol prototype, not production MER or a GPU worker. Source/model/KV identity strings are synthetic fixtures, not loaded model data. Kernel scheduling, spawn latency and uninterruptible processes cannot be given hard real-time guarantees. A reap timeout returns `ExitUnconfirmed`, retains the owned handle in the supervisor's `unconfirmed` list, and never reports a successful reap or a healthy replacement. This exceptional path has model coverage; inducing an unkillable OS process is outside scope. A fatal supervisor exit can still leave children; crash containment and OS service integration remain unproven.
- PID recycling is deterministically tested with the same numeric PID and different generation in the contract; actual kernel PID reuse is not forced. Live tests reject wrong PID/generation frames and stale tickets across real replacement processes. Same-user malicious socket impersonation, OS-level sandboxing, multi-tenant authentication and cryptographic code attestation are not established. The local trusted fake binary is the process boundary under test.
- Tombstones and generations persist for the lifetime of one `Supervisor`, not across a supervisor crash/restart. Durable replay suppression, external idempotency, actual client delivery and cross-process real KV recovery need separate design. The client channel tests simulate the client boundary with real threads. There is no automated restart/replay mechanism.
- Checked fake resource release is not in-process last CUDA-context Drop verification. `last_cuda_context_drop=UNOBSERVABLE`, `leak_free=NOT_CLAIMED`, `automatic_retry=FORBIDDEN`. No physical post-exit telemetry is acquired; the observed process report retains baseline UNKNOWN even after normal exit.
- [A2 #358](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/358#issuecomment-6407208972) remains scoped operator-host normal synthetic process-exit evidence; its GPU token is consumed. Historical [P5D #347](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/347) STOP remains 8/60 vectors and 10/3840 scalars against the unchanged strict criterion. No CUDA/driver query, Qwen restore, AWS action, model execution, numerical-acceptance change or production merge is authorized here.
