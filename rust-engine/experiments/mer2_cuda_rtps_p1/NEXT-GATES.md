# RTPS-P1 R1 next gates — proposals only

The original P1 Mac CPU gate is complete: one offline check and one offline test,
24/24 passed before R1 edits. Native implementation is source-only. No automatic
successor, VM operation, native build/run, merge or production authority follows.

1. **Independent source and CPU review.** Review the R1 diff, exact published
   blobs and source manifest, Stage A logs and unchanged fixture/oracle bytes.
   Review the raw CUDA ABI, primary-context assumption, symbol/PTX linkage,
   five actual readbacks, explicit cleanup and quarantine. The staged API's
   uninhabited gate and public `native-once` STOP must remain closed.
2. **Best next one-use Linux build gate, prepared by ChatGPT.** Bind the accepted
   source commit and manifest; founder identifies the intended existing host
   without Codex AWS/SSH operations. Establish exact Linux/toolchain/toolkit
   executables, complete vendor/header inventory, registry archive checksums,
   native lock and transitive build-script closure. Do not reuse the Stage A
   CPU-only lock as native evidence. Resolve a native lock offline only under
   explicit fresh authority; stop on missing cache or unreviewed resolution.
   Review cudarc's build.rs and unchanged candle-kernels/cudaforge build path.
   Pin `CUDA_COMPUTE_CAP` from already-reviewed host evidence to suppress
   cudaforge's `nvidia-smi` fallback; bind `NVCC`, `PATH`, compiler, compatibility
   header and CUDA-version overrides. No automatic driver probe or downloading.
   The vendor builds additional PTX and an MoE static library and links cudart;
   this wider build closure must be admitted even though only three modules
   are loaded by the diagnostic. Set wall/process/memory/disk/output ceilings,
   a separate target directory and external monitoring before execution.
   Proposed single compile/link command, only after that review and grant:

   ```text
   cargo test --offline --locked --manifest-path rust-engine/experiments/mer2_cuda_rtps_p1/Cargo.toml --lib --no-default-features --features native-diagnostic --no-run
   ```

   This compiles test declarations without running them. Seal the resulting
   binary, all three embedded PTX images, generated bindings and complete build
   logs. Verify exact f32 symbol definitions, parameter widths, actual SiLU
   expression and compiler flags. A successful build is not numerical evidence.
3. **Separate host-gate implementation and review.** Define a real externally
   reviewed process/host permit before changing `ReviewedHostGate`. No hashes or
   booleans from a caller suffice. Bind the physical device and library mapping,
   fresh isolated single-threaded process, absence of other primary-context
   users, one initialization/invocation, deadline, output custody, terminal
   receipt and independent reaper. CUDA/FFI may block or abort before a terminal;
   in-process cleanup cannot close this gap. Do not open the public CLI as an
   incidental build fix. No permit implementation is supplied in R1.
4. **Separately authorized one-use synthetic GPU parity gate.** Only after the
   previous reviews: founder executes ChatGPT-prepared SHA-pinned GitHub scripts
   under an explicit one-use grant. Submit the fixed raw Q4 bytes once; retain
   all five actual device vectors; report 5,120 actual/expected bits and deltas,
   symbol resolution, launch/fence/copy results and cleanup or missing terminal.
   Stop on first failure, no retry/fallback. Numerical equality cannot detect
   identity substituted for this saturated SiLU; actual `usilu_f32` PTX and
   launch evidence are mandatory. This is only synthetic geometry/stage parity.
5. **Authentic owner/model/token/TPS work stays separate.** W3 R01 independent
   run authority, R02 full checkpoint dense/tokenizer/SSD expert provenance,
   P5D strict stage STOP, O1/H14, CPU-REC, and CPU_Q2=HOLD_NO_AUTHORITY (119
   separate Rust tests) are unchanged. No authentic Qwen, token or TPS claim.

Codex must not SSH, start/stop EC2, or write a VM launcher. The founder controls
AWS manually. ChatGPT prepares/publishes reviewed SHA-pinned VM-operation
scripts only when separately authorized. This file is not such a script/grant.
