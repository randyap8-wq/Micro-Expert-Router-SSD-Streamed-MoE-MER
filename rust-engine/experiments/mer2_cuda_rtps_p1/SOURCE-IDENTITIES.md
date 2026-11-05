# RTPS-P1 R1 source and execution identities

R1 authorization is the submitted handoff, SHA-256
`542c8010859acaa1aa1e63044d232b1fd420f2dc77a3bf44c0b06da0f7c1c3f8`.
It authorizes one bounded local offline CPU check/test attempt and source-only
native adapter work on the existing draft PR. No earlier grant was reused.

## Exact admitted source

| Item | Verified identity |
| --- | --- |
| Repository | randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER |
| Draft PR | [380](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/pull/380), open, draft, unmerged |
| Existing head branch | codex/mer2-rtps-p1-source-20261104-01a1a973 |
| Original head commit | 3a29f3cce5ea74ae0b1099c4081aa8d42618f7b6 |
| Original head tree | 5a057f6f2298315573472899a6991fd074469fa3 |
| Sole parent / exact PR base commit | b55fd52d5043e7865a340c04f02433b0fa5dc974 |
| Base branch | fix/mer2-g3-p6a-a3-s3-p1a-r1-source-blockers |
| Original source manifest SHA-256 | 0877eff309e7b425df4f5ef94f8e83b413fdbc7baedf216a1b7e6f7a6ddc7ecf |
| Original source patch SHA-256 | 38466c0f346391d5fa8ca9b81d8193c28e2a01f7ddc6675104974e971c83553a |
| Separate frozen W3 tree | 59e545e2d394c9eb159c3547cc826e41a5df9b97 |

The live Git commit and untruncated recursive tree contained 263 blobs.
All 15 local P1 files matched their live original-head Git blob IDs, lengths,
regular-file modes and source SHA-256 inventory before execution. The source
manifest excludes itself; its own hash above binds that fifteenth file.

The local MER root has an unborn unrelated `main` Git repository and no remote.
It is not represented as a checkout of the PR branch. The exact local source
package was bound directly to the live Git objects; only the same remote branch
may receive the authorized ordinary successor commit. Existing local evidence
and Finder `.DS_Store` files are excluded, untouched, and not source inputs.

Final source hashes are in `SOURCE-PAYLOAD.sha256`. The final commit/tree and
remote readback belong in the local R1 publication receipt, avoiding a circular
self-commit claim in these source bytes. No W3 or other experimental tree is
imported. W3's 258 original file hashes are rechecked without modification.

## Frozen vendor provenance and public APIs

The same exact-head Git tree already contains all 26 candle-kernels vendor
files and `rust-engine/cuda/glibc_cuda_compat.h`. All 27 local reference files
were hashed against those live Git blobs. They were read from the available
S0 archive tree only after proving byte equality to the current PR ancestor;
they were neither borrowed as replacement experimental code nor copied/edited.
The optional path dependency refers to the existing repository vendor path.

| Exact-head path | SHA-256 |
| --- | --- |
| rust-engine/vendor/candle-kernels/Cargo.toml | fc19f7428b8df705b699c386dac86f25f7b3ad02dfa624146b88d3b351c39805 |
| rust-engine/vendor/candle-kernels/build.rs | d2683e67d2c82d7d3b2e03e937c625155e2218702fe21cf38a164a82b8bdeedf |
| rust-engine/vendor/candle-kernels/src/lib.rs | 91992b48ff931befcaccc854f639a173de1d1f831c7f5b45185b6bd811a7d7ad |
| rust-engine/vendor/candle-kernels/src/quantized.cu | 510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972 |
| rust-engine/vendor/candle-kernels/src/unary.cu | bea116bebb9b84106661dd425ed56ceddabbff337762edcee05f6e9134fb7e5b |
| rust-engine/vendor/candle-kernels/src/binary.cu | dc9961a0013fe24972adf27a37a986df5d78081912a6de21684a0433ba8d1b17 |
| rust-engine/vendor/candle-kernels/src/binary_op_macros.cuh | a83f97301a9ee52790b2c39b80e4f24393cf60dee153dbc68c0f802ea3ad95c4 |
| rust-engine/cuda/glibc_cuda_compat.h | 7f84797c9c440978534dd54d7ee476b83e37e6d591063582f48e8f8b1b964fee |

candle-kernels is the unchanged path-vendored 0.10.2, with no registry checksum
claimed for its locally modified build script. Its public `Module::ptx()`
provides the exact build-embedded QUANTIZED, UNARY and BINARY inputs. Its public
constants are Module objects, not raw strings. Generated PTX, compiler flags,
binary identity and Linux link behavior remain UNBOUND.

Already-cached registry archives were independently hashed; every archive file
matched its extracted source (cudarc: 130 files; cudaforge: 15 files).

| Package | Exact archive SHA-256 |
| --- | --- |
| cudarc 0.19.7 | 1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc |
| cudaforge 0.1.5 | 6f7a0d45b139b5beeeb1c34188717e12241c44a0120afb498815ce7f5373c691 |

The adapter uses cudarc's public `driver::result` and `driver::sys` types, not
P6B Native/Lease/permit constructors and not `CudaContext::Drop`. Reviewed API
locations in cudarc 0.19.7 `src/driver/result.rs`: init 88, device::get 109,
primary_ctx retain/release 449/462, ctx set_current/get_current/synchronize
546/553/580, stream create/synchronize/destroy 667/721/733, malloc/free sync
833/938, memset_d8_sync 977, memcpy_htod_sync/dtoh_sync 1020/1058, module
load_data/get_function/unload 1157/1169/1203, launch_kernel 1309.
The source archive inventory binds these APIs even if line display shifts.

Manifest source pins are exact and default-off. cudaforge is additionally pinned
as an optional build dependency to constrain the vendor's original ^0.1.2
requirement. This is NOT evidence that Cargo has resolved the native closure.
The only generated lock has one CPU package and predates these manifest edits.
It remains local. No fabricated native lock, dependency install, package
download, native build script or dynamic runtime package resolution was used.
cudarc's future dynamic driver-library loading still needs a reviewed host
library binding; a source pin alone does not establish that runtime mapping.

## Observed Stage A CPU evidence

Rust/Cargo were the installed Homebrew 1.97.1, host aarch64-apple-darwin.
rustc commit: `8bab26f4f68e0e26f0bb7960be334d5b520ea452`.
Cargo commit: `c980f4866141969fab6254a680546a277789d6f0`.
No applicable Cargo/rust-toolchain configuration, package dependency, build.rs
or proc macro was present in the original P1 crate. Environment was constructed
explicitly with offline mode, one job, one test thread, separate local
CARGO_HOME/target/tmp, and an exact Apple Clang linker.

| Item | SHA-256 |
| --- | --- |
| installed Cargo executable | b7a341737d8777ef0e21a71f0af90723ccf692914255110c12bbcfed79d75a53 |
| installed rustc executable | d69d40bfd2e11825feb3538512b6ffcd63de91c35ec36bb876849f0f9f8fe6bd |
| Stage A generated CPU-only Cargo.lock | db06d5433feb175594c57c86e4cd95ca1960ee32a785f376f6d2e6a6dd86ffa3 |
| check.log | ed2630ba9853fd4ca19a4c3cb259771aaf70753ec242a7e414e838257a92827e |
| test.log | cdeabc9d851e6ffe1acc827d36310e8b91e4e6a2c4307d6190dc20a5abf95579 |
| STAGE-A-PLAN.json | b88ef39485cd3fa108707d89a87029478594965203ac8e8c203488941f76f9f1 |
| STAGE-A-RESULTS.json | 24e3735bfb757f126012c8589ae1d4b69403b8dd77132174238b7a5fa82dc293 |
| STAGE-A-ARTIFACTS.json | 04a026b013b188180fda6b5a2437ff06985b8d7d1438f39bede853a0660fdc24 |

Full local evidence is retained at
`/Users/r/Documents/ChatGPT/MER/outputs/MER2-RTPS-P1-R1/evidence/`.
It includes all 24 individual outcomes and compiled CPU artifact hashes.
The native source received syntax/manifest/byte review only after Stage A.
