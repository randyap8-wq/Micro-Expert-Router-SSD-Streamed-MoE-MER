# Source identities and reviewed inputs

User authorization: attached `Pasted text.txt`, SHA-256
`d3e7adf808da9b37104395c60f3b91f6c1ad1211c56bfb604bc1a012cfa7c337`.
The #351 comment references a separately named handoff with a different hash;
these are not represented as byte-identical documents.

Read live: [issue #351](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351),
[P0 scope](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351#issuecomment-6622523867),
[P0 independent review / P1 decision](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351#issuecomment-6637541321),
and [founder-operated workflow](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/351#issuecomment-6647522767).
All five P0 reports were read locally; their five hashes and eight evidence
manifest entries matched. No earlier memory was used as current source proof.

| Identity | Fresh P1 verification |
| --- | --- |
| Baseline commit `b55fd52d5043e7865a340c04f02433b0fa5dc974` | Live GitHub Git-commit API binds tree `e0b0b698bf49d4d6dbe30bb31922aa02b4c4f44d`; all 248 baseline blob bytes independently reconstructed in memory from unchanged S0 bytes plus four archived originals and rehashed; live recursive tree untruncated. |
| S0 tree `31c1e50984ca270498ff1e98470e8faa7c0e0a32` | 254 actual files, modes, SHA-256 and Git blob IDs rechecked, Git tree recomputed; original directory unchanged. |
| W3 tree `59e545e2d394c9eb159c3547cc826e41a5df9b97` | 258 actual files/modes rechecked and tree recomputed; every local source byte equals the archived W3 source member. |
| D1-R1 evidence ZIP | SHA-256 `e3c20c2a4d56cada839904dc1b97ac6e99f4c768f26197b36f786ff3986fe4df`, CRC readback; archived baseline/S0 inventories and original after-reversal bytes read. |
| W3 source evidence ZIP | SHA-256 `cb8b7594d8e073fe0a4508b74792efee499480336febed4803fbf66c9ee562f1`, CRC readback; 258 source members match local bytes. |
| Current main, read only | Live ref `23ac889408dc479afe7bf8fefbe97c1cad880d2f`; not used as experiment parent or changed. |

Exact roots: S0 is `/Users/r/Documents/Codex/MER2-G3-P6B-S0-source-candidate/`;
W3 is `/Users/r/Documents/ChatGPT/MER/development-candidates/MER2-G3-P6B-R02-W3-S2-S1-source-only/source/`.
The P0 reports/receipt are under
`/Users/r/Documents/ChatGPT/MER/outputs/MER2-P6B-RTPS-P0-readonly/`.
Those trees reconcile as 249 identical paths, five conflicts and four W3-only
paths. They are tree identities, not interchangeable commits. The baseline
contains no P6B directory; P1 source has no dependency on a silently imported
S0/W3 snapshot. No Git history was repaired or rebased.

## Inspected W3 reference blobs — SHA-256

All paths below are relative to the frozen W3 source root.

| Path | SHA-256 |
| --- | --- |
| `rust-engine/experiments/mer2_cuda_p6b/src/abi.rs` | `733da5fb26c6383075a108e426e6aad1d8f47f42b8a36138322039868b5113c1` |
| `rust-engine/experiments/mer2_cuda_p6b/src/cuda.rs` | `58f2ff6bd67c9854215179743a93baccd0fabf0c3f24bf2bf7d0749f6e8122ae` |
| `rust-engine/experiments/mer2_cuda_p6b/src/identity.rs` | `f835972949a49ba9e1ce45333c5127419c853eaf6b5438e7ca860178caa59acd` |
| `rust-engine/experiments/mer2_cuda_p6b/src/mod.rs` | `b877b6a4bfd48fc143ba6d938d6098697d1549f57e536e9d2d234be1db4a6541` |
| `rust-engine/experiments/mer2_cuda_g3_native/src/cuda.rs` | `459e6b7baf461acac2e1ea84d8d8f7364be0f70e6802125a8f3b05638b375fdb` |
| `rust-engine/experiments/mer2_cuda_g3_s1/src/lib.rs` | `f21140b5852922825bf4f9dd0358ae199f9948e6bd30a91c906c91db75b30caf` |
| `rust-engine/vendor/candle-kernels/src/quantized.cu` | `510c665e914346cbf361c4c60463a9cbad28032c09760cd90a227d9d2ab92972` |
| `rust-engine/vendor/candle-kernels/src/unary.cu` | `bea116bebb9b84106661dd425ed56ceddabbff337762edcee05f6e9134fb7e5b` |
| `rust-engine/vendor/candle-kernels/src/binary.cu` | `dc9961a0013fe24972adf27a37a986df5d78081912a6de21684a0433ba8d1b17` |
| `rust-engine/vendor/candle-kernels/build.rs` | `d2683e67d2c82d7d3b2e03e937c625155e2218702fe21cf38a164a82b8bdeedf` |
| `rust-engine/cuda/glibc_cuda_compat.h` | `7f84797c9c440978534dd54d7ee476b83e37e6d591063582f48e8f8b1b964fee` |
| `rust-engine/Cargo.lock` | `939ba0c96f0b5f1ed5e6df6beb01d2a7720c82f4654d33ad67a7ef4792c59dca` |

No vendor code was copied or modified. PTX/module source hashes at a future
build must be checked against this reference and the complete unchanged vendor
inventory, not just these three `.cu` files. QUANTIZED/UNARY/BINARY PTX hashes,
binary hash, compiler/toolkit, driver, host boot and L4 identity are all UNBOUND.

The new source has no external dependencies. Referenced W3 registry identities
are historical source pins, not the new crate's installed closure:
`cudarc 0.19.7` checksum
`1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc`;
`cudaforge 0.1.5` checksum
`6f7a0d45b139b5beeeb1c34188717e12241c44a0120afb498815ce7f5373c691`.
The path-patched `candle-kernels 0.10.2` has no registry checksum in that lock;
its bytes/build-script/header identity must be retained separately.
