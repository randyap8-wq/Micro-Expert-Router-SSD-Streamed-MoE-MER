# HMA-1F mapped destination VMA characterization

`probe-gpu-native-mapped-destination-vma --report-out <new-report.json>`

This diagnostic implements the read-only observation requested after
[issue #180 comment 5691436019](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/180#issuecomment-5691436019).
The consumed capability probe is not called. Implementation and portable testing
provide no hardware characterization result or authorization for a performance
experiment.

The command dispatches before logging, ordinary configuration, affinity, worker
pools, model, storage, and production runtime initialization. It requires Linux
and the exact NVIDIA L4 discrete Vulkan adapter with vendor `0x10de`. It creates
one `UPLOAD_BYTES = FULL + 4096 = 2,662,400` byte `MAP_WRITE | COPY_SRC` buffer,
maps it for writing, and obtains one mutable mapped view. The existing
`aligned_offset` helper identifies the exact 2,658,304-byte aligned range. The
command does not write to the mapped bytes, submit a queue, copy a GPU buffer,
read expert sources, or perform any lock/policy/pinning intervention.

While the view is alive, the command reads `/proc/self/smaps` exactly once into a
complete string, then `/proc/self/status` exactly once. It drops the view, unmaps,
polls with `Maintain::Wait`, and pops the validation scope before parsing or
writing evidence. Error paths after buffer creation perform the same cleanup.
Any WGPU validation error fails the command.

The strict parser processes every entry in the captured smaps snapshot, including
entries following the containing VMA. Missing or duplicate required fields,
malformed headers, invalid units, numeric overflow, and incomplete snapshots
fail closed. Unknown numeric fields, two-letter flags, and the kernel's `??` unknown-bit
marker are preserved, following the [Linux smaps emitter](https://github.com/torvalds/linux/blob/master/fs/proc/task_mmu.c). Exactly
one VMA must contain the entire actual pointer range; pathname is descriptive.
Adjacent VMAs are never combined to satisfy containment.

The JSON schema is `mer.gpu-native-mapped-destination-vma.v1`:

- `hardware` contains adapter identity and driver information.
- `mapped_range` contains the pointer, exact length, alignment remainder, aligned
  offset, and upload buffer size.
- `vma` contains the address range, header metadata, pointer offset, full-range
  containment, required and available optional Linux fields, ordered `VmFlags`,
  and literal decoded flag booleans. Every `kB` field is converted to **bytes**;
  `THPeligible` and `ProtectionKey` remain scalars. The top-level
  `memory_field_unit` documents this distinction.
- `vma.raw_header_line` preserves the header without its newline.
  `vma.raw_entry` preserves the complete entry including original whitespace and
  line endings. `vma.raw_entry_sha256` hashes those exact UTF-8 bytes.
- `smaps_snapshot` binds the absolute path, SHA-256, and byte length of the
  complete captured snapshot, saved as `<report-out>.smaps`.
- `process_status` contains `VmLck`, `VmPin`, and `VmRSS` in bytes. These fields
  are descriptive; no locked-memory growth is required.
- `cleanup` records mapped-view drop, unmap, polling, scope pop, and any WGPU
  validation error.

Both output paths must be new and their parent directory must exist. Files are
reserved with create-new semantics before hardware construction, so existing
files or symlinks fail before capture. Available smaps evidence is preserved even
when parsing, status capture, or validation fails. A failure before smaps capture
may leave an empty reserved sibling with `smaps_snapshot: null`. Output files are
not overwritten or deleted on failure.

Exit success and `complete: true`, `characterization_pass: true` require capture,
cleanup, parsing, unique containment, and evidence persistence to succeed.
Otherwise the command returns an error and writes a failure report where possible.
The report makes no performance, mechanism-selection, qualification, or FIRST
recommendation.

Portable verification:

```sh
RUST_MIN_STACK=8388608 cargo test --manifest-path rust-engine/Cargo.toml \
  --bin micro-expert-router --features tokenizer gpu_native_mapped_vma::tests
```

These tests use synthetic proc snapshots and evidence files. They never invoke
Linux hardware capture; the non-Linux test verifies rejection before adapter
construction.
