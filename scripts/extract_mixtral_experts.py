#!/usr/bin/env python3
"""
Extract per-expert SwiGLU FFN weights from a Mixtral checkpoint and write
them as `expert_<id>.bin` files plus a `metadata.json`, in the on-disk
format expected by `micro-expert-router run`.

The Rust engine streams real expert weights from SSD; this script is the
bridge from a Hugging Face Mixtral checkpoint to that on-disk layout.

### On-disk layout (per expert, little-endian f32)

    gate_proj  : [d_ff   x d_model]   row-major
    up_proj    : [d_ff   x d_model]   row-major
    down_proj  : [d_model x d_ff  ]   row-major

The file is then zero-padded to a multiple of `block_align` (default
4096) so that `O_DIRECT` reads work without `EINVAL` on Linux NVMe.

### metadata.json

    {
      "model": "<HF model name>",
      "layer": <which layer's experts were dumped>,
      "num_experts": N,
      "top_k": K,
      "d_model": d_model,
      "d_ff": d_ff,
      "expert_size": <bytes per file, including padding>,
      "block_align": 4096,
      "dtype": "f32",
      "weight_layout": "gate_proj || up_proj || down_proj (row-major)"
    }

`micro-expert-router run --data-dir <out>` reads this file and fills in
`--num-experts`, `--d-model`, `--d-ff`, `--top-k`, `--expert-size`
automatically (CLI flags still override).

### Constraints

* No GPU / accelerator code paths — weights are dequantised to f32 and
  saved on the host. The engine itself runs CPU-only by design.
* No quantisation in the on-disk format (the engine reinterprets bytes
  as `&[f32]` directly). If you're working with a quantised Mixtral
  checkpoint, this script materialises the dequantised weights — which
  is what you want to measure SSD-streamed dense FFN compute.

### Usage

    python scripts/extract_mixtral_experts.py \\
        --model mistralai/Mixtral-8x7B-v0.1 \\
        --layer 0 \\
        --out ./data \\
        --block-align 4096

Add `--limit N` while iterating to write only the first N experts (handy
for quick local smoke tests; the full Mixtral expert is ~176 MiB on
disk).

Requires `torch` and `transformers`. These are *not* listed as engine
dependencies because the Rust code never sees them — they're only
needed if you want to feed real Mixtral weights into the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Mixtral expert FFN weights to expert_<id>.bin + metadata.json",
    )
    p.add_argument(
        "--model",
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Hugging Face model id or local path.",
    )
    p.add_argument(
        "--layer",
        type=str,
        default="0",
        help=(
            "Which transformer layer's expert FFNs to dump. Accepts: "
            "a single integer (e.g. `--layer 0`), a comma-separated list "
            "(`--layer 0,2,5`), an inclusive range (`--layer 0-3`), or "
            "the literal `all` for every MoE layer in the checkpoint. "
            "Multi-layer dumps use `expert_<layer>_<id>.bin` filenames "
            "(consumed by the Rust `MultiLayerExpertCache`); single-"
            "integer dumps keep the legacy `expert_<id>.bin` naming."
        ),
    )
    p.add_argument(
        "--all-layers",
        action="store_true",
        help="Deprecated alias for `--layer all`. Kept for backward compat.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./data"),
        help="Output directory. Will be created if missing.",
    )
    p.add_argument(
        "--block-align",
        type=int,
        default=4096,
        help="O_DIRECT block alignment (bytes). File size is padded to this.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only dump the first N experts (debug / smoke tests).",
    )
    p.add_argument(
        "--dtype",
        choices=["f32", "f16", "int8"],
        default="f32",
        help=(
            "On-disk dtype. `f32` (legacy) writes 4 bytes per weight; "
            "`f16` writes 2 bytes per weight, halving the SSD bytes the "
            "engine must read on every cache miss (the dominant energy "
            "term). `int8` quartizes per-tensor symmetrically and "
            "prepends a 12-byte `[gate, up, down]` scale header per "
            "expert — quartering SSD bytes. The Rust engine dequantises "
            "to f32 in DRAM."
        ),
    )
    return p.parse_args()


def parse_layer_spec(spec: str, available: list[int]) -> list[int]:
    """Resolve a `--layer` argument into a sorted list of layer indices.

    Accepts: ``"all"``, a single int (``"3"``), a comma-separated list
    (``"0,2,5"``), or an inclusive range (``"0-3"``). Returns layer ids
    in ascending order, deduplicated. Raises ``ValueError`` with a
    user-friendly message on malformed input.
    """
    s = spec.strip().lower()
    if s == "all":
        return sorted(available)
    out: set[int] = set()
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_str, hi_str = chunk.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if lo > hi:
                raise ValueError(f"empty range {chunk!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def quantize_int8_per_tensor(arr):
    """Symmetrically quantize a tensor to int8 with a single scalar scale.

    Returns ``(scale: float, qbytes: bytes)`` where the dequantization is
    ``f32_value = i8_value * scale``. ``scale`` is ``max_abs / 127`` so
    that the range fully covers ``[-127, +127]``; ``-128`` is avoided
    for symmetry. Tiny tensors with all-zero values get a scale of
    ``1.0`` so dequantization remains a no-op rather than producing
    NaNs from a zero divisor.
    """
    import numpy as np
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    max_abs = float(np.abs(flat).max()) if flat.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
    return scale, q.tobytes(order="C")


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:  # pragma: no cover — runtime guidance only
        print(
            f"error: missing dependency: {e}\n"
            "install with: pip install 'transformers>=4.38' torch",
            file=sys.stderr,
        )
        return 2

    # tqdm is optional: if it isn't installed, fall back to a stub that
    # just yields the iterable unchanged. We don't want to fail a
    # multi-hour Mixtral dump because someone missed `pip install tqdm`.
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover — runtime guidance only
        def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
            return iterable

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading config + model: {args.model}", file=sys.stderr)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)

    # Mixtral / Llama-MoE expose these names in the HF config; assert up
    # front so a non-MoE model fails with a clear error rather than a
    # late KeyError deep in the state-dict walk.
    required_attrs = {
        "num_local_experts": "number of experts per MoE layer",
        "num_experts_per_tok": "top-K experts activated per token",
        "hidden_size": "d_model (residual-stream dim)",
        "intermediate_size": "d_ff (FFN intermediate dim)",
    }
    for attr, descr in required_attrs.items():
        if not hasattr(config, attr):
            print(
                f"error: model {args.model!r} doesn't look like a Mixtral-style MoE: "
                f"config is missing `{attr}` ({descr}).",
                file=sys.stderr,
            )
            return 2

    num_experts: int = int(config.num_local_experts)
    top_k: int = int(config.num_experts_per_tok)
    d_model: int = int(config.hidden_size)
    d_ff: int = int(config.intermediate_size)
    # SwiGLU expert holds three weight matrices (gate, up, down). The
    # Rust engine reinterprets the bytes as either `&[f32]` (4 B / weight)
    # or dequantises from little-endian `f16` (2 B / weight) depending on
    # `--dtype`. The on-disk byte order is always little-endian, matching
    # `compile_error!` guards in `inference.rs`.
    NUM_WEIGHT_MATRICES = 3
    if args.dtype == "f32":
        bytes_per_weight = 4
        header_bytes = 0
    elif args.dtype == "f16":
        bytes_per_weight = 2
        header_bytes = 0
    else:  # int8
        bytes_per_weight = 1
        header_bytes = 12  # 3 × f32 per-tensor scales (gate, up, down)
    weight_bytes = header_bytes + NUM_WEIGHT_MATRICES * d_model * d_ff * bytes_per_weight
    expert_size = ((weight_bytes + args.block_align - 1) // args.block_align) * args.block_align

    print(
        f"model has {num_experts} experts/layer, top_k={top_k}, "
        f"d_model={d_model}, d_ff={d_ff} dtype={args.dtype} -> "
        f"{weight_bytes / 1024 / 1024:.1f} MiB/expert "
        f"(padded to {expert_size / 1024 / 1024:.1f} MiB on disk)",
        file=sys.stderr,
    )

    # Load weights in fp32 on CPU. Mixtral-8x7B's full FFN is large
    # (~88 GiB at fp32 across all 32 layers); for a single layer it
    # fits in modest RAM.
    print("loading model weights to CPU (this can take a while)...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # Walk the state dict for the requested layer(s)' experts.
    # Mixtral parameter names look like:
    #   model.layers.<layer>.block_sparse_moe.experts.<expert_id>.w1.weight  (gate_proj)
    #   model.layers.<layer>.block_sparse_moe.experts.<expert_id>.w2.weight  (down_proj)
    #   model.layers.<layer>.block_sparse_moe.experts.<expert_id>.w3.weight  (up_proj)
    # Note: Mixtral's `w1` is the gate, `w3` is the up, `w2` is the down.
    sd = model.state_dict()

    # Discover MoE layers present in the state dict, then resolve the
    # `--layer` spec (or legacy `--all-layers`) against them.
    available_layers = sorted({
        int(name.split(".")[2])
        for name in sd.keys()
        if name.startswith("model.layers.") and ".block_sparse_moe.experts." in name
    })
    if not available_layers:
        print(
            "error: no MoE layers found in state dict; "
            "is this actually a Mixtral-style model?",
            file=sys.stderr,
        )
        return 1

    layer_spec = "all" if args.all_layers else args.layer
    try:
        layer_ids = parse_layer_spec(layer_spec, available_layers)
    except ValueError as e:
        print(f"error: malformed --layer {args.layer!r}: {e}", file=sys.stderr)
        return 2
    if not layer_ids:
        print(f"error: --layer {args.layer!r} resolved to an empty layer set", file=sys.stderr)
        return 2
    missing = [l for l in layer_ids if l not in available_layers]
    if missing:
        print(
            f"error: requested layers {missing} not present in checkpoint "
            f"(available: {available_layers[0]}..{available_layers[-1]})",
            file=sys.stderr,
        )
        return 2

    # Multi-layer dumps use the global `expert_<layer>_<id>.bin` naming
    # so the Rust `MultiLayerExpertCache` can map (layer, expert_id) ->
    # global expert index. Single-layer dumps keep the legacy name.
    multi_layer = len(layer_ids) > 1
    if multi_layer:
        print(
            f"--layer {layer_spec}: dumping {len(layer_ids)} layers "
            f"({layer_ids[0]}..{layer_ids[-1]})",
            file=sys.stderr,
        )
    else:
        print(f"--layer {layer_spec}: dumping layer {layer_ids[0]}", file=sys.stderr)

    n = num_experts if args.limit is None else min(num_experts, args.limit)
    written = 0
    total_experts = len(layer_ids) * n
    progress = tqdm(
        total=total_experts,
        desc=f"writing experts ({args.dtype})",
        unit="expert",
        file=sys.stderr,
    )
    for layer in layer_ids:
        prefix = f"model.layers.{layer}.block_sparse_moe.experts"
        for expert_id in range(n):
            w1 = sd.get(f"{prefix}.{expert_id}.w1.weight")  # gate_proj [d_ff, d_model]
            w3 = sd.get(f"{prefix}.{expert_id}.w3.weight")  # up_proj   [d_ff, d_model]
            w2 = sd.get(f"{prefix}.{expert_id}.w2.weight")  # down_proj [d_model, d_ff]
            if w1 is None or w2 is None or w3 is None:
                progress.close()
                print(
                    f"error: missing tensors for expert {expert_id} on layer {layer}; "
                    "is the layer index correct?",
                    file=sys.stderr,
                )
                return 1
            # Sanity-check shapes — fail loudly if HF ever changes the layout.
            assert tuple(w1.shape) == (d_ff, d_model), (
                f"w1 (gate_proj) for layer {layer} expert {expert_id} has shape "
                f"{tuple(w1.shape)}, expected ({d_ff}, {d_model})"
            )
            assert tuple(w3.shape) == (d_ff, d_model), (
                f"w3 (up_proj) for layer {layer} expert {expert_id} has shape "
                f"{tuple(w3.shape)}, expected ({d_ff}, {d_model})"
            )
            assert tuple(w2.shape) == (d_model, d_ff), (
                f"w2 (down_proj) for layer {layer} expert {expert_id} has shape "
                f"{tuple(w2.shape)}, expected ({d_model}, {d_ff})"
            )

            gate_f32 = w1.to(torch.float32).contiguous().cpu().numpy()
            up_f32 = w3.to(torch.float32).contiguous().cpu().numpy()
            down_f32 = w2.to(torch.float32).contiguous().cpu().numpy()

            # Multi-layer dumps use a global filename; single-layer
            # dumps keep the legacy name for backward compat.
            if multi_layer:
                path = out / f"expert_{layer}_{expert_id}.bin"
            else:
                path = out / f"expert_{expert_id}.bin"
            with open(path, "wb") as f:
                if args.dtype == "f32":
                    f.write(gate_f32.astype("float32", copy=False).tobytes(order="C"))
                    f.write(up_f32.astype("float32", copy=False).tobytes(order="C"))
                    f.write(down_f32.astype("float32", copy=False).tobytes(order="C"))
                elif args.dtype == "f16":
                    f.write(gate_f32.astype("float16", copy=False).tobytes(order="C"))
                    f.write(up_f32.astype("float16", copy=False).tobytes(order="C"))
                    f.write(down_f32.astype("float16", copy=False).tobytes(order="C"))
                else:  # int8: per-tensor symmetric quantization
                    import struct
                    g_scale, g_bytes = quantize_int8_per_tensor(gate_f32)
                    u_scale, u_bytes = quantize_int8_per_tensor(up_f32)
                    d_scale, d_bytes = quantize_int8_per_tensor(down_f32)
                    f.write(struct.pack("<fff", g_scale, u_scale, d_scale))
                    f.write(g_bytes)
                    f.write(u_bytes)
                    f.write(d_bytes)
                # Pad up to expert_size with zero bytes so O_DIRECT alignment
                # holds. The engine ignores anything past `weight_bytes`.
                wrote = f.tell()
                assert wrote == weight_bytes, (
                    f"internal: wrote {wrote} bytes, expected {weight_bytes}"
                )
                pad = expert_size - wrote
                if pad > 0:
                    f.write(b"\x00" * pad)
            written += 1
            progress.update(1)
    progress.close()
    print(f"finished writing {written} expert file(s)", file=sys.stderr)

    metadata = {
        "model": args.model,
        "layer": layer_ids[0] if not multi_layer else None,
        "layers": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": n,
        "top_k": top_k,
        "d_model": d_model,
        "d_ff": d_ff,
        "expert_size": expert_size,
        "block_align": args.block_align,
        "dtype": args.dtype,
        "weight_layout": (
            "[i8 header (12 B): gate_scale, up_scale, down_scale (f32)] || "
            "gate_proj || up_proj || down_proj (row-major, little-endian)"
            if args.dtype == "int8"
            else "gate_proj || up_proj || down_proj (row-major, little-endian)"
        ),
        "file_naming": "expert_<layer>_<id>.bin" if multi_layer else "expert_<id>.bin",
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {meta_path}", file=sys.stderr)
    print(
        f"\nnext step: cargo run --release --manifest-path rust-engine/Cargo.toml -- "
        f"run --data-dir {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
