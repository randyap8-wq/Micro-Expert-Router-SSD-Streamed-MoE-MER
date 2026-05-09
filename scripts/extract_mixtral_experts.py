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
        type=int,
        default=0,
        help="Which transformer layer's expert FFNs to dump (Mixtral has 32). "
             "Ignored when --all-layers is set.",
    )
    p.add_argument(
        "--all-layers",
        action="store_true",
        help="Dump experts from every layer. Files are named "
             "`expert_<layer>_<id>.bin` (vs `expert_<id>.bin` for single-layer). "
             "metadata.json includes `num_layers`. This is the format "
             "consumed by the Rust engine's MultiLayerExpertCache.",
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
        choices=["f32"],
        default="f32",
        help="On-disk dtype. Currently only f32 is supported by the engine.",
    )
    return p.parse_args()


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
    # SwiGLU expert holds three weight matrices (gate, up, down), each
    # of shape (d_ff, d_model) or (d_model, d_ff), stored as little-
    # endian f32 (4 bytes per element). The Rust engine reinterprets
    # these bytes as `&[f32]` directly — no quantisation on disk.
    NUM_WEIGHT_MATRICES = 3
    F32_BYTES = 4
    weight_bytes = NUM_WEIGHT_MATRICES * d_model * d_ff * F32_BYTES
    expert_size = ((weight_bytes + args.block_align - 1) // args.block_align) * args.block_align

    print(
        f"model has {num_experts} experts/layer, top_k={top_k}, "
        f"d_model={d_model}, d_ff={d_ff} -> {weight_bytes / 1024 / 1024:.1f} MiB/expert "
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

    # Discover the set of layers to dump. With --all-layers we walk every
    # block_sparse_moe layer present in the state dict; otherwise we
    # dump just `--layer`.
    if args.all_layers:
        layer_ids = sorted({
            int(name.split(".")[2])
            for name in sd.keys()
            if name.startswith("model.layers.") and ".block_sparse_moe.experts." in name
        })
        if not layer_ids:
            print(
                "error: --all-layers set but no MoE layers found in state dict; "
                "is this actually a Mixtral-style model?",
                file=sys.stderr,
            )
            return 1
        print(f"--all-layers: dumping {len(layer_ids)} layers ({layer_ids[0]}..{layer_ids[-1]})", file=sys.stderr)
    else:
        layer_ids = [args.layer]

    n = num_experts if args.limit is None else min(num_experts, args.limit)
    written = 0
    for layer in layer_ids:
        prefix = f"model.layers.{layer}.block_sparse_moe.experts"
        for expert_id in range(n):
            w1 = sd.get(f"{prefix}.{expert_id}.w1.weight")  # gate_proj [d_ff, d_model]
            w3 = sd.get(f"{prefix}.{expert_id}.w3.weight")  # up_proj   [d_ff, d_model]
            w2 = sd.get(f"{prefix}.{expert_id}.w2.weight")  # down_proj [d_model, d_ff]
            if w1 is None or w2 is None or w3 is None:
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

            # Concatenate as gate || up || down, contiguous row-major f32, LE.
            gate = w1.to(torch.float32).contiguous().cpu().numpy()
            up = w3.to(torch.float32).contiguous().cpu().numpy()
            down = w2.to(torch.float32).contiguous().cpu().numpy()

            # Single-layer: keep the legacy `expert_<id>.bin` name so the
            # existing run path keeps working without changes.
            # Multi-layer: `expert_<layer>_<id>.bin`, consumed by the
            # Rust `MultiLayerExpertCache`.
            if args.all_layers:
                path = out / f"expert_{layer}_{expert_id}.bin"
            else:
                path = out / f"expert_{expert_id}.bin"
            with open(path, "wb") as f:
                f.write(gate.tobytes(order="C"))
                f.write(up.tobytes(order="C"))
                f.write(down.tobytes(order="C"))
                # Pad up to expert_size with zero bytes so O_DIRECT alignment
                # holds. The engine ignores anything past `weight_bytes`.
                wrote = gate.nbytes + up.nbytes + down.nbytes
                assert wrote == weight_bytes, (
                    f"internal: wrote {wrote} bytes, expected {weight_bytes}"
                )
                pad = expert_size - wrote
                if pad > 0:
                    f.write(b"\x00" * pad)
            written += 1
            if written % 8 == 0:
                print(f"  wrote layer {layer} expert {expert_id} -> {path}", file=sys.stderr)
    # Always log the final write so the user sees a clean tail line even
    # when `written` isn't a multiple of 8 (e.g. with `--limit`).
    print(f"  finished writing {written} expert file(s)", file=sys.stderr)

    metadata = {
        "model": args.model,
        "layer": layer_ids[0] if not args.all_layers else None,
        "num_layers": len(layer_ids),
        "num_experts": n,
        "top_k": top_k,
        "d_model": d_model,
        "d_ff": d_ff,
        "expert_size": expert_size,
        "block_align": args.block_align,
        "dtype": args.dtype,
        "weight_layout": "gate_proj || up_proj || down_proj (row-major, little-endian f32)",
        "file_naming": "expert_<layer>_<id>.bin" if args.all_layers else "expert_<id>.bin",
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
