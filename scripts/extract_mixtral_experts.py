#!/usr/bin/env python3
"""
Extract per-expert SwiGLU FFN weights from a Hugging Face MoE checkpoint
and write them as `expert_<id>.bin` files plus a `metadata.json`, in the
on-disk format expected by `micro-expert-router run`.

The Rust engine streams real expert weights from SSD; this script is the
bridge from a Hugging Face checkpoint to that on-disk layout.

### Supported architectures

The script is architecture-aware (matching the Rust `Architecture` enum):

* **Mixtral** (`mixtral`) — experts under
  `model.layers.{L}.block_sparse_moe.experts.{j}.w1/w3/w2.weight`
  (`w1`=gate, `w3`=up, `w2`=down), router gate
  `…block_sparse_moe.gate.weight`. No shared expert.
* **Qwen3-MoE** (`qwen3_moe`) — experts under
  `model.layers.{L}.mlp.experts.{j}.{gate,up,down}_proj.weight`, router
  gate `…mlp.gate.weight`. No shared expert.
* **DeepSeek-V3 / V3.1** (`deepseek_v3`) — same `mlp.experts.{j}.…`
  expert naming plus an always-on shared expert
  (`…mlp.shared_experts.{gate,up,down}_proj.weight`). FP8
  block-quantised checkpoints are de-quantised on the fly: every
  `F8_E4M3` tensor is decoded and multiplied by its companion
  `weight_scale_inv` per-128x128-block reciprocal scale, mirroring the
  engine's `dequant_fp8_e4m3_blockwise` (`rust-engine/src/mla.rs`).

Fully **dense** architectures (Qwen3 dense `qwen3`, Mistral Small 3
`mistral3`, Phi-4 `phi3`) have no routed experts to stream and are
refused with a clear error — run them directly from `.safetensors`
instead.

The architecture is auto-detected from the checkpoint's `config.json`
(`architectures` / `model_type`); override with `--architecture`.

### On-disk layout (per expert, little-endian f32)

    gate_proj  : [d_ff   x d_model]   row-major
    up_proj    : [d_ff   x d_model]   row-major
    down_proj  : [d_model x d_ff  ]   row-major

The file is then zero-padded to a multiple of `block_align` (default
4096) so that `O_DIRECT` reads work without `EINVAL` on Linux NVMe.

For MoE checkpoints the script additionally emits, alongside the
per-expert files, the resident tensors the engine's `from_dir` loader
auto-discovers:

    gate_<L>.bin                       : router gate, [num_experts x d_model] f32
    layer_<L>_shexp_gate.bin           : shared-expert gate_proj  (f32)
    layer_<L>_shexp_up.bin             : shared-expert up_proj    (f32)
    layer_<L>_shexp_down.bin           : shared-expert down_proj  (f32)
    layer_<L>_shexp_gate_inp.bin       : shared-expert sigmoid gate (f32, optional)

Router-gate and shared-expert files are always written as little-endian
`f32` (the engine reads them through `read_full_f32`), independent of the
per-expert `--dtype`.

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

    # Mixtral (auto-detected)
    python scripts/extract_mixtral_experts.py \\
        --model mistralai/Mixtral-8x7B-v0.1 \\
        --layer 0 \\
        --out ./data \\
        --block-align 4096

    # Qwen3-MoE (auto-detected from config.json)
    python scripts/extract_mixtral_experts.py \\
        --model Qwen/Qwen3-30B-A3B \\
        --layer all \\
        --out ./data-qwen3

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
        "--architecture",
        choices=["auto", "mixtral", "qwen3_moe", "deepseek_v3"],
        default="auto",
        help=(
            "MoE architecture / tensor-name schema. `auto` (default) "
            "detects it from the checkpoint's config.json (`architectures` "
            "/ `model_type`). Override only if auto-detection fails."
        ),
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


# Per-architecture MoE tensor-name schemas. `{L}` is the layer index and
# `{j}` the routed-expert id. These match the exact HuggingFace
# `safetensors` keys for each family and mirror the Rust `TensorNaming`
# map in `rust-engine/src/architecture.rs`.
MOE_SCHEMAS: dict[str, dict[str, str | None]] = {
    "mixtral": {
        # Mixtral: `w1`=gate_proj, `w3`=up_proj, `w2`=down_proj.
        "expert_marker": ".block_sparse_moe.experts.",
        "expert_gate": "model.layers.{L}.block_sparse_moe.experts.{j}.w1.weight",
        "expert_up": "model.layers.{L}.block_sparse_moe.experts.{j}.w3.weight",
        "expert_down": "model.layers.{L}.block_sparse_moe.experts.{j}.w2.weight",
        "router_gate": "model.layers.{L}.block_sparse_moe.gate.weight",
        # Mixtral has no shared expert.
        "shared_gate": None,
        "shared_up": None,
        "shared_down": None,
        "shared_gate_inp": None,
    },
    "qwen3_moe": {
        "expert_marker": ".mlp.experts.",
        "expert_gate": "model.layers.{L}.mlp.experts.{j}.gate_proj.weight",
        "expert_up": "model.layers.{L}.mlp.experts.{j}.up_proj.weight",
        "expert_down": "model.layers.{L}.mlp.experts.{j}.down_proj.weight",
        "router_gate": "model.layers.{L}.mlp.gate.weight",
        # Qwen3-MoE has no shared expert.
        "shared_gate": None,
        "shared_up": None,
        "shared_down": None,
        "shared_gate_inp": None,
    },
    "deepseek_v3": {
        "expert_marker": ".mlp.experts.",
        "expert_gate": "model.layers.{L}.mlp.experts.{j}.gate_proj.weight",
        "expert_up": "model.layers.{L}.mlp.experts.{j}.up_proj.weight",
        "expert_down": "model.layers.{L}.mlp.experts.{j}.down_proj.weight",
        "router_gate": "model.layers.{L}.mlp.gate.weight",
        # DeepSeek-MoE keeps one always-on shared expert and no sigmoid
        # shared-gate scalar.
        "shared_gate": "model.layers.{L}.mlp.shared_experts.gate_proj.weight",
        "shared_up": "model.layers.{L}.mlp.shared_experts.up_proj.weight",
        "shared_down": "model.layers.{L}.mlp.shared_experts.down_proj.weight",
        "shared_gate_inp": None,
    },
}

# Fully dense decoder families (no routed experts to stream). Listed so
# the script can refuse with a clear, architecture-specific message
# instead of a late "no MoE layers found" error.
DENSE_ARCHS = {"qwen3", "mistral3", "phi3"}

# HuggingFace `architectures[]` entry -> our schema key.
_HF_ARCH_TO_KEY = {
    "MixtralForCausalLM": "mixtral",
    "Qwen3MoeForCausalLM": "qwen3_moe",
    "DeepseekV3ForCausalLM": "deepseek_v3",
    "Qwen3ForCausalLM": "qwen3",
    "Mistral3ForConditionalGeneration": "mistral3",
    "Phi3ForCausalLM": "phi3",
}


def detect_architecture(config) -> str | None:
    """Map an HF `AutoConfig` to one of our schema keys.

    Prefers the `architectures` list (most specific), then falls back to
    `model_type`. Returns `None` if neither is recognised.
    """
    for arch in getattr(config, "architectures", None) or []:
        if arch in _HF_ARCH_TO_KEY:
            return _HF_ARCH_TO_KEY[arch]
    model_type = (getattr(config, "model_type", "") or "").strip()
    if model_type in MOE_SCHEMAS or model_type in DENSE_ARCHS:
        return model_type
    return None


def _first_attr(config, names: list[str]):
    """Return the first present, non-None config attribute among `names`."""
    for name in names:
        val = getattr(config, name, None)
        if val is not None:
            return int(val)
    return None


def resolve_moe_config(config, arch: str) -> dict:
    """Resolve the architecture-specific MoE hyperparameters.

    The expert count and FFN-width spelling differ per family:
    Mixtral uses `num_local_experts` + `intermediate_size`; Qwen3-MoE
    and DeepSeek use `num_experts` / `n_routed_experts` +
    `moe_intermediate_size`.
    """
    num_experts = _first_attr(
        config, ["num_local_experts", "num_experts", "n_routed_experts"]
    )
    top_k = _first_attr(config, ["num_experts_per_tok"])
    d_model = _first_attr(config, ["hidden_size"])
    if arch == "mixtral":
        d_ff = _first_attr(config, ["intermediate_size"])
    else:
        d_ff = _first_attr(config, ["moe_intermediate_size", "intermediate_size"])
    num_shared_experts = _first_attr(config, ["n_shared_experts"]) or 0
    first_k_dense_replace = _first_attr(config, ["first_k_dense_replace"]) or 0
    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_shared_experts": num_shared_experts,
        "first_k_dense_replace": first_k_dense_replace,
    }


def detect_fp8(config) -> bool:
    """True if the checkpoint is FP8 block-quantised (DeepSeek-V3).

    Such weights carry companion `weight_scale_inv` tensors: a per-block
    f32 reciprocal scale laid out over a 128x128 block grid. This script
    de-quantises them on the fly (mirroring the Rust engine's
    `dequant_fp8_e4m3_blockwise` in `rust-engine/src/mla.rs`) by reading
    the raw `safetensors` shards directly instead of instantiating the
    model through `transformers`.
    """
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    return bool(method) and "fp8" in str(method).lower()


# Default FP8 quantisation block edge (DeepSeek-V3 `weight_block_size`).
FP8_BLOCK = 128

# Largest finite magnitude in the OCP `e4m3fn` format (S.1110.111 =
# 1.875 * 2^8). The S.1111.111 NaN pattern is clamped to this value so
# downstream matmuls stay finite — bit-for-bit the behaviour of the Rust
# engine's `f8_e4m3_to_f32` (`rust-engine/src/mla.rs`).
F8_E4M3_MAX_FINITE = 448.0


def f8_e4m3_lut():
    """256-entry `uint8 -> f32` decode table for FP8 `e4m3fn`.

    1 sign / 4 exponent / 3 mantissa, bias 7, no infinities; the
    all-ones encoding (`0x7F`/`0xFF`) is NaN in the spec and clamped to
    +-448 here, matching the Rust engine's decoder exactly.
    """
    import numpy as np

    lut = np.empty(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0:
            val = sign * (mant / 8.0) * 2.0 ** (1 - 7)
        elif exp == 0x0F and mant == 0x07:
            val = sign * F8_E4M3_MAX_FINITE
        else:
            val = sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        lut[b] = val
    return lut


def dequant_fp8_e4m3_blockwise(q_u8, scale_inv, block: int = FP8_BLOCK):
    """Block-wise de-quantise an FP8 `e4m3fn` matrix to f32.

    * `q_u8`      — `[rows, cols]` uint8 array of raw e4m3 bytes.
    * `scale_inv` — `[ceil(rows/block), ceil(cols/block)]` f32 per-block
                    reciprocal scales (DeepSeek `weight_scale_inv`).
    * `block`     — square block edge (DeepSeek uses 128).

    Mirrors `dequant_fp8_e4m3_blockwise` in `rust-engine/src/mla.rs`:
    `out[r, c] = decode_e4m3(q[r, c]) * scale_inv[r // block, c // block]`.
    """
    import numpy as np

    q_u8 = np.asarray(q_u8, dtype=np.uint8)
    scale_inv = np.asarray(scale_inv, dtype=np.float32)
    if q_u8.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError("dequant_fp8_e4m3_blockwise expects 2-D weight and scale arrays")
    rows, cols = q_u8.shape
    want = (-(-rows // block), -(-cols // block))  # ceil-div
    if scale_inv.shape != want:
        raise ValueError(
            f"weight_scale_inv shape {scale_inv.shape} does not match the "
            f"{want} block grid of a {rows}x{cols} weight (block={block})"
        )
    out = f8_e4m3_lut()[q_u8]
    # Expand the per-block scales over the element grid and crop the
    # ragged trailing blocks.
    expanded = np.repeat(np.repeat(scale_inv, block, axis=0), block, axis=1)[:rows, :cols]
    out *= expanded
    return out


def _np_from_raw(dtype: str, raw: bytes, shape):
    """Decode a raw safetensors buffer (`F32`/`F16`/`BF16`) to f32."""
    import numpy as np

    if dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    return arr.reshape(shape).copy()


class Fp8StateDict:
    """Lazy state-dict view over an FP8 `safetensors` checkpoint.

    Instantiating an FP8 DeepSeek checkpoint through
    `AutoModelForCausalLM` requires a working fp8 quantisation backend
    (and usually a GPU). This view instead reads the raw shards
    directly — the 8-byte header-length prefix + JSON header layout is
    trivial — and de-quantises any `F8_E4M3` tensor with its companion
    `<name>_scale_inv` per-block scales on access. Non-FP8 tensors
    (`F32`/`F16`/`BF16`, e.g. layernorms and the router gate) decode
    straight to f32. Returned tensors are `torch.Tensor`s so the rest
    of the extraction pipeline is unchanged.
    """

    def __init__(self, ckpt_dir: Path):
        self.dir = Path(ckpt_dir)
        # name -> (shard_path, dtype, shape, start, end); offsets are
        # relative to the shard's data section (after the JSON header).
        self._index: dict[str, tuple[Path, str, list[int], int, int]] = {}
        shards = sorted(self.dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors shards in {self.dir}")
        for shard in shards:
            for name, entry in self._read_header(shard).items():
                self._index[name] = entry

    @staticmethod
    def _read_header(shard: Path) -> dict:
        import struct

        out = {}
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        data_base = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            out[name] = (shard, meta["dtype"], meta["shape"], data_base + start, data_base + end)
        return out

    def keys(self):
        return self._index.keys()

    def _raw(self, name: str) -> tuple[str, list[int], bytes]:
        shard, dtype, shape, start, end = self._index[name]
        with open(shard, "rb") as f:
            f.seek(start)
            raw = f.read(end - start)
        return dtype, shape, raw

    def get(self, name: str):
        import numpy as np
        import torch

        if name not in self._index:
            return None
        dtype, shape, raw = self._raw(name)
        if dtype == "F8_E4M3":
            q = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            scale_name = f"{name}_scale_inv"
            if scale_name not in self._index:
                raise KeyError(
                    f"FP8 tensor {name!r} has no companion {scale_name!r}; "
                    "cannot de-quantise"
                )
            s_dtype, s_shape, s_raw = self._raw(scale_name)
            scale = _np_from_raw(s_dtype, s_raw, s_shape)
            arr = dequant_fp8_e4m3_blockwise(q, scale, FP8_BLOCK)
        else:
            arr = _np_from_raw(dtype, raw, shape)
        return torch.from_numpy(np.ascontiguousarray(arr))


def resolve_checkpoint_dir(model: str) -> Path:
    """Local checkpoint directory for `model` (a path or an HF repo id).

    HF repo ids are materialised via `huggingface_hub.snapshot_download`
    restricted to the `safetensors` shards + JSON sidecars the FP8 path
    reads.
    """
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
    )


def write_f32_matrix(path: Path, arr) -> None:
    """Write a tensor as a row-major little-endian `f32` `.bin` file.

    Used for the router gate and shared-expert tensors, which the Rust
    `from_dir` loader always reads as `f32` via `read_full_f32`
    (independent of the per-expert `--dtype`).
    """
    import numpy as np

    flat = arr.astype(np.float32, copy=False).reshape(-1)
    path.write_bytes(flat.tobytes(order="C"))


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

    # tqdm is optional: if it isn't installed, fall back to a no-op
    # progress shim. We don't want to fail a multi-hour Mixtral dump
    # because someone missed `pip install tqdm`. The shim mirrors the
    # subset of the tqdm API used below (`update`/`close`), so it works
    # with the keyword-only `tqdm(total=..., desc=...)` call.
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover — runtime guidance only
        class tqdm:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                pass

            def update(self, _n=1):
                pass

            def close(self):
                pass

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading config + model: {args.model}", file=sys.stderr)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)

    # Resolve the architecture (and therefore the tensor-name schema)
    # from the checkpoint's config, unless explicitly overridden.
    if args.architecture == "auto":
        arch = detect_architecture(config)
        if arch is None:
            print(
                f"error: could not auto-detect the architecture of {args.model!r} "
                f"(model_type={getattr(config, 'model_type', None)!r}, "
                f"architectures={getattr(config, 'architectures', None)!r}). "
                "Pass --architecture {mixtral,qwen3_moe,deepseek_v3} explicitly.",
                file=sys.stderr,
            )
            return 2
    else:
        arch = args.architecture

    # Fail fast on fully dense families: they have no routed experts to
    # stream off SSD, so there is nothing for this script to extract.
    if arch in DENSE_ARCHS:
        print(
            f"error: {args.model!r} is a dense architecture ({arch!r}) with no "
            "routed experts to extract. Dense models do not exercise the SSD "
            "expert-streaming path — run them directly from their .safetensors "
            "(see the engine's safetensors loader).",
            file=sys.stderr,
        )
        return 2

    if arch not in MOE_SCHEMAS:
        print(
            f"error: unsupported architecture {arch!r}; expected one of "
            f"{sorted(MOE_SCHEMAS)}.",
            file=sys.stderr,
        )
        return 2

    # DeepSeek-V3 ships FP8 block-quantised weights with companion
    # `weight_scale_inv` tensors. The engine de-quantises these at load
    # (MLA + FP8 are implemented in `rust-engine/src/mla.rs` /
    # `model.rs`); this script mirrors that de-quantisation by reading
    # the raw safetensors shards directly — no fp8 runtime backend (or
    # GPU) needed.
    is_fp8 = detect_fp8(config)
    if is_fp8:
        print(
            f"{args.model!r} ({arch!r}) is FP8 block-quantised; de-quantising "
            "expert tensors via the companion `weight_scale_inv` per-block "
            f"scales (block={FP8_BLOCK}) while extracting.",
            file=sys.stderr,
        )

    schema = MOE_SCHEMAS[arch]

    # The expert-count / FFN-width spelling differs per family; resolve
    # them up front so a non-MoE model fails clearly rather than with a
    # late KeyError deep in the state-dict walk.
    resolved = resolve_moe_config(config, arch)
    # The config attribute names tried for each resolved field, surfaced
    # in the error message so a malformed config is easy to debug.
    field_sources = {
        "num_experts": ["num_local_experts", "num_experts", "n_routed_experts"],
        "top_k": ["num_experts_per_tok"],
        "d_model": ["hidden_size"],
        "d_ff": ["intermediate_size"] if arch == "mixtral" else ["moe_intermediate_size", "intermediate_size"],
    }
    missing = [k for k in field_sources if resolved[k] is None]
    if missing:
        detail = "; ".join(f"{k} (tried {field_sources[k]})" for k in missing)
        print(
            f"error: model {args.model!r} ({arch!r}) is missing MoE config "
            f"fields: {detail}. Is this actually a {arch} MoE checkpoint?",
            file=sys.stderr,
        )
        return 2

    num_experts: int = resolved["num_experts"]
    top_k: int = resolved["top_k"]
    d_model: int = resolved["d_model"]
    d_ff: int = resolved["d_ff"]
    num_shared_experts: int = resolved["num_shared_experts"]
    first_k_dense_replace: int = resolved["first_k_dense_replace"]
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
        f"architecture {arch!r}: {num_experts} experts/layer, top_k={top_k}, "
        f"d_model={d_model}, d_ff={d_ff} dtype={args.dtype} "
        f"(shared_experts={num_shared_experts}, first_k_dense_replace="
        f"{first_k_dense_replace}) -> "
        f"{weight_bytes / 1024 / 1024:.1f} MiB/expert "
        f"(padded to {expert_size / 1024 / 1024:.1f} MiB on disk)",
        file=sys.stderr,
    )

    # Load weights on CPU. FP8 checkpoints are read straight from their
    # safetensors shards (lazy, de-quantised per tensor on access — see
    # `Fp8StateDict`); everything else goes through the standard
    # `transformers` fp32 instantiation. Mixtral-8x7B's full FFN is
    # large (~88 GiB at fp32 across all 32 layers); for a single layer
    # it fits in modest RAM.
    if is_fp8:
        print("opening FP8 safetensors shards (lazy, no model instantiation)...", file=sys.stderr)
        sd = Fp8StateDict(resolve_checkpoint_dir(args.model))
    else:
        print("loading model weights to CPU (this can take a while)...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        # Walk the state dict for the requested layer(s)' experts, using the
        # architecture's tensor-name schema (Mixtral `block_sparse_moe.*` vs
        # Qwen3-MoE / DeepSeek `mlp.experts.{j}.{gate,up,down}_proj`).
        sd = model.state_dict()

    # Discover MoE layers present in the state dict, then resolve the
    # `--layer` spec (or legacy `--all-layers`) against them. The expert
    # tensors live under an architecture-specific marker.
    expert_marker = schema["expert_marker"]
    available_layers = sorted({
        int(name.split(".")[2])
        for name in sd.keys()
        if name.startswith("model.layers.") and expert_marker in name
    })
    if not available_layers:
        print(
            f"error: no MoE layers found in state dict (looked for "
            f"{expert_marker!r}); is this actually a {arch} model?",
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
        for expert_id in range(n):
            gate_key = schema["expert_gate"].format(L=layer, j=expert_id)
            up_key = schema["expert_up"].format(L=layer, j=expert_id)
            down_key = schema["expert_down"].format(L=layer, j=expert_id)
            w_gate = sd.get(gate_key)  # gate_proj [d_ff, d_model]
            w_up = sd.get(up_key)      # up_proj   [d_ff, d_model]
            w_down = sd.get(down_key)  # down_proj [d_model, d_ff]
            if w_gate is None or w_up is None or w_down is None:
                progress.close()
                print(
                    f"error: missing expert tensors for layer {layer} expert "
                    f"{expert_id} (looked for {gate_key!r}); is the layer index "
                    "correct?",
                    file=sys.stderr,
                )
                return 1
            # Sanity-check shapes — fail loudly if HF ever changes the layout.
            assert tuple(w_gate.shape) == (d_ff, d_model), (
                f"gate_proj for layer {layer} expert {expert_id} has shape "
                f"{tuple(w_gate.shape)}, expected ({d_ff}, {d_model})"
            )
            assert tuple(w_up.shape) == (d_ff, d_model), (
                f"up_proj for layer {layer} expert {expert_id} has shape "
                f"{tuple(w_up.shape)}, expected ({d_ff}, {d_model})"
            )
            assert tuple(w_down.shape) == (d_model, d_ff), (
                f"down_proj for layer {layer} expert {expert_id} has shape "
                f"{tuple(w_down.shape)}, expected ({d_model}, {d_ff})"
            )

            gate_f32 = w_gate.to(torch.float32).contiguous().cpu().numpy()
            up_f32 = w_up.to(torch.float32).contiguous().cpu().numpy()
            down_f32 = w_down.to(torch.float32).contiguous().cpu().numpy()

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

    # Emit the resident router gate (`gate_<L>.bin`) and any shared expert
    # (`layer_<L>_shexp_*.bin`) for each dumped layer. The Rust `from_dir`
    # loader auto-discovers these and reads them as little-endian `f32`
    # (via `read_full_f32`), independent of the per-expert `--dtype`.
    gates_written = 0
    shared_written = 0
    for layer in layer_ids:
        gate_w = sd.get(schema["router_gate"].format(L=layer))
        if gate_w is not None:
            assert tuple(gate_w.shape) == (num_experts, d_model), (
                f"router gate for layer {layer} has shape {tuple(gate_w.shape)}, "
                f"expected ({num_experts}, {d_model})"
            )
            write_f32_matrix(
                out / f"gate_{layer}.bin",
                gate_w.to(torch.float32).contiguous().cpu().numpy(),
            )
            gates_written += 1

        if schema["shared_gate"] is not None:
            sh_gate = sd.get(schema["shared_gate"].format(L=layer))
            sh_up = sd.get(schema["shared_up"].format(L=layer))
            sh_down = sd.get(schema["shared_down"].format(L=layer))
            if sh_gate is not None and sh_up is not None and sh_down is not None:
                write_f32_matrix(
                    out / f"layer_{layer}_shexp_gate.bin",
                    sh_gate.to(torch.float32).contiguous().cpu().numpy(),
                )
                write_f32_matrix(
                    out / f"layer_{layer}_shexp_up.bin",
                    sh_up.to(torch.float32).contiguous().cpu().numpy(),
                )
                write_f32_matrix(
                    out / f"layer_{layer}_shexp_down.bin",
                    sh_down.to(torch.float32).contiguous().cpu().numpy(),
                )
                # Optional sigmoid shared-gate scalar (Qwen2-MoE); DeepSeek
                # omits it. The engine treats it as optional.
                if schema["shared_gate_inp"] is not None:
                    sh_gate_inp = sd.get(schema["shared_gate_inp"].format(L=layer))
                    if sh_gate_inp is not None:
                        write_f32_matrix(
                            out / f"layer_{layer}_shexp_gate_inp.bin",
                            sh_gate_inp.to(torch.float32).contiguous().cpu().numpy(),
                        )
                shared_written += 1
    if gates_written:
        print(f"wrote {gates_written} router gate file(s) (gate_<L>.bin)", file=sys.stderr)
    if shared_written:
        print(
            f"wrote {shared_written} shared-expert set(s) (layer_<L>_shexp_*.bin)",
            file=sys.stderr,
        )

    metadata = {
        "model": args.model,
        "architecture": arch,
        "layer": layer_ids[0] if not multi_layer else None,
        "layers": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": n,
        "top_k": top_k,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_shared_experts": num_shared_experts,
        "first_k_dense_replace": first_k_dense_replace,
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
