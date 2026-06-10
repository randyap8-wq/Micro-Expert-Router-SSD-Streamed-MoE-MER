#!/usr/bin/env python3
"""Tests for the FP8 e4m3 de-quantisation path in
`extract_mixtral_experts.py`.

Mirrors the reference vectors of the Rust engine's
`rust-engine/src/mla.rs` tests (`fp8_e4m3_decodes_reference_values`,
`fp8_blockwise_dequant_*`) so both sides of the extraction pipeline are
guaranteed to agree bit-for-bit on the e4m3 decode and the per-block
scale application.

Run with: `python3 scripts/test_extract_fp8.py` (requires `numpy`;
the `Fp8StateDict` round-trip additionally runs when `torch` is
installed and is skipped otherwise).
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from extract_mixtral_experts import (
    F8_E4M3_MAX_FINITE,
    Fp8StateDict,
    dequant_fp8_e4m3_blockwise,
    f8_e4m3_lut,
)


def test_lut_decodes_reference_values():
    lut = f8_e4m3_lut()
    # Exact, well-known e4m3 encodings (same vectors as mla.rs tests).
    assert lut[0x00] == 0.0  # +0
    assert lut[0x38] == 1.0  # exp=7 (bias) mant=0 -> 1.0
    assert lut[0x40] == 2.0  # exp=8 -> 2.0
    assert lut[0xB8] == -1.0  # sign + 1.0
    assert lut[0x34] == 0.75  # exp=6 mant=4 -> (1+0.5)*0.5
    # Subnormal: mant/8 * 2^-6.
    assert lut[0x01] == (1.0 / 8.0) * 2.0 ** -6
    # NaN encodings clamp to +-448 (e4m3fn, matching the Rust decoder).
    assert lut[0x7F] == F8_E4M3_MAX_FINITE
    assert lut[0xFF] == -F8_E4M3_MAX_FINITE
    # Max normal finite: S.1110.111.
    assert lut[0x7E] == F8_E4M3_MAX_FINITE


def test_blockwise_dequant_applies_per_block_scale():
    # 2x2 matrix, block=1 so each element has its own scale
    # (mla.rs: fp8_blockwise_dequant_applies_per_block_scale).
    q = np.array([[0x38, 0x40], [0x38, 0x38]], dtype=np.uint8)  # 1, 2, 1, 1
    scale = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    out = dequant_fp8_e4m3_blockwise(q, scale, block=1)
    assert out.tolist() == [[2.0, 6.0], [4.0, 5.0]]


def test_blockwise_dequant_shares_scale_within_block():
    # 2x2 matrix, block=2 -> a single shared scale.
    q = np.array([[0x38, 0x40], [0x38, 0x40]], dtype=np.uint8)  # 1, 2, 1, 2
    scale = np.array([[10.0]], dtype=np.float32)
    out = dequant_fp8_e4m3_blockwise(q, scale, block=2)
    assert out.tolist() == [[10.0, 20.0], [10.0, 20.0]]


def test_blockwise_dequant_handles_ragged_blocks():
    # 3x5 matrix with block=2 -> ceil grid of 2x3 scales; trailing
    # ragged rows/cols must use the right block's scale.
    q = np.full((3, 5), 0x38, dtype=np.uint8)  # all 1.0
    scale = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    out = dequant_fp8_e4m3_blockwise(q, scale, block=2)
    expected = np.array(
        [
            [1.0, 1.0, 2.0, 2.0, 3.0],
            [1.0, 1.0, 2.0, 2.0, 3.0],
            [4.0, 4.0, 5.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(out, expected)


def test_blockwise_dequant_rejects_bad_shapes():
    q = np.zeros((2, 2), dtype=np.uint8)
    bad_scale = np.array([[1.0, 2.0]], dtype=np.float32)  # want 2x2 at block=1
    try:
        dequant_fp8_e4m3_blockwise(q, bad_scale, block=1)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched scale grid must raise ValueError")


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]):
    """Minimal safetensors writer: 8-byte LE header length + JSON header."""
    header = {}
    offset = 0
    blob = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        blob += raw
        offset += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_fp8_state_dict_dequantises_on_access():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (torch not installed; skipping Fp8StateDict round-trip)")
        return
    with tempfile.TemporaryDirectory() as d:
        dirp = Path(d)
        # 2x2 fp8 weight (values 1, 2, 1, 1) + per-element scales, and a
        # plain f32 tensor alongside.
        q = bytes([0x38, 0x40, 0x38, 0x38])
        scale = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        gate = np.array([[0.5, -0.5]], dtype=np.float32)
        _write_safetensors(
            dirp / "model-00001-of-00001.safetensors",
            {
                "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [2, 2], q),
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": (
                    "F32",
                    [2, 2],
                    scale.tobytes(),
                ),
                "model.layers.0.mlp.gate.weight": ("F32", [1, 2], gate.tobytes()),
            },
        )
        # Block edge 128 with a 2x2 weight needs a 1x1 scale grid; the
        # synthetic file uses per-element scales, so monkeypatch the
        # block to 1 for the round-trip.
        import extract_mixtral_experts as ex

        old_block = ex.FP8_BLOCK
        ex.FP8_BLOCK = 1
        try:
            sd = Fp8StateDict(dirp)
            w = sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
            assert w is not None
            assert w.shape == (2, 2)
            assert w.numpy().tolist() == [[2.0, 6.0], [4.0, 5.0]]
            g = sd.get("model.layers.0.mlp.gate.weight")
            assert g.numpy().tolist() == [[0.5, -0.5]]
            assert sd.get("missing.tensor") is None
        finally:
            ex.FP8_BLOCK = old_block


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"running {t.__name__} ...", file=sys.stderr)
        t()
    print(f"ok: {len(tests)} test(s) passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
