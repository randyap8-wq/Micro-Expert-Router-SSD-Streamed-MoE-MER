#!/usr/bin/env python3
"""Compute an expert alias map for runtime deduplication.

Given a directory of `expert_<id>.bin` files (single-layer) or
`expert_<layer>_<id>.bin` files (multi-layer), find pairs of experts
that are numerically near-identical — i.e. their cosine similarity over
the full weight blob is above a threshold — and emit a JSON map
`{ "src_id": canonical_id, ... }`.

The Rust engine accepts this map via `--alias-map` and remaps any
routed/predicted expert id through it before consulting the cache.
Pairs of near-identical experts therefore share a *single* resident
copy, halving the SSD bytes those redirects would otherwise burn.

Usage:

    python scripts/compute_expert_aliases.py \\
        --data-dir ./data \\
        --out      ./data/aliases.json \\
        --threshold 0.995

The `metadata.json` file written by `extract_mixtral_experts.py` is
read for `dtype` / `d_model` / `d_ff` / `expert_size` so the script
knows the on-disk layout. If `metadata.json` is missing you can pass
those values on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="JSON output path (passed to engine via --alias-map).")
    p.add_argument("--threshold", type=float, default=0.995,
                   help="Cosine similarity threshold above which two experts "
                        "are considered aliases (default 0.995).")
    p.add_argument("--dtype", choices=["f32", "f16"], default=None,
                   help="On-disk dtype (overrides metadata.json).")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--expert-size", type=int, default=None,
                   help="On-disk file size in bytes (overrides metadata.json).")
    p.add_argument("--num-experts", type=int, default=None,
                   help="Restrict scan to the first N experts.")
    return p.parse_args()


def load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def expert_files(data_dir: Path) -> List[Tuple[int, Path]]:
    """Return `(expert_id, path)` tuples for single-layer or layer-0 of
    multi-layer dumps. Stable sort by id."""
    out: List[Tuple[int, Path]] = []
    for p in sorted(data_dir.glob("expert_*.bin")):
        stem = p.stem.removeprefix("expert_")
        # Single-layer: "expert_<id>.bin"; multi-layer: "expert_<layer>_<id>.bin".
        # We only consider single-layer here for simplicity; for
        # multi-layer dumps, run this once per layer with --data-dir
        # filtered to that layer's files.
        if "_" in stem:
            continue
        try:
            out.append((int(stem), p))
        except ValueError:
            continue
    return out


def load_weights(path: Path, dtype: str, weight_floats: int):
    """Load the first `weight_floats` weights from `path` as a flat
    `numpy.float32` vector. Trailing padding (the file may be larger
    than the actual weights to satisfy O_DIRECT alignment) is ignored.
    Lazy-imports numpy so the rest of the script can be inspected
    without the dep."""
    import numpy as np  # local import keeps the help text usable without numpy

    if dtype == "f32":
        nbytes = weight_floats * 4
        raw = np.fromfile(path, dtype=np.float32, count=weight_floats)
        if raw.size < weight_floats:
            raise RuntimeError(
                f"{path}: expected {weight_floats} f32 weights, found {raw.size} "
                f"(file too small)."
            )
        return raw
    else:  # f16
        raw = np.fromfile(path, dtype=np.float16, count=weight_floats)
        if raw.size < weight_floats:
            raise RuntimeError(
                f"{path}: expected {weight_floats} f16 weights, found {raw.size}."
            )
        return raw.astype(np.float32, copy=False)


def cosine(a, b) -> float:
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    args = parse_args()
    meta = load_metadata(args.data_dir)
    dtype = args.dtype or meta.get("dtype", "f32")
    d_model = args.d_model or meta.get("d_model")
    d_ff = args.d_ff or meta.get("d_ff")
    if d_model is None or d_ff is None:
        print(
            "error: need d_model and d_ff. Pass --d-model / --d-ff or place "
            "metadata.json in --data-dir.",
            file=sys.stderr,
        )
        return 2

    weight_floats = 3 * d_model * d_ff  # gate || up || down
    print(
        f"scanning {args.data_dir} for aliases: dtype={dtype} d_model={d_model} "
        f"d_ff={d_ff} weight_floats={weight_floats} threshold={args.threshold}",
        file=sys.stderr,
    )

    files = expert_files(args.data_dir)
    if args.num_experts is not None:
        files = files[: args.num_experts]
    if not files:
        print(f"error: no expert_<id>.bin files found in {args.data_dir}", file=sys.stderr)
        return 2

    # Load all expert vectors. For Mixtral-8x7B (8 experts, ~88 MiB each)
    # that's ~700 MiB; for very large models, fall back to a streaming
    # similarity (left as future work).
    vecs = []
    for (eid, path) in files:
        v = load_weights(path, dtype, weight_floats)
        vecs.append((eid, v))

    # Pairwise cosine. Greedy clustering: each expert is either its own
    # canonical id, or aliases to a smaller id with which it has cosine
    # >= threshold. This produces a forest where every non-canonical
    # entry maps directly to a canonical (no chains).
    alias_map: Dict[int, int] = {}
    canonical_ids: List[int] = []
    for (eid, v) in vecs:
        chosen = None
        for cid in canonical_ids:
            cv = next(x for (xid, x) in vecs if xid == cid)
            sim = cosine(v, cv)
            if sim >= args.threshold:
                chosen = cid
                break
        if chosen is None:
            canonical_ids.append(eid)
        else:
            alias_map[eid] = chosen
            print(f"  alias: {eid} -> {chosen}  (cosine ~ {args.threshold:.3f}+)",
                  file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in sorted(alias_map.items())}, f, indent=2)
    print(
        f"wrote {len(alias_map)} alias entries to {args.out} "
        f"({len(canonical_ids)} canonical / {len(vecs)} total experts).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
