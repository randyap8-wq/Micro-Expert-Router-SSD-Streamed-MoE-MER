#!/usr/bin/env python3
"""Compute an expert-routing transition matrix and per-K cache-hit
table from a JSONL routing trace produced by the engine.

The engine writes one record per token to the trace path passed via
`run --trace-out PATH`:

    {"token": 42, "layer": 0, "experts": [3, 7], "cache_hit": [false, true], "predicted": [3, 9]}

The `predicted` field is always present in the trace; it is empty when no speculator is
wired (i.e. when `run --speculator` is not set). This script ignores it,
but it lets other tooling diff Predicted vs. Actual experts per layer.

This script:

1. Aggregates `(prev_top1_expert -> next_top1_expert)` counts per layer
   and per-layer marginals.
2. Normalises rows into a transition probability matrix `P` where
   `P[i][j] = Pr(next top-1 = j | prev top-1 = i)`. Used by the engine's
   `PredictiveLoader` to seed its 1st-order Markov priors.
3. For K in {2, 4, 8, 16}, simulates a tiny LRU of size K replaying the
   trace's expert ids one token at a time, and prints the hit rate. This
   is the same simulation the `validate-predictor` Rust subcommand
   runs — having both lets you cross-check the predictor's design from
   either side of the FFI boundary.

Outputs:

* `transition_matrix.json` next to the input trace, with one entry per
  layer: `{ "layer": L, "num_experts": E, "matrix": [[...], ...] }`.
* `cache_hit_table.csv`: `K,hit_rate,hits,total`.

Gist Phase 6 — routing trace + predictor evaluation.

Usage:
    scripts/compute_transition_matrix.py /path/to/trace.jsonl
    scripts/compute_transition_matrix.py /path/to/trace.jsonl --cache-slots 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path, help="Path to a JSONL routing trace")
    p.add_argument(
        "--cache-slots",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Cache sizes to sweep for the hit-rate simulation (default: 2 4 8 16)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to the trace file's parent)",
    )
    return p.parse_args()


def load_trace(path: Path):
    """Yield (token, layer, experts) tuples from a JSONL trace."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield int(rec.get("token", 0)), int(rec.get("layer", 0)), [int(x) for x in rec.get("experts", [])]


def build_transition_matrices(trace_path: Path):
    """Return dict[layer] -> (num_experts, matrix) where matrix is a
    list of lists of floats forming a row-stochastic transition matrix."""
    counts: dict[int, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    max_id_per_layer: dict[int, int] = defaultdict(int)
    prev_per_layer: dict[int, int | None] = defaultdict(lambda: None)

    for _t, layer, experts in load_trace(trace_path):
        if not experts:
            continue
        cur = experts[0]
        max_id_per_layer[layer] = max(max_id_per_layer[layer], max(experts))
        prev = prev_per_layer[layer]
        if prev is not None:
            counts[layer][(prev, cur)] += 1
        prev_per_layer[layer] = cur

    out: dict[int, tuple[int, list[list[float]]]] = {}
    for layer, c in counts.items():
        n = max_id_per_layer[layer] + 1
        mat = [[0.0] * n for _ in range(n)]
        # Row counts
        row_total = [0] * n
        for (i, j), v in c.items():
            mat[i][j] += float(v)
            row_total[i] += v
        # Row-normalise (rows with no observations stay all-zero —
        # the engine treats those as a uniform fallback at runtime).
        for i in range(n):
            if row_total[i] > 0:
                inv = 1.0 / row_total[i]
                for j in range(n):
                    mat[i][j] *= inv
        out[layer] = (n, mat)
    return out


def simulate_lru(trace_path: Path, k: int) -> tuple[int, int]:
    """Replay the trace through a single shared LRU of size `k` and
    return (hits, total). This matches the simulation the Rust
    `validate-predictor` subcommand performs."""
    lru: deque[int] = deque()
    in_set: set[int] = set()
    hits = 0
    total = 0
    for _t, _layer, experts in load_trace(trace_path):
        for e in experts:
            if e in in_set:
                hits += 1
                # Move-to-back to keep LRU semantics. Guaranteed-present
                # because `in_set` and `lru` are kept in sync below.
                lru.remove(e)
            else:
                if len(lru) == k:
                    old = lru.popleft()
                    in_set.discard(old)
            lru.append(e)
            in_set.add(e)
            total += 1
    return hits, total


def main() -> int:
    args = parse_args()
    if not args.trace.is_file():
        print(f"error: trace file {args.trace} not found", file=sys.stderr)
        return 1
    out_dir = args.out_dir or args.trace.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading trace: {args.trace}")
    matrices = build_transition_matrices(args.trace)
    print(f"per-layer transition matrices computed for {len(matrices)} layer(s)")

    tm_path = out_dir / "transition_matrix.json"
    payload = [
        {"layer": layer, "num_experts": n, "matrix": mat}
        for layer, (n, mat) in sorted(matrices.items())
    ]
    with tm_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {tm_path}")

    cht_path = out_dir / "cache_hit_table.csv"
    with cht_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "hit_rate", "hits", "total"])
        for k in args.cache_slots:
            hits, total = simulate_lru(args.trace, k)
            rate = hits / total if total else 0.0
            w.writerow([k, f"{rate:.6f}", hits, total])
            print(f"  K={k:>3}  hit_rate={rate:.4f}  hits={hits}/{total}")
    print(f"wrote {cht_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
