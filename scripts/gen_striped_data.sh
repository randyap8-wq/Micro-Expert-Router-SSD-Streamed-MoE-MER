#!/usr/bin/env bash
# Generate striped synthetic expert data across N directories.
#
# Usage:
#   scripts/gen_striped_data.sh DIR1,DIR2[,DIR3...] NUM_EXPERTS EXPERT_SIZE [extra-gen-data-flags]
#
# Generates all experts into the *first* directory using the engine's
# `gen-data` subcommand, then moves expert files into the other
# directories using the `id % n_drives` shard map the engine uses at
# read time (`NvmeStorage::striped`). After this script completes, you
# can run the engine with `--data-dir DIR1,DIR2[,DIR3...]` and reads
# will be distributed across the listed mountpoints.
#
# Gist Phase 4 — multi-drive striping.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 DIR1,DIR2[,DIR3...] NUM_EXPERTS EXPERT_SIZE [extra-gen-data-flags]" >&2
    exit 1
fi

DIRS_CSV="$1"
NUM_EXPERTS="$2"
EXPERT_SIZE="$3"
shift 3

# Split on comma.
IFS=',' read -r -a DIRS <<< "$DIRS_CSV"
N=${#DIRS[@]}
if [[ $N -lt 1 ]]; then
    echo "error: at least one directory required" >&2
    exit 1
fi

for d in "${DIRS[@]}"; do
    mkdir -p "$d"
done

PRIMARY="${DIRS[0]}"

# Locate the engine binary. Prefer release, fall back to debug. Use
# `git rev-parse` when available so the script works when invoked from
# anywhere; fall back to the script's parent directory otherwise.
if ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
ENGINE_BIN=""
for cand in "$ROOT/rust-engine/target/release/micro-expert-router" \
            "$ROOT/rust-engine/target/debug/micro-expert-router"; do
    if [[ -x "$cand" ]]; then
        ENGINE_BIN="$cand"
        break
    fi
done
if [[ -z "$ENGINE_BIN" ]]; then
    echo "error: micro-expert-router binary not found. Build with 'cargo build --release' first." >&2
    exit 1
fi

echo "[1/2] generating $NUM_EXPERTS experts into primary dir $PRIMARY"
"$ENGINE_BIN" gen-data \
    --data-dir "$PRIMARY" \
    --num-experts "$NUM_EXPERTS" \
    --expert-size "$EXPERT_SIZE" \
    "$@"

if [[ $N -eq 1 ]]; then
    echo "[2/2] single-drive layout — nothing to redistribute."
    exit 0
fi

echo "[2/2] redistributing experts across $N drives by id % $N"
for ((id = 0; id < NUM_EXPERTS; id++)); do
    target_idx=$((id % N))
    if [[ $target_idx -eq 0 ]]; then
        continue
    fi
    target_dir="${DIRS[$target_idx]}"
    src="$PRIMARY/expert_${id}.bin"
    dst="$target_dir/expert_${id}.bin"
    if [[ -f "$src" ]]; then
        mv "$src" "$dst"
    fi
done
# Copy metadata.json to every shard (engine only reads from the primary,
# but downstream tooling may scan each mountpoint).
if [[ -f "$PRIMARY/metadata.json" ]]; then
    for ((i = 1; i < N; i++)); do
        cp "$PRIMARY/metadata.json" "${DIRS[$i]}/metadata.json"
    done
fi

echo "done. Run the engine with '--data-dir $DIRS_CSV'."
