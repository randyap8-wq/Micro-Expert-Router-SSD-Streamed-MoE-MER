#!/usr/bin/env bash
# --------------------------------------------------------------------
# scripts/quickstart.sh — one-shot SMB bring-up.
#
# Generates a small synthetic expert dataset, then starts the engine in
# server mode. Intended for kicking the tyres on a fresh laptop / VM
# without a real Mixtral checkpoint. Replace the `gen-data` step with
# `extract_mixtral_experts.py` once you have actual weights to serve.
# --------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/config.toml}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
EXPERT_SIZE="${EXPERT_SIZE:-16777216}" # 16 MiB
D_MODEL="${D_MODEL:-512}"
D_FF="${D_FF:-2048}"

mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/expert_0.bin" ]; then
  echo ">> Generating ${NUM_EXPERTS} synthetic experts in ${DATA_DIR}"
  ( cd "$ROOT/rust-engine" && cargo run --release -- gen-data \
      --data-dir "$DATA_DIR" \
      --num-experts "$NUM_EXPERTS" \
      --expert-size "$EXPERT_SIZE" \
      --d-model "$D_MODEL" \
      --d-ff "$D_FF" )
else
  echo ">> Reusing existing data in ${DATA_DIR}"
fi

echo ">> Starting the server (Ctrl-C to stop)"
exec "$ROOT/rust-engine/target/release/micro-expert-router" \
  serve --config "$CONFIG_FILE"
