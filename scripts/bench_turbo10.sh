#!/usr/bin/env bash
# --------------------------------------------------------------------
# scripts/bench_turbo10.sh — Turbo-10 compute / I/O latency guardrail.
#
# Runs the engine's `run` (CLI bench) sub-command end-to-end against a
# small synthetic dataset, parses the `compute: p50=...us` and
# `i/o latency: p50=...us` lines from the structured run summary, and
# reports the **compute / io ratio**.
#
# The gist's Task 4 deliverable requires this to be **less than 10%**
# once the math kernels are dispatched through the AVX-512 fused path
# and the io_uring batched-submit path is live. The script exits 0 on
# PASS, 1 on FAIL — drop it into CI to keep the guardrail enforced.
#
# Usage:
#   scripts/bench_turbo10.sh                 # default: 200 tokens, 8 experts
#   TOKENS=1000 NUM_EXPERTS=16 scripts/bench_turbo10.sh
#
# Environment overrides:
#   DATA_DIR     directory holding `expert_<id>.bin` (default: ./data)
#   NUM_EXPERTS  number of experts to generate / serve (default: 8)
#   EXPERT_SIZE  bytes per expert on disk             (default: 16 MiB)
#   D_MODEL      hidden dim                           (default: 512)
#   D_FF         intermediate FFN dim                 (default: 2048)
#   TOP_K        experts activated per token          (default: 2)
#   TOKENS       number of tokens to drive            (default: 200)
#   CACHE_SLOTS  resident LRU capacity                (default: 4)
#   THRESHOLD    PASS threshold (percent)             (default: 10)
#   FEATURES     extra cargo features                 (default: avx512)
# --------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
EXPERT_SIZE="${EXPERT_SIZE:-16777216}"
D_MODEL="${D_MODEL:-512}"
D_FF="${D_FF:-2048}"
TOP_K="${TOP_K:-2}"
TOKENS="${TOKENS:-200}"
CACHE_SLOTS="${CACHE_SLOTS:-4}"
THRESHOLD="${THRESHOLD:-10}"
FEATURES="${FEATURES:-avx512}"

# On filesystems that don't support O_DIRECT (tmpfs, overlay, macOS),
# the engine accepts `--no-direct`. Probe portably: GNU `stat -f -c`
# and BSD/macOS `stat -f` use different flags, so try the GNU form
# first and fall back to the BSD form. On any failure (no `stat`, an
# exotic filesystem, …) we leave NO_DIRECT empty and let the engine
# emit its own warning if needed.
NO_DIRECT=""
FS_TYPE="$( (stat -f -c '%T' "$(dirname "$DATA_DIR")" 2>/dev/null) \
         || (stat -f '%T' "$(dirname "$DATA_DIR")" 2>/dev/null) \
         || echo unknown)"
case "$FS_TYPE" in
  tmpfs|overlayfs|overlay|hfs|apfs) NO_DIRECT="--no-direct" ;;
esac

mkdir -p "$DATA_DIR"

echo ">> Building micro-expert-router with features: ${FEATURES}"
( cd "$ROOT/rust-engine" && cargo build --release --features "$FEATURES" >/dev/null )

if [ ! -f "$DATA_DIR/expert_$((NUM_EXPERTS - 1)).bin" ]; then
  echo ">> Generating ${NUM_EXPERTS} synthetic experts in ${DATA_DIR}"
  "$ROOT/rust-engine/target/release/micro-expert-router" gen-data \
    --data-dir "$DATA_DIR" \
    --num-experts "$NUM_EXPERTS" \
    --expert-size "$EXPERT_SIZE" \
    --d-model "$D_MODEL" \
    --d-ff "$D_FF" >/dev/null
fi

echo ">> Running bench: tokens=${TOKENS} top_k=${TOP_K} cache_slots=${CACHE_SLOTS}"
RUN_OUT="$(mktemp)"
trap 'rm -f "$RUN_OUT"' EXIT

"$ROOT/rust-engine/target/release/micro-expert-router" run \
  --data-dir "$DATA_DIR" \
  --num-experts "$NUM_EXPERTS" \
  --expert-size "$EXPERT_SIZE" \
  --d-model "$D_MODEL" \
  --d-ff "$D_FF" \
  --top-k "$TOP_K" \
  --tokens "$TOKENS" \
  --cache-slots "$CACHE_SLOTS" \
  $NO_DIRECT 2>&1 | tee "$RUN_OUT"

# `tracing` writes the run summary to stderr (captured above via 2>&1).
IO_P50="$(grep -Eo 'i/o latency: *p50=[0-9]+us' "$RUN_OUT" \
            | grep -Eo '[0-9]+' | head -n 1)"
COMPUTE_P50="$(grep -Eo 'compute: *p50=[0-9]+us' "$RUN_OUT" \
                | grep -Eo '[0-9]+' | head -n 1)"

if [ -z "${IO_P50:-}" ] || [ -z "${COMPUTE_P50:-}" ]; then
  echo "!! Could not parse latency from run output." >&2
  echo "   Expected lines starting with 'i/o latency: p50=…us' and 'compute: p50=…us'." >&2
  exit 2
fi

if [ "$IO_P50" -eq 0 ]; then
  echo "!! io_latency_us p50 is 0 — every expert hit the cache. " >&2
  echo "   Re-run with --cache-slots smaller than --top-k (e.g. CACHE_SLOTS=1)." >&2
  exit 2
fi

# Integer percent: 100 * compute / io.
RATIO_PCT=$(( 100 * COMPUTE_P50 / IO_P50 ))

echo ""
echo "================ Turbo-10 latency guardrail ================"
printf "  io_latency_us       (p50) = %u us\n"  "$IO_P50"
printf "  compute_latency_us  (p50) = %u us\n"  "$COMPUTE_P50"
printf "  compute / io              = %u %%   (threshold: <%u %%)\n" \
  "$RATIO_PCT" "$THRESHOLD"
echo "============================================================"

if [ "$RATIO_PCT" -lt "$THRESHOLD" ]; then
  echo "PASS: compute is less than ${THRESHOLD}% of I/O — guardrail met."
  exit 0
else
  echo "FAIL: compute is ${RATIO_PCT}% of I/O — exceeds ${THRESHOLD}% guardrail."
  echo "      Investigate kernel dispatch overhead or warm the cache further."
  exit 1
fi
