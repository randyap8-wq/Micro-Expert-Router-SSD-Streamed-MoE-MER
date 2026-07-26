#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
MATRIX_DIR=${1:-}

if [[ -z "$MATRIX_DIR" ]]; then
  echo "usage: $0 MATRIX_DIR" >&2
  exit 2
fi
if [[ -n "${MER_PROMPT2_PHASE4B_TRACE_PATH:-}" ]]; then
  echo "Phase 4C primary matrix must be untraced; unset MER_PROMPT2_PHASE4B_TRACE_PATH" >&2
  exit 2
fi

mkdir -p "$MATRIX_DIR"
MATRIX_DIR=$(cd "$MATRIX_DIR" && pwd)
variants=(
  demand-only
  current-f2
  current-f2-governed
  second-only-f2
  second-only-f1
  second-only-f1-governed
)

first=true
for variant in "${variants[@]}"; do
  if [[ "$first" == true ]]; then
    MER_PROMPT2_PREFETCH_VARIANT="$variant" \
      bash "$COLLECTOR" "$MATRIX_DIR/$variant" phase4c-untraced
    first=false
  else
    MER_SKIP_BUILD=1 \
    MER_PROMPT2_PREFETCH_VARIANT="$variant" \
      bash "$COLLECTOR" "$MATRIX_DIR/$variant" phase4c-untraced
  fi
done

jq -n \
  --slurpfile demand "$MATRIX_DIR/demand-only/phase4c-variant-summary.json" \
  --slurpfile current "$MATRIX_DIR/current-f2/phase4c-variant-summary.json" \
  --slurpfile current_governed "$MATRIX_DIR/current-f2-governed/phase4c-variant-summary.json" \
  --slurpfile second_f2 "$MATRIX_DIR/second-only-f2/phase4c-variant-summary.json" \
  --slurpfile second_f1 "$MATRIX_DIR/second-only-f1/phase4c-variant-summary.json" \
  --slurpfile second_f1_governed "$MATRIX_DIR/second-only-f1-governed/phase4c-variant-summary.json" \
  --slurpfile demand_short "$MATRIX_DIR/demand-only/baseline-1536-short.json" \
  --slurpfile demand_medium "$MATRIX_DIR/demand-only/baseline-1536-medium.json" \
  --slurpfile current_short "$MATRIX_DIR/current-f2/baseline-1536-short.json" \
  --slurpfile current_medium "$MATRIX_DIR/current-f2/baseline-1536-medium.json" \
  --slurpfile current_governed_short "$MATRIX_DIR/current-f2-governed/baseline-1536-short.json" \
  --slurpfile current_governed_medium "$MATRIX_DIR/current-f2-governed/baseline-1536-medium.json" \
  --slurpfile second_f2_short "$MATRIX_DIR/second-only-f2/baseline-1536-short.json" \
  --slurpfile second_f2_medium "$MATRIX_DIR/second-only-f2/baseline-1536-medium.json" \
  --slurpfile second_f1_short "$MATRIX_DIR/second-only-f1/baseline-1536-short.json" \
  --slurpfile second_f1_medium "$MATRIX_DIR/second-only-f1/baseline-1536-medium.json" \
  --slurpfile second_f1_governed_short "$MATRIX_DIR/second-only-f1-governed/baseline-1536-short.json" \
  --slurpfile second_f1_governed_medium "$MATRIX_DIR/second-only-f1-governed/baseline-1536-medium.json" \
  -f "$ROOT/scripts/qwen3_coder_prompt2_phase4c_matrix.jq" \
  > "$MATRIX_DIR/phase4c-matrix-summary.json"

jq -e '
  .gates.exact_six_variant_matrix and
  .gates.all_variant_qualifications_passed and
  .gates.neural_speculator_disabled_everywhere and
  .gates.governor_assignment_exact and
  .gates.governed_cases_share_configuration and
  .gates.current_governor_pair_differs_only_by_governor and
  .gates.second_f1_governor_pair_differs_only_by_governor and
  .gates.cross_variant_short_output_parity and
  .gates.cross_variant_medium_output_parity
' "$MATRIX_DIR/phase4c-matrix-summary.json" >/dev/null

echo "Phase 4C untraced matrix complete: $MATRIX_DIR"
