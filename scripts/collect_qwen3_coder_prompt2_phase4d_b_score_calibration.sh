#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
FILTER="$ROOT/scripts/qwen3_coder_prompt2_phase4d_b_score_calibration.jq"
DIAGNOSTIC_DIR=${1:-}

if [[ -z "$DIAGNOSTIC_DIR" ]]; then
  echo "usage: $0 DIAGNOSTIC_DIR" >&2
  exit 2
fi
if [[ -n "${MER_PROMPT2_PHASE4B_TRACE_PATH:-}" ]]; then
  echo "Phase 4D-B score calibration must be untraced; unset MER_PROMPT2_PHASE4B_TRACE_PATH" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --short)" ]]; then
  echo "Phase 4D-B score calibration requires a clean worktree" >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi
RUNNER_GIT_COMMIT_FULL=$(git -C "$ROOT" rev-parse HEAD)

mkdir -p "$DIAGNOSTIC_DIR"
DIAGNOSTIC_DIR=$(cd "$DIAGNOSTIC_DIR" && pwd)
variant_specs=(
  "demand-only|0.05|0.02|1.0"
  "second-only-f1|0.05|0.02|1.0"
  "second-only-f1-governed-current|0.05|0.02|1.0"
  "second-only-f1-governed-bt005-cw000|0.05|0.005|0.0"
)

first=true
for spec in "${variant_specs[@]}"; do
  IFS='|' read -r variant precision_floor base_threshold contention_weight <<<"$spec"
  skip_build=1
  if [[ "$first" == true ]]; then
    skip_build=0
    first=false
  fi
  MER_ALLOW_DIRTY=0 \
  MER_SKIP_BUILD="$skip_build" \
  MER_PROMPT2_PREFETCH_VARIANT="$variant" \
  MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR="$precision_floor" \
  MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD="$base_threshold" \
  MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT="$contention_weight" \
    bash "$COLLECTOR" "$DIAGNOSTIC_DIR/$variant" \
      phase4d-b-score-calibration
done

jq -n \
  --arg runner_git_commit_full "$RUNNER_GIT_COMMIT_FULL" \
  --slurpfile demand "$DIAGNOSTIC_DIR/demand-only/phase4d-b-score-calibration-variant-summary.json" \
  --slurpfile second "$DIAGNOSTIC_DIR/second-only-f1/phase4d-b-score-calibration-variant-summary.json" \
  --slurpfile current "$DIAGNOSTIC_DIR/second-only-f1-governed-current/phase4d-b-score-calibration-variant-summary.json" \
  --slurpfile low "$DIAGNOSTIC_DIR/second-only-f1-governed-bt005-cw000/phase4d-b-score-calibration-variant-summary.json" \
  -f "$FILTER" \
  > "$DIAGNOSTIC_DIR/phase4d-b-governor-score-calibration-summary.json"

jq -e '
  .qualification_kind == "phase4d-b-score-calibration-diagnostic" and
  .performance_qualification_applicable == false and
  .qualification_passed == false and
  .diagnostic_gates_passed == true
' "$DIAGNOSTIC_DIR/phase4d-b-governor-score-calibration-summary.json" >/dev/null

echo "Phase 4D-B governor score calibration complete: $DIAGNOSTIC_DIR"
