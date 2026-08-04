#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
FILTER="$ROOT/scripts/qwen3_coder_prompt2_phase4d_c_sparse_admission.jq"
SCREENING_DIR=${1:-}

if [[ -z "$SCREENING_DIR" ]]; then
  echo "usage: $0 SCREENING_DIR" >&2
  exit 2
fi
if [[ -n "${MER_PROMPT2_PHASE4B_TRACE_PATH:-}" ]]; then
  echo "Phase 4D-C sparse admission screening must be untraced; unset MER_PROMPT2_PHASE4B_TRACE_PATH" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --short)" ]]; then
  echo "Phase 4D-C sparse admission screening requires a clean worktree" >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi
RUNNER_GIT_COMMIT_FULL=$(git -C "$ROOT" rev-parse HEAD)

mkdir -p "$SCREENING_DIR"
SCREENING_DIR=$(cd "$SCREENING_DIR" && pwd)
variant_specs=(
  "demand-only|0.05|0.02|1.0"
  "second-only-f1|0.05|0.02|1.0"
  "second-only-f1-governed-bt001-cw000|0.05|0.001|0.0"
  "second-only-f1-governed-bt0005-cw000|0.05|0.0005|0.0"
  "second-only-f1-governed-bt00025-cw000|0.05|0.00025|0.0"
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
    bash "$COLLECTOR" "$SCREENING_DIR/$variant" phase4d-c-sparse-admission
done

jq -n \
  --arg runner_git_commit_full "$RUNNER_GIT_COMMIT_FULL" \
  --slurpfile demand "$SCREENING_DIR/demand-only/phase4d-c-sparse-admission-variant-summary.json" \
  --slurpfile second "$SCREENING_DIR/second-only-f1/phase4d-c-sparse-admission-variant-summary.json" \
  --slurpfile high "$SCREENING_DIR/second-only-f1-governed-bt001-cw000/phase4d-c-sparse-admission-variant-summary.json" \
  --slurpfile mid "$SCREENING_DIR/second-only-f1-governed-bt0005-cw000/phase4d-c-sparse-admission-variant-summary.json" \
  --slurpfile low "$SCREENING_DIR/second-only-f1-governed-bt00025-cw000/phase4d-c-sparse-admission-variant-summary.json" \
  -f "$FILTER" > "$SCREENING_DIR/phase4d-c-sparse-admission-summary.json"

jq -e '
  .qualification_kind == "phase4d-c-sparse-admission-screening" and
  .performance_qualification_applicable == false and
  .qualification_passed == false and
  .screening_gates_passed == true
' "$SCREENING_DIR/phase4d-c-sparse-admission-summary.json" >/dev/null

echo "Phase 4D-C sparse admission screening complete: $SCREENING_DIR"
