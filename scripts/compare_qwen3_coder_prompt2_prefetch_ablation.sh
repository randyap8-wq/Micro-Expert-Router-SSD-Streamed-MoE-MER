#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if (( $# != 3 )); then
  echo "usage: $0 CONTROL_A1_DIR NO_PREFETCH_B_DIR CONTROL_A2_DIR" >&2
  exit 2
fi

command -v jq >/dev/null
command -v realpath >/dev/null

CONTROL_A1_DIR=$(realpath "$1")
NO_PREFETCH_B_DIR=$(realpath "$2")
CONTROL_A2_DIR=$(realpath "$3")

find_one() {
  local root=$1
  local name=$2
  local matches=()
  while IFS= read -r path; do
    matches+=("$path")
  done < <(find "$root" -maxdepth 3 -type f -name "$name" -print | sort)
  if (( ${#matches[@]} != 1 )); then
    echo "expected exactly one $name below $root; found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

A1_SUMMARY=$(find_one "$CONTROL_A1_DIR" four-case-summary.json)
B_SUMMARY=$(find_one "$NO_PREFETCH_B_DIR" four-case-summary.json)
A2_SUMMARY=$(find_one "$CONTROL_A2_DIR" four-case-summary.json)
A1_PROVENANCE=$(find_one "$CONTROL_A1_DIR" ablation-provenance.json)
B_PROVENANCE=$(find_one "$NO_PREFETCH_B_DIR" ablation-provenance.json)
A2_PROVENANCE=$(find_one "$CONTROL_A2_DIR" ablation-provenance.json)

for artifact in \
  "$A1_SUMMARY" "$B_SUMMARY" "$A2_SUMMARY" \
  "$A1_PROVENANCE" "$B_PROVENANCE" "$A2_PROVENANCE"; do
  jq empty "$artifact"
done

jq -n \
  --arg control_a1_dir "$CONTROL_A1_DIR" \
  --arg no_prefetch_b_dir "$NO_PREFETCH_B_DIR" \
  --arg control_a2_dir "$CONTROL_A2_DIR" \
  --slurpfile control_a1_summary "$A1_SUMMARY" \
  --slurpfile no_prefetch_b_summary "$B_SUMMARY" \
  --slurpfile control_a2_summary "$A2_SUMMARY" \
  --slurpfile control_a1_provenance "$A1_PROVENANCE" \
  --slurpfile no_prefetch_b_provenance "$B_PROVENANCE" \
  --slurpfile control_a2_provenance "$A2_PROVENANCE" \
  -f "$ROOT/scripts/qwen3_coder_prompt2_prefetch_ablation_compare.jq"
