#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PHASE3A_DIR=${1:-}
PHASE3B_DIR=${2:-}

if [[ -z "$PHASE3A_DIR" || -z "$PHASE3B_DIR" ]]; then
  echo "usage: $0 PHASE3A_ARTIFACT_DIR PHASE3B_ARTIFACT_DIR" >&2
  echo "each directory must contain one four-case-summary.json and one resident-control-summary.json" >&2
  exit 2
fi

command -v jq >/dev/null
command -v realpath >/dev/null

PHASE3A_DIR=$(realpath "$PHASE3A_DIR")
PHASE3B_DIR=$(realpath "$PHASE3B_DIR")

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

A_STREAM=$(find_one "$PHASE3A_DIR" four-case-summary.json)
A_RESIDENT=$(find_one "$PHASE3A_DIR" resident-control-summary.json)
B_STREAM=$(find_one "$PHASE3B_DIR" four-case-summary.json)
B_RESIDENT=$(find_one "$PHASE3B_DIR" resident-control-summary.json)

for artifact in "$A_STREAM" "$A_RESIDENT" "$B_STREAM" "$B_RESIDENT"; do
  jq empty "$artifact"
done

A_STREAM_UNAME=$(dirname "$A_STREAM")/uname.txt
A_RESIDENT_UNAME=$(dirname "$A_RESIDENT")/uname.txt
B_STREAM_UNAME=$(dirname "$B_STREAM")/uname.txt
B_RESIDENT_UNAME=$(dirname "$B_RESIDENT")/uname.txt
for artifact in "$A_STREAM_UNAME" "$A_RESIDENT_UNAME" "$B_STREAM_UNAME" "$B_RESIDENT_UNAME"; do
  test -s "$artifact"
done
A_STREAM_HOST=$(awk 'NR == 1 {print $2}' "$A_STREAM_UNAME")
A_RESIDENT_HOST=$(awk 'NR == 1 {print $2}' "$A_RESIDENT_UNAME")
B_STREAM_HOST=$(awk 'NR == 1 {print $2}' "$B_STREAM_UNAME")
B_RESIDENT_HOST=$(awk 'NR == 1 {print $2}' "$B_RESIDENT_UNAME")

jq -n \
  --arg phase3a_artifact_dir "$PHASE3A_DIR" \
  --arg phase3b_artifact_dir "$PHASE3B_DIR" \
  --arg phase3a_streaming_hostname "$A_STREAM_HOST" \
  --arg phase3a_resident_hostname "$A_RESIDENT_HOST" \
  --arg phase3b_streaming_hostname "$B_STREAM_HOST" \
  --arg phase3b_resident_hostname "$B_RESIDENT_HOST" \
  --slurpfile phase3a_stream "$A_STREAM" \
  --slurpfile phase3a_resident "$A_RESIDENT" \
  --slurpfile phase3b_stream "$B_STREAM" \
  --slurpfile phase3b_resident "$B_RESIDENT" \
  -f "$ROOT/scripts/qwen3_coder_prompt2_same_host_compare.jq"
