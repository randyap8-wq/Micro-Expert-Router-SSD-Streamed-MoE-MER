#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FINALIZER="$ROOT/scripts/finalize_qwen3_coder_prompt2_resident.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-resident-finalization.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

write_raw_case() {
  local path=$1
  jq -n '{
    aggregate: {
      output_token_parity: true,
      cache_misses_total: 0,
      ssd_bytes_total: 0
    },
    runs: [
      {cache_io: {
        cache_misses: 0,
        foreground_read_operations: 0,
        foreground_expert_bytes: 0,
        foreground_expert_io_wait_seconds: 0,
        total_expert_bytes_read: 0,
        prefetch_submitted: 0,
        prefetch_completed: 0,
        prefetch_bytes: 0
      }}
    ]
  }' > "$path"
}

write_case_summary() {
  local path=$1
  local decode_tps=$2
  jq -n --argjson decode_tps "$decode_tps" '{
    schema: {name:"mer-prompt2-case-summary", version:1},
    decode_tps_mean: $decode_tps,
    external_peak_rss_bytes: 1024,
    output_token_parity: true,
    qualification_passed: true
  }' > "$path"
}

write_complete_cases() {
  local dir=$1
  local short_tps=$2
  local medium_tps=$3
  mkdir -p "$dir"
  write_raw_case "$dir/baseline-6144-short.json"
  write_raw_case "$dir/baseline-6144-medium.json"
  write_case_summary "$dir/baseline-6144-short.case-summary.json" "$short_tps"
  write_case_summary "$dir/baseline-6144-medium.case-summary.json" "$medium_tps"
}

hash_artifacts() {
  local dir=$1
  find "$dir" -type f ! -name artifact-sha256.txt -print0 |
    sort -z |
    xargs -0 sha256sum > "$dir/artifact-sha256.txt"
}

PASS_DIR="$TEST_ROOT/pass"
write_complete_cases "$PASS_DIR" 3.4921515988 3.5081547663
bash "$FINALIZER" "$PASS_DIR" test-commit
jq -e '
  .qualification_passed == true and
  .failure_reasons == [] and
  .resident_reference_tolerance_passed == true
' "$PASS_DIR/qualification.json" >/dev/null

FAIL_DIR="$TEST_ROOT/fail"
write_complete_cases "$FAIL_DIR" 3.2589499509 3.0904253666
if bash "$FINALIZER" "$FAIL_DIR" test-commit; then
  echo "expected the complete performance-gate failure to return nonzero" >&2
  exit 1
else
  status=$?
fi
test "$status" -eq 1
test -s "$FAIL_DIR/resident-control-summary.json"
test -s "$FAIL_DIR/qualification.json"
jq -e '
  .qualification_passed == false and
  .resident_reference_tolerance_passed == false and
  (.failure_reasons | any(contains("resident performance gate failed")))
' "$FAIL_DIR/qualification.json" >/dev/null
hash_artifacts "$FAIL_DIR"
(cd "$FAIL_DIR" && sha256sum -c artifact-sha256.txt >/dev/null)

INCOMPLETE_DIR="$TEST_ROOT/incomplete"
mkdir -p "$INCOMPLETE_DIR"
write_raw_case "$INCOMPLETE_DIR/baseline-6144-short.json"
write_case_summary "$INCOMPLETE_DIR/baseline-6144-short.case-summary.json" 3.5
if bash "$FINALIZER" "$INCOMPLETE_DIR" test-commit; then
  echo "expected an interrupted resident run to be incomplete" >&2
  exit 1
else
  status=$?
fi
test "$status" -eq 2
test ! -e "$INCOMPLETE_DIR/resident-control-summary.json"
test ! -e "$INCOMPLETE_DIR/qualification.json"
test ! -e "$INCOMPLETE_DIR/artifact-sha256.txt"

echo "resident collector finalization tests: PASS"
