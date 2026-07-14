#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FINALIZER="$ROOT/scripts/finalize_qwen3_coder_prompt2_resident.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-resident-finalization.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

write_raw_case() {
  local path=$1
  jq -n '
  def zero_phase3:
    {
      routed_layers_observed: 48,
      misses_at_initial_layer_lookup: 0,
      missing_experts_per_routed_layer: [48,0,0,0,0,0,0,0,0],
      layers_with_multiple_simultaneous_misses: 0,
      layers_with_one_physical_read: 0,
      layers_with_no_foreground_physical_read: 0,
      layers_with_serial_physical_reads: 0,
      layers_with_overlapping_physical_reads: 0,
      layers_beginning_compute_before_all_misses_available: 0,
      foreground_physical_read_operations: 0,
      peak_foreground_physical_reads_in_flight: 0,
      foreground_physical_read_concurrency_integral_seconds: 0,
      foreground_physical_read_active_seconds: 0,
      average_foreground_physical_read_concurrency: 0,
      physical_read_issue_to_completion_seconds: 0,
      physical_read_issue_to_completion_mean_seconds: 0,
      physical_read_issue_to_completion_max_seconds: 0,
      physical_read_issue_to_completion_histogram: [range(0;16) | {count:0}],
      primary_buffer_acquisition_wait_seconds: 0,
      primary_buffer_acquisition_wait_mean_seconds: 0,
      primary_buffer_acquisition_wait_max_seconds: 0,
      foreground_admission_wait_seconds: 0,
      foreground_admission_wait_mean_seconds: 0,
      singleflight_wait_seconds: 0,
      completion_to_consumption_delay_seconds: 0,
      completion_to_consumption_delay_mean_seconds: 0,
      completion_to_consumption_delay_max_seconds: 0,
      first_miss_to_first_read_issue_seconds: 0,
      first_miss_to_last_read_issue_seconds: 0,
      first_to_last_read_issue_spread_seconds: 0,
      first_miss_to_first_required_expert_available_seconds: 0,
      first_miss_to_final_required_expert_available_seconds: 0,
      first_to_last_required_expert_completion_spread_seconds: 0,
      first_miss_to_expert_compute_begin_seconds: 0,
      first_miss_to_layer_completion_seconds: 0,
      layer_expert_fetch_critical_path_seconds: 0,
      layer_expert_fetch_critical_path_mean_seconds: 0,
      layer_expert_fetch_critical_path_max_seconds: 0,
      demand_reads_issued_while_speculative_reads_active: 0,
      demand_critical_reads_delayed_by_speculative_activity: null,
      final_straggler_routed_slot_histogram: [0,0,0,0,0,0,0,0,0],
      worst_layer_fetch: null
    };
  {
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
      }, demand_miss_fanout: {
        prompt: zero_phase3,
        decode: zero_phase3
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

TELEMETRY_FAIL_DIR="$TEST_ROOT/telemetry-fail"
write_complete_cases "$TELEMETRY_FAIL_DIR" 3.4921515988 3.5081547663
jq '(.runs[0].demand_miss_fanout.decode.foreground_physical_read_operations) = 1' \
  "$TELEMETRY_FAIL_DIR/baseline-6144-short.json" \
  > "$TELEMETRY_FAIL_DIR/baseline-6144-short.json.tmp"
mv "$TELEMETRY_FAIL_DIR/baseline-6144-short.json.tmp" \
  "$TELEMETRY_FAIL_DIR/baseline-6144-short.json"
if bash "$FINALIZER" "$TELEMETRY_FAIL_DIR" test-commit; then
  echo "expected resident Phase 3A telemetry activation to fail qualification" >&2
  exit 1
else
  status=$?
fi
test "$status" -eq 1
jq -e '
  .qualification_passed == false and
  (.failure_reasons | index("resident cases activated Phase 3A miss-only telemetry")) != null
' "$TELEMETRY_FAIL_DIR/qualification.json" >/dev/null
jq -e '.resident_gates.zero_phase3_miss_telemetry == false' \
  "$TELEMETRY_FAIL_DIR/resident-control-summary.json" >/dev/null

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
