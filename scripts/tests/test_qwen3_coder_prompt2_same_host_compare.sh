#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMPARE="$ROOT/scripts/compare_qwen3_coder_prompt2_same_host.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-same-host.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

write_case() {
  local tps=$1
  local hit_rate=$2
  local misses=$3
  local bytes=$4
  local service=$5
  local critical=$6
  local fraction=$7
  local deferred=$8
  jq -n \
    --argjson tps "$tps" \
    --argjson hit_rate "$hit_rate" \
    --argjson misses "$misses" \
    --argjson bytes "$bytes" \
    --argjson service "$service" \
    --argjson critical "$critical" \
    --argjson fraction "$fraction" \
    --argjson deferred "$deferred" '
    {
      decode_tps_mean: $tps,
      cache_hit_rate: $hit_rate,
      cache_misses_total: $misses,
      ssd_bytes_total: $bytes,
      output_token_parity: true,
      qualification_passed: true,
      phase3a_decode: {
        physical_read_issue_to_completion_mean_seconds: $service,
        physical_read_issue_to_completion_max_seconds: (2 * $service),
        physical_read_issue_to_completion_histogram: ([range(0;15) | {upper_bound_microseconds: 1000, count: 0}] + [{upper_bound_microseconds: 2000, count: 10}]),
        layer_expert_fetch_critical_path_mean_seconds: $critical,
        layer_expert_fetch_critical_path_max_seconds: (2 * $critical),
        decode_wall_fraction_attributable_to_layer_expert_fetch: $fraction
      },
      phase3b_decode: {
        speculative_physical_reads_deferred_for_demand_pressure: $deferred,
        deferred_speculative_physical_reads_resumed: $deferred,
        deferred_speculative_physical_reads_dropped_stale_duplicate_or_cache_hit: 0,
        speculative_physical_reads_admitted_without_demand_pressure: 4,
        speculative_physical_reads_active_when_demand_burst_began: 1,
        demand_layers_final_straggler_issued_while_speculative_reads_active: 1,
        demand_physical_read_service_without_speculation_operations: 8,
        demand_physical_read_service_without_speculation_mean_seconds: $service,
        demand_physical_read_service_without_speculation_max_seconds: (2 * $service),
        demand_physical_read_service_with_speculation_operations: 2,
        demand_physical_read_service_with_speculation_mean_seconds: (1.5 * $service),
        demand_physical_read_service_with_speculation_max_seconds: (3 * $service)
      }
    }'
}

write_phase() {
  local dir=$1
  local stream_scale=$2
  local resident_scale=$3
  local deferred=$4
  mkdir -p "$dir/stream" "$dir/resident"
  printf 'Linux qualified-g2 6.0.0 x86_64 GNU/Linux\n' > "$dir/stream/uname.txt"
  printf 'Linux qualified-g2 6.0.0 x86_64 GNU/Linux\n' > "$dir/resident/uname.txt"

  local short
  local medium
  short=$(write_case "$((100 * stream_scale))" 0.75 100 1000 0.010 0.020 0.30 "$deferred")
  medium=$(write_case "$((64 * stream_scale))" 0.80 80 800 0.012 0.025 0.35 "$deferred")
  jq -n \
    --argjson short "$short" \
    --argjson medium "$medium" '
    {
      cases: [
        ($short + {cache_slots:1536, prompt_fixture:"short"}),
        ($medium + {cache_slots:1536, prompt_fixture:"medium"})
      ],
      qualification_passed: true
    }' > "$dir/stream/four-case-summary.json"

  jq -n \
    --argjson short_tps "$((90 * resident_scale))" \
    --argjson medium_tps "$((81 * resident_scale))" '
    {
      cases: [
        {cache_slots:6144, prompt_fixture:"short", decode_tps_mean:$short_tps},
        {cache_slots:6144, prompt_fixture:"medium", decode_tps_mean:$medium_tps}
      ],
      resident_gates: {output_token_parity_passed:true},
      performance_gate: {passed:false},
      qualification_passed: false
    }' > "$dir/resident/resident-control-summary.json"
}

write_phase "$TEST_ROOT/phase3a" 1 1 0
write_phase "$TEST_ROOT/phase3b" 2 1 3

SOURCE_CKSUM_BEFORE=$(find "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" -type f -exec cksum {} \; | sort | cksum)
bash "$COMPARE" "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" > "$TEST_ROOT/comparison.json"
SOURCE_CKSUM_AFTER=$(find "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" -type f -exec cksum {} \; | sort | cksum)
test "$SOURCE_CKSUM_BEFORE" = "$SOURCE_CKSUM_AFTER"
jq -e '
  .same_host.hostname_matches == true and
  .streaming_1536.short.decode_tps.percent_delta == 100 and
  .streaming_1536.medium.decode_tps.percent_delta == 100 and
  .streaming_1536.geometric_mean_decode_tps.percent_delta == 100 and
  .resident_6144.geometric_mean_decode_tps.percent_delta == 0 and
  .resident_6144.contemporaneous_within_two_percent == true and
  .speculative_arbitration_decode_1536.deferred == 6 and
  .sources.source_artifacts_modified == false and
  .correctness_and_qualification.phase3b_streaming_output_parity == true and
  .correctness_and_qualification.phase3b_resident_qualification_passed == false
' "$TEST_ROOT/comparison.json" >/dev/null

echo "same-host comparison helper tests passed"
