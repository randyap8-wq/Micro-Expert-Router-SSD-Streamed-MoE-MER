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
  local resumed=$9
  jq -n \
    --argjson tps "$tps" \
    --argjson hit_rate "$hit_rate" \
    --argjson misses "$misses" \
    --argjson bytes "$bytes" \
    --argjson service "$service" \
    --argjson critical "$critical" \
    --argjson fraction "$fraction" \
    --argjson deferred "$deferred" \
    --argjson resumed "$resumed" '
    {
      speculative_physical_reads_deferred_for_demand_pressure: $deferred,
      deferred_speculative_physical_reads_resumed: $resumed,
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
    } as $phase3b |
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
      phase3b_prompt: $phase3b,
      phase3b_decode: $phase3b
    }'
}

write_phase() {
  local dir=$1
  local stream_scale=$2
  local resident_scale=$3
  local deferred=$4
  local resumed=$5
  mkdir -p "$dir/stream" "$dir/resident"
  printf 'Linux qualified-g2 6.0.0 x86_64 GNU/Linux\n' > "$dir/stream/uname.txt"
  printf 'Linux qualified-g2 6.0.0 x86_64 GNU/Linux\n' > "$dir/resident/uname.txt"

  local short
  local medium
  short=$(write_case "$((100 * stream_scale))" 0.75 100 1000 0.010 0.020 0.30 "$deferred" "$resumed")
  medium=$(write_case "$((64 * stream_scale))" 0.80 80 800 0.012 0.025 0.35 "$deferred" "$resumed")
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
    --argjson resident_scale "$resident_scale" '
    {
      cases: [
        {cache_slots:6144, prompt_fixture:"short", decode_tps_mean:(90 * $resident_scale)},
        {cache_slots:6144, prompt_fixture:"medium", decode_tps_mean:(81 * $resident_scale)}
      ],
      resident_gates: {output_token_parity_passed:true},
      performance_gate: {passed:false},
      qualification_passed: false
    }' > "$dir/resident/resident-control-summary.json"
}

write_phase "$TEST_ROOT/phase3a" 1 1 0 0
write_phase "$TEST_ROOT/phase3b" 2 0.99 3 2

SOURCE_CKSUM_BEFORE=$(find "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" -type f -exec cksum {} \; | sort | cksum)
bash "$COMPARE" "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" > "$TEST_ROOT/comparison.json"
SOURCE_CKSUM_AFTER=$(find "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" -type f -exec cksum {} \; | sort | cksum)
test "$SOURCE_CKSUM_BEFORE" = "$SOURCE_CKSUM_AFTER"
jq -e '
  .same_host.hostname_matches == true and
  .streaming_1536.short.decode_tps.percent_delta == 100 and
  .streaming_1536.medium.decode_tps.percent_delta == 100 and
  .streaming_1536.geometric_mean_decode_tps.percent_delta == 100 and
  .resident_6144.geometric_mean_decode_tps.percent_delta > -1.000001 and
  .resident_6144.geometric_mean_decode_tps.percent_delta < -0.999999 and
  .resident_6144.contemporaneous_within_two_percent == true and
  .speculative_arbitration_decode_1536.deferred == 6 and
  .speculative_arbitration_decode_1536.resumed == 4 and
  .speculative_arbitration_decode_1536.not_classified_by_phase3b_terminal_counters_at_snapshot == 2 and
  .speculative_arbitration_decode_1536.fully_classified_at_snapshot == false and
  .sources.source_artifacts_modified == false and
  .correctness_and_qualification.phase3b_streaming_output_parity == true and
  .correctness_and_qualification.phase3b_resident_qualification_passed == false and
  .acceptance.same_host_identity_passed == true and
  .acceptance.resident_contemporaneous.within_plus_or_minus_two_percent == true and
  .acceptance.throughput_regression_status.short.regressed == false and
  .acceptance.streaming_geometric_mean.preferred_threshold_reached == true and
  .acceptance.deferred_resumed_drop_accounting.all_deferred_units_classified_at_snapshot == false and
  .acceptance.policy_accepted == false and
  (.acceptance.rejection_reasons | index("deferred speculative operations were not fully classified by the Phase 3B terminal counters at snapshot time")) != null
' "$TEST_ROOT/comparison.json" >/dev/null

update_stream_case() {
  local input=$1
  local output=$2
  local phase=$3
  jq --arg phase "$phase" '
    .cases |= map(
      if .prompt_fixture == "short" then
        .decode_tps_mean = (if $phase == "a" then 1.297117606206983 else 1.305594662395702 end) |
        .cache_misses_total = (if $phase == "a" then 100 else 101.6641 end) |
        .ssd_bytes_total = (if $phase == "a" then 1000 else 994.219 end) |
        .phase3a_decode.physical_read_issue_to_completion_mean_seconds =
          (if $phase == "a" then 0.010 else 0.01036406 end) |
        .phase3a_decode.physical_read_issue_to_completion_max_seconds =
          (if $phase == "a" then 0.020 else 0.021 end) |
        .phase3a_decode.layer_expert_fetch_critical_path_mean_seconds =
          (if $phase == "a" then 0.020 else 0.02106698 end) |
        .phase3a_decode.layer_expert_fetch_critical_path_max_seconds =
          (if $phase == "a" then 0.040 else 0.042 end) |
        .phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch =
          (if $phase == "a" then 0.30 else 0.3180945 end) |
        (if $phase == "b" then
          .phase3b_prompt.speculative_physical_reads_deferred_for_demand_pressure = 2501 |
          .phase3b_prompt.deferred_speculative_physical_reads_resumed = 2212 |
          .phase3b_decode.speculative_physical_reads_deferred_for_demand_pressure = 8180 |
          .phase3b_decode.deferred_speculative_physical_reads_resumed = 7525
        else . end)
      else
        .decode_tps_mean = (if $phase == "a" then 1.494644032414944 else 1.5249795388009992 end) |
        .cache_misses_total = (if $phase == "a" then 80 else 80.97392 end) |
        .ssd_bytes_total = (if $phase == "a" then 800 else 793.5616 end) |
        .phase3a_decode.physical_read_issue_to_completion_mean_seconds =
          (if $phase == "a" then 0.012 else 0.011966292 end) |
        .phase3a_decode.physical_read_issue_to_completion_max_seconds =
          (if $phase == "a" then 0.024 else 0.023 end) |
        .phase3a_decode.layer_expert_fetch_critical_path_mean_seconds =
          (if $phase == "a" then 0.025 else 0.026008 end) |
        .phase3a_decode.layer_expert_fetch_critical_path_max_seconds =
          (if $phase == "a" then 0.050 else 0.049 end) |
        .phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch =
          (if $phase == "a" then 0.35 else 0.3723461 end) |
        (if $phase == "b" then
          .phase3b_prompt.speculative_physical_reads_deferred_for_demand_pressure = 2501 |
          .phase3b_prompt.deferred_speculative_physical_reads_resumed = 2212 |
          .phase3b_decode.speculative_physical_reads_deferred_for_demand_pressure = 8180 |
          .phase3b_decode.deferred_speculative_physical_reads_resumed = 7526
        else . end)
      end
    )
  ' "$input" > "$output"
}

update_stream_case "$TEST_ROOT/phase3a/stream/four-case-summary.json" "$TEST_ROOT/phase3a/stream/four-case-summary.json.tmp" a
mv "$TEST_ROOT/phase3a/stream/four-case-summary.json.tmp" "$TEST_ROOT/phase3a/stream/four-case-summary.json"
update_stream_case "$TEST_ROOT/phase3b/stream/four-case-summary.json" "$TEST_ROOT/phase3b/stream/four-case-summary.json.tmp" b
mv "$TEST_ROOT/phase3b/stream/four-case-summary.json.tmp" "$TEST_ROOT/phase3b/stream/four-case-summary.json"

jq '.cases |= map(.decode_tps_mean = 3.2437048410808447)' "$TEST_ROOT/phase3a/resident/resident-control-summary.json" > "$TEST_ROOT/phase3a/resident/resident-control-summary.json.tmp"
mv "$TEST_ROOT/phase3a/resident/resident-control-summary.json.tmp" "$TEST_ROOT/phase3a/resident/resident-control-summary.json"
jq '.cases |= map(.decode_tps_mean = 3.532783148464943)' "$TEST_ROOT/phase3b/resident/resident-control-summary.json" > "$TEST_ROOT/phase3b/resident/resident-control-summary.json.tmp"
mv "$TEST_ROOT/phase3b/resident/resident-control-summary.json.tmp" "$TEST_ROOT/phase3b/resident/resident-control-summary.json"

CAPTURED_CKSUM_BEFORE=$(find "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" -type f -exec cksum {} \; | sort | cksum)
bash "$COMPARE" "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" > "$TEST_ROOT/captured-comparison.json"
CAPTURED_CKSUM_AFTER=$(find "$TEST_ROOT/phase3a" "$TEST_ROOT/phase3b" -type f -exec cksum {} \; | sort | cksum)
test "$CAPTURED_CKSUM_BEFORE" = "$CAPTURED_CKSUM_AFTER"
jq -e '
  .streaming_1536.geometric_mean_decode_tps.percent_delta > 1.3391 and
  .streaming_1536.geometric_mean_decode_tps.percent_delta < 1.3393 and
  .acceptance.streaming_geometric_mean.preferred_threshold_reached == false and
  .acceptance.demand_storage_changes.mean_improved_in_both_fixtures == false and
  .acceptance.layer_fetch_critical_path_changes.mean_improved_in_both_fixtures == false and
  .acceptance.decode_wall_fetch_fraction_changes.did_not_worsen == false and
  .acceptance.cache_miss_changes.did_not_increase == false and
  .acceptance.deferred_resumed_drop_accounting.prompt.not_classified_by_phase3b_terminal_counters_at_snapshot == 578 and
  .acceptance.deferred_resumed_drop_accounting.decode.not_classified_by_phase3b_terminal_counters_at_snapshot == 1309 and
  .acceptance.policy_accepted == false and
  (.acceptance.rejection_reasons | index("streaming geometric-mean improvement did not reach the preferred plus 2 percent threshold")) != null
' "$TEST_ROOT/captured-comparison.json" >/dev/null

echo "same-host comparison helper tests passed"
