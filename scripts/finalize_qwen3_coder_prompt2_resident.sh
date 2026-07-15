#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR=${1:-}
EXPECTED_COMMIT=${2:-}
if [[ -z "$ARTIFACT_DIR" || -z "$EXPECTED_COMMIT" ]]; then
  echo "usage: $0 ARTIFACT_DIR EXPECTED_COMMIT" >&2
  exit 2
fi

command -v jq >/dev/null

SHORT_RAW="$ARTIFACT_DIR/baseline-6144-short.json"
MEDIUM_RAW="$ARTIFACT_DIR/baseline-6144-medium.json"
SHORT_CASE="$ARTIFACT_DIR/baseline-6144-short.case-summary.json"
MEDIUM_CASE="$ARTIFACT_DIR/baseline-6144-medium.case-summary.json"

for artifact in "$SHORT_RAW" "$MEDIUM_RAW" "$SHORT_CASE" "$MEDIUM_CASE"; do
  if [[ ! -s "$artifact" ]]; then
    echo "INCOMPLETE: required resident case artifact is missing or empty: $artifact" >&2
    exit 2
  fi
  if ! jq empty "$artifact" >/dev/null 2>&1; then
    echo "INCOMPLETE: required resident case artifact is not valid JSON: $artifact" >&2
    exit 2
  fi
done

jq -n \
  --arg commit "$EXPECTED_COMMIT" \
  --slurpfile short "$SHORT_CASE" \
  --slurpfile medium "$MEDIUM_CASE" \
  --slurpfile short_raw "$SHORT_RAW" \
  --slurpfile medium_raw "$MEDIUM_RAW" \
  --argjson reference 3.500144036461 '
  ($short[0]) as $short |
  ($medium[0]) as $medium |
  ($short_raw[0]) as $short_raw |
  ($medium_raw[0]) as $medium_raw |
  (($short.decode_tps_mean * $medium.decode_tps_mean) | sqrt) as $gm |
  ($reference * 0.98) as $lower |
  ($reference * 1.02) as $upper |
  {
    case_qualification_passed: ([$short, $medium] | all(.qualification_passed == true)),
    output_token_parity_passed: ([$short_raw, $medium_raw] | all(.aggregate.output_token_parity == true)),
    zero_cache_misses: ([$short_raw, $medium_raw] | all(
      .aggregate.cache_misses_total == 0 and
      ([.runs[].cache_io.cache_misses] | all(. == 0))
    )),
    zero_ssd_bytes: ([$short_raw, $medium_raw] | all(
      .aggregate.ssd_bytes_total == 0 and
      ([.runs[].cache_io.total_expert_bytes_read] | all(. == 0))
    )),
    zero_foreground_io: ([$short_raw, $medium_raw] | all(
      [.runs[].cache_io] | all(
        .foreground_read_operations == 0 and
        .foreground_expert_bytes == 0 and
        .foreground_expert_io_wait_seconds == 0
      )
    )),
    zero_prefetch_io: ([$short_raw, $medium_raw] | all(
      [.runs[].cache_io] | all(
        .prefetch_submitted == 0 and
        .prefetch_completed == 0 and
        .prefetch_bytes == 0
      )
    )),
    zero_phase3_miss_telemetry: ([$short_raw, $medium_raw] | all(
      [.runs[].demand_miss_fanout.prompt, .runs[].demand_miss_fanout.decode] | all(
        .misses_at_initial_layer_lookup == 0 and
        .missing_experts_per_routed_layer[0] == .routed_layers_observed and
        ([.missing_experts_per_routed_layer[1:][]] | all(. == 0)) and
        .layers_with_multiple_simultaneous_misses == 0 and
        .layers_with_one_physical_read == 0 and
        .layers_with_no_foreground_physical_read == 0 and
        .layers_with_serial_physical_reads == 0 and
        .layers_with_overlapping_physical_reads == 0 and
        .layers_beginning_compute_before_all_misses_available == 0 and
        .foreground_physical_read_operations == 0 and
        .peak_foreground_physical_reads_in_flight == 0 and
        .foreground_physical_read_concurrency_integral_seconds == 0 and
        .foreground_physical_read_active_seconds == 0 and
        .average_foreground_physical_read_concurrency == 0 and
        .physical_read_issue_to_completion_seconds == 0 and
        .physical_read_issue_to_completion_mean_seconds == 0 and
        .physical_read_issue_to_completion_max_seconds == 0 and
        ([.physical_read_issue_to_completion_histogram[].count] | all(. == 0)) and
        .primary_buffer_acquisition_wait_seconds == 0 and
        .primary_buffer_acquisition_wait_mean_seconds == 0 and
        .primary_buffer_acquisition_wait_max_seconds == 0 and
        .foreground_admission_wait_seconds == 0 and
        .foreground_admission_wait_mean_seconds == 0 and
        .singleflight_wait_seconds == 0 and
        .completion_to_consumption_delay_seconds == 0 and
        .completion_to_consumption_delay_mean_seconds == 0 and
        .completion_to_consumption_delay_max_seconds == 0 and
        .first_miss_to_first_read_issue_seconds == 0 and
        .first_miss_to_last_read_issue_seconds == 0 and
        .first_to_last_read_issue_spread_seconds == 0 and
        .first_miss_to_first_required_expert_available_seconds == 0 and
        .first_miss_to_final_required_expert_available_seconds == 0 and
        .first_to_last_required_expert_completion_spread_seconds == 0 and
        .first_miss_to_expert_compute_begin_seconds == 0 and
        .first_miss_to_layer_completion_seconds == 0 and
        .layer_expert_fetch_critical_path_seconds == 0 and
        .layer_expert_fetch_critical_path_mean_seconds == 0 and
        .layer_expert_fetch_critical_path_max_seconds == 0 and
        .foreground_demand_bursts_entered == 0 and
        .foreground_demand_pressure_active_seconds == 0 and
        .speculative_physical_reads_admitted_without_demand_pressure == 0 and
        .speculative_physical_reads_deferred_for_demand_pressure == 0 and
        .deferred_speculative_physical_reads_resumed == 0 and
        .deferred_speculative_physical_reads_dropped_stale_duplicate_or_cache_hit == 0 and
        .speculative_physical_reads_active_when_demand_burst_began == 0 and
        .demand_reads_issued_while_speculative_reads_active == 0 and
        .demand_physical_read_service_without_speculation_operations == 0 and
        .demand_physical_read_service_without_speculation_seconds == 0 and
        .demand_physical_read_service_without_speculation_mean_seconds == 0 and
        .demand_physical_read_service_without_speculation_max_seconds == 0 and
        ([.demand_physical_read_service_without_speculation_histogram[].count] | all(. == 0)) and
        .demand_physical_read_service_with_speculation_operations == 0 and
        .demand_physical_read_service_with_speculation_seconds == 0 and
        .demand_physical_read_service_with_speculation_mean_seconds == 0 and
        .demand_physical_read_service_with_speculation_max_seconds == 0 and
        ([.demand_physical_read_service_with_speculation_histogram[].count] | all(. == 0)) and
        .demand_layers_final_straggler_issued_while_speculative_reads_active == 0 and
        .demand_critical_reads_delayed_by_speculative_activity == null and
        ([.final_straggler_routed_slot_histogram[]] | all(. == 0)) and
        .worst_layer_fetch == null
      )
    ))
  } as $resident_gates |
  (($gm >= $lower) and ($gm <= $upper)) as $performance_passed |
  (
    [
      if $resident_gates.case_qualification_passed then empty
      else "one or more resident case qualification gates failed" end,
      if $resident_gates.output_token_parity_passed then empty
      else "resident output-token parity failed" end,
      if $resident_gates.zero_cache_misses then empty
      else "resident cases recorded cache misses" end,
      if $resident_gates.zero_ssd_bytes then empty
      else "resident cases recorded SSD bytes" end,
      if $resident_gates.zero_foreground_io then empty
      else "resident cases recorded foreground expert I/O" end,
      if $resident_gates.zero_prefetch_io then empty
      else "resident cases recorded speculative prefetch I/O" end,
      if $resident_gates.zero_phase3_miss_telemetry then empty
      else "resident cases activated Phase 3 miss-only telemetry or arbitration" end,
      if $performance_passed then empty
      elif $gm < $lower then
        "resident performance gate failed: geometric mean decode TPS \($gm) is below the lower bound \($lower)"
      else
        "resident performance gate failed: geometric mean decode TPS \($gm) is above the upper bound \($upper)"
      end
    ]
  ) as $failure_reasons |
  {
    schema: {name:"mer-prompt2-resident-control-summary", version:1},
    git_commit_full: $commit,
    completed_case_count: 2,
    cases: [$short, $medium],
    reference: {
      commit: "327d263193c48f9dde9f6a716562260ab49fa7ef",
      source: "immediate Phase 1 A-B-A recheck",
      geometric_mean_decode_tps: $reference,
      tolerance_percent: 2.0
    },
    resident_geometric_mean_decode_tps: $gm,
    delta_from_reference_percent: (100 * ($gm / $reference - 1)),
    resident_gates: $resident_gates,
    handoff_runtime_present: false,
    handoff_activity: "not applicable; Phase 2 handoff runtime was removed",
    no_ssd_or_handoff_activity: (
      $resident_gates.zero_cache_misses and
      $resident_gates.zero_ssd_bytes and
      $resident_gates.zero_foreground_io and
      $resident_gates.zero_prefetch_io and
      $resident_gates.zero_phase3_miss_telemetry
    ),
    performance_gate: {
      lower_bound_decode_tps: $lower,
      upper_bound_decode_tps: $upper,
      passed: $performance_passed
    },
    failure_reasons: $failure_reasons,
    qualification_passed: ($failure_reasons | length == 0)
  }' > "$ARTIFACT_DIR/resident-control-summary.json"

jq -n \
  --arg commit "$EXPECTED_COMMIT" \
  --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --slurpfile summary "$ARTIFACT_DIR/resident-control-summary.json" '
  ($summary[0]) as $summary |
  {
    schema: {name:"mer-prompt2-resident-qualification", version:1},
    git_commit_full: $commit,
    captured_at_utc: $captured_at_utc,
    two_required_resident_cases_present: ($summary.completed_case_count == 2),
    schema_and_required_fields_passed: $summary.resident_gates.case_qualification_passed,
    provenance_and_backend_gates_passed: $summary.resident_gates.case_qualification_passed,
    prompt_identity_gates_passed: $summary.resident_gates.case_qualification_passed,
    strictness_and_correctness_gates_passed: (
      $summary.resident_gates.case_qualification_passed and
      $summary.resident_gates.output_token_parity_passed
    ),
    critical_path_coverage_gates_passed: $summary.resident_gates.case_qualification_passed,
    external_peak_rss_present: ([ $summary.cases[].external_peak_rss_bytes ] | all(. > 0)),
    handoff_runtime_present: false,
    no_ssd_or_handoff_activity: $summary.no_ssd_or_handoff_activity,
    resident_reference_tolerance_passed: $summary.performance_gate.passed,
    failure_reasons: $summary.failure_reasons,
    qualification_passed: $summary.qualification_passed
  }' > "$ARTIFACT_DIR/qualification.json"

if jq -e '.qualification_passed == true' "$ARTIFACT_DIR/qualification.json" >/dev/null; then
  exit 0
fi

jq -r '.failure_reasons[] | "QUALIFICATION FAILURE: \(.)"' \
  "$ARTIFACT_DIR/qualification.json" >&2
exit 1
