#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FILTER="$ROOT/scripts/qwen3_coder_prompt2_phase4c_matrix.jq"
RUNNER="$ROOT/scripts/collect_qwen3_coder_prompt2_phase4c_matrix.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4c-matrix.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

make_summary() {
  local variant=$1
  local predictor_mode=$2
  local fanout=$3
  local upper_bound=$4
  local first_order=$5
  local second_order=$6
  local fallback_fill=$7
  local governor=$8
  local short_tps=$9
  local medium_tps=${10}
  local prefetch_reads=${11}
  local governor_rejected=${12}
  local output=${13}
  jq -n \
    --arg variant "$variant" \
    --arg predictor_mode "$predictor_mode" \
    --argjson fanout "$fanout" \
    --argjson upper_bound "$upper_bound" \
    --argjson first_order "$first_order" \
    --argjson second_order "$second_order" \
    --argjson fallback_fill "$fallback_fill" \
    --argjson governor "$governor" \
    --argjson short_tps "$short_tps" \
    --argjson medium_tps "$medium_tps" \
    --argjson prefetch_reads "$prefetch_reads" \
    --argjson governor_rejected "$governor_rejected" '
    def metadata:
      {
        variant: $variant,
        predictor_mode: $predictor_mode,
        predict_fanout: $fanout,
        pipeline_depth: 3,
        first_order_enabled: $first_order,
        second_order_enabled: $second_order,
        fallback_prior_fill_enabled: $fallback_fill,
        fanout_is_upper_bound: $upper_bound,
        governor_enabled: $governor,
        neural_speculator_enabled: false
      };
    def governor_config:
      {
        enabled: $governor,
        precision_floor: 0.05,
        contention_weight: 1.0,
        runtime_default_precision_alpha: 0.2,
        runtime_default_base_threshold: 0.02
      };
    def governor_run($index):
      {
        run_index: $index,
        governor_enabled: $governor,
        neural_speculator_enabled: false,
        candidates_rejected_by_governor: $governor_rejected,
        speculative_work_admitted: $prefetch_reads,
        governor_admitted_candidates_derived: $prefetch_reads,
        governor_admission_derivation:
          "prefetch_submitted + prefetch_dropped_concurrency; governor admission precedes the concurrency gate",
        speculative_work_completed: $prefetch_reads,
        speculative_work_used: 1,
        speculative_work_dropped_by_concurrency: 2,
        speculative_work_dropped_by_pool_pressure: 3,
        demand_read_activity_while_speculation_active:
          {prompt: 4, decode: 5, total: 9},
        foreground_pressure: {
          prompt_peak_foreground_physical_reads_in_flight: 6,
          decode_peak_foreground_physical_reads_in_flight: 7,
          prompt_average_foreground_physical_read_concurrency: 1.5,
          decode_average_foreground_physical_read_concurrency: 2.5,
          prompt_foreground_physical_read_active_seconds: 3.5,
          decode_foreground_physical_read_active_seconds: 4.5
        }
      };
    def governor_runs: [range(0; 5) | governor_run(.)];
    def governor_totals:
      {
        candidates_rejected_by_governor: (5 * $governor_rejected),
        speculative_work_admitted: (5 * $prefetch_reads),
        governor_admitted_candidates_derived: (5 * $prefetch_reads),
        speculative_work_dropped_by_concurrency: 10,
        speculative_work_dropped_by_pool_pressure: 15,
        demand_reads_observed_while_speculation_active: 45
      };
    def case($prompt; $tps):
      {
        prompt_fixture: $prompt,
        decode_tps_mean: $tps,
        ssd_bytes_total: 1000,
        phase3a_decode: {
          physical_read_issue_to_completion_mean_seconds: 1
        },
        phase4a_prefetch: {
          counters: {prefetch_completed: $prefetch_reads}
        },
        phase4c_predictor: metadata,
        phase4c_governor: {
          configuration: governor_config,
          counters_by_run: governor_runs,
          totals: governor_totals
        }
      };
    [case("short"; $short_tps), case("medium"; $medium_tps)] as $cases |
    {
      variant: $variant,
      cache_slots: 1536,
      traced: false,
      metadata: metadata,
      governor_configuration: governor_config,
      governor_counters_by_case:
        [$cases[] | {
          case: ("baseline-1536-" + .prompt_fixture),
          prompt_fixture,
          counters_by_run: .phase4c_governor.counters_by_run,
          totals: .phase4c_governor.totals
        }],
      qualification_passed: true,
      cases: $cases
    }
    ' > "$output"
}

make_raw() {
  local output=$1
  local first_id=$2
  jq -n --argjson first_id "$first_id" '
    {runs: [{output_token_ids: [$first_id, 2, 3]}]}
  ' > "$output"
}

make_summary demand-only demand-only 0 false true true true false \
  100 100 0 0 "$TEST_ROOT/demand.json"
make_summary current-f2 legacy-combined 2 false true true true false \
  95 96 100 0 "$TEST_ROOT/current.json"
make_summary current-f2-governed legacy-combined 2 false true true true true \
  103 102 70 10 "$TEST_ROOT/current-governed.json"
make_summary second-only-f2 second-order-only 2 true false true false false \
  105 104 40 0 "$TEST_ROOT/second-f2.json"
make_summary second-only-f1 second-order-only 1 true false true false false \
  101 102 20 0 "$TEST_ROOT/second-f1.json"
make_summary second-only-f1-governed second-order-only 1 true false true false true \
  106 105 10 4 "$TEST_ROOT/second-f1-governed.json"

for name in \
  demand-short demand-medium \
  current-short current-medium \
  current-governed-short current-governed-medium \
  second-f2-short second-f2-medium \
  second-f1-short second-f1-medium \
  second-f1-governed-short second-f1-governed-medium; do
  make_raw "$TEST_ROOT/$name.json" 1
done

render_matrix() {
  local second_f1_governed_medium=$1
  local output=$2
  jq -n \
    --slurpfile demand "$TEST_ROOT/demand.json" \
    --slurpfile current "$TEST_ROOT/current.json" \
    --slurpfile current_governed "$TEST_ROOT/current-governed.json" \
    --slurpfile second_f2 "$TEST_ROOT/second-f2.json" \
    --slurpfile second_f1 "$TEST_ROOT/second-f1.json" \
    --slurpfile second_f1_governed "$TEST_ROOT/second-f1-governed.json" \
    --slurpfile demand_short "$TEST_ROOT/demand-short.json" \
    --slurpfile demand_medium "$TEST_ROOT/demand-medium.json" \
    --slurpfile current_short "$TEST_ROOT/current-short.json" \
    --slurpfile current_medium "$TEST_ROOT/current-medium.json" \
    --slurpfile current_governed_short "$TEST_ROOT/current-governed-short.json" \
    --slurpfile current_governed_medium "$TEST_ROOT/current-governed-medium.json" \
    --slurpfile second_f2_short "$TEST_ROOT/second-f2-short.json" \
    --slurpfile second_f2_medium "$TEST_ROOT/second-f2-medium.json" \
    --slurpfile second_f1_short "$TEST_ROOT/second-f1-short.json" \
    --slurpfile second_f1_medium "$TEST_ROOT/second-f1-medium.json" \
    --slurpfile second_f1_governed_short "$TEST_ROOT/second-f1-governed-short.json" \
    --slurpfile second_f1_governed_medium "$second_f1_governed_medium" \
    -f "$FILTER" > "$output"
}

render_matrix "$TEST_ROOT/second-f1-governed-medium.json" "$TEST_ROOT/matrix.json"
jq -e '
  .schema == {"name":"mer-prompt2-phase4c-matrix-summary","version":1} and
  .traced == false and
  .cache_slots == 1536 and
  .prompt_fixtures == ["short", "medium"] and
  .expected_variant_order == [
    "demand-only",
    "current-f2",
    "current-f2-governed",
    "second-only-f2",
    "second-only-f1",
    "second-only-f1-governed"
  ] and
  ([.variants[].variant] == .expected_variant_order) and
  (.variants | length) == 6 and
  ([.variants[].variant] | index("second-only-f2-governed")) == null and
  [.variants[].metadata] == [
    {
      variant: "demand-only",
      predictor_mode: "demand-only",
      predict_fanout: 0,
      pipeline_depth: 3,
      first_order_enabled: true,
      second_order_enabled: true,
      fallback_prior_fill_enabled: true,
      fanout_is_upper_bound: false,
      governor_enabled: false,
      neural_speculator_enabled: false
    },
    {
      variant: "current-f2",
      predictor_mode: "legacy-combined",
      predict_fanout: 2,
      pipeline_depth: 3,
      first_order_enabled: true,
      second_order_enabled: true,
      fallback_prior_fill_enabled: true,
      fanout_is_upper_bound: false,
      governor_enabled: false,
      neural_speculator_enabled: false
    },
    {
      variant: "current-f2-governed",
      predictor_mode: "legacy-combined",
      predict_fanout: 2,
      pipeline_depth: 3,
      first_order_enabled: true,
      second_order_enabled: true,
      fallback_prior_fill_enabled: true,
      fanout_is_upper_bound: false,
      governor_enabled: true,
      neural_speculator_enabled: false
    },
    {
      variant: "second-only-f2",
      predictor_mode: "second-order-only",
      predict_fanout: 2,
      pipeline_depth: 3,
      first_order_enabled: false,
      second_order_enabled: true,
      fallback_prior_fill_enabled: false,
      fanout_is_upper_bound: true,
      governor_enabled: false,
      neural_speculator_enabled: false
    },
    {
      variant: "second-only-f1",
      predictor_mode: "second-order-only",
      predict_fanout: 1,
      pipeline_depth: 3,
      first_order_enabled: false,
      second_order_enabled: true,
      fallback_prior_fill_enabled: false,
      fanout_is_upper_bound: true,
      governor_enabled: false,
      neural_speculator_enabled: false
    },
    {
      variant: "second-only-f1-governed",
      predictor_mode: "second-order-only",
      predict_fanout: 1,
      pipeline_depth: 3,
      first_order_enabled: false,
      second_order_enabled: true,
      fallback_prior_fill_enabled: false,
      fanout_is_upper_bound: true,
      governor_enabled: true,
      neural_speculator_enabled: false
    }
  ] and
  .gates.exact_six_variant_matrix == true and
  .gates.all_variant_qualifications_passed == true and
  .gates.neural_speculator_disabled_everywhere == true and
  .gates.governor_assignment_exact == true and
  .gates.governed_cases_share_configuration == true and
  .gates.current_governor_pair_differs_only_by_governor == true and
  .gates.second_f1_governor_pair_differs_only_by_governor == true and
  .gates.cross_variant_short_output_parity == true and
  .gates.cross_variant_medium_output_parity == true and
  ([.variants[].metadata.neural_speculator_enabled] | all(. == false)) and
  ([.variants[] | select(.metadata.governor_enabled) | .variant] ==
    ["current-f2-governed", "second-only-f1-governed"]) and
  ([.variants[].cases[] |
    (.phase4c_governor.counters_by_run | length)] | all(. == 5)) and
  ([.variants[].cases[].phase4c_governor.counters_by_run[] |
    has("candidates_rejected_by_governor") and
    has("speculative_work_admitted") and
    has("speculative_work_dropped_by_concurrency") and
    has("speculative_work_dropped_by_pool_pressure") and
    has("demand_read_activity_while_speculation_active") and
    has("foreground_pressure")] | all) and
  [.causal_comparisons[].name] == [
    "current-f2_vs_current-f2-governed",
    "second-only-f1_vs_second-only-f1-governed",
    "current-f2_vs_second-only-f1",
    "current-f2-governed_vs_second-only-f1-governed"
  ] and
  (.causal_comparisons | length) == 4 and
  ([.causal_comparisons[].prompts | length] | all(. == 2)) and
  ([.causal_comparisons[].prompts[].governor_counters_by_run.control | length] |
   all(. == 5)) and
  .candidate_promotion_evidence[0].resolved_primary_gates.beats_current_f2_on_short == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.beats_current_f2_on_medium == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.recovers_demand_only_on_short == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.recovers_demand_only_on_medium == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.geometric_mean_at_least_three_percent_over_demand_only == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.reduces_speculative_reads_vs_current_f2 == true and
  .candidate_promotion_evidence[0].thresholded_gates_requiring_experiment_judgment.ssd_bytes_not_materially_higher_than_demand_only == null and
  (.candidate_promotion_evidence[0].diagnostic_gates_pending | length) == 5
' "$TEST_ROOT/matrix.json" >/dev/null

make_raw "$TEST_ROOT/second-f1-governed-medium-mismatch.json" 9
render_matrix \
  "$TEST_ROOT/second-f1-governed-medium-mismatch.json" \
  "$TEST_ROOT/parity-failure.json"
jq -e '.gates.cross_variant_medium_output_parity == false' \
  "$TEST_ROOT/parity-failure.json" >/dev/null

if grep -F '6144' "$RUNNER" >/dev/null; then
  echo "Phase 4C matrix runner must not generate a 6,144-slot case" >&2
  exit 1
fi

echo "Phase 4C six-variant governor matrix fixtures: PASS"
