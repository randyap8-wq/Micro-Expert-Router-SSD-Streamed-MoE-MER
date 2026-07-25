#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FILTER="$ROOT/scripts/qwen3_coder_prompt2_phase4c_matrix.jq"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4c-matrix.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

make_summary() {
  local variant=$1
  local short_tps=$2
  local medium_tps=$3
  local prefetch_reads=$4
  local output=$5
  jq -n \
    --arg variant "$variant" \
    --argjson short_tps "$short_tps" \
    --argjson medium_tps "$medium_tps" \
    --argjson prefetch_reads "$prefetch_reads" '
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
        }
      };
    {
      variant: $variant,
      qualification_passed: true,
      cases: [case("short"; $short_tps), case("medium"; $medium_tps)]
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

make_summary demand-only 100 100 0 "$TEST_ROOT/demand.json"
make_summary current-f2 95 96 100 "$TEST_ROOT/current.json"
make_summary second-only-f2 105 104 40 "$TEST_ROOT/second-f2.json"
make_summary second-only-f1 101 102 20 "$TEST_ROOT/second-f1.json"
for name in demand-short demand-medium current-short current-medium second-f2-short second-f2-medium second-f1-short second-f1-medium; do
  make_raw "$TEST_ROOT/$name.json" 1
done

render_matrix() {
  local second_f1_medium=$1
  local output=$2
  jq -n \
    --slurpfile demand "$TEST_ROOT/demand.json" \
    --slurpfile current "$TEST_ROOT/current.json" \
    --slurpfile second_f2 "$TEST_ROOT/second-f2.json" \
    --slurpfile second_f1 "$TEST_ROOT/second-f1.json" \
    --slurpfile demand_short "$TEST_ROOT/demand-short.json" \
    --slurpfile demand_medium "$TEST_ROOT/demand-medium.json" \
    --slurpfile current_short "$TEST_ROOT/current-short.json" \
    --slurpfile current_medium "$TEST_ROOT/current-medium.json" \
    --slurpfile second_f2_short "$TEST_ROOT/second-f2-short.json" \
    --slurpfile second_f2_medium "$TEST_ROOT/second-f2-medium.json" \
    --slurpfile second_f1_short "$TEST_ROOT/second-f1-short.json" \
    --slurpfile second_f1_medium "$second_f1_medium" \
    -f "$FILTER" > "$output"
}

render_matrix "$TEST_ROOT/second-f1-medium.json" "$TEST_ROOT/matrix.json"
jq -e '
  .schema == {"name":"mer-prompt2-phase4c-matrix-summary","version":1} and
  .traced == false and
  (.variants | length) == 4 and
  .gates.all_variant_qualifications_passed == true and
  .gates.cross_variant_short_output_parity == true and
  .gates.cross_variant_medium_output_parity == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.beats_current_f2_on_short == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.beats_current_f2_on_medium == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.recovers_demand_only_on_short == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.recovers_demand_only_on_medium == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.geometric_mean_at_least_three_percent_over_demand_only == true and
  .candidate_promotion_evidence[0].resolved_primary_gates.reduces_speculative_reads_vs_current_f2 == true and
  .candidate_promotion_evidence[0].thresholded_gates_requiring_experiment_judgment.ssd_bytes_not_materially_higher_than_demand_only == null and
  (.candidate_promotion_evidence[0].diagnostic_gates_pending | length) == 5 and
  .candidate_promotion_evidence[1].resolved_primary_gates.geometric_mean_at_least_three_percent_over_demand_only == false
' "$TEST_ROOT/matrix.json" >/dev/null

make_raw "$TEST_ROOT/second-f1-medium-mismatch.json" 9
render_matrix "$TEST_ROOT/second-f1-medium-mismatch.json" "$TEST_ROOT/parity-failure.json"
jq -e '.gates.cross_variant_medium_output_parity == false' \
  "$TEST_ROOT/parity-failure.json" >/dev/null

echo "Phase 4C matrix fixtures: PASS"
