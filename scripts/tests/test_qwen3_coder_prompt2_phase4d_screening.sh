#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FILTER="$ROOT/scripts/qwen3_coder_prompt2_phase4d_screening.jq"
RUNNER="$ROOT/scripts/collect_qwen3_coder_prompt2_phase4d_screening.sh"
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
CONFIG_HELPER="$ROOT/scripts/qwen3_coder_prompt2_collector_config.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4d-screening.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT
TEST_GIT_COMMIT_FULL=0123456789012345678901234567890123456789
TEST_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

source "$CONFIG_HELPER"

make_variant() {
  local variant=$1
  local predictor_mode=$2
  local fanout=$3
  local governor=$4
  local base_threshold=$5
  local contention_weight=$6
  local admitted_per_run=$7
  local rejected_per_run=$8
  local completed=$9
  local used=${10}
  local output_id=${11}
  local output=${12}
  local first_order=true
  local fallback_fill=true
  local upper_bound=false
  if [[ "$predictor_mode" == second-order-only ]]; then
    first_order=false
    fallback_fill=false
    upper_bound=true
  fi

  jq -n \
    --arg variant "$variant" \
    --arg predictor_mode "$predictor_mode" \
    --argjson fanout "$fanout" \
    --argjson governor "$governor" \
    --argjson base_threshold "$base_threshold" \
    --argjson contention_weight "$contention_weight" \
    --argjson first_order "$first_order" \
    --argjson fallback_fill "$fallback_fill" \
    --argjson upper_bound "$upper_bound" \
    --argjson admitted "$admitted_per_run" \
    --argjson rejected "$rejected_per_run" \
    --argjson completed "$completed" \
    --argjson used "$used" \
    --argjson output_id "$output_id" \
    --arg git_commit_full "$TEST_GIT_COMMIT_FULL" \
    --arg identity_sha256 "$TEST_SHA256" '
    def run($index):
      {
        run_index: $index,
        governor_enabled: $governor,
        neural_speculator_enabled: false,
        candidates_rejected_by_governor: $rejected,
        governor_admitted_candidates: $admitted,
        governor_rejected_candidates: $rejected,
        governor_total_decisions: ($admitted + $rejected),
        governor_admission_rate:
          (if ($admitted + $rejected) == 0 then 0
           else ($admitted / ($admitted + $rejected)) end),
        governor_precision_ewma_final: (0.5 - ($index * 0.1)),
        governor_foreground_inflight_final: 0,
        direct_governor_decisions: {
          admitted_candidates: $admitted,
          rejected_candidates: $rejected,
          total_decisions: ($admitted + $rejected),
          admission_rate:
            (if ($admitted + $rejected) == 0 then 0
             else ($admitted / ($admitted + $rejected)) end)
        },
        speculative_work_admitted: $admitted,
        governor_admitted_candidates_derived: $admitted,
        governor_admission_derivation:
          "prefetch_submitted + prefetch_dropped_concurrency; governor admission precedes the concurrency gate",
        derived_governor_admission: {
          admitted_candidates: $admitted,
          derivation:
            "prefetch_submitted + prefetch_dropped_concurrency; governor admission precedes the concurrency gate"
        },
        speculative_work_completed: $completed,
        speculative_work_used: $used,
        speculative_work_dropped_by_concurrency: 0,
        speculative_work_dropped_by_pool_pressure: 0,
        demand_read_activity_while_speculation_active:
          {prompt: 0, decode: $admitted, total: $admitted},
        foreground_pressure: {}
      };
    [run(0), run(1)] as $runs |
    ($admitted * 2) as $admitted_total |
    ($rejected * 2) as $rejected_total |
    {
      schema: {name:"mer-prompt2-phase4d-screening-variant", version:1},
      qualification_kind: "phase4d-governor-screening",
      variant: $variant,
      cache_slots: 1536,
      prompt_fixture: "short",
      output_tokens: 128,
      warmup_runs: 1,
      measured_runs: 2,
      greedy: true,
      traced: false,
      metadata: {
        variant: $variant,
        predictor_mode: $predictor_mode,
        predict_fanout: $fanout,
        pipeline_depth: 3,
        first_order_enabled: $first_order,
        second_order_enabled: true,
        fallback_prior_fill_enabled: $fallback_fill,
        fanout_is_upper_bound: $upper_bound,
        governor_enabled: $governor,
        neural_speculator_enabled: false
      },
      governor_configuration: {
        enabled: $governor,
        precision_floor: 0.05,
        contention_weight: $contention_weight,
        base_threshold: $base_threshold,
        runtime_default_precision_alpha: 0.2,
        runtime_default_base_threshold: 0.02
      },
      provenance: {
        git_commit_full: $git_commit_full,
        binary_sha256: $identity_sha256,
        model_hashes: {
          config_json_sha256: $identity_sha256,
          dense_manifest_sha256: $identity_sha256
        },
        tokenizer_identity: {
          path: "/mnt/localssd/model/tokenizer.json",
          sha256: $identity_sha256
        },
        target_host: {
          hostname: "phase4d-target",
          logical_cpu_count: 32,
          requested_cpu_mask: "0-31",
          effective_cpu_mask: "0-31"
        },
        model_mount_identity: {
          target: "/mnt/localssd",
          source: "/dev/nvme0n1",
          fstype: "ext4",
          options: "rw,noatime,nodiratime"
        },
        cargo_features: [
          "avx512", "blas", "io_uring", "q8-candle-reference", "tokenizer"
        ],
        prompt_hashes: {
          short_sha256: $identity_sha256,
          medium_sha256: $identity_sha256
        }
      },
      governor_counters_by_run: $runs,
      governor_totals: {
        candidates_rejected_by_governor: $rejected_total,
        governor_admitted_candidates: $admitted_total,
        governor_rejected_candidates: $rejected_total,
        governor_total_decisions: ($admitted_total + $rejected_total),
        governor_admission_rate:
          (if ($admitted_total + $rejected_total) == 0 then 0
           else ($admitted_total / ($admitted_total + $rejected_total)) end),
        governor_precision_ewma_final: ($runs | last | .governor_precision_ewma_final),
        governor_foreground_inflight_final: 0,
        speculative_work_admitted: $admitted_total,
        governor_admitted_candidates_derived: $admitted_total,
        speculative_work_dropped_by_concurrency: 0,
        speculative_work_dropped_by_pool_pressure: 0,
        demand_reads_observed_while_speculation_active: $admitted_total
      },
      prefetch_counters: {
        prefetch_submitted:
          (if $fanout == 0 then 0
           elif $governor then $admitted_total
           else $completed
           end),
        prefetch_completed: $completed,
        prefetch_used: $used,
        prefetch_bytes: (if $fanout == 0 then 0 else 100 end),
        useful_prefetch_bytes: (if $fanout == 0 then 0 else 50 end),
        unused_prefetch_bytes_at_sample: (if $fanout == 0 then 0 else 50 end),
        prefetch_dropped_concurrency: 0,
        prefetch_dropped_pool_starved: 0,
        prefetch_dropped_governor: $rejected_total,
        prefetch_dropped_bytes: 0
      },
      decode_tps_mean: 1.5,
      ssd_bytes: 1000,
      demand_read_service_mean_seconds: 0.2,
      demand_reads_observed_while_speculation_active: $admitted_total,
      output_token_ids: [$output_id, 2, 3],
      output_parity_within_variant: true,
      screening_collection_valid: true,
      performance_qualification_applicable: false,
      performance_qualification_reason: "screening only",
      qualification_passed: false
    }
  ' > "$output"
}

make_variant demand-only demand-only 0 false 0.02 1.0 0 0 0 0 1 \
  "$TEST_ROOT/demand.json"
make_variant second-only-f1 second-order-only 1 false 0.02 1.0 0 0 8 3 1 \
  "$TEST_ROOT/second.json"
# Zero admissions are valid screening evidence as long as decisions occurred.
make_variant second-only-f1-governed-current second-order-only 1 true \
  0.02 1.0 0 5 0 0 1 "$TEST_ROOT/current.json"
make_variant second-only-f1-governed-cw025 second-order-only 1 true \
  0.02 0.25 2 3 3 4 1 "$TEST_ROOT/cw025.json"
make_variant second-only-f1-governed-bt010-cw025 second-order-only 1 true \
  0.01 0.25 4 1 6 2 1 "$TEST_ROOT/bt010.json"
make_variant second-only-f1-governed-bt005-cw000 second-order-only 1 true \
  0.005 0.0 5 0 8 4 1 "$TEST_ROOT/bt005.json"

render_summary() {
  local second=$1
  local current=$2
  local bt005=$3
  local output=$4
  jq -n \
    --arg runner_git_commit_full "$TEST_GIT_COMMIT_FULL" \
    --slurpfile demand "$TEST_ROOT/demand.json" \
    --slurpfile second "$second" \
    --slurpfile current "$current" \
    --slurpfile cw025 "$TEST_ROOT/cw025.json" \
    --slurpfile bt010 "$TEST_ROOT/bt010.json" \
    --slurpfile bt005 "$bt005" \
    -f "$FILTER" > "$output"
}

render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/current.json" \
  "$TEST_ROOT/bt005.json" "$TEST_ROOT/summary.json"
jq -e --arg commit "$TEST_GIT_COMMIT_FULL" '
  .schema == {name:"mer-prompt2-phase4d-governor-screening-summary", version:1} and
  .qualification_kind == "phase4d-governor-screening" and
  .performance_qualification_applicable == false and
  .qualification_passed == false and
  .expected_variant_order == [
    "demand-only",
    "second-only-f1",
    "second-only-f1-governed-current",
    "second-only-f1-governed-cw025",
    "second-only-f1-governed-bt010-cw025",
    "second-only-f1-governed-bt005-cw000"
  ] and
  [.variants[].variant] == .expected_variant_order and
  .gates.exact_variant_order and
  .gates.deterministic_output_parity and
  .gates.neural_speculator_disabled_everywhere and
  .gates.governor_assignment_exact and
  .gates.predictor_configuration_exact and
  .gates.governor_configuration_values_exact and
  .gates.direct_decision_counters_reconcile and
  .gates.direct_and_derived_admissions_reconcile and
  .gates.foreground_read_accounting_balanced and
  .gates.demand_only_has_zero_speculative_work and
  .gates.ungoverned_second_only_f1_is_active and
  .gates.every_governed_case_made_an_admission_decision and
  .gates.cross_variant_provenance_identical and
  .gates.screening_mode_is_non_qualifying and
  .screening_gates_passed and
  .runner_git_commit_full == $commit and
  .provenance.git_commit_full == $commit and
  .variants[2].governor_admitted_candidates == 0 and
  .variants[2].governor_rejected_candidates > 0 and
  .variants[2].measured_window_prefetch_used_per_completed == null and
  .variants[3].measured_window_prefetch_used_per_completed > 1 and
  (.variants[3].counter_semantics.measured_window_prefetch_used_per_completed |
   contains("not exact cohort precision")) and
  ([.variants[] | has("demand_read_service_mean_seconds")] | all) and
  ([.variants[] | has("demand_read_service_seconds")] | any | not) and
  ([.variants[] | has("completed_prefetch_precision")] | any | not) and
  ([.variants[] | has("governor_admitted_candidates_derived")] | all)
' "$TEST_ROOT/summary.json" >/dev/null

jq '.governor_counters_by_run[0].governor_total_decisions += 1' \
  "$TEST_ROOT/current.json" > "$TEST_ROOT/counter-mismatch.json"
render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/counter-mismatch.json" \
  "$TEST_ROOT/bt005.json" "$TEST_ROOT/counter-mismatch-summary.json"
jq -e '
  .gates.direct_decision_counters_reconcile == false and
  .screening_gates_passed == false and
  .qualification_passed == false
' "$TEST_ROOT/counter-mismatch-summary.json" >/dev/null

jq '.governor_counters_by_run[0].governor_admitted_candidates_derived += 1' \
  "$TEST_ROOT/current.json" > "$TEST_ROOT/derived-mismatch.json"
render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/derived-mismatch.json" \
  "$TEST_ROOT/bt005.json" "$TEST_ROOT/derived-mismatch-summary.json"
jq -e '
  .gates.direct_and_derived_admissions_reconcile == false and
  .screening_gates_passed == false and
  .qualification_passed == false
' "$TEST_ROOT/derived-mismatch-summary.json" >/dev/null

jq '.governor_counters_by_run[1].governor_foreground_inflight_final = 1' \
  "$TEST_ROOT/current.json" > "$TEST_ROOT/foreground-unbalanced.json"
render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/foreground-unbalanced.json" \
  "$TEST_ROOT/bt005.json" "$TEST_ROOT/foreground-unbalanced-summary.json"
jq -e '
  .gates.foreground_read_accounting_balanced == false and
  .screening_gates_passed == false and
  .qualification_passed == false
' "$TEST_ROOT/foreground-unbalanced-summary.json" >/dev/null

jq '
  .prefetch_counters.prefetch_submitted = 0 |
  .prefetch_counters.prefetch_completed = 0
' "$TEST_ROOT/second.json" > "$TEST_ROOT/inactive-control.json"
render_summary "$TEST_ROOT/inactive-control.json" "$TEST_ROOT/current.json" \
  "$TEST_ROOT/bt005.json" "$TEST_ROOT/inactive-control-summary.json"
jq -e '
  .gates.ungoverned_second_only_f1_is_active == false and
  .screening_gates_passed == false and
  .qualification_passed == false
' "$TEST_ROOT/inactive-control-summary.json" >/dev/null

jq '.provenance.binary_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"' \
  "$TEST_ROOT/bt005.json" > "$TEST_ROOT/provenance-mismatch.json"
render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/current.json" \
  "$TEST_ROOT/provenance-mismatch.json" \
  "$TEST_ROOT/provenance-mismatch-summary.json"
jq -e '
  .gates.cross_variant_provenance_identical == false and
  .screening_gates_passed == false and
  .qualification_passed == false
' "$TEST_ROOT/provenance-mismatch-summary.json" >/dev/null

jq '.output_token_ids[0] = 99' "$TEST_ROOT/bt005.json" \
  > "$TEST_ROOT/output-mismatch.json"
render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/current.json" \
  "$TEST_ROOT/output-mismatch.json" "$TEST_ROOT/output-mismatch-summary.json"
jq -e '
  .gates.deterministic_output_parity == false and
  .screening_gates_passed == false and
  .qualification_passed == false
' "$TEST_ROOT/output-mismatch-summary.json" >/dev/null

expected_specs='  "demand-only|0.05|0.02|1.0"
  "second-only-f1|0.05|0.02|1.0"
  "second-only-f1-governed-current|0.05|0.02|1.0"
  "second-only-f1-governed-cw025|0.05|0.02|0.25"
  "second-only-f1-governed-bt010-cw025|0.05|0.01|0.25"
  "second-only-f1-governed-bt005-cw000|0.05|0.005|0.0"'
actual_specs=$(sed -n '/^variant_specs=($/,/^)/p' "$RUNNER" | sed '1d;$d')
test "$actual_specs" = "$expected_specs"

unset \
  MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR \
  MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT \
  MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD
for spec in \
  'second-only-f1-governed-current|0.02|1.0' \
  'second-only-f1-governed-cw025|0.02|0.25' \
  'second-only-f1-governed-bt010-cw025|0.01|0.25' \
  'second-only-f1-governed-bt005-cw000|0.005|0.0'; do
  IFS='|' read -r variant expected_base expected_weight <<<"$spec"
  MER_PROMPT2_PREFETCH_VARIANT=$variant
  prompt2_resolve_ablation_config
  test "$PREDICTOR_MODE" = second-order-only
  test "$PREDICT_FANOUT" -eq 1
  test "$PIPELINE_DEPTH" -eq 3
  test "$FIRST_ORDER_ENABLED" = false
  test "$SECOND_ORDER_ENABLED" = true
  test "$FALLBACK_PRIOR_FILL_ENABLED" = false
  test "$FANOUT_IS_UPPER_BOUND" = true
  test "$PREFETCH_GOVERNOR_ENABLED" = true
  test "$PREFETCH_GOVERNOR_PRECISION_FLOOR" = 0.05
  test "$PREFETCH_GOVERNOR_BASE_THRESHOLD" = "$expected_base"
  test "$PREFETCH_GOVERNOR_CONTENTION_WEIGHT" = "$expected_weight"
done
unset MER_PROMPT2_PREFETCH_VARIANT

phase4d_plan=$(sed -n \
  '/^# BEGIN PHASE4D_SCREENING_COLLECTION$/,/^# END PHASE4D_SCREENING_COLLECTION$/p' \
  "$COLLECTOR")
test "$(grep -Fxc '# BEGIN PHASE4D_SCREENING_COLLECTION' <<<"$phase4d_plan")" -eq 1
test "$(grep -Fxc '# END PHASE4D_SCREENING_COLLECTION' <<<"$phase4d_plan")" -eq 1
test "$(grep -c '^elif \[\[ "$COLLECTOR_MODE" ==' <<<"$phase4d_plan")" -eq 1
grep -Fx 'elif [[ "$COLLECTOR_MODE" == phase4d-screening ]]; then' \
  <<<"$phase4d_plan" >/dev/null
grep -F 'run_case 1536 short 14 "$SHORT_PROMPT_SHA"' <<<"$phase4d_plan" >/dev/null
if grep -F 'run_case 1536 medium' <<<"$phase4d_plan" >/dev/null ||
   grep -F 'run_case 6144' <<<"$phase4d_plan" >/dev/null; then
  echo "Phase 4D-A collector must run only the 1,536-slot short prompt" >&2
  exit 1
fi
grep -F 'QUALIFICATION_KIND=phase4d-governor-screening' "$COLLECTOR" >/dev/null
grep -F 'MEASURED_RUNS=2' "$COLLECTOR" >/dev/null
grep -F '.governor_foreground_inflight_final == 0' "$COLLECTOR" >/dev/null
grep -F 'demand_read_service_mean_seconds:' <<<"$phase4d_plan" >/dev/null
if grep -F 'demand_read_service_seconds:' <<<"$phase4d_plan" >/dev/null; then
  echo "Phase 4D-A must export the weighted mean-per-read service metric" >&2
  exit 1
fi
grep -F -- '--arg runner_git_commit_full "$RUNNER_GIT_COMMIT_FULL"' \
  "$RUNNER" >/dev/null
grep -F '.qualification_passed == false' "$RUNNER" >/dev/null

echo "Phase 4D-A governor screening fixtures: PASS"
