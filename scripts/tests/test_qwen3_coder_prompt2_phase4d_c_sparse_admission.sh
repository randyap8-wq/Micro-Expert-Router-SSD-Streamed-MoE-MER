#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FILTER="$ROOT/scripts/qwen3_coder_prompt2_phase4d_c_sparse_admission.jq"
RUNNER="$ROOT/scripts/collect_qwen3_coder_prompt2_phase4d_c_sparse_admission.sh"
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4d-c-sparse.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

TEST_GIT_COMMIT_FULL=0123456789012345678901234567890123456789
TEST_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

make_variant() {
  local variant=$1
  local governor_enabled=$2
  local base_threshold=$3
  local contention_weight=$4
  local admitted_per_run=$5
  local rejected_per_run=$6
  local prefetch_submitted=$7
  local prefetch_completed=$8
  local output=$9

  jq -n \
    --arg variant "$variant" \
    --argjson governor_enabled "$governor_enabled" \
    --argjson base_threshold "$base_threshold" \
    --argjson contention_weight "$contention_weight" \
    --argjson admitted "$admitted_per_run" \
    --argjson rejected "$rejected_per_run" \
    --argjson prefetch_submitted "$prefetch_submitted" \
    --argjson prefetch_completed "$prefetch_completed" \
    --arg git_commit_full "$TEST_GIT_COMMIT_FULL" \
    --arg sha "$TEST_SHA256" '
    def distribution($count; $minimum; $maximum; $mean):
      if $count == 0 then
        {
          count: 0,
          minimum: null,
          maximum: null,
          mean: null,
          p50: null,
          p90: null,
          p95: null,
          p99: null
        }
      else
        {
          count: $count,
          minimum: $minimum,
          maximum: $maximum,
          mean: $mean,
          p50: $mean,
          p90: $maximum,
          p95: $maximum,
          p99: $maximum
        }
      end;

    def diagnostics($enabled; $base; $weight; $admitted; $rejected):
      ($admitted + $rejected) as $total |
      ($base * 0.2) as $rejected_score |
      ($base * 1.2) as $admitted_score |
      {
        semantics:
          "exact counts and bounded nearest-rank percentiles over the most recent 512 finite decisions",
        enabled: $enabled,
        sample_capacity: 512,
        sampled_decisions: $total,
        total_decisions: $total,
        admitted: $admitted,
        rejected: $rejected,
        invalid_numeric_decisions: 0,
        ratio_undefined_decisions: 0,
        base_threshold: $base,
        contention_weight: $weight,
        candidate_probability:
          distribution($total; 0.1; 0.5; 0.3),
        effective_precision:
          distribution($total; 0.05; 0.5; 0.2),
        candidate_score:
          distribution(
            $total;
            (if $rejected > 0 then $rejected_score else $admitted_score end);
            (if $admitted > 0 then $admitted_score else $rejected_score end);
            (if $admitted > 0 and $rejected > 0 then
               (($rejected_score + $admitted_score) / 2)
             elif $admitted > 0 then $admitted_score
             else $rejected_score
             end)),
        effective_threshold:
          distribution($total; $base; $base; $base),
        score_to_threshold_ratio:
          distribution(
            $total;
            (if $rejected > 0 then 0.2 else 1.2 end);
            (if $admitted > 0 then 1.2 else 0.2 end);
            (if $admitted > 0 and $rejected > 0 then 0.7
             elif $admitted > 0 then 1.2
             else 0.2
             end)),
        decision_boundary: {
          maximum_rejected_candidate_score:
            (if $rejected > 0 then $rejected_score else null end),
          maximum_rejected_score_to_threshold_ratio:
            (if $rejected > 0 then 0.2 else null end),
          minimum_admitted_candidate_score:
            (if $admitted > 0 then $admitted_score else null end),
          minimum_admitted_score_to_threshold_ratio:
            (if $admitted > 0 then 1.2 else null end),
          closest_rejected_score_minus_threshold_margin:
            (if $rejected > 0 then ($rejected_score - $base) else null end),
          closest_admitted_score_minus_threshold_margin:
            (if $admitted > 0 then ($admitted_score - $base) else null end)
        },
        foreground_inflight_decisions: {
          foreground_inflight_zero: $total,
          foreground_inflight_zero_admitted: $admitted,
          foreground_inflight_zero_rejected: $rejected,
          foreground_inflight_positive: 0,
          foreground_inflight_positive_admitted: 0,
          foreground_inflight_positive_rejected: 0
        }
      };

    def run($index):
      diagnostics(
        $governor_enabled;
        $base_threshold;
        $contention_weight;
        $admitted;
        $rejected
      ) as $diagnostics |
      {
        run_index: $index,
        governor_enabled: $governor_enabled,
        governor_admitted_candidates: $admitted,
        governor_rejected_candidates: $rejected,
        governor_total_decisions: ($admitted + $rejected),
        governor_admission_rate:
          (if ($admitted + $rejected) == 0 then 0
           else ($admitted / ($admitted + $rejected)) end),
        governor_precision_ewma_final: 0.05,
        governor_foreground_inflight_final: 0,
        governor_score_diagnostics: $diagnostics,
        governor_score_diagnostics_reconcile: true,
        direct_governor_decisions: {
          admitted_candidates: $admitted,
          rejected_candidates: $rejected,
          total_decisions: ($admitted + $rejected),
          admission_rate:
            (if ($admitted + $rejected) == 0 then 0
             else ($admitted / ($admitted + $rejected)) end)
        }
      };

    [run(0), run(1)] as $runs |
    ($admitted * 2) as $admitted_total |
    ($rejected * 2) as $rejected_total |
    diagnostics(
      $governor_enabled;
      $base_threshold;
      $contention_weight;
      $admitted_total;
      $rejected_total
    ) as $aggregate_diagnostics |
    {
      schema: {
        name: "mer-prompt2-phase4d-c-sparse-admission-variant",
        version: 1
      },
      qualification_kind: "phase4d-c-sparse-admission-screening",
      variant: $variant,
      cache_slots: 1536,
      prompt_fixture: "short",
      output_tokens: 128,
      warmup_runs: 1,
      measured_runs: 2,
      cache_reset: "keep",
      greedy: true,
      traced: false,
      metadata:
        (if $variant == "demand-only" then
          {
            variant: $variant,
            predictor_mode: "demand-only",
            predict_fanout: 0,
            pipeline_depth: 3,
            first_order_enabled: true,
            second_order_enabled: true,
            fallback_prior_fill_enabled: true,
            fanout_is_upper_bound: false,
            governor_enabled: false,
            neural_speculator_enabled: false
          }
        else
          {
            variant: $variant,
            predictor_mode: "second-order-only",
            predict_fanout: 1,
            pipeline_depth: 3,
            first_order_enabled: false,
            second_order_enabled: true,
            fallback_prior_fill_enabled: false,
            fanout_is_upper_bound: true,
            governor_enabled: $governor_enabled,
            neural_speculator_enabled: false
          }
        end),
      governor_configuration: {
        enabled: $governor_enabled,
        precision_floor: 0.05,
        contention_weight: $contention_weight,
        base_threshold: $base_threshold,
        runtime_default_precision_alpha: 0.2,
        runtime_default_base_threshold: 0.02
      },
      governor_counters_by_run: $runs,
      governor_totals: {
        governor_admitted_candidates: $admitted_total,
        governor_rejected_candidates: $rejected_total,
        governor_total_decisions: ($admitted_total + $rejected_total),
        governor_foreground_inflight_final: 0
      },
      governor_score_diagnostics_by_run:
        [$runs[] | {
          run_index,
          reconcile: .governor_score_diagnostics_reconcile,
          diagnostics: .governor_score_diagnostics
        }],
      governor_score_diagnostics: $aggregate_diagnostics,
      governor_score_diagnostics_reconcile: true,
      foreground_accounting: {
        final_by_run: [0, 0],
        final_aggregate: 0
      },
      prefetch_counters: {
        prefetch_submitted: $prefetch_submitted,
        prefetch_completed: $prefetch_completed,
        prefetch_used: 0,
        prefetch_bytes: 0,
        useful_prefetch_bytes: 0,
        unused_prefetch_bytes_at_sample: 0,
        prefetch_dropped_concurrency: 0,
        prefetch_dropped_pool_starved: 0,
        prefetch_dropped_governor: $rejected_total,
        prefetch_dropped_bytes: 0
      },
      provenance: {
        git_commit_full: $git_commit_full,
        binary_sha256: $sha,
        model_hashes: {
          config_json_sha256: $sha,
          dense_manifest_sha256: $sha
        },
        tokenizer_identity: {
          path: "/mnt/localssd/qwen/tokenizer.json",
          sha256: $sha
        },
        target_host: {
          hostname: "qwen-host",
          logical_cpu_count: 32,
          requested_cpu_mask: "0-31",
          effective_cpu_mask: "0-31"
        },
        model_mount_identity: {
          target: "/mnt/localssd",
          fstype: "ext4",
          options: "rw,noatime"
        },
        cargo_features: ["avx512", "blas", "io_uring", "tokenizer"],
        prompt_hashes: {short_sha256: $sha}
      },
      decode_tps_mean: 4.2,
      ssd_bytes: 4096,
      demand_read_service_mean_seconds: 0.01,
      output_token_ids: [10, 20, 30],
      output_parity_within_variant: true,
      screening_collection_valid: true,
      performance_qualification_applicable: false,
      qualification_passed: false
    }
  ' > "$output"
}

render_summary() {
  jq -n --arg runner_git_commit_full "$TEST_GIT_COMMIT_FULL" \
    --slurpfile demand "$1" --slurpfile second "$2" \
    --slurpfile high "$3" --slurpfile mid "$4" --slurpfile low "$5" \
    -f "$FILTER" > "$6"
}

make_variant demand-only false 0.02 1.0 0 0 0 0 "$TEST_ROOT/demand.json"
make_variant second-only-f1 false 0.02 1.0 0 0 4 4 "$TEST_ROOT/second.json"
make_variant second-only-f1-governed-bt001-cw000 true 0.001 0.0 1 3 2 2 "$TEST_ROOT/high.json"
make_variant second-only-f1-governed-bt0005-cw000 true 0.0005 0.0 2 2 4 4 "$TEST_ROOT/mid.json"
make_variant second-only-f1-governed-bt00025-cw000 true 0.00025 0.0 3 1 6 6 "$TEST_ROOT/low.json"

render() {
  render_summary "$TEST_ROOT/demand.json" "$TEST_ROOT/second.json" \
    "${1:-$TEST_ROOT/high.json}" "${2:-$TEST_ROOT/mid.json}" \
    "${3:-$TEST_ROOT/low.json}" "$4"
}

render "" "" "" "$TEST_ROOT/valid.json"
jq -e '.screening_gates_passed and (.qualification_passed == false) and
  (.variants | length == 5) and .gates.governed_variants_admit_candidates and
  .gates.governed_variants_reject_candidates and
  (.monotonicity_observations.structural_gate == false)' "$TEST_ROOT/valid.json" >/dev/null

# Wrong order.
render_summary "$TEST_ROOT/second.json" "$TEST_ROOT/demand.json" \
  "$TEST_ROOT/high.json" "$TEST_ROOT/mid.json" "$TEST_ROOT/low.json" "$TEST_ROOT/order.json"
jq -e '.gates.exact_variant_order == false and .screening_gates_passed == false' "$TEST_ROOT/order.json" >/dev/null

assert_gate_fails() {
  local mutation=$1 gate=$2 label=$3 source=${4:-$TEST_ROOT/high.json}
  jq "$mutation" "$source" > "$TEST_ROOT/$label.json"
  render "$TEST_ROOT/$label.json" "" "" "$TEST_ROOT/$label-summary.json"
  jq -e --arg gate "$gate" '.gates[$gate] == false and .screening_gates_passed == false and .qualification_passed == false' \
    "$TEST_ROOT/$label-summary.json" >/dev/null
}

assert_gate_fails '.governor_configuration.base_threshold = 0.01' expected_governor_configuration wrong-threshold
assert_gate_fails '.metadata.governor_enabled = false' expected_predictor_configuration wrong-assignment
assert_gate_fails '.governor_totals.governor_total_decisions += 1' direct_counters_reconcile counter-mismatch
assert_gate_fails '.governor_score_diagnostics.total_decisions += 1' diagnostic_decision_counters_reconcile diagnostic-mismatch
assert_gate_fails '.governor_totals.governor_admitted_candidates = 0' governed_variants_admit_candidates zero-admissions
assert_gate_fails '.governor_totals.governor_rejected_candidates = 0' governed_variants_reject_candidates zero-rejections
assert_gate_fails '.provenance.binary_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"' cross_variant_provenance_identical provenance
assert_gate_fails '.output_token_ids[0] = 99' deterministic_output_parity output-token
assert_gate_fails '.governor_score_diagnostics.candidate_score.mean = "NaN"' all_statistics_finite_or_explicitly_null invalid-statistic
assert_gate_fails '.performance_qualification_applicable = true | .qualification_passed = true' screening_mode_is_non_qualifying accidental-qualification

expected_specs='  "demand-only|0.05|0.02|1.0"
  "second-only-f1|0.05|0.02|1.0"
  "second-only-f1-governed-bt001-cw000|0.05|0.001|0.0"
  "second-only-f1-governed-bt0005-cw000|0.05|0.0005|0.0"
  "second-only-f1-governed-bt00025-cw000|0.05|0.00025|0.0"'
actual_specs=$(sed -n '/^variant_specs=($/,/^)/p' "$RUNNER" | sed '1d;$d')
test "$actual_specs" = "$expected_specs"

phase4d_c_plan=$(sed -n '/^# BEGIN PHASE4D_C_SPARSE_ADMISSION_COLLECTION$/,/^# END PHASE4D_C_SPARSE_ADMISSION_COLLECTION$/p' "$COLLECTOR")
grep -Fx 'elif [[ "$COLLECTOR_MODE" == phase4d-c-sparse-admission ]]; then' <<<"$phase4d_c_plan" >/dev/null
grep -F 'run_case 1536 short 14 "$SHORT_PROMPT_SHA"' <<<"$phase4d_c_plan" >/dev/null
! grep -E 'run_case (1536 medium|6144)' <<<"$phase4d_c_plan" >/dev/null
grep -F 'phase4d-c-sparse-admission-variant-summary.json' <<<"$phase4d_c_plan" >/dev/null

echo "Phase 4D-C sparse admission fixtures: PASS"
