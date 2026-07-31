def ids($report): $report[0].runs[0].output_token_ids;
def case($summary; $prompt):
  $summary.cases[] | select(.prompt_fixture == $prompt);
def gm($summary):
  ((case($summary; "short").decode_tps_mean *
    case($summary; "medium").decode_tps_mean) | sqrt);
def prefetch_completed($summary):
  ([case($summary; "short"), case($summary; "medium")] |
   map(.phase4a_prefetch.counters.prefetch_completed) |
   add);
def governor_total($summary; $field):
  ($summary.governor_counters_by_case |
   map(.totals[$field]) |
   add);
def comparable_metadata($summary):
  $summary.metadata | del(.variant, .governor_enabled);
def prompt_comparison($control; $candidate; $prompt):
  {
    prompt_fixture: $prompt,
    decode_tps: {
      control: case($control; $prompt).decode_tps_mean,
      candidate: case($candidate; $prompt).decode_tps_mean,
      delta:
        (case($candidate; $prompt).decode_tps_mean -
         case($control; $prompt).decode_tps_mean),
      ratio:
        (case($candidate; $prompt).decode_tps_mean /
         case($control; $prompt).decode_tps_mean)
    },
    ssd_bytes: {
      control: case($control; $prompt).ssd_bytes_total,
      candidate: case($candidate; $prompt).ssd_bytes_total,
      ratio:
        (case($candidate; $prompt).ssd_bytes_total /
         case($control; $prompt).ssd_bytes_total)
    },
    demand_read_service_seconds: {
      control:
        case($control; $prompt).phase3a_decode.physical_read_issue_to_completion_mean_seconds,
      candidate:
        case($candidate; $prompt).phase3a_decode.physical_read_issue_to_completion_mean_seconds,
      ratio:
        (case($candidate; $prompt).phase3a_decode.physical_read_issue_to_completion_mean_seconds /
         case($control; $prompt).phase3a_decode.physical_read_issue_to_completion_mean_seconds)
    },
    speculative_reads_completed: {
      control: case($control; $prompt).phase4a_prefetch.counters.prefetch_completed,
      candidate: case($candidate; $prompt).phase4a_prefetch.counters.prefetch_completed
    },
    governor_counters_by_run: {
      control: case($control; $prompt).phase4c_governor.counters_by_run,
      candidate: case($candidate; $prompt).phase4c_governor.counters_by_run
    }
  };
def causal_comparison($name; $control; $candidate; $change):
  {
    name: $name,
    control_variant: $control.variant,
    candidate_variant: $candidate.variant,
    isolated_change: $change,
    control_metadata: $control.metadata,
    candidate_metadata: $candidate.metadata,
    prompts: [
      prompt_comparison($control; $candidate; "short"),
      prompt_comparison($control; $candidate; "medium")
    ],
    combined_governor_counters: {
      control: {
        candidates_rejected_by_governor:
          governor_total($control; "candidates_rejected_by_governor"),
        speculative_work_admitted:
          governor_total($control; "speculative_work_admitted"),
        speculative_work_dropped_by_concurrency:
          governor_total($control; "speculative_work_dropped_by_concurrency"),
        speculative_work_dropped_by_pool_pressure:
          governor_total($control; "speculative_work_dropped_by_pool_pressure"),
        demand_reads_observed_while_speculation_active:
          governor_total($control; "demand_reads_observed_while_speculation_active")
      },
      candidate: {
        candidates_rejected_by_governor:
          governor_total($candidate; "candidates_rejected_by_governor"),
        speculative_work_admitted:
          governor_total($candidate; "speculative_work_admitted"),
        speculative_work_dropped_by_concurrency:
          governor_total($candidate; "speculative_work_dropped_by_concurrency"),
        speculative_work_dropped_by_pool_pressure:
          governor_total($candidate; "speculative_work_dropped_by_pool_pressure"),
        demand_reads_observed_while_speculation_active:
          governor_total($candidate; "demand_reads_observed_while_speculation_active")
      }
    }
  };
def candidate_gates($candidate; $name):
  {
    variant: $name,
    metrics: {
      short_decode_tps: case($candidate; "short").decode_tps_mean,
      medium_decode_tps: case($candidate; "medium").decode_tps_mean,
      decode_tps_geometric_mean: gm($candidate),
      geometric_mean_ratio_vs_demand_only: (gm($candidate) / gm($demand[0])),
      short_ssd_bytes_ratio_vs_demand_only:
        (case($candidate; "short").ssd_bytes_total /
         case($demand[0]; "short").ssd_bytes_total),
      medium_ssd_bytes_ratio_vs_demand_only:
        (case($candidate; "medium").ssd_bytes_total /
         case($demand[0]; "medium").ssd_bytes_total),
      short_demand_read_service_ratio_vs_demand_only:
        (case($candidate; "short").phase3a_decode.physical_read_issue_to_completion_mean_seconds /
         case($demand[0]; "short").phase3a_decode.physical_read_issue_to_completion_mean_seconds),
      medium_demand_read_service_ratio_vs_demand_only:
        (case($candidate; "medium").phase3a_decode.physical_read_issue_to_completion_mean_seconds /
         case($demand[0]; "medium").phase3a_decode.physical_read_issue_to_completion_mean_seconds),
      speculative_reads_completed: prefetch_completed($candidate)
    },
    resolved_primary_gates: {
      beats_current_f2_on_short:
        (case($candidate; "short").decode_tps_mean >
         case($current[0]; "short").decode_tps_mean),
      beats_current_f2_on_medium:
        (case($candidate; "medium").decode_tps_mean >
         case($current[0]; "medium").decode_tps_mean),
      recovers_demand_only_on_short:
        (case($candidate; "short").decode_tps_mean >=
         case($demand[0]; "short").decode_tps_mean),
      recovers_demand_only_on_medium:
        (case($candidate; "medium").decode_tps_mean >=
         case($demand[0]; "medium").decode_tps_mean),
      geometric_mean_at_least_three_percent_over_demand_only:
        (gm($candidate) >= 1.03 * gm($demand[0])),
      reduces_speculative_reads_vs_current_f2:
        (prefetch_completed($candidate) < prefetch_completed($current[0]))
    },
    thresholded_gates_requiring_experiment_judgment: {
      ssd_bytes_not_materially_higher_than_demand_only: null,
      demand_read_service_not_materially_worse_than_demand_only: null
    },
    diagnostic_gates_pending: [
      "published lifecycle first-use efficiency",
      "fallback-prior physical prefetches",
      "first-order physical prefetches",
      "ready-before-lookup precision",
      "final-straggler and ordinary-demand blocking"
    ]
  };
{
  schema: {name:"mer-prompt2-phase4c-matrix-summary", version:1},
  traced: false,
  cache_slots: 1536,
  prompt_fixtures: ["short", "medium"],
  expected_variant_order: [
    "demand-only",
    "current-f2",
    "current-f2-governed",
    "second-only-f2",
    "second-only-f1",
    "second-only-f1-governed"
  ],
  variants: [
    $demand[0],
    $current[0],
    $current_governed[0],
    $second_f2[0],
    $second_f1[0],
    $second_f1_governed[0]
  ],
  gates: {
    exact_six_variant_matrix:
      ([
        $demand[0].variant,
        $current[0].variant,
        $current_governed[0].variant,
        $second_f2[0].variant,
        $second_f1[0].variant,
        $second_f1_governed[0].variant
      ] == [
        "demand-only",
        "current-f2",
        "current-f2-governed",
        "second-only-f2",
        "second-only-f1",
        "second-only-f1-governed"
      ]),
    all_variant_qualifications_passed:
      ([
        $demand[0],
        $current[0],
        $current_governed[0],
        $second_f2[0],
        $second_f1[0],
        $second_f1_governed[0]
      ] | all(.qualification_passed == true)),
    neural_speculator_disabled_everywhere:
      ([
        $demand[0],
        $current[0],
        $current_governed[0],
        $second_f2[0],
        $second_f1[0],
        $second_f1_governed[0]
      ] | all(.metadata.neural_speculator_enabled == false)),
    governor_assignment_exact:
      ($demand[0].metadata.governor_enabled == false and
       $current[0].metadata.governor_enabled == false and
       $current_governed[0].metadata.governor_enabled == true and
       $second_f2[0].metadata.governor_enabled == false and
       $second_f1[0].metadata.governor_enabled == false and
       $second_f1_governed[0].metadata.governor_enabled == true),
    governed_cases_share_configuration:
      ($current_governed[0].governor_configuration ==
       $second_f1_governed[0].governor_configuration),
    current_governor_pair_differs_only_by_governor:
      (comparable_metadata($current[0]) ==
       comparable_metadata($current_governed[0])),
    second_f1_governor_pair_differs_only_by_governor:
      (comparable_metadata($second_f1[0]) ==
       comparable_metadata($second_f1_governed[0])),
    cross_variant_short_output_parity:
      ([
        ids($demand_short),
        ids($current_short),
        ids($current_governed_short),
        ids($second_f2_short),
        ids($second_f1_short),
        ids($second_f1_governed_short)
      ] | unique | length == 1),
    cross_variant_medium_output_parity:
      ([
        ids($demand_medium),
        ids($current_medium),
        ids($current_governed_medium),
        ids($second_f2_medium),
        ids($second_f1_medium),
        ids($second_f1_governed_medium)
      ] | unique | length == 1)
  },
  causal_comparisons: [
    causal_comparison(
      "current-f2_vs_current-f2-governed";
      $current[0];
      $current_governed[0];
      "governor-only"
    ),
    causal_comparison(
      "second-only-f1_vs_second-only-f1-governed";
      $second_f1[0];
      $second_f1_governed[0];
      "governor-only"
    ),
    causal_comparison(
      "current-f2_vs_second-only-f1";
      $current[0];
      $second_f1[0];
      "predictor-admission-and-fanout"
    ),
    causal_comparison(
      "current-f2-governed_vs_second-only-f1-governed";
      $current_governed[0];
      $second_f1_governed[0];
      "predictor-admission-and-fanout-under-shared-governor"
    )
  ],
  candidate_promotion_evidence: [
    candidate_gates($second_f2[0]; "second-only-f2"),
    candidate_gates($second_f1[0]; "second-only-f1"),
    candidate_gates($second_f1_governed[0]; "second-only-f1-governed")
  ]
}
