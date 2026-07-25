def ids($report): $report[0].runs[0].output_token_ids;
def case($summary; $prompt):
  $summary.cases[] | select(.prompt_fixture == $prompt);
def gm($summary):
  ((case($summary; "short").decode_tps_mean *
    case($summary; "medium").decode_tps_mean) | sqrt);
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
      speculative_reads_completed:
        (case($candidate; "short").phase4a_prefetch.counters.prefetch_completed +
         case($candidate; "medium").phase4a_prefetch.counters.prefetch_completed)
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
        ((case($candidate; "short").phase4a_prefetch.counters.prefetch_completed +
          case($candidate; "medium").phase4a_prefetch.counters.prefetch_completed) <
         (case($current[0]; "short").phase4a_prefetch.counters.prefetch_completed +
          case($current[0]; "medium").phase4a_prefetch.counters.prefetch_completed))
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
  variants: [$demand[0], $current[0], $second_f2[0], $second_f1[0]],
  gates: {
    all_variant_qualifications_passed:
      ([$demand[0], $current[0], $second_f2[0], $second_f1[0]] |
       all(.qualification_passed == true)),
    cross_variant_short_output_parity:
      ([ids($demand_short), ids($current_short), ids($second_f2_short), ids($second_f1_short)] |
       unique | length == 1),
    cross_variant_medium_output_parity:
      ([ids($demand_medium), ids($current_medium), ids($second_f2_medium), ids($second_f1_medium)] |
       unique | length == 1)
  },
  candidate_promotion_evidence: [
    candidate_gates($second_f2[0]; "second-only-f2"),
    candidate_gates($second_f1[0]; "second-only-f1")
  ]
}
