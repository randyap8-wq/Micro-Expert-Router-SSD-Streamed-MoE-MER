def absolute:
  if . < 0 then - . else . end;

def percent_delta($before; $after):
  if $before == 0 then null else (100 * ($after / $before - 1)) end;

def metric_delta($before; $after):
  if $before == null or $after == null then
    {phase3a: $before, phase3b: $after, absolute_delta: null, percent_delta: null}
  else
    {
      phase3a: $before,
      phase3b: $after,
      absolute_delta: ($after - $before),
      percent_delta: percent_delta($before; $after)
    }
  end;

def case_for($summary; $slots; $fixture):
  first($summary.cases[] | select(.cache_slots == $slots and .prompt_fixture == $fixture));

def histogram_quantile_upper_us($histogram; $quantile):
  ($histogram | map(.count) | add) as $total |
  if $total == 0 then null
  else
    (($total * $quantile) | ceil) as $rank |
    reduce $histogram[] as $bucket
      ({seen: 0, value: null};
       if .value != null then .
       else
         .seen += $bucket.count |
         if .seen >= $rank then .value = $bucket.upper_bound_microseconds else . end
       end) |
    .value
  end;

def streaming_case($before; $after):
  {
    decode_tps: metric_delta($before.decode_tps_mean; $after.decode_tps_mean),
    cache_hit_rate: metric_delta($before.cache_hit_rate; $after.cache_hit_rate),
    cache_misses: metric_delta($before.cache_misses_total; $after.cache_misses_total),
    ssd_bytes: metric_delta($before.ssd_bytes_total; $after.ssd_bytes_total),
    demand_storage_mean_seconds: metric_delta(
      $before.phase3a_decode.physical_read_issue_to_completion_mean_seconds;
      $after.phase3a_decode.physical_read_issue_to_completion_mean_seconds
    ),
    demand_storage_max_seconds: metric_delta(
      $before.phase3a_decode.physical_read_issue_to_completion_max_seconds;
      $after.phase3a_decode.physical_read_issue_to_completion_max_seconds
    ),
    demand_storage_p95_histogram_upper_microseconds: metric_delta(
      histogram_quantile_upper_us($before.phase3a_decode.physical_read_issue_to_completion_histogram; 0.95);
      histogram_quantile_upper_us($after.phase3a_decode.physical_read_issue_to_completion_histogram; 0.95)
    ),
    layer_fetch_critical_path_mean_seconds: metric_delta(
      $before.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds;
      $after.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds
    ),
    layer_fetch_critical_path_max_seconds: metric_delta(
      $before.phase3a_decode.layer_expert_fetch_critical_path_max_seconds;
      $after.phase3a_decode.layer_expert_fetch_critical_path_max_seconds
    ),
    decode_wall_fetch_fraction: metric_delta(
      $before.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch;
      $after.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch
    ),
    phase3b_demand_service_split: {
      without_speculation_operations: $after.phase3b_decode.demand_physical_read_service_without_speculation_operations,
      without_speculation_mean_seconds: $after.phase3b_decode.demand_physical_read_service_without_speculation_mean_seconds,
      without_speculation_max_seconds: $after.phase3b_decode.demand_physical_read_service_without_speculation_max_seconds,
      with_speculation_operations: $after.phase3b_decode.demand_physical_read_service_with_speculation_operations,
      with_speculation_mean_seconds: $after.phase3b_decode.demand_physical_read_service_with_speculation_mean_seconds,
      with_speculation_max_seconds: $after.phase3b_decode.demand_physical_read_service_with_speculation_max_seconds
    }
  };

def sum_numbers($values):
  if ($values | all(type == "number")) then ($values | add) else null end;

def arbitration_totals($short; $medium; $phase):
  [$short[$phase], $medium[$phase]] as $samples |
  (sum_numbers([$samples[].speculative_physical_reads_deferred_for_demand_pressure])) as $deferred |
  (sum_numbers([$samples[].deferred_speculative_physical_reads_resumed])) as $resumed |
  (sum_numbers([$samples[].deferred_speculative_physical_reads_dropped_stale_duplicate_or_cache_hit])) as $dropped |
  (if $resumed == null or $dropped == null then null else ($resumed + $dropped) end) as $classified |
  {
    deferred: $deferred,
    resumed: $resumed,
    dropped_stale_duplicate_or_cache_hit: $dropped,
    classified_by_phase3b_terminal_counters: $classified,
    not_classified_by_phase3b_terminal_counters_at_snapshot: (
      if $deferred == null or $classified == null then null else ($deferred - $classified) end
    ),
    fully_classified_at_snapshot: (
      $deferred != null and $classified != null and $deferred == $classified
    ),
    admitted_without_demand_pressure: sum_numbers([$samples[].speculative_physical_reads_admitted_without_demand_pressure]),
    already_active_when_demand_burst_began: sum_numbers([$samples[].speculative_physical_reads_active_when_demand_burst_began]),
    final_stragglers_issued_with_speculation_active: sum_numbers([$samples[].demand_layers_final_straggler_issued_while_speculative_reads_active]),
    causal_delay_claim: null,
    semantics: {
      deferred: "logical speculative operations at their first demand-pressure encounter",
      resumed: "previously deferred operations later admitted to physical storage service; not read completions",
      dropped_stale_duplicate_or_cache_hit: "previously deferred operations made unnecessary by a later cache or singleflight recheck",
      not_classified: "snapshot difference only; may include detached work still pending or exits through existing governor, concurrency, or pool-starvation counters, and is not evidence of leaked work"
    }
  };

def all_negative($values):
  ($values | length) > 0 and ($values | all(type == "number" and . < 0));

def all_nonpositive($values):
  ($values | length) > 0 and ($values | all(type == "number" and . <= 0));

($phase3a_stream[0]) as $a_stream |
($phase3b_stream[0]) as $b_stream |
($phase3a_resident[0]) as $a_resident |
($phase3b_resident[0]) as $b_resident |
(case_for($a_stream; 1536; "short")) as $a_short |
(case_for($a_stream; 1536; "medium")) as $a_medium |
(case_for($b_stream; 1536; "short")) as $b_short |
(case_for($b_stream; 1536; "medium")) as $b_medium |
(case_for($a_resident; 6144; "short")) as $a_resident_short |
(case_for($a_resident; 6144; "medium")) as $a_resident_medium |
(case_for($b_resident; 6144; "short")) as $b_resident_short |
(case_for($b_resident; 6144; "medium")) as $b_resident_medium |
(($a_short.decode_tps_mean * $a_medium.decode_tps_mean) | sqrt) as $a_stream_gm |
(($b_short.decode_tps_mean * $b_medium.decode_tps_mean) | sqrt) as $b_stream_gm |
(($a_resident_short.decode_tps_mean * $a_resident_medium.decode_tps_mean) | sqrt) as $a_resident_gm |
(($b_resident_short.decode_tps_mean * $b_resident_medium.decode_tps_mean) | sqrt) as $b_resident_gm |
(streaming_case($a_short; $b_short)) as $short_stream |
(streaming_case($a_medium; $b_medium)) as $medium_stream |
(metric_delta($a_stream_gm; $b_stream_gm)) as $stream_gm |
(metric_delta($a_resident_gm; $b_resident_gm)) as $resident_gm |
(arbitration_totals($b_short; $b_medium; "phase3b_prompt")) as $prompt_arbitration |
(arbitration_totals($b_short; $b_medium; "phase3b_decode")) as $decode_arbitration |
(
  $phase3a_streaming_hostname != "" and
  $phase3a_streaming_hostname == $phase3a_resident_hostname and
  $phase3a_streaming_hostname == $phase3b_streaming_hostname and
  $phase3a_streaming_hostname == $phase3b_resident_hostname
) as $same_host_passed |
(
  $a_stream.qualification_passed == true and
  $b_stream.qualification_passed == true
) as $streaming_qualifications_passed |
(
  $a_resident.qualification_passed == true and
  $b_resident.qualification_passed == true
) as $resident_qualifications_passed |
([$a_stream.cases[].output_token_parity] | all(. == true)) as $a_stream_parity |
([$b_stream.cases[].output_token_parity] | all(. == true)) as $b_stream_parity |
($a_resident.resident_gates.output_token_parity_passed == true) as $a_resident_parity |
($b_resident.resident_gates.output_token_parity_passed == true) as $b_resident_parity |
(
  $a_stream_parity and $b_stream_parity and
  $a_resident_parity and $b_resident_parity
) as $output_parity_passed |
(
  $resident_gm.percent_delta != null and
  (($resident_gm.percent_delta | absolute) <= 2)
) as $resident_within_two_percent |
(
  $short_stream.decode_tps.percent_delta != null and
  $short_stream.decode_tps.percent_delta >= 0
) as $short_no_regression |
(
  $medium_stream.decode_tps.percent_delta != null and
  $medium_stream.decode_tps.percent_delta >= 0
) as $medium_no_regression |
(
  $stream_gm.percent_delta != null and
  $stream_gm.percent_delta >= 2
) as $preferred_threshold_reached |
(all_negative([
  $short_stream.demand_storage_mean_seconds.percent_delta,
  $medium_stream.demand_storage_mean_seconds.percent_delta
])) as $storage_mean_improved |
(all_negative([
  $short_stream.demand_storage_max_seconds.percent_delta,
  $medium_stream.demand_storage_max_seconds.percent_delta
])) as $storage_max_improved |
(all_negative([
  $short_stream.layer_fetch_critical_path_mean_seconds.percent_delta,
  $medium_stream.layer_fetch_critical_path_mean_seconds.percent_delta
])) as $critical_mean_improved |
(all_negative([
  $short_stream.layer_fetch_critical_path_max_seconds.percent_delta,
  $medium_stream.layer_fetch_critical_path_max_seconds.percent_delta
])) as $critical_max_improved |
(
  ($storage_mean_improved and $storage_max_improved) or
  ($critical_mean_improved and $critical_max_improved)
) as $latency_evidence_passed |
(all_nonpositive([
  $short_stream.decode_wall_fetch_fraction.percent_delta,
  $medium_stream.decode_wall_fetch_fraction.percent_delta
])) as $fetch_fraction_not_worse |
(all_nonpositive([
  $short_stream.cache_misses.percent_delta,
  $medium_stream.cache_misses.percent_delta
])) as $cache_misses_not_increased |
(
  $prompt_arbitration.fully_classified_at_snapshot and
  $decode_arbitration.fully_classified_at_snapshot
) as $deferrals_fully_classified |
(
  $same_host_passed and
  $streaming_qualifications_passed and
  $resident_qualifications_passed and
  $output_parity_passed and
  $resident_within_two_percent and
  $short_no_regression and
  $medium_no_regression and
  $preferred_threshold_reached and
  $latency_evidence_passed and
  $fetch_fraction_not_worse and
  $cache_misses_not_increased and
  $deferrals_fully_classified
) as $policy_accepted |
{
  schema: {name: "mer-prompt2-phase3-same-host-comparison", version: 2},
  sources: {
    phase3a_artifact_dir: $phase3a_artifact_dir,
    phase3b_artifact_dir: $phase3b_artifact_dir,
    source_artifacts_modified: false
  },
  same_host: {
    phase3a_streaming_hostname: $phase3a_streaming_hostname,
    phase3a_resident_hostname: $phase3a_resident_hostname,
    phase3b_streaming_hostname: $phase3b_streaming_hostname,
    phase3b_resident_hostname: $phase3b_resident_hostname,
    hostname_matches: $same_host_passed
  },
  streaming_1536: {
    short: $short_stream,
    medium: $medium_stream,
    geometric_mean_decode_tps: $stream_gm
  },
  resident_6144: {
    short_decode_tps: metric_delta($a_resident_short.decode_tps_mean; $b_resident_short.decode_tps_mean),
    medium_decode_tps: metric_delta($a_resident_medium.decode_tps_mean; $b_resident_medium.decode_tps_mean),
    geometric_mean_decode_tps: $resident_gm,
    contemporaneous_within_two_percent: $resident_within_two_percent,
    historical_reference_gate_phase3a: $a_resident.performance_gate,
    historical_reference_gate_phase3b: $b_resident.performance_gate
  },
  speculative_arbitration_prompt_1536: $prompt_arbitration,
  speculative_arbitration_decode_1536: $decode_arbitration,
  correctness_and_qualification: {
    phase3a_streaming_output_parity: $a_stream_parity,
    phase3b_streaming_output_parity: $b_stream_parity,
    phase3a_resident_output_parity: $a_resident_parity,
    phase3b_resident_output_parity: $b_resident_parity,
    phase3a_streaming_qualification_passed: $a_stream.qualification_passed,
    phase3b_streaming_qualification_passed: $b_stream.qualification_passed,
    phase3a_resident_qualification_passed: $a_resident.qualification_passed,
    phase3b_resident_qualification_passed: $b_resident.qualification_passed
  },
  acceptance: {
    same_host_identity_passed: $same_host_passed,
    streaming_qualifications: {
      phase3a_passed: ($a_stream.qualification_passed == true),
      phase3b_passed: ($b_stream.qualification_passed == true),
      all_passed: $streaming_qualifications_passed
    },
    resident_qualifications: {
      phase3a_passed: ($a_resident.qualification_passed == true),
      phase3b_passed: ($b_resident.qualification_passed == true),
      all_passed: $resident_qualifications_passed
    },
    output_parity: {
      phase3a_streaming_passed: $a_stream_parity,
      phase3b_streaming_passed: $b_stream_parity,
      phase3a_resident_passed: $a_resident_parity,
      phase3b_resident_passed: $b_resident_parity,
      all_passed: $output_parity_passed
    },
    resident_contemporaneous: {
      geometric_mean_percent_delta: $resident_gm.percent_delta,
      within_plus_or_minus_two_percent: $resident_within_two_percent
    },
    throughput_regression_status: {
      short: {
        percent_delta: $short_stream.decode_tps.percent_delta,
        regressed: ($short_no_regression | not)
      },
      medium: {
        percent_delta: $medium_stream.decode_tps.percent_delta,
        regressed: ($medium_no_regression | not)
      }
    },
    streaming_geometric_mean: {
      percent_delta: $stream_gm.percent_delta,
      preferred_threshold_percent: 2,
      preferred_threshold_reached: $preferred_threshold_reached
    },
    demand_storage_changes: {
      short: {
        mean_percent_delta: $short_stream.demand_storage_mean_seconds.percent_delta,
        maximum_percent_delta: $short_stream.demand_storage_max_seconds.percent_delta
      },
      medium: {
        mean_percent_delta: $medium_stream.demand_storage_mean_seconds.percent_delta,
        maximum_percent_delta: $medium_stream.demand_storage_max_seconds.percent_delta
      },
      mean_improved_in_both_fixtures: $storage_mean_improved,
      maximum_improved_in_both_fixtures: $storage_max_improved
    },
    layer_fetch_critical_path_changes: {
      short: {
        mean_percent_delta: $short_stream.layer_fetch_critical_path_mean_seconds.percent_delta,
        maximum_percent_delta: $short_stream.layer_fetch_critical_path_max_seconds.percent_delta
      },
      medium: {
        mean_percent_delta: $medium_stream.layer_fetch_critical_path_mean_seconds.percent_delta,
        maximum_percent_delta: $medium_stream.layer_fetch_critical_path_max_seconds.percent_delta
      },
      mean_improved_in_both_fixtures: $critical_mean_improved,
      maximum_improved_in_both_fixtures: $critical_max_improved
    },
    decode_wall_fetch_fraction_changes: {
      short_percent_delta: $short_stream.decode_wall_fetch_fraction.percent_delta,
      medium_percent_delta: $medium_stream.decode_wall_fetch_fraction.percent_delta,
      did_not_worsen: $fetch_fraction_not_worse
    },
    cache_miss_changes: {
      short_percent_delta: $short_stream.cache_misses.percent_delta,
      medium_percent_delta: $medium_stream.cache_misses.percent_delta,
      did_not_increase: $cache_misses_not_increased
    },
    ssd_byte_changes: {
      short_percent_delta: $short_stream.ssd_bytes.percent_delta,
      medium_percent_delta: $medium_stream.ssd_bytes.percent_delta
    },
    deferred_resumed_drop_accounting: {
      prompt: $prompt_arbitration,
      decode: $decode_arbitration,
      all_deferred_units_classified_at_snapshot: $deferrals_fully_classified
    },
    consistent_latency_improvement_passed: $latency_evidence_passed,
    policy_accepted: $policy_accepted,
    rejection_reasons: [
      if $same_host_passed then empty else "same-host identity did not pass" end,
      if $streaming_qualifications_passed then empty else "one or more streaming qualifications did not pass" end,
      if $resident_qualifications_passed then empty else "one or more resident qualifications did not pass" end,
      if $output_parity_passed then empty else "output parity did not pass in every streaming and resident comparison" end,
      if $resident_within_two_percent then empty else "resident geometric-mean delta exceeded plus or minus 2 percent" end,
      if $short_no_regression then empty else "short streaming throughput regressed" end,
      if $medium_no_regression then empty else "medium streaming throughput regressed" end,
      if $preferred_threshold_reached then empty else "streaming geometric-mean improvement did not reach the preferred plus 2 percent threshold" end,
      if $latency_evidence_passed then empty else "neither demand-storage latency nor layer-fetch critical path improved consistently in both fixtures" end,
      if $fetch_fraction_not_worse then empty else "decode-wall fetch fraction worsened in at least one fixture" end,
      if $cache_misses_not_increased then empty else "cache misses increased in at least one fixture" end,
      if $deferrals_fully_classified then empty else "deferred speculative operations were not fully classified by the Phase 3B terminal counters at snapshot time" end
    ]
  }
}
