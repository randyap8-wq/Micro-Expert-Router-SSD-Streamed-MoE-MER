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

def arbitration_totals($short; $medium; $phase):
  [$short[$phase], $medium[$phase]] as $samples |
  {
    deferred: ([$samples[].speculative_physical_reads_deferred_for_demand_pressure] | add),
    resumed: ([$samples[].deferred_speculative_physical_reads_resumed] | add),
    dropped_stale_duplicate_or_cache_hit: ([$samples[].deferred_speculative_physical_reads_dropped_stale_duplicate_or_cache_hit] | add),
    admitted_without_demand_pressure: ([$samples[].speculative_physical_reads_admitted_without_demand_pressure] | add),
    already_active_when_demand_burst_began: ([$samples[].speculative_physical_reads_active_when_demand_burst_began] | add),
    final_stragglers_issued_with_speculation_active: ([$samples[].demand_layers_final_straggler_issued_while_speculative_reads_active] | add),
    causal_delay_claim: null
  };

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
{
  schema: {name: "mer-prompt2-phase3-same-host-comparison", version: 1},
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
    hostname_matches: (
      $phase3a_streaming_hostname != "" and
      $phase3a_streaming_hostname == $phase3a_resident_hostname and
      $phase3a_streaming_hostname == $phase3b_streaming_hostname and
      $phase3a_streaming_hostname == $phase3b_resident_hostname
    )
  },
  streaming_1536: {
    short: streaming_case($a_short; $b_short),
    medium: streaming_case($a_medium; $b_medium),
    geometric_mean_decode_tps: metric_delta($a_stream_gm; $b_stream_gm)
  },
  resident_6144: {
    short_decode_tps: metric_delta($a_resident_short.decode_tps_mean; $b_resident_short.decode_tps_mean),
    medium_decode_tps: metric_delta($a_resident_medium.decode_tps_mean; $b_resident_medium.decode_tps_mean),
    geometric_mean_decode_tps: metric_delta($a_resident_gm; $b_resident_gm),
    contemporaneous_within_two_percent: ((percent_delta($a_resident_gm; $b_resident_gm) | abs) <= 2),
    historical_reference_gate_phase3a: $a_resident.performance_gate,
    historical_reference_gate_phase3b: $b_resident.performance_gate
  },
  speculative_arbitration_prompt_1536: arbitration_totals($b_short; $b_medium; "phase3b_prompt"),
  speculative_arbitration_decode_1536: arbitration_totals($b_short; $b_medium; "phase3b_decode"),
  correctness_and_qualification: {
    phase3a_streaming_output_parity: ([$a_stream.cases[].output_token_parity] | all(. == true)),
    phase3b_streaming_output_parity: ([$b_stream.cases[].output_token_parity] | all(. == true)),
    phase3a_resident_output_parity: $a_resident.resident_gates.output_token_parity_passed,
    phase3b_resident_output_parity: $b_resident.resident_gates.output_token_parity_passed,
    phase3a_streaming_qualification_passed: $a_stream.qualification_passed,
    phase3b_streaming_qualification_passed: $b_stream.qualification_passed,
    phase3a_resident_qualification_passed: $a_resident.qualification_passed,
    phase3b_resident_qualification_passed: $b_resident.qualification_passed
  }
}
