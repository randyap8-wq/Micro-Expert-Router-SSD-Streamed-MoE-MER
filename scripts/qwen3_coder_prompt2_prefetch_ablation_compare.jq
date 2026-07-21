def absolute:
  if . < 0 then - . else . end;

def percent_delta($before; $after):
  if $before == null or $after == null or $before == 0 then
    null
  else
    (100 * ($after / $before - 1))
  end;

def arithmetic_mean($first; $second):
  if $first == null or $second == null then null else (($first + $second) / 2) end;

def metric_vs_control_mean($a1; $a2; $b):
  arithmetic_mean($a1; $a2) as $control_mean |
  {
    control_a1: $a1,
    control_a2: $a2,
    control_mean: $control_mean,
    no_prefetch_b: $b,
    absolute_delta: (if $control_mean == null or $b == null then null else ($b - $control_mean) end),
    percent_delta: percent_delta($control_mean; $b)
  };

def control_delta($a1; $a2):
  {
    control_a1: $a1,
    control_a2: $a2,
    absolute_delta: (if $a1 == null or $a2 == null then null else ($a2 - $a1) end),
    percent_delta: percent_delta($a1; $a2)
  };

def case_for($summary; $fixture):
  first($summary.cases[] | select(.cache_slots == 1536 and .prompt_fixture == $fixture));

def streaming_cases_qualified($summary):
  ([
    $summary.cases[] |
    select(.cache_slots == 1536 and (.prompt_fixture == "short" or .prompt_fixture == "medium"))
  ]) as $cases |
  ($cases | length) == 2 and
  ($cases | all(.qualification_passed == true));

def streaming_output_parity($summary):
  ([
    $summary.cases[] |
    select(.cache_slots == 1536 and (.prompt_fixture == "short" or .prompt_fixture == "medium"))
  ]) as $cases |
  ($cases | length) == 2 and
  ($cases | all(.output_token_parity == true));

def geometric_mean_tps($short; $medium):
  (($short.decode_tps_mean * $medium.decode_tps_mean) | sqrt);

def mount_options($provenance):
  ($provenance.model_mount_identity.options // "" | split(","));

def qualifying_cpu($provenance):
  $provenance.host.logical_cpu_count == 32 and
  $provenance.host.requested_cpu_mask == "0-31" and
  $provenance.host.effective_cpu_mask == "0-31";

def qualifying_storage($provenance):
  $provenance.model_mount_identity.target == "/mnt/localssd" and
  $provenance.model_mount_identity.fstype == "ext4" and
  ((["rw", "noatime", "nodiratime"] - mount_options($provenance)) | length == 0);

def expected_provenance_schema($provenance):
  $provenance.schema == {name:"mer-prompt2-phase4a-ablation-provenance", version:1} and
  $provenance.experiment_name == "prompt2-phase4a-prefetch-ablation" and
  $provenance.collector_mode == "four-case";

def expected_configuration($provenance; $fanout; $depth; $active):
  $provenance.predict_fanout == $fanout and
  $provenance.pipeline_depth == $depth and
  $provenance.prefetch_expected_active == $active;

def nonempty_string($value):
  ($value | type) == "string" and ($value | length) > 0;

def negative_number($value):
  ($value | type) == "number" and $value < 0;

def case_prefetch_disabled($case):
  $case.phase4a_prefetch.expected_active == false and
  $case.phase4a_prefetch.all_runs_reported_expected_enabled == true and
  $case.phase4a_prefetch.reported_enabled_values == [false];

def case_prefetch_counters_zero($case):
  $case.phase4a_prefetch.all_prefetch_counters_zero == true and
  ([$case.phase4a_prefetch.counters[]] | all(type == "number" and . == 0));

def sum_prefetch_counters($short; $medium):
  reduce ($short.phase4a_prefetch.counters | keys[]) as $key
    ({}; .[$key] = (
      ($short.phase4a_prefetch.counters[$key] // 0) +
      ($medium.phase4a_prefetch.counters[$key] // 0)
    ));

def case_control_stability($a1; $a2):
  {
    decode_tps: control_delta($a1.decode_tps_mean; $a2.decode_tps_mean),
    cache_misses: control_delta($a1.cache_misses_total; $a2.cache_misses_total),
    ssd_bytes: control_delta($a1.ssd_bytes_total; $a2.ssd_bytes_total),
    demand_storage_mean_seconds: control_delta(
      $a1.phase3a_decode.physical_read_issue_to_completion_mean_seconds;
      $a2.phase3a_decode.physical_read_issue_to_completion_mean_seconds
    ),
    demand_storage_max_seconds: control_delta(
      $a1.phase3a_decode.physical_read_issue_to_completion_max_seconds;
      $a2.phase3a_decode.physical_read_issue_to_completion_max_seconds
    ),
    layer_fetch_critical_path_mean_seconds: control_delta(
      $a1.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds;
      $a2.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds
    ),
    layer_fetch_critical_path_max_seconds: control_delta(
      $a1.phase3a_decode.layer_expert_fetch_critical_path_max_seconds;
      $a2.phase3a_decode.layer_expert_fetch_critical_path_max_seconds
    ),
    decode_wall_fetch_fraction: control_delta(
      $a1.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch;
      $a2.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch
    )
  };

def case_no_prefetch_comparison($a1; $a2; $b):
  {
    decode_tps: metric_vs_control_mean($a1.decode_tps_mean; $a2.decode_tps_mean; $b.decode_tps_mean),
    cache_hit_rate: (
      metric_vs_control_mean($a1.cache_hit_rate; $a2.cache_hit_rate; $b.cache_hit_rate) |
      . + {
        percentage_point_delta: (
          if .absolute_delta == null then null else (100 * .absolute_delta) end
        )
      }
    ),
    cache_misses: metric_vs_control_mean($a1.cache_misses_total; $a2.cache_misses_total; $b.cache_misses_total),
    ssd_bytes: metric_vs_control_mean($a1.ssd_bytes_total; $a2.ssd_bytes_total; $b.ssd_bytes_total),
    demand_storage_mean_seconds: metric_vs_control_mean(
      $a1.phase3a_decode.physical_read_issue_to_completion_mean_seconds;
      $a2.phase3a_decode.physical_read_issue_to_completion_mean_seconds;
      $b.phase3a_decode.physical_read_issue_to_completion_mean_seconds
    ),
    demand_storage_max_seconds: metric_vs_control_mean(
      $a1.phase3a_decode.physical_read_issue_to_completion_max_seconds;
      $a2.phase3a_decode.physical_read_issue_to_completion_max_seconds;
      $b.phase3a_decode.physical_read_issue_to_completion_max_seconds
    ),
    layer_fetch_critical_path_mean_seconds: metric_vs_control_mean(
      $a1.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds;
      $a2.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds;
      $b.phase3a_decode.layer_expert_fetch_critical_path_mean_seconds
    ),
    layer_fetch_critical_path_max_seconds: metric_vs_control_mean(
      $a1.phase3a_decode.layer_expert_fetch_critical_path_max_seconds;
      $a2.phase3a_decode.layer_expert_fetch_critical_path_max_seconds;
      $b.phase3a_decode.layer_expert_fetch_critical_path_max_seconds
    ),
    decode_wall_fetch_fraction: metric_vs_control_mean(
      $a1.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch;
      $a2.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch;
      $b.phase3a_decode.decode_wall_fraction_attributable_to_layer_expert_fetch
    )
  };

($control_a1_summary[0]) as $a1_summary |
($no_prefetch_b_summary[0]) as $b_summary |
($control_a2_summary[0]) as $a2_summary |
($control_a1_provenance[0]) as $a1_provenance |
($no_prefetch_b_provenance[0]) as $b_provenance |
($control_a2_provenance[0]) as $a2_provenance |
(case_for($a1_summary; "short")) as $a1_short |
(case_for($a1_summary; "medium")) as $a1_medium |
(case_for($b_summary; "short")) as $b_short |
(case_for($b_summary; "medium")) as $b_medium |
(case_for($a2_summary; "short")) as $a2_short |
(case_for($a2_summary; "medium")) as $a2_medium |
(geometric_mean_tps($a1_short; $a1_medium)) as $a1_gm |
(geometric_mean_tps($b_short; $b_medium)) as $b_gm |
(geometric_mean_tps($a2_short; $a2_medium)) as $a2_gm |
(arithmetic_mean($a1_gm; $a2_gm)) as $control_mean_gm |
(case_control_stability($a1_short; $a2_short)) as $short_control_stability |
(case_control_stability($a1_medium; $a2_medium)) as $medium_control_stability |
(case_no_prefetch_comparison($a1_short; $a2_short; $b_short)) as $short_comparison |
(case_no_prefetch_comparison($a1_medium; $a2_medium; $b_medium)) as $medium_comparison |
(
  nonempty_string($a1_provenance.host.hostname) and
  $a1_provenance.host.hostname == $b_provenance.host.hostname and
  $a1_provenance.host.hostname == $a2_provenance.host.hostname
) as $same_hostname |
(
  nonempty_string($a1_provenance.git_commit_full) and
  $a1_provenance.git_commit_full == $b_provenance.git_commit_full and
  $a1_provenance.git_commit_full == $a2_provenance.git_commit_full
) as $same_commit |
(
  nonempty_string($a1_provenance.model_hashes.config_json_sha256) and
  nonempty_string($a1_provenance.model_hashes.dense_manifest_sha256) and
  $a1_provenance.model_hashes == $b_provenance.model_hashes and
  $a1_provenance.model_hashes == $a2_provenance.model_hashes
) as $same_model_hashes |
(
  nonempty_string($a1_provenance.prompt_hashes.short_sha256) and
  nonempty_string($a1_provenance.prompt_hashes.medium_sha256) and
  $a1_provenance.prompt_hashes == $b_provenance.prompt_hashes and
  $a1_provenance.prompt_hashes == $a2_provenance.prompt_hashes
) as $same_prompt_hashes |
(
  nonempty_string($a1_provenance.binary_sha256) and
  $a1_provenance.binary_sha256 == $b_provenance.binary_sha256 and
  $a1_provenance.binary_sha256 == $a2_provenance.binary_sha256
) as $same_binary_hash |
(
  ($a1_provenance.cargo_features | type) == "array" and
  ($a1_provenance.cargo_features | length) > 0 and
  $a1_provenance.cargo_features == $b_provenance.cargo_features and
  $a1_provenance.cargo_features == $a2_provenance.cargo_features
) as $same_cargo_features |
(
  qualifying_cpu($a1_provenance) and
  qualifying_cpu($b_provenance) and
  qualifying_cpu($a2_provenance)
) as $cpu_qualified |
(
  qualifying_storage($a1_provenance) and
  qualifying_storage($b_provenance) and
  qualifying_storage($a2_provenance)
) as $storage_qualified |
(
  $a1_provenance.model_mount_identity == $b_provenance.model_mount_identity and
  $a1_provenance.model_mount_identity == $a2_provenance.model_mount_identity
) as $same_mount_identity |
(
  expected_provenance_schema($a1_provenance) and
  expected_provenance_schema($b_provenance) and
  expected_provenance_schema($a2_provenance)
) as $schemas_qualified |
(
  expected_configuration($a1_provenance; 2; 3; true) and
  expected_configuration($b_provenance; 0; 1; false) and
  expected_configuration($a2_provenance; 2; 3; true)
) as $configuration_matrix_passed |
(
  streaming_cases_qualified($a1_summary) and
  streaming_cases_qualified($b_summary) and
  streaming_cases_qualified($a2_summary) and
  $a1_summary.qualification_passed == true and
  $b_summary.qualification_passed == true and
  $a2_summary.qualification_passed == true
) as $streaming_qualifications_passed |
(
  streaming_output_parity($a1_summary) and
  streaming_output_parity($b_summary) and
  streaming_output_parity($a2_summary)
) as $streaming_output_parity_passed |
(
  $same_hostname and $same_commit and $same_model_hashes and
  $same_prompt_hashes and $same_binary_hash and $same_cargo_features and
  $cpu_qualified and $storage_qualified and $same_mount_identity and
  $schemas_qualified and $configuration_matrix_passed and
  $streaming_qualifications_passed and $streaming_output_parity_passed
) as $provenance_passed |
(
  $short_control_stability.decode_tps.percent_delta != null and
  (($short_control_stability.decode_tps.percent_delta | absolute) <= 2)
) as $short_control_stable |
(
  $medium_control_stability.decode_tps.percent_delta != null and
  (($medium_control_stability.decode_tps.percent_delta | absolute) <= 2)
) as $medium_control_stable |
(control_delta($a1_gm; $a2_gm)) as $gm_control_stability |
(
  $gm_control_stability.percent_delta != null and
  (($gm_control_stability.percent_delta | absolute) <= 2)
) as $gm_control_stable |
($short_control_stable and $medium_control_stable and $gm_control_stable) as $controls_stable |
(
  case_prefetch_disabled($b_short) and case_prefetch_disabled($b_medium)
) as $prefetch_disabled |
(
  case_prefetch_counters_zero($b_short) and case_prefetch_counters_zero($b_medium)
) as $prefetch_counters_zero |
(
  $b_short.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active == 0 and
  $b_medium.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active == 0
) as $prompt_overlap_zero |
(
  $b_short.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active == 0 and
  $b_medium.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active == 0
) as $decode_overlap_zero |
(
  streaming_cases_qualified($b_summary) and $b_summary.qualification_passed == true
) as $b_qualification_passed |
(streaming_output_parity($b_summary)) as $b_output_parity_passed |
(
  $prefetch_disabled and $prefetch_counters_zero and
  $prompt_overlap_zero and $decode_overlap_zero and
  $b_qualification_passed and $b_output_parity_passed
) as $no_prefetch_invariants_passed |
(
  negative_number($short_comparison.demand_storage_mean_seconds.percent_delta) and
  negative_number($medium_comparison.demand_storage_mean_seconds.percent_delta)
) as $storage_mean_improved_consistently |
(
  negative_number($short_comparison.layer_fetch_critical_path_mean_seconds.percent_delta) and
  negative_number($medium_comparison.layer_fetch_critical_path_mean_seconds.percent_delta)
) as $critical_path_mean_improved_consistently |
(
  $storage_mean_improved_consistently or $critical_path_mean_improved_consistently
) as $foreground_latency_improved_consistently |
(percent_delta($control_mean_gm; $b_gm)) as $b_gm_percent_delta |
(
  if (
    ($controls_stable | not) or
    ($provenance_passed | not) or
    ($no_prefetch_invariants_passed | not)
  ) then
    "inconclusive"
  elif ($b_gm_percent_delta | type) != "number" then
    "inconclusive"
  elif $b_gm_percent_delta >= -0.5 and $foreground_latency_improved_consistently then
    "prefetch_net_negative_or_unnecessary"
  elif $b_gm_percent_delta <= -2 and $foreground_latency_improved_consistently then
    "prefetch_helpful_but_contentious"
  elif $b_gm_percent_delta <= -2 and ($foreground_latency_improved_consistently | not) then
    "prefetch_not_primary_bottleneck"
  else
    "inconclusive"
  end
) as $classification |
{
  schema: {
    name: "mer-prompt2-phase4a-prefetch-ablation",
    version: 1
  },
  sources: {
    control_a1_dir: $control_a1_dir,
    no_prefetch_b_dir: $no_prefetch_b_dir,
    control_a2_dir: $control_a2_dir,
    source_artifacts_modified: false
  },
  provenance: {
    hostnames: {
      control_a1: $a1_provenance.host.hostname,
      no_prefetch_b: $b_provenance.host.hostname,
      control_a2: $a2_provenance.host.hostname,
      all_match: $same_hostname
    },
    git_commits: {
      control_a1: $a1_provenance.git_commit_full,
      no_prefetch_b: $b_provenance.git_commit_full,
      control_a2: $a2_provenance.git_commit_full,
      all_match: $same_commit
    },
    model_hashes_match: $same_model_hashes,
    prompt_hashes_match: $same_prompt_hashes,
    binary_hashes_match: $same_binary_hash,
    cargo_feature_sets_match: $same_cargo_features,
    cpu_identity_qualified: $cpu_qualified,
    storage_identity_qualified: $storage_qualified,
    model_mount_identities_match: $same_mount_identity,
    schemas_and_experiment_names_qualified: $schemas_qualified,
    configuration_matrix_passed: $configuration_matrix_passed,
    streaming_qualifications_passed: $streaming_qualifications_passed,
    streaming_output_parity_passed: $streaming_output_parity_passed,
    source_values: {
      control_a1: {
        model_hashes: $a1_provenance.model_hashes,
        prompt_hashes: $a1_provenance.prompt_hashes,
        binary_sha256: $a1_provenance.binary_sha256,
        cargo_features: $a1_provenance.cargo_features,
        cpu: $a1_provenance.host,
        model_mount_identity: $a1_provenance.model_mount_identity,
        configuration: {
          predict_fanout: $a1_provenance.predict_fanout,
          pipeline_depth: $a1_provenance.pipeline_depth,
          prefetch_expected_active: $a1_provenance.prefetch_expected_active
        }
      },
      no_prefetch_b: {
        model_hashes: $b_provenance.model_hashes,
        prompt_hashes: $b_provenance.prompt_hashes,
        binary_sha256: $b_provenance.binary_sha256,
        cargo_features: $b_provenance.cargo_features,
        cpu: $b_provenance.host,
        model_mount_identity: $b_provenance.model_mount_identity,
        configuration: {
          predict_fanout: $b_provenance.predict_fanout,
          pipeline_depth: $b_provenance.pipeline_depth,
          prefetch_expected_active: $b_provenance.prefetch_expected_active
        }
      },
      control_a2: {
        model_hashes: $a2_provenance.model_hashes,
        prompt_hashes: $a2_provenance.prompt_hashes,
        binary_sha256: $a2_provenance.binary_sha256,
        cargo_features: $a2_provenance.cargo_features,
        cpu: $a2_provenance.host,
        model_mount_identity: $a2_provenance.model_mount_identity,
        configuration: {
          predict_fanout: $a2_provenance.predict_fanout,
          pipeline_depth: $a2_provenance.pipeline_depth,
          prefetch_expected_active: $a2_provenance.prefetch_expected_active
        }
      }
    },
    all_passed: $provenance_passed
  },
  control_stability: {
    threshold_absolute_percent: 2,
    short: $short_control_stability,
    medium: $medium_control_stability,
    streaming_geometric_mean_decode_tps: $gm_control_stability,
    short_fixture_within_threshold: $short_control_stable,
    medium_fixture_within_threshold: $medium_control_stable,
    streaming_geometric_mean_within_threshold: $gm_control_stable,
    all_passed: $controls_stable
  },
  no_prefetch_invariants: {
    prefetch_disabled: $prefetch_disabled,
    all_prefetch_counters_zero: $prefetch_counters_zero,
    prompt_speculative_overlap_zero: $prompt_overlap_zero,
    decode_speculative_overlap_zero: $decode_overlap_zero,
    qualification_passed: $b_qualification_passed,
    output_parity_passed: $b_output_parity_passed,
    all_passed: $no_prefetch_invariants_passed
  },
  no_prefetch_vs_control_mean: {
    short: $short_comparison,
    medium: $medium_comparison,
    streaming_geometric_mean_decode_tps: metric_vs_control_mean($a1_gm; $a2_gm; $b_gm),
    prompt_demand_reads_issued_while_speculative_reads_active: {
      short: $b_short.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active,
      medium: $b_medium.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active,
      total: (
        $b_short.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active +
        $b_medium.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active
      )
    },
    decode_demand_reads_issued_while_speculative_reads_active: {
      short: $b_short.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active,
      medium: $b_medium.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active,
      total: (
        $b_short.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active +
        $b_medium.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active
      )
    },
    prefetch_counters: {
      short: $b_short.phase4a_prefetch.counters,
      medium: $b_medium.phase4a_prefetch.counters,
      total: sum_prefetch_counters($b_short; $b_medium)
    }
  },
  interpretation: {
    thresholds: {
      controls_stable_absolute_percent: 2,
      throughput_neutral_lower_bound_percent: -0.5,
      material_throughput_regression_percent: -2
    },
    controls_stable: $controls_stable,
    provenance_and_qualification_passed: $provenance_passed,
    no_prefetch_invariants_passed: $no_prefetch_invariants_passed,
    no_prefetch_streaming_geometric_mean_percent_delta: $b_gm_percent_delta,
    demand_storage_mean_improved_consistently: $storage_mean_improved_consistently,
    layer_fetch_critical_path_mean_improved_consistently: $critical_path_mean_improved_consistently,
    foreground_storage_or_critical_path_improved_consistently: $foreground_latency_improved_consistently,
    classification: $classification
  }
}
