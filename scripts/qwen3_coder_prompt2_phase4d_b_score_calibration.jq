def item($value): $value[0];

def finite_number:
  type == "number" and (isnan or isinfinite) == false;

def nullable_finite:
  . == null or finite_number;

def distribution_valid($expected_count):
  .count == $expected_count and
  (if $expected_count == 0 then
     [.minimum, .maximum, .mean, .p50, .p90, .p95, .p99] |
     all(. == null)
   else
     [.minimum, .maximum, .mean, .p50, .p90, .p95, .p99] |
     all(finite_number)
   end);

def boundary_valid($admitted; $rejected):
  (if $rejected == 0 then
     .maximum_rejected_candidate_score == null and
     .maximum_rejected_score_to_threshold_ratio == null and
     .closest_rejected_score_minus_threshold_margin == null
   else
     (.maximum_rejected_candidate_score | finite_number) and
     (.maximum_rejected_score_to_threshold_ratio | finite_number) and
     .maximum_rejected_score_to_threshold_ratio < 1 and
     (.closest_rejected_score_minus_threshold_margin | finite_number) and
     .closest_rejected_score_minus_threshold_margin < 0
   end) and
  (if $admitted == 0 then
     .minimum_admitted_candidate_score == null and
     .minimum_admitted_score_to_threshold_ratio == null and
     .closest_admitted_score_minus_threshold_margin == null
   else
     (.minimum_admitted_candidate_score | finite_number) and
     (.minimum_admitted_score_to_threshold_ratio | finite_number) and
     .minimum_admitted_score_to_threshold_ratio >= 1 and
     (.closest_admitted_score_minus_threshold_margin | finite_number) and
     .closest_admitted_score_minus_threshold_margin >= 0
   end);

def foreground_counts_valid($total; $admitted; $rejected):
  .foreground_inflight_zero >= 0 and
  .foreground_inflight_zero_admitted >= 0 and
  .foreground_inflight_zero_rejected >= 0 and
  .foreground_inflight_positive >= 0 and
  .foreground_inflight_positive_admitted >= 0 and
  .foreground_inflight_positive_rejected >= 0 and
  .foreground_inflight_zero ==
    (.foreground_inflight_zero_admitted +
     .foreground_inflight_zero_rejected) and
  .foreground_inflight_positive ==
    (.foreground_inflight_positive_admitted +
     .foreground_inflight_positive_rejected) and
  (.foreground_inflight_zero + .foreground_inflight_positive) == $total and
  (.foreground_inflight_zero_admitted +
   .foreground_inflight_positive_admitted) == $admitted and
  (.foreground_inflight_zero_rejected +
   .foreground_inflight_positive_rejected) == $rejected;

def diagnostics_valid($enabled; $total; $admitted; $rejected):
  .enabled == $enabled and
  (.semantics | type == "string" and length > 0) and
  .sample_capacity == 512 and
  .sampled_decisions == ([$total, 512] | min) and
  .total_decisions == $total and
  .admitted == $admitted and
  .rejected == $rejected and
  .total_decisions == (.admitted + .rejected) and
  .invalid_numeric_decisions == 0 and
  .ratio_undefined_decisions == 0 and
  (.base_threshold | finite_number) and
  (.contention_weight | finite_number) and
  (.candidate_probability | distribution_valid($total)) and
  (.effective_precision | distribution_valid($total)) and
  (.candidate_score | distribution_valid($total)) and
  (.effective_threshold | distribution_valid($total)) and
  (.score_to_threshold_ratio | distribution_valid($total)) and
  (.decision_boundary | boundary_valid($admitted; $rejected)) and
  (.foreground_inflight_decisions |
   foreground_counts_valid($total; $admitted; $rejected));

def expected_governor_config($enabled; $base_threshold; $contention_weight):
  {
    enabled: $enabled,
    precision_floor: 0.05,
    contention_weight: $contention_weight,
    base_threshold: $base_threshold,
    runtime_default_precision_alpha: 0.2,
    runtime_default_base_threshold: 0.02
  };

def second_only_metadata_valid($variant; $enabled):
  .metadata == {
    variant: $variant,
    predictor_mode: "second-order-only",
    predict_fanout: 1,
    pipeline_depth: 3,
    first_order_enabled: false,
    second_order_enabled: true,
    fallback_prior_fill_enabled: false,
    fanout_is_upper_bound: true,
    governor_enabled: $enabled,
    neural_speculator_enabled: false
  };

def direct_run_reconciles:
  .governor_total_decisions ==
    (.governor_admitted_candidates + .governor_rejected_candidates) and
  .direct_governor_decisions.total_decisions ==
    .governor_total_decisions and
  .direct_governor_decisions.admitted_candidates ==
    .governor_admitted_candidates and
  .direct_governor_decisions.rejected_candidates ==
    .governor_rejected_candidates;

def direct_counters_reconcile:
  ([.governor_counters_by_run[] | direct_run_reconciles] | all) and
  .governor_totals.governor_admitted_candidates ==
    ([.governor_counters_by_run[].governor_admitted_candidates] | add) and
  .governor_totals.governor_rejected_candidates ==
    ([.governor_counters_by_run[].governor_rejected_candidates] | add) and
  .governor_totals.governor_total_decisions ==
    (.governor_totals.governor_admitted_candidates +
     .governor_totals.governor_rejected_candidates);

def diagnostic_counters_reconcile:
  ([.governor_counters_by_run[] |
    .governor_score_diagnostics_reconcile == true and
    (.governor_score_diagnostics |
     diagnostics_valid(.enabled;
       .total_decisions; .admitted; .rejected)) and
    .governor_score_diagnostics.total_decisions ==
      .governor_total_decisions and
    .governor_score_diagnostics.admitted ==
      .governor_admitted_candidates and
    .governor_score_diagnostics.rejected ==
      .governor_rejected_candidates] | all) and
  .governor_score_diagnostics_reconcile == true and
  (.governor_score_diagnostics |
   diagnostics_valid(.enabled;
     .total_decisions; .admitted; .rejected)) and
  .governor_score_diagnostics.total_decisions ==
    .governor_totals.governor_total_decisions and
  .governor_score_diagnostics.admitted ==
    .governor_totals.governor_admitted_candidates and
  .governor_score_diagnostics.rejected ==
    .governor_totals.governor_rejected_candidates;

def nonempty_string:
  type == "string" and length > 0;

def git_sha:
  type == "string" and test("^[0-9a-f]{40}$");

def sha256:
  type == "string" and test("^[0-9a-f]{64}$");

def provenance_complete($runner_git_commit_full):
  .provenance.git_commit_full == $runner_git_commit_full and
  (.provenance.git_commit_full | git_sha) and
  (.provenance.binary_sha256 | sha256) and
  (.provenance.model_hashes.config_json_sha256 | sha256) and
  (.provenance.model_hashes.dense_manifest_sha256 | sha256) and
  (.provenance.tokenizer_identity.path | nonempty_string) and
  (.provenance.tokenizer_identity.sha256 | sha256) and
  (.provenance.target_host.hostname | nonempty_string) and
  .provenance.target_host.logical_cpu_count == 32 and
  .provenance.target_host.requested_cpu_mask == "0-31" and
  .provenance.target_host.effective_cpu_mask == "0-31" and
  (.provenance.model_mount_identity.target | nonempty_string) and
  (.provenance.model_mount_identity.fstype | nonempty_string) and
  (.provenance.model_mount_identity.options | nonempty_string) and
  (.provenance.cargo_features | type) == "array" and
  (.provenance.cargo_features | length) > 0 and
  (.provenance.prompt_hashes.short_sha256 | sha256);

def reported_variant($variant):
  {
    variant: $variant.variant,
    governor_configuration: $variant.governor_configuration,
    provenance: $variant.provenance,
    decode_tps_mean: $variant.decode_tps_mean,
    ssd_bytes: $variant.ssd_bytes,
    demand_read_service_mean_seconds:
      $variant.demand_read_service_mean_seconds,
    governor_counters_by_run: $variant.governor_counters_by_run,
    governor_totals: $variant.governor_totals,
    governor_direct_counters: {
      admitted: $variant.governor_totals.governor_admitted_candidates,
      rejected: $variant.governor_totals.governor_rejected_candidates,
      total: $variant.governor_totals.governor_total_decisions
    },
    governor_score_diagnostics: $variant.governor_score_diagnostics,
    governor_score_diagnostics_by_run:
      $variant.governor_score_diagnostics_by_run,
    governor_score_diagnostics_reconcile:
      $variant.governor_score_diagnostics_reconcile,
    foreground_accounting: $variant.foreground_accounting,
    prefetch_counters: $variant.prefetch_counters,
    output_token_ids: $variant.output_token_ids,
    diagnostic_collection_valid: $variant.diagnostic_collection_valid,
    qualification_passed: false
  };

[
  item($demand),
  item($second),
  item($current),
  item($low)
] as $variants |
[
  "demand-only",
  "second-only-f1",
  "second-only-f1-governed-current",
  "second-only-f1-governed-bt005-cw000"
] as $expected_order |
($variants[0].provenance) as $reference_provenance |
{
  exact_variant_order:
    ([$variants[].variant] == $expected_order),
  deterministic_output_parity:
    (($variants | all(.output_parity_within_variant == true)) and
     ([$variants[].output_token_ids] | unique | length) == 1),
  expected_predictor_configuration:
    ($variants[0].metadata == {
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
     } and
     ($variants[1] |
      second_only_metadata_valid("second-only-f1"; false)) and
     ($variants[2] |
      second_only_metadata_valid("second-only-f1-governed-current"; true)) and
     ($variants[3] |
      second_only_metadata_valid(
        "second-only-f1-governed-bt005-cw000"; true))),
  expected_governor_configuration:
    ($variants[0].governor_configuration ==
       expected_governor_config(false; 0.02; 1.0) and
     $variants[1].governor_configuration ==
       expected_governor_config(false; 0.02; 1.0) and
     $variants[2].governor_configuration ==
       expected_governor_config(true; 0.02; 1.0) and
     $variants[3].governor_configuration ==
       expected_governor_config(true; 0.005; 0.0)),
  direct_counters_reconcile:
    ($variants | all(direct_counters_reconcile)),
  diagnostic_decision_counters_reconcile:
    ($variants | all(diagnostic_counters_reconcile)),
  foreground_read_accounting_balanced:
    ($variants | all(
      ([.foreground_accounting.final_by_run[]] | all(. == 0)) and
      .foreground_accounting.final_aggregate == 0)),
  cross_variant_provenance_identical:
    (($runner_git_commit_full | git_sha) and
     ($variants | all(
       .provenance == $reference_provenance and
       provenance_complete($runner_git_commit_full)
     ))),
  demand_only_has_no_governor_decisions:
    ($variants[0].governor_totals.governor_total_decisions == 0 and
     $variants[0].governor_score_diagnostics.total_decisions == 0),
  ungoverned_control_performs_speculative_work:
    ($variants[1].prefetch_counters.prefetch_submitted > 0 and
     $variants[1].prefetch_counters.prefetch_completed > 0 and
     $variants[1].governor_totals.governor_total_decisions == 0 and
     $variants[1].governor_score_diagnostics.total_decisions == 0),
  governed_variants_make_governor_decisions:
    ($variants[2:4] | all(
      .governor_totals.governor_total_decisions > 0 and
      .governor_score_diagnostics.total_decisions > 0)),
  all_statistics_finite_or_explicitly_null:
    ($variants | all(diagnostic_counters_reconcile)),
  diagnostic_mode_is_non_qualifying:
    ($variants | all(
      .qualification_kind == "phase4d-b-score-calibration-diagnostic" and
      .diagnostic_collection_valid == true and
      .performance_qualification_applicable == false and
      .qualification_passed == false and
      .traced == false and
      .cache_slots == 1536 and
      .prompt_fixture == "short" and
      .output_tokens == 128 and
      .warmup_runs == 1 and
      .measured_runs == 2 and
      .cache_reset == "keep" and
      .greedy == true and
      .metadata.neural_speculator_enabled == false
    ))
} as $gates |
{
  schema: {
    name:"mer-prompt2-phase4d-b-governor-score-calibration-summary",
    version:1
  },
  qualification_kind: "phase4d-b-score-calibration-diagnostic",
  performance_qualification_applicable: false,
  performance_qualification_reason:
    "Phase 4D-B measures bounded governor score and threshold distributions and cannot qualify as a production performance baseline",
  qualification_passed: false,
  traced: false,
  cache_slots: 1536,
  prompt_fixture: "short",
  output_tokens: 128,
  warmup_runs: 1,
  measured_runs: 2,
  cache_reset: "keep",
  greedy: true,
  neural_speculator_enabled: false,
  percentile_semantics:
    "nearest-rank percentiles over a deterministic ring of the most recent 512 finite decisions; counts, means, extrema, boundary values, and foreground splits cover every decision in each reset window",
  empty_population_semantics:
    "distribution statistics and decision-boundary values are null when their population is empty; disabled-governor controls report zero decisions and do not fabricate scores",
  runner_git_commit_full: $runner_git_commit_full,
  expected_variant_order: $expected_order,
  variants: [$variants[] | reported_variant(.)],
  gates: $gates,
  diagnostic_gates_passed: ([$gates[]] | all(. == true))
}
