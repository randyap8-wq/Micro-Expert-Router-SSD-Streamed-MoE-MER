def item($value): $value[0];

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

def run_decisions_reconcile:
  .governor_total_decisions ==
    (.governor_admitted_candidates + .governor_rejected_candidates) and
  .direct_governor_decisions.total_decisions == .governor_total_decisions and
  .direct_governor_decisions.admitted_candidates ==
    .governor_admitted_candidates and
  .direct_governor_decisions.rejected_candidates ==
    .governor_rejected_candidates;

def direct_decisions_reconcile:
  ([.governor_counters_by_run[] | run_decisions_reconcile] | all) and
  .governor_totals.governor_total_decisions ==
    (.governor_totals.governor_admitted_candidates +
     .governor_totals.governor_rejected_candidates);

def direct_and_derived_admissions_reconcile:
  ([.governor_counters_by_run[] |
    .governor_admitted_candidates ==
      .governor_admitted_candidates_derived and
    .derived_governor_admission.admitted_candidates ==
      .governor_admitted_candidates_derived] | all) and
  .governor_totals.governor_admitted_candidates ==
    .governor_totals.governor_admitted_candidates_derived and
  .governor_totals.governor_admitted_candidates ==
    ([.governor_counters_by_run[].governor_admitted_candidates] | add) and
  .governor_totals.governor_admitted_candidates_derived ==
    ([.governor_counters_by_run[].governor_admitted_candidates_derived] | add) and
  (if .metadata.governor_enabled then
     true
   else
     ([.governor_counters_by_run[] |
       .governor_admitted_candidates == 0 and
       .governor_admitted_candidates_derived == 0] | all) and
     .governor_totals.governor_admitted_candidates == 0 and
     .governor_totals.governor_admitted_candidates_derived == 0
   end);

def foreground_read_accounting_balanced:
  ([.governor_counters_by_run[].governor_foreground_inflight_final] |
   all(. == 0)) and
  .governor_totals.governor_foreground_inflight_final == 0;

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
    governor_admitted_candidates:
      $variant.governor_totals.governor_admitted_candidates,
    governor_rejected_candidates:
      $variant.governor_totals.governor_rejected_candidates,
    governor_total_decisions:
      $variant.governor_totals.governor_total_decisions,
    governor_admission_rate:
      $variant.governor_totals.governor_admission_rate,
    governor_admitted_candidates_derived:
      $variant.governor_totals.governor_admitted_candidates_derived,
    governor_admission_derivation:
      "prefetch_submitted + prefetch_dropped_concurrency; governor admission precedes the concurrency gate",
    completed_prefetches: $variant.prefetch_counters.prefetch_completed,
    used_prefetches: $variant.prefetch_counters.prefetch_used,
    measured_window_prefetch_used_per_completed:
      (if $variant.prefetch_counters.prefetch_completed == 0 then null
       else ($variant.prefetch_counters.prefetch_used /
             $variant.prefetch_counters.prefetch_completed)
       end),
    governor_precision_ewma_final:
      $variant.governor_totals.governor_precision_ewma_final,
    governor_foreground_inflight_final:
      $variant.governor_totals.governor_foreground_inflight_final,
    pool_pressure_drops:
      $variant.governor_totals.speculative_work_dropped_by_pool_pressure,
    concurrency_drops:
      $variant.governor_totals.speculative_work_dropped_by_concurrency,
    demand_reads_observed_while_speculation_active:
      $variant.demand_reads_observed_while_speculation_active,
    output_token_ids: $variant.output_token_ids,
    counter_semantics: {
      governor_admitted_candidates:
        "authoritative direct bench-real admission-controller counter",
      governor_rejected_candidates: "direct bench-real governor decision counter",
      governor_admitted_candidates_derived:
        "backward-compatible prefetch_submitted + prefetch_dropped_concurrency estimate",
      measured_window_prefetch_used_per_completed:
        "operational ratio of measured-run counter deltas; cache-reset keep allows warmup or prior-run prefetch lifecycle overlap, so this is not exact cohort precision and is not clamped"
    }
  };

[
  item($demand),
  item($second),
  item($current),
  item($cw025),
  item($bt010),
  item($bt005)
] as $variants |
[
  "demand-only",
  "second-only-f1",
  "second-only-f1-governed-current",
  "second-only-f1-governed-cw025",
  "second-only-f1-governed-bt010-cw025",
  "second-only-f1-governed-bt005-cw000"
] as $expected_order |
($variants[0].provenance) as $reference_provenance |
{
  exact_variant_order:
    ([$variants[].variant] == $expected_order),
  deterministic_output_parity:
    (($variants | all(.output_parity_within_variant == true)) and
     ([$variants[].output_token_ids] | unique | length) == 1),
  neural_speculator_disabled_everywhere:
    ($variants | all(.metadata.neural_speculator_enabled == false)),
  governor_assignment_exact:
    ($variants[0].metadata.governor_enabled == false and
     $variants[1].metadata.governor_enabled == false and
     ($variants[2:6] | all(.metadata.governor_enabled == true))),
  predictor_configuration_exact:
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
       second_only_metadata_valid("second-only-f1-governed-cw025"; true)) and
     ($variants[4] |
       second_only_metadata_valid("second-only-f1-governed-bt010-cw025"; true)) and
     ($variants[5] |
       second_only_metadata_valid("second-only-f1-governed-bt005-cw000"; true))),
  governor_configuration_values_exact:
    ($variants[0].governor_configuration ==
       expected_governor_config(false; 0.02; 1.0) and
     $variants[1].governor_configuration ==
       expected_governor_config(false; 0.02; 1.0) and
     $variants[2].governor_configuration ==
       expected_governor_config(true; 0.02; 1.0) and
     $variants[3].governor_configuration ==
       expected_governor_config(true; 0.02; 0.25) and
     $variants[4].governor_configuration ==
       expected_governor_config(true; 0.01; 0.25) and
     $variants[5].governor_configuration ==
       expected_governor_config(true; 0.005; 0.0)),
  direct_decision_counters_reconcile:
    ($variants | all(direct_decisions_reconcile)),
  direct_and_derived_admissions_reconcile:
    ($variants | all(direct_and_derived_admissions_reconcile)),
  foreground_read_accounting_balanced:
    ($variants | all(foreground_read_accounting_balanced)),
  demand_only_has_zero_speculative_work:
    (($variants[0].prefetch_counters | [.[]] | all(. == 0)) and
     $variants[0].governor_totals.governor_total_decisions == 0),
  ungoverned_second_only_f1_is_active:
    ($variants[1].prefetch_counters.prefetch_submitted > 0 and
     $variants[1].prefetch_counters.prefetch_completed > 0 and
     $variants[1].governor_totals.governor_total_decisions == 0),
  every_governed_case_made_an_admission_decision:
    ($variants[2:6] |
     all(.governor_totals.governor_total_decisions > 0)),
  cross_variant_provenance_identical:
    (($runner_git_commit_full | git_sha) and
     ($variants | all(
       .provenance == $reference_provenance and
       provenance_complete($runner_git_commit_full)
     ))),
  screening_mode_is_non_qualifying:
    ($variants | all(
      .qualification_kind == "phase4d-governor-screening" and
      .screening_collection_valid == true and
      .performance_qualification_applicable == false and
      .qualification_passed == false and
      .traced == false and
      .cache_slots == 1536 and
      .prompt_fixture == "short" and
      .output_tokens == 128 and
      .warmup_runs == 1 and
      .measured_runs == 2 and
      .greedy == true
    ))
} as $gates |
{
  schema: {name:"mer-prompt2-phase4d-governor-screening-summary", version:1},
  qualification_kind: "phase4d-governor-screening",
  performance_qualification_applicable: false,
  performance_qualification_reason:
    "Phase 4D-A is a two-run short-prompt calibration screen, not a qualified production performance baseline",
  qualification_passed: false,
  traced: false,
  cache_slots: 1536,
  prompt_fixture: "short",
  output_tokens: 128,
  warmup_runs: 1,
  measured_runs: 2,
  greedy: true,
  neural_speculator_enabled: false,
  runner_git_commit_full: $runner_git_commit_full,
  provenance: $reference_provenance,
  expected_variant_order: $expected_order,
  variants: [$variants[] | reported_variant(.)],
  gates: $gates,
  screening_gates_passed: ([$gates[]] | all(. == true))
}
