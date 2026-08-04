def lifecycle_reconciles:
  .lifecycle_reconciliation_passed == true and
  ((.lifecycle_reconciliation_errors // []) | length) == 0 and
  .lifecycle.physical_read_issued ==
    (.lifecycle.physical_read_completed +
     .lifecycle.physical_read_failed +
     .lifecycle.physical_read_inflight_at_sample) and
  .lifecycle.physical_read_completed ==
    (.lifecycle.published +
     .lifecycle.publication_rejected +
     .lifecycle.completion_not_yet_published_at_sample) and
  .lifecycle.published ==
    (.lifecycle.first_use +
     .lifecycle.evicted_before_first_use +
     .lifecycle.still_resident_unused_at_sample);

def phase4b_snapshot_valid:
  .schema_name == "mer-prompt2-phase4b-routing-trace" and
  .schema_version == 1 and
  (.trace_path | type) == "string" and
  (.trace_path | length) > 0 and
  .max_events > 0 and
  .events_written > 0 and
  .events_dropped == 0 and
  .trace_truncated == false and
  .trace_write_failed == false and
  lifecycle_reconciles;

def critical_path_observation_valid:
  .wall_seconds > 0 and
  .attributed_seconds >= 0 and
  (.unattributed_residual_seconds | type) == "number" and
  .unattributed_residual_seconds >= 0 and
  (.coverage_ratio | type) == "number" and
  .coverage_ratio >= 0 and
  .non_overlap_invariant_passed == true and
  (.coverage_95_percent_passed | type) == "boolean" and
  (.qualification_passed | type) == "boolean" and
  ([.categories[]] | all(type == "number" and . >= 0));

def production_critical_path_valid:
  .wall_seconds > 0 and
  .attributed_seconds >= 0 and
  has("unattributed_residual_seconds") and
  .coverage_ratio >= 0.95 and
  .non_overlap_invariant_passed == true and
  .coverage_95_percent_passed == true and
  .qualification_passed == true and
  ([.categories[]] | all(type == "number" and . >= 0));

def performance_collection_valid:
  .phase4b_trace_enabled == false and
  ([.runs[] | has("phase4b_diagnostics")] | all(. == false)) and
  ([.runs[].critical_path.prompt, .runs[].critical_path.decode] |
    all(production_critical_path_valid));

def diagnostic_collection_valid:
  .phase4b_trace_enabled == true and
  (.phase4b_trace | phase4b_snapshot_valid) and
  ([.runs[].phase4b_diagnostics] | all(phase4b_snapshot_valid)) and
  ([.runs[].critical_path.prompt, .runs[].critical_path.decode] |
    all(critical_path_observation_valid));

def screening_collection_valid:
  .phase4b_trace_enabled == false and
  (.runs | length) == 2 and
  ([.runs[] | has("phase4b_diagnostics")] | all(. == false)) and
  ([.runs[].critical_path.prompt, .runs[].critical_path.decode] |
    all(critical_path_observation_valid));

(if $qualification_kind == "performance-baseline" then
  performance_collection_valid
elif $qualification_kind == "phase4b-diagnostic" then
  diagnostic_collection_valid
elif $qualification_kind == "phase4d-governor-screening" then
  screening_collection_valid
elif $qualification_kind == "phase4d-b-score-calibration-diagnostic" then
  screening_collection_valid
elif $qualification_kind == "phase4d-c-sparse-admission-screening" then
  screening_collection_valid
else
  false
end) as $collection_valid |
([.runs[].critical_path.prompt.coverage_ratio]) as $prompt_coverage |
([.runs[].critical_path.decode.coverage_ratio]) as $decode_coverage |
{
  qualification_kind: $qualification_kind,
  collection_qualification_valid: $collection_valid,
  qualification_passed: (
    if $qualification_kind == "performance-baseline"
    then $collection_valid
    else false
    end
  ),
  diagnostic_qualification_passed: (
    if $qualification_kind == "phase4b-diagnostic"
    then $collection_valid
    else null
    end
  ),
  screening_collection_valid: (
    if $qualification_kind == "phase4d-governor-screening"
    then $collection_valid
    else null
    end
  ),
  score_calibration_collection_valid: (
    if $qualification_kind == "phase4d-b-score-calibration-diagnostic"
    then $collection_valid
    else null
    end
  ),
  sparse_admission_collection_valid: (
    if $qualification_kind == "phase4d-c-sparse-admission-screening"
    then $collection_valid
    else null
    end
  ),
  performance_qualification_applicable: ($qualification_kind == "performance-baseline"),
  performance_qualification_passed: (
    if $qualification_kind == "performance-baseline"
    then $collection_valid
    else null
    end
  ),
  performance_qualification_reason: (
    if $qualification_kind == "phase4b-diagnostic" then
      "synchronous Phase 4B JSONL tracing adds diagnostic wall time outside production critical-path categories; traced TPS and coverage are not comparable with untraced Phase 4A performance baselines"
    elif $qualification_kind == "phase4d-governor-screening" then
      "Phase 4D-A uses one warmup and two measured short-prompt runs for calibration screening; it is diagnostic evidence and not a qualified production performance baseline"
    elif $qualification_kind == "phase4d-b-score-calibration-diagnostic" then
      "Phase 4D-B records bounded governor score and threshold distributions from one warmup and two measured short-prompt runs; it is diagnostic evidence and not a qualified production performance baseline"
    elif $qualification_kind == "phase4d-c-sparse-admission-screening" then
      "Phase 4D-C screens sparse governor admissions with one warmup and two measured short-prompt runs; it identifies a candidate for later qualification and is not a qualified production performance baseline"
    else
      null
    end
  ),
  production_critical_path_coverage_gates_passed: (
    [.runs[].critical_path.prompt, .runs[].critical_path.decode] |
    all(production_critical_path_valid)
  ),
  observed_critical_path_coverage: {
    prompt: $prompt_coverage,
    decode: $decode_coverage,
    prompt_min: ($prompt_coverage | min),
    decode_min: ($decode_coverage | min)
  }
}
