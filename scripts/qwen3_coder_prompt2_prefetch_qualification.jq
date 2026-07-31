.predictive_policy.markov_prefetch_fanout == $predict_fanout and
.predictive_policy.pipeline_depth == $pipeline_depth and
.predictive_policy.prefetch_governor_enabled == $prefetch_governor_enabled and
.predictive_policy.speculator_enabled == false and
.memory_layout.primary_expert_pool_allocated_bytes > 0 and
.memory_layout.shadow_expert_pool_allocated_bytes >= 0 and
.memory_layout.total_expert_pool_allocated_bytes ==
  (.memory_layout.primary_expert_pool_allocated_bytes + .memory_layout.shadow_expert_pool_allocated_bytes) and
([.runs[].memory] | all(
  .primary_expert_pool_allocated_bytes > 0 and
  .shadow_expert_pool_allocated_bytes >= 0 and
  .total_expert_pool_allocated_bytes ==
    (.primary_expert_pool_allocated_bytes + .shadow_expert_pool_allocated_bytes)
)) and
([.runs[].cache_io] | all(
  .prefetch_enabled == ($predict_fanout > 0) and
  .prefetch_submitted >= 0 and
  .prefetch_completed >= 0 and
  .prefetch_used >= 0 and
  .prefetch_bytes >= 0 and
  .useful_prefetch_bytes >= 0 and
  .unused_prefetch_bytes_at_sample >= 0 and
  .prefetch_dropped_concurrency >= 0 and
  .prefetch_dropped_pool_starved >= 0 and
  (if $prefetch_governor_enabled then
    .prefetch_dropped_governor >= 0
  else
    .prefetch_dropped_governor == 0
  end) and
  .prefetch_dropped_bytes == 0 and
  (if $predict_fanout == 0 then
    .prefetch_submitted == 0 and
    .prefetch_completed == 0 and
    .prefetch_used == 0 and
    .prefetch_bytes == 0 and
    .useful_prefetch_bytes == 0 and
    .unused_prefetch_bytes_at_sample == 0 and
    .prefetch_dropped_concurrency == 0 and
    .prefetch_dropped_pool_starved == 0
  else true end)
)) and
([.runs[].demand_miss_fanout.prompt, .runs[].demand_miss_fanout.decode] | all(
  if $predict_fanout == 0 then
    .demand_reads_issued_while_speculative_reads_active == 0
  else
    .demand_reads_issued_while_speculative_reads_active >= 0
  end
)) and
(if $predict_fanout > 0 then
  .memory_layout.shadow_expert_pool_allocated_bytes > 0 and
  ([.runs[].memory.shadow_expert_pool_allocated_bytes] | all(. > 0))
else true end) and
(if (.phase4b_trace_enabled // false) then
  .phase4b_trace.schema_name == "mer-prompt2-phase4b-routing-trace" and
  .phase4b_trace.schema_version == 1 and
  (.phase4b_trace.trace_path | length) > 0 and
  .phase4b_trace.max_events > 0 and
  .phase4b_trace.events_written > 0 and
  .phase4b_trace.events_dropped >= 0 and
  (.phase4b_trace.trace_truncated | type) == "boolean" and
  .phase4b_trace.trace_write_failed == false and
  .phase4b_trace.lifecycle_reconciliation_passed == true and
  .phase4b_trace.lifecycle.physical_read_issued ==
    (.phase4b_trace.lifecycle.physical_read_completed +
     .phase4b_trace.lifecycle.physical_read_failed +
     .phase4b_trace.lifecycle.physical_read_inflight_at_sample) and
  .phase4b_trace.lifecycle.physical_read_completed ==
    (.phase4b_trace.lifecycle.published +
     .phase4b_trace.lifecycle.publication_rejected +
     .phase4b_trace.lifecycle.completion_not_yet_published_at_sample) and
  .phase4b_trace.lifecycle.published ==
    (.phase4b_trace.lifecycle.first_use +
     .phase4b_trace.lifecycle.evicted_before_first_use +
     .phase4b_trace.lifecycle.still_resident_unused_at_sample) and
  ([.runs[].phase4b_diagnostics] | all(
    .schema_name == "mer-prompt2-phase4b-routing-trace" and
    .schema_version == 1 and
    .max_events > 0 and
    .trace_write_failed == false and
    .lifecycle_reconciliation_passed == true and
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
       .lifecycle.still_resident_unused_at_sample)
  ))
else
  ([.runs[] | has("phase4b_diagnostics")] | all(. == false))
end)
