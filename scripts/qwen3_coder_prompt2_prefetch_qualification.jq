.predictive_policy.markov_prefetch_fanout == $predict_fanout and
.predictive_policy.pipeline_depth == $pipeline_depth and
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
  .prefetch_dropped_governor == 0 and
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
else true end)
