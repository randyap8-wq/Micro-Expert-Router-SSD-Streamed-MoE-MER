#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
CONFIG_HELPER="$ROOT/scripts/qwen3_coder_prompt2_collector_config.sh"
QUALIFIER="$ROOT/scripts/qwen3_coder_prompt2_prefetch_qualification.jq"
COLLECTION_QUALIFIER="$ROOT/scripts/qwen3_coder_prompt2_collection_qualification.jq"
TEMPLATE="$ROOT/benchmarks/qwen3-coder-single-stream/qwen3-coder-q8.toml.in"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4a-collector.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

source "$CONFIG_HELPER"

unset \
  MER_PROMPT2_PREDICT_FANOUT \
  MER_PROMPT2_PIPELINE_DEPTH \
  MER_PROMPT2_PREFETCH_VARIANT \
  MER_PROMPT2_FIRST_ORDER_ENABLED \
  MER_PROMPT2_SECOND_ORDER_ENABLED \
  MER_PROMPT2_FALLBACK_PRIOR_FILL_ENABLED \
  MER_PROMPT2_FANOUT_IS_UPPER_BOUND \
  MER_PROMPT2_PREFETCH_GOVERNOR_ENABLED \
  MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR \
  MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT \
  MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD
prompt2_resolve_ablation_config
test "$PREDICT_FANOUT" -eq 2
test "$PIPELINE_DEPTH" -eq 3
test "$PREFETCH_EXPECTED_ACTIVE" = true
test "$PREFETCH_GOVERNOR_PRECISION_FLOOR" = 0.05
test "$PREFETCH_GOVERNOR_CONTENTION_WEIGHT" = 1.0
test "$PREFETCH_GOVERNOR_BASE_THRESHOLD" = 0.02
prompt2_render_config "$TEMPLATE" "$TEST_ROOT/default.toml" \
  /mnt/localssd/model /mnt/localssd/model/tokenizer.json 1536 \
  "$PREDICT_FANOUT" "$PIPELINE_DEPTH" \
  "$FIRST_ORDER_ENABLED" "$SECOND_ORDER_ENABLED" \
  "$FALLBACK_PRIOR_FILL_ENABLED" "$FANOUT_IS_UPPER_BOUND" \
  "$PREFETCH_GOVERNOR_ENABLED"
grep -Fx 'predict_fanout = 2' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'pipeline_depth = 3' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'first_order_enabled = true' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'second_order_enabled = true' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'fallback_prior_fill_enabled = true' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'fanout_is_upper_bound = false' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'prefetch_governor = false' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'prefetch_precision_floor = 0.05' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'prefetch_contention_weight = 1.0' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'prefetch_governor_base_threshold = 0.02' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'speculator_enabled = false' "$TEST_ROOT/default.toml" >/dev/null
if grep -E '@[A-Z0-9_]+@' "$TEST_ROOT/default.toml" >/dev/null; then
  echo "default render left a collector placeholder unresolved" >&2
  exit 1
fi

MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR=0.125
MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT=0.25
MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD=0.005
prompt2_resolve_ablation_config
test "$PREFETCH_GOVERNOR_PRECISION_FLOOR" = 0.125
test "$PREFETCH_GOVERNOR_CONTENTION_WEIGHT" = 0.25
test "$PREFETCH_GOVERNOR_BASE_THRESHOLD" = 0.005
prompt2_render_config "$TEMPLATE" "$TEST_ROOT/governor-overrides.toml" \
  /mnt/localssd/model /mnt/localssd/model/tokenizer.json 1536 \
  "$PREDICT_FANOUT" "$PIPELINE_DEPTH" \
  "$FIRST_ORDER_ENABLED" "$SECOND_ORDER_ENABLED" \
  "$FALLBACK_PRIOR_FILL_ENABLED" "$FANOUT_IS_UPPER_BOUND" \
  "$PREFETCH_GOVERNOR_ENABLED"
grep -Fx 'prefetch_precision_floor = 0.125' \
  "$TEST_ROOT/governor-overrides.toml" >/dev/null
grep -Fx 'prefetch_contention_weight = 0.25' \
  "$TEST_ROOT/governor-overrides.toml" >/dev/null
grep -Fx 'prefetch_governor_base_threshold = 0.005' \
  "$TEST_ROOT/governor-overrides.toml" >/dev/null
unset \
  MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR \
  MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT \
  MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD

assert_invalid_governor_override() {
  local name=$1
  local value=$2
  if (
    unset \
      MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR \
      MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT \
      MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD
    export "$name=$value"
    prompt2_resolve_ablation_config >/dev/null 2>&1
  ); then
    echo "$name accepted invalid value: $value" >&2
    exit 1
  fi
}

for value in malformed NaN Infinity -0.1 1.1; do
  assert_invalid_governor_override \
    MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR "$value"
done
for value in malformed NaN Infinity -0.1; do
  assert_invalid_governor_override \
    MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT "$value"
  assert_invalid_governor_override \
    MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD "$value"
done

MER_PROMPT2_PREDICT_FANOUT=0
MER_PROMPT2_PIPELINE_DEPTH=1
prompt2_resolve_ablation_config
test "$PREDICT_FANOUT" -eq 0
test "$PIPELINE_DEPTH" -eq 1
test "$PREFETCH_EXPECTED_ACTIVE" = false
prompt2_render_config "$TEMPLATE" "$TEST_ROOT/no-prefetch.toml" \
  /mnt/localssd/model /mnt/localssd/model/tokenizer.json 1536 \
  "$PREDICT_FANOUT" "$PIPELINE_DEPTH" \
  "$FIRST_ORDER_ENABLED" "$SECOND_ORDER_ENABLED" \
  "$FALLBACK_PRIOR_FILL_ENABLED" "$FANOUT_IS_UPPER_BOUND" \
  "$PREFETCH_GOVERNOR_ENABLED"
grep -Fx 'predict_fanout = 0' "$TEST_ROOT/no-prefetch.toml" >/dev/null
grep -Fx 'pipeline_depth = 1' "$TEST_ROOT/no-prefetch.toml" >/dev/null

unset MER_PROMPT2_PREDICT_FANOUT MER_PROMPT2_PIPELINE_DEPTH
for variant in \
  demand-only \
  current-f2 \
  current-f2-governed \
  second-only-f2 \
  second-only-f1 \
  second-only-f1-governed; do
  MER_PROMPT2_PREFETCH_VARIANT=$variant
  prompt2_resolve_ablation_config
  test "$PIPELINE_DEPTH" -eq 3
  test "$NEURAL_SPECULATOR_ENABLED" = false
  case "$variant" in
    demand-only)
      test "$PREDICT_FANOUT" -eq 0
      test "$PREDICTOR_MODE" = demand-only
      test "$FIRST_ORDER_ENABLED" = true
      test "$FALLBACK_PRIOR_FILL_ENABLED" = true
      test "$FANOUT_IS_UPPER_BOUND" = false
      test "$PREFETCH_GOVERNOR_ENABLED" = false
      ;;
    current-f2)
      test "$PREDICT_FANOUT" -eq 2
      test "$PREDICTOR_MODE" = legacy-combined
      test "$FIRST_ORDER_ENABLED" = true
      test "$FALLBACK_PRIOR_FILL_ENABLED" = true
      test "$FANOUT_IS_UPPER_BOUND" = false
      test "$PREFETCH_GOVERNOR_ENABLED" = false
      ;;
    current-f2-governed)
      test "$PREDICT_FANOUT" -eq 2
      test "$PREDICTOR_MODE" = legacy-combined
      test "$FIRST_ORDER_ENABLED" = true
      test "$SECOND_ORDER_ENABLED" = true
      test "$FALLBACK_PRIOR_FILL_ENABLED" = true
      test "$FANOUT_IS_UPPER_BOUND" = false
      test "$PREFETCH_GOVERNOR_ENABLED" = true
      ;;
    second-only-f2)
      test "$PREDICT_FANOUT" -eq 2
      test "$PREDICTOR_MODE" = second-order-only
      test "$FIRST_ORDER_ENABLED" = false
      test "$SECOND_ORDER_ENABLED" = true
      test "$FALLBACK_PRIOR_FILL_ENABLED" = false
      test "$FANOUT_IS_UPPER_BOUND" = true
      test "$PREFETCH_GOVERNOR_ENABLED" = false
      ;;
    second-only-f1)
      test "$PREDICT_FANOUT" -eq 1
      test "$PREDICTOR_MODE" = second-order-only
      test "$FIRST_ORDER_ENABLED" = false
      test "$SECOND_ORDER_ENABLED" = true
      test "$FALLBACK_PRIOR_FILL_ENABLED" = false
      test "$FANOUT_IS_UPPER_BOUND" = true
      test "$PREFETCH_GOVERNOR_ENABLED" = false
      ;;
    second-only-f1-governed)
      test "$PREDICT_FANOUT" -eq 1
      test "$PREDICTOR_MODE" = second-order-only
      test "$FIRST_ORDER_ENABLED" = false
      test "$SECOND_ORDER_ENABLED" = true
      test "$FALLBACK_PRIOR_FILL_ENABLED" = false
      test "$FANOUT_IS_UPPER_BOUND" = true
      test "$PREFETCH_GOVERNOR_ENABLED" = true
      ;;
  esac
  prompt2_render_config "$TEMPLATE" "$TEST_ROOT/$variant.toml" \
    /mnt/localssd/model /mnt/localssd/model/tokenizer.json 1536 \
    "$PREDICT_FANOUT" "$PIPELINE_DEPTH" \
    "$FIRST_ORDER_ENABLED" "$SECOND_ORDER_ENABLED" \
    "$FALLBACK_PRIOR_FILL_ENABLED" "$FANOUT_IS_UPPER_BOUND" \
    "$PREFETCH_GOVERNOR_ENABLED"
  grep -Fx 'speculator_enabled = false' "$TEST_ROOT/$variant.toml" >/dev/null
done
diff -u \
  <(grep -v '^prefetch_governor = ' "$TEST_ROOT/current-f2.toml") \
  <(grep -v '^prefetch_governor = ' "$TEST_ROOT/current-f2-governed.toml") \
  >/dev/null
diff -u \
  <(grep -v '^prefetch_governor = ' "$TEST_ROOT/second-only-f1.toml") \
  <(grep -v '^prefetch_governor = ' "$TEST_ROOT/second-only-f1-governed.toml") \
  >/dev/null
grep -Fx 'prefetch_governor = false' "$TEST_ROOT/current-f2.toml" >/dev/null
grep -Fx 'prefetch_governor = true' "$TEST_ROOT/current-f2-governed.toml" >/dev/null
grep -Fx 'prefetch_governor = false' "$TEST_ROOT/second-only-f1.toml" >/dev/null
grep -Fx 'prefetch_governor = true' "$TEST_ROOT/second-only-f1-governed.toml" >/dev/null
unset MER_PROMPT2_PREFETCH_VARIANT

jq -n '
  def cache_io:
    {
      prefetch_enabled: false,
      prefetch_submitted: 0,
      prefetch_completed: 0,
      prefetch_used: 0,
      prefetch_bytes: 0,
      useful_prefetch_bytes: 0,
      unused_prefetch_bytes_at_sample: 0,
      prefetch_dropped_concurrency: 0,
      prefetch_dropped_pool_starved: 0,
      prefetch_dropped_governor: 0,
      prefetch_dropped_bytes: 0
    };
  def phase:
    {demand_reads_issued_while_speculative_reads_active: 0};
  {
    predictive_policy: {
      markov_prefetch_fanout: 0,
      pipeline_depth: 1,
      prefetch_governor_enabled: false,
      speculator_enabled: false
    },
    memory_layout: {
      primary_expert_pool_allocated_bytes: 5017600,
      shadow_expert_pool_allocated_bytes: 0,
      total_expert_pool_allocated_bytes: 5017600
    },
    runs: [{
      memory: {
        primary_expert_pool_allocated_bytes: 5017600,
        shadow_expert_pool_allocated_bytes: 0,
        total_expert_pool_allocated_bytes: 5017600
      },
      cache_io: cache_io,
      demand_miss_fanout: {prompt: phase, decode: phase}
    }]
  }
' > "$TEST_ROOT/no-prefetch-report.json"

jq '
  .predictive_policy = {
    markov_prefetch_fanout: 2,
    pipeline_depth: 3,
    prefetch_governor_enabled: false,
    speculator_enabled: false
  } |
  .memory_layout.shadow_expert_pool_allocated_bytes = 30105600 |
  .memory_layout.total_expert_pool_allocated_bytes = 35123200 |
  .runs[0].memory.shadow_expert_pool_allocated_bytes = 30105600 |
  .runs[0].memory.total_expert_pool_allocated_bytes = 35123200 |
  .runs[0].cache_io.prefetch_enabled = true
' "$TEST_ROOT/no-prefetch-report.json" > "$TEST_ROOT/default-report.json"
jq -e \
  --argjson predict_fanout 2 \
  --argjson pipeline_depth 3 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/default-report.json" >/dev/null

jq -e \
  --argjson predict_fanout 0 \
  --argjson pipeline_depth 1 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/no-prefetch-report.json" >/dev/null

jq '
  .predictive_policy.prefetch_governor_enabled = true |
  .runs[0].cache_io.prefetch_dropped_governor = 2
' "$TEST_ROOT/default-report.json" > "$TEST_ROOT/governed-report.json"
jq -e \
  --argjson predict_fanout 2 \
  --argjson pipeline_depth 3 \
  --argjson prefetch_governor_enabled true \
  -f "$QUALIFIER" \
  "$TEST_ROOT/governed-report.json" >/dev/null
if jq -e \
  --argjson predict_fanout 2 \
  --argjson pipeline_depth 3 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/governed-report.json" >/dev/null; then
  echo "ungoverned qualification accepted governor-enabled metadata" >&2
  exit 1
fi

jq '
  def phase4b:
    {
      schema_name: "mer-prompt2-phase4b-routing-trace",
      schema_version: 1,
      trace_path: "/tmp/phase4b.jsonl",
      max_events: 100,
      events_written: 1,
      events_dropped: 0,
      trace_truncated: false,
      trace_write_failed: false,
      lifecycle_reconciliation_passed: true,
      lifecycle_reconciliation_errors: [],
      lifecycle: {
        physical_read_issued: 0,
        physical_read_completed: 0,
        physical_read_failed: 0,
        physical_read_inflight_at_sample: 0,
        published: 0,
        publication_rejected: 0,
        completion_not_yet_published_at_sample: 0,
        first_use: 0,
        evicted_before_first_use: 0,
        still_resident_unused_at_sample: 0
      }
    };
  .phase4b_trace_enabled = true |
  .phase4b_trace = phase4b |
  .runs = [.runs[] | .phase4b_diagnostics = phase4b]
' "$TEST_ROOT/default-report.json" > "$TEST_ROOT/phase4b-report.json"
jq -e \
  --argjson predict_fanout 2 \
  --argjson pipeline_depth 3 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/phase4b-report.json" >/dev/null

jq -n '
  def critical($coverage):
    {
      wall_seconds: 1,
      attributed_seconds: $coverage,
      unattributed_residual_seconds: (1 - $coverage),
      coverage_ratio: $coverage,
      non_overlap_invariant_passed: true,
      coverage_95_percent_passed: ($coverage >= 0.95),
      qualification_passed: ($coverage >= 0.95),
      categories: [0, $coverage]
    };
  {
    phase4b_trace_enabled: false,
    runs: [
      range(0; 5) | {
        critical_path: {
          prompt: critical(0.99),
          decode: critical(0.98)
        }
      }
    ]
  }
' > "$TEST_ROOT/performance-collection.json"
jq \
  --arg qualification_kind performance-baseline \
  -f "$COLLECTION_QUALIFIER" \
  "$TEST_ROOT/performance-collection.json" \
  > "$TEST_ROOT/performance-qualification.json"
jq -e '
  .qualification_kind == "performance-baseline" and
  .collection_qualification_valid == true and
  .qualification_passed == true and
  .diagnostic_qualification_passed == null and
  .performance_qualification_applicable == true and
  .performance_qualification_passed == true
' "$TEST_ROOT/performance-qualification.json" >/dev/null

jq '.runs = .runs[:2]' "$TEST_ROOT/performance-collection.json" \
  > "$TEST_ROOT/phase4d-screening-collection.json"
jq \
  --arg qualification_kind phase4d-governor-screening \
  -f "$COLLECTION_QUALIFIER" \
  "$TEST_ROOT/phase4d-screening-collection.json" \
  > "$TEST_ROOT/phase4d-screening-qualification.json"
jq -e '
  .qualification_kind == "phase4d-governor-screening" and
  .collection_qualification_valid == true and
  .screening_collection_valid == true and
  .diagnostic_qualification_passed == null and
  .performance_qualification_applicable == false and
  .performance_qualification_passed == null and
  .qualification_passed == false and
  (.performance_qualification_reason | contains("not a qualified production performance baseline"))
' "$TEST_ROOT/phase4d-screening-qualification.json" >/dev/null

jq '
  .runs[].critical_path.prompt.coverage_ratio = 0.75 |
  .runs[].critical_path.prompt.coverage_95_percent_passed = false |
  .runs[].critical_path.prompt.qualification_passed = false
' "$TEST_ROOT/performance-collection.json" > "$TEST_ROOT/low-coverage-performance.json"
jq \
  --arg qualification_kind performance-baseline \
  -f "$COLLECTION_QUALIFIER" \
  "$TEST_ROOT/low-coverage-performance.json" \
  > "$TEST_ROOT/low-coverage-performance-qualification.json"
jq -e '
  .collection_qualification_valid == false and
  .qualification_passed == false and
  .performance_qualification_applicable == true
' "$TEST_ROOT/low-coverage-performance-qualification.json" >/dev/null

jq -n '
  def lifecycle:
    {
      physical_read_issued: 1,
      physical_read_completed: 1,
      physical_read_failed: 0,
      physical_read_inflight_at_sample: 0,
      published: 1,
      publication_rejected: 0,
      completion_not_yet_published_at_sample: 0,
      first_use: 1,
      evicted_before_first_use: 0,
      still_resident_unused_at_sample: 0
    };
  def phase4b:
    {
      schema_name: "mer-prompt2-phase4b-routing-trace",
      schema_version: 1,
      trace_path: "/tmp/phase4b.jsonl",
      max_events: 2000000,
      events_written: 1368511,
      events_dropped: 0,
      trace_truncated: false,
      trace_write_failed: false,
      lifecycle_reconciliation_passed: true,
      lifecycle_reconciliation_errors: [],
      lifecycle: lifecycle
    };
  def critical($coverage):
    {
      wall_seconds: 10,
      attributed_seconds: (10 * $coverage),
      unattributed_residual_seconds: (10 * (1 - $coverage)),
      coverage_ratio: $coverage,
      non_overlap_invariant_passed: true,
      coverage_95_percent_passed: false,
      qualification_passed: false,
      categories: [0, (10 * $coverage)]
    };
  {
    phase4b_trace_enabled: true,
    phase4b_trace: phase4b,
    runs: [
      range(0; 5) | {
        phase4b_diagnostics: phase4b,
        critical_path: {
          prompt: critical([0.7616837846, 0.6142133217, 0.4355608912, 0.3643167004, 0.2229585082][.]),
          decode: critical([0.4424307114, 0.3213106218, 0.2733205465, 0.1989155920, 0.1656464919][.])
        }
      }
    ]
  }
' > "$TEST_ROOT/phase4b-diagnostic-collection.json"
jq \
  --arg qualification_kind phase4b-diagnostic \
  -f "$COLLECTION_QUALIFIER" \
  "$TEST_ROOT/phase4b-diagnostic-collection.json" \
  > "$TEST_ROOT/phase4b-diagnostic-qualification.json"
jq -e '
  .qualification_kind == "phase4b-diagnostic" and
  .collection_qualification_valid == true and
  .diagnostic_qualification_passed == true and
  .performance_qualification_applicable == false and
  .performance_qualification_passed == null and
  .qualification_passed == false and
  .production_critical_path_coverage_gates_passed == false and
  (.performance_qualification_reason | contains("not comparable")) and
  (.observed_critical_path_coverage.prompt | length) == 5 and
  (.observed_critical_path_coverage.decode | length) == 5 and
  .observed_critical_path_coverage.prompt_min == 0.2229585082 and
  .observed_critical_path_coverage.decode_min == 0.1656464919
' "$TEST_ROOT/phase4b-diagnostic-qualification.json" >/dev/null

assert_diagnostic_rejected() {
  local fixture=$1
  local label=$2
  jq \
    --arg qualification_kind phase4b-diagnostic \
    -f "$COLLECTION_QUALIFIER" \
    "$fixture" > "$TEST_ROOT/$label-qualification.json"
  jq -e '
    .collection_qualification_valid == false and
    .diagnostic_qualification_passed == false and
    .performance_qualification_applicable == false and
    .qualification_passed == false
  ' "$TEST_ROOT/$label-qualification.json" >/dev/null
}

jq '.phase4b_trace.events_dropped = 1' \
  "$TEST_ROOT/phase4b-diagnostic-collection.json" \
  > "$TEST_ROOT/diagnostic-dropped.json"
assert_diagnostic_rejected "$TEST_ROOT/diagnostic-dropped.json" diagnostic-dropped

jq '.phase4b_trace.trace_truncated = true' \
  "$TEST_ROOT/phase4b-diagnostic-collection.json" \
  > "$TEST_ROOT/diagnostic-truncated.json"
assert_diagnostic_rejected "$TEST_ROOT/diagnostic-truncated.json" diagnostic-truncated

jq '.phase4b_trace.trace_write_failed = true' \
  "$TEST_ROOT/phase4b-diagnostic-collection.json" \
  > "$TEST_ROOT/diagnostic-writer-failed.json"
assert_diagnostic_rejected "$TEST_ROOT/diagnostic-writer-failed.json" diagnostic-writer-failed

jq '.runs[0].phase4b_diagnostics.lifecycle.published = 2' \
  "$TEST_ROOT/phase4b-diagnostic-collection.json" \
  > "$TEST_ROOT/diagnostic-lifecycle-invalid.json"
assert_diagnostic_rejected \
  "$TEST_ROOT/diagnostic-lifecycle-invalid.json" diagnostic-lifecycle-invalid

jq '.phase4b_trace.lifecycle.published = 1' \
  "$TEST_ROOT/phase4b-report.json" > "$TEST_ROOT/invalid-phase4b-report.json"
if jq -e \
  --argjson predict_fanout 2 \
  --argjson pipeline_depth 3 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/invalid-phase4b-report.json" >/dev/null; then
  echo "Phase 4B qualification accepted impossible lifecycle arithmetic" >&2
  exit 1
fi

jq '.runs[0].cache_io.prefetch_submitted = 1' \
  "$TEST_ROOT/no-prefetch-report.json" > "$TEST_ROOT/nonzero-prefetch-report.json"
if jq -e \
  --argjson predict_fanout 0 \
  --argjson pipeline_depth 1 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/nonzero-prefetch-report.json" >/dev/null; then
  echo "fanout=0 qualification accepted a nonzero prefetch counter" >&2
  exit 1
fi

jq '.runs[0].demand_miss_fanout.decode.demand_reads_issued_while_speculative_reads_active = 1' \
  "$TEST_ROOT/no-prefetch-report.json" > "$TEST_ROOT/nonzero-overlap-report.json"
if jq -e \
  --argjson predict_fanout 0 \
  --argjson pipeline_depth 1 \
  --argjson prefetch_governor_enabled false \
  -f "$QUALIFIER" \
  "$TEST_ROOT/nonzero-overlap-report.json" >/dev/null; then
  echo "fanout=0 qualification accepted speculative overlap" >&2
  exit 1
fi

assert_rejected_before_output() {
  local fanout=$1
  local depth=$2
  local artifact_dir=$3
  local status
  set +e
  MER_QWEN_CONVERTED_DIR=/does/not/exist \
  MER_EXPECTED_NVME_MOUNT=/does/not/exist \
  MER_PROMPT2_PREDICT_FANOUT="$fanout" \
  MER_PROMPT2_PIPELINE_DEPTH="$depth" \
    bash "$COLLECTOR" "$artifact_dir" >/dev/null 2>&1
  status=$?
  set -e
  test "$status" -eq 2
  test ! -e "$artifact_dir"
}

assert_rejected_before_output malformed 3 "$TEST_ROOT/malformed-fanout-output"
assert_rejected_before_output -1 3 "$TEST_ROOT/negative-fanout-output"
assert_rejected_before_output 2 malformed "$TEST_ROOT/malformed-depth-output"
assert_rejected_before_output 2 0 "$TEST_ROOT/zero-depth-output"

set +e
MER_PROMPT2_PREFETCH_VARIANT=invalid \
MER_QWEN_CONVERTED_DIR=/does/not/exist \
MER_EXPECTED_NVME_MOUNT=/does/not/exist \
  bash "$COLLECTOR" "$TEST_ROOT/invalid-variant-output" >/dev/null 2>&1
status=$?
set -e
test "$status" -eq 2
test ! -e "$TEST_ROOT/invalid-variant-output"

set +e
MER_QWEN_CONVERTED_DIR=/does/not/exist \
MER_EXPECTED_NVME_MOUNT=/does/not/exist \
  bash "$COLLECTOR" \
    "$TEST_ROOT/phase4c-missing-variant-output" \
    phase4c-untraced >/dev/null 2>&1
status=$?
set -e
test "$status" -eq 2
test ! -e "$TEST_ROOT/phase4c-missing-variant-output"

set +e
MER_QWEN_CONVERTED_DIR=/does/not/exist \
MER_EXPECTED_NVME_MOUNT=/does/not/exist \
MER_PROMPT2_PHASE4B_TRACE_PATH=/tmp/no-case-placeholder.jsonl \
  bash "$COLLECTOR" "$TEST_ROOT/invalid-phase4b-path-output" >/dev/null 2>&1
status=$?
set -e
test "$status" -eq 2
test ! -e "$TEST_ROOT/invalid-phase4b-path-output"

set +e
MER_QWEN_CONVERTED_DIR=/does/not/exist \
MER_EXPECTED_NVME_MOUNT=/does/not/exist \
  bash "$COLLECTOR" \
    "$TEST_ROOT/missing-phase4b-trace-output" \
    phase4b-diagnostic >/dev/null 2>&1
status=$?
set -e
test "$status" -eq 2
test ! -e "$TEST_ROOT/missing-phase4b-trace-output"

set +e
MER_QWEN_CONVERTED_DIR=/does/not/exist \
MER_EXPECTED_NVME_MOUNT=/does/not/exist \
MER_PROMPT2_PHASE4B_TRACE_PATH='/tmp/{case}.jsonl' \
  bash "$COLLECTOR" \
    "$TEST_ROOT/traced-performance-output" \
    four-case >/dev/null 2>&1
status=$?
set -e
test "$status" -eq 2
test ! -e "$TEST_ROOT/traced-performance-output"

sed -n \
  '/^RESIDENT_STATUS=0$/,/^elif \[\[ "\$COLLECTOR_MODE" == resident-only \]\]; then$/p' \
  "$COLLECTOR" > "$TEST_ROOT/diagnostic-case-plan.txt"
grep -F 'run_case 1536 short 14 "$SHORT_PROMPT_SHA"' \
  "$TEST_ROOT/diagnostic-case-plan.txt" >/dev/null
grep -F 'run_case 1536 medium 65 "$MEDIUM_PROMPT_SHA"' \
  "$TEST_ROOT/diagnostic-case-plan.txt" >/dev/null

sed -n \
  '/^elif \[\[ "\$COLLECTOR_MODE" == phase4c-untraced \]\]; then$/,/^elif \[\[ "\$COLLECTOR_MODE" == resident-only \]\]; then$/p' \
  "$COLLECTOR" > "$TEST_ROOT/phase4c-case-plan.txt"
grep -F 'run_case 1536 short 14 "$SHORT_PROMPT_SHA"' \
  "$TEST_ROOT/phase4c-case-plan.txt" >/dev/null
grep -F 'run_case 1536 medium 65 "$MEDIUM_PROMPT_SHA"' \
  "$TEST_ROOT/phase4c-case-plan.txt" >/dev/null
if grep -F 'run_case 6144' "$TEST_ROOT/phase4c-case-plan.txt" >/dev/null; then
  echo "Phase 4C case plan must not collect a 6144-slot case" >&2
  exit 1
fi
grep -F -- '--output-tokens "$OUTPUT_TOKENS"' "$COLLECTOR" >/dev/null
if grep -F 'run_case 6144' "$TEST_ROOT/diagnostic-case-plan.txt" >/dev/null; then
  echo "Phase 4B diagnostic case plan included a resident 6,144-slot run" >&2
  exit 1
fi

echo "Prompt 2 collector qualification fixtures: PASS"
