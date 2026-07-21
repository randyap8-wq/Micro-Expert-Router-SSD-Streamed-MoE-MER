#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COLLECTOR="$ROOT/scripts/collect_qwen3_coder_prompt2_baseline.sh"
CONFIG_HELPER="$ROOT/scripts/qwen3_coder_prompt2_collector_config.sh"
QUALIFIER="$ROOT/scripts/qwen3_coder_prompt2_prefetch_qualification.jq"
TEMPLATE="$ROOT/benchmarks/qwen3-coder-single-stream/qwen3-coder-q8.toml.in"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4a-collector.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

source "$CONFIG_HELPER"

unset MER_PROMPT2_PREDICT_FANOUT MER_PROMPT2_PIPELINE_DEPTH
prompt2_resolve_ablation_config
test "$PREDICT_FANOUT" -eq 2
test "$PIPELINE_DEPTH" -eq 3
test "$PREFETCH_EXPECTED_ACTIVE" = true
prompt2_render_config "$TEMPLATE" "$TEST_ROOT/default.toml" \
  /mnt/localssd/model /mnt/localssd/model/tokenizer.json 1536 \
  "$PREDICT_FANOUT" "$PIPELINE_DEPTH"
grep -Fx 'predict_fanout = 2' "$TEST_ROOT/default.toml" >/dev/null
grep -Fx 'pipeline_depth = 3' "$TEST_ROOT/default.toml" >/dev/null
if grep -E '@(PREDICT_FANOUT|PIPELINE_DEPTH)@' "$TEST_ROOT/default.toml" >/dev/null; then
  echo "default render left a Phase 4A placeholder unresolved" >&2
  exit 1
fi

MER_PROMPT2_PREDICT_FANOUT=0
MER_PROMPT2_PIPELINE_DEPTH=1
prompt2_resolve_ablation_config
test "$PREDICT_FANOUT" -eq 0
test "$PIPELINE_DEPTH" -eq 1
test "$PREFETCH_EXPECTED_ACTIVE" = false
prompt2_render_config "$TEMPLATE" "$TEST_ROOT/no-prefetch.toml" \
  /mnt/localssd/model /mnt/localssd/model/tokenizer.json 1536 \
  "$PREDICT_FANOUT" "$PIPELINE_DEPTH"
grep -Fx 'predict_fanout = 0' "$TEST_ROOT/no-prefetch.toml" >/dev/null
grep -Fx 'pipeline_depth = 1' "$TEST_ROOT/no-prefetch.toml" >/dev/null

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
    predictive_policy: {markov_prefetch_fanout: 0, pipeline_depth: 1},
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
  .predictive_policy = {markov_prefetch_fanout: 2, pipeline_depth: 3} |
  .memory_layout.shadow_expert_pool_allocated_bytes = 30105600 |
  .memory_layout.total_expert_pool_allocated_bytes = 35123200 |
  .runs[0].memory.shadow_expert_pool_allocated_bytes = 30105600 |
  .runs[0].memory.total_expert_pool_allocated_bytes = 35123200 |
  .runs[0].cache_io.prefetch_enabled = true
' "$TEST_ROOT/no-prefetch-report.json" > "$TEST_ROOT/default-report.json"
jq -e \
  --argjson predict_fanout 2 \
  --argjson pipeline_depth 3 \
  -f "$QUALIFIER" \
  "$TEST_ROOT/default-report.json" >/dev/null

jq -e \
  --argjson predict_fanout 0 \
  --argjson pipeline_depth 1 \
  -f "$QUALIFIER" \
  "$TEST_ROOT/no-prefetch-report.json" >/dev/null

jq '.runs[0].cache_io.prefetch_submitted = 1' \
  "$TEST_ROOT/no-prefetch-report.json" > "$TEST_ROOT/nonzero-prefetch-report.json"
if jq -e \
  --argjson predict_fanout 0 \
  --argjson pipeline_depth 1 \
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

echo "Prompt 2 Phase 4A collector fixtures: PASS"
