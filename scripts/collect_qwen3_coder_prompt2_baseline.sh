#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FIXTURES="$ROOT/benchmarks/qwen3-coder-single-stream"
TEMPLATE="$FIXTURES/qwen3-coder-q8.toml.in"
FEATURES="avx512,blas,tokenizer,io_uring,q8-candle-reference"

: "${MER_QWEN_CONVERTED_DIR:?set MER_QWEN_CONVERTED_DIR to the converted Qwen directory on local NVMe}"
: "${MER_EXPECTED_NVME_MOUNT:?set MER_EXPECTED_NVME_MOUNT to the local-NVMe mount, for example /mnt/localssd}"

TOKENIZER=${MER_QWEN_TOKENIZER:-$MER_QWEN_CONVERTED_DIR/tokenizer.json}
ARTIFACT_DIR=${1:-}
if [[ -z "$ARTIFACT_DIR" ]]; then
  echo "usage: $0 ARTIFACT_DIR" >&2
  exit 2
fi

for value in "$MER_QWEN_CONVERTED_DIR" "$TOKENIZER"; do
  if [[ "$value" == *'&'* || "$value" == *'|'* || "$value" == *$'\n'* ]]; then
    echo "benchmark paths must not contain &, |, or a newline: $value" >&2
    exit 2
  fi
done

command -v realpath >/dev/null
command -v findmnt >/dev/null
MODEL_REAL=$(realpath -e "$MER_QWEN_CONVERTED_DIR")
MOUNT_REAL=$(realpath -e "$MER_EXPECTED_NVME_MOUNT")
TOKENIZER_REAL=$(realpath -e "$TOKENIZER")
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Prompt 2 baselines require Linux x86_64; found $(uname -s) $(uname -m)" >&2
  exit 1
fi
if [[ "$(getconf _NPROCESSORS_ONLN)" != 32 ]]; then
  echo "Prompt 2 baselines require 32 online logical CPUs; found $(getconf _NPROCESSORS_ONLN)" >&2
  exit 1
fi
case "$MODEL_REAL/" in
  "$MOUNT_REAL/"*) ;;
  *)
    echo "converted model is not below expected local-NVMe mount: model=$MODEL_REAL mount=$MOUNT_REAL" >&2
    exit 1
    ;;
esac
ACTUAL_MOUNT=$(findmnt -T "$MODEL_REAL" -n -o TARGET)
if [[ "$(realpath -e "$ACTUAL_MOUNT")" != "$MOUNT_REAL" ]]; then
  echo "model filesystem mount does not match MER_EXPECTED_NVME_MOUNT: actual=$ACTUAL_MOUNT expected=$MOUNT_REAL" >&2
  exit 1
fi

test -f "$MODEL_REAL/config.json"
test -f "$MODEL_REAL/dense_manifest.json"
test -f "$TOKENIZER_REAL"
command -v jq >/dev/null
command -v lscpu >/dev/null
command -v lsblk >/dev/null
command -v findmnt >/dev/null
command -v taskset >/dev/null
command -v sha256sum >/dev/null
test -x /usr/bin/time

if [[ -n "$(git -C "$ROOT" status --short)" && "${MER_ALLOW_DIRTY:-0}" != 1 ]]; then
  echo "refusing a baseline from a dirty worktree; commit/stash changes or set MER_ALLOW_DIRTY=1 for a non-qualifying diagnostic" >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR/configs"
ARTIFACT_DIR=$(realpath -e "$ARTIFACT_DIR")

exec > >(tee -a "$ARTIFACT_DIR/collector.log")
exec 2> >(tee -a "$ARTIFACT_DIR/collector.stderr.log" >&2)

{
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_path=$MODEL_REAL"
  echo "tokenizer_path=$TOKENIZER_REAL"
  echo "expected_nvme_mount=$MOUNT_REAL"
  echo "requested_cpu_mask=0-31"
  echo "requested_rayon_threads=30"
  echo "cargo_features=$FEATURES"
  echo "omp_num_threads=1"
  echo "openblas_num_threads=1"
  echo "mkl_num_threads=1"
  echo "blis_num_threads=1"
  echo "malloc_arena_max=2"
  echo "rust_log=off"
  echo "no_color=1"
} > "$ARTIFACT_DIR/environment.txt"

uname -a > "$ARTIFACT_DIR/uname.txt"
lscpu > "$ARTIFACT_DIR/lscpu.txt"
lsblk -o NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL > "$ARTIFACT_DIR/lsblk.txt"
findmnt > "$ARTIFACT_DIR/findmnt.txt"
findmnt -T "$MODEL_REAL" -o TARGET,SOURCE,FSTYPE,OPTIONS > "$ARTIFACT_DIR/model-findmnt.txt"
findmnt --json -T "$MODEL_REAL" -o TARGET,SOURCE,FSTYPE,OPTIONS > "$ARTIFACT_DIR/model-findmnt.json"
jq -e --arg mount "$MOUNT_REAL" '
  .filesystems[0].target == $mount and
  .filesystems[0].fstype == "ext4" and
  ((["rw", "noatime", "nodiratime"] - (.filesystems[0].options | split(","))) | length == 0)
' "$ARTIFACT_DIR/model-findmnt.json" >/dev/null
df -T "$MODEL_REAL" > "$ARTIFACT_DIR/model-df.txt"
grep -E 'MemTotal|MemAvailable' /proc/meminfo > "$ARTIFACT_DIR/memory-preflight.txt"
taskset -pc $$ > "$ARTIFACT_DIR/collector-effective-cpu-mask.txt"
git -C "$ROOT" rev-parse HEAD > "$ARTIFACT_DIR/git-commit.txt"
git -C "$ROOT" status --short > "$ARTIFACT_DIR/git-status-short.txt"
EXPECTED_COMMIT=$(tr -d '\n' < "$ARTIFACT_DIR/git-commit.txt")

jq -j .prompt "$FIXTURES/prompts/short.json" | sha256sum > "$ARTIFACT_DIR/prompt-short.sha256"
jq -j .prompt "$FIXTURES/prompts/medium.json" | sha256sum > "$ARTIFACT_DIR/prompt-medium.sha256"
SHORT_PROMPT_SHA=$(awk '{print $1}' "$ARTIFACT_DIR/prompt-short.sha256")
MEDIUM_PROMPT_SHA=$(awk '{print $1}' "$ARTIFACT_DIR/prompt-medium.sha256")
sha256sum "$MODEL_REAL/config.json" "$MODEL_REAL/dense_manifest.json" > "$ARTIFACT_DIR/checkpoint-metadata.sha256"
CONFIG_SHA=$(sha256sum "$MODEL_REAL/config.json" | awk '{print $1}')
DENSE_MANIFEST_SHA=$(sha256sum "$MODEL_REAL/dense_manifest.json" | awk '{print $1}')

render_config() {
  local slots=$1
  local output=$2
  sed \
    -e "s|@MODEL_DIR@|$MODEL_REAL|g" \
    -e "s|@TOKENIZER_PATH@|$TOKENIZER_REAL|g" \
    -e "s|@CACHE_SLOTS@|$slots|g" \
    "$TEMPLATE" > "$output"
}

render_config 1536 "$ARTIFACT_DIR/configs/qwen3-coder-q8-1536.toml"
render_config 6144 "$ARTIFACT_DIR/configs/qwen3-coder-q8-6144.toml"

# Physical pool sizing from build_bench_real_runtime:
# primary=(cache_slots+1), shadow=predict_fanout*pipeline_depth=6.
for slots in 1536 6144; do
  primary_slots=$((slots + 1))
  shadow_slots=6
  primary_bytes=$((primary_slots * 5017600))
  shadow_bytes=$((shadow_slots * 5017600))
  total_bytes=$((primary_bytes + shadow_bytes))
  printf 'cache_slots=%s primary_slots=%s shadow_slots=%s primary_bytes=%s shadow_bytes=%s total_pool_bytes=%s\n' \
    "$slots" "$primary_slots" "$shadow_slots" "$primary_bytes" "$shadow_bytes" "$total_bytes"
done > "$ARTIFACT_DIR/pool-sizing.txt"

unset RAYON_NUM_THREADS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export RUST_LOG=off
export NO_COLOR=1

if [[ "${MER_SKIP_BUILD:-0}" != 1 ]]; then
  cargo build \
    --manifest-path "$ROOT/rust-engine/Cargo.toml" \
    --release \
    --features "$FEATURES" \
    2>&1 | tee "$ARTIFACT_DIR/cargo-build.log"
fi

BIN=${MER_BIN:-$ROOT/rust-engine/target/release/micro-expert-router}
test -x "$BIN"
sha256sum "$BIN" > "$ARTIFACT_DIR/binary.sha256"

run_case() {
  local slots=$1
  local prompt_id=$2
  local expected_prompt_tokens=$3
  local expected_prompt_sha=$4
  local stem="baseline-${slots}-${prompt_id}"
  local json="$ARTIFACT_DIR/$stem.json"
  local request="$FIXTURES/prompts/$prompt_id.json"
  local config="$ARTIFACT_DIR/configs/qwen3-coder-q8-$slots.toml"

  /usr/bin/time -v \
    -o "$ARTIFACT_DIR/$stem.time.txt" \
    "$BIN" \
    --rayon-threads 30 \
    --cpu-mask 0-31 \
    bench-real \
    --config "$config" \
    --request-json "$request" \
    --output-tokens 128 \
    --warmup-runs 1 \
    --measured-runs 5 \
    --cache-reset keep \
    --greedy \
    --format json \
    > "$json" \
    2> "$ARTIFACT_DIR/$stem.stderr.log"

  test -s "$json"
  jq empty "$json"
  jq -e \
    --arg commit "$EXPECTED_COMMIT" \
    --arg prompt_id "$prompt_id" \
    --arg prompt_sha "$expected_prompt_sha" \
    --arg config_sha "$CONFIG_SHA" \
    --arg dense_manifest_sha "$DENSE_MANIFEST_SHA" \
    --argjson slots "$slots" \
    --argjson prompt_tokens "$expected_prompt_tokens" '
    .schema == {"name":"mer-bench-real","version":2} and
    .benchmark == "bench-real" and
    .warmup_runs == 1 and
    .measured_runs == 5 and
    .cache_reset == "keep" and
    .greedy == true and
    .execution.git_commit_full == $commit and
    .execution.worktree_dirty_at_execution == false and
    .execution.build_profile == "release" and
    (.execution.cargo_features | contains(["tokenizer", "io_uring", "blas", "avx512", "q8-candle-reference"])) and
    .execution.operating_system == "linux" and
    .execution.architecture == "x86_64" and
    (.execution.cpu_vendor | length > 0) and
    (.execution.cpu_model | length > 0) and
    (.execution.cpu_instruction_features | contains(["avx2", "fma"])) and
    .execution.logical_cpu_count == 32 and
    .execution.requested_cpu_mask == "0-31" and
    .execution.requested_cpu_mask_source == "cli" and
    .execution.effective_cpu_affinity == "0-31" and
    .execution.requested_rayon_workers == 30 and
    .execution.actual_rayon_pool_size == 30 and
    .execution.rayon_selection_source == "cli" and
    .execution.dense_matvec_backend == "rayon-matrixmultiply" and
    .execution.direct_q8_expert_backend == "avx2-fma" and
    .execution.direct_q8_avx2_fma_selected == true and
    .execution.direct_q8_avx512_selected == false and
    (.model.checkpoint_identifier | length > 0) and
    .model.config_json_sha256 == $config_sha and
    .model.dense_manifest_sha256 == $dense_manifest_sha and
    (.model.converted_manifest_identity | test("^[0-9a-f]{64}$")) and
    .model.architecture == "qwen3_moe" and
    .model.d_model == 2048 and
    .model.d_ff == 768 and
    .model.layer_count == 48 and
    .model.expert_count_per_layer == 128 and
    .model.layer_qualified_expert_count == 6144 and
    .model.top_k == 8 and
    .model.query_head_count == 32 and
    .model.kv_head_count == 4 and
    .model.head_dim == 128 and
    .model.vocab_size == 151936 and
    (.model.dense_dtype | contains("q8_0")) and
    .model.expert_dtype == "q8_0" and
    .prompt_identity.fixture_identifier == $prompt_id and
    .prompt_identity.sha256 == $prompt_sha and
    .prompt_identity.requested_completion_tokens == 128 and
    .storage.active_expert_io_backend == "pread-odirect" and
    .storage.direct_io_enabled == true and
    .storage.io_uring_compiled == true and
    .storage.io_uring_active_for_expert_reads == false and
    .storage.packed_expert_storage == false and
    .strictness.strict_weights == true and
    (.strictness.loader | length > 0) and
    .strictness.required_tensor_count > 0 and
    .strictness.loaded_tensor_count == .strictness.required_tensor_count and
    .strictness.seeded_fallback_remained == false and
    .strictness.inference_policy == {
      "allow_degraded_experts":false,
      "allow_nonfinite_attention_fallback":false,
      "allow_truncated_expert_payloads":false
    } and
    .memory_layout.primary_expert_pool_allocated_bytes > 0 and
    .memory_layout.shadow_expert_pool_allocated_bytes > 0 and
    .memory_layout.total_expert_pool_allocated_bytes ==
      (.memory_layout.primary_expert_pool_allocated_bytes + .memory_layout.shadow_expert_pool_allocated_bytes) and
    .memory_layout.prepared_duplicate_expert_bytes == 0 and
    .memory_layout.external_peak_rss_source == "collector:/usr/bin/time-v" and
    .predictive_policy == {
      "markov_prefetch_fanout":2,
      "pipeline_depth":3,
      "locality_enabled":false,
      "speculator_enabled":false,
      "affinity_enabled":false,
      "prefetch_governor_enabled":false,
      "cost_aware_eviction_enabled":false,
      "pregate_enabled":false,
      "static_residency_fraction":0
    } and
    .aggregate.output_token_parity == true and
    (.runs | length == 5) and
    ([.runs[].prompt_tokens] | all(. == $prompt_tokens)) and
    ([.runs[].completion_tokens] | all(. == 128)) and
    ([.runs[].total_api_tokens] | all(. == ($prompt_tokens + 128))) and
    ([.runs[].output_token_ids] | unique | length == 1) and
    ([.runs[].correctness] | all(
      .strict_weights == true and
      .required_tensor_count > 0 and
      .loaded_tensor_count == .required_tensor_count and
      .seeded_fallback_remained == false and
      .degraded_expert_substitutions == 0 and
      .expert_read_failures == 0 and
      .truncated_expert_payload_uses == 0 and
      .nonfinite_attention_fallbacks == 0 and
      has("nonfinite_output_count") and
      .q8_scalar_layout_fallbacks == 0 and
      .q8_direct_kernel_dispatches > 0 and
      .prepared_duplicate_expert_bytes == 0 and
      (.inference_policy | all(. == false))
    )) and
    ([.runs[].cache_io] | all(
      .cache_capacity_experts == $slots and
      .cache_resident_experts_at_sample >= 0 and
      .cache_resident_experts_at_sample <= .cache_capacity_experts and
      .shadow_resident_experts_at_sample >= 0 and
      .expert_read_failures == 0 and
      .prefetch_enabled == true and
      .cache_evictions >= 0 and
      .foreground_read_operations >= 0 and
      .foreground_read_operations_issued >= .foreground_read_operations and
      .foreground_expert_bytes >= 0 and
      .foreground_expert_io_wait_seconds >= 0 and
      .total_expert_bytes_read >= 0 and
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
      .speculative_read_operations_issued >= 0 and
      .demand_requests_joined_inflight_prefetch >= 0 and
      .demand_requests_joined_inflight_foreground >= 0 and
      .speculative_loads_promoted_to_demand >= 0 and
      .duplicate_physical_reads_avoided ==
        (.demand_requests_joined_inflight_prefetch + .demand_requests_joined_inflight_foreground) and
      .completed_prefetch_direct_handoffs >= 0 and
      .in_flight_registry_peak_size >= .in_flight_registry_size_at_sample and
      .in_flight_entries_removed >= .in_flight_failed_or_abandoned_entries_removed and
      .in_flight_failed_or_abandoned_entries_removed >= 0
    )) and
    ([.runs[].memory] | all(
      .current_rss_bytes > 0 and
      .current_rss_sample_point == "after_completion_decode_before_report_serialization" and
      .resident_expert_buffer_bytes >= 0 and
      .primary_expert_pool_allocated_bytes > 0 and
      .shadow_expert_pool_allocated_bytes > 0 and
      .total_expert_pool_allocated_bytes ==
        (.primary_expert_pool_allocated_bytes + .shadow_expert_pool_allocated_bytes) and
      .prepared_duplicate_expert_bytes == 0 and
      .external_peak_rss_bytes == null
    )) and
    ([.runs[].critical_path.prompt, .runs[].critical_path.decode] | all(
      .wall_seconds > 0 and
      .attributed_seconds >= 0 and
      has("unattributed_residual_seconds") and
      .coverage_ratio >= 0.95 and
      .non_overlap_invariant_passed == true and
      .coverage_95_percent_passed == true and
      .qualification_passed == true and
      ([.categories[]] | all(type == "number" and . >= 0))
    ))
  ' "$json" >/dev/null

  local peak_rss_kib
  peak_rss_kib=$(awk -F: '/Maximum resident set size \(kbytes\)/ {gsub(/[[:space:]]/, "", $2); print $2}' "$ARTIFACT_DIR/$stem.time.txt")
  if [[ ! "$peak_rss_kib" =~ ^[0-9]+$ || "$peak_rss_kib" == 0 ]]; then
    echo "failed to parse external peak RSS for $stem" >&2
    exit 1
  fi
  local peak_rss_bytes=$((peak_rss_kib * 1024))
  jq \
    --arg case "$stem" \
    --arg prompt_id "$prompt_id" \
    --argjson cache_slots "$slots" \
    --argjson peak_rss_bytes "$peak_rss_bytes" \
    '{
      schema: {name:"mer-prompt2-case-summary", version:1},
      case: $case,
      cache_slots: $cache_slots,
      prompt_fixture: $prompt_id,
      prompt_tokens: ([.runs[].prompt_tokens] | unique | first),
      requested_completion_tokens: .prompt_identity.requested_completion_tokens,
      actual_completion_tokens: ([.runs[].completion_tokens] | unique | first),
      decode_tps_mean: .aggregate.decode_tps_mean,
      prompt_tps_mean: .aggregate.prompt_tps_mean,
      time_to_first_token_p50_seconds: .aggregate.time_to_first_token_p50_seconds,
      decode_seconds_mean: .aggregate.decode_seconds_mean,
      total_seconds_mean: .aggregate.total_seconds_mean,
      cache_hit_rate: .aggregate.hit_rate,
      cache_hits_total: .aggregate.cache_hits_total,
      cache_misses_total: .aggregate.cache_misses_total,
      cache_evictions_total: ([.runs[].cache_io.cache_evictions] | add),
      ssd_bytes_total: .aggregate.ssd_bytes_total,
      foreground_read_operations_total: ([.runs[].cache_io.foreground_read_operations] | add),
      foreground_read_operations_issued_total: ([.runs[].cache_io.foreground_read_operations_issued] | add),
      foreground_expert_bytes_total: ([.runs[].cache_io.foreground_expert_bytes] | add),
      foreground_expert_io_wait_seconds_total: ([.runs[].cache_io.foreground_expert_io_wait_seconds] | add),
      prefetch_submitted_total: ([.runs[].cache_io.prefetch_submitted] | add),
      prefetch_completed_total: ([.runs[].cache_io.prefetch_completed] | add),
      prefetch_used_total: ([.runs[].cache_io.prefetch_used] | add),
      prefetch_bytes_total: ([.runs[].cache_io.prefetch_bytes] | add),
      useful_prefetch_bytes_total: ([.runs[].cache_io.useful_prefetch_bytes] | add),
      unused_prefetch_bytes_at_sample_total: ([.runs[].cache_io.unused_prefetch_bytes_at_sample] | add),
      speculative_read_operations_issued_total: ([.runs[].cache_io.speculative_read_operations_issued] | add),
      demand_requests_joined_inflight_prefetch_total: ([.runs[].cache_io.demand_requests_joined_inflight_prefetch] | add),
      demand_requests_joined_inflight_foreground_total: ([.runs[].cache_io.demand_requests_joined_inflight_foreground] | add),
      speculative_loads_promoted_to_demand_total: ([.runs[].cache_io.speculative_loads_promoted_to_demand] | add),
      duplicate_physical_reads_avoided_total: ([.runs[].cache_io.duplicate_physical_reads_avoided] | add),
      completed_prefetch_direct_handoffs_total: ([.runs[].cache_io.completed_prefetch_direct_handoffs] | add),
      in_flight_registry_peak_size: ([.runs[].cache_io.in_flight_registry_peak_size] | max),
      in_flight_entries_removed_total: ([.runs[].cache_io.in_flight_entries_removed] | add),
      in_flight_failed_or_abandoned_entries_removed_total: ([.runs[].cache_io.in_flight_failed_or_abandoned_entries_removed] | add),
      external_peak_rss_bytes: $peak_rss_bytes,
      storage_identity_artifact: "model-findmnt.json",
      prompt_critical_path_coverage_min: ([.runs[].critical_path.prompt.coverage_ratio] | min),
      decode_critical_path_coverage_min: ([.runs[].critical_path.decode.coverage_ratio] | min),
      output_token_parity: .aggregate.output_token_parity,
      qualification_passed: true
    }' "$json" > "$ARTIFACT_DIR/$stem.case-summary.json"
}

run_case 1536 short 14 "$SHORT_PROMPT_SHA"
run_case 1536 medium 65 "$MEDIUM_PROMPT_SHA"
run_case 6144 short 14 "$SHORT_PROMPT_SHA"
run_case 6144 medium 65 "$MEDIUM_PROMPT_SHA"

jq -s '{
  schema: {name:"mer-prompt2-four-case-summary", version:1},
  cases: .,
  qualification_passed: (all(.qualification_passed == true))
}' \
  "$ARTIFACT_DIR/baseline-1536-short.case-summary.json" \
  "$ARTIFACT_DIR/baseline-1536-medium.case-summary.json" \
  "$ARTIFACT_DIR/baseline-6144-short.case-summary.json" \
  "$ARTIFACT_DIR/baseline-6144-medium.case-summary.json" \
  > "$ARTIFACT_DIR/four-case-summary.json"

jq -n \
  --arg commit "$EXPECTED_COMMIT" \
  --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema: {name:"mer-prompt2-qualification", version:1},
    git_commit_full: $commit,
    captured_at_utc: $captured_at_utc,
    four_required_cases_present: true,
    schema_and_required_fields_passed: true,
    provenance_and_backend_gates_passed: true,
    prompt_identity_gates_passed: true,
    strictness_and_correctness_gates_passed: true,
    critical_path_coverage_gates_passed: true,
    external_peak_rss_present: true,
    qualification_passed: true
  }' > "$ARTIFACT_DIR/qualification.json"

# Hash immutable artifacts last. The two tee-backed collector logs and this
# manifest are excluded because they are still open while the manifest is made.
find "$ARTIFACT_DIR" -type f \
  ! -name 'collector.log' \
  ! -name 'collector.stderr.log' \
  ! -name 'artifact-sha256.txt' \
  -print0 | sort -z | xargs -0 sha256sum > "$ARTIFACT_DIR/artifact-sha256.txt"

echo "QUALIFICATION: PASS"
echo "four-case instrumented baseline collection completed: $ARTIFACT_DIR"
