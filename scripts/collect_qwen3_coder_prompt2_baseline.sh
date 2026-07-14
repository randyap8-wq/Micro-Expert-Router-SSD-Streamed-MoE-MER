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
MODE_ARG=${2:-four-case}
if [[ -z "$ARTIFACT_DIR" ]]; then
  echo "usage: $0 ARTIFACT_DIR [four-case|--resident-only]" >&2
  exit 2
fi
case "$MODE_ARG" in
  four-case) COLLECTOR_MODE=four-case ;;
  resident-only|--resident-only) COLLECTOR_MODE=resident-only ;;
  *)
    echo "unknown collector mode: $MODE_ARG (expected four-case or --resident-only)" >&2
    exit 2
    ;;
esac

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
  echo "collector_mode=$COLLECTOR_MODE"
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
      .prefetch_dropped_bytes == 0
    )) and
    ([.runs[].demand_miss_fanout] | all(
      (.semantics | length > 0) and
      ([.prompt, .decode] | all(
        .routed_layers_observed > 0 and
        .routed_expert_activations_observed == (.routed_layers_observed * 8) and
        .resident_hits_at_initial_layer_lookup + .misses_at_initial_layer_lookup ==
          .routed_expert_activations_observed and
        (.missing_experts_per_routed_layer | length) == 9 and
        ([.missing_experts_per_routed_layer[]] | all(type == "number" and . >= 0)) and
        ([.missing_experts_per_routed_layer[]] | add) == .routed_layers_observed and
        ([.missing_experts_per_routed_layer | to_entries[] | (.key * .value)] | add) ==
          .misses_at_initial_layer_lookup and
        .layers_with_multiple_simultaneous_misses ==
          ([.missing_experts_per_routed_layer[2:9][]] | add) and
        (.layers_with_no_foreground_physical_read +
         .layers_with_one_physical_read +
         .layers_with_serial_physical_reads +
         .layers_with_overlapping_physical_reads) ==
          ([.missing_experts_per_routed_layer[1:9][]] | add) and
        .foreground_admission_control == "none" and
        .primary_buffer_partition == "primary-buffer-a" and
        .speculative_buffer_partition == "shadow-buffer-b" and
        .expert_compute_start_policy == "wait-for-all-required-experts" and
        .cache_lookup_seconds >= 0 and
        .foreground_physical_read_operations >= 0 and
        .peak_foreground_physical_reads_in_flight >= 0 and
        .foreground_physical_read_concurrency_integral_seconds >= 0 and
        .foreground_physical_read_active_seconds >= 0 and
        .average_foreground_physical_read_concurrency >= 0 and
        .physical_read_issue_to_completion_seconds >= 0 and
        .physical_read_issue_to_completion_mean_seconds >= 0 and
        .physical_read_issue_to_completion_max_seconds >= 0 and
        (.physical_read_issue_to_completion_histogram | length) == 16 and
        ([.physical_read_issue_to_completion_histogram[].count] | add) ==
          .foreground_physical_read_operations and
        .primary_buffer_acquisition_wait_seconds >= 0 and
        .foreground_admission_wait_seconds == 0 and
        .singleflight_wait_seconds >= 0 and
        .completion_to_consumption_delay_seconds >= 0 and
        .layer_expert_fetch_critical_path_seconds >= 0 and
        .layer_expert_fetch_critical_path_wall_fraction >= 0 and
        .layers_beginning_compute_before_all_misses_available == 0 and
        .demand_reads_issued_while_speculative_reads_active >= 0 and
        .demand_critical_reads_delayed_by_speculative_activity == null and
        (.final_straggler_routed_slot_histogram | length) == 9 and
        ([.final_straggler_routed_slot_histogram[]] | add) ==
          ([.missing_experts_per_routed_layer[1:9][]] | add)
      ))
    )) and
    ([.runs[]] | all(
      (.demand_miss_fanout.prompt.foreground_physical_read_operations +
       .demand_miss_fanout.decode.foreground_physical_read_operations) >=
        .cache_io.foreground_read_operations
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
      cache_misses_total: .aggregate.cache_misses_total,
      ssd_bytes_total: .aggregate.ssd_bytes_total,
      external_peak_rss_bytes: $peak_rss_bytes,
      storage_identity_artifact: "model-findmnt.json",
      prompt_critical_path_coverage_min: ([.runs[].critical_path.prompt.coverage_ratio] | min),
      decode_critical_path_coverage_min: ([.runs[].critical_path.decode.coverage_ratio] | min),
      output_token_parity: .aggregate.output_token_parity,
      qualification_passed: true,
      phase3a_decode: {
        routed_layers_observed: ([.runs[].demand_miss_fanout.decode.routed_layers_observed] | add),
        routed_expert_activations_observed: ([.runs[].demand_miss_fanout.decode.routed_expert_activations_observed] | add),
        resident_hits_at_initial_layer_lookup: ([.runs[].demand_miss_fanout.decode.resident_hits_at_initial_layer_lookup] | add),
        misses_at_initial_layer_lookup: ([.runs[].demand_miss_fanout.decode.misses_at_initial_layer_lookup] | add),
        cache_lookup_seconds: ([.runs[].demand_miss_fanout.decode.cache_lookup_seconds] | add),
        missing_experts_per_routed_layer: [
          range(0; 9) as $i |
          ([.runs[].demand_miss_fanout.decode.missing_experts_per_routed_layer[$i]] | add)
        ],
        layers_with_multiple_simultaneous_misses: ([.runs[].demand_miss_fanout.decode.layers_with_multiple_simultaneous_misses] | add),
        layers_with_no_foreground_physical_read: ([.runs[].demand_miss_fanout.decode.layers_with_no_foreground_physical_read] | add),
        layers_with_one_physical_read: ([.runs[].demand_miss_fanout.decode.layers_with_one_physical_read] | add),
        layers_with_serial_physical_reads: ([.runs[].demand_miss_fanout.decode.layers_with_serial_physical_reads] | add),
        layers_with_overlapping_physical_reads: ([.runs[].demand_miss_fanout.decode.layers_with_overlapping_physical_reads] | add),
        layers_beginning_compute_before_all_misses_available: ([.runs[].demand_miss_fanout.decode.layers_beginning_compute_before_all_misses_available] | add),
        foreground_physical_read_operations: ([.runs[].demand_miss_fanout.decode.foreground_physical_read_operations] | add),
        peak_foreground_physical_reads_in_flight: ([.runs[].demand_miss_fanout.decode.peak_foreground_physical_reads_in_flight] | max),
        foreground_physical_read_concurrency_integral_seconds: ([.runs[].demand_miss_fanout.decode.foreground_physical_read_concurrency_integral_seconds] | add),
        foreground_physical_read_active_seconds: ([.runs[].demand_miss_fanout.decode.foreground_physical_read_active_seconds] | add),
        average_foreground_physical_read_concurrency: (
          ([.runs[].demand_miss_fanout.decode.foreground_physical_read_concurrency_integral_seconds] | add) as $integral |
          ([.runs[].demand_miss_fanout.decode.foreground_physical_read_active_seconds] | add) as $active |
          if $active == 0 then 0 else ($integral / $active) end
        ),
        physical_read_issue_to_completion_seconds: ([.runs[].demand_miss_fanout.decode.physical_read_issue_to_completion_seconds] | add),
        physical_read_issue_to_completion_mean_seconds: (
          ([.runs[].demand_miss_fanout.decode.physical_read_issue_to_completion_seconds] | add) as $service |
          ([.runs[].demand_miss_fanout.decode.foreground_physical_read_operations] | add) as $reads |
          if $reads == 0 then 0 else ($service / $reads) end
        ),
        physical_read_issue_to_completion_max_seconds: ([.runs[].demand_miss_fanout.decode.physical_read_issue_to_completion_max_seconds] | max),
        physical_read_issue_to_completion_histogram: [
          range(0; 16) as $i |
          {
            upper_bound_microseconds: .runs[0].demand_miss_fanout.decode.physical_read_issue_to_completion_histogram[$i].upper_bound_microseconds,
            count: ([.runs[].demand_miss_fanout.decode.physical_read_issue_to_completion_histogram[$i].count] | add)
          }
        ],
        primary_buffer_acquisition_wait_seconds: ([.runs[].demand_miss_fanout.decode.primary_buffer_acquisition_wait_seconds] | add),
        foreground_admission_wait_seconds: ([.runs[].demand_miss_fanout.decode.foreground_admission_wait_seconds] | add),
        singleflight_wait_seconds: ([.runs[].demand_miss_fanout.decode.singleflight_wait_seconds] | add),
        completion_to_consumption_delay_seconds: ([.runs[].demand_miss_fanout.decode.completion_to_consumption_delay_seconds] | add),
        first_miss_to_first_read_issue_seconds: ([.runs[].demand_miss_fanout.decode.first_miss_to_first_read_issue_seconds] | add),
        first_miss_to_last_read_issue_seconds: ([.runs[].demand_miss_fanout.decode.first_miss_to_last_read_issue_seconds] | add),
        first_to_last_read_issue_spread_seconds: ([.runs[].demand_miss_fanout.decode.first_to_last_read_issue_spread_seconds] | add),
        first_miss_to_first_required_expert_available_seconds: ([.runs[].demand_miss_fanout.decode.first_miss_to_first_required_expert_available_seconds] | add),
        first_miss_to_final_required_expert_available_seconds: ([.runs[].demand_miss_fanout.decode.first_miss_to_final_required_expert_available_seconds] | add),
        first_to_last_required_expert_completion_spread_seconds: ([.runs[].demand_miss_fanout.decode.first_to_last_required_expert_completion_spread_seconds] | add),
        first_miss_to_expert_compute_begin_seconds: ([.runs[].demand_miss_fanout.decode.first_miss_to_expert_compute_begin_seconds] | add),
        first_miss_to_layer_completion_seconds: ([.runs[].demand_miss_fanout.decode.first_miss_to_layer_completion_seconds] | add),
        layer_expert_fetch_critical_path_seconds: ([.runs[].demand_miss_fanout.decode.layer_expert_fetch_critical_path_seconds] | add),
        layer_expert_fetch_critical_path_mean_seconds: (
          ([.runs[].demand_miss_fanout.decode.layer_expert_fetch_critical_path_seconds] | add) as $critical |
          ([.runs[].demand_miss_fanout.decode.missing_experts_per_routed_layer[1:9][]] | add) as $layers |
          if $layers == 0 then 0 else ($critical / $layers) end
        ),
        layer_expert_fetch_critical_path_max_seconds: ([.runs[].demand_miss_fanout.decode.layer_expert_fetch_critical_path_max_seconds] | max),
        decode_wall_seconds: ([.runs[].decode_seconds] | add),
        decode_wall_fraction_attributable_to_layer_expert_fetch: (
          ([.runs[].demand_miss_fanout.decode.layer_expert_fetch_critical_path_seconds] | add) /
          ([.runs[].decode_seconds] | add)
        ),
        demand_reads_issued_while_speculative_reads_active: ([.runs[].demand_miss_fanout.decode.demand_reads_issued_while_speculative_reads_active] | add),
        demand_critical_reads_delayed_by_speculative_activity: null,
        final_straggler_routed_slot_histogram: [
          range(0; 9) as $i |
          ([.runs[].demand_miss_fanout.decode.final_straggler_routed_slot_histogram[$i]] | add)
        ],
        worst_layer_fetch: (
          [.runs[] |
            select(.demand_miss_fanout.decode.worst_layer_fetch != null) |
            {
              run_index,
              sample: .demand_miss_fanout.decode.worst_layer_fetch
            }
          ] |
          if length == 0 then null else max_by(.sample.critical_path_seconds) end
        )
      }
    }' "$json" > "$ARTIFACT_DIR/$stem.case-summary.json"
}

RESIDENT_STATUS=0
if [[ "$COLLECTOR_MODE" == resident-only ]]; then
  run_case 6144 short 14 "$SHORT_PROMPT_SHA"
  run_case 6144 medium 65 "$MEDIUM_PROMPT_SHA"

  if bash "$ROOT/scripts/finalize_qwen3_coder_prompt2_resident.sh" \
    "$ARTIFACT_DIR" "$EXPECTED_COMMIT"; then
    RESIDENT_STATUS=0
  else
    RESIDENT_STATUS=$?
  fi
  if (( RESIDENT_STATUS != 0 && RESIDENT_STATUS != 1 )); then
    echo "resident-only collection is incomplete; final artifacts were not written" >&2
    exit "$RESIDENT_STATUS"
  fi
else
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
fi

# Hash immutable artifacts last. The two tee-backed collector logs and this
# manifest are excluded because they are still open while the manifest is made.
find "$ARTIFACT_DIR" -type f \
  ! -name 'collector.log' \
  ! -name 'collector.stderr.log' \
  ! -name 'artifact-sha256.txt' \
  -print0 | sort -z | xargs -0 sha256sum > "$ARTIFACT_DIR/artifact-sha256.txt"

if (( RESIDENT_STATUS == 1 )); then
  echo "QUALIFICATION: FAIL"
  echo "resident-only collection completed with an auditable gate failure: $ARTIFACT_DIR"
  exit 1
fi

echo "QUALIFICATION: PASS"
echo "$COLLECTOR_MODE instrumented baseline collection completed: $ARTIFACT_DIR"
