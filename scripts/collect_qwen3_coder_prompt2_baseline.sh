#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FIXTURES="$ROOT/benchmarks/qwen3-coder-single-stream"
TEMPLATE="$FIXTURES/qwen3-coder-q8.toml.in"
FEATURES="avx512,blas,tokenizer,io_uring,q8-candle-reference"
COLLECTION_QUALIFIER="$ROOT/scripts/qwen3_coder_prompt2_collection_qualification.jq"
source "$ROOT/scripts/qwen3_coder_prompt2_collector_config.sh"

: "${MER_QWEN_CONVERTED_DIR:?set MER_QWEN_CONVERTED_DIR to the converted Qwen directory on local NVMe}"
: "${MER_EXPECTED_NVME_MOUNT:?set MER_EXPECTED_NVME_MOUNT to the local-NVMe mount, for example /mnt/localssd}"

prompt2_resolve_ablation_config

TOKENIZER=${MER_QWEN_TOKENIZER:-$MER_QWEN_CONVERTED_DIR/tokenizer.json}
ARTIFACT_DIR=${1:-}
MODE_ARG=${2:-four-case}
if [[ -z "$ARTIFACT_DIR" ]]; then
  echo "usage: $0 ARTIFACT_DIR [four-case|phase4c-untraced|phase4d-screening|phase4d-b-score-calibration|phase4d-c-sparse-admission|phase4b-diagnostic|--resident-only]" >&2
  exit 2
fi
case "$MODE_ARG" in
  four-case)
    COLLECTOR_MODE=four-case
    QUALIFICATION_KIND=performance-baseline
    EXPERIMENT_NAME=prompt2-phase4a-prefetch-ablation
    ;;
  phase4c-untraced)
    COLLECTOR_MODE=phase4c-untraced
    QUALIFICATION_KIND=performance-baseline
    EXPERIMENT_NAME=prompt2-phase4c-precision-first-prefetch
    if [[ "$PREFETCH_VARIANT" == custom ]]; then
      echo "phase4c-untraced requires a named Phase 4C matrix variant" >&2
      exit 2
    fi
    ;;
  phase4b-diagnostic)
    COLLECTOR_MODE=phase4b-diagnostic
    QUALIFICATION_KIND=phase4b-diagnostic
    EXPERIMENT_NAME=prompt2-phase4b-routing-trace-diagnostic
    ;;
  phase4d-screening)
    COLLECTOR_MODE=phase4d-screening
    QUALIFICATION_KIND=phase4d-governor-screening
    EXPERIMENT_NAME=prompt2-phase4d-governor-screening
    case "$PREFETCH_VARIANT" in
      demand-only|second-only-f1|second-only-f1-governed-current|second-only-f1-governed-cw025|second-only-f1-governed-bt010-cw025|second-only-f1-governed-bt005-cw000) ;;
      *)
        echo "phase4d-screening requires a named Phase 4D-A screening variant; found: $PREFETCH_VARIANT" >&2
        exit 2
        ;;
    esac
    ;;
  phase4d-b-score-calibration)
    COLLECTOR_MODE=phase4d-b-score-calibration
    QUALIFICATION_KIND=phase4d-b-score-calibration-diagnostic
    EXPERIMENT_NAME=prompt2-phase4d-b-governor-score-calibration
    case "$PREFETCH_VARIANT" in
      demand-only|second-only-f1|second-only-f1-governed-current|second-only-f1-governed-bt005-cw000) ;;
      *)
        echo "phase4d-b-score-calibration requires a named Phase 4D-B diagnostic variant; found: $PREFETCH_VARIANT" >&2
        exit 2
        ;;
    esac
    ;;
  phase4d-c-sparse-admission)
    COLLECTOR_MODE=phase4d-c-sparse-admission
    QUALIFICATION_KIND=phase4d-c-sparse-admission-screening
    EXPERIMENT_NAME=prompt2-phase4d-c-sparse-admission
    case "$PREFETCH_VARIANT" in
      demand-only|second-only-f1|second-only-f1-governed-bt001-cw000|second-only-f1-governed-bt0005-cw000|second-only-f1-governed-bt00025-cw000) ;;
      *)
        echo "phase4d-c-sparse-admission requires a named Phase 4D-C screening variant; found: $PREFETCH_VARIANT" >&2
        exit 2
        ;;
    esac
    ;;
  resident-only|--resident-only)
    COLLECTOR_MODE=resident-only
    QUALIFICATION_KIND=performance-baseline
    EXPERIMENT_NAME=prompt2-phase4a-prefetch-ablation
    ;;
  *)
    echo "unknown collector mode: $MODE_ARG (expected four-case, phase4c-untraced, phase4d-screening, phase4d-b-score-calibration, phase4d-c-sparse-admission, phase4b-diagnostic, or --resident-only)" >&2
    exit 2
    ;;
esac

OUTPUT_TOKENS=128
MEASURED_RUNS=5
if [[ "$COLLECTOR_MODE" == phase4b-diagnostic ]]; then
  OUTPUT_TOKENS=${MER_PROMPT2_PHASE4B_OUTPUT_TOKENS-128}
  if [[ ! "$OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MER_PROMPT2_PHASE4B_OUTPUT_TOKENS must be a positive integer" >&2
    exit 2
  fi
fi
if [[ "$COLLECTOR_MODE" == phase4d-screening ||
      "$COLLECTOR_MODE" == phase4d-b-score-calibration ||
      "$COLLECTOR_MODE" == phase4d-c-sparse-admission ]]; then
  MEASURED_RUNS=2
fi

PHASE4B_TRACE_TEMPLATE=${MER_PROMPT2_PHASE4B_TRACE_PATH:-}
PHASE4B_TRACE_MAX_EVENTS=${MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS:-1000000}
if [[ -n "$PHASE4B_TRACE_TEMPLATE" ]]; then
  if [[ "$PHASE4B_TRACE_TEMPLATE" != *'{case}'* ]]; then
    echo "MER_PROMPT2_PHASE4B_TRACE_PATH must contain {case} for collector runs (one bounded trace per case)" >&2
    exit 2
  fi
  if [[ ! "$PHASE4B_TRACE_MAX_EVENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS must be a positive integer" >&2
    exit 2
  fi
  PHASE4B_TRACE_ENABLED=true
else
  PHASE4B_TRACE_ENABLED=false
fi
if [[ "$COLLECTOR_MODE" == phase4b-diagnostic && "$PHASE4B_TRACE_ENABLED" != true ]]; then
  echo "phase4b-diagnostic mode requires MER_PROMPT2_PHASE4B_TRACE_PATH with a {case} placeholder" >&2
  exit 2
fi
if [[ "$COLLECTOR_MODE" != phase4b-diagnostic && "$PHASE4B_TRACE_ENABLED" == true ]]; then
  echo "trace-enabled collector runs must use phase4b-diagnostic mode and cannot qualify as performance baselines" >&2
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

WORKTREE_STATUS=$(git -C "$ROOT" status --short)
if [[ -n "$WORKTREE_STATUS" &&
      ("$COLLECTOR_MODE" == phase4d-screening ||
       "$COLLECTOR_MODE" == phase4d-b-score-calibration ||
       "$COLLECTOR_MODE" == phase4d-c-sparse-admission ||
       "${MER_ALLOW_DIRTY:-0}" != 1) ]]; then
  echo "refusing a baseline from a dirty worktree; commit/stash changes or set MER_ALLOW_DIRTY=1 for a non-qualifying diagnostic" >&2
  if [[ "$COLLECTOR_MODE" == phase4d-screening ]]; then
    echo "Phase 4D-A screening always requires a clean worktree" >&2
  elif [[ "$COLLECTOR_MODE" == phase4d-b-score-calibration ]]; then
    echo "Phase 4D-B score calibration always requires a clean worktree" >&2
  elif [[ "$COLLECTOR_MODE" == phase4d-c-sparse-admission ]]; then
    echo "Phase 4D-C sparse admission screening always requires a clean worktree" >&2
  fi
  printf '%s\n' "$WORKTREE_STATUS" >&2
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
  if [[ "$QUALIFICATION_KIND" != performance-baseline ]]; then
    echo "qualification_kind=$QUALIFICATION_KIND"
  fi
  echo "experiment_name=$EXPERIMENT_NAME"
  echo "predict_fanout=$PREDICT_FANOUT"
  echo "pipeline_depth=$PIPELINE_DEPTH"
  echo "prefetch_variant=$PREFETCH_VARIANT"
  echo "predictor_mode=$PREDICTOR_MODE"
  echo "first_order_enabled=$FIRST_ORDER_ENABLED"
  echo "second_order_enabled=$SECOND_ORDER_ENABLED"
  echo "fallback_prior_fill_enabled=$FALLBACK_PRIOR_FILL_ENABLED"
  echo "fanout_is_upper_bound=$FANOUT_IS_UPPER_BOUND"
  echo "prefetch_governor_enabled=$PREFETCH_GOVERNOR_ENABLED"
  echo "prefetch_governor_precision_floor=$PREFETCH_GOVERNOR_PRECISION_FLOOR"
  echo "prefetch_governor_contention_weight=$PREFETCH_GOVERNOR_CONTENTION_WEIGHT"
  echo "prefetch_governor_base_threshold=$PREFETCH_GOVERNOR_BASE_THRESHOLD"
  echo "neural_speculator_enabled=$NEURAL_SPECULATOR_ENABLED"
  echo "output_tokens=$OUTPUT_TOKENS"
  echo "measured_runs=$MEASURED_RUNS"
  echo "prefetch_expected_active=$PREFETCH_EXPECTED_ACTIVE"
  echo "phase4b_trace_enabled=$PHASE4B_TRACE_ENABLED"
  echo "phase4b_trace_path_template=$PHASE4B_TRACE_TEMPLATE"
  echo "phase4b_trace_max_events=$PHASE4B_TRACE_MAX_EVENTS"
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
sha256sum "$TOKENIZER_REAL" > "$ARTIFACT_DIR/tokenizer.sha256"
TOKENIZER_SHA=$(awk '{print $1}' "$ARTIFACT_DIR/tokenizer.sha256")

prompt2_render_config "$TEMPLATE" "$ARTIFACT_DIR/configs/qwen3-coder-q8-1536.toml" \
  "$MODEL_REAL" "$TOKENIZER_REAL" 1536 "$PREDICT_FANOUT" "$PIPELINE_DEPTH" \
  "$FIRST_ORDER_ENABLED" "$SECOND_ORDER_ENABLED" \
  "$FALLBACK_PRIOR_FILL_ENABLED" "$FANOUT_IS_UPPER_BOUND" \
  "$PREFETCH_GOVERNOR_ENABLED"
if [[ "$COLLECTOR_MODE" != phase4b-diagnostic &&
      "$COLLECTOR_MODE" != phase4c-untraced &&
      "$COLLECTOR_MODE" != phase4d-screening &&
      "$COLLECTOR_MODE" != phase4d-b-score-calibration &&
      "$COLLECTOR_MODE" != phase4d-c-sparse-admission ]]; then
  prompt2_render_config "$TEMPLATE" "$ARTIFACT_DIR/configs/qwen3-coder-q8-6144.toml" \
    "$MODEL_REAL" "$TOKENIZER_REAL" 6144 "$PREDICT_FANOUT" "$PIPELINE_DEPTH" \
    "$FIRST_ORDER_ENABLED" "$SECOND_ORDER_ENABLED" \
    "$FALLBACK_PRIOR_FILL_ENABLED" "$FANOUT_IS_UPPER_BOUND" \
    "$PREFETCH_GOVERNOR_ENABLED"
fi

# Physical pool sizing from build_bench_real_runtime:
# primary=(cache_slots+1), shadow=predict_fanout*pipeline_depth.
POOL_SLOTS=(1536 6144)
if [[ "$COLLECTOR_MODE" == phase4b-diagnostic ||
      "$COLLECTOR_MODE" == phase4c-untraced ||
      "$COLLECTOR_MODE" == phase4d-screening ||
      "$COLLECTOR_MODE" == phase4d-b-score-calibration ||
      "$COLLECTOR_MODE" == phase4d-c-sparse-admission ]]; then
  POOL_SLOTS=(1536)
fi
for slots in "${POOL_SLOTS[@]}"; do
  primary_slots=$((slots + 1))
  shadow_slots=$((PREDICT_FANOUT * PIPELINE_DEPTH))
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
BINARY_SHA=$(awk '{print $1}' "$ARTIFACT_DIR/binary.sha256")
HOSTNAME_VALUE=$(hostname)
LOGICAL_CPU_COUNT=$(getconf _NPROCESSORS_ONLN)
EFFECTIVE_CPU_MASK=$(awk -F': ' 'NR == 1 {print $2}' "$ARTIFACT_DIR/collector-effective-cpu-mask.txt")
jq -n \
  --arg experiment_name "$EXPERIMENT_NAME" \
  --argjson predict_fanout "$PREDICT_FANOUT" \
  --argjson pipeline_depth "$PIPELINE_DEPTH" \
  --arg prefetch_variant "$PREFETCH_VARIANT" \
  --arg predictor_mode "$PREDICTOR_MODE" \
  --argjson first_order_enabled "$FIRST_ORDER_ENABLED" \
  --argjson second_order_enabled "$SECOND_ORDER_ENABLED" \
  --argjson fallback_prior_fill_enabled "$FALLBACK_PRIOR_FILL_ENABLED" \
  --argjson fanout_is_upper_bound "$FANOUT_IS_UPPER_BOUND" \
  --argjson prefetch_governor_enabled "$PREFETCH_GOVERNOR_ENABLED" \
  --argjson prefetch_governor_precision_floor "$PREFETCH_GOVERNOR_PRECISION_FLOOR" \
  --argjson prefetch_governor_contention_weight "$PREFETCH_GOVERNOR_CONTENTION_WEIGHT" \
  --argjson prefetch_governor_base_threshold "$PREFETCH_GOVERNOR_BASE_THRESHOLD" \
  --argjson neural_speculator_enabled "$NEURAL_SPECULATOR_ENABLED" \
  --argjson output_tokens "$OUTPUT_TOKENS" \
  --argjson prefetch_expected_active "$PREFETCH_EXPECTED_ACTIVE" \
  --arg git_commit_full "$EXPECTED_COMMIT" \
  --arg config_json_sha256 "$CONFIG_SHA" \
  --arg dense_manifest_sha256 "$DENSE_MANIFEST_SHA" \
  --arg tokenizer_path "$TOKENIZER_REAL" \
  --arg tokenizer_sha256 "$TOKENIZER_SHA" \
  --arg short_prompt_sha256 "$SHORT_PROMPT_SHA" \
  --arg medium_prompt_sha256 "$MEDIUM_PROMPT_SHA" \
  --arg hostname "$HOSTNAME_VALUE" \
  --argjson logical_cpu_count "$LOGICAL_CPU_COUNT" \
  --arg requested_cpu_mask "0-31" \
  --arg effective_cpu_mask "$EFFECTIVE_CPU_MASK" \
  --arg cargo_features "$FEATURES" \
  --arg binary_sha256 "$BINARY_SHA" \
  --arg collector_mode "$COLLECTOR_MODE" \
  --arg qualification_kind "$QUALIFICATION_KIND" \
  --slurpfile model_mount "$ARTIFACT_DIR/model-findmnt.json" '
  {
    schema: {name:"mer-prompt2-phase4a-ablation-provenance", version:1},
    experiment_name: $experiment_name,
    predict_fanout: $predict_fanout,
    pipeline_depth: $pipeline_depth,
    prefetch_variant: $prefetch_variant,
    predictor_mode: $predictor_mode,
    prefetch_predictor: {
      first_order_enabled: $first_order_enabled,
      second_order_enabled: $second_order_enabled,
      fallback_prior_fill_enabled: $fallback_prior_fill_enabled,
      fanout_is_upper_bound: $fanout_is_upper_bound
    },
    prefetch_governor: {
      enabled: $prefetch_governor_enabled,
      precision_floor: $prefetch_governor_precision_floor,
      contention_weight: $prefetch_governor_contention_weight,
      base_threshold: $prefetch_governor_base_threshold,
      runtime_default_precision_alpha: 0.2,
      runtime_default_base_threshold: 0.02
    },
    neural_speculator_enabled: $neural_speculator_enabled,
    output_tokens: $output_tokens,
    prefetch_expected_active: $prefetch_expected_active,
    git_commit_full: $git_commit_full,
    model_hashes: {
      config_json_sha256: $config_json_sha256,
      dense_manifest_sha256: $dense_manifest_sha256
    },
    tokenizer_identity: {
      path: $tokenizer_path,
      sha256: $tokenizer_sha256
    },
    prompt_hashes: {
      short_sha256: $short_prompt_sha256,
      medium_sha256: $medium_prompt_sha256
    },
    host: {
      hostname: $hostname,
      logical_cpu_count: $logical_cpu_count,
      requested_cpu_mask: $requested_cpu_mask,
      effective_cpu_mask: $effective_cpu_mask
    },
    model_mount_identity: $model_mount[0].filesystems[0],
    cargo_features: ($cargo_features | split(",") | sort),
    binary_sha256: $binary_sha256,
    collector_mode: $collector_mode
  } +
  (if $qualification_kind != "performance-baseline" then
    {qualification_kind: $qualification_kind}
  else
    {}
  end)
' > "$ARTIFACT_DIR/ablation-provenance.json"

run_case() {
  local slots=$1
  local prompt_id=$2
  local expected_prompt_tokens=$3
  local expected_prompt_sha=$4
  local stem="baseline-${slots}-${prompt_id}"
  local json="$ARTIFACT_DIR/$stem.json"
  local request="$FIXTURES/prompts/$prompt_id.json"
  local config="$ARTIFACT_DIR/configs/qwen3-coder-q8-$slots.toml"
  local -a phase4b_env=()
  if [[ "$PHASE4B_TRACE_ENABLED" == true ]]; then
    local phase4b_trace_path=${PHASE4B_TRACE_TEMPLATE//\{case\}/$stem}
    phase4b_env=(
      env
      "MER_PROMPT2_PHASE4B_TRACE_PATH=$phase4b_trace_path"
      "MER_PROMPT2_PHASE4B_TRACE_MAX_EVENTS=$PHASE4B_TRACE_MAX_EVENTS"
    )
  fi

  "${phase4b_env[@]}" /usr/bin/time -v \
    -o "$ARTIFACT_DIR/$stem.time.txt" \
    "$BIN" \
    --rayon-threads 30 \
    --cpu-mask 0-31 \
    bench-real \
    --config "$config" \
    --request-json "$request" \
    --output-tokens "$OUTPUT_TOKENS" \
    --warmup-runs 1 \
    --measured-runs "$MEASURED_RUNS" \
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
    --argjson prompt_tokens "$expected_prompt_tokens" \
    --argjson predict_fanout "$PREDICT_FANOUT" \
    --argjson pipeline_depth "$PIPELINE_DEPTH" \
    --argjson first_order_enabled "$FIRST_ORDER_ENABLED" \
    --argjson second_order_enabled "$SECOND_ORDER_ENABLED" \
    --argjson fallback_prior_fill_enabled "$FALLBACK_PRIOR_FILL_ENABLED" \
    --argjson fanout_is_upper_bound "$FANOUT_IS_UPPER_BOUND" \
    --argjson prefetch_governor_enabled "$PREFETCH_GOVERNOR_ENABLED" \
    --argjson prefetch_governor_precision_floor "$PREFETCH_GOVERNOR_PRECISION_FLOOR" \
    --argjson prefetch_governor_contention_weight "$PREFETCH_GOVERNOR_CONTENTION_WEIGHT" \
    --argjson prefetch_governor_base_threshold "$PREFETCH_GOVERNOR_BASE_THRESHOLD" \
    --argjson neural_speculator_enabled "$NEURAL_SPECULATOR_ENABLED" \
    --argjson output_tokens "$OUTPUT_TOKENS" \
    --argjson measured_runs "$MEASURED_RUNS" \
    --argjson phase4b_trace_enabled "$PHASE4B_TRACE_ENABLED" '
    .schema == {"name":"mer-bench-real","version":2} and
    .benchmark == "bench-real" and
    .warmup_runs == 1 and
    .measured_runs == $measured_runs and
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
    .prompt_identity.requested_completion_tokens == $output_tokens and
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
    .memory_layout.shadow_expert_pool_allocated_bytes >= 0 and
    .memory_layout.total_expert_pool_allocated_bytes ==
      (.memory_layout.primary_expert_pool_allocated_bytes + .memory_layout.shadow_expert_pool_allocated_bytes) and
    .memory_layout.prepared_duplicate_expert_bytes == 0 and
    .memory_layout.external_peak_rss_source == "collector:/usr/bin/time-v" and
    .predictive_policy == {
      "markov_prefetch_fanout":$predict_fanout,
      "pipeline_depth":$pipeline_depth,
      "first_order_enabled":$first_order_enabled,
      "second_order_enabled":$second_order_enabled,
      "fallback_prior_fill_enabled":$fallback_prior_fill_enabled,
      "fanout_is_upper_bound":$fanout_is_upper_bound,
      "locality_enabled":false,
      "speculator_enabled":$neural_speculator_enabled,
      "affinity_enabled":false,
      "prefetch_governor_enabled":$prefetch_governor_enabled,
      "prefetch_governor_precision_floor":$prefetch_governor_precision_floor,
      "prefetch_governor_contention_weight":$prefetch_governor_contention_weight,
      "prefetch_governor_base_threshold":$prefetch_governor_base_threshold,
      "cost_aware_eviction_enabled":false,
      "pregate_enabled":false,
      "static_residency_fraction":0
    } and
    .phase4b_trace_enabled == $phase4b_trace_enabled and
    .aggregate.output_token_parity == true and
    (.runs | length == $measured_runs) and
    ([.runs[].prompt_tokens] | all(. == $prompt_tokens)) and
    ([.runs[].completion_tokens] | all(. == $output_tokens)) and
    ([.runs[].total_api_tokens] | all(. == ($prompt_tokens + $output_tokens))) and
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
      .prefetch_enabled == ($predict_fanout > 0) and
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
      (if $prefetch_governor_enabled then
        .prefetch_dropped_governor >= 0 and
        .governor_rejected_candidates == .prefetch_dropped_governor
      else
        .prefetch_dropped_governor == 0 and
        .governor_admitted_candidates == 0 and
        .governor_rejected_candidates == 0 and
        .governor_total_decisions == 0
      end) and
      .governor_admitted_candidates >= 0 and
      .governor_rejected_candidates >= 0 and
      .governor_total_decisions ==
        (.governor_admitted_candidates + .governor_rejected_candidates) and
      .governor_admission_rate >= 0 and
      .governor_admission_rate <= 1 and
      .governor_admission_rate ==
        (if .governor_total_decisions == 0 then 0
         else (.governor_admitted_candidates / .governor_total_decisions)
         end) and
      .governor_score_diagnostics_reconcile == true and
      .governor_score_diagnostics.total_decisions ==
        .governor_total_decisions and
      .governor_score_diagnostics.admitted ==
        .governor_admitted_candidates and
      .governor_score_diagnostics.rejected ==
        .governor_rejected_candidates and
      .governor_score_diagnostics.total_decisions ==
        (.governor_score_diagnostics.admitted +
         .governor_score_diagnostics.rejected) and
      .governor_score_diagnostics.invalid_numeric_decisions == 0 and
      .governor_score_diagnostics.sample_capacity == 512 and
      .governor_score_diagnostics.sampled_decisions >= 0 and
      .governor_score_diagnostics.sampled_decisions <= 512 and
      .governor_precision_ewma_final >= 0 and
      .governor_precision_ewma_final <= 1 and
      .governor_foreground_inflight_final == 0 and
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
        (if $predict_fanout == 0 then
          .demand_reads_issued_while_speculative_reads_active == 0
        else
          .demand_reads_issued_while_speculative_reads_active >= 0
        end) and
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
      .shadow_expert_pool_allocated_bytes >= 0 and
      .total_expert_pool_allocated_bytes ==
        (.primary_expert_pool_allocated_bytes + .shadow_expert_pool_allocated_bytes) and
      .prepared_duplicate_expert_bytes == 0 and
      .external_peak_rss_bytes == null
    )) and
    (if $predict_fanout > 0 then
      .memory_layout.shadow_expert_pool_allocated_bytes > 0 and
      ([.runs[].memory.shadow_expert_pool_allocated_bytes] | all(. > 0))
    else true end)
  ' "$json" >/dev/null

  local collection_qualification
  collection_qualification=$(jq \
    --arg qualification_kind "$QUALIFICATION_KIND" \
    -f "$COLLECTION_QUALIFIER" \
    "$json")
  jq -e '.collection_qualification_valid == true' \
    <<<"$collection_qualification" >/dev/null

  jq -e \
    --argjson predict_fanout "$PREDICT_FANOUT" \
    --argjson pipeline_depth "$PIPELINE_DEPTH" \
    --argjson prefetch_governor_enabled "$PREFETCH_GOVERNOR_ENABLED" \
    -f "$ROOT/scripts/qwen3_coder_prompt2_prefetch_qualification.jq" \
    "$json" >/dev/null

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
    --argjson predict_fanout "$PREDICT_FANOUT" \
    --argjson pipeline_depth "$PIPELINE_DEPTH" \
    --arg prefetch_variant "$PREFETCH_VARIANT" \
    --arg predictor_mode "$PREDICTOR_MODE" \
    --argjson first_order_enabled "$FIRST_ORDER_ENABLED" \
    --argjson second_order_enabled "$SECOND_ORDER_ENABLED" \
    --argjson fallback_prior_fill_enabled "$FALLBACK_PRIOR_FILL_ENABLED" \
    --argjson fanout_is_upper_bound "$FANOUT_IS_UPPER_BOUND" \
    --argjson prefetch_governor_enabled "$PREFETCH_GOVERNOR_ENABLED" \
    --argjson prefetch_governor_precision_floor "$PREFETCH_GOVERNOR_PRECISION_FLOOR" \
    --argjson prefetch_governor_contention_weight "$PREFETCH_GOVERNOR_CONTENTION_WEIGHT" \
    --argjson prefetch_governor_base_threshold "$PREFETCH_GOVERNOR_BASE_THRESHOLD" \
    --argjson neural_speculator_enabled "$NEURAL_SPECULATOR_ENABLED" \
    --argjson collection_qualification "$collection_qualification" \
    '([.runs[].cache_io] | {
      prefetch_submitted: (map(.prefetch_submitted) | add),
      prefetch_completed: (map(.prefetch_completed) | add),
      prefetch_used: (map(.prefetch_used) | add),
      prefetch_bytes: (map(.prefetch_bytes) | add),
      useful_prefetch_bytes: (map(.useful_prefetch_bytes) | add),
      unused_prefetch_bytes_at_sample: (map(.unused_prefetch_bytes_at_sample) | add),
      prefetch_dropped_concurrency: (map(.prefetch_dropped_concurrency) | add),
      prefetch_dropped_pool_starved: (map(.prefetch_dropped_pool_starved) | add),
      prefetch_dropped_governor: (map(.prefetch_dropped_governor) | add),
      prefetch_dropped_bytes: (map(.prefetch_dropped_bytes) | add)
    }) as $prefetch_counters |
    ([.runs[] | {
      run_index,
      governor_enabled: $prefetch_governor_enabled,
      neural_speculator_enabled: $neural_speculator_enabled,
      candidates_rejected_by_governor: .cache_io.governor_rejected_candidates,
      governor_admitted_candidates: .cache_io.governor_admitted_candidates,
      governor_rejected_candidates: .cache_io.governor_rejected_candidates,
      governor_total_decisions: .cache_io.governor_total_decisions,
      governor_admission_rate: .cache_io.governor_admission_rate,
      governor_score_diagnostics:
        .cache_io.governor_score_diagnostics,
      governor_score_diagnostics_reconcile:
        .cache_io.governor_score_diagnostics_reconcile,
      governor_precision_ewma_final: .cache_io.governor_precision_ewma_final,
      governor_foreground_inflight_final:
        .cache_io.governor_foreground_inflight_final,
      direct_governor_decisions: {
        admitted_candidates: .cache_io.governor_admitted_candidates,
        rejected_candidates: .cache_io.governor_rejected_candidates,
        total_decisions: .cache_io.governor_total_decisions,
        admission_rate: .cache_io.governor_admission_rate
      },
      speculative_work_admitted: .cache_io.prefetch_submitted,
      governor_admitted_candidates_derived:
        (if $prefetch_governor_enabled then
          (.cache_io.prefetch_submitted + .cache_io.prefetch_dropped_concurrency)
        else
          0
        end),
      governor_admission_derivation:
        "prefetch_submitted + prefetch_dropped_concurrency; governor admission precedes the concurrency gate",
      derived_governor_admission: {
        admitted_candidates:
          (if $prefetch_governor_enabled then
            (.cache_io.prefetch_submitted + .cache_io.prefetch_dropped_concurrency)
          else
            0
          end),
        derivation:
          "prefetch_submitted + prefetch_dropped_concurrency; governor admission precedes the concurrency gate"
      },
      speculative_work_completed: .cache_io.prefetch_completed,
      speculative_work_used: .cache_io.prefetch_used,
      speculative_work_dropped_by_concurrency:
        .cache_io.prefetch_dropped_concurrency,
      speculative_work_dropped_by_pool_pressure:
        .cache_io.prefetch_dropped_pool_starved,
      demand_read_activity_while_speculation_active: {
        prompt:
          .demand_miss_fanout.prompt.demand_reads_issued_while_speculative_reads_active,
        decode:
          .demand_miss_fanout.decode.demand_reads_issued_while_speculative_reads_active,
        total:
          (.demand_miss_fanout.prompt.demand_reads_issued_while_speculative_reads_active +
           .demand_miss_fanout.decode.demand_reads_issued_while_speculative_reads_active)
      },
      foreground_pressure: {
        prompt_peak_foreground_physical_reads_in_flight:
          .demand_miss_fanout.prompt.peak_foreground_physical_reads_in_flight,
        decode_peak_foreground_physical_reads_in_flight:
          .demand_miss_fanout.decode.peak_foreground_physical_reads_in_flight,
        prompt_average_foreground_physical_read_concurrency:
          .demand_miss_fanout.prompt.average_foreground_physical_read_concurrency,
        decode_average_foreground_physical_read_concurrency:
          .demand_miss_fanout.decode.average_foreground_physical_read_concurrency,
        prompt_foreground_physical_read_active_seconds:
          .demand_miss_fanout.prompt.foreground_physical_read_active_seconds,
        decode_foreground_physical_read_active_seconds:
          .demand_miss_fanout.decode.foreground_physical_read_active_seconds
      }
    }]) as $governor_runs |
    {
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
      qualification_passed: $collection_qualification.qualification_passed,
      phase4a_prefetch: {
        requested_predict_fanout: $predict_fanout,
        requested_pipeline_depth: $pipeline_depth,
        expected_active: ($predict_fanout > 0),
        reported_enabled_values: ([.runs[].cache_io.prefetch_enabled] | unique),
        all_runs_reported_expected_enabled: (
          [.runs[].cache_io.prefetch_enabled] | all(. == ($predict_fanout > 0))
        ),
        counters: $prefetch_counters,
        all_prefetch_counters_zero: (
          [$prefetch_counters[]] | all(. == 0)
        ),
        prompt_demand_reads_issued_while_speculative_reads_active: (
          [.runs[].demand_miss_fanout.prompt.demand_reads_issued_while_speculative_reads_active] | add
        ),
        decode_demand_reads_issued_while_speculative_reads_active: (
          [.runs[].demand_miss_fanout.decode.demand_reads_issued_while_speculative_reads_active] | add
        )
      },
      phase4c_predictor: {
        variant: $prefetch_variant,
        predictor_mode: $predictor_mode,
        predict_fanout: $predict_fanout,
        pipeline_depth: $pipeline_depth,
        first_order_enabled: $first_order_enabled,
        second_order_enabled: $second_order_enabled,
        fallback_prior_fill_enabled: $fallback_prior_fill_enabled,
        fanout_is_upper_bound: $fanout_is_upper_bound,
        governor_enabled: $prefetch_governor_enabled,
        neural_speculator_enabled: $neural_speculator_enabled
      },
      phase4c_governor: {
        configuration: {
          enabled: $prefetch_governor_enabled,
          precision_floor: $prefetch_governor_precision_floor,
          contention_weight: $prefetch_governor_contention_weight,
          base_threshold: $prefetch_governor_base_threshold,
          runtime_default_precision_alpha: 0.2,
          runtime_default_base_threshold: 0.02
        },
        counters_by_run: $governor_runs,
        totals: {
          candidates_rejected_by_governor:
            ($governor_runs | map(.candidates_rejected_by_governor) | add),
          governor_admitted_candidates:
            ($governor_runs | map(.governor_admitted_candidates) | add),
          governor_rejected_candidates:
            ($governor_runs | map(.governor_rejected_candidates) | add),
          governor_total_decisions:
            ($governor_runs | map(.governor_total_decisions) | add),
          governor_admission_rate:
            (($governor_runs | map(.governor_admitted_candidates) | add) as $admitted |
             ($governor_runs | map(.governor_total_decisions) | add) as $decisions |
             if $decisions == 0 then 0 else ($admitted / $decisions) end),
          governor_precision_ewma_final:
            ($governor_runs | last | .governor_precision_ewma_final),
          governor_foreground_inflight_final:
            ($governor_runs | last | .governor_foreground_inflight_final),
          speculative_work_admitted:
            ($governor_runs | map(.speculative_work_admitted) | add),
          governor_admitted_candidates_derived:
            ($governor_runs | map(.governor_admitted_candidates_derived) | add),
          speculative_work_dropped_by_concurrency:
            ($governor_runs | map(.speculative_work_dropped_by_concurrency) | add),
          speculative_work_dropped_by_pool_pressure:
            ($governor_runs | map(.speculative_work_dropped_by_pool_pressure) | add),
          demand_reads_observed_while_speculation_active:
            ($governor_runs |
             map(.demand_read_activity_while_speculation_active.total) |
             add)
        },
        score_diagnostics_by_run:
          [$governor_runs[] | {
            run_index,
            reconcile: .governor_score_diagnostics_reconcile,
            diagnostics: .governor_score_diagnostics
          }],
        score_diagnostics_aggregate:
          .aggregate.governor_score_diagnostics,
        score_diagnostics_aggregate_reconcile:
          .aggregate.governor_score_diagnostics_reconcile
      },
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
    } +
    (if $collection_qualification.qualification_kind == "phase4b-diagnostic" then
      {
        qualification_kind: $collection_qualification.qualification_kind,
        diagnostic_qualification_passed:
          $collection_qualification.diagnostic_qualification_passed,
        performance_qualification_applicable:
          $collection_qualification.performance_qualification_applicable,
        performance_qualification_reason:
          $collection_qualification.performance_qualification_reason,
        production_critical_path_coverage_gates_passed:
          $collection_qualification.production_critical_path_coverage_gates_passed,
        observed_critical_path_coverage:
          $collection_qualification.observed_critical_path_coverage
      }
    elif $collection_qualification.qualification_kind == "phase4d-governor-screening" then
      {
        qualification_kind: $collection_qualification.qualification_kind,
        screening_collection_valid:
          $collection_qualification.screening_collection_valid,
        performance_qualification_applicable: false,
        performance_qualification_reason:
          $collection_qualification.performance_qualification_reason,
        production_critical_path_coverage_gates_passed:
          $collection_qualification.production_critical_path_coverage_gates_passed,
        observed_critical_path_coverage:
          $collection_qualification.observed_critical_path_coverage
      }
    elif $collection_qualification.qualification_kind ==
         "phase4d-b-score-calibration-diagnostic" then
      {
        qualification_kind: $collection_qualification.qualification_kind,
        score_calibration_collection_valid:
          $collection_qualification.score_calibration_collection_valid,
        performance_qualification_applicable: false,
        performance_qualification_reason:
          $collection_qualification.performance_qualification_reason,
        production_critical_path_coverage_gates_passed:
          $collection_qualification.production_critical_path_coverage_gates_passed,
        observed_critical_path_coverage:
          $collection_qualification.observed_critical_path_coverage
      }
    else
      {}
    end)' "$json" > "$ARTIFACT_DIR/$stem.case-summary.json"
}

RESIDENT_STATUS=0
if [[ "$COLLECTOR_MODE" == phase4b-diagnostic ]]; then
  run_case 1536 short 14 "$SHORT_PROMPT_SHA"
  run_case 1536 medium 65 "$MEDIUM_PROMPT_SHA"

  jq -s '{
    schema: {name:"mer-prompt2-phase4b-diagnostic-summary", version:1},
    qualification_kind: "phase4b-diagnostic",
    cases: .,
    diagnostic_qualification_passed:
      (all(.diagnostic_qualification_passed == true)),
    performance_qualification_applicable: false,
    performance_qualification_reason:
      "synchronous Phase 4B JSONL tracing adds diagnostic wall time outside production critical-path categories; traced TPS and coverage are not comparable with untraced Phase 4A performance baselines",
    qualification_passed: false,
    observed_critical_path_coverage: [
      .[] | {
        case,
        prompt: .observed_critical_path_coverage.prompt,
        decode: .observed_critical_path_coverage.decode,
        prompt_min: .observed_critical_path_coverage.prompt_min,
        decode_min: .observed_critical_path_coverage.decode_min,
        production_critical_path_coverage_gates_passed
      }
    ]
  }' \
    "$ARTIFACT_DIR/baseline-1536-short.case-summary.json" \
    "$ARTIFACT_DIR/baseline-1536-medium.case-summary.json" \
    > "$ARTIFACT_DIR/phase4b-diagnostic-summary.json"

  jq -n \
    --arg commit "$EXPECTED_COMMIT" \
    --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --slurpfile diagnostic_summary \
      "$ARTIFACT_DIR/phase4b-diagnostic-summary.json" \
    '{
      schema: {name:"mer-prompt2-qualification", version:1},
      git_commit_full: $commit,
      captured_at_utc: $captured_at_utc,
      qualification_kind: "phase4b-diagnostic",
      two_required_cases_present: true,
      schema_and_required_fields_passed: true,
      provenance_and_backend_gates_passed: true,
      prompt_identity_gates_passed: true,
      strictness_and_correctness_gates_passed: true,
      phase4b_trace_gates_passed: true,
      prefetch_policy_gates_passed: true,
      diagnostic_qualification_passed: true,
      performance_qualification_applicable: false,
      performance_qualification_reason:
        "synchronous Phase 4B JSONL tracing adds diagnostic wall time outside production critical-path categories; traced TPS and coverage are not comparable with untraced Phase 4A performance baselines",
      production_critical_path_coverage_gates_passed: (
        [$diagnostic_summary[0].observed_critical_path_coverage[] |
          .production_critical_path_coverage_gates_passed] |
        all(. == true)
      ),
      observed_critical_path_coverage:
        $diagnostic_summary[0].observed_critical_path_coverage,
      critical_path_coverage_gates_applicable: false,
      critical_path_coverage_gates_passed: null,
      external_peak_rss_present: true,
      qualification_passed: false
    }' > "$ARTIFACT_DIR/qualification.json"
# BEGIN PHASE4D_SCREENING_COLLECTION
elif [[ "$COLLECTOR_MODE" == phase4d-screening ]]; then
  run_case 1536 short 14 "$SHORT_PROMPT_SHA"

  jq -n \
    --arg variant "$PREFETCH_VARIANT" \
    --slurpfile case_summary \
      "$ARTIFACT_DIR/baseline-1536-short.case-summary.json" \
    --slurpfile provenance "$ARTIFACT_DIR/ablation-provenance.json" \
    --slurpfile raw "$ARTIFACT_DIR/baseline-1536-short.json" '
    $case_summary[0] as $case |
    $provenance[0] as $provenance |
    {
      schema: {name:"mer-prompt2-phase4d-screening-variant", version:1},
      qualification_kind: "phase4d-governor-screening",
      variant: $variant,
      cache_slots: 1536,
      prompt_fixture: "short",
      output_tokens: 128,
      warmup_runs: 1,
      measured_runs: 2,
      greedy: true,
      traced: false,
      metadata: $case.phase4c_predictor,
      governor_configuration: $case.phase4c_governor.configuration,
      governor_counters_by_run: $case.phase4c_governor.counters_by_run,
      governor_totals: $case.phase4c_governor.totals,
      prefetch_counters: $case.phase4a_prefetch.counters,
      provenance: {
        git_commit_full: $provenance.git_commit_full,
        binary_sha256: $provenance.binary_sha256,
        model_hashes: $provenance.model_hashes,
        tokenizer_identity: $provenance.tokenizer_identity,
        target_host: $provenance.host,
        model_mount_identity: $provenance.model_mount_identity,
        cargo_features: $provenance.cargo_features,
        prompt_hashes: $provenance.prompt_hashes
      },
      decode_tps_mean: $case.decode_tps_mean,
      ssd_bytes: $case.ssd_bytes_total,
      demand_read_service_mean_seconds:
        $case.phase3a_decode.physical_read_issue_to_completion_mean_seconds,
      demand_reads_observed_while_speculation_active:
        ($case.phase4a_prefetch.prompt_demand_reads_issued_while_speculative_reads_active +
         $case.phase4a_prefetch.decode_demand_reads_issued_while_speculative_reads_active),
      output_token_ids: $raw[0].runs[0].output_token_ids,
      output_parity_within_variant: $raw[0].aggregate.output_token_parity,
      screening_collection_valid: $case.screening_collection_valid,
      performance_qualification_applicable: false,
      performance_qualification_reason: $case.performance_qualification_reason,
      qualification_passed: false
    }
  ' > "$ARTIFACT_DIR/phase4d-screening-variant-summary.json"

  jq -n \
    --arg commit "$EXPECTED_COMMIT" \
    --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg variant "$PREFETCH_VARIANT" \
    --slurpfile variant_summary \
      "$ARTIFACT_DIR/phase4d-screening-variant-summary.json" '
    {
      schema: {name:"mer-prompt2-qualification", version:1},
      git_commit_full: $commit,
      captured_at_utc: $captured_at_utc,
      qualification_kind: "phase4d-governor-screening",
      phase4d_variant: $variant,
      one_required_1536_short_case_present: true,
      no_6144_cases_collected: true,
      trace_disabled: true,
      screening_collection_valid:
        $variant_summary[0].screening_collection_valid,
      performance_qualification_applicable: false,
      performance_qualification_reason:
        "Phase 4D-A uses one warmup and two measured short-prompt runs for calibration screening; it is diagnostic evidence and not a qualified production performance baseline",
      qualification_passed: false
    }
  ' > "$ARTIFACT_DIR/qualification.json"
# END PHASE4D_SCREENING_COLLECTION
# BEGIN PHASE4D_B_SCORE_CALIBRATION_COLLECTION
elif [[ "$COLLECTOR_MODE" == phase4d-b-score-calibration ]]; then
  run_case 1536 short 14 "$SHORT_PROMPT_SHA"

  jq -n \
    --arg variant "$PREFETCH_VARIANT" \
    --slurpfile case_summary \
      "$ARTIFACT_DIR/baseline-1536-short.case-summary.json" \
    --slurpfile provenance "$ARTIFACT_DIR/ablation-provenance.json" \
    --slurpfile raw "$ARTIFACT_DIR/baseline-1536-short.json" '
    $case_summary[0] as $case |
    $provenance[0] as $provenance |
    $raw[0] as $raw |
    {
      schema: {
        name:"mer-prompt2-phase4d-b-governor-score-calibration-variant",
        version:1
      },
      qualification_kind: "phase4d-b-score-calibration-diagnostic",
      variant: $variant,
      cache_slots: 1536,
      prompt_fixture: "short",
      output_tokens: 128,
      warmup_runs: 1,
      measured_runs: 2,
      cache_reset: "keep",
      greedy: true,
      traced: false,
      metadata: $case.phase4c_predictor,
      governor_configuration: $case.phase4c_governor.configuration,
      governor_counters_by_run: $case.phase4c_governor.counters_by_run,
      governor_totals: $case.phase4c_governor.totals,
      governor_score_diagnostics_by_run:
        $case.phase4c_governor.score_diagnostics_by_run,
      governor_score_diagnostics:
        $case.phase4c_governor.score_diagnostics_aggregate,
      governor_score_diagnostics_reconcile:
        $case.phase4c_governor.score_diagnostics_aggregate_reconcile,
      foreground_accounting: {
        final_by_run:
          [$case.phase4c_governor.counters_by_run[] |
           .governor_foreground_inflight_final],
        final_aggregate:
          $case.phase4c_governor.totals.governor_foreground_inflight_final
      },
      prefetch_counters: $case.phase4a_prefetch.counters,
      provenance: {
        git_commit_full: $provenance.git_commit_full,
        binary_sha256: $provenance.binary_sha256,
        model_hashes: $provenance.model_hashes,
        tokenizer_identity: $provenance.tokenizer_identity,
        target_host: $provenance.host,
        model_mount_identity: $provenance.model_mount_identity,
        cargo_features: $provenance.cargo_features,
        prompt_hashes: $provenance.prompt_hashes
      },
      decode_tps_mean: $case.decode_tps_mean,
      ssd_bytes: $case.ssd_bytes_total,
      demand_read_service_mean_seconds:
        $case.phase3a_decode.physical_read_issue_to_completion_mean_seconds,
      output_token_ids: $raw.runs[0].output_token_ids,
      output_parity_within_variant: $raw.aggregate.output_token_parity,
      diagnostic_collection_valid:
        $case.score_calibration_collection_valid,
      performance_qualification_applicable: false,
      performance_qualification_reason: $case.performance_qualification_reason,
      qualification_passed: false
    }
  ' > "$ARTIFACT_DIR/phase4d-b-score-calibration-variant-summary.json"

  jq -n \
    --arg commit "$EXPECTED_COMMIT" \
    --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg variant "$PREFETCH_VARIANT" \
    --slurpfile variant_summary \
      "$ARTIFACT_DIR/phase4d-b-score-calibration-variant-summary.json" '
    {
      schema: {name:"mer-prompt2-qualification", version:1},
      git_commit_full: $commit,
      captured_at_utc: $captured_at_utc,
      qualification_kind: "phase4d-b-score-calibration-diagnostic",
      phase4d_b_variant: $variant,
      one_required_1536_short_case_present: true,
      no_6144_cases_collected: true,
      trace_disabled: true,
      diagnostic_collection_valid:
        $variant_summary[0].diagnostic_collection_valid,
      governor_score_diagnostics_reconcile:
        $variant_summary[0].governor_score_diagnostics_reconcile,
      performance_qualification_applicable: false,
      performance_qualification_reason:
        "Phase 4D-B records bounded governor score distributions for diagnosis and is not a qualified production performance baseline",
      qualification_passed: false
    }
  ' > "$ARTIFACT_DIR/qualification.json"
# END PHASE4D_B_SCORE_CALIBRATION_COLLECTION
# BEGIN PHASE4D_C_SPARSE_ADMISSION_COLLECTION
elif [[ "$COLLECTOR_MODE" == phase4d-c-sparse-admission ]]; then
  run_case 1536 short 14 "$SHORT_PROMPT_SHA"

  jq -n \
    --arg variant "$PREFETCH_VARIANT" \
    --slurpfile case_summary "$ARTIFACT_DIR/baseline-1536-short.case-summary.json" \
    --slurpfile provenance "$ARTIFACT_DIR/ablation-provenance.json" \
    --slurpfile raw "$ARTIFACT_DIR/baseline-1536-short.json" '
    $case_summary[0] as $case |
    $provenance[0] as $provenance |
    $raw[0] as $raw |
    {
      schema: {name:"mer-prompt2-phase4d-c-sparse-admission-variant", version:1},
      qualification_kind: "phase4d-c-sparse-admission-screening",
      variant: $variant,
      cache_slots: 1536,
      prompt_fixture: "short",
      output_tokens: 128,
      warmup_runs: 1,
      measured_runs: 2,
      cache_reset: "keep",
      greedy: true,
      traced: false,
      metadata: $case.phase4c_predictor,
      governor_configuration: $case.phase4c_governor.configuration,
      governor_counters_by_run: $case.phase4c_governor.counters_by_run,
      governor_totals: $case.phase4c_governor.totals,
      governor_score_diagnostics_by_run: $case.phase4c_governor.score_diagnostics_by_run,
      governor_score_diagnostics: $case.phase4c_governor.score_diagnostics_aggregate,
      governor_score_diagnostics_reconcile: $case.phase4c_governor.score_diagnostics_aggregate_reconcile,
      foreground_accounting: {
        final_by_run: [$case.phase4c_governor.counters_by_run[] | .governor_foreground_inflight_final],
        final_aggregate: $case.phase4c_governor.totals.governor_foreground_inflight_final
      },
      prefetch_counters: $case.phase4a_prefetch.counters,
      provenance: {
        git_commit_full: $provenance.git_commit_full,
        binary_sha256: $provenance.binary_sha256,
        model_hashes: $provenance.model_hashes,
        tokenizer_identity: $provenance.tokenizer_identity,
        target_host: $provenance.host,
        model_mount_identity: $provenance.model_mount_identity,
        cargo_features: $provenance.cargo_features,
        prompt_hashes: $provenance.prompt_hashes
      },
      decode_tps_mean: $case.decode_tps_mean,
      ssd_bytes: $case.ssd_bytes_total,
      demand_read_service_mean_seconds: $case.phase3a_decode.physical_read_issue_to_completion_mean_seconds,
      output_token_ids: $raw.runs[0].output_token_ids,
      output_parity_within_variant: $raw.aggregate.output_token_parity,
      screening_collection_valid: $case.sparse_admission_collection_valid,
      performance_qualification_applicable: false,
      performance_qualification_reason: $case.performance_qualification_reason,
      qualification_passed: false
    }
  ' > "$ARTIFACT_DIR/phase4d-c-sparse-admission-variant-summary.json"

  jq -n \
    --arg commit "$EXPECTED_COMMIT" \
    --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg variant "$PREFETCH_VARIANT" \
    --slurpfile variant_summary "$ARTIFACT_DIR/phase4d-c-sparse-admission-variant-summary.json" '
    {
      schema: {name:"mer-prompt2-qualification", version:1},
      git_commit_full: $commit,
      captured_at_utc: $captured_at_utc,
      qualification_kind: "phase4d-c-sparse-admission-screening",
      phase4d_c_variant: $variant,
      one_required_1536_short_case_present: true,
      no_6144_cases_collected: true,
      trace_disabled: true,
      screening_collection_valid: $variant_summary[0].screening_collection_valid,
      governor_score_diagnostics_reconcile: $variant_summary[0].governor_score_diagnostics_reconcile,
      performance_qualification_applicable: false,
      performance_qualification_reason: "Phase 4D-C screens sparse admissions to identify a candidate for a later qualified benchmark",
      qualification_passed: false
    }
  ' > "$ARTIFACT_DIR/qualification.json"
# END PHASE4D_C_SPARSE_ADMISSION_COLLECTION
elif [[ "$COLLECTOR_MODE" == phase4c-untraced ]]; then
  run_case 1536 short 14 "$SHORT_PROMPT_SHA"
  run_case 1536 medium 65 "$MEDIUM_PROMPT_SHA"

  jq -s \
    --arg variant "$PREFETCH_VARIANT" \
    '{
      schema: {name:"mer-prompt2-phase4c-variant-summary", version:1},
      variant: $variant,
      cache_slots: 1536,
      traced: false,
      metadata: .[0].phase4c_predictor,
      governor_configuration: .[0].phase4c_governor.configuration,
      governor_counters_by_case:
        [ .[] | {
            case,
            prompt_fixture,
            counters_by_run: .phase4c_governor.counters_by_run,
            totals: .phase4c_governor.totals
          }
        ],
      cases: .,
      qualification_passed:
        (all(.qualification_passed == true) and
         all(.phase4c_predictor.neural_speculator_enabled == false) and
         all(.phase4c_predictor.variant == $variant) and
         all((.phase4c_governor.counters_by_run | length) == 5))
    }' \
    "$ARTIFACT_DIR/baseline-1536-short.case-summary.json" \
    "$ARTIFACT_DIR/baseline-1536-medium.case-summary.json" \
    > "$ARTIFACT_DIR/phase4c-variant-summary.json"

  jq -n \
    --arg commit "$EXPECTED_COMMIT" \
    --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg variant "$PREFETCH_VARIANT" \
    --arg predictor_mode "$PREDICTOR_MODE" \
    --argjson prefetch_governor_enabled "$PREFETCH_GOVERNOR_ENABLED" \
    --argjson neural_speculator_enabled "$NEURAL_SPECULATOR_ENABLED" \
    '{
      schema: {name:"mer-prompt2-qualification", version:1},
      git_commit_full: $commit,
      captured_at_utc: $captured_at_utc,
      qualification_kind: "performance-baseline",
      phase4c_variant: $variant,
      predictor_mode: $predictor_mode,
      prefetch_governor_enabled: $prefetch_governor_enabled,
      neural_speculator_enabled: $neural_speculator_enabled,
      two_required_1536_cases_present: true,
      no_6144_cases_collected: true,
      trace_disabled: true,
      schema_and_required_fields_passed: true,
      provenance_and_backend_gates_passed: true,
      prompt_identity_gates_passed: true,
      strictness_and_correctness_gates_passed: true,
      critical_path_coverage_gates_passed: true,
      external_peak_rss_present: true,
      qualification_passed: true
    }' > "$ARTIFACT_DIR/qualification.json"
elif [[ "$COLLECTOR_MODE" == resident-only ]]; then
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

if [[ "$COLLECTOR_MODE" == phase4b-diagnostic ]]; then
  echo "DIAGNOSTIC QUALIFICATION: PASS"
  echo "phase4b-diagnostic collection completed; performance qualification is not applicable: $ARTIFACT_DIR"
else
  echo "QUALIFICATION: PASS"
  echo "$COLLECTOR_MODE instrumented baseline collection completed: $ARTIFACT_DIR"
fi
