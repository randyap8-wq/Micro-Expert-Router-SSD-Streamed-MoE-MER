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
} > "$ARTIFACT_DIR/environment.txt"

uname -a > "$ARTIFACT_DIR/uname.txt"
lscpu > "$ARTIFACT_DIR/lscpu.txt"
lsblk -o NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL > "$ARTIFACT_DIR/lsblk.txt"
findmnt > "$ARTIFACT_DIR/findmnt.txt"
findmnt -T "$MODEL_REAL" -o TARGET,SOURCE,FSTYPE,OPTIONS > "$ARTIFACT_DIR/model-findmnt.txt"
df -T "$MODEL_REAL" > "$ARTIFACT_DIR/model-df.txt"
grep -E 'MemTotal|MemAvailable' /proc/meminfo > "$ARTIFACT_DIR/memory-preflight.txt"
taskset -pc $$ > "$ARTIFACT_DIR/collector-effective-cpu-mask.txt"
git -C "$ROOT" rev-parse HEAD > "$ARTIFACT_DIR/git-commit.txt"
git -C "$ROOT" status --short > "$ARTIFACT_DIR/git-status-short.txt"

jq -j .prompt "$FIXTURES/prompts/short.json" | sha256sum > "$ARTIFACT_DIR/prompt-short.sha256"
jq -j .prompt "$FIXTURES/prompts/medium.json" | sha256sum > "$ARTIFACT_DIR/prompt-medium.sha256"
sha256sum "$MODEL_REAL/config.json" "$MODEL_REAL/dense_manifest.json" > "$ARTIFACT_DIR/checkpoint-metadata.sha256"

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
  jq -e '
    .benchmark == "bench-real" and
    .measured_runs == 5 and
    .cache_reset == "keep" and
    .greedy == true and
    .aggregate.output_token_parity == true and
    (.runs | length == 5) and
    ([.runs[].completion_tokens] | all(. == 128)) and
    ([.runs[].prompt_tokens] | unique | length == 1) and
    ([.runs[].output_token_ids] | unique | length == 1)
  ' "$json" >/dev/null
  jq '{prompt_tokens: [.runs[].prompt_tokens] | unique, completion_tokens: [.runs[].completion_tokens] | unique, output_token_parity: .aggregate.output_token_parity}' \
    "$json" > "$ARTIFACT_DIR/$stem.validity.json"
}

run_case 1536 short
run_case 1536 medium
run_case 6144 short
run_case 6144 medium

echo "baseline collection completed; jq syntax/parity checks passed"
echo "IMPORTANT: Prompt 2 qualification still requires the schema and critical-path gates listed in docs/benchmarks/qwen3-coder-single-stream-decode-phase-0.md"
