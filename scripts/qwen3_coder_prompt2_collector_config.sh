#!/usr/bin/env bash

prompt2_resolve_ablation_config() {
  local predict_fanout_raw=${MER_PROMPT2_PREDICT_FANOUT-2}
  local pipeline_depth_raw=${MER_PROMPT2_PIPELINE_DEPTH-3}

  if [[ ! "$predict_fanout_raw" =~ ^[0-9]+$ ]]; then
    echo "MER_PROMPT2_PREDICT_FANOUT must be an integer greater than or equal to zero; found: $predict_fanout_raw" >&2
    return 2
  fi
  if [[ ! "$pipeline_depth_raw" =~ ^[0-9]+$ ]]; then
    echo "MER_PROMPT2_PIPELINE_DEPTH must be an integer greater than or equal to one; found: $pipeline_depth_raw" >&2
    return 2
  fi

  PREDICT_FANOUT=$((10#$predict_fanout_raw))
  PIPELINE_DEPTH=$((10#$pipeline_depth_raw))
  if (( PIPELINE_DEPTH < 1 )); then
    echo "MER_PROMPT2_PIPELINE_DEPTH must be an integer greater than or equal to one; found: $pipeline_depth_raw" >&2
    return 2
  fi
  if (( PREDICT_FANOUT > 0 )); then
    PREFETCH_EXPECTED_ACTIVE=true
  else
    PREFETCH_EXPECTED_ACTIVE=false
  fi
}

prompt2_render_config() {
  local template=$1
  local output=$2
  local model_dir=$3
  local tokenizer_path=$4
  local cache_slots=$5
  local predict_fanout=$6
  local pipeline_depth=$7

  sed \
    -e "s|@MODEL_DIR@|$model_dir|g" \
    -e "s|@TOKENIZER_PATH@|$tokenizer_path|g" \
    -e "s|@CACHE_SLOTS@|$cache_slots|g" \
    -e "s|@PREDICT_FANOUT@|$predict_fanout|g" \
    -e "s|@PIPELINE_DEPTH@|$pipeline_depth|g" \
    "$template" > "$output"
}
