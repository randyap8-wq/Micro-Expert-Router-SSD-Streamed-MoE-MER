#!/usr/bin/env bash

prompt2_resolve_ablation_config() {
  PREFETCH_VARIANT=${MER_PROMPT2_PREFETCH_VARIANT-custom}
  local predict_fanout_raw
  local pipeline_depth_raw
  case "$PREFETCH_VARIANT" in
    custom)
      predict_fanout_raw=${MER_PROMPT2_PREDICT_FANOUT-2}
      pipeline_depth_raw=${MER_PROMPT2_PIPELINE_DEPTH-3}
      FIRST_ORDER_ENABLED=${MER_PROMPT2_FIRST_ORDER_ENABLED-true}
      SECOND_ORDER_ENABLED=${MER_PROMPT2_SECOND_ORDER_ENABLED-true}
      FALLBACK_PRIOR_FILL_ENABLED=${MER_PROMPT2_FALLBACK_PRIOR_FILL_ENABLED-true}
      FANOUT_IS_UPPER_BOUND=${MER_PROMPT2_FANOUT_IS_UPPER_BOUND-false}
      ;;
    demand-only)
      predict_fanout_raw=0
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=true
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=true
      FANOUT_IS_UPPER_BOUND=false
      ;;
    current-f2)
      predict_fanout_raw=2
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=true
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=true
      FANOUT_IS_UPPER_BOUND=false
      ;;
    second-only-f2)
      predict_fanout_raw=2
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      ;;
    second-only-f1)
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      ;;
    *)
      echo "MER_PROMPT2_PREFETCH_VARIANT must be custom, demand-only, current-f2, second-only-f2, or second-only-f1; found: $PREFETCH_VARIANT" >&2
      return 2
      ;;
  esac

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
  local boolean_name
  for boolean_name in \
    FIRST_ORDER_ENABLED \
    SECOND_ORDER_ENABLED \
    FALLBACK_PRIOR_FILL_ENABLED \
    FANOUT_IS_UPPER_BOUND; do
    if [[ "${!boolean_name}" != true && "${!boolean_name}" != false ]]; then
      echo "$boolean_name must resolve to true or false; found: ${!boolean_name}" >&2
      return 2
    fi
  done
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
  local first_order_enabled=$8
  local second_order_enabled=$9
  local fallback_prior_fill_enabled=${10}
  local fanout_is_upper_bound=${11}

  sed \
    -e "s|@MODEL_DIR@|$model_dir|g" \
    -e "s|@TOKENIZER_PATH@|$tokenizer_path|g" \
    -e "s|@CACHE_SLOTS@|$cache_slots|g" \
    -e "s|@PREDICT_FANOUT@|$predict_fanout|g" \
    -e "s|@PIPELINE_DEPTH@|$pipeline_depth|g" \
    -e "s|@FIRST_ORDER_ENABLED@|$first_order_enabled|g" \
    -e "s|@SECOND_ORDER_ENABLED@|$second_order_enabled|g" \
    -e "s|@FALLBACK_PRIOR_FILL_ENABLED@|$fallback_prior_fill_enabled|g" \
    -e "s|@FANOUT_IS_UPPER_BOUND@|$fanout_is_upper_bound|g" \
    "$template" > "$output"
}
