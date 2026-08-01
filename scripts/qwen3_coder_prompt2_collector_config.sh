#!/usr/bin/env bash

prompt2_resolve_governor_tunable() {
  local env_name=$1
  local raw_value=$2
  local maximum=$3
  local output_name=$4
  local resolved

  if ! resolved=$(jq -nre \
    --arg raw "$raw_value" \
    --argjson maximum "$maximum" '
      ($raw | tonumber?) as $value |
      select($value != null) |
      select(($value | (isnan or isinfinite)) == false) |
      select($value >= 0) |
      select($maximum == null or $value <= $maximum) |
      ($value | tostring)
    '); then
    if [[ "$maximum" == null ]]; then
      echo "$env_name must be a finite number greater than or equal to zero; found: $raw_value" >&2
    else
      echo "$env_name must be a finite number in [0.0, $maximum]; found: $raw_value" >&2
    fi
    return 2
  fi

  # TOML distinguishes integers from floats. Canonical jq output for an
  # integral value is `1`, so add a decimal point before rendering into an
  # f64 field. Exponent and decimal forms are already TOML floats.
  if [[ "$resolved" != *.* && "$resolved" != *e* && "$resolved" != *E* ]]; then
    resolved="$resolved.0"
  fi
  printf -v "$output_name" '%s' "$resolved"
}

prompt2_resolve_ablation_config() {
  PREFETCH_VARIANT=${MER_PROMPT2_PREFETCH_VARIANT-custom}
  NEURAL_SPECULATOR_ENABLED=false
  local predict_fanout_raw
  local pipeline_depth_raw
  local governor_precision_floor_default=0.05
  local governor_contention_weight_default=1.0
  local governor_base_threshold_default=0.02
  case "$PREFETCH_VARIANT" in
    custom)
      PREDICTOR_MODE=custom
      predict_fanout_raw=${MER_PROMPT2_PREDICT_FANOUT-2}
      pipeline_depth_raw=${MER_PROMPT2_PIPELINE_DEPTH-3}
      FIRST_ORDER_ENABLED=${MER_PROMPT2_FIRST_ORDER_ENABLED-true}
      SECOND_ORDER_ENABLED=${MER_PROMPT2_SECOND_ORDER_ENABLED-true}
      FALLBACK_PRIOR_FILL_ENABLED=${MER_PROMPT2_FALLBACK_PRIOR_FILL_ENABLED-true}
      FANOUT_IS_UPPER_BOUND=${MER_PROMPT2_FANOUT_IS_UPPER_BOUND-false}
      PREFETCH_GOVERNOR_ENABLED=${MER_PROMPT2_PREFETCH_GOVERNOR_ENABLED-false}
      ;;
    demand-only)
      PREDICTOR_MODE=demand-only
      predict_fanout_raw=0
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=true
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=true
      FANOUT_IS_UPPER_BOUND=false
      PREFETCH_GOVERNOR_ENABLED=false
      ;;
    current-f2)
      PREDICTOR_MODE=legacy-combined
      predict_fanout_raw=2
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=true
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=true
      FANOUT_IS_UPPER_BOUND=false
      PREFETCH_GOVERNOR_ENABLED=false
      ;;
    current-f2-governed)
      PREDICTOR_MODE=legacy-combined
      predict_fanout_raw=2
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=true
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=true
      FANOUT_IS_UPPER_BOUND=false
      PREFETCH_GOVERNOR_ENABLED=true
      ;;
    second-only-f2)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=2
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=false
      ;;
    second-only-f1)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=false
      ;;
    second-only-f1-governed)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=true
      ;;
    second-only-f1-governed-current)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=true
      ;;
    second-only-f1-governed-cw025)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=true
      governor_contention_weight_default=0.25
      ;;
    second-only-f1-governed-bt010-cw025)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=true
      governor_base_threshold_default=0.01
      governor_contention_weight_default=0.25
      ;;
    second-only-f1-governed-bt005-cw000)
      PREDICTOR_MODE=second-order-only
      predict_fanout_raw=1
      pipeline_depth_raw=3
      FIRST_ORDER_ENABLED=false
      SECOND_ORDER_ENABLED=true
      FALLBACK_PRIOR_FILL_ENABLED=false
      FANOUT_IS_UPPER_BOUND=true
      PREFETCH_GOVERNOR_ENABLED=true
      governor_base_threshold_default=0.005
      governor_contention_weight_default=0.0
      ;;
    *)
      echo "MER_PROMPT2_PREFETCH_VARIANT is not a supported Prompt 2 variant; found: $PREFETCH_VARIANT" >&2
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
    FANOUT_IS_UPPER_BOUND \
    PREFETCH_GOVERNOR_ENABLED \
    NEURAL_SPECULATOR_ENABLED; do
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

  prompt2_resolve_governor_tunable \
    MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR \
    "${MER_PROMPT2_PREFETCH_GOVERNOR_PRECISION_FLOOR-$governor_precision_floor_default}" \
    1.0 PREFETCH_GOVERNOR_PRECISION_FLOOR || return
  prompt2_resolve_governor_tunable \
    MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT \
    "${MER_PROMPT2_PREFETCH_GOVERNOR_CONTENTION_WEIGHT-$governor_contention_weight_default}" \
    null PREFETCH_GOVERNOR_CONTENTION_WEIGHT || return
  prompt2_resolve_governor_tunable \
    MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD \
    "${MER_PROMPT2_PREFETCH_GOVERNOR_BASE_THRESHOLD-$governor_base_threshold_default}" \
    null PREFETCH_GOVERNOR_BASE_THRESHOLD || return
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
  local prefetch_governor_enabled=${12}
  local prefetch_governor_precision_floor=${13:-${PREFETCH_GOVERNOR_PRECISION_FLOOR:-0.05}}
  local prefetch_governor_contention_weight=${14:-${PREFETCH_GOVERNOR_CONTENTION_WEIGHT:-1.0}}
  local prefetch_governor_base_threshold=${15:-${PREFETCH_GOVERNOR_BASE_THRESHOLD:-0.02}}

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
    -e "s|@PREFETCH_GOVERNOR_ENABLED@|$prefetch_governor_enabled|g" \
    -e "s|@PREFETCH_GOVERNOR_PRECISION_FLOOR@|$prefetch_governor_precision_floor|g" \
    -e "s|@PREFETCH_GOVERNOR_CONTENTION_WEIGHT@|$prefetch_governor_contention_weight|g" \
    -e "s|@PREFETCH_GOVERNOR_BASE_THRESHOLD@|$prefetch_governor_base_threshold|g" \
    "$template" > "$output"
}
