#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

TRACE="$TMP/trace.jsonl"
REPORT="$TMP/report.json"

cat > "$TRACE" <<'EOF'
{"schema_name":"mer-prompt2-phase4b-routing-trace","schema_version":1,"event_id":1,"event_type":"routing","monotonic_timestamp_ns":1,"payload":{"request_id":1,"token_index":0,"layer_index":0,"ordered_global_topk_expert_ids":[0,1,2,3,4,5,6,7]}}
{"schema_name":"mer-prompt2-phase4b-routing-trace","schema_version":1,"event_id":2,"event_type":"initial_demand_lookup","monotonic_timestamp_ns":2,"payload":{"classification":"ordinary_miss"}}
{"schema_name":"mer-prompt2-phase4b-routing-trace","schema_version":1,"event_id":3,"event_type":"physical_read_issued","monotonic_timestamp_ns":3,"payload":{"read_id":1,"issue_timestamp_ns":3,"read_classification":"demand"}}
{"schema_name":"mer-prompt2-phase4b-routing-trace","schema_version":1,"event_id":4,"event_type":"physical_read_completed","monotonic_timestamp_ns":13,"payload":{"read_id":1,"completion_timestamp_ns":13,"read_classification":"demand"}}
{"schema_name":"mer-prompt2-phase4b-routing-trace","schema_version":1,"event_id":5,"event_type":"routing","monotonic_timestamp_ns":14,"payload":{"request_id":1,"token_index":0,"layer_index":1,"ordered_global_topk_expert_ids":[8,9,10,11,12,13,14,15]}}
{"schema_name":"mer-prompt2-phase4b-routing-trace","schema_version":1,"event_id":6,"event_type":"initial_demand_lookup","monotonic_timestamp_ns":15,"payload":{"classification":"ready_prefetched_resident"}}
EOF

python3 "$ROOT/scripts/replay_qwen3_coder_prompt2_phase4b.py" \
  "$TRACE" \
  --pretty > "$REPORT"

jq -e '
  .schema == {"name":"mer-prompt2-phase4b-replay","version":1} and
  .validation.valid == true and
  .validation.routing_events == 2 and
  .policies.current_policy_reconstruction.lookup_classes.ordinary_miss == 1 and
  .policies.current_policy_reconstruction.lookup_classes.ready_prefetched_resident == 1 and
  .policies.no_prefetch_reconstruction.misses == 16 and
  .policies.perfect_next_layer_fanout_8.misses == 8 and
  .policies.global_pooled_lru.cache_geometry.kind == "global_pooled_lru" and
  .latency_oracles.ideal_zero_latency_oracle_seconds == 0 and
  .latency_oracles.recorded_latency_oracle.demand.samples == 1
' "$REPORT" >/dev/null

echo "phase4b replay fixture: PASS"
