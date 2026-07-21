#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMPARE="$ROOT/scripts/compare_qwen3_coder_prompt2_prefetch_ablation.sh"
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mer-prompt2-phase4a-compare.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

write_collection() {
  local dir=$1
  local fanout=$2
  local depth=$3
  local active=$4
  local tps_scale=$5
  local latency_scale=$6
  local qualification=$7
  local parity=$8
  local host=$9
  local commit=${10}
  local binary=${11}

  mkdir -p "$dir"
  jq -n \
    --argjson fanout "$fanout" \
    --argjson depth "$depth" \
    --argjson active "$active" \
    --arg host "$host" \
    --arg commit "$commit" \
    --arg binary "$binary" '
    {
      schema: {name:"mer-prompt2-phase4a-ablation-provenance", version:1},
      experiment_name: "prompt2-phase4a-prefetch-ablation",
      predict_fanout: $fanout,
      pipeline_depth: $depth,
      prefetch_expected_active: $active,
      git_commit_full: $commit,
      model_hashes: {
        config_json_sha256: "config-hash",
        dense_manifest_sha256: "manifest-hash"
      },
      prompt_hashes: {
        short_sha256: "short-hash",
        medium_sha256: "medium-hash"
      },
      host: {
        hostname: $host,
        logical_cpu_count: 32,
        requested_cpu_mask: "0-31",
        effective_cpu_mask: "0-31"
      },
      model_mount_identity: {
        target: "/mnt/localssd",
        source: "/dev/nvme0n1",
        fstype: "ext4",
        options: "rw,noatime,nodiratime"
      },
      cargo_features: ["avx512", "blas", "io_uring", "q8-candle-reference", "tokenizer"],
      binary_sha256: $binary,
      collector_mode: "four-case"
    }
  ' > "$dir/ablation-provenance.json"

  jq -n \
    --argjson fanout "$fanout" \
    --argjson depth "$depth" \
    --argjson active "$active" \
    --argjson tps_scale "$tps_scale" \
    --argjson latency_scale "$latency_scale" \
    --argjson qualification "$qualification" \
    --argjson parity "$parity" '
    def counters($value):
      {
        prefetch_submitted: $value,
        prefetch_completed: $value,
        prefetch_used: $value,
        prefetch_bytes: $value,
        useful_prefetch_bytes: $value,
        unused_prefetch_bytes_at_sample: $value,
        prefetch_dropped_concurrency: $value,
        prefetch_dropped_pool_starved: $value,
        prefetch_dropped_governor: $value,
        prefetch_dropped_bytes: $value
      };
    def make_case($fixture; $base_tps; $hit_rate; $misses; $bytes; $storage; $critical; $fraction):
      (if $active then 5 else 0 end) as $counter |
      {
        schema: {name:"mer-prompt2-case-summary", version:1},
        cache_slots: 1536,
        prompt_fixture: $fixture,
        decode_tps_mean: ($base_tps * $tps_scale),
        cache_hit_rate: $hit_rate,
        cache_misses_total: $misses,
        ssd_bytes_total: $bytes,
        output_token_parity: $parity,
        qualification_passed: $qualification,
        phase4a_prefetch: {
          requested_predict_fanout: $fanout,
          requested_pipeline_depth: $depth,
          expected_active: $active,
          reported_enabled_values: [$active],
          all_runs_reported_expected_enabled: true,
          counters: counters($counter),
          all_prefetch_counters_zero: ($counter == 0),
          prompt_demand_reads_issued_while_speculative_reads_active: (if $active then 3 else 0 end),
          decode_demand_reads_issued_while_speculative_reads_active: (if $active then 7 else 0 end)
        },
        phase3a_decode: {
          physical_read_issue_to_completion_mean_seconds: ($storage * $latency_scale),
          physical_read_issue_to_completion_max_seconds: (2 * $storage * $latency_scale),
          layer_expert_fetch_critical_path_mean_seconds: ($critical * $latency_scale),
          layer_expert_fetch_critical_path_max_seconds: (2 * $critical * $latency_scale),
          decode_wall_fraction_attributable_to_layer_expert_fetch: ($fraction * $latency_scale)
        }
      };
    {
      schema: {name:"mer-prompt2-four-case-summary", version:1},
      cases: [
        make_case("short"; 100; 0.75; 100; 1000; 0.010; 0.020; 0.30),
        make_case("medium"; 64; 0.80; 80; 800; 0.012; 0.025; 0.35)
      ],
      qualification_passed: $qualification
    }
  ' > "$dir/four-case-summary.json"
}

write_matrix() {
  local root=$1
  local b_tps=$2
  local b_latency=$3
  local a2_tps=$4
  local b_qualification=${5:-true}
  local b_parity=${6:-true}
  local b_commit=${7:-phase4a-commit}

  write_collection "$root/a1" 2 3 true 1 1 true true qualified-g2 phase4a-commit binary-hash
  write_collection "$root/b" 0 1 false "$b_tps" "$b_latency" \
    "$b_qualification" "$b_parity" qualified-g2 "$b_commit" binary-hash
  write_collection "$root/a2" 2 3 true "$a2_tps" 1 true true qualified-g2 phase4a-commit binary-hash
}

write_matrix "$TEST_ROOT/neutral" 1 0.90 1
SOURCE_CKSUM_BEFORE=$(find "$TEST_ROOT/neutral" -type f -exec cksum {} \; | sort | cksum)
bash "$COMPARE" \
  "$TEST_ROOT/neutral/a1" "$TEST_ROOT/neutral/b" "$TEST_ROOT/neutral/a2" \
  > "$TEST_ROOT/neutral.json"
SOURCE_CKSUM_AFTER=$(find "$TEST_ROOT/neutral" -type f -exec cksum {} \; | sort | cksum)
test "$SOURCE_CKSUM_BEFORE" = "$SOURCE_CKSUM_AFTER"
jq -e '
  .schema == {name:"mer-prompt2-phase4a-prefetch-ablation", version:1} and
  (keys | length) == 7 and
  .provenance.all_passed == true and
  .control_stability.all_passed == true and
  .no_prefetch_invariants.all_passed == true and
  .no_prefetch_vs_control_mean.streaming_geometric_mean_decode_tps.percent_delta == 0 and
  .no_prefetch_vs_control_mean.prefetch_counters.total.prefetch_submitted == 0 and
  .interpretation.classification == "prefetch_net_negative_or_unnecessary"
' "$TEST_ROOT/neutral.json" >/dev/null

write_matrix "$TEST_ROOT/helpful" 0.97 0.90 1
bash "$COMPARE" \
  "$TEST_ROOT/helpful/a1" "$TEST_ROOT/helpful/b" "$TEST_ROOT/helpful/a2" \
  > "$TEST_ROOT/helpful.json"
jq -e '
  .interpretation.no_prefetch_streaming_geometric_mean_percent_delta < -2.999 and
  .interpretation.foreground_storage_or_critical_path_improved_consistently == true and
  .interpretation.classification == "prefetch_helpful_but_contentious"
' "$TEST_ROOT/helpful.json" >/dev/null

write_matrix "$TEST_ROOT/not-primary" 0.97 1.10 1
bash "$COMPARE" \
  "$TEST_ROOT/not-primary/a1" "$TEST_ROOT/not-primary/b" "$TEST_ROOT/not-primary/a2" \
  > "$TEST_ROOT/not-primary.json"
jq -e '
  .interpretation.foreground_storage_or_critical_path_improved_consistently == false and
  .interpretation.classification == "prefetch_not_primary_bottleneck"
' "$TEST_ROOT/not-primary.json" >/dev/null

write_matrix "$TEST_ROOT/unstable" 1 0.90 1.05
bash "$COMPARE" \
  "$TEST_ROOT/unstable/a1" "$TEST_ROOT/unstable/b" "$TEST_ROOT/unstable/a2" \
  > "$TEST_ROOT/unstable.json"
jq -e '
  .control_stability.short_fixture_within_threshold == false and
  .control_stability.all_passed == false and
  .interpretation.classification == "inconclusive"
' "$TEST_ROOT/unstable.json" >/dev/null

write_matrix "$TEST_ROOT/parity-failure" 1 0.90 1 true false
bash "$COMPARE" \
  "$TEST_ROOT/parity-failure/a1" "$TEST_ROOT/parity-failure/b" "$TEST_ROOT/parity-failure/a2" \
  > "$TEST_ROOT/parity-failure.json"
jq -e '
  .provenance.streaming_output_parity_passed == false and
  .no_prefetch_invariants.output_parity_passed == false and
  .interpretation.classification == "inconclusive"
' "$TEST_ROOT/parity-failure.json" >/dev/null

write_matrix "$TEST_ROOT/provenance-failure" 1 0.90 1 true true different-commit
bash "$COMPARE" \
  "$TEST_ROOT/provenance-failure/a1" "$TEST_ROOT/provenance-failure/b" "$TEST_ROOT/provenance-failure/a2" \
  > "$TEST_ROOT/provenance-failure.json"
jq -e '
  .provenance.git_commits.all_match == false and
  .provenance.all_passed == false and
  .interpretation.classification == "inconclusive"
' "$TEST_ROOT/provenance-failure.json" >/dev/null

set +e
bash "$COMPARE" "$TEST_ROOT/neutral/a1" "$TEST_ROOT/neutral/b" >/dev/null 2>&1
TOO_FEW_STATUS=$?
bash "$COMPARE" \
  "$TEST_ROOT/neutral/a1" "$TEST_ROOT/neutral/b" "$TEST_ROOT/neutral/a2" extra \
  >/dev/null 2>&1
TOO_MANY_STATUS=$?
set -e
test "$TOO_FEW_STATUS" -eq 2
test "$TOO_MANY_STATUS" -eq 2

echo "Prompt 2 Phase 4A A-B-A comparison fixtures (jq 1.6 syntax): PASS"
