# Distributed serving (gRPC transport behind `--features grpc`)

The default `micro-expert-router` server runs as a single process.
This document records the partitioning plan referenced in
`docs/production.md` and the gRPC transport that now implements it
behind the off-by-default `grpc` cargo feature.

## Goals

* Scale expert capacity past one host's RAM + NVMe budget.
* Stay strictly OpenAI-shape compatible from the client's POV.
* Re-use the existing `gating::Router` interface — partitioning is a
  transport detail, not an algorithm change.

## Sharding scheme

Hash expert ids by `id % num_nodes` and assign each shard to one
node. The gating layer on the request-receiving node computes the
full top-K (which is cheap — it's a single `d_model × num_experts`
SGEMV) and then issues one RPC per shard with the **subset** of
top-K ids that live there.

```
client → router-node (gates) → [shard-0, shard-1, …] → router-node (combines) → client
```

The combiner sums weighted FFN outputs the same way the
single-process `combine_moe_outputs` does today. Top-K is small
(typically 2 to 8) so the fan-out is bounded.

## On-wire format (sketch)

* gRPC, `proto3`, with two RPCs:
  * `RouteExperts(request_id, layer_idx, expert_ids[], hidden_state)
    → (ffn_out: f16[d_model])`
  * `Health() → (free_blocks, expert_read_failures, …)`
* `hidden_state` and `ffn_out` carried as packed `bytes` (f16 to
  halve wire size; the engine already loses no precision against
  the f32 path because gate/up are followed by SwiGLU + downcast).
* HTTP/2 with KEEPALIVE so connections survive idle periods.

## Failure semantics

* If a shard times out, the combiner contributes a zero vector for
  the missing experts and increments
  `mer_expert_read_failures_total` — same code path as the local
  fault-tolerance work, surfaced via `/v1/admin/health/experts`.
* Shards must be horizontally replicated (active/active or
  active/standby) since losing one shard otherwise loses
  `1/num_nodes` of the experts.

## Status

The sharded `RouteExperts` RPC is **implemented**, and the expert
namespace partitioning is **wired into the serving path**: the batch
scheduler's warm pre-pass consults a `ShardRouter` for every peeked
expert id, keeping locally-owned ids on the NVMe path and fetching
remote ids from their owning shard (best-effort, with structured
failure counters). The dependency-free routing/wire layer is always in
tree; the `tonic`/`prost` gRPC transport compiles in behind
`--features grpc`.

Enable partitioning with the `[distributed]` config section:

```toml
[distributed]
enabled = true
# Every node in the mesh, in shard order (expert id % nodes.len()).
nodes = ["node-a:50051", "node-b:50051", "node-c:50051"]
# This process's position in `nodes` (its experts stay local).
self_index = 0
# Per-call deadline for remote fetches (milliseconds).
remote_fetch_timeout_ms = 250
```

`[distributed]` requires `[real_transformer] enabled = true` — expert
sharding is wired through the batch scheduler, which only runs with
the real transformer (`serve` rejects the config otherwise).

At startup `serve` builds an `RpcShardRouter` over the documented
`id % num_nodes` placement (`RpcShardRouter::from_modulo_placement`)
spanning the layer-qualified global expert namespace
(`num_layers × num_experts`); with the section absent (the default)
the scheduler keeps the single-node `LocalShardRouter` and behaves
identically to before.

Always available (no feature flag, no extra dependencies):

* The shard-routing function ([`crate::rpc::shard_for_expert`]) and
  the top-K bucketing helper
  ([`crate::rpc::group_top_k_by_shard`]) — the only routing
  decisions the request-receiving node makes when the partitioning
  is enabled.
* The packed wire-format frames
  ([`crate::rpc::RouteExpertsRequest`] /
  [`crate::rpc::RouteExpertsResponse`]) with explicit
  `encode` / `decode` — bit-identical to the layout the
  `tonic`-generated decoder produces against the matching
  `proto/route_experts.proto` schema. Carrying both `hidden_state`
  and `ffn_out` as packed `bytes` keeps the on-wire path zero-copy,
  and the `grpc` module reuses these exact frames + their f16
  pack/unpack helpers as the single source of truth for the layout.
* `proto/route_experts.proto` — the gRPC contract. Service surface is
  two RPCs: `RouteExperts` and `Health`, matching the sketch above.
* `crate::distributed::RpcShardRouter` and `map_tonic_status` — the
  `ShardRouter` impl and the tonic-status → `ShardRouterError`
  translation. Without the `grpc` feature its `fetch_remote` returns a
  structured `Unreachable` (no `tonic` linked in).

Compiled in with `--features grpc` ([`crate::grpc`]):

* **Server** — implement the `grpc::ShardCompute` trait (produce one
  FFN output per owned expert id) and call `grpc::serve(addr, backend)`
  (or `serve_with_shutdown`) to expose the `ExpertShard` gRPC service.
* **Client** — `grpc::ShardClient::connect(node)` issues one
  `route_experts` call per shard plus a `health` probe, bridging the
  packed-f16 `rpc` frames to the generated `prost` messages.
* **Router probe** — `grpc::probe_remote` performs a `Health`
  round-trip (with a connect+call timeout) that `RpcShardRouter::
  fetch_remote` uses to confirm a shard is reachable before routing to
  it; any `tonic::Status` is mapped through `map_tonic_status`.

### Regenerating the gRPC stubs

The generated tonic/prost code is committed at `src/grpc_gen.rs`, so
the build needs **no `protoc`** and the default build stays lean
(`tonic`/`prost` pull in ~150 transitive crates only under
`--features grpc`). Regenerate it only when `proto/route_experts.proto`
changes:

1. In a throwaway crate, add `tonic-build` (0.12) as a build
   dependency and, from a `build.rs`, call
   `tonic_build::configure().out_dir("gen").compile_protos(
   &["proto/route_experts.proto"], &["proto"])` (requires `protoc` on
   the build host).
2. Copy the generated `gen/*.rs` into `rust-engine/src/grpc_gen.rs`,
   keeping the header comment and the `#![allow(clippy::all,
   missing_docs, unreachable_pub)]` lint relaxations at the top.
3. Rebuild with `cargo build --features grpc` and run
   `cargo test --features grpc` to confirm the round-trip tests pass.

This document captures the design and the implementation surface; the
routing/wire layer is exercised by the unit tests in `src/rpc.rs` and
the gRPC server/client by the live round-trip tests in `src/grpc.rs`.
