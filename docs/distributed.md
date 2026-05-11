# Distributed serving sketch (planned, not yet implemented)

The current `micro-expert-router` server runs as a single process.
This document records the partitioning plan referenced in
`docs/production.md` so we can implement it incrementally without
re-doing the design.

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

## Out of scope for this PR

The gist asks to *document* the distributed path, not to implement
it. The intent is to keep the surface area visible so a follow-up
PR can land it without re-deriving the design.
