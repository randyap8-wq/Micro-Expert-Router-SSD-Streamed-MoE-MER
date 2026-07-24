#!/usr/bin/env python3
"""Read-only Prompt 2 Phase 4B trace validator and first cache/prefetch oracle.

The tool never writes into benchmark artifact directories. It reconstructs the
observed current policy, a demand-only baseline, perfect next-layer fanout
1/2/4/8, fixed per-layer LRU, pooled global LRU, and ideal/recorded latency
oracles from a bounded Phase 4B JSONL trace.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA_NAME = "mer-prompt2-phase4b-routing-trace"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Route:
    request_id: int
    token_index: int
    layer: int
    experts: tuple[int, ...]


class Lru:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(capacity, 0)
        self.items: collections.OrderedDict[int, None] = collections.OrderedDict()

    def access(self, expert: int) -> bool:
        hit = expert in self.items
        if hit:
            self.items.move_to_end(expert)
        elif self.capacity:
            self.items[expert] = None
            if len(self.items) > self.capacity:
                self.items.popitem(last=False)
        return hit

    def prefetch(self, expert: int) -> None:
        self.access(expert)


def load_events(path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    truncated = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if event.get("schema_name") != SCHEMA_NAME:
                raise ValueError(
                    f"{path}:{line_number}: schema_name must be {SCHEMA_NAME!r}"
                )
            if event.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: schema_version must be {SCHEMA_VERSION}"
                )
            event_id = event.get("event_id")
            timestamp = event.get("monotonic_timestamp_ns")
            if not isinstance(event_id, int) or event_id <= 0:
                raise ValueError(f"{path}:{line_number}: event_id must be positive")
            if event_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate event_id {event_id}")
            if not isinstance(timestamp, int) or timestamp < 0:
                raise ValueError(
                    f"{path}:{line_number}: monotonic_timestamp_ns must be nonnegative"
                )
            seen_ids.add(event_id)
            truncated |= event.get("event_type") == "trace_truncated"
            events.append(event)
    events.sort(key=lambda event: (event["monotonic_timestamp_ns"], event["event_id"]))
    return events, {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "event_count": len(events),
        "trace_truncated": truncated,
        "unique_event_ids": len(seen_ids) == len(events),
    }


def payload(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("payload", {})
    return value if isinstance(value, dict) else {}


def extract_routes(events: Iterable[dict[str, Any]]) -> list[Route]:
    routes: list[Route] = []
    for event in events:
        if event.get("event_type") != "routing":
            continue
        body = payload(event)
        experts = body.get("ordered_global_topk_expert_ids")
        if not isinstance(experts, list) or not all(isinstance(item, int) for item in experts):
            raise ValueError(f"routing event {event['event_id']} has invalid expert list")
        routes.append(
            Route(
                request_id=int(body["request_id"]),
                token_index=int(body["token_index"]),
                layer=int(body["layer_index"]),
                experts=tuple(experts),
            )
        )
    return routes


def cache_simulation(
    routes: list[Route],
    *,
    per_layer_capacity: int | None,
    global_capacity: int | None,
    perfect_fanout: int,
) -> dict[str, Any]:
    if (per_layer_capacity is None) == (global_capacity is None):
        raise ValueError("select exactly one cache geometry")
    global_lru = Lru(global_capacity or 0)
    layer_lrus: dict[int, Lru] = collections.defaultdict(
        lambda: Lru(per_layer_capacity or 0)
    )
    hits = 0
    misses = 0
    layers_with_misses = 0
    request_boundaries = 0
    for index, route in enumerate(routes):
        lru = global_lru if global_capacity is not None else layer_lrus[route.layer]
        layer_misses = 0
        for expert in route.experts:
            if lru.access(expert):
                hits += 1
            else:
                misses += 1
                layer_misses += 1
        layers_with_misses += layer_misses > 0
        if perfect_fanout and index + 1 < len(routes):
            nxt = routes[index + 1]
            if nxt.request_id != route.request_id:
                request_boundaries += 1
                continue
            next_lru = (
                global_lru
                if global_capacity is not None
                else layer_lrus[nxt.layer]
            )
            for expert in nxt.experts[:perfect_fanout]:
                next_lru.prefetch(expert)
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "layers_with_misses": layers_with_misses,
        "request_boundaries_not_prefetched_across": request_boundaries,
        "perfect_next_layer_fanout": perfect_fanout,
        "cache_geometry": (
            {"kind": "per_layer_lru", "slots_per_layer": per_layer_capacity}
            if per_layer_capacity is not None
            else {"kind": "global_pooled_lru", "slots": global_capacity}
        ),
    }


def current_policy(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    classes: collections.Counter[str] = collections.Counter()
    for event in events:
        if event.get("event_type") == "initial_demand_lookup":
            classes[str(payload(event).get("classification", "unknown"))] += 1
    ready = classes["ready_prefetched_resident"]
    resident = classes["ordinary_resident"]
    joins = classes["speculative_read_in_flight"]
    misses = classes["ordinary_miss"]
    total = ready + resident + joins + misses
    return {
        "reconstruction_source": "initial_demand_lookup events",
        "lookup_classes": dict(sorted(classes.items())),
        "resident_or_ready_hit_rate": (ready + resident) / total if total else 0.0,
        "late_join_rate": joins / total if total else 0.0,
        "ordinary_miss_rate": misses / total if total else 0.0,
    }


def recorded_latencies(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    issued: dict[int, int] = {}
    classes: dict[int, str] = {}
    durations: dict[str, list[int]] = collections.defaultdict(list)
    for event in events:
        event_type = event.get("event_type")
        body = payload(event)
        if event_type == "physical_read_issued":
            issued[int(body["read_id"])] = int(body["issue_timestamp_ns"])
            classes[int(body["read_id"])] = str(body["read_classification"])
        elif event_type == "physical_read_completed":
            read_id = int(body["read_id"])
            start = issued.get(read_id)
            if start is not None:
                durations[classes.get(read_id, "unknown")].append(
                    int(body["completion_timestamp_ns"]) - start
                )
    by_class: dict[str, Any] = {}
    for read_class, samples in sorted(durations.items()):
        by_class[read_class] = {
            "samples": len(samples),
            "total_seconds": sum(samples) / 1_000_000_000,
            "mean_seconds": statistics.fmean(samples) / 1_000_000_000,
            "max_seconds": max(samples) / 1_000_000_000,
        }
    return {
        "ideal_zero_latency_oracle_seconds": 0.0,
        "recorded_latency_oracle": by_class,
        "causal_claim": (
            "none; recorded service durations are observations, not proof that "
            "speculation caused foreground delay"
        ),
    }


def build_report(
    events: list[dict[str, Any]],
    validation: dict[str, Any],
    per_layer_slots: int,
    global_slots: int,
) -> dict[str, Any]:
    routes = extract_routes(events)
    no_prefetch_per_layer = cache_simulation(
        routes,
        per_layer_capacity=per_layer_slots,
        global_capacity=None,
        perfect_fanout=0,
    )
    policies: dict[str, Any] = {
        "current_policy_reconstruction": current_policy(events),
        "no_prefetch_reconstruction": no_prefetch_per_layer,
        "current_per_layer_lru": no_prefetch_per_layer,
        "global_pooled_lru": cache_simulation(
            routes,
            per_layer_capacity=None,
            global_capacity=global_slots,
            perfect_fanout=0,
        ),
    }
    for fanout in (1, 2, 4, 8):
        policies[f"perfect_next_layer_fanout_{fanout}"] = cache_simulation(
            routes,
            per_layer_capacity=per_layer_slots,
            global_capacity=None,
            perfect_fanout=fanout,
        )
    return {
        "schema": {"name": "mer-prompt2-phase4b-replay", "version": 1},
        "validation": {
            **validation,
            "routing_events": len(routes),
            "valid": bool(routes) and validation["unique_event_ids"],
        },
        "inputs": {
            "per_layer_lru_slots": per_layer_slots,
            "global_pooled_lru_slots": global_slots,
        },
        "policies": policies,
        "latency_oracles": recorded_latencies(events),
        "unsupported_in_first_replay": [
            "Belady cache replacement",
            "optimized per-layer allocation",
            "lookahead 2/3/6/12 measured scheduling",
            "demand-priority queue simulation",
            "probationary speculative cache",
            "incremental expert execution timing",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--per-layer-slots", type=int, default=32)
    parser.add_argument("--global-slots", type=int, default=1536)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if args.per_layer_slots <= 0 or args.global_slots <= 0:
        parser.error("cache slot counts must be positive")
    try:
        events, validation = load_events(args.trace)
        report = build_report(
            events, validation, args.per_layer_slots, args.global_slots
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"phase4b replay validation failed: {error}", file=sys.stderr)
        return 1
    json.dump(
        report,
        sys.stdout,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
