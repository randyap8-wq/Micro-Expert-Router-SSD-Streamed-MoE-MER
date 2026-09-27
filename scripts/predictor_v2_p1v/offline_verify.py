"""Immutable-byte P1R verification and separate, new-run P1V derived evidence.

Raw trace decoding and classifier ordering are ported from exact ed60f P1R.
Only the cross-arm generation/payload premise can change. Ownership, epochs,
writer identities, publication, binding, clean routes and accounting cannot.
The externally retained seal SHA256 is required; a self-asserted root is not
authentication. No timing comparison, throughput or performance verdict runs.
"""

import argparse
import copy
import struct
from pathlib import Path

from . import (AUTHORITY, IDS, LAUNCHES, MODEL, SOURCE_FILES, InvalidEvidence, canonical_json, digest,
               publish_new, require, sha256, strict_json, uint)
from .supervisor import validate_attempt, validate_enforcement

SCHEMA = "mer.predictor-v2-p1r-critical-path-attribution.v1"
PROTOCOL = "mer.predictor-v2-p1r-arm.v1"
POSITIONS, ATTEMPTS, WIDTH = 143, 98, 8
PASS_STRIDE, ATTEMPT_STRIDE = 56, 116
STRIDE = 24 + ATTEMPTS * ATTEMPT_STRIDE
CAPACITY = 144 * STRIDE
EVENTS = {name: i for i, name in enumerate((
    "Observation", "Freeze", "Launch", "Lease", "Claim", "Spawn", "WriterStart",
    "Payload", "Close", "LeaseRelease", "DEntry", "DP1e", "Publication", "Credit",
    "Retire", "DeferredRetire", "CleanRoute", "RequestStart", "RequestEnd", "P0Install", "P0Reservation"))}
EVENTS.update({name: 24 + i for i, name in enumerate((
    "Segment", "Boundary", "Binding", "Service", "Selected", "Probe", "SourceSet",
    "SourceBranch", "LogicalSet", "InstallSet", "CopySet", "LockedSet", "SourceExpert",
    "LogicalSource", "Reservation", "Stage", "Commit", "Installed"))})
LAUNCH_FIELDS = (
    "launch_considered", "source_first_attempt_clean", "source_checkpoint_recovered_clean",
    "source_not_eligible", "p1j_not_ready", "no_pending_freeze", "pending_sidecar_existing",
    "retirement_busy", "candidate_identity_invalid", "freeze_incomplete",
    "candidate_already_current_at_f", "physical_evidence_missing_or_invalid",
    "not_logical_materialized", "missing_logical_generation", "host_lease_busy",
    "host_lease_missing", "host_lease_stale", "host_lease_wrong_payload_kind",
    "host_lease_wrong_dtype", "host_lease_wrong_length", "sidecar_lock_busy",
    "sidecar_occupied", "sidecar_identity_rejected", "sidecar_epoch_exhausted",
    "sidecar_writer_sequence_exhausted", "p0_acquire_failed", "writer_spawned")
P0_FIELDS = (
    "emitted", "terminal_predictions", "live_predictions", "admission_pending", "accepted",
    "rejected", "skipped", "source_leaders", "source_followers", "source_completed",
    "source_failed", "source_cancelled", "source_live", "reservations", "reservations_committed",
    "reservations_aborted", "reservations_live", "install_owners", "install_followers",
    "available_installs", "direct_matching_demand_credits")


def slot(position, event, attempt=0, pass_index=0, member=0):
    require(0 <= position <= POSITIONS and 0 <= attempt < ATTEMPTS
            and pass_index in (0, 1) and 0 <= member < WIDTH, "trace index bounds")
    e = EVENTS[event]
    if e < 24:
        offset = e
    elif e < 28:
        offset = 24 + attempt * ATTEMPT_STRIDE + e - 24
    elif e < 36:
        offset = 24 + attempt * ATTEMPT_STRIDE + 4 + pass_index * PASS_STRIDE + e - 28
    else:
        offset = 24 + attempt * ATTEMPT_STRIDE + 4 + pass_index * PASS_STRIDE + 8 + member * 6 + e - 36
    return position * STRIDE + offset


class Trace:
    def __init__(self, snapshot):
        require(snapshot["invalid"] is False and snapshot["capacity"] == CAPACITY
                and snapshot["storage_bytes"] == CAPACITY * 104, "invalid recorder bounds")
        layout = snapshot["layout"]
        for key, value in {"words": 12, "position_stride": STRIDE, "position_slots": 24,
                           "attempt_stride": ATTEMPT_STRIDE, "pass_stride": PASS_STRIDE,
                           "member_stride": 6, "timestamp_words": [0, 1],
                           "event_offsets": [[k, v] for k, v in EVENTS.items()],
                           "clock": "checked u64 ns from the same request-local P1E monotonic origin"}.items():
            require(layout[key] == value, "recorder layout mismatch: " + key)
        self.records = {}
        for record in snapshot["records"]:
            index, words = uint(record["slot"]), record["words"]
            require(index < CAPACITY and index % STRIDE not in (21, 22, 23)
                    and index not in self.records and len(words) == 12, "duplicate/invalid trace slot")
            for word in words:
                uint(word)
            require(words[1] >= words[0], "trace chronology")
            self.records[index] = words

    def get(self, p, event, attempt=0, pass_index=0, member=0):
        return self.records.get(slot(p, event, attempt, pass_index, member))

    def need(self, *args):
        value = self.get(*args)
        require(value is not None, "missing trace event " + str(args))
        return value

    def route(self, p):
        r = self.need(p, "CleanRoute")
        require(r[3] == 8 and 1 <= r[2] <= ATTEMPTS, "clean route width/attempt")
        ids = r[4:12]
        require(len(set(ids)) == 8 and all(i < 128 for i in ids), "clean route IDs")
        boundary = self.need(p, "Boundary", r[2] - 1)
        require(boundary[2] == 1 and boundary[3:] == r[3:], "clean route boundary mismatch")
        return ids


def semantic(candidate):
    result = copy.deepcopy(candidate)
    for key in ("runtime_namespace", "request_sequence"):
        result["request"][key] = 0
    for key in ("runtime", "context", "arena"):
        result["namespace"][key] = 0
    return result


def duration(record):
    return record[1] - record[0]


def service(trace, p, expert):
    """Reconstruct causal service fields from raw slots, never trust row claims."""
    result, global_id = [], 47 * 128 + expert
    for attempt in range(ATTEMPTS):
        total = trace.get(p, "Service", attempt)
        if total is None:
            continue
        require(total[2] == 1, "failed foreground service")
        for probe_index in range(2):
            selected = trace.get(p, "Selected", attempt, probe_index)
            if selected is None:
                require(probe_index == 1, "missing selected set")
                continue
            require(selected[3] == 8 and len(set(selected[4:])) == 8, "selected set invalid")
            ids = selected[4:]
            probe = trace.need(p, "Probe", attempt, probe_index)
            require(probe[3] <= 8, "missing set overflow")
            missing = probe[4:4 + probe[3]]
            require(set(missing) <= set(ids) and len(set(missing)) == len(missing), "missing set invalid")
            member = ids.index(global_id) if global_id in ids else None

            def get(event):
                return None if member is None else trace.get(p, event, attempt, probe_index, member)

            stage, commit, installed = (get(e) for e in ("Stage", "Commit", "Installed"))
            is_missing = global_id in missing
            if not is_missing:
                require(all(get(e) is None for e in ("SourceExpert", "LogicalSource", "Reservation", "Stage", "Commit", "Installed")),
                        "physical hit acquired source/install attribution")
            if installed is not None:
                require(installed[2] == 1 and installed[3] == global_id and stage is not None
                        and commit is not None and stage[2:5] == [1, global_id, installed[4]]
                        and commit[2:5] == [1, global_id, installed[4]], "installed stage/commit identity")
            source = trace.get(p, "SourceSet", attempt, probe_index)
            result.append({"attempt": attempt, "re_probe": probe_index, "selected_set": ids,
                           "physical_missing_set": missing, "candidate_physically_missing": is_missing,
                           "foreground_set_ns": duration(total),
                           "source_set_ns": None if source is None else duration(source),
                           "installed_generation": None if installed is None else installed[4],
                           "installed": installed is not None,
                           "scope": "SINGLETON_MISSING_SET" if is_missing and len(missing) == 1 else "SET",
                           "exact_candidate_saved_ns": None})
    return result


def owner(candidate, generation, claim, event):
    return (event[8] == candidate["request"]["runtime_namespace"]
            and event[9] == candidate["request"]["request_sequence"]
            and event[10] == candidate["namespace"]["context"]
            and event[11] == candidate["namespace"]["arena"]
            and event[3] == candidate["sequence"] and event[4] == candidate["expert"]
            and event[5] == generation and event[6] != 0 and event[7] != 0
            and event[3:8] == claim[3:8])


def readiness(close_record, d, consumed):
    if close_record is not None and close_record[2] == 2 and d is not None and close_record[0] > d:
        return "AFTER_D"
    if close_record is None or close_record[2] != 1 or d is None:
        return "UNAVAILABLE"
    lo, hi = close_record[:2]
    if hi < d:
        return "BEFORE_D"
    if lo > d:
        return "READY_DURING_D_WINDOW" if consumed else "AFTER_D"
    return "D_WINDOW_AMBIGUOUS"


def classify(launched, outcome, ordinary, consumed, divergent, available, ready,
             publication, control_absent, services, payload_equivalent=False):
    """Exact frozen P1R order; only two cross-arm generation branches are skipped."""
    missing_paid = any(s["candidate_physically_missing"] and s["installed"]
                       and s["foreground_set_ns"] > 0 and s["source_set_ns"] is not None for s in services)
    if not launched:
        return "NOT_LAUNCHED_OR_INELIGIBLE"
    if outcome == "Censored" or outcome == {"Resolved": {"prediction_hit": False}}:
        return "UNUSED"
    if ordinary and not consumed:
        return "ORDINARY_WON_RACE"
    if divergent and not payload_equivalent:
        return "GENERATION_DIVERGENT"
    if not available and not payload_equivalent:
        return "GENERATION_UNAVAILABLE"
    if ready in ("D_WINDOW_AMBIGUOUS", "READY_DURING_D_WINDOW"):
        return ready
    if ready == "AFTER_D" and publication == 2:
        return "LATE"
    if publication == 1:
        return "PUBLICATION_BUSY"
    if consumed and ready == "BEFORE_D" and control_absent and missing_paid:
        return "READY_AND_CRITICAL"
    if consumed and ready == "BEFORE_D" and not any(s["candidate_physically_missing"] for s in services):
        return "READY_BUT_NOT_CRITICAL"
    return "ATTRIBUTION_UNRESOLVED"


def counter_delta(before, after, delta):
    require(type(before) is dict and bool(before) and set(before) == set(after) == set(delta), "counter schema")
    for key in before:
        require(uint(after[key]) >= uint(before[key]) and after[key] - before[key] == uint(delta[key]), "counter delta: " + key)


def opportunities(observation):
    outcome = observation["outcome"]
    hit = outcome["Resolved"]["prediction_hit"] if isinstance(outcome, dict) else None
    f = observation["freeze"]["current"]
    d = observation["deadline"]["current"] if observation["deadline"] else None
    for value in (hit, f, d):
        require(value is None or type(value) is bool, "nonboolean P1E current/outcome")
    useful = None if hit is None or f is None else hit and not f
    return {"prediction_hit": hit, "physical_miss_at_f": None if f is None else not f,
            "physical_miss_at_d": None if d is None else not d, "useful": useful,
            "target_confirmed_useful": None if useful is None or d is None else useful and not d,
            "already_resident": f, "redundant_route_hit": None if hit is None or f is None else hit and f}


def physical_current(evidence, candidate):
    if evidence is None:
        return None
    snapshot = evidence["snapshot"]
    require(snapshot["namespace"] == candidate["namespace"], "physical namespace mismatch")
    residents = snapshot["residents"]
    require(len(residents) == 8, "physical snapshot width")
    ids, slots, gap = set(), set(), False
    for resident in residents:
        if resident is None:
            gap = True
            continue
        require(not gap and uint(resident["expert"], 32) < 128 and uint(resident["generation"]) > 0
                and uint(resident["bank"], 32) < 4 and uint(resident["slot"], 32) < 8
                and uint(resident["epoch"], 32) > 0, "physical resident invalid")
        location = resident["bank"], resident["slot"]
        require(resident["expert"] not in ids and location not in slots, "physical duplicate")
        ids.add(resident["expert"])
        slots.add(location)
    for key in ("event_cutoff", "committed_installs", "physical_victims"):
        uint(evidence[key])
    return candidate["expert"] in ids


def validate_arm(artifact, name):
    report = artifact["report"]
    require(artifact["arm"] == name and report["arm"] == name and uint(artifact["pid"], 32) > 0, "arm identity")
    require(report["mode"] == ("InertResourceOnly" if name == "Control" else "Active"), "arm mode")
    for key in ("runtime_build_attempted", "runtime_constructed", "ordinary_invariants_pass",
                "initial_state_pass", "sidecar_freshly_initialized"):
        require(report[key] is True, "invalid arm: " + key)
    require(report["errors"] == [] and report["completed_positions"] == POSITIONS, "arm completion")
    require(report["generated_tps"] is None and report["planned_position_tps"] is None, "unexpected performance fields")
    uint(report["request_wall_ns"])
    shutdown = report["runtime_shutdown"]
    require(shutdown["controlled_shutdown_requested"] is True
            and shutdown["all_runtime_resources_released"] is True, "arm shutdown incomplete")
    tokens = report["generated_token_ids"]
    require(len(tokens) == 128, "generated token count")
    encoded = struct.pack("<Q", len(tokens)) + b"".join(struct.pack("<I", uint(t, 32)) for t in tokens)
    require(report["generated_token_ids_sha256"] == digest(encoded), "generated token digest")
    provenance = report["provenance"]
    build = provenance["provenance"]["build"]
    require(build["git_sha"] == AUTHORITY["commit"] and build["dirty"] is False
            and provenance["model_identity"] == MODEL, "frozen build/model identity")
    require(provenance["config_sha256"] == AUTHORITY["config_sha256"]
            and provenance["provenance"]["executable_sha256"] == AUTHORITY["binary_sha256"]
            and report["runtime_resolved_config_sha256"] == provenance["provenance"]["resolved_config_sha256"], "arm input provenance")
    adapter = report["adapter"]
    require(adapter["name"] == "NVIDIA L4" and "580.178.04" in adapter["driver_info"]
            and adapter["software_adapter"] is False, "adapter identity")
    require(report["resource_identity"] is not None and report["runtime_contract"] is not None
            and report["model_load"] is not None, "runtime evidence absent")
    initial, before = report["initial"], report["resources_before_opt_in"]
    resources = initial["resources"]
    require(initial["mode"] == report["mode"] and before["sidecar_initialized"] is False
            and before["sidecar_allocated"] is False and before["namespace"] is None
            and resources["sidecar_initialized"] is True and resources["sidecar_allocated"] is True
            and resources["namespace"] is not None, "fresh sidecar resources")
    require(resources["sidecar_stride_bytes"] == 2654212 and resources["sidecar_bank"] == 1
            and resources["sidecar_slot"] == 0 and resources["ordinary_total_expert_budget_bytes"] == 2147483648
            and len(resources["ordinary_layer_capacities"]) == 48 and resources["ordinary_layer_capacities"][47] == 8
            and resources["ordinary_layer_resident_counts"] == [0] * 48, "ordinary capacity/sidecar geometry")
    for key in ("ordinary_layer_resident_counts", "ordinary_layer_capacities",
                "ordinary_total_expert_budget_bytes", "ordinary_arena_allocation_bytes", "activity_counters"):
        require(before[key] == resources[key], "resource changed: " + key)
    require(resources["activity_counters"] == [0] * 14 and resources["p1e_shadow_present"] is True
            and before["p1e_shadow_present"] is False and initial["committed_position"] == 0
            and initial["pending_sidecar"] is False, "initial activity")
    require(initial["observation_capacity"] == report["observation_capacity_per_collection"] == 439232,
            "frozen observation capacity")
    require(initial["p0"] == {**dict.fromkeys(P0_FIELDS, 0), "incomplete": None, "terminal_categories": []}, "initial P0 activity")
    for key in ("token_loop", "recovery", "production_install", "launch"):
        require(initial[key] and all(v == 0 and type(v) in (int, bool) for v in initial[key].values()), "nonzero initial counters")
    counter_delta(initial["production_install"], report["production_install_after"], report["production_install_delta"])
    counters = report["runtime_counters"]
    for key in ("token_loop", "recovery", "routed_execution"):
        counter_delta(counters[key + "_before"], counters[key + "_after"], counters[key + "_delta"])
    token = counters["token_loop_delta"]
    require(token["tokens_completed"] == 143 and token["fatal_failures"] == 0
            and token["no_progress_failures"] == 0 and token["replay_attempts"] == 0
            and token["token_attempts"] == token["queue_submissions"] == token["boundary_maps"] == token["boundary_readbacks"], "ordinary token accounting")
    recovery = counters["recovery_delta"]
    require(recovery["full_token_replay_attempts"] == 0
            and (token["residency_miss_attempts"] == 0 or recovery["resume_attempts"] > 0), "recovery accounting")
    routed = counters["routed_execution_delta"]
    require(all(routed[k] == 0 for k in ("cpu_routed_expert_dispatches", "gpu_cpu_fallbacks", "degraded_expert_substitutions")), "fallback/degradation")
    launch = []
    for suffix in ("before", "after", "delta"):
        value = report["p1j_launch_" + suffix]
        require(set(value) == set(LAUNCH_FIELDS) | {"incomplete"} and value["incomplete"] is False, "P1J counter schema")
        for key in LAUNCH_FIELDS:
            uint(value[key])
        require(sum(value[k] for k in LAUNCH_FIELDS[1:4]) == value["launch_considered"]
                and sum(value[k] for k in LAUNCH_FIELDS[3:]) == value["launch_considered"], "P1J counters unreconciled")
        launch.append({k: value[k] for k in LAUNCH_FIELDS})
    counter_delta(*launch)
    if name == "Control":
        require(all(v == 0 for s in launch for v in s.values()), "CONTROL P1J movement")
    p0 = report["predictor_v2_snapshot"]
    require(set(p0) == set(P0_FIELDS) | {"incomplete", "terminal_categories"}
            and p0["incomplete"] is None, "P0 incomplete/schema")
    for key in P0_FIELDS:
        uint(p0[key])
    require(all(p0[k] == 0 for k in ("live_predictions", "source_live", "reservations_live", "available_installs", "source_failed"))
            and p0["terminal_predictions"] == p0["emitted"], "P0 live accounting")
    if name == "Control":
        require(all(p0[k] == 0 for k in P0_FIELDS) and p0["terminal_categories"] == [], "CONTROL P0 movement")
    raw = report["raw_p1e_report"]
    require(raw["incomplete"] is None, "P1E incomplete")
    index, sources = {}, set()
    partition = dict.fromkeys(("emitted", "resolved", "pending", "censored", "prediction_hits", "route_misses",
                               "valid_f", "current_at_f", "absent_at_f", "useful", "target_confirmed_useful"), 0)
    computed_opportunities = []
    for obs in raw["observations"]:
        freeze, deadline = obs["freeze"], obs["deadline"]
        c = freeze["candidate"]
        require(set(c) == {"request", "model", "namespace", "source_position", "target_position", "source_layer",
                           "target_layer", "position_distance", "nominal_layer_lead", "source_set", "expert", "score",
                           "signal_revision", "generation", "sequence", "committed_position_cutoff", "table_update_cutoff"},
                "incomplete candidate schema")
        require(c["model"] == {"num_layers": 48, "num_experts": 128, "top_k": 8}
                and c["namespace"]["layer"] == 47 and c["namespace"]["capacity"] == 8,
                "candidate model/namespace geometry")
        for key in ("runtime", "context", "arena"):
            require(uint(c["namespace"][key]) > 0, "zero candidate namespace")
        for key in ("score", "generation", "nominal_layer_lead", "committed_position_cutoff", "table_update_cutoff"):
            uint(c[key])
        require(len(c["source_set"]) == 8 and len(set(c["source_set"])) == 8
                and all(uint(e, 32) < 128 for e in c["source_set"]), "candidate source set")
        for position in (c["source_position"], c["target_position"]):
            absolute = uint(position["absolute_position"])
            require(position == {"absolute_position": absolute,
                                 "position_kind": "Prompt" if absolute < 16 else "Decode",
                                 "decode_index": None if absolute < 16 else absolute - 16}, "position identity")
        source_identity = freeze["source"]
        require(set(source_identity) == {"logical_generation", "logical_materialized", "ram_resident", "permanence"}
                and type(source_identity["logical_materialized"]) is bool
                and type(source_identity["ram_resident"]) is bool and source_identity["permanence"] == "Unknown", "host source schema")
        if source_identity["logical_generation"] is not None:
            uint(source_identity["logical_generation"])
        require(freeze["current"] is physical_current(freeze["physical"], c), "freeze physical/current mismatch")
        seq = uint(c["sequence"])
        require(seq > 0 and seq not in index and obs["incomplete"] is None
                and freeze["incomplete"] is None, "duplicate/incomplete candidate")
        source, target = c["source_position"]["absolute_position"], c["target_position"]["absolute_position"]
        require(c["request"] == raw["request"] and c["namespace"]["runtime"] == raw["request"]["runtime_namespace"]
                and c["source_layer"] == c["target_layer"] == 47 and c["signal_revision"] == 1
                and c["position_distance"] == 1 and source + 1 == target
                and source not in sources and 0 <= source < 143 and 0 <= c["expert"] < 128, "P1E causal identity")
        if deadline is not None:
            require(deadline["incomplete"] is None and deadline["request"] == c["request"]
                    and deadline["position"] == c["target_position"], "P1E deadline identity")
            require(deadline["current"] is physical_current(deadline["physical"], c), "deadline physical/current mismatch")
        outcome = obs["outcome"]
        require(outcome in ("Censored", {"Resolved": {"prediction_hit": True}},
                            {"Resolved": {"prediction_hit": False}}), "pending/invalid outcome")
        partition["emitted"] += 1
        if outcome == "Censored":
            partition["censored"] += 1
        else:
            partition["resolved"] += 1
            partition["prediction_hits" if outcome["Resolved"]["prediction_hit"] else "route_misses"] += 1
        if freeze["current"] is not None:
            partition["valid_f"] += 1
            partition["current_at_f" if freeze["current"] else "absent_at_f"] += 1
        opp = opportunities(obs)
        for key in ("useful", "target_confirmed_useful"):
            partition[key] += int(opp[key] is True)
        computed_opportunities.append([seq, opp])
        index[seq] = obs
        sources.add(source)
    for no in raw["no_emissions"]:
        source = no["source"]["absolute_position"]
        require(no["reason"] == "NoPositiveHistory" and source not in sources, "invalid no-emission coverage")
        sources.add(source)
    require(sources == set(range(143)) and raw["partitions"] == partition, "P1E coverage/partitions")
    require(sorted(raw["opportunities"]) == sorted(computed_opportunities), "P1E opportunity mismatch")
    trace = Trace(report["diagnostics"])
    for p in range(143):
        trace.route(p)
    trace.need(0, "RequestStart")
    trace.need(0, "RequestEnd")
    return report, trace, index


def attribute(control, treatment, parent_rows):
    cr, cs, ci = control
    tr, ts, ti = treatment
    require(set(ci) == set(ti), "orphan semantic nomination")
    require(sorted(cr["raw_p1e_report"]["no_emissions"], key=lambda r: r["source"]["absolute_position"])
            == sorted(tr["raw_p1e_report"]["no_emissions"], key=lambda r: r["source"]["absolute_position"]), "no-emission divergence")
    for p in range(143):
        require(cs.route(p) == ts.route(p), "cross-arm clean route mismatch")
    targets = {r["freeze"]["candidate"]["target_position"]["absolute_position"] for r in ti.values()}
    require(len(targets) == len(ti), "duplicate target")
    for trace in (cs, ts):
        for index in trace.records:
            if index % STRIDE in set(range(3, 10)) | set(range(11, 16)) | {19, 20}:
                require(index // STRIDE in targets, "orphan ownership record")
    require(len(parent_rows) == len(ti), "parent attribution coverage")
    rows, launches, claims, credits = [], [], 0, 0
    for raw_row, seq in zip(parent_rows, sorted(ti)):
        cobs, tobs = ci[seq], ti[seq]
        candidate = tobs["freeze"]["candidate"]
        p, local = candidate["target_position"]["absolute_position"], candidate["expert"]
        require(semantic(cobs["freeze"]["candidate"]) == semantic(candidate)
                and cobs["outcome"] == tobs["outcome"], "cross-arm semantic mismatch")
        for trace, obs in ((cs, cobs), (ts, tobs)):
            freeze = trace.need(p, "Freeze")
            require(freeze[0] == obs["freeze"]["timestamp_ns"]
                    and freeze[2:6] == [seq, local, obs["freeze"]["source"]["logical_generation"] or 0, candidate["generation"]], "freeze identity/clock")
            if obs["deadline"] is not None:
                deadline = trace.need(p, "DP1e")
                require(deadline[0] == obs["deadline"]["timestamp_ns"] and deadline[2] == seq, "deadline identity/clock")
        require(all(cs.get(p, e) is None for e in ("Claim", "Lease", "WriterStart")), "inert CONTROL movement")
        launch = ts.need(p, "Launch")
        launched = launch[2] == 6
        if launched:
            launches.append((seq, p, local, 47 * 128 + local))
        claim = ts.get(p, "Claim")
        claim = claim if claim is not None and claim[2] == 1 else None
        pub, credit = ts.get(p, "Publication"), ts.get(p, "Credit")
        consumed = credit is not None and credit[2] == 1
        if claim is not None:
            claims += 1
            generation = tobs["freeze"]["source"]["logical_generation"]
            uint(generation)
            require(owner(candidate, generation, claim, claim), "claim identity")
            for event in ("WriterStart", "Payload", "Close", "LeaseRelease", "Publication", "Credit", "DeferredRetire"):
                record = ts.get(p, event)
                if record is not None:
                    require(owner(candidate, generation, claim, record), "stale owner/epoch/writer/generation")
            release = ts.need(p, "LeaseRelease")
            close = ts.need(p, "Close")
            ts.need(p, "DeferredRetire")
            require(release[2] == 1, "lease incomplete")
            if launched:
                spawn, start, payload = (ts.need(p, e) for e in ("Spawn", "WriterStart", "Payload"))
                require(start[0] >= spawn[0] and payload[0] >= start[0]
                        and close[0] >= payload[1] and payload[2] == 1, "writer chronology/failure")
        else:
            require(not consumed and not launched, "launch/credit without claim")
        if consumed:
            clean = ts.need(p, "CleanRoute")
            binding = ts.need(p, "Binding", clean[2] - 1)
            require(pub is not None and pub[2] == 0 and local in ts.route(p)
                    and owner(candidate, claim[5], claim, binding), "exact binding/credit mismatch")
            credits += 1
        control_service, treatment_service = service(cs, p, local), service(ts, p, local)
        cg = cobs["freeze"]["source"]["logical_generation"]
        tg = tobs["freeze"]["source"]["logical_generation"]
        divergent = (cg is not None and tg is not None and cg != tg) or any(
            s["installed_generation"] is not None and s["installed_generation"] != tg for s in control_service)
        available = cg is not None and tg is not None
        d = ts.get(p, "DEntry")
        ready = readiness(ts.get(p, "Close"), None if d is None else d[0], consumed)
        publication = None if pub is None else pub[2]
        ordinary = (tobs["deadline"] is not None and tobs["deadline"]["current"] is True) or publication == 5
        missing_paid = any(s["candidate_physically_missing"] and s["installed"] and s["foreground_set_ns"] > 0
                           and s["source_set_ns"] is not None for s in control_service)
        singleton = missing_paid and all(s["scope"] == "SINGLETON_MISSING_SET" for s in control_service if s["candidate_physically_missing"])
        statuses = (["GENERATION_DIVERGENT"] if divergent else []) + (["GENERATION_UNAVAILABLE"] if not available else [])
        if publication == 1:
            statuses.append("PUBLICATION_BUSY")
        if tobs["outcome"] == "Censored":
            statuses.append("CENSORED")
        if missing_paid and not singleton:
            statuses.append("ATTRIBUTION_UNRESOLVED")
        args = (launched, tobs["outcome"], ordinary, consumed, divergent, available, ready, publication,
                cobs["deadline"] is not None and cobs["deadline"]["current"] is False, control_service)
        raw_class = classify(*args)
        expected = {"semantic_candidate": semantic(candidate), "control_raw_candidate": cobs["freeze"]["candidate"],
                    "treatment_raw_candidate": candidate, "control_logical_generation": cg,
                    "treatment_logical_generation": tg, "class": raw_class, "statuses": statuses,
                    "readiness": ready, "consumed": consumed, "published": None if pub is None else publication == 0,
                    "criticality_scope": "FOREGROUND_DEMAND_SET", "exact_candidate_saved_ns": None,
                    "candidate_marginal_attribution": "UNRESOLVED_GENERATION" if divergent or not available
                    else "SINGLETON_FOREGROUND_EXPOSURE" if singleton else "UNRESOLVED"}
        for key, value in expected.items():
            require(raw_row[key] == value, "parent attribution/raw disagreement: " + key)
        for key, services in (("control_service", control_service), ("treatment_service", treatment_service)):
            require(len(raw_row[key]) == len(services), "service coverage mismatch")
            for retained, measured in zip(raw_row[key], services):
                require(all(retained[k] == v for k, v in measured.items()), "raw service disagreement")
        rows.append({"semantic_candidate": semantic(candidate), "raw_class": raw_class,
                     "raw_statuses": statuses, "control_logical_generation": cg,
                     "treatment_logical_generation": tg, "semantic_payload_equivalent": launched,
                     "derived_class": classify(*args, payload_equivalent=launched),
                     "readiness": ready, "consumed": consumed, "criticality_scope": "FOREGROUND_DEMAND_SET",
                     "candidate_marginal_attribution": raw_row["candidate_marginal_attribution"] if not launched
                     else "SINGLETON_FOREGROUND_EXPOSURE" if singleton else "UNRESOLVED",
                     "exact_candidate_saved_ns": None})
    require(tuple(launches) == LAUNCHES, "exact 13-launch equality failed")
    require(sum(1 for i, w in ts.records.items() if i % STRIDE == EVENTS["Launch"] and w[2] == 6) == 13, "orphan/additional launch")
    require(not any(i % STRIDE == EVENTS["Launch"] and w[2] == 6 for i, w in cs.records.items()), "CONTROL launch")
    p0 = tr["predictor_v2_snapshot"]
    require(claims == p0["reservations"] and credits == p0["direct_matching_demand_credits"]
            and tr["p1j_launch_delta"]["writer_spawned"] == 13, "P0/P1J claim-credit accounting")
    return rows


def verify_reports(raw_bytes, envelope, allow_simulated=False, sealed=True):
    content = validate_enforcement(envelope, allow_simulated, through_hash=not sealed)
    require(set(raw_bytes) == {"parent", "control", "treatment"}, "exact three raw reports required")
    hashes = {key: digest(value) for key, value in raw_bytes.items()}
    require(hashes == envelope["receipts"]["P1R_PARENT"]["raw_report_hashes"], "immutable report byte hash mismatch")
    parent, control, treatment = (strict_json(raw_bytes[k]) for k in ("parent", "control", "treatment"))
    attempt = validate_attempt(envelope["attempt"])
    for artifact in (parent, control, treatment):
        require(artifact["schema"] == SCHEMA and artifact["protocol"] == PROTOCOL
                and artifact["diagnostic_id"] == attempt and artifact["pair_index"] == 1, "new-run P1R identity")
        require(artifact["binary_sha256"] == AUTHORITY["binary_sha256"]
                and artifact["request_sha256"] == AUTHORITY["request_sha256"], "binary/request mismatch")
        for key in ("commit", "tree", "parent"):
            require(artifact["source_identity"][key] == AUTHORITY[key], "source identity mismatch")
        require(artifact["source_identity"]["files"] == SOURCE_FILES, "frozen source bytes mismatch")
    require(parent["source_identity"] == control["source_identity"] == treatment["source_identity"], "source divergence")
    require(parent["diagnostic_evidence_valid"] is True and parent["errors"] == []
            and parent["diagnostic_only"] is True and parent["execution_order"] == ["Control", "Treatment"]
            and parent["profile"] == AUTHORITY["profile"] and parent["performance_comparison_authorized"] is False
            and parent["performance_verdict"] == "NOT_AUTHORIZED" and parent["exact_candidate_saved_ns"] is None,
            "invalid P1R parent")
    require(parent["arms"] == [control, treatment], "embedded/standalone child mismatch")
    require(control["pid"] != treatment["pid"] and len(parent["processes"]) == 2, "process isolation")
    for process, artifact, key in zip(parent["processes"], (control, treatment), ("control", "treatment")):
        require(process["arm"] == artifact["arm"] and process["pid"] == artifact["pid"]
                and process["exit_code"] == 0 and process["error"] is None
                and process["artifact_sha256"] == hashes[key], "child exit/byte hash binding")
    c, t = validate_arm(control, "Control"), validate_arm(treatment, "Treatment")
    for key in ("provenance", "resource_identity", "generated_token_ids"):
        require(c[0][key] == t[0][key], "cross-arm identity: " + key)
    rows = attribute(c, t, parent["attribution"])
    derived = {"schema": "mer.p1v.derived.v1", "attempt": attempt, "evidence_mode": envelope["mode"],
               "raw_report_hashes": hashes, "content_root": content["content_root"],
               "continuity": "PASS_CONDITIONAL_TRUSTED_HOST", "exact_launch_gate": "PASS",
               "semantic_payload_equivalent": True, "rows": rows,
               "payload_equivalence_scope": {"global_expert_ids": list(IDS), "full_model": False},
               "exact_candidate_saved_ns": None, "performance_verdict": "NOT_AUTHORIZED",
               "historical_reclassification": False}
    if sealed:
        require(envelope["receipts"]["VERIFY"]["derived"] == derived, "sealed derived evidence mismatch")
    return derived


def verify_sealed(raw_bytes, envelope_bytes, expected_seal_sha256, allow_simulated=False):
    require(digest(envelope_bytes) == sha256(expected_seal_sha256), "trusted seal digest mismatch")
    envelope = strict_json(envelope_bytes)
    require(canonical_json(envelope) == envelope_bytes, "noncanonical evidence envelope")
    return verify_reports(raw_bytes, envelope, allow_simulated=allow_simulated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "control", "treatment", "enforcement", "expected-seal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    inputs = [args.parent, args.control, args.treatment, args.enforcement]
    # Refuse before reading if output aliases any input. publish_new also uses O_EXCL.
    require(all(Path(args.output).resolve() != Path(p).resolve() for p in inputs), "immutable report output target")
    raw = {k: Path(getattr(args, k)).read_bytes() for k in ("parent", "control", "treatment")}
    result = verify_sealed(raw, Path(args.enforcement).read_bytes(), args.expected_seal_sha256)
    publish_new(args.output, canonical_json(result), forbidden=inputs)


if __name__ == "__main__":
    try:
        main()
    except (InvalidEvidence, KeyError, TypeError, OSError, ValueError) as error:
        raise SystemExit("P1V_VERIFY_FAIL: " + str(error)) from error
