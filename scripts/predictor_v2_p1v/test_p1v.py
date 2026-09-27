"""Portable fixtures only: no root, mounts, devices, model files or runtime.

Run: python3 -B -m unittest scripts.predictor_v2_p1v.test_p1v -v
All expert bytes are generated in memory. No live authorization token exists
in these tests. Temporary files contain only synthetic reports/evidence.
"""

import ast
import copy
import io
import stat
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from . import (ALIGN, AUTHORITY, CANONICAL, CONFIGURED, EXPECTATIONS, EXPERT_SIZE,
               IDS, LAUNCHES, MODEL, SOURCE_FILES, STARTUP_IDS, TOOLS, InvalidEvidence, canonical_json,
               digest, publish_new, strict_json, tool_hashes)
from .content_hasher import (Identity, LinuxDirectReader, alternatives, encode_ledger,
                             hash_content, resolve, verify_content)
from .offline_verify import (ATTEMPT_STRIDE, CAPACITY, EVENTS, LAUNCH_FIELDS,
                             P0_FIELDS, PASS_STRIDE, PROTOCOL, SCHEMA, STRIDE,
                             Trace, classify, opportunities, readiness,
                             semantic, slot, verify_reports, verify_sealed)
from .supervisor import (ALIAS_PARENT, HASH_WINDOW, FakeBackend, FutureBackend,
                         Supervisor, main, parent_argv, plan, validate_enforcement)


class FakeReader:
    def __init__(self):
        self.identities = {}
        for g in STARTUP_IDS + IDS:
            self.identities[alternatives(g)[0]] = Identity(259, 4, 10000 + g, EXPERT_SIZE,
                                                        stat.S_IFREG | 0o400, 1, 100, 100)
        self.trace, self.opened = [], set()
        self.error, self.short, self.direct, self.drift, self.path_drift = None, False, True, False, False
        self.phase = "POST_SHUTDOWN"

    def metadata(self, path):
        if self.error:
            raise self.error
        if path not in self.identities:
            raise FileNotFoundError(path)
        value = self.identities[path]
        return replace(value, inode=value.inode + 1) if self.path_drift and self.trace else value

    def canonical(self, path):
        return CANONICAL + "/" + path.rsplit("/", 1)[1]

    def open_direct(self, path):
        self.opened.add(path)
        return path

    def is_direct(self, handle):
        return self.direct

    def identity(self, handle):
        value = self.identities[handle]
        return replace(value, inode=value.inode + 1) if self.drift and self.trace else value

    def read_direct(self, handle, offset, length):
        if self.phase != "POST_SHUTDOWN":
            raise InvalidEvidence("fake helper payload read before shutdown")
        self.trace.append((self.phase, handle, offset, length))
        # Include different header/padding-sized regions, all generated in memory.
        data = bytes([offset // ALIGN % 251]) * length
        return data[:-1] if self.short else data

    def close(self, handle):
        self.opened.remove(handle)


def pre_metadata(reader):
    return {g: resolve(reader, g) for g in IDS}


def fake_content():
    reader = FakeReader()
    return hash_content(reader, pre_metadata(reader), HASH_WINDOW)


def word(start=10, end=None, *data):
    return [start, start if end is None else end, *data, *([0] * (10 - len(data)))]


def counters():
    return {**dict.fromkeys(LAUNCH_FIELDS, 0), "incomplete": False}


def p0():
    return {**dict.fromkeys(P0_FIELDS, 0), "incomplete": None, "terminal_categories": []}


def candidate(seq, target, local, namespace):
    return {"sequence": seq, "expert": local, "generation": seq, "source_layer": 47,
            "target_layer": 47, "signal_revision": 1, "position_distance": 1,
            "source_position": position_identity(target - 1),
            "target_position": position_identity(target),
            "model": {"num_layers": 48, "num_experts": 128, "top_k": 8},
            "nominal_layer_lead": 48, "source_set": list(range(8)), "score": 1,
            "committed_position_cutoff": target - 1, "table_update_cutoff": seq,
            "request": {"runtime_namespace": namespace, "request_sequence": 1,
                        "phase": "Measured", "phase_run_index": 0},
            "namespace": {"runtime": namespace, "context": namespace + 10,
                          "arena": namespace + 20, "layer": 47, "capacity": 8}}


def position_identity(p):
    return {"absolute_position": p, "position_kind": "Prompt" if p < 16 else "Decode",
            "decode_index": None if p < 16 else p - 16}


def empty_physical(c):
    return {"snapshot": {"namespace": copy.deepcopy(c["namespace"]), "residents": [None] * 8},
            "event_cutoff": 0, "committed_installs": 0, "physical_victims": 0}


def raw_fixture():
    """Complete 143-position recorder fixture, with 13 fresh successful launches.

    These outcomes are deliberately synthetic, not the historical 9/4 result.
    The first launch has a shared paid service; all others have no missing set.
    """
    artifacts = []
    for arm, namespace, generation in (("Control", 10, 100), ("Treatment", 20, 200)):
        records = {}

        def put(p, event, words, attempt=0, probe=0, member=0):
            index = slot(p, event, attempt, probe, member)
            assert index not in records
            records[index] = {"slot": index, "words": words}

        targets = {p: local for _, p, local, _ in LAUNCHES}
        for p in range(143):
            local = targets.get(p, 8)
            ids = [local, 0, 1, 2, 3, 4, 5, 6]
            put(p, "CleanRoute", word(120, 120, 1, 8, *ids))
            put(p, "Boundary", word(115, 115, 1, 8, *ids))
        put(0, "RequestStart", word(0))
        put(0, "RequestEnd", word(200))
        observations = []
        for seq, target, local, _ in LAUNCHES:
            c = candidate(seq, target, local, namespace)
            obs = {"freeze": {"candidate": c, "timestamp_ns": 10,
                               "source": {"logical_generation": generation, "logical_materialized": True,
                                          "ram_resident": True, "permanence": "Unknown"}, "current": False,
                               "incomplete": None, "physical": empty_physical(c)},
                   "deadline": {"timestamp_ns": 100, "current": False,
                                 "request": c["request"], "position": c["target_position"], "incomplete": None,
                                 "physical": empty_physical(c)},
                   "outcome": {"Resolved": {"prediction_hit": True}}, "incomplete": None}
            observations.append(obs)
            put(target, "Freeze", word(10, 10, seq, local, generation, seq))
            put(target, "DP1e", word(100, 100, seq))
            if arm == "Treatment":
                put(target, "Launch", word(11, 12, 6))

                def owned(code, stamp):
                    return word(stamp, stamp + 1, code, seq, local, generation, seq, seq,
                                namespace, 1, namespace + 10, namespace + 20)

                for event, stamp, code in (("Claim", 13, 1), ("WriterStart", 21, 1),
                                            ("Payload", 25, 1), ("Close", 30, 1),
                                            ("LeaseRelease", 33, 1), ("Publication", 105, 0),
                                            ("Credit", 125, 1), ("DeferredRetire", 130, 1)):
                    put(target, event, owned(code, stamp))
                put(target, "Spawn", word(20, 22, 1))
                put(target, "DEntry", word(99))
                put(target, "Binding", owned(1, 110))
            elif seq == 25:
                global_id = 47 * 128 + local
                selected = [global_id, *range(6016, 6023)]
                put(target, "Service", word(100, 110, 1))
                put(target, "Selected", word(100, 100, 1, 8, *selected))
                put(target, "Probe", word(100, 100, 1, 2, global_id, selected[1]))
                put(target, "SourceSet", word(101, 106, 1, 2, global_id, selected[1]))
                for event in ("Stage", "Commit", "Installed"):
                    put(target, event, word(107, 108, 1, global_id, generation))
        request = observations[0]["freeze"]["candidate"]["request"]
        sources = {o["freeze"]["candidate"]["source_position"]["absolute_position"] for o in observations}
        raw = {"request": request, "observations": observations,
               "no_emissions": [{"source": position_identity(p), "target": position_identity(p + 1),
                                 "reason": "NoPositiveHistory"} for p in range(143) if p not in sources],
               "incomplete": None, "opportunities": [[o["freeze"]["candidate"]["sequence"], opportunities(o)] for o in observations],
               "partitions": {"emitted": 13, "resolved": 13, "pending": 0, "censored": 0,
                              "prediction_hits": 13, "route_misses": 0, "valid_f": 13, "current_at_f": 0,
                              "absent_at_f": 13, "useful": 13, "target_confirmed_useful": 13}}
        launch = counters()
        movement = p0()
        if arm == "Treatment":
            for key in ("launch_considered", "source_first_attempt_clean", "writer_spawned"):
                launch[key] = 13
            for key in ("emitted", "terminal_predictions", "reservations", "direct_matching_demand_credits"):
                movement[key] = 13
        resources = {"sidecar_initialized": True, "sidecar_allocated": True, "namespace": {"runtime": namespace},
                     "sidecar_stride_bytes": 2654212, "sidecar_bank": 1, "sidecar_slot": 0,
                     "ordinary_total_expert_budget_bytes": 2147483648, "ordinary_layer_capacities": [8] * 48,
                     "ordinary_layer_resident_counts": [0] * 48, "ordinary_arena_allocation_bytes": 1024,
                     "activity_counters": [0] * 14, "p1e_shadow_present": True}
        before = dict(resources, sidecar_initialized=False, sidecar_allocated=False, namespace=None, p1e_shadow_present=False)
        initial = {"resources": resources, "mode": "InertResourceOnly" if arm == "Control" else "Active",
                   "token_loop": {"count": 0}, "recovery": {"count": 0}, "production_install": {"count": 0},
                   "launch": counters(), "p0": p0(), "committed_position": 0, "pending_sidecar": False,
                   "observation_capacity": 439232}
        token = {"tokens_completed": 143, "fatal_failures": 0, "no_progress_failures": 0, "replay_attempts": 0,
                 "token_attempts": 143, "queue_submissions": 143, "boundary_maps": 143, "boundary_readbacks": 143,
                 "residency_miss_attempts": 0}
        runtime_counters = {}
        for key, delta in (("token_loop", token), ("recovery", {"full_token_replay_attempts": 0, "resume_attempts": 0}),
                           ("routed_execution", {"cpu_routed_expert_dispatches": 0, "gpu_cpu_fallbacks": 0,
                                                  "degraded_expert_substitutions": 0})):
            runtime_counters[key + "_before"] = dict.fromkeys(delta, 0)
            runtime_counters[key + "_after"] = delta.copy()
            runtime_counters[key + "_delta"] = delta.copy()
        tokens = list(range(128))
        report = {"arm": arm, "mode": initial["mode"], "runtime_build_attempted": True,
                  "runtime_constructed": True, "ordinary_invariants_pass": True, "initial_state_pass": True,
                  "sidecar_freshly_initialized": True, "errors": [], "completed_positions": 143,
                  "observation_capacity_per_collection": 439232,
                  "generated_tps": None, "planned_position_tps": None, "request_wall_ns": 200,
                  "runtime_shutdown": {"controlled_shutdown_requested": True, "all_runtime_resources_released": True},
                  "generated_token_ids": tokens,
                  "generated_token_ids_sha256": digest(struct.pack("<Q", 128) + b"".join(struct.pack("<I", t) for t in tokens)),
                  "provenance": {"config_sha256": AUTHORITY["config_sha256"], "model_identity": MODEL,
                                 "provenance": {"executable_sha256": AUTHORITY["binary_sha256"],
                                                "resolved_config_sha256": "a" * 64,
                                                "build": {"git_sha": AUTHORITY["commit"], "dirty": False}}},
                  "runtime_resolved_config_sha256": "a" * 64,
                  "adapter": {"name": "NVIDIA L4", "driver_info": "580.178.04", "software_adapter": False},
                  "resource_identity": {"fixture": True}, "runtime_contract": {"fixture": True}, "model_load": {"fixture": True},
                  "initial": initial, "resources_before_opt_in": before,
                  "production_install_after": {"count": 1}, "production_install_delta": {"count": 1},
                  "runtime_counters": runtime_counters, "p1j_launch_before": counters(),
                  "p1j_launch_after": launch.copy(), "p1j_launch_delta": launch.copy(),
                  "predictor_v2_snapshot": movement, "raw_p1e_report": raw,
                  "diagnostics": {"invalid": False, "capacity": CAPACITY, "storage_bytes": CAPACITY * 104,
                                  "layout": {"words": 12, "position_stride": STRIDE, "position_slots": 24,
                                             "attempt_stride": ATTEMPT_STRIDE, "pass_stride": PASS_STRIDE,
                                             "member_stride": 6, "timestamp_words": [0, 1],
                                             "event_offsets": [[k, v] for k, v in EVENTS.items()],
                                             "clock": "checked u64 ns from the same request-local P1E monotonic origin"},
                                  "records": list(records.values())}}
        artifacts.append({"schema": SCHEMA, "protocol": PROTOCOL, "diagnostic_id": "p1v-fixture", "pair_index": 1,
                          "arm": arm, "pid": namespace, "binary_sha256": AUTHORITY["binary_sha256"],
                          "request_sha256": AUTHORITY["request_sha256"],
                          "source_identity": {**{k: AUTHORITY[k] for k in ("commit", "tree", "parent")},
                                              "files": SOURCE_FILES}, "report": report})
    control, treatment = artifacts
    rows = []
    from .offline_verify import service
    cs, ts = (Trace(a["report"]["diagnostics"]) for a in artifacts)
    for cobs, tobs in zip(control["report"]["raw_p1e_report"]["observations"], treatment["report"]["raw_p1e_report"]["observations"]):
        c = tobs["freeze"]["candidate"]
        statuses = ["GENERATION_DIVERGENT"] + (["ATTRIBUTION_UNRESOLVED"] if c["sequence"] == 25 else [])
        rows.append({"semantic_candidate": semantic(c), "control_raw_candidate": cobs["freeze"]["candidate"],
                     "treatment_raw_candidate": c, "control_logical_generation": 100, "treatment_logical_generation": 200,
                     "class": "GENERATION_DIVERGENT", "statuses": statuses, "readiness": "BEFORE_D", "consumed": True,
                     "published": True, "criticality_scope": "FOREGROUND_DEMAND_SET", "exact_candidate_saved_ns": None,
                     "candidate_marginal_attribution": "UNRESOLVED_GENERATION",
                     "control_service": service(cs, c["target_position"]["absolute_position"], c["expert"]),
                     "treatment_service": service(ts, c["target_position"]["absolute_position"], c["expert"])})
    parent = {k: copy.deepcopy(control[k]) for k in ("schema", "protocol", "diagnostic_id", "pair_index",
                                                     "binary_sha256", "request_sha256", "source_identity")}
    parent.update(diagnostic_evidence_valid=True, errors=[], diagnostic_only=True,
                  execution_order=["Control", "Treatment"], profile=AUTHORITY["profile"],
                  performance_comparison_authorized=False, performance_verdict="NOT_AUTHORIZED",
                  exact_candidate_saved_ns=None, arms=artifacts, attribution=rows,
                  processes=[{"arm": a["arm"], "pid": a["pid"], "exit_code": 0, "error": None,
                              "artifact_sha256": digest(canonical_json(a))} for a in artifacts])
    return parent, control, treatment


def encode_raw(fixture):
    parent, control, treatment = copy.deepcopy(fixture)
    parent["arms"] = [control, treatment]
    for process, arm in zip(parent["processes"], (control, treatment)):
        process["artifact_sha256"] = digest(canonical_json(arm))
    return {k: canonical_json(v) for k, v in zip(("parent", "control", "treatment"), (parent, control, treatment))}


def receipts(raw, content):
    protection = {"ok": True, "expectations": EXPECTATIONS, "filesystem_ro": True, "device_ro": True,
                  "alias_locked": True, "exclusion_held": True, "namespace_pinned": True,
                  "healthy": True, "no_uncontrolled_aliases": True, "namespace_id": "synthetic-ns",
                  "boot_id": "synthetic-boot", "disk_sequence": 1, "mount_id": 100}
    reader = FakeReader()
    metadata = [{"global_expert_id": g, "selected": alternatives(g)[0],
                 "canonical": reader.canonical(alternatives(g)[0]), "identity": asdict(reader.identities[alternatives(g)[0]]),
                 "dio_eligible": True, "o_direct_retained": True, "payload_bytes_read": 0,
                 "redirects": [], "disallowed_features": []} for g in STARTUP_IDS + IDS]
    return {
        "AUTHORITY": {"ok": True, "authority": AUTHORITY, "tools": tool_hashes(), "inputs_verified": True,
                      "outputs_absent": True, "marker_absent": True, "frozen_environment": True, "frozen_limits": True,
                      "argv": list(parent_argv("p1v-fixture"))},
        "EXCLUSION": {"ok": True, "held": True, "trusted_host": True, "uncontrolled_writers": [], "raw_holders": [], "uncontrolled_mount_namespaces": []},
        "TOPOLOGY": {"ok": True, "expectations": EXPECTATIONS, "filesystem_rw": True, "device_rw": True,
                     "healthy": True, "external_journal": False, "stacked_or_overlapping_devices": [],
                     "uncontrolled_aliases": [], "writes_on_data_backing": [], "boot_id": "synthetic-boot",
                     "disk_sequence": 1, "mount_id": 100, "original_mount_options": ["rw", "relatime"], "output_major_minor": "259:1"},
        "NAMESPACE": {"ok": True, "private_recursive": True, "initial_user_namespace": True,
                      "references_pinned": True, "namespace_id": "synthetic-ns"},
        "ALIAS_BIND": {"ok": True, "path": ALIAS_PARENT, "private_bind": True},
        "ALIAS_RO": {"ok": True, "alias_parent_ro": True, "configured": CONFIGURED,
                     "symlink_target": CANONICAL, "runtime_cannot_replace": True},
        "METADATA": {"ok": True, "records": metadata},
        "FILESYSTEM_RO": {"ok": True, "superblock_ro": True, "normal_unforced_transition": True,
                          "journal_writeback_complete": True, "device_still_rw": True, "healthy": True},
        "DEVICE_RO": {"ok": True, "device_ro": True, "filesystem_still_ro": True},
        "PROTECTED": protection.copy(),
        "CONFINEMENT": {"ok": True, "uids": [1000] * 3, "gids": [1000] * 3,
                        "capabilities": dict.fromkeys(("effective", "permitted", "inheritable", "ambient", "bounding"), 0),
                        **dict.fromkeys(("no_new_privs", "groups_reviewed", "brokers_excluded", "setup_fds_closed",
                                         "seccomp_enforced", "cgroup_nondelegated", "namespace_apis_denied", "raw_device_access_denied"), True),
                        "policy_sha256": "b" * 64, "inherited_fds": [0, 1, 2], "cgroup_id": "synthetic-runtime"},
        "MARKER": {"ok": True, "created_exclusive": True, "durable": True, "attempt": "p1v-fixture"},
        "P1R_PARENT": {"ok": True, "invocations": 1, "argv": list(parent_argv("p1v-fixture")),
                       "external_actions_between_arms": [], "exit_code": 0, "report_origin": "created_by_this_invocation",
                       "attempt": "p1v-fixture", "raw_report_hashes": {k: digest(v) for k, v in raw.items()}},
        "SHUTDOWN": {"ok": True, "parent_reaped": True, "runtime_tree_empty": True,
                     "uncertain_descendants": [], "cgroup_id": "synthetic-runtime"},
        "REVALIDATE": protection.copy(),
        "HASH": {"ok": True, "window": HASH_WINDOW, "hasher_stopped": True, "content": content},
        "VERIFY": {},
        "SEAL": {"ok": True, "durable": True, "exclusive": True, "runtime_cannot_write": True},
        "DEVICE_RW": {"ok": True, "device_rw": True, "filesystem_still_ro": True,
                      "runtime_tree_empty": True, "hasher_stopped": True},
        "FILESYSTEM_RW": {"ok": True, "filesystem_rw": True, "device_rw": True, "healthy": True, "mount_options": ["rw", "relatime"]},
        "RESTORATION_SEAL": {"ok": True, "durable": True, "exclusive": True},
        "RELEASE": {"ok": True, "released": True},
    }


class PipelineFake(FakeBackend):
    def __init__(self, raw, content, **kwargs):
        super().__init__(receipts(raw, content), **kwargs)
        self.raw = raw

    def perform(self, action, evidence):
        if action.name == "VERIFY":
            self.receipts["VERIFY"] = {"ok": True, "raw_report_hashes": {k: digest(v) for k, v in self.raw.items()},
                                       "derived": verify_reports(self.raw, evidence, allow_simulated=True, sealed=False)}
        if action.name == "RESTORATION_SEAL":
            self.receipts["RESTORATION_SEAL"]["evidence_seal_sha256"] = evidence["receipts"]["SEAL"]["sha256"]
        return super().perform(action, evidence)


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = fake_content()
        cls.fixture = raw_fixture()
        cls.raw = encode_raw(cls.fixture)

    def backend(self, **kwargs):
        return PipelineFake(self.raw, self.content, **kwargs)

    def run_backend(self, backend):
        return Supervisor(backend, "p1v-fixture").run()

    def seal(self):
        backend = self.backend()
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        return backend.sealed_bytes


class SupervisorTests(FixtureTests):
    def test_exact_happy_order_and_seal(self):
        backend = self.backend()
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        expected = [a.name for a in plan()]
        i = expected.index("P1R_PARENT")
        expected[i + 1:i + 1] = ["CONTROL", "TREATMENT"]
        self.assertEqual(backend.trace, expected)
        self.assertLess(expected.index("ALIAS_RO"), expected.index("MARKER"))
        self.assertLess(expected.index("FILESYSTEM_RO"), expected.index("DEVICE_RO"))
        self.assertLess(expected.index("SHUTDOWN"), expected.index("HASH"))
        self.assertLess(expected.index("HASH"), expected.index("SEAL"))
        self.assertLess(expected.index("SEAL"), expected.index("DEVICE_RW"))
        self.assertLess(expected.index("DEVICE_RW"), expected.index("FILESYSTEM_RW"))
        self.assertNotIn("DEVICE_RW", strict_json(backend.sealed_bytes)["receipts"])

    def test_every_action_failure_stops_without_retry(self):
        for action in plan():
            with self.subTest(action=action.name):
                backend = self.backend(fail_at=action.name)
                result = self.run_backend(backend)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(backend.trace[-1], action.name)
                self.assertEqual(backend.trace.count(action.name), 1)
                self.assertEqual(backend.journal[-1]["status"], "FAIL")
                if action.name in [a.name for a in plan()[:18]]:
                    self.assertNotIn("DEVICE_RW", backend.trace)

    def test_filesystem_failure_before_device(self):
        backend = self.backend(fail_at="FILESYSTEM_RO")
        self.run_backend(backend)
        self.assertNotIn("DEVICE_RO", backend.trace)
        self.assertNotIn("MARKER", backend.trace)

    def test_device_and_alias_failure_before_marker(self):
        for action in ("DEVICE_RO", "ALIAS_BIND", "ALIAS_RO"):
            backend = self.backend(fail_at=action)
            self.run_backend(backend)
            self.assertNotIn("MARKER", backend.trace)
            self.assertNotIn("CONTROL", backend.trace)

    def test_marker_collision_and_one_shot(self):
        backend = self.backend()
        backend.marker = True
        supervisor = Supervisor(backend, "p1v-fixture")
        self.assertEqual(supervisor.run()["status"], "FAIL")
        self.assertNotIn("CONTROL", backend.trace)
        with self.assertRaises(InvalidEvidence):
            supervisor.run()

    def test_control_failure_prevents_treatment(self):
        backend = self.backend(control_ok=False)
        self.assertEqual(self.run_backend(backend)["status"], "FAIL")
        self.assertIn("CONTROL", backend.trace)
        for forbidden in ("TREATMENT", "HASH", "DEVICE_RW"):
            self.assertNotIn(forbidden, backend.trace)

    def test_uncertain_descendants_block_hash_and_restore(self):
        backend = self.backend()
        backend.receipts["SHUTDOWN"]["runtime_tree_empty"] = False
        self.assertEqual(self.run_backend(backend)["status"], "FAIL")
        self.assertNotIn("HASH", backend.trace)
        self.assertNotIn("DEVICE_RW", backend.trace)

    def test_receipt_drift_fail_closed(self):
        cases = [("ALIAS_RO", "symlink_target", "/elsewhere"), ("FILESYSTEM_RO", "superblock_ro", False),
                 ("DEVICE_RO", "filesystem_still_ro", False), ("PROTECTED", "exclusion_held", False),
                 ("CONFINEMENT", "uids", [0, 0, 0]), ("REVALIDATE", "namespace_id", "other"),
                 ("HASH", "hasher_stopped", False), ("SEAL", "durable", False),
                 ("DEVICE_RW", "filesystem_still_ro", False)]
        for action, key, value in cases:
            with self.subTest(action=action, key=key):
                backend = self.backend()
                backend.receipts[action][key] = value
                self.assertEqual(self.run_backend(backend)["status"], "FAIL")
                self.assertEqual(backend.trace[-1], action)

    def test_plan_is_side_effect_free_and_exact_argv(self):
        output = io.StringIO()
        with patch("os.open", side_effect=AssertionError("plan opened file")), redirect_stdout(output):
            main(["--plan"])
        data = strict_json(output.getvalue())
        self.assertEqual(data["mode"], "PLAN_ONLY")
        self.assertEqual(data["actions"][12]["argv"][1], "diagnose-gpu-native-predictor-v2-critical-path-attribution")
        self.assertEqual(data["post_shutdown_hash_ids"], list(IDS))

    def test_no_live_authority_in_portable_suite(self):
        with self.assertRaises(InvalidEvidence):
            FutureBackend(None, lambda *_: None, lambda *_: None)
        with self.assertRaises(InvalidEvidence):
            LinuxDirectReader(None, HASH_WINDOW)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--execute"])
        backend = self.backend()
        backend.simulated = False
        with self.assertRaises(InvalidEvidence):
            Supervisor(backend, "p1v-fixture")

    def test_no_payload_helpers_in_pre_or_pair(self):
        backend = self.backend()
        self.run_backend(backend)
        i = backend.trace.index("CONTROL")
        self.assertEqual(backend.trace[i:i + 2], ["CONTROL", "TREATMENT"])
        self.assertEqual([a.name for a in plan() if a.name == "HASH"], ["HASH"])
        for phase in ("PRE", "CONTROL", "TREATMENT"):
            reader = FakeReader()
            reader.phase = phase
            with self.assertRaises(InvalidEvidence):
                reader.read_direct("synthetic", 0, ALIGN)
            self.assertEqual(reader.trace, [])


class HasherTests(unittest.TestCase):
    def test_exact_full_bytes_six_ids_and_root_encoding(self):
        reader = FakeReader()
        content = hash_content(reader, pre_metadata(reader), HASH_WINDOW)
        self.assertEqual([r["global_expert_id"] for r in content["records"]], list(IDS))
        self.assertEqual(len(reader.trace), 6 * (EXPERT_SIZE // ALIGN))
        expected = bytearray(b"MER-P1V-CONTENT-v1\0\x06\0\0\0")
        for record in content["records"]:
            body = struct.pack("<I", record["global_expert_id"])
            for key in ("authoritative_selected_path", "canonical_path"):
                value = record[key].encode()
                body += struct.pack("<I", len(value)) + value
            body += struct.pack("<IIQQ", 259, 4, record["inode"], EXPERT_SIZE)
            body += bytes.fromhex(record["sha256"])
            expected += struct.pack("<I", len(body)) + body
        self.assertEqual(bytes.fromhex(content["ledger_hex"]), bytes(expected))
        self.assertEqual(content["content_root"], digest(bytes(expected)))
        self.assertEqual(content["content_root"], "5ac121bb9062dcf132ba96498bdb796e838c87f88a2e11e14bb50a4ae6345a52")
        payload = b"".join(bytes([i % 251]) * ALIGN for i in range(EXPERT_SIZE // ALIGN))
        self.assertEqual(content["records"][0]["sha256"], digest(payload))
        self.assertEqual(reader.opened, set())

    def test_short_direct_missing_flag_identity_drift(self):
        for flag in ("short", "direct", "drift", "path_drift"):
            reader = FakeReader()
            before = pre_metadata(reader)
            setattr(reader, flag, flag != "direct")
            with self.subTest(flag=flag), self.assertRaises(InvalidEvidence):
                hash_content(reader, before, HASH_WINDOW)
            self.assertEqual(reader.opened, set())

    def test_direct_error_no_retry(self):
        reader = FakeReader()
        before = pre_metadata(reader)
        with patch.object(reader, "read_direct", side_effect=OSError("direct EINVAL")) as read:
            with self.assertRaises(OSError):
                hash_content(reader, before, HASH_WINDOW)
            self.assertEqual(read.call_count, 1)
        self.assertEqual(reader.opened, set())

    def test_symlink_hardlink_device_size_and_fallback_errors(self):
        for field, value in (("mode", stat.S_IFLNK | 0o777), ("links", 2), ("device_minor", 1), ("file_length", EXPERT_SIZE - 1)):
            reader = FakeReader()
            primary = alternatives(IDS[0])[0]
            reader.identities[primary] = replace(reader.identities[primary], **{field: value})
            with self.subTest(field=field), self.assertRaises(InvalidEvidence):
                resolve(reader, IDS[0])
        for error in (PermissionError("denied"), OSError("I/O failure")):
            reader = FakeReader()
            reader.error = error
            with self.assertRaises(OSError):
                resolve(reader, IDS[0])
        reader = FakeReader()
        primary, fallback = alternatives(IDS[0])
        reader.identities[fallback] = reader.identities.pop(primary)
        self.assertEqual(resolve(reader, IDS[0])[0], fallback)
        reader.identities[primary] = reader.identities[fallback]
        self.assertEqual(resolve(reader, IDS[0])[0], primary)

    def test_hash_window_and_coverage_failures(self):
        for key in HASH_WINDOW:
            reader = FakeReader()
            bad = dict(HASH_WINDOW, **{key: False})
            with self.assertRaises(InvalidEvidence):
                hash_content(reader, pre_metadata(reader), bad)
            self.assertEqual(reader.trace, [])
        reader = FakeReader()
        before = pre_metadata(reader)
        before.pop(IDS[-1])
        with self.assertRaises(InvalidEvidence):
            hash_content(reader, before, HASH_WINDOW)

    def test_ledger_order_extra_missing_corruption(self):
        content = fake_content()
        for records in (content["records"][::-1], content["records"][:-1], content["records"] + content["records"][:1]):
            with self.assertRaises(InvalidEvidence):
                encode_ledger(records)
        for key, value in (("content_root", "f" * 64), ("ledger_hex", "00"), ("buffered_fallback", True)):
            with self.assertRaises(InvalidEvidence):
                verify_content(dict(content, **{key: value}))


class VerifierTests(FixtureTests):
    def test_valid_exact_launch_gate_raw_divergence_separate_derived(self):
        seal = self.seal()
        before = copy.deepcopy(self.raw)
        result = verify_sealed(self.raw, seal, digest(seal), allow_simulated=True)
        self.assertTrue(result["semantic_payload_equivalent"])
        self.assertEqual(result["rows"][0]["derived_class"], "READY_AND_CRITICAL")
        self.assertEqual(result["rows"][1]["derived_class"], "READY_BUT_NOT_CRITICAL")
        for row in result["rows"]:
            self.assertEqual(row["raw_class"], "GENERATION_DIVERGENT")
            self.assertIn("GENERATION_DIVERGENT", row["raw_statuses"])
            self.assertEqual((row["control_logical_generation"], row["treatment_logical_generation"]), (100, 200))
            self.assertIsNone(row["exact_candidate_saved_ns"])
        self.assertEqual(self.raw, before)

    def test_simulation_never_qualifies_live(self):
        seal = self.seal()
        with self.assertRaises(InvalidEvidence):
            verify_sealed(self.raw, seal, digest(seal))

    def test_generation_unavailable_preserved_end_to_end(self):
        fixture = copy.deepcopy(self.fixture)
        report = fixture[1]["report"]
        report["raw_p1e_report"]["observations"][1]["freeze"]["source"]["logical_generation"] = None
        freeze = next(r for r in report["diagnostics"]["records"] if r["slot"] == slot(46, "Freeze"))
        freeze["words"][4] = 0
        row = fixture[0]["attribution"][1]
        row.update(control_logical_generation=None, statuses=["GENERATION_UNAVAILABLE"])
        row["class"] = "GENERATION_UNAVAILABLE"
        raw = encode_raw(fixture)
        backend = PipelineFake(raw, self.content)
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        derived = verify_sealed(raw, backend.sealed_bytes, digest(backend.sealed_bytes), allow_simulated=True)
        row = derived["rows"][1]
        self.assertEqual(row["raw_statuses"], ["GENERATION_UNAVAILABLE"])
        self.assertIsNone(row["control_logical_generation"])
        self.assertTrue(row["semantic_payload_equivalent"])
        self.assertEqual(row["derived_class"], "READY_BUT_NOT_CRITICAL")

    def test_source_model_capacity_and_physical_evidence_are_required(self):
        for kind in ("source", "build", "model", "capacity", "initial-p0", "physical", "candidate-field"):
            fixture = copy.deepcopy(self.fixture)
            artifact = fixture[2]
            report = artifact["report"]
            if kind == "source":
                artifact["source_identity"]["files"]["engine.rs"] = "0" * 64
            elif kind == "build":
                report["provenance"]["provenance"]["build"]["dirty"] = True
            elif kind == "model":
                report["provenance"]["model_identity"]["num_experts_per_layer"] = 127
            elif kind == "capacity":
                report["initial"]["observation_capacity"] -= 1
            elif kind == "initial-p0":
                report["initial"]["p0"]["emitted"] = 1
            elif kind == "physical":
                report["raw_p1e_report"]["observations"][0]["freeze"]["physical"] = None
            else:
                del report["raw_p1e_report"]["observations"][0]["freeze"]["candidate"]["table_update_cutoff"]
            backend = PipelineFake(encode_raw(fixture), self.content)
            with self.subTest(kind=kind):
                self.assertEqual(self.run_backend(backend)["status"], "FAIL")

    def test_trusted_seal_and_derived_tampering_rejected(self):
        seal = self.seal()
        with self.assertRaises(InvalidEvidence):
            verify_sealed(self.raw, seal, "0" * 64, allow_simulated=True)
        envelope = strict_json(seal)
        envelope["receipts"]["VERIFY"]["derived"]["rows"][0]["derived_class"] = "LATE"
        receipt = envelope["receipts"]["VERIFY"]
        envelope["events"][-1]["receipt_sha256"] = digest(canonical_json(receipt))
        # Even when an operator supplies the modified envelope digest, the raw
        # classifier replay detects the unjustified derived result.
        altered = canonical_json(envelope)
        with self.assertRaises(InvalidEvidence):
            verify_sealed(self.raw, altered, digest(altered), allow_simulated=True)

    def test_mutated_original_bytes_rejected(self):
        seal = self.seal()
        for key in self.raw:
            raw = dict(self.raw, **{key: self.raw[key] + b" "})
            with self.assertRaises(InvalidEvidence):
                verify_sealed(raw, seal, digest(seal), allow_simulated=True)

    def test_continuity_missing_changed_or_endpoint_only_rejected(self):
        envelope = strict_json(self.seal())
        for mutation in (lambda e: e["events"].pop(5),
                         lambda e: e["receipts"]["ALIAS_RO"].update(alias_parent_ro=False),
                         lambda e: e.update(events=[]),
                         lambda e: e["receipts"]["PROTECTED"].update(exclusion_held=False)):
            bad = copy.deepcopy(envelope)
            mutation(bad)
            with self.assertRaises(InvalidEvidence):
                validate_enforcement(bad, allow_simulated=True)

    def test_launch_missing_extra_duplicate_or_wrong_tuple(self):
        for kind in ("missing", "extra", "duplicate", "wrong"):
            fixture = copy.deepcopy(self.fixture)
            report = fixture[2]["report"]
            records = report["diagnostics"]["records"]
            found = next(r for r in records if r["slot"] == slot(29, "Launch"))
            if kind == "missing":
                records.remove(found)
            elif kind == "extra":
                records.append({"slot": slot(140, "Launch"), "words": word(0, 0, 6)})
            elif kind == "duplicate":
                records.append(copy.deepcopy(found))
            else:
                report["raw_p1e_report"]["observations"][0]["freeze"]["candidate"]["expert"] = 12
            backend = PipelineFake(encode_raw(fixture), self.content)
            with self.subTest(kind=kind):
                self.assertEqual(self.run_backend(backend)["status"], "FAIL")
                self.assertNotIn("DEVICE_RW", backend.trace)

    def test_stale_generation_binding_and_accounting_fail(self):
        for kind in ("generation", "epoch", "binding", "counter", "shutdown", "raw-row"):
            fixture = copy.deepcopy(self.fixture)
            report = fixture[2]["report"]
            records = report["diagnostics"]["records"]
            if kind in ("generation", "epoch", "binding"):
                event = "Binding" if kind == "binding" else "Payload"
                record = next(r for r in records if r["slot"] == slot(29, event))
                record["words"][6 if kind == "epoch" else 5] += 1
            elif kind == "counter":
                report["predictor_v2_snapshot"]["direct_matching_demand_credits"] = 12
            elif kind == "shutdown":
                report["runtime_shutdown"]["all_runtime_resources_released"] = False
            else:
                fixture[0]["attribution"][0]["class"] = "READY_AND_CRITICAL"
            backend = PipelineFake(encode_raw(fixture), self.content)
            with self.subTest(kind=kind):
                self.assertEqual(self.run_backend(backend)["status"], "FAIL")

    def test_no_performance_or_historical_reclassification(self):
        seal = self.seal()
        result = verify_sealed(self.raw, seal, digest(seal), allow_simulated=True)
        self.assertEqual(result["performance_verdict"], "NOT_AUTHORIZED")
        self.assertFalse(result["historical_reclassification"])
        self.assertNotIn("tps", canonical_json(result).decode())
        fixture = copy.deepcopy(self.fixture)
        for report in fixture:
            report["diagnostic_id"] = "predictor-v2-p1r-historical"
        backend = PipelineFake(encode_raw(fixture), self.content)
        self.assertEqual(self.run_backend(backend)["status"], "FAIL")

    def test_classifier_frozen_order_all_branches(self):
        base = [True, {"Resolved": {"prediction_hit": True}}, False, True, True, True,
                "BEFORE_D", 0, True, [{"candidate_physically_missing": True, "installed": True,
                                      "foreground_set_ns": 2, "source_set_ns": 1}]]
        cases = [({0: False}, "NOT_LAUNCHED_OR_INELIGIBLE"), ({1: "Censored"}, "UNUSED"),
                 ({1: {"Resolved": {"prediction_hit": False}}}, "UNUSED"),
                 ({2: True, 3: False}, "ORDINARY_WON_RACE"),
                 ({6: "D_WINDOW_AMBIGUOUS"}, "D_WINDOW_AMBIGUOUS"),
                 ({6: "READY_DURING_D_WINDOW"}, "READY_DURING_D_WINDOW"),
                 ({6: "AFTER_D", 7: 2}, "LATE"), ({7: 1}, "PUBLICATION_BUSY"),
                 ({}, "READY_AND_CRITICAL"), ({9: []}, "READY_BUT_NOT_CRITICAL"),
                 ({3: False}, "ATTRIBUTION_UNRESOLVED")]
        for changes, expected in cases:
            args = base.copy()
            for index, value in changes.items():
                args[index] = value
            with self.subTest(expected=expected):
                self.assertEqual(classify(*args, payload_equivalent=True), expected)
        self.assertEqual(classify(*base), "GENERATION_DIVERGENT")
        base[4:6] = [False, False]
        self.assertEqual(classify(*base), "GENERATION_UNAVAILABLE")
        self.assertEqual(classify(*base, payload_equivalent=True), "READY_AND_CRITICAL")

    def test_readiness_boundaries(self):
        for close, d, consumed, expected in ((word(1, 2, 1), 3, True, "BEFORE_D"),
                                             (word(3, 4, 1), 3, True, "D_WINDOW_AMBIGUOUS"),
                                             (word(4, 5, 1), 3, True, "READY_DURING_D_WINDOW"),
                                             (word(4, 5, 1), 3, False, "AFTER_D"),
                                             (None, 3, True, "UNAVAILABLE"),
                                             (word(4, 5, 2), 3, True, "AFTER_D")):
            self.assertEqual(readiness(close, d, consumed), expected)


class PublicationTests(unittest.TestCase):
    def test_no_overwrite_or_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "historical-p1r.json"
            original.write_bytes(b"immutable historical fixture")
            original_bytes = original.read_bytes()
            with self.assertRaises(InvalidEvidence):
                publish_new(original, b"changed", forbidden=[original])
            with self.assertRaises(FileExistsError):
                publish_new(original, b"changed")
            alias = Path(directory) / "alias"
            alias.symlink_to(original)
            with self.assertRaises(InvalidEvidence):
                publish_new(alias, b"changed")
            self.assertEqual(original.read_bytes(), original_bytes)
            fresh = Path(directory) / "derived.json"
            self.assertEqual(publish_new(fresh, b"derived\n", forbidden=[original]), digest(b"derived\n"))
            self.assertEqual(fresh.read_bytes(), b"derived\n")

    def test_strict_json_and_numeric_lexemes(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(InvalidEvidence):
                strict_json(data)
        a = strict_json(b'{"a":127.99906291886037}')
        b = strict_json(b'{"a":127.99906291886038}')
        self.assertNotEqual(a, b)

    def test_no_shell_or_subprocess_execution_in_package(self):
        # Static safety assertion: the software-only package has no hidden
        # executor bypass. ctypes is limited to the future-gated direct reader.
        root = Path(__file__).parent
        for name in TOOLS:
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ("system", "popen", "Popen", "execv", "execve"))


if __name__ == "__main__":
    unittest.main()
