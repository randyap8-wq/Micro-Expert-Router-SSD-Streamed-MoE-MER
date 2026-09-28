"""Portable P1V4 validation: synthetic interfaces and temporary evidence only.

No valid adapter execution grant is created anywhere in this suite. Native
entry points use rejection tests or fake receivers with fully mocked syscalls.
"""

import ast
import copy
import importlib
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from . import (AUTHORITY, EXPECTATIONS, FutureAuthorization, InvalidEvidence,
               canonical_json, digest, strict_json)
from . import linux_backend as lb
from .supervisor import (FutureBackend, Supervisor, parent_argv, plan,
                         validate_enforcement)
from .test_p1v import (FakeReader, PipelineFake, adapter_fixture, closure_fixture,
                      confinement_fixture, encode_raw, fake_content,
                      launch_fixture, raw_fixture, receipts)


class FixtureSystem:
    simulated = True

    def __init__(self, raw, content):
        self.raw = copy.deepcopy(raw)
        self.receipts = receipts(raw, content)
        self.calls = []
        self.ro = 0
        self.fail = None
        self.fail_count = 0
        self.running = False
        self.started, self.finish = threading.Event(), threading.Event()
        self.finish.set()
        self.backing = {"device_fd": 40, "filesystem_fd": 41,
                        "identity": {"path": lb.DEVICE, "major_minor": "259:4", "uuid": EXPECTATIONS["data_uuid"],
                                     "disk_sequence": 1, "sysfs_identity": "synthetic-nvme"},
                        "host_mount_id": 100, "private_mount_id": 200,
                        "mount_flags": 1024, "mount_data": "synthetic-original-ext4-options", "alias_flags": 1024}
        self.observations = {
            "authority": {**self.receipts["AUTHORITY"], "launch_observed": launch_fixture()},
            "exclusion": self.receipts["EXCLUSION"], "topology": self.receipts["TOPOLOGY"],
            "roles": {"guardian_original_namespace": True, "worker_single_threaded": True,
                      "initial_user_namespace": True, "worker_privileged": True},
            "namespace": {**self.receipts["NAMESPACE"], "original_namespace_id": "synthetic-original-ns",
                          "original_mount_id": 100, "private_mount_id": 200},
            "alias_bind": self.receipts["ALIAS_BIND"],
            "alias_ro": {**self.receipts["ALIAS_RO"], "preserved_flags": 1024,
                         "canonical_target": lb.CANONICAL, "output_still_rw": True},
            "metadata": self.receipts["METADATA"],
            "sync_health": {"syncfs_ok": True, "journal_clean": True, "healthy": True},
            "filesystem_ro": {**self.receipts["FILESYSTEM_RO"], "mount_ro": True},
            "device_ro": self.receipts["DEVICE_RO"], "protected": self.receipts["PROTECTED"],
            "cgroup_preflight": {"baseline": launch_fixture()["resource_domain"],
                                 "prospective": launch_fixture()["resource_domain"],
                                 "nondelegated": True, "controllers_to_enable": [], "settings_to_write": [], "bpf_to_attach": []},
            "confinement": self.receipts["CONFINEMENT"],
            "shutdown": {**self.receipts["SHUTDOWN"], "recursive_populated": 0, "unexpected_child_groups": []},
            "device_rw": self.receipts["DEVICE_RW"], "filesystem_rw": self.receipts["FILESYSTEM_RW"],
            "release": self.receipts["RELEASE"],
        }
        for key in ("filesystem_ro", "device_ro", "protected", "device_rw", "filesystem_rw"):
            self.observations[key].update(device_identity=self.backing["identity"].copy(), private_mount_id=200,
                                          healthy=True, journal_clean=True)

    def call(self, name, *args):
        if self.running:
            raise AssertionError("system operation during parent: " + name)
        self.calls.append((name, *args))
        if name == self.fail:
            self.fail_count += 1
            raise OSError("synthetic uncertain failure: " + name)

    def observe(self, topic):
        self.call("observe:" + topic)
        return copy.deepcopy(self.observations[topic])

    def acquire_exclusion(self):
        self.call("exclusion")

    def retain_backing(self):
        self.call("retain_backing")
        return copy.deepcopy(self.backing)

    def unshare(self, flags):
        self.call("unshare", flags)

    def mount(self, source, target, fstype, flags, data):
        self.call("mount", source, target, fstype, flags, data)

    def syncfs(self, fd):
        self.call("syncfs", fd)

    def block_ro(self, fd, read_only):
        self.call("block_ro", fd, read_only)
        self.ro = int(read_only)

    def block_get(self, fd):
        self.call("block_get", fd)
        return self.ro

    def create_accounting_cgroup(self, attempt):
        self.call("cgroup_create", attempt)
        return "synthetic-runtime"

    def prepare_stopped_child(self, argv, manifest, cgroup):
        self.call("prepare_child", argv, manifest, cgroup)
        return 1234

    def resume_wait_parent(self, pid):
        self.call("parent_once", pid)
        self.running = True
        self.started.set()
        if not self.finish.wait(10):
            raise AssertionError("synthetic parent not released by test")
        self.running = False
        return 0

    def raw_reports(self, attempt):
        self.call("raw_reports", attempt)
        return copy.deepcopy(self.raw)

    def release(self):
        self.call("release")


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = fake_content()
        cls.raw = encode_raw(raw_fixture())

    def make(self, system=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        journal = lb.DurableJournal(directory.name, "p1v-fixture", fixture=directory)
        system = system or FixtureSystem(self.raw, self.content)
        created = []

        def reader():
            created.append("HASH")
            self.assertEqual(backend.done[-3:], ["P1R_PARENT", "SHUTDOWN", "REVALIDATE"])
            self.assertFalse(system.running)
            return FakeReader()

        backend = lb.LinuxBackend(system, journal, launch_fixture(), adapter_fixture(), closure_fixture(),
                                  host_id="synthetic-host", attempt="p1v-fixture", fixture_reader_factory=reader)
        backend.created_readers = created
        return backend

    def run_backend(self, backend):
        return Supervisor(backend, "p1v-fixture").run()

    def test_full_adapter_seal_and_unchanged_classifier(self):
        backend = self.make()
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        self.assertEqual(backend.done, [a.name for a in plan()])
        self.assertEqual(backend.created_readers, ["HASH"])
        seal = (backend.journal.root / "enforcement.json").read_bytes()
        from .offline_verify import verify_sealed
        derived = verify_sealed(self.raw, seal, digest(seal), allow_simulated=True)
        self.assertTrue(derived["semantic_payload_equivalent"])
        self.assertIsNone(derived["exact_candidate_saved_ns"])
        self.assertEqual(derived["performance_verdict"], "NOT_AUTHORIZED")
        self.assertFalse(derived["historical_reclassification"])
        self.assertEqual(derived["rows"][0]["raw_class"], "GENERATION_DIVERGENT")
        restoration = strict_json((backend.journal.root / "restoration.json").read_bytes())
        self.assertEqual(restoration["evidence_seal_sha256"], digest(seal))
        self.assertNotIn("DEVICE_RW", strict_json(seal)["receipts"])
        self.assertEqual((backend.journal.root / "enforcement.json").read_bytes(), seal)

    def test_no_operation_or_journal_during_blocked_parent(self):
        backend = self.make()
        backend.system.finish.clear()
        result = []
        thread = threading.Thread(target=lambda: result.append(self.run_backend(backend)))
        thread.start()
        try:
            self.assertTrue(backend.system.started.wait(5))
            calls = copy.deepcopy(backend.system.calls)
            sequence = backend.journal.sequence
            self.assertTrue(backend.running)
            self.assertEqual(backend.created_readers, [])
            with self.assertRaises(InvalidEvidence):
                backend._observe("protected")
            with self.assertRaises(InvalidEvidence):
                backend.retain({"action": "P1R_PARENT"})
            with self.assertRaises(InvalidEvidence):
                backend.perform(plan()[15], {})
            self.assertEqual(backend.system.calls, calls)
            self.assertEqual(backend.journal.sequence, sequence)
        finally:
            backend.system.finish.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["status"], "PASS_SIMULATION", result[0].get("failure"))
        parents = [c for c in backend.system.calls if c[0] == "parent_once"]
        self.assertEqual(parents, [("parent_once", 1234)])
        prepare = next(c for c in backend.system.calls if c[0] == "prepare_child")
        self.assertEqual(prepare[1], parent_argv("p1v-fixture"))
        self.assertFalse(any(c[0] in ("CONTROL", "TREATMENT") for c in backend.system.calls))

    def test_alias_filesystem_device_and_restore_order(self):
        backend = self.make()
        self.assertEqual(self.run_backend(backend)["status"], "PASS_SIMULATION")
        calls = backend.system.calls
        mounts = [c for c in calls if c[0] == "mount"]
        self.assertEqual(mounts, [
            ("mount", None, "/", None, lb.MS_REC | lb.MS_PRIVATE, None),
            ("mount", lb.ALIAS_PARENT, lb.ALIAS_PARENT, None, lb.MS_BIND, None),
            ("mount", None, lb.ALIAS_PARENT, None, 1024 | lb.MS_BIND | lb.MS_REMOUNT | lb.MS_RDONLY, None),
            ("mount", None, lb.MOUNT, "ext4", 1024 | lb.MS_REMOUNT | lb.MS_RDONLY, "synthetic-original-ext4-options"),
            ("mount", None, lb.MOUNT, "ext4", 1024 | lb.MS_REMOUNT, "synthetic-original-ext4-options"),
        ])
        self.assertEqual([c for c in calls if c[0] == "unshare"], [("unshare", lb.CLONE_NEWNS)])
        self.assertLess(calls.index(("syncfs", 41)), calls.index(mounts[3]))
        self.assertLess(calls.index(mounts[3]), calls.index(("block_ro", 40, True)))
        self.assertLess(calls.index(("block_ro", 40, False)), calls.index(mounts[4]))
        restore = calls[calls.index(("block_ro", 40, False)):calls.index(mounts[4])]
        self.assertIn(("block_get", 40), restore)
        self.assertIn(("observe:device_rw",), restore)

    def test_all_system_failpoints_retain_and_never_retry(self):
        complete = self.make()
        self.assertEqual(self.run_backend(complete)["status"], "PASS_SIMULATION")
        for operation in dict.fromkeys(c[0] for c in complete.system.calls):
            with self.subTest(operation=operation):
                backend = self.make()
                backend.system.fail = operation
                result = self.run_backend(backend)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(backend.system.fail_count, 1)
                self.assertEqual(backend.system.calls[-1][0], operation)
                self.assertTrue((backend.journal.root / "bootstrap.json").exists())
                with self.assertRaises(InvalidEvidence):
                    backend.perform(plan()[0], {"attempt": "p1v-fixture", "receipts": {}})

    def test_readback_uncertainty_stops_before_runtime_or_restore(self):
        cases = [
            ("namespace", "initial_user_namespace", False), ("namespace", "private_recursive", False),
            ("namespace", "original_namespace_id", "synthetic-ns"), ("namespace", "private_mount_id", 100),
            ("roles", "worker_single_threaded", False), ("roles", "guardian_original_namespace", False),
            ("alias_ro", "alias_parent_ro", False), ("alias_ro", "canonical_target", "/wrong"),
            ("alias_ro", "symlink_target", "/wrong"), ("alias_ro", "preserved_flags", 0),
            ("alias_ro", "output_still_rw", False),
            ("filesystem_ro", "superblock_ro", False), ("filesystem_ro", "mount_ro", False),
            ("sync_health", "syncfs_ok", False), ("sync_health", "journal_clean", False),
            ("sync_health", "healthy", False), ("filesystem_ro", "journal_clean", False),
            ("device_ro", "device_identity", {}), ("device_ro", "filesystem_still_ro", False),
            ("shutdown", "runtime_tree_empty", False), ("shutdown", "recursive_populated", 1),
            ("shutdown", "uncertain_descendants", [999]),
            ("protected", "exclusion_held", False), ("protected", "alias_locked", False),
            ("device_rw", "filesystem_still_ro", False), ("filesystem_rw", "healthy", False),
            ("filesystem_rw", "mount_options", ["wrong"]),
        ]
        for topic, key, value in cases:
            with self.subTest(topic=topic, key=key):
                backend = self.make()
                backend.system.observations[topic][key] = value
                result = self.run_backend(backend)
                self.assertEqual(result["status"], "FAIL")
                self.assertNotIn("RELEASE", backend.done)
                if topic not in ("device_rw", "filesystem_rw"):
                    self.assertEqual(backend.created_readers, [])
                    self.assertNotIn(("block_ro", 40, False), backend.system.calls)

    def test_device_readback_failure_blocks_following_action(self):
        backend = self.make()
        original = backend.system.block_get

        def wrong(fd):
            value = original(fd)
            return 0 if backend.system.ro else value

        backend.system.block_get = wrong
        result = self.run_backend(backend)
        self.assertEqual(result["failure"]["action"], "DEVICE_RO")
        self.assertEqual(backend.created_readers, [])
        self.assertNotIn(("block_ro", 40, False), backend.system.calls)

    def test_every_resource_domain_drift_rejected(self):
        for field in lb.DOMAIN_FIELDS:
            for stage in ("prospective", "actual"):
                with self.subTest(field=field, stage=stage):
                    backend = self.make()
                    domain = (backend.system.observations["cgroup_preflight"]["prospective"] if stage == "prospective"
                              else backend.system.observations["confinement"]["resource_domain_observed"])
                    domain[field] = {"domain": "other", "constraints": {}} if field in lb.CONTROLLERS else [99]
                    result = self.run_backend(backend)
                    self.assertEqual(result["failure"]["action"], "CONFINEMENT")
                    self.assertFalse(any(c[0] == "parent_once" for c in backend.system.calls))
                    if stage == "prospective":
                        self.assertFalse(any(c[0] == "cgroup_create" for c in backend.system.calls))

    def test_cgroup_policy_changes_rejected_before_creation(self):
        for key in ("controllers_to_enable", "settings_to_write", "bpf_to_attach"):
            backend = self.make()
            backend.system.observations["cgroup_preflight"][key] = ["added"]
            self.assertEqual(self.run_backend(backend)["status"], "FAIL")
            self.assertNotIn("cgroup_create", [c[0] for c in backend.system.calls])

    def test_launch_drift_all_fields_and_unknown_rejected(self):
        for field in lb.LAUNCH_FIELDS:
            baseline = launch_fixture()
            for value in (None, "UNKNOWN"):
                baseline[field] = value
                with self.subTest(field=field, value=value), self.assertRaises((InvalidEvidence, TypeError)):
                    lb.LaunchManifest.parse(baseline)
        for field in ("environment", "rlimits", "scheduler", "cpu_affinity", "numa", "gpu", "adapter_pins"):
            backend = self.make()
            manifest = backend.system.observations["confinement"]["launch_observed"]
            if field == "environment":
                manifest[field]["values"]["RAYON_NUM_THREADS"] = "2"
            elif field == "rlimits":
                manifest[field]["NOFILE"] = [1023, 2048]
            elif field == "cpu_affinity":
                manifest[field] = [0]
            elif field == "adapter_pins":
                manifest[field]["python"] = "0" * 64
            else:
                first = next(iter(manifest[field]))
                manifest[field][first] = "changed"
            with self.subTest(field=field):
                self.assertEqual(self.run_backend(backend)["status"], "FAIL")
                self.assertNotIn("parent_once", [c[0] for c in backend.system.calls])

    def test_payload_factory_never_reachable_out_of_order(self):
        backend = self.make()
        with self.assertRaises(InvalidEvidence):
            backend.perform(plan()[15], {"attempt": "p1v-fixture", "receipts": {}})
        self.assertEqual(backend.created_readers, [])
        self.assertEqual(backend.system.calls, [])

    def test_parent_wait_uncertainty_keeps_intent_and_protection(self):
        backend = self.make()

        def uncertain(pid):
            backend.system.call("uncertain_wait", pid)
            raise InterruptedError("synthetic reap uncertainty")

        backend.system.resume_wait_parent = uncertain
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(backend.running)
        records = sorted(backend.journal.root.glob("[0-9]*.json"))
        self.assertEqual(strict_json(records[-1].read_bytes())["record"]["status"], "INTENT")
        self.assertEqual(backend.created_readers, [])
        self.assertNotIn(("block_ro", 40, False), backend.system.calls)


class EnforcementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = PipelineFake(encode_raw(raw_fixture()), fake_content())
        result = Supervisor(backend, "p1v-fixture").run()
        assert result["status"] == "PASS_SIMULATION", result
        cls.envelope = strict_json(backend.sealed_bytes)

    def test_v2_accepted_v1_and_wrong_profile_rejected(self):
        self.assertEqual(validate_enforcement(self.envelope, True)["schema"], "mer.p1v.content.v1")
        for key, value in (("schema", "mer.p1v.enforcement.v1"), ("profile", "legacy")):
            bad = copy.deepcopy(self.envelope)
            bad[key] = value
            with self.assertRaises(InvalidEvidence):
                validate_enforcement(bad, True)

    def test_every_confinement_field_required_and_strict(self):
        base = confinement_fixture()
        for key in base:
            bad = copy.deepcopy(base)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(InvalidEvidence):
                lb.validate_confinement(bad, launch_fixture(), {"namespace_id": "synthetic-ns"})
        for key in ("seccomp_enforced", "namespace_apis_denied", "policy_sha256"):
            bad = dict(base, **{key: True})
            with self.assertRaises(InvalidEvidence):
                lb.validate_confinement(bad, launch_fixture(), {"namespace_id": "synthetic-ns"})

    def test_seccomp_bpf_credentials_fds_and_trust_rejected(self):
        cases = [("seccomp_mode", 1), ("seccomp_mode", 2), ("seccomp_mode", False),
                 ("p1v_device_bpf_attached", True), ("cgroup_bpf_added", ["filter"]),
                 ("uids", [0, 0, 0]), ("uids", [1000, 1001, 1000]), ("uids", [True] * 3),
                 ("gids", [0] * 3), ("gids", [1000, 1000, 1001]), ("groups", [1000, 27]),
                 ("groups_exact", False), ("no_new_privs", 1), ("inherited_fds", [0, 1, 2, 40]),
                 ("raw_or_privileged_fds", [40]), ("writable_alias_routes", ["/proc/1/root"]),
                 ("initial_user_namespace", False), ("private_mount_namespace", False),
                 ("trusted_execution_closure_verified", False), ("brokers_excluded", False),
                 ("cgroup_mode", "ACTIVE"), ("cgroup_nondelegated", False),
                 ("cgroup_active_monitoring_during_pair", True), ("cgroup_limits_changed", ["pids.max"]),
                 ("cgroup_controllers_changed", ["cpu"]), ("resource_domain_equal", False),
                 ("launch_profile_equal", False), ("setup_fds_closed", False)]
        for key, value in cases:
            r = confinement_fixture()
            r[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(InvalidEvidence):
                lb.validate_confinement(r, launch_fixture(), {"namespace_id": "synthetic-ns"})
        for cap in confinement_fixture()["capabilities"]:
            for value in (1, False):
                r = confinement_fixture()
                r["capabilities"][cap] = value
                with self.subTest(cap=cap), self.assertRaises(InvalidEvidence):
                    lb.validate_confinement(r, launch_fixture(), {"namespace_id": "synthetic-ns"})

    def test_manifest_and_closure_binding_tampering(self):
        for item in ("module_sha256", "dependencies", "launch_manifest_sha256", "execution_closure_sha256"):
            manifest = adapter_fixture()
            manifest[item] = {} if item == "dependencies" else "bad"
            with self.assertRaises(InvalidEvidence):
                lb.validate_adapter_manifest(manifest, launch_fixture(), closure_fixture())
        closure = closure_fixture()
        closure["premise"]["hostile_namespace_substitution"] = "KERNEL_DENIED"
        with self.assertRaises(InvalidEvidence):
            lb.validate_adapter_manifest(adapter_fixture(), launch_fixture(), closure)

    def test_legacy_future_authority_never_enables_adapter(self):
        # Fake old-seam tokens do not constitute an adapter-specific live grant.
        token = b"synthetic legacy authorization only" * 2
        for issue in (215, 216, 217, 218):
            authorization = FutureAuthorization(token, digest(token), issue)
            with self.subTest(issue=issue), patch("os.open", side_effect=AssertionError("unauthorized open")):
                with self.assertRaises(InvalidEvidence):
                    FutureBackend(authorization, lambda *_: None, lambda *_: None)
                with self.assertRaises(InvalidEvidence):
                    lb.NativePrimitives(authorization, None, "host", "p1v-fixture", adapter_fixture(), lambda: "AUTHORITY")
                with self.assertRaises(InvalidEvidence):
                    lb.NativeProcesses(authorization, None, "host", "p1v-fixture", adapter_fixture())
                with self.assertRaises(InvalidEvidence):
                    lb.DurableJournal("/not-used", "p1v-fixture", authorization=authorization)

    def test_implementation_grant_cannot_be_issued(self):
        token = b"synthetic implementation authority only" * 2
        for issue in (215, 216, 217, 218):
            scope = {"authority_issue": issue, "host_id": "synthetic", "attempt": "p1v-fixture",
                     "operations": [a.name for a in plan()], "adapter_manifest_sha256": "a" * 64}
            # All supplied pins match: the issue boundary itself must reject.
            with self.subTest(issue=issue), self.assertRaisesRegex(InvalidEvidence, "follow #218"):
                lb.ExecutionGrant(scope, digest(canonical_json(scope)), token, digest(token))

    def test_import_no_effect_and_no_shell_or_native_invocation(self):
        import ctypes
        import fcntl
        with patch("os.open", side_effect=AssertionError("open on import")), \
                patch("os.fork", side_effect=AssertionError("fork on import")), \
                patch.object(ctypes, "CDLL", side_effect=AssertionError("libc on import")), \
                patch.object(fcntl, "ioctl", side_effect=AssertionError("ioctl on import")):
            importlib.reload(lb)
        tree = ast.parse(Path(lb.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name == "subprocess" for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "subprocess")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ("system", "popen", "Popen", "seccomp", "bpf"))
                self.assertFalse(any(k.arg == "shell" for k in node.keywords))
        self.assertNotIn("/bin/sh", Path(lb.__file__).read_text())


class JournalTests(unittest.TestCase):
    def make(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory, lb.DurableJournal(directory.name, "p1v-fixture", fixture=directory)

    def test_chain_exclusive_marker_seals_and_no_reuse(self):
        directory, journal = self.make()
        record = {"step": 0, "action": "AUTHORITY", "status": "INTENT"}
        journal.retain(record)
        journal.retain(dict(record, status="PASS"))
        previous = "0" * 64
        for seq, path in enumerate(sorted(journal.root.glob("[0-9]*.json"))):
            data = path.read_bytes()
            decoded = strict_json(data)
            self.assertEqual(decoded["sequence"], seq)
            self.assertEqual(decoded["previous_sha256"], previous)
            previous = digest(data)
        self.assertEqual(journal.previous, previous)
        with self.assertRaises(FileExistsError):
            lb.DurableJournal(directory.name, "p1v-fixture", fixture=directory)
        for name in ("marker.json", "enforcement.json", "restoration.json"):
            _, journal = self.make()
            journal.publish(name, b"immutable", "MARKER")
            with self.assertRaises(FileExistsError):
                journal.publish(name, b"changed", "MARKER")
            self.assertEqual((journal.root / name).read_bytes(), b"immutable")
            self.assertTrue(journal.failed)

    def test_missing_intent_duplicate_result_and_failure_stop(self):
        _, journal = self.make()
        r = {"step": 0, "action": "AUTHORITY", "status": "PASS"}
        with self.assertRaises(InvalidEvidence):
            journal.retain(r)
        journal.retain(dict(r, status="INTENT"))
        journal.retain(dict(r, status="FAIL"))
        with self.assertRaises(InvalidEvidence):
            journal.retain(r)

    def test_file_and_directory_fsync_failure_retains_bytes(self):
        for fail_at in (1, 2):
            _, journal = self.make()
            calls = []
            original = os.fsync

            def fsync(fd):
                calls.append(fd)
                if len(calls) == fail_at:
                    raise OSError("synthetic fsync failure")
                original(fd)

            with patch("os.fsync", side_effect=fsync), self.assertRaises(OSError):
                journal.publish("retained.json", b"retained", "AUTHORITY")
            self.assertTrue(journal.failed)
            self.assertEqual((journal.root / "retained.json").read_bytes(), b"retained")
            with self.assertRaises(InvalidEvidence):
                journal.publish("retry.json", b"no", "AUTHORITY")

    def test_short_write_stops_without_retry(self):
        _, journal = self.make()
        with patch("os.write", return_value=1) as write, self.assertRaises(InvalidEvidence):
            journal.publish("short.json", b"longer", "AUTHORITY")
        self.assertEqual(write.call_count, 1)
        self.assertTrue(journal.failed)


class RoleTests(unittest.TestCase):
    class Processes:
        simulated = True

        def __init__(self):
            self.calls = []
            self.fork_pid = 10
            self.exit_code = 0

        def fork(self):
            self.calls.append("fork")
            return self.fork_pid

        def wait_once(self, pid, stopped):
            self.calls.append(("wait", pid, stopped))
            return {"pid": pid, "stopped": True} if stopped else {"pid": pid, "exited": True, "exit_code": self.exit_code}

        def resume_wait_once(self, pid):
            self.calls.append(("continue", pid))
            return self.wait_once(pid, False)

    def test_guardian_passive_single_wait_worker_failure(self):
        processes = self.Processes()
        roles = lb.ProcessRoles(processes, attempt="p1v-fixture", manifest=adapter_fixture())
        roles.guardian(lambda: self.fail("parent must not execute worker callback"))
        self.assertEqual(processes.calls, ["fork", ("wait", 10, True), ("continue", 10), ("wait", 10, False)])
        with self.assertRaises(InvalidEvidence):
            roles.guardian(lambda: {})
        processes = self.Processes()
        processes.exit_code = 1
        roles = lb.ProcessRoles(processes, attempt="p1v-fixture", manifest=adapter_fixture())
        with self.assertRaises(InvalidEvidence):
            roles.guardian(lambda: {})
        self.assertEqual(processes.calls, ["fork", ("wait", 10, True), ("continue", 10), ("wait", 10, False)])

    def test_stopped_handoff_one_resume_one_wait(self):
        processes = self.Processes()
        roles = lb.ProcessRoles(processes, attempt="p1v-fixture", manifest=adapter_fixture())
        pid = roles.prepare_stopped_child(parent_argv("p1v-fixture"), lb.LaunchManifest.parse(launch_fixture()).encoded, "synthetic-runtime")
        self.assertEqual(roles.resume_wait_parent(pid), 0)
        self.assertEqual(processes.calls, ["fork", ("wait", 10, True), ("continue", 10), ("wait", 10, False)])
        with self.assertRaises(InvalidEvidence):
            roles.resume_wait_parent(pid)


class AdditionalGateTests(unittest.TestCase):
    def test_all_native_constructors_refuse_before_libc_or_process_access(self):
        import ctypes
        import fcntl
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("real libc")), \
                patch.object(fcntl, "ioctl", side_effect=AssertionError("real ioctl")), \
                patch("os.fork", side_effect=AssertionError("real fork")), \
                patch("os.open", side_effect=AssertionError("real device open")):
            manifest = adapter_fixture()
            for factory in (
                lambda: lb.NativePrimitives(None, None, "synthetic", "p1v-fixture", manifest, lambda: "AUTHORITY"),
                lambda: lb.NativeProcesses(None, None, "synthetic", "p1v-fixture", manifest),
            ):
                with self.assertRaises(InvalidEvidence):
                    factory()

    def test_namespace_and_device_bindings_validated_offline(self):
        raw, content = encode_raw(raw_fixture()), fake_content()
        cases = [("NAMESPACE", "private_mount_id", 100), ("NAMESPACE", "original_namespace_id", "synthetic-ns"),
                 ("ALIAS_RO", "output_still_rw", False), ("FILESYSTEM_RO", "syncfs_checked", False),
                 ("DEVICE_RO", "device_identity", {}), ("PROTECTED", "private_mount_id", 999),
                 ("REVALIDATE", "superblock_ro", False), ("SHUTDOWN", "recursive_populated", 1)]
        for action, key, value in cases:
            backend = PipelineFake(raw, content)
            backend.receipts[action][key] = value
            with self.subTest(action=action):
                result = Supervisor(backend, "p1v-fixture").run()
                self.assertEqual(result["failure"]["action"], action)
                self.assertNotIn("DEVICE_RW", backend.trace)

    def test_complete_manifest_and_environment_empty_vs_absent(self):
        baseline = launch_fixture()
        baseline["environment"]["values"]["EMPTY_VARIABLE"] = ""
        frozen = lb.LaunchManifest.parse(baseline)
        changed = copy.deepcopy(baseline)
        del changed["environment"]["values"]["EMPTY_VARIABLE"]
        with self.assertRaises(InvalidEvidence):
            frozen.compare(changed)
        for limit in lb.RLIMITS:
            bad = launch_fixture()
            del bad["rlimits"][limit]
            with self.assertRaises(InvalidEvidence):
                lb.LaunchManifest.parse(bad)
        for field, keys in lb.POLICY_FIELDS.items():
            for key in keys:
                bad = launch_fixture()
                del bad[field][key]
                with self.assertRaises(InvalidEvidence):
                    lb.LaunchManifest.parse(bad)

    def test_worker_failure_has_no_restore_callback(self):
        class ChildExit(BaseException):
            pass

        class Processes(RoleTests.Processes):
            def __init__(self):
                super().__init__()
                self.fork_pid = 0

            def stop_self(self):
                self.calls.append("stop")

            def exit_worker(self, code):
                self.calls.append(("exit", code))
                raise ChildExit()

        processes = Processes()
        roles = lb.ProcessRoles(processes, attempt="p1v-fixture", manifest=adapter_fixture())

        def failed_worker():
            raise OSError("synthetic worker failure with protection retained")

        with self.assertRaises(ChildExit):
            roles.guardian(failed_worker)
        self.assertEqual(processes.calls, ["fork", "stop", ("exit", 1)])

    def test_child_role_order_closes_fds_stops_then_execs_parent(self):
        class Executed(BaseException):
            pass

        class Processes(RoleTests.Processes):
            def __init__(self):
                super().__init__()
                self.fork_pid = 0

            def enter_cgroup(self, cgroup):
                self.calls.append(("membership", cgroup))

            def apply_launch_credentials_and_close_fds(self, manifest):
                self.calls.append(("credentials_fds", manifest))

            def stop_then_exec_parent(self, argv, manifest):
                self.calls.append(("stop_then_exec_parent", argv, manifest))
                raise Executed()

        processes = Processes()
        roles = lb.ProcessRoles(processes, attempt="p1v-fixture", manifest=adapter_fixture())
        baseline = lb.LaunchManifest.parse(launch_fixture())
        with self.assertRaises(Executed):
            roles.prepare_stopped_child(parent_argv("p1v-fixture"), baseline.encoded, "synthetic")
        self.assertEqual(processes.calls, ["fork", ("membership", "synthetic"),
                                          ("credentials_fds", baseline.encoded),
                                          ("stop_then_exec_parent", parent_argv("p1v-fixture"), baseline.encoded)])


class MountAndScopeTests(unittest.TestCase):
    def test_mount_only_ro_is_distinct_from_superblock_ro(self):
        mount = lb.parse_mountinfo("200 1 259:4 / /mnt/localssd ro,noatime - ext4 /dev/nvme0n1 rw,discard\n")[0]
        self.assertTrue(mount["mount_ro"])
        self.assertFalse(mount["superblock_ro"])
        mount = lb.parse_mountinfo("200 1 259:4 / /mnt/localssd ro,noatime - ext4 /dev/nvme0n1 ro,discard\n")[0]
        self.assertTrue(mount["superblock_ro"])
        self.assertEqual(mount["mount_id"], 200)
        for raw in ("", "malformed", "200 1 259:4 / /mnt/localssd noatime - ext4 /dev/nvme0n1 rw\n",
                    "200 1 259:4 / /mnt/localssd ro,rw - ext4 /dev/nvme0n1 ro\n"):
            with self.assertRaises(InvalidEvidence):
                lb.parse_mountinfo(raw)

    def test_native_release_to_wait_has_no_intervening_adapter_call(self):
        tree = ast.parse(Path(lb.__file__).read_text())
        native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeProcesses")
        method = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == "resume_wait_once")
        release = next(i for i, n in enumerate(method.body)
                       if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "kill")
        following = method.body[release + 1]
        self.assertIsInstance(following, ast.Assign)
        self.assertEqual(ast.unparse(following.value), "os.waitpid(pid, 0)")
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "waitpid" for n in ast.walk(method)), 1)

    def test_simulation_cannot_be_relabelled_live(self):
        backend = PipelineFake(encode_raw(raw_fixture()), fake_content())
        Supervisor(backend, "p1v-fixture").run()
        envelope = strict_json(backend.sealed_bytes)
        envelope["mode"] = "LIVE"
        with self.assertRaises(InvalidEvidence):
            validate_enforcement(envelope)

    def test_later_grant_requires_scope_and_matching_external_token(self):
        # Deliberately invalid: this suite never creates a valid live grant.
        scope = {"authority_issue": 219, "host_id": "synthetic", "attempt": "p1v-fixture",
                 "operations": [a.name for a in plan()], "adapter_manifest_sha256": "a" * 64}
        with self.assertRaises(InvalidEvidence):
            lb.ExecutionGrant(scope, digest(canonical_json(scope)), b"invalid" * 10, "0" * 64)
        for key in scope:
            bad = copy.deepcopy(scope)
            del bad[key]
            with self.assertRaises(InvalidEvidence):
                lb.ExecutionGrant(bad, "0" * 64, b"invalid" * 10, "0" * 64)


class ReleaseExecBoundaryTests(unittest.TestCase):
    """Exercise the resumed-child defect with syntax and synthetic call traces.

    No NativeProcesses instance or live grant is constructed. The unbound
    primitive receives a fake authorization interface; every OS call is mocked.
    """
    class Executed(BaseException):
        pass

    class PreparedChild:
        def __init__(self, calls, deny=False):
            self.attempt = "p1v-fixture"
            self.context = (None, None, "synthetic", self.attempt, adapter_fixture())
            self.calls, self.deny = calls, deny

        def _check(self, action):
            self.calls.append(("authorize", action))
            if self.deny:
                raise InvalidEvidence("synthetic denied authority")

    @staticmethod
    def method(owner, name):
        tree = ast.parse(Path(lb.__file__).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_native_stop_is_immediately_followed_by_only_direct_execve(self):
        method = self.method("NativeProcesses", "stop_then_exec_parent")
        stops = [i for i, n in enumerate(method.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == "os.kill"]
        self.assertEqual(len(stops), 1)
        stop = stops[0]
        self.assertEqual(ast.unparse(method.body[stop]), "os.kill(pid, signal.SIGSTOP)")
        self.assertEqual(len(method.body[stop + 1:]), 1)
        following = method.body[stop + 1]
        self.assertEqual(ast.unparse(following), "os.execve(executable, argv, environment)")
        self.assertEqual([ast.unparse(n.func) for n in ast.walk(following) if isinstance(n, ast.Call)],
                         ["os.execve"])
        self.assertTrue(all(isinstance(arg, ast.Name) for arg in following.value.args))
        before = ast.unparse(ast.Module(body=method.body[:stop], type_ignores=[]))
        for required in ('self._check(\'P1R_PARENT\')', 'import signal', 'parent_argv(self.attempt)',
                         'LaunchManifest.parse', 'launch.sha256', "['launch_manifest_sha256']",
                         "['environment']['values']", 'executable = argv[0]', 'pid = os.getpid()'):
            self.assertIn(required, before)

    def test_runtime_role_has_one_atomic_stop_exec_and_no_old_sequence(self):
        method = self.method("ProcessRoles", "prepare_stopped_child")
        child = next(n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "pid == 0")
        calls = [ast.unparse(n.value.func) for n in child.body if isinstance(n, ast.Expr)]
        self.assertEqual(calls, ['self.processes.enter_cgroup',
                                 'self.processes.apply_launch_credentials_and_close_fds',
                                 'self.processes.stop_then_exec_parent'])
        self.assertNotIn('stop_self', ast.unparse(child))
        tree = ast.parse(Path(lb.__file__).read_text())
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name == 'exec_parent' for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == 'exec_parent' for n in ast.walk(tree)))
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'os.execve'
                             for n in ast.walk(tree)), 1)

    def test_guardian_authorization_precedes_sigcont_then_single_waitpid(self):
        method = self.method("NativeProcesses", "resume_wait_once")
        release = next(i for i, n in enumerate(method.body)
                       if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                       and ast.unparse(n.value.func) == 'os.kill')
        self.assertEqual(ast.unparse(method.body[release]), 'os.kill(pid, signal.SIGCONT)')
        wait = method.body[release + 1]
        self.assertEqual(ast.unparse(wait.value), 'os.waitpid(pid, 0)')
        self.assertEqual([ast.unparse(n.func) for n in ast.walk(wait) if isinstance(n, ast.Call)], ['os.waitpid'])
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'os.waitpid'
                             for n in ast.walk(method)), 1)
        self.assertIn("self._check('P1R_PARENT')", ast.unparse(ast.Module(body=method.body[:release], type_ignores=[])))

    def test_preparation_calls_all_precede_fake_stop_and_exec_is_exact(self):
        import signal
        from contextlib import ExitStack
        calls = []
        child = self.PreparedChild(calls)
        baseline = lb.LaunchManifest.parse(launch_fixture()).encoded
        argv = parent_argv(child.attempt)
        released = False

        def traced(name, function):
            def invoke(*args, **kwargs):
                self.assertFalse(released, 'helper call after SIGCONT: ' + name)
                calls.append(name)
                return function(*args, **kwargs)
            return invoke

        def stop(pid, sig):
            nonlocal released
            self.assertEqual((pid, sig), (1234, signal.SIGSTOP))
            calls.append('SIGSTOP/SIGCONT')
            released = True

        def execute(executable, actual_argv, environment):
            self.assertTrue(released)
            self.assertEqual(calls[-1], 'SIGSTOP/SIGCONT')
            self.assertEqual((executable, actual_argv, environment),
                             (argv[0], argv, launch_fixture()['environment']['values']))
            calls.append('execve')
            raise self.Executed()

        with ExitStack() as stack:
            for name in ('require', 'parent_argv', 'canonical_json', 'digest'):
                stack.enter_context(patch.object(lb, name, traced(name, getattr(lb, name))))
            stack.enter_context(patch.object(lb.os, 'getpid', side_effect=traced('getpid', lambda: 1234)))
            stack.enter_context(patch.object(lb.os, 'kill', side_effect=stop))
            exec_call = stack.enter_context(patch.object(lb.os, 'execve', side_effect=execute))
            with self.assertRaises(self.Executed):
                lb.NativeProcesses.stop_then_exec_parent(child, argv, baseline)
        self.assertEqual(calls[0], ('authorize', 'P1R_PARENT'))
        self.assertEqual(calls[-2:], ['SIGSTOP/SIGCONT', 'execve'])
        exec_call.assert_called_once()

    def test_denied_authority_argv_environment_and_manifest_never_stop(self):
        baseline = lb.LaunchManifest.parse(launch_fixture()).encoded
        changed = launch_fixture()
        changed['environment']['values']['UNREVIEWED'] = 'injection'
        invalid = launch_fixture()
        invalid['environment']['values']['BAD=NAME'] = 'x'
        cases = [('authority', baseline, parent_argv('p1v-fixture'), True),
                 ('argv', baseline, ('/wrong',), False),
                 ('environment', lb.LaunchManifest.parse(changed).encoded, parent_argv('p1v-fixture'), False),
                 ('invalid_environment', canonical_json(invalid), parent_argv('p1v-fixture'), False),
                 ('noncanonical', baseline + b' ', parent_argv('p1v-fixture'), False),
                 ('manifest_type', baseline.decode(), parent_argv('p1v-fixture'), False)]
        for name, manifest, argv, deny in cases:
            child = self.PreparedChild([], deny=deny)
            with self.subTest(case=name), patch.object(lb.os, 'getpid') as getpid, \
                    patch.object(lb.os, 'kill') as kill, patch.object(lb.os, 'execve') as execute:
                with self.assertRaises(InvalidEvidence):
                    lb.NativeProcesses.stop_then_exec_parent(child, argv, manifest)
                getpid.assert_not_called()
                kill.assert_not_called()
                execute.assert_not_called()

    def test_stop_failure_never_execs_or_retries(self):
        child = self.PreparedChild([])
        with patch.object(lb.os, 'getpid', return_value=1234), \
                patch.object(lb.os, 'kill', side_effect=OSError('synthetic stop failure')) as kill, \
                patch.object(lb.os, 'execve') as execute:
            with self.assertRaises(OSError):
                lb.NativeProcesses.stop_then_exec_parent(child, parent_argv(child.attempt),
                                                        lb.LaunchManifest.parse(launch_fixture()).encoded)
        kill.assert_called_once()
        execute.assert_not_called()

    def test_exec_failure_propagates_without_retry_or_recovery(self):
        child = self.PreparedChild([])
        with patch.object(lb.os, 'getpid', return_value=1234), patch.object(lb.os, 'kill') as kill, \
                patch.object(lb.os, 'execve', side_effect=OSError('synthetic exec failure')) as execute:
            with self.assertRaises(OSError):
                lb.NativeProcesses.stop_then_exec_parent(child, parent_argv(child.attempt),
                                                        lb.LaunchManifest.parse(launch_fixture()).encoded)
        kill.assert_called_once()
        execute.assert_called_once()

    def test_implementation_issue_rejected_on_scope_readback_and_recheck(self):
        token = b'synthetic legacy seam only' * 2
        authorization = FutureAuthorization(token, digest(token), 218)
        scope = {'authority_issue': 218, 'host_id': 'synthetic', 'attempt': 'p1v-fixture',
                 'operations': [a.name for a in plan()],
                 'adapter_manifest_sha256': digest(canonical_json(adapter_fixture()))}
        # Fake serialized input only; never an ExecutionGrant instance.
        class StoredScope:
            _scope = canonical_json(scope)

        with self.assertRaisesRegex(InvalidEvidence, 'same future issue'):
            lb.ExecutionGrant.check(StoredScope(), authorization, 'synthetic', 'p1v-fixture',
                                    adapter_fixture(), 'P1R_PARENT')
        receipt = {'host_id': 'synthetic', 'execution_grant': scope, 'adapter_manifest': adapter_fixture()}
        with self.assertRaisesRegex(InvalidEvidence, 'postdate implementation'):
            lb.validate_execution_scope(receipt, {'mode': 'LIVE', 'attempt': 'p1v-fixture'})


if __name__ == "__main__":
    unittest.main()
