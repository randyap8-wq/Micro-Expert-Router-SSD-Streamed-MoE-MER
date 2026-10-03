"""Portable P1V4 validation: synthetic interfaces and temporary evidence only.

Only synthetic adapter scopes are constructed for pure simulated tests. Native
entry points use rejection tests or fake receivers with fully mocked syscalls.
"""

import ast
import copy
import importlib
import io
import os
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from . import (AUTHORITY, EXPECTATIONS, FutureAuthorization, InvalidEvidence,
               canonical_json, digest, strict_json)
from . import linux_backend as lb
from .supervisor import (FutureBackend, Supervisor, parent_argv, plan,
                         validate_enforcement)
from .test_p1v import (FakeReader, PipelineFake, adapter_fixture, closure_fixture,
                      confinement_fixture, encode_raw, fake_content,
                      launch_fixture, raw_fixture, receipts, storage_fixture, execution_fixture)


PROTECTED_P1V_FILES = {
    "__init__.py": "a936eec1bd452981180dd06ad236caf84040ba3d3275a5ad56de5a570228b1be",
    "content_hasher.py": "dec979b29cb55f0e61afd08173267d338bafe120ebffa98b4305cbaca42f3085",
    "offline_verify.py": "6008eceb36210b2755ff4ba3d5f0970f0656a919bdcfc6151a65f52562b86537",
    "supervisor.py": "c0f57a524814ae12677af8d42a65fa08b900a10ba183093d6006bdbf5df354f8",
    "test_p1v.py": "ef2eb74ed4a469e80a9f2ec4c8faa8e991f29109e04e396ec37b92402b0f5a4a",
    "linux_backend.py": "91888947b5610dd9ee6757e8488cbc66e185b95cf8f2285f3a250f871635a8ed",
}
REQUIRED_P1V_FILES = frozenset(PROTECTED_P1V_FILES) | {"test_linux_backend.py"}
OPTIONAL_P1V5_FILES = frozenset({"live_system.py", "test_live_system.py",
                                 "linux_gate.py", "test_linux_gate.py"})


def assert_package_scope(test, root):
    """Shared real/fixture witness: #221 permits only frozen #217 successors."""
    actual = {p.name for p in root.iterdir() if p.is_file()}
    test.assertLessEqual(REQUIRED_P1V_FILES, actual, "missing required regular files")
    test.assertLessEqual(actual, REQUIRED_P1V_FILES | OPTIONAL_P1V5_FILES,
                         "unauthorized regular files")
    for name, expected in PROTECTED_P1V_FILES.items():
        test.assertEqual(digest((root / name).read_bytes()), expected,
                         "protected SHA256: " + name)


class FixtureSystem:
    simulated = True

    def __init__(self, raw, content, storage_identity=None):
        storage_identity = storage_fixture() if storage_identity is None else storage_identity
        self.raw = copy.deepcopy(raw)
        self.receipts = receipts(raw, content, storage_identity)
        self.calls = []
        self.ro = 0
        self.fail = None
        self.fail_count = 0
        self.running = False
        self.started, self.finish = threading.Event(), threading.Event()
        self.finish.set()
        self.backing = {"device_fd": 40, "filesystem_fd": 41,
                        "identity": copy.deepcopy(storage_identity),
                        "host_mount_id": 100, "private_mount_id": 200,
                        "mount_flags": 1024, "mount_data": "synthetic-original-ext4-options", "alias_flags": 1024}
        self.observations = {
            "authority": self.receipts["AUTHORITY"],
            "exclusion": self.receipts["EXCLUSION"], "topology": self.receipts["TOPOLOGY"],
            "roles": {"guardian_original_namespace": True, "worker_single_threaded": True,
                      "initial_user_namespace": True, "worker_privileged": True},
            "namespace": {**self.receipts["NAMESPACE"], "original_namespace_id": "synthetic-original-ns",
                          "original_mount_id": 100, "private_mount_id": 200},
            "alias_bind": self.receipts["ALIAS_BIND"],
            "alias_ro": {**self.receipts["ALIAS_RO"], "preserved_flags": 1024,
                         "canonical_target": lb.CANONICAL, "output_still_rw": True},
            "metadata": self.receipts["METADATA"],
            "sync_health": {"syncfs_ok": True, "healthy": True, "external_journal": False, "recovery_required": True},
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
                                          healthy=True, external_journal=False)

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

        authorization, grant = execution_fixture()
        backend = lb.LinuxBackend(system, journal, launch_fixture(), adapter_fixture(), closure_fixture(),
                                  host_id="synthetic-host", attempt="p1v-fixture", authorization=authorization,
                                  grant=grant, fixture_reader_factory=reader)
        backend.created_readers = created
        return backend

    def run_backend(self, backend):
        return Supervisor(backend, "p1v-fixture").run()

    def test_authority_returns_prospective_manifest_without_child_observation(self):
        backend = self.make()
        self.assertNotIn("launch_observed", backend.system.observations["authority"])
        with patch.object(lb.LaunchManifest, "compare", side_effect=AssertionError("premature child observation")):
            receipt = backend._perform("AUTHORITY", {"receipts": {}})
        self.assertEqual(receipt["launch_manifest"], launch_fixture())
        self.assertEqual(receipt["adapter_manifest"], adapter_fixture())
        self.assertEqual(receipt["execution_closure"], closure_fixture())
        self.assertEqual(receipt["execution_grant"], {"status": "NOT_GRANTED", "storage_identity": storage_fixture()})
        self.assertNotIn("launch_observed", receipt)
        self.assertIsNone(backend.child)
        self.assertEqual(backend.system.calls, [("observe:authority",)])

    def test_authority_rejects_even_exact_or_empty_launch_observed(self):
        for observed in (launch_fixture(), {}, None, False):
            with self.subTest(observed=observed):
                backend = self.make()
                backend.system.observations["authority"]["launch_observed"] = observed
                with patch.object(lb.LaunchManifest, "compare", side_effect=AssertionError("must reject presence")):
                    with self.assertRaisesRegex(InvalidEvidence, "AUTHORITY forbids child launch observation"):
                        backend.perform(plan("p1v-fixture")[0], {"attempt": "p1v-fixture", "receipts": {}})
                self.assertTrue(backend.failed)
                self.assertEqual(backend.done, [])
                self.assertIsNone(backend.child)
                self.assertEqual(backend.system.calls, [("observe:authority",)])

    def test_authority_rejects_privileged_setup_credentials_as_launch_receipt(self):
        observed = launch_fixture()
        observed["credentials"] = {"uid": 0, "gid": 0, "groups": [0]}
        backend = self.phase_failure("AUTHORITY", "authority", {"launch_observed": observed})
        self.assertEqual(backend.system.calls, [("observe:authority",)])
        self.assertIsNone(backend.child)

    def test_authority_still_rejects_closure_dependency_and_host_drift(self):
        closure = closure_fixture()
        closure["source_commit"] = "0" * 40
        manifest = adapter_fixture()
        manifest["dependencies"]["python"] = "0" * 64
        for field, value in (("execution_closure", closure), ("adapter_manifest", manifest),
                             ("host_id", "other-host")):
            with self.subTest(field=field):
                backend = self.phase_failure("AUTHORITY", "authority", {field: value})
                self.assertEqual(backend.system.calls, [("observe:authority",)])
        for field in ("inputs_verified", "outputs_absent", "marker_absent", "frozen_environment", "frozen_limits"):
            with self.subTest(field=field):
                self.phase_failure("AUTHORITY", "authority", {field: False})

    def test_invalid_prospective_manifest_rejected_before_any_observation(self):
        variants = []
        for field in ("uid", "gid"):
            bad = launch_fixture()
            bad["credentials"][field] = 0
            variants.append(bad)
        bad = launch_fixture()
        del bad["rlimits"]["LOCKS"]
        variants.append(bad)
        bad = launch_fixture()
        bad["environment"]["complete"] = False
        variants.append(bad)
        bad = launch_fixture()
        bad["adapter_pins"]["python"] = "invalid"
        variants.append(bad)
        for baseline in variants:
            with self.subTest(baseline=baseline):
                backend = self.make()
                with self.assertRaises(InvalidEvidence):
                    lb.LaunchManifest.parse(baseline)
                with self.assertRaises(InvalidEvidence):
                    lb.validate_adapter_manifest(adapter_fixture(), baseline, closure_fixture())
                with self.assertRaises(InvalidEvidence):
                    lb.LinuxBackend(backend.system, backend.journal, baseline, adapter_fixture(), closure_fixture(),
                                    host_id="synthetic-host", attempt="p1v-fixture",
                                    fixture_reader_factory=backend.reader_factory)
                self.assertEqual(backend.system.calls, [])

    def confinement_failure(self, changes, absent=()):
        backend = self.phase_failure("CONFINEMENT", "confinement", changes, absent)
        self.assertEqual(backend.child, 1234)
        self.assertEqual(backend.system.calls[-1], ("observe:confinement",))
        self.assertNotIn("MARKER", backend.done)
        self.assertNotIn("parent_once", [c[0] for c in backend.system.calls])
        self.assertEqual(backend.created_readers, [])
        return backend

    def test_confinement_rejects_other_or_probe_pid_even_with_exact_manifest(self):
        for pid in (1235, 9999):
            with self.subTest(pid=pid):
                self.confinement_failure({"handoff_pid": pid, "launch_observed": launch_fixture()})

    def test_confinement_requires_strict_stopped_true(self):
        for stopped in (False, 1, None, "true"):
            with self.subTest(stopped=stopped):
                self.confinement_failure({"handoff_stopped": stopped})

    def test_confinement_requires_actual_launch_observed(self):
        self.confinement_failure({}, ("launch_observed",))

    def test_confinement_preserves_namespace_and_cgroup_binding(self):
        for field in ("namespace_id", "cgroup_id"):
            with self.subTest(field=field):
                self.confinement_failure({field: "other"})

    def test_confinement_rejects_each_wrong_actual_uid_gid_and_groups(self):
        for field in ("uids", "gids"):
            for index in range(3):
                for value in (0, 1001):
                    actual = [1000] * 3
                    actual[index] = value
                    with self.subTest(field=field, index=index, value=value):
                        self.confinement_failure({field: actual})
        for groups in ([], [0], [1001], [1000, 1001]):
            with self.subTest(groups=groups):
                self.confinement_failure({"groups": groups})
        for field, value in (("uid", 1001), ("gid", 1001), ("groups", [1001])):
            observed = launch_fixture()
            observed["credentials"][field] = value
            with self.subTest(launch_credential=field):
                self.confinement_failure({"launch_observed": observed})

    def test_confinement_rejects_valid_drift_in_every_full_launch_field(self):
        changed = launch_fixture()
        changed["environment"]["values"]["PATH"] = "/other/bin"
        changed["rlimits"]["LOCKS"] = [1023, 2048]
        changed["umask"] = 0o077
        changed["stdio"][0]["backing"] = "other"
        changed["scheduler"]["attributes"]["priority"] = 1
        changed["nice"] = 1
        changed["cpu_affinity"] = [0]
        changed["io_priority"]["priority"] = 1
        changed["cpu_policy"]["boost"] = True
        changed["numa"]["memory_policy"] = "other"
        changed["thp"]["policy"] = "other"
        changed["numa_balancing"]["policy"] = "other"
        changed["resource_domain"]["cpu"]["domain"] = "other"
        changed["gpu"]["clocks"] = "other-normalization-certificate"
        changed["cwd"] = "/other"
        changed["signals"]["mask"] = [1]
        changed["credentials"]["uid"] = 1001
        changed["adapter_pins"]["python"] = "0" * 64
        changed["session"]["session_id"] = 101
        changed["security_context"]["labels"] = ["other"]
        self.assertEqual(set(changed), set(lb.LAUNCH_FIELDS))
        for field in lb.LAUNCH_FIELDS:
            observed = launch_fixture()
            observed[field] = changed[field]
            with self.subTest(field=field):
                lb.LaunchManifest.parse(observed)
                self.assertNotEqual(observed[field], launch_fixture()[field])
                self.confinement_failure({"launch_observed": observed})

    def test_confinement_rejects_drift_in_all_sixteen_rlimits(self):
        self.assertEqual(len(lb.RLIMITS), 16)
        self.assertIn("LOCKS", lb.RLIMITS)
        for limit in lb.RLIMITS:
            observed = launch_fixture()
            observed["rlimits"][limit] = [1023, 2048]
            with self.subTest(limit=limit):
                self.confinement_failure({"launch_observed": observed})

    def test_prospective_gpu_clock_certificate_requires_exact_confinement_receipt(self):
        backend = self.make()
        observed = launch_fixture()
        observed["gpu"]["clocks"] = "different-actual-normalization-certificate"
        backend.system.observations["confinement"]["launch_observed"] = observed
        result = self.run_backend(backend)
        self.assertEqual(result["failure"]["action"], "CONFINEMENT")
        self.assertEqual(result["receipts"]["AUTHORITY"]["launch_manifest"]["gpu"]["clocks"],
                         launch_fixture()["gpu"]["clocks"])
        self.assertNotIn("MARKER", backend.done)
        self.assertNotIn("parent_once", [c[0] for c in backend.system.calls])

    def test_full_observation_follows_exact_child_preparation_in_unchanged_action_order(self):
        backend = self.make()
        original = backend.system.observe

        def observe(topic):
            if topic == "authority":
                self.assertIsNone(backend.child)
                self.assertNotIn("prepare_child", [c[0] for c in backend.system.calls])
            if topic == "confinement":
                self.assertEqual(backend.child, 1234)
                self.assertEqual(backend.system.calls[-1],
                                 ("prepare_child", parent_argv("p1v-fixture"), backend.baseline.encoded,
                                  "synthetic-runtime"))
            return original(topic)

        backend.system.observe = observe
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        self.assertEqual(backend.done, [
            "AUTHORITY", "EXCLUSION", "TOPOLOGY", "NAMESPACE", "ALIAS_BIND", "ALIAS_RO", "METADATA",
            "FILESYSTEM_RO", "DEVICE_RO", "PROTECTED", "CONFINEMENT", "MARKER", "P1R_PARENT", "SHUTDOWN",
            "REVALIDATE", "HASH", "VERIFY", "SEAL", "DEVICE_RW", "FILESYSTEM_RW", "RESTORATION_SEAL", "RELEASE"])
        calls = backend.system.calls
        start = calls.index(("observe:cgroup_preflight",))
        self.assertEqual([c[0] for c in calls[start:start + 4]],
                         ["observe:cgroup_preflight", "cgroup_create", "prepare_child", "observe:confinement"])
        self.assertEqual(calls.count(("observe:confinement",)), 1)
        self.assertEqual(calls.count(("parent_once", 1234)), 1)
        self.assertEqual(result["receipts"]["CONFINEMENT"]["launch_observed"],
                         result["receipts"]["AUTHORITY"]["launch_manifest"])

    def phase_failure(self, action, topic, changes, absent=()):
        """Inject one observation at the named phase, after earlier gates passed."""
        backend = self.make()
        original = backend.system.observe

        def observe(name):
            snapshot = original(name)
            if name == topic and plan()[len(backend.done)].name == action:
                snapshot.update(changes)
                for key in absent:
                    del snapshot[key]
            return snapshot

        backend.system.observe = observe
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure"]["action"], action)
        self.assertTrue(backend.failed)
        self.assertNotIn(action, backend.done)
        self.assertNotIn("RELEASE", backend.done)
        return backend

    def pre_ro_failure(self, changes, absent=()):
        backend = self.phase_failure("FILESYSTEM_RO", "sync_health", changes, absent)
        self.assertEqual(backend.system.calls[-1], ("observe:sync_health",))
        self.assertFalse(any(c[0] == "mount" and c[2] == lb.MOUNT for c in backend.system.calls))
        self.assertFalse(any(c[0] == "block_ro" for c in backend.system.calls))
        self.assertEqual(backend.created_readers, [])

    def test_pre_ro_synced_active_without_journal_clean_passes_in_order(self):
        backend = self.make()
        self.assertEqual(backend.system.observations["sync_health"], {
            "syncfs_ok": True, "healthy": True, "external_journal": False, "recovery_required": True})
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        calls = backend.system.calls
        start = calls.index(("syncfs", 41))
        self.assertEqual(calls[start:start + 4], [
            ("syncfs", 41), ("observe:sync_health",),
            ("mount", None, lb.MOUNT, "ext4", 1024 | lb.MS_REMOUNT | lb.MS_RDONLY,
             "synthetic-original-ext4-options"), ("observe:filesystem_ro",)])

    def test_pre_ro_sync_failure_rejects_before_remount(self):
        self.pre_ro_failure({"syncfs_ok": False})
        backend = self.make()
        backend.system.fail = "syncfs"
        result = self.run_backend(backend)
        self.assertEqual(result["failure"]["action"], "FILESYSTEM_RO")
        self.assertEqual(backend.system.calls[-1], ("syncfs", 41))
        self.assertNotIn(("observe:sync_health",), backend.system.calls)

    def test_pre_ro_health_failure_rejects_before_remount(self):
        self.pre_ro_failure({"healthy": False})

    def test_pre_ro_external_journal_rejects_before_remount(self):
        self.pre_ro_failure({"external_journal": True})

    def test_pre_ro_recovery_clear_rejects_before_remount(self):
        self.pre_ro_failure({"recovery_required": False})

    def test_pre_ro_fabricated_journal_clean_field_rejects(self):
        for value in (True, False, None, 1, "true"):
            with self.subTest(value=value):
                self.pre_ro_failure({"journal_clean": value})

    def test_pre_ro_fields_are_required_strict_booleans(self):
        for key in ("syncfs_ok", "healthy", "external_journal", "recovery_required"):
            with self.subTest(missing=key):
                self.pre_ro_failure({}, (key,))
            for value in (0, 1, None, "true"):
                with self.subTest(key=key, value=value):
                    self.pre_ro_failure({key: value})

    def test_ro_remount_success_with_recovery_still_set_rejects(self):
        # Model ext4 reconfigure returning successfully despite a failed recovery
        # completion. RO flags and syncfs success cannot replace the clear bit.
        backend = self.phase_failure("FILESYSTEM_RO", "filesystem_ro", {"recovery_required": True})
        self.assertEqual(backend.system.calls[-1], ("observe:filesystem_ro",))
        self.assertFalse(any(c[0] == "block_ro" for c in backend.system.calls))

    def test_ro_remount_unclean_journal_rejects(self):
        backend = self.phase_failure("FILESYSTEM_RO", "filesystem_ro", {"journal_clean": False})
        self.assertFalse(any(c[0] == "block_ro" for c in backend.system.calls))

    def test_quiesced_phases_and_restored_active_rw_pass(self):
        backend = self.make()
        result = self.run_backend(backend)
        self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
        for action in ("FILESYSTEM_RO", "DEVICE_RO", "PROTECTED", "REVALIDATE", "DEVICE_RW"):
            r = result["receipts"][action]
            self.assertIs(r["recovery_required"], False)
            self.assertIs(r["journal_clean"], True)
            self.assertIs(r["external_journal"], False)
            self.assertIs(r["healthy"], True)
            self.assertIs(r["mount_ro"], True)
            self.assertIs(r["superblock_ro"], True)
        rw = result["receipts"]["FILESYSTEM_RW"]
        self.assertIs(rw["filesystem_rw"], True)
        self.assertIs(rw["device_rw"], True)
        self.assertIs(rw["healthy"], True)
        self.assertIs(rw["external_journal"], False)
        self.assertIs(rw["recovery_required"], True)
        self.assertNotIn("journal_clean", rw)
        self.assertEqual(backend.done[-5:], ["SEAL", "DEVICE_RW", "FILESYSTEM_RW", "RESTORATION_SEAL", "RELEASE"])
        restoration = strict_json((backend.journal.root / "restoration.json").read_bytes())
        self.assertIs(restoration["device"]["journal_clean"], True)
        self.assertNotIn("journal_clean", restoration["filesystem"])

    def test_device_ro_requires_quiescence_before_block_ro(self):
        for key, value in (("recovery_required", True), ("journal_clean", False),
                           ("mount_ro", False), ("superblock_ro", False)):
            with self.subTest(key=key):
                backend = self.phase_failure("DEVICE_RO", "filesystem_ro", {key: value})
                self.assertFalse(any(c[0] == "block_ro" for c in backend.system.calls))

    def test_every_quiesced_readback_requires_all_phase_fields(self):
        for action, topic in (("FILESYSTEM_RO", "filesystem_ro"), ("DEVICE_RO", "device_ro"),
                              ("PROTECTED", "protected"), ("REVALIDATE", "protected"),
                              ("DEVICE_RW", "device_rw")):
            for key, value in (("healthy", False), ("external_journal", True),
                               ("recovery_required", True), ("journal_clean", False),
                               ("mount_ro", False), ("superblock_ro", False),
                               ("device_identity", {}), ("private_mount_id", 999)):
                with self.subTest(action=action, key=key):
                    self.phase_failure(action, topic, {key: value})
                    self.phase_failure(action, topic, {}, (key,))

    def test_device_rw_requires_quiescence_before_device_write_enable(self):
        for key, value in (("recovery_required", True), ("journal_clean", False)):
            with self.subTest(key=key):
                backend = self.phase_failure("DEVICE_RW", "protected", {key: value})
                self.assertNotIn(("block_ro", 40, False), backend.system.calls)

    def test_filesystem_rw_requires_quiescence_before_rw_remount(self):
        for key, value in (("recovery_required", True), ("journal_clean", False),
                           ("mount_ro", False), ("superblock_ro", False)):
            with self.subTest(key=key):
                backend = self.phase_failure("FILESYSTEM_RW", "device_rw", {key: value})
                self.assertFalse(any(c[0] == "mount" and c[2] == lb.MOUNT
                                     and not c[4] & lb.MS_RDONLY for c in backend.system.calls))

    def test_restored_rw_recovery_clear_rejects(self):
        self.phase_failure("FILESYSTEM_RW", "filesystem_rw", {"recovery_required": False})

    def test_restored_rw_any_journal_clean_field_rejects(self):
        for value in (True, False, None, 1, "true"):
            with self.subTest(value=value):
                self.phase_failure("FILESYSTEM_RW", "filesystem_rw", {"journal_clean": value})

    def test_restored_rw_requires_identity_health_and_internal_journal(self):
        for key, value in (("healthy", False), ("external_journal", True),
                           ("device_identity", {}), ("private_mount_id", 999),
                           ("filesystem_rw", False), ("device_rw", False)):
            with self.subTest(key=key):
                self.phase_failure("FILESYSTEM_RW", "filesystem_rw", {key: value})
                self.phase_failure("FILESYSTEM_RW", "filesystem_rw", {}, (key,))

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

    def test_namespace_uses_one_combined_mount_and_network_unshare(self):
        backend = self.make()
        backend.backing = copy.deepcopy(backend.system.backing)
        result = backend._perform("NAMESPACE", {"receipts": {}})
        self.assertEqual(lb.CLONE_NEWNET, 0x40000000)
        self.assertEqual(backend.system.calls, [
            ("observe:roles",), ("unshare", lb.CLONE_NEWNS | lb.CLONE_NEWNET),
            ("mount", None, "/", None, lb.MS_REC | lb.MS_PRIVATE, None), ("observe:namespace",)])
        self.assertIs(result["private_network_namespace"], True)
        self.assertIs(result["private_mount_namespace"], True)
        for field, value in (("private_network_namespace", False), ("private_mount_namespace", False),
                             ("network_namespace_id", "synthetic-original-net"),
                             ("network_namespace_id", ""), ("mount_namespace_id", "wrong"),
                             ("original_mount_namespace_id", "wrong")):
            with self.subTest(field=field, value=value):
                bad = self.make()
                bad.backing = copy.deepcopy(bad.system.backing)
                bad.system.observations["namespace"][field] = value
                with self.assertRaises(InvalidEvidence):
                    bad._perform("NAMESPACE", {"receipts": {}})
        native = object.__new__(lb.NativePrimitives)
        native.context = (None, None, "synthetic", "p1v-fixture", {})
        native.current_action = lambda: "NAMESPACE"
        libc = SimpleNamespace(unshare=Mock(return_value=0))
        with patch.object(lb, "require_adapter_grant"), patch("os.geteuid", return_value=0), \
                patch.object(native, "_libc", return_value=libc):
            for flags in (0, True, float(lb.CLONE_NEWNS | lb.CLONE_NEWNET), lb.CLONE_NEWNS, lb.CLONE_NEWNET,
                          lb.CLONE_NEWNS | lb.CLONE_NEWNET | 0x10000000):
                with self.subTest(flags=flags), self.assertRaises(InvalidEvidence):
                    native.unshare(flags)
            libc.unshare.assert_not_called()
            native.unshare(lb.CLONE_NEWNS | lb.CLONE_NEWNET)
            libc.unshare.assert_called_once_with(lb.CLONE_NEWNS | lb.CLONE_NEWNET)
            native.current_action = lambda: "CONFINEMENT"
            with self.assertRaisesRegex(InvalidEvidence, "outside phase"):
                native.unshare(lb.CLONE_NEWNS | lb.CLONE_NEWNET)
            libc.unshare.assert_called_once()

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
        self.assertEqual([c for c in calls if c[0] == "unshare"], [("unshare", lb.CLONE_NEWNS | lb.CLONE_NEWNET)])
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

    def test_confinement_requires_strict_private_network_namespace(self):
        receipt = confinement_fixture()
        namespace = {"namespace_id": "synthetic-ns"}
        lb.validate_confinement(receipt, launch_fixture(), namespace)
        self.assertIs(receipt["private_network_namespace"], True)
        for value in (False, None, 1, "true"):
            with self.subTest(value=value), self.assertRaises(InvalidEvidence):
                lb.validate_confinement(dict(receipt, private_network_namespace=value), launch_fixture(), namespace)
        for field in receipt:
            bad = copy.deepcopy(receipt)
            del bad[field]
            with self.subTest(missing=field), self.assertRaises(InvalidEvidence):
                lb.validate_confinement(bad, launch_fixture(), namespace)

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
        for issue in (215, 216, 217, 218, 219, 220, 221, 222):
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
        for issue in (215, 216, 217, 218, 219, 220, 221, 222):
            scope = {"authority_issue": issue, "host_id": "synthetic", "attempt": "p1v-fixture",
                     "operations": [a.name for a in plan()], "adapter_manifest_sha256": "a" * 64, "storage_identity": storage_fixture()}
            # All supplied pins match: the issue boundary itself must reject.
            with self.subTest(issue=issue), self.assertRaisesRegex(InvalidEvidence, "follow #230"):
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
        scope = {"authority_issue": 231, "host_id": "synthetic", "attempt": "p1v-fixture",
                 "operations": [a.name for a in plan()], "adapter_manifest_sha256": "a" * 64, "storage_identity": storage_fixture()}
        with self.assertRaisesRegex(InvalidEvidence, "external adapter grant required"):
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
                 'adapter_manifest_sha256': digest(canonical_json(adapter_fixture())), 'storage_identity': storage_fixture()}
        # Fake serialized input only; never an ExecutionGrant instance.
        class StoredScope:
            _scope = canonical_json(scope)

        with self.assertRaisesRegex(InvalidEvidence, 'same future issue'):
            lb.ExecutionGrant.check(StoredScope(), authorization, 'synthetic', 'p1v-fixture',
                                    adapter_fixture(), 'P1R_PARENT')
        receipt = {'host_id': 'synthetic', 'execution_grant': scope, 'adapter_manifest': adapter_fixture()}
        with self.assertRaisesRegex(InvalidEvidence, 'postdate implementation'):
            lb.validate_execution_scope(receipt, {'mode': 'LIVE', 'attempt': 'p1v-fixture'})


class RlimitCompatibilityTests(unittest.TestCase):
    FROZEN_LIMITS = ("CPU", "FSIZE", "DATA", "STACK", "CORE", "RSS", "NPROC", "NOFILE",
                     "MEMLOCK", "AS", "LOCKS", "SIGPENDING", "MSGQUEUE", "NICE", "RTPRIO", "RTTIME")

    def test_frozen_manifest_still_requires_locks_in_exact_position(self):
        self.assertEqual(lb.RLIMITS, self.FROZEN_LIMITS)
        baseline = launch_fixture()
        self.assertEqual(set(lb.LaunchManifest.parse(baseline).values()["rlimits"]), set(self.FROZEN_LIMITS))
        del baseline["rlimits"]["LOCKS"]
        with self.assertRaisesRegex(InvalidEvidence, "all Linux RLIMITs required"):
            lb.LaunchManifest.parse(baseline)

    def test_ordinary_limits_use_exported_integers_without_numbering_guesses(self):
        for index, name in enumerate(self.FROZEN_LIMITS):
            if name == "LOCKS":
                continue
            # Deliberately unlike Linux numbering, including the historic 5-9 range.
            exported = 100 + index
            resource = SimpleNamespace(**{"RLIMIT_" + name: exported})
            for platform in ("linux", "darwin"):
                with self.subTest(name=name, platform=platform):
                    self.assertIs(lb.linux_rlimit_id(resource, name, platform), exported)

    def test_missing_locks_on_linux_resolves_to_integer_ten_without_mutation(self):
        resource = SimpleNamespace()
        result = lb.linux_rlimit_id(resource, "LOCKS", "linux")
        self.assertIs(type(result), int)
        self.assertEqual(result, 10)
        self.assertEqual(vars(resource), {})

    def test_exported_locks_integer_ten_is_accepted_without_mutation(self):
        resource = SimpleNamespace(RLIMIT_LOCKS=10)
        for platform in ("linux", "darwin"):
            self.assertEqual(lb.linux_rlimit_id(resource, "LOCKS", platform), 10)
        self.assertEqual(vars(resource), {"RLIMIT_LOCKS": 10})

    def test_wrong_or_noninteger_exported_locks_rejects(self):
        for value in (-1, 0, 9, 11, 10.0, "10", None, True, False):
            resource = SimpleNamespace(RLIMIT_LOCKS=value)
            for platform in ("linux", "darwin"):
                with self.subTest(value=value, platform=platform), self.assertRaises(InvalidEvidence):
                    lb.linux_rlimit_id(resource, "LOCKS", platform)
            self.assertEqual(vars(resource), {"RLIMIT_LOCKS": value})

    def test_missing_locks_on_nonlinux_fails_closed(self):
        for platform in ("darwin", "win32", "freebsd14", "linux2", ""):
            with self.subTest(platform=platform), self.assertRaisesRegex(InvalidEvidence, "missing RLIMIT symbol"):
                lb.linux_rlimit_id(SimpleNamespace(), "LOCKS", platform)

    def test_every_other_missing_symbol_fails_closed(self):
        for name in self.FROZEN_LIMITS:
            if name != "LOCKS":
                with self.subTest(name=name), self.assertRaisesRegex(InvalidEvidence, "missing RLIMIT symbol"):
                    lb.linux_rlimit_id(SimpleNamespace(), name, "linux")

    def test_noninteger_ordinary_export_and_unknown_name_reject(self):
        for value in (True, 5.0, "5", None):
            with self.subTest(value=value), self.assertRaisesRegex(InvalidEvidence, "must be integer"):
                lb.linux_rlimit_id(SimpleNamespace(RLIMIT_CPU=value), "CPU", "linux")
        with self.assertRaisesRegex(InvalidEvidence, "unknown frozen RLIMIT name"):
            lb.linux_rlimit_id(SimpleNamespace(RLIMIT_UNKNOWN=10), "UNKNOWN", "linux")

    def handoff_fixture(self, mismatch=None):
        """Run the real unbound method with no native receiver or host interfaces."""
        baseline = launch_fixture()
        baseline["rlimits"] = {name: [800 + i, 900 + i] for i, name in enumerate(self.FROZEN_LIMITS)}
        baseline["rlimits"]["CPU"] = ["INFINITY", "INFINITY"]
        baseline["rlimits"]["FSIZE"][1] = "INFINITY"
        baseline["signals"] = {"mask": [], "dispositions": {"1": "DEFAULT", "2": "IGNORE"}}
        identifiers = {name: 10 if name == "LOCKS" else 100 + i for i, name in enumerate(self.FROZEN_LIMITS)}
        pairs = {identifiers[name]: tuple(-1 if v == "INFINITY" else v for v in pair)
                 for name, pair in baseline["rlimits"].items()}
        if mismatch is not None:
            pairs[identifiers[mismatch]] = (0, 0)
        resource = SimpleNamespace(RLIM_INFINITY=-1, getrlimit=Mock(side_effect=pairs.__getitem__),
                                   **{"RLIMIT_" + name: identifier for name, identifier in identifiers.items()
                                      if name != "LOCKS"})
        original_resource = vars(resource).copy()
        fake_os = SimpleNamespace(
            geteuid=Mock(return_value=0), sched_getaffinity=Mock(return_value={0, 1}),
            getpriority=Mock(return_value=0), PRIO_PROCESS=0,
            setgroups=Mock(), setresgid=Mock(), setresuid=Mock(), umask=Mock(), chdir=Mock(),
            listdir=Mock(return_value=["0", "1", "2", "40", "41"]), closerange=Mock(),
            kill=Mock(), execve=Mock())
        signals = SimpleNamespace(valid_signals=Mock(return_value={1, 2, 9, 19}), SIGKILL=9, SIGSTOP=19,
                                  SIG_DFL=0, SIG_IGN=1, SIG_SETMASK=2, signal=Mock(), pthread_sigmask=Mock())
        libc = SimpleNamespace(prctl=Mock(return_value=0), capset=Mock(return_value=0))
        child = SimpleNamespace(_check=Mock(), _prctl=Mock())
        path = Mock()
        path.return_value.read_text.return_value = "2\n"
        encoded = lb.LaunchManifest.parse(baseline).encoded
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"resource": resource, "signal": signals}))
            stack.enter_context(patch.object(sys, "platform", "linux"))
            stack.enter_context(patch.object(lb, "os", fake_os))
            stack.enter_context(patch.object(lb, "Path", path))
            loader = stack.enter_context(patch("ctypes.CDLL", return_value=libc))
            if mismatch is None:
                lb.NativeProcesses.apply_launch_credentials_and_close_fds(child, encoded)
            else:
                with self.assertRaisesRegex(InvalidEvidence, "inherited RLIMIT drift"):
                    lb.NativeProcesses.apply_launch_credentials_and_close_fds(child, encoded)
        self.assertEqual(vars(resource), original_resource)
        self.assertFalse(hasattr(resource, "RLIMIT_LOCKS"))
        child._check.assert_called_once_with("CONFINEMENT")
        fake_os.kill.assert_not_called()
        fake_os.execve.assert_not_called()
        return SimpleNamespace(resource=resource, os=fake_os, signals=signals, libc=libc, child=child,
                               loader=loader, path=path, identifiers=identifiers)

    def test_stock_cpython_unbound_handoff_checks_all_sixteen_in_frozen_order(self):
        fixture = self.handoff_fixture()
        self.assertEqual(fixture.resource.getrlimit.call_args_list,
                         [call(fixture.identifiers[name]) for name in self.FROZEN_LIMITS])
        locks_argument = fixture.resource.getrlimit.call_args_list[10].args[0]
        self.assertIs(type(locks_argument), int)
        self.assertEqual(locks_argument, 10)
        fixture.os.sched_getaffinity.assert_called_once_with(0)
        fixture.os.getpriority.assert_called_once_with(0, 0)
        fixture.loader.assert_called_once_with(None, use_errno=True)
        fixture.libc.prctl.assert_called_once_with(21, 0, 0, 0, 0)
        self.assertEqual(fixture.child._prctl.call_args_list,
                         [call(fixture.libc, 38, 1), call(fixture.libc, 47, 4),
                          *[call(fixture.libc, 24, cap) for cap in range(3)], call(fixture.libc, 8, 0)])
        fixture.os.setgroups.assert_called_once_with([1000])
        fixture.os.setresgid.assert_called_once_with(1000, 1000, 1000)
        fixture.os.setresuid.assert_called_once_with(1000, 1000, 1000)
        fixture.libc.capset.assert_called_once()
        self.assertEqual(fixture.signals.signal.call_args_list, [call(1, 0), call(2, 1)])
        fixture.signals.pthread_sigmask.assert_called_once_with(2, [])
        fixture.os.closerange.assert_called_once_with(3, 42)

    def assert_rejected_before_mutation(self, name):
        fixture = self.handoff_fixture(mismatch=name)
        index = self.FROZEN_LIMITS.index(name)
        self.assertEqual(fixture.resource.getrlimit.call_args_list,
                         [call(fixture.identifiers[n]) for n in self.FROZEN_LIMITS[:index + 1]])
        for interface in (fixture.os, fixture.signals, fixture.libc):
            for field, mock in vars(interface).items():
                if isinstance(mock, Mock) and field != "geteuid":
                    mock.assert_not_called()
        fixture.loader.assert_not_called()
        fixture.child._prctl.assert_not_called()
        fixture.path.assert_not_called()

    def test_locks_value_mismatch_rejects_before_credentials_fds_or_stop(self):
        self.assert_rejected_before_mutation("LOCKS")

    def test_each_other_inherited_pair_mismatch_still_rejects_before_mutation(self):
        for name in self.FROZEN_LIMITS:
            if name != "LOCKS":
                with self.subTest(name=name):
                    self.assert_rejected_before_mutation(name)

    def test_no_setrlimit_or_resource_monkeypatch_in_adapter(self):
        source = Path(lb.__file__).read_text()
        self.assertNotIn("setrlimit", source)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.assertNotIn(ast.unparse(node.value), ("resource", "resource_module"))
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("setattr", "delattr"):
                self.assertNotIn(ast.unparse(node.args[0]), ("resource", "resource_module"))

    def test_two_path_p1v_scope_preserves_all_other_files(self):
        assert_package_scope(self, Path(lb.__file__).parent)

    def test_220_rejected_on_grant_recheck_and_live_receipt(self):
        token = b"synthetic legacy seam only" * 2
        manifest = adapter_fixture()
        for issue in (219, 220, 221, 222):
            authorization = FutureAuthorization(token, digest(token), issue)
            scope = {"authority_issue": issue, "host_id": "synthetic", "attempt": "p1v-fixture",
                     "operations": [a.name for a in plan()],
                     "adapter_manifest_sha256": digest(canonical_json(manifest)), "storage_identity": storage_fixture()}
            stored = SimpleNamespace(_scope=canonical_json(scope))  # Never a live ExecutionGrant.
            with self.subTest(issue=issue):
                with self.assertRaisesRegex(InvalidEvidence, "same future issue"):
                    lb.ExecutionGrant.check(stored, authorization, "synthetic", "p1v-fixture", manifest, "P1R_PARENT")
                receipt = {"host_id": "synthetic", "execution_grant": scope, "adapter_manifest": manifest}
                with self.assertRaisesRegex(InvalidEvidence, "postdate implementation"):
                    lb.validate_execution_scope(receipt, {"mode": "LIVE", "attempt": "p1v-fixture"})

    def test_later_authorities_must_match_without_issuing_live_grant(self):
        token = b"synthetic legacy seam only" * 2
        manifest = adapter_fixture()
        scope = {"authority_issue": 231, "host_id": "synthetic", "attempt": "p1v-fixture",
                 "operations": [a.name for a in plan()],
                 "adapter_manifest_sha256": digest(canonical_json(manifest)), "storage_identity": storage_fixture()}
        stored = SimpleNamespace(_scope=canonical_json(scope))
        for issue in (220, 221, 222, 224):
            with self.subTest(issue=issue), self.assertRaisesRegex(InvalidEvidence, "same future issue"):
                lb.ExecutionGrant.check(stored, FutureAuthorization(token, digest(token), issue),
                                        "synthetic", "p1v-fixture", manifest, "P1R_PARENT")
        # The pure unbound check accepts matching later scope; this fake receiver
        # still cannot pass require_adapter_grant's exact ExecutionGrant type gate.
        authorization = FutureAuthorization(token, digest(token), 231)
        lb.ExecutionGrant.check(stored, authorization, "synthetic", "p1v-fixture", manifest, "P1R_PARENT")
        with self.assertRaisesRegex(InvalidEvidence, "adapter-specific external execution grant required"):
            lb.require_adapter_grant(authorization, stored, "synthetic", "p1v-fixture", manifest, "P1R_PARENT")


class PackageScopeWitnessTests(unittest.TestCase):
    """Package expansion is synthetic; no successor is created in the checkout."""

    def package_fixture(self, successors=()):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = Path(lb.__file__).parent
        for name in REQUIRED_P1V_FILES:
            (root / name).write_bytes((source / name).read_bytes())
        for name in successors:
            (root / name).write_bytes(b"# Synthetic package-scope fixture only.\n")
        return root

    def test_baseline_seven_files_pass_without_optional_successors(self):
        root = self.package_fixture()
        self.assertEqual({p.name for p in root.iterdir()}, REQUIRED_P1V_FILES)
        self.assertEqual(len(REQUIRED_P1V_FILES), 7)
        self.assertFalse(any((root / name).exists() for name in OPTIONAL_P1V5_FILES))
        assert_package_scope(self, root)

    def test_each_authorized_successor_individually_passes(self):
        for name in sorted(OPTIONAL_P1V5_FILES):
            with self.subTest(successor=name):
                assert_package_scope(self, self.package_fixture((name,)))

    def test_all_four_authorized_successors_together_pass(self):
        root = self.package_fixture(OPTIONAL_P1V5_FILES)
        self.assertEqual(len(list(root.iterdir())), 11)
        assert_package_scope(self, root)

    def test_all_sixteen_optional_successor_subsets_pass(self):
        successors = sorted(OPTIONAL_P1V5_FILES)
        checked = 0
        for count in range(len(successors) + 1):
            for subset in combinations(successors, count):
                with self.subTest(successors=subset):
                    assert_package_scope(self, self.package_fixture(subset))
                checked += 1
        self.assertEqual(checked, 16)

    def test_unknown_hidden_backup_and_generated_regular_files_rejected(self):
        for name in ("unrelated.py", "README.txt", ".hidden", ".hidden.py",
                     "linux_backend.py.bak", "test_linux_backend.py~", "generated.pyc"):
            with self.subTest(filename=name):
                root = self.package_fixture(OPTIONAL_P1V5_FILES)
                (root / name).write_bytes(b"unauthorized")
                with self.assertRaisesRegex(AssertionError, "unauthorized regular files"):
                    assert_package_scope(self, root)

    def test_typo_and_alternate_successor_names_rejected(self):
        for name in ("live_systen.py", "test_live_systen.py", "linux_gates.py",
                     "test_linux_gates.py", "Live_system.py", "linux_gate.py.bak"):
            with self.subTest(filename=name):
                root = self.package_fixture((name,))
                with self.assertRaisesRegex(AssertionError, "unauthorized regular files"):
                    assert_package_scope(self, root)

    def test_deleting_each_required_file_rejected_even_with_all_successors(self):
        for name in sorted(REQUIRED_P1V_FILES):
            with self.subTest(filename=name):
                root = self.package_fixture(OPTIONAL_P1V5_FILES)
                (root / name).unlink()
                with self.assertRaisesRegex(AssertionError, "missing required regular files"):
                    assert_package_scope(self, root)

    def test_required_filename_as_directory_does_not_satisfy_witness(self):
        root = self.package_fixture()
        (root / "test_linux_backend.py").unlink()
        (root / "test_linux_backend.py").mkdir()
        with self.assertRaisesRegex(AssertionError, "missing required regular files"):
            assert_package_scope(self, root)

    def test_each_protected_file_mutation_fails_its_hash_check(self):
        self.assertEqual(len(PROTECTED_P1V_FILES), 6)
        for name in sorted(PROTECTED_P1V_FILES):
            with self.subTest(filename=name):
                root = self.package_fixture(OPTIONAL_P1V5_FILES)
                path = root / name
                path.write_bytes(path.read_bytes() + b"\n# synthetic mutation\n")
                with self.assertRaisesRegex(AssertionError, "protected SHA256: " + name):
                    assert_package_scope(self, root)

    def test_authorized_witness_does_not_pin_its_own_bytes(self):
        root = self.package_fixture()
        (root / "test_linux_backend.py").write_bytes(b"# Authorized witness change.\n")
        assert_package_scope(self, root)



class ExternalStorageAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = encode_raw(raw_fixture())

    def make(self, observed, granted=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        content = fake_content(observed)
        system = FixtureSystem(self.raw, content, observed)
        authorization, grant = execution_fixture(observed if granted is None else granted)
        backend = lb.LinuxBackend(system, lb.DurableJournal(directory.name, "p1v-fixture", fixture=directory),
                                  launch_fixture(), adapter_fixture(), closure_fixture(), host_id="synthetic-host",
                                  attempt="p1v-fixture", authorization=authorization, grant=grant,
                                  fixture_reader_factory=lambda: FakeReader(observed))
        return backend

    def test_complete_adapter_content_restore_pipeline_for_both_grants(self):
        from .content_hasher import verify_content
        for second in (False, True):
            storage = storage_fixture(second)
            backend = self.make(storage)
            result = Supervisor(backend, "p1v-fixture").run()
            self.assertEqual(result["status"], "PASS_SIMULATION", result.get("failure"))
            for phase in ("TOPOLOGY", "FILESYSTEM_RO", "DEVICE_RO", "PROTECTED", "REVALIDATE", "DEVICE_RW", "FILESYSTEM_RW"):
                self.assertEqual(result["receipts"][phase]["device_identity"], storage)
            self.assertEqual(backend.grant.storage_identity, backend.storage_identity)
            self.assertEqual(result["receipts"]["AUTHORITY"]["execution_grant"]["storage_identity"], storage)
            content = result["receipts"]["HASH"]["content"]
            verify_content(content, storage)
            with self.assertRaises(InvalidEvidence):
                verify_content(content, storage_fixture(not second))
            self.assertEqual([c for c in backend.system.calls if c[0] == "block_ro"],
                             [("block_ro", 40, True), ("block_ro", 40, False)])

    def test_old_grant_new_storage_and_new_grant_old_storage_reject(self):
        for second in (False, True):
            backend = self.make(storage_fixture(second), storage_fixture(not second))
            result = Supervisor(backend, "p1v-fixture").run()
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["failure"]["action"], "TOPOLOGY")
            self.assertEqual(backend.storage_identity, storage_fixture(not second))
            self.assertFalse(any(c[0] == "parent_once" for c in backend.system.calls))

    def test_every_identity_coordinate_drift_rejects_through_restoration(self):
        for second in (False, True):
            storage, other = storage_fixture(second), storage_fixture(not second)
            for phase in ("filesystem_ro", "device_ro", "protected", "device_rw", "filesystem_rw"):
                for field in storage:
                    with self.subTest(second=second, phase=phase, field=field):
                        backend = self.make(storage)
                        receipt = backend.system.observations[phase]
                        self.assertIn("device_identity", receipt)
                        replacement = other[field] if other[field] != storage[field] else ["/dev/disk/by-id/google-local-ssd-1"]
                        receipt["device_identity"][field] = replacement
                        result = Supervisor(backend, "p1v-fixture").run()
                        self.assertEqual(result["status"], "FAIL")
                        self.assertEqual(backend.storage_identity, storage)

    def test_backend_hash_passes_identical_grant_object_to_reader_and_hasher(self):
        from . import content_hasher as ch
        backend = self.make(storage_fixture(True))
        evidence = {"receipts": backend.system.receipts}
        backend.done = ["P1R_PARENT", "SHUTDOWN", "REVALIDATE"]
        backend.simulated = False
        reader = Mock()
        with patch.object(backend, "_protected"), patch.object(ch, "LinuxDirectReader", return_value=reader) as factory, \
             patch.object(lb, "hash_content", return_value={}) as hashed:
            backend._perform("HASH", evidence)
        self.assertIs(factory.call_args.args[2], hashed.call_args.args[3])
        self.assertEqual(factory.call_args.args[2], backend.grant.storage_identity)

    def test_execution_grant_storage_is_frozen_and_constructor_validates(self):
        from dataclasses import FrozenInstanceError
        for second in (False, True):
            storage = storage_fixture(second)
            _, grant = execution_fixture(storage)
            storage["by_id_aliases"].append("bad")
            grant.storage_identity["major_minor"] = "9:9"
            self.assertEqual(grant.storage_identity, storage_fixture(second))
            with self.assertRaises(FrozenInstanceError):
                grant._scope = b"{}"
            with self.assertRaises(InvalidEvidence):
                execution_fixture(storage)

    def native_retention(self, second, variant=None):
        import stat
        storage = storage_fixture(second)
        authorization, grant = execution_fixture(storage)
        native = lb.NativePrimitives(authorization, grant, "synthetic-host", "p1v-fixture", adapter_fixture(), lambda: "TOPOLOGY")
        major, minor = map(int, storage["major_minor"].split(":"))
        rdev = (major << 32) | minor
        block = SimpleNamespace(st_mode=stat.S_IFBLK | 0o600, st_rdev=rdev)
        filesystem = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=rdev)
        mount = {"target": lb.MOUNT, "major_minor": storage["major_minor"], "fstype": "ext4", "root": "/"}
        path = copy.copy(block)
        if variant == "symlink": path.st_mode = stat.S_IFLNK
        if variant == "filesystem": filesystem.st_dev += 1
        if variant == "rdev": block.st_rdev += 1
        if variant == "mount": mount["major_minor"] = "1:2"
        def follow(alias):
            return SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=rdev + int(variant == "alias"))
        aliases = [p.rsplit("/", 1)[1] for p in storage["by_id_aliases"]]
        if variant == "extra_alias": aliases.append("google-local-ssd-1")
        with ExitStack() as stack:
            stack.enter_context(patch.object(os, "geteuid", return_value=0))
            opened = stack.enter_context(patch.object(os, "open", side_effect=[40, 41, 42]))
            stack.enter_context(patch.object(os, "fstat", side_effect=lambda fd: block if fd == 40 else filesystem))
            stack.enter_context(patch.object(os, "lstat", return_value=path))
            stack.enter_context(patch.object(os, "stat", side_effect=follow))
            stack.enter_context(patch.object(os, "listdir", return_value=aliases))
            stack.enter_context(patch.object(os, "major", side_effect=lambda n: n >> 32))
            stack.enter_context(patch.object(os, "minor", side_effect=lambda n: n & 0xffffffff))
            stack.enter_context(patch.object(native, "mount_snapshot", return_value=[mount]))
            result = native.retain_objects()
            self.assertEqual(opened.call_args_list[0].args[0], storage["device_path"])
            self.assertTrue(opened.call_args_list[0].args[1] & os.O_NOFOLLOW)
            with self.assertRaisesRegex(InvalidEvidence, "reuse"):
                native.retain_objects()
            with self.assertRaisesRegex(InvalidEvidence, "retained block"):
                native._device(99)
            self.assertEqual(opened.call_count, 3)
            return result

    def test_native_retains_granted_path_rdev_aliases_mount_for_both_storage(self):
        for second in (False, True):
            self.assertEqual(self.native_retention(second), {"device": 40, "filesystem": 41, "host_mount_ns": 42})
            for variant in ("symlink", "filesystem", "rdev", "mount", "alias", "extra_alias"):
                with self.subTest(second=second, variant=variant), self.assertRaises(InvalidEvidence):
                    self.native_retention(second, variant)


if __name__ == "__main__":
    unittest.main()
