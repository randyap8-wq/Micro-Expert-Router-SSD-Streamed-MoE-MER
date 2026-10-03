"""Software-only P1V5 tests. No valid external live grant is created."""

import ast
import builtins
import copy
import ctypes
import errno
import importlib
import os
import stat
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import InvalidEvidence, canonical_json, digest, strict_json
from . import live_system as ls
from . import linux_backend as lb
from .test_p1v import launch_fixture, adapter_fixture, closure_fixture, storage_fixture


def mask(*nodes):
    data = bytearray(128)
    for node in nodes:
        data[node // 8] |= 1 << (node % 8)
    return bytes(data)


def numa_baseline():
    value = launch_fixture()
    value["credentials"]["groups"] = []
    value["numa"] = {"allowed_nodes": [0], "memory_policy": ls.task_policy(0, mask())}
    return value


def snapshot_fixture():
    baseline = numa_baseline()
    values = {"attempt": "p1v-fixture", "host_id": "fixture-host",
              "worker": {"pid": 40, "starttime": 900, "threads": 1}, "kernel": ls.NUMA_ABI,
              "policy": baseline["numa"]["memory_policy"], "allowed": [0],
              "source": baseline["resource_domain"], "target": copy.deepcopy(baseline["resource_domain"]),
              "hierarchy": "cgroup2", "exclusion": True, "baseline": baseline}
    return values


def child_fixture():
    v = snapshot_fixture()
    return {"attempt": v["attempt"], "host_id": v["host_id"], "worker": v["worker"], "returned_pid": 41,
            "child": {"pid": 41, "ppid": 40, "threads": 1, "stopped": True,
                      "allowed_mems": [0], "allowed_cpus": [0, 1]},
            "source": v["source"], "target": v["target"], "baseline": v["baseline"]}


def superblock(recovery=True):
    import uuid
    data = bytearray(1024)
    struct.pack_into("<H", data, 0x38, 0xEF53)
    struct.pack_into("<H", data, 0x3A, 1)
    struct.pack_into("<I", data, 0x5C, 4)
    struct.pack_into("<I", data, 0x60, 4 if recovery else 0)
    struct.pack_into("<I", data, 0xE0, 8)
    data[0x68:0x78] = uuid.UUID(storage_fixture()["uuid"]).bytes
    return bytes(data)


def bundle_fixture():
    baseline = numa_baseline()
    closure = closure_fixture()
    adapter = adapter_fixture()
    adapter.update(module_sha256=ls.PROTECTED["linux_backend.py"], launch_manifest_sha256=lb.LaunchManifest.parse(baseline).sha256)
    profile = {"source_root": "/fixture/source", "dependency_files": {"/fixture/dependency": "a" * 64},
               "cgroup_parent": "/sys/fs/cgroup/fixture", "lock_path": "/fixture/lock",
               "trusted_evidence_root": "/fixture/evidence", "namespace_services": {}, "policy_files": {},
               "gpu_inventory_roots": {}, "kernel_source_profile": "Linux-v6.8",
               "administrative_noninterference": "TRUSTED_HOST_PREMISE"}
    profile["policy_files"] = {
        "cpu_policy": {name: ["/sys/fixture/" + name] for name in ("governors", "frequency_limits", "boost", "power")},
        "thp": {"enabled": ["/sys/fixture/thp"]},
        "numa_balancing": {"enabled": ["/proc/sys/kernel/numa_balancing"]}}
    profile["gpu_inventory_roots"] = {name: ["/fixture/" + name] for name in ("nodes", "libraries", "icds", "layers", "cache")}
    provider = {"schema": "mer.p1v.provider.v1", "implementation_issue": 217,
                "modules": {"live_system.py": "1" * 64, "linux_gate.py": "2" * 64,
                            "linux_backend.py": ls.PROTECTED["linux_backend.py"]},
                "enforcement_schema": lb.SCHEMA, "enforcement_profile": lb.PROFILE,
                "runtime": {"implementation": "cpython", "version": [3, 12, 3], "cache_tag": "cpython-312",
                            "machine": "x86_64", "byteorder": "little"},
                "dependencies": baseline["adapter_pins"], "launch_manifest_sha256": lb.LaunchManifest.parse(baseline).sha256,
                "execution_closure_sha256": digest(canonical_json(closure)), "numa_abi": copy.deepcopy(ls.NUMA_ABI),
                "nvml": {"path": "/fixture/libnvidia-ml.so", "sha256": "b" * 64, "uuid": "GPU-fixture",
                         "pci_bus_id": "00000000:00:01.0", "driver": "580.178.04", "version": "fixture-version",
                         "dedicated_host": True, "normalize_default_unlocked": True}, "host_profile": profile}
    return {"adapter": adapter, "provider": provider, "launch": baseline, "closure": closure}


class PolicyTests(unittest.TestCase):
    def test_encoding_deterministic_versioned_and_lossless(self):
        a = ls.task_policy(2 | ls.STATIC, mask(1, 73, 1023))
        self.assertEqual(a, ls.task_policy(2 | ls.STATIC, mask(1, 73, 1023)))
        v = strict_json(a)
        self.assertEqual(v["schema"], "mer.p1v.numa-task-policy.v1")
        self.assertEqual(v["nodemask"], [1, 73, 1023])
        self.assertEqual(v["mode_flags"], ls.STATIC)
        self.assertEqual(v["maxnode"], 1024)
        self.assertEqual(v["source"], "GET_MEMPOLICY_CURRENT_TASK")
        self.assertEqual(v["inheritance"], "FORK_EXEC")

    def test_default_distinct_from_local(self):
        self.assertNotEqual(ls.task_policy(0, mask()), ls.task_policy(4, mask()))

    def test_static_nodes_preserves_user_mask(self):
        self.assertEqual(strict_json(ls.task_policy(2 | ls.STATIC, mask(73)))["nodemask"], [73])

    def test_relative_nodes_preserves_user_mask(self):
        v = strict_json(ls.task_policy(3 | ls.RELATIVE, mask(1, 4)))
        self.assertEqual(v["mode_flags"], ls.RELATIVE)
        self.assertEqual(v["nodemask"], [1, 4])

    def test_user_nodemask_not_allowed_nodes(self):
        v = snapshot_fixture()
        v["policy"] = ls.task_policy(2 | ls.STATIC, mask(0, 73))
        v["baseline"]["numa"]["memory_policy"] = v["policy"]
        receipt = strict_json(ls.InheritanceSnapshot.capture(**v).encoded)
        self.assertEqual(receipt["allowed"], [0])
        self.assertEqual(receipt["abi_fields"]["nodemask"], [0, 73])

    def test_incompatible_flags_reject(self):
        with self.assertRaises(InvalidEvidence):
            ls.task_policy(2 | ls.STATIC | ls.RELATIVE, mask(0))

    def test_wrong_balancing_mode_rejects(self):
        with self.assertRaises(InvalidEvidence):
            ls.task_policy(3 | ls.BALANCING, mask(0))

    def test_direct_query_mocked_pinned_flags_zero(self):
        libc = SimpleNamespace(syscall=Mock())
        def query(number, mode, nodes, maxnode, address, flags):
            self.assertEqual((number.value, maxnode.value, address.value, flags.value), (239, 1024, None, 0))
            ctypes.cast(mode, ctypes.POINTER(ctypes.c_int))[0] = 2 | ls.RELATIVE
            nodes[0] = 10
            return 0
        libc.syscall.side_effect = query
        io = object.__new__(ls.DirectLinux)
        io._storage_identity = canonical_json(storage_fixture())
        io.check = lambda: None
        with patch.object(io, "libc", return_value=libc):
            value = strict_json(io.task_policy())
        self.assertEqual(value["nodemask"], [1, 3])
        self.assertEqual(value["mode_flags"], ls.RELATIVE)
        libc.syscall.assert_called_once()

    def test_direct_query_failure_rejects(self):
        io = object.__new__(ls.DirectLinux)
        io._storage_identity = canonical_json(storage_fixture())
        io.check = lambda: None
        with patch.object(io, "libc", return_value=SimpleNamespace(syscall=Mock(return_value=-1))):
            with self.assertRaisesRegex(InvalidEvidence, "syscall failure"):
                io.task_policy()

    def test_denied_query_before_library_load(self):
        io = object.__new__(ls.DirectLinux)
        io._storage_identity = canonical_json(storage_fixture())
        io.check = Mock(side_effect=InvalidEvidence("denied"))
        with patch("ctypes.CDLL", side_effect=AssertionError("live library load")):
            with self.assertRaisesRegex(InvalidEvidence, "denied"):
                io.task_policy()

    def test_valid_snapshot_child_certificate(self):
        snap = ls.InheritanceSnapshot.capture(**snapshot_fixture())
        self.assertEqual(snap.bind_child(**child_fixture()), numa_baseline()["numa"])

    def test_snapshot_immutable_copy(self):
        v = snapshot_fixture()
        snap = ls.InheritanceSnapshot.capture(**v)
        before = snap.encoded
        v["source"]["effective_mems"].append(1)
        self.assertEqual(snap.encoded, before)

    def test_snapshot_not_reused_by_provider(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.child, obj.snapshot, obj.snapshot_consumed = 41, Mock(), True
        with self.assertRaisesRegex(InvalidEvidence, "one exact child"):
            obj._observe_confinement()


def add_policy_tests():
    for mode in (1, 2, 3, 5):
        def test(self, mode=mode):
            self.assertEqual(strict_json(ls.task_policy(mode, mask(1, 9)))["nodemask"], [1, 9])
        setattr(PolicyTests, "test_exact_mask_" + ls.MPOL_NAMES[mode].lower(), test)
    cases = {"unsupported_mode": (6, mask()), "unsupported_flags": (0x1000 | 2, mask(0)),
             "negative_mode": (-1, mask()), "bool_mode": (True, mask()),
             "truncated_mask": (2, b"\1"), "overlong_mask": (2, bytes(129)),
             "wrong_mask_type": (2, [0]), "default_with_nodes": (0, mask(1))}
    for name, args in cases.items():
        def test(self, args=args):
            with self.assertRaises(InvalidEvidence):
                ls.task_policy(*args)
        setattr(PolicyTests, "test_" + name, test)
    variants = {"policy_mismatch": ("policy", ls.task_policy(4, mask())),
                "allowed_mismatch": ("allowed", [1]), "non_cgroup2": ("hierarchy", "cgroup"),
                "exclusion_missing": ("exclusion", False), "kernel_drift": ("kernel", {}),
                "worker_multithreaded": ("worker", {"pid": 40, "starttime": 900, "threads": 2})}
    for name, (key, value) in variants.items():
        def test(self, key=key, value=value):
            v = snapshot_fixture()
            v[key] = value
            with self.assertRaises(InvalidEvidence):
                ls.InheritanceSnapshot.capture(**v)
        setattr(PolicyTests, "test_prefork_" + name, test)
    for field in lb.DOMAIN_FIELDS:
        def test(self, field=field):
            v = snapshot_fixture()
            if field.startswith("effective"):
                v["target"][field] = [7]
            elif field in ("ancestor_constraints", "bpf_ancestry"):
                v["target"][field] = [{"unexpected": 1}]
            else:
                v["target"][field]["constraints"]["mutation"] = True
            with self.assertRaises(InvalidEvidence):
                ls.InheritanceSnapshot.capture(**v)
        setattr(PolicyTests, "test_no_rebind_" + field, test)
    cases = {"wrong_attempt": ("attempt", "p1v-other"), "wrong_host": ("host_id", "other"),
             "wrong_returned_pid": ("returned_pid", 42), "wrong_worker": ("worker", {"pid": 40, "starttime": 901, "threads": 1})}
    for name, (key, value) in cases.items():
        def test(self, key=key, value=value):
            snap = ls.InheritanceSnapshot.capture(**snapshot_fixture())
            child = child_fixture()
            child[key] = value
            with self.assertRaises(InvalidEvidence):
                snap.bind_child(**child)
        setattr(PolicyTests, "test_child_" + name, test)
    for key, value in {"pid": 42, "ppid": 42, "threads": 2, "stopped": False,
                       "allowed_mems": [1], "allowed_cpus": [1]}.items():
        def test(self, key=key, value=value):
            snap = ls.InheritanceSnapshot.capture(**snapshot_fixture())
            child = child_fixture()
            child["child"][key] = value
            with self.assertRaises(InvalidEvidence):
                snap.bind_child(**child)
        setattr(PolicyTests, "test_child_drift_" + key, test)


add_policy_tests()


class ParserTests(unittest.TestCase):
    def test_ext4_rw_and_ro_distinct(self):
        self.assertTrue(ls.ext4_superblock(superblock())["recovery_required"])
        self.assertFalse(ls.ext4_superblock(superblock(False))["recovery_required"])
        self.assertNotIn("journal_clean", ls.ext4_superblock(superblock()))

    def test_ext4_truncated(self):
        with self.assertRaises(InvalidEvidence):
            ls.ext4_superblock(superblock()[:-1])

    def test_ext4_bad_magic(self):
        data = bytearray(superblock())
        data[0x38] = 0
        with self.assertRaises(InvalidEvidence):
            ls.ext4_superblock(bytes(data))

    def test_ext4_recorded_errors(self):
        data = bytearray(superblock())
        struct.pack_into("<I", data, 0x194, 1)
        with self.assertRaises(InvalidEvidence):
            ls.ext4_superblock(bytes(data))

    def test_ext4_external_journal(self):
        data = bytearray(superblock())
        data[0xD0] = 1
        with self.assertRaises(InvalidEvidence):
            ls.ext4_superblock(bytes(data))

    def test_ext4_error_state(self):
        data = bytearray(superblock())
        data[0x3A] = 3
        with self.assertRaises(InvalidEvidence):
            ls.ext4_superblock(bytes(data))

    def test_ext4_checksum_failure(self):
        data = bytearray(superblock())
        struct.pack_into("<I", data, 0x64, 0x400)
        data[0x175] = 1
        with self.assertRaisesRegex(InvalidEvidence, "checksum"):
            ls.ext4_superblock(bytes(data))

    def test_mount_ro_not_superblock_ro(self):
        records = lb.parse_mountinfo("20 1 259:4 / /mnt/localssd ro,relatime - ext4 /dev/nvme0n1 rw\n")
        self.assertTrue(records[0]["mount_ro"])
        self.assertFalse(records[0]["superblock_ro"])

    def test_ambiguous_mount_rejected(self):
        records = [{"target": "/"}, {"target": "/"}]
        with self.assertRaises(InvalidEvidence):
            ls.mount_at(records, "/fixture")

    def test_lists_reject_overlap_reverse_and_malformed(self):
        self.assertEqual(ls.integer_list("0-2,4"), [0, 1, 2, 4])
        for value in ("", "2-1", "1,1", "0-2,2", "x", "1,0", "65536"):
            with self.subTest(value=value), self.assertRaises(InvalidEvidence):
                ls.integer_list(value)

    def test_stat_comm_parentheses(self):
        text = "40 (name ) with ( parentheses) T 10 20 30 " + "0 " * 15 + "99"
        result = ls.parse_stat(text)
        self.assertEqual((result["pid"], result["ppid"], result["starttime"]), (40, 10, 99))

    def test_malformed_status_rejected(self):
        with self.assertRaises(InvalidEvidence):
            ls.parse_status("Pid: 1\nPid: 2\n")


class NvmlTests(unittest.TestCase):
    def make_nvml(self):
        spec = bundle_fixture()["provider"]["nvml"]
        io = Mock()
        io.file_hash.return_value = spec["sha256"]
        obj = object.__new__(ls.PinnedNvml)
        obj.io, obj.spec, obj.check = io, spec, lambda: None
        obj.library = obj.handle = obj.certificate = None
        obj.identity = Mock(return_value={"device_uuid": spec["uuid"], "pci_bus_id": spec["pci_bus_id"],
            "driver_version": spec["driver"], "nvml_version": spec["version"],
            "device_name": "NVIDIA L4", "nvml_library_sha256": spec["sha256"]})
        obj.query_policy = Mock(return_value={"application_clocks": {}, "offsets": []})
        obj._call = Mock(return_value=0)
        return obj

    def test_reset_order_and_certificate(self):
        obj = self.make_nvml()
        with patch("ctypes.CDLL", return_value=object()):
            certificate = obj.normalize()
        self.assertEqual([c.args[0] for c in obj._call.call_args_list],
                         ["nvmlInit_v2", "nvmlDeviceGetHandleByUUID", "nvmlDeviceResetGpuLockedClocks", "nvmlDeviceResetMemoryLockedClocks"])
        self.assertEqual(certificate["normalization"], "DEFAULT_UNLOCKED")
        self.assertEqual(certificate["schema"], "mer.p1v.gpu-clock-policy.v1")
        self.assertNotIn("current_frequency", certificate)

    def test_identity_before_resets(self):
        obj = self.make_nvml()
        obj.identity.return_value["device_uuid"] = "wrong"
        with patch("ctypes.CDLL", return_value=object()), self.assertRaises(InvalidEvidence):
            obj.normalize()
        self.assertNotIn("nvmlDeviceResetGpuLockedClocks", [c.args[0] for c in obj._call.call_args_list])

    def test_library_hash_before_load(self):
        obj = self.make_nvml()
        obj.io.file_hash.return_value = "0" * 64
        with patch("ctypes.CDLL", side_effect=AssertionError("library must not load")), self.assertRaises(InvalidEvidence):
            obj.normalize()

    def test_gpu_reset_failure_stops_memory_reset(self):
        obj = self.make_nvml()
        def call(name, *args):
            if name == "nvmlDeviceResetGpuLockedClocks":
                raise InvalidEvidence("reset failed")
        obj._call.side_effect = call
        with patch("ctypes.CDLL", return_value=object()), self.assertRaises(InvalidEvidence):
            obj.normalize()
        self.assertIsNone(obj.certificate)
        self.assertNotIn("nvmlDeviceResetMemoryLockedClocks", [c.args[0] for c in obj._call.call_args_list])

    def test_memory_reset_failure_has_no_certificate(self):
        obj = self.make_nvml()
        def call(name, *args):
            if name == "nvmlDeviceResetMemoryLockedClocks":
                raise InvalidEvidence("reset failed")
        obj._call.side_effect = call
        with patch("ctypes.CDLL", return_value=object()), self.assertRaises(InvalidEvidence):
            obj.normalize()
        self.assertIsNone(obj.certificate)

    def test_query_state_cannot_replace_transition(self):
        obj = self.make_nvml()
        with self.assertRaises(InvalidEvidence):
            obj.observed()
        obj.query_policy.assert_not_called()

    def test_denial_before_nvml(self):
        with patch("ctypes.CDLL", side_effect=AssertionError("real NVML forbidden")), self.assertRaises(InvalidEvidence):
            ls.PinnedNvml(Mock(), {}, Mock(side_effect=InvalidEvidence("denied")))

    def test_nondefault_application_policy_rejects(self):
        obj = self.make_nvml()
        obj._uint = Mock(side_effect=[{"status": "NVML_SUCCESS", "value": 10}, {"status": "NVML_SUCCESS", "value": 20}])
        with self.assertRaises(InvalidEvidence):
            ls.PinnedNvml.query_policy(obj)

    def test_documented_optional_unsupported(self):
        obj = self.make_nvml()
        obj._call.return_value = 3
        self.assertEqual(ls.PinnedNvml._uint(obj, "optional", optional=True), {"status": "NVML_ERROR_NOT_SUPPORTED"})

    def test_unknown_optional_failure_not_unsupported(self):
        obj = self.make_nvml()
        function = Mock(return_value=999)
        obj.library = SimpleNamespace(optional=function)
        with self.assertRaises(InvalidEvidence):
            ls.PinnedNvml._call(obj, "optional", [], [], True)

    def test_normalization_reuse_rejected(self):
        obj = self.make_nvml()
        obj.certificate = b"{}"
        with self.assertRaises(InvalidEvidence):
            obj.normalize()


class BoundaryTests(unittest.TestCase):
    def test_import_has_no_host_side_effect(self):
        with patch("os.open", side_effect=AssertionError("open")), patch("os.listdir", side_effect=AssertionError("listdir")), \
             patch("ctypes.CDLL", side_effect=AssertionError("library")), patch("os.fork", create=True, side_effect=AssertionError("fork")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("read")), patch.object(Path, "read_text", side_effect=AssertionError("read")):
            importlib.reload(ls)

    def test_current_authorities_cannot_grant(self):
        for issue in range(217, 229):
            scope = {"authority_issue": issue, "host_id": "fixture", "attempt": "p1v-fixture",
                     "operations": [a.name for a in plan_fixture()], "provider_manifest_sha256": "a" * 64, "storage_identity": storage_fixture()}
            with self.subTest(issue=issue), self.assertRaisesRegex(InvalidEvidence, "follow #244"):
                ls.ProviderGrant(scope, digest(canonical_json(scope)), b"x" * 32, digest(b"x" * 32))

    def test_constructor_denies_before_host_access(self):
        with patch("os.getpid", side_effect=AssertionError("host")), patch("ctypes.CDLL", side_effect=AssertionError("library")), \
             self.assertRaises(InvalidEvidence):
            ls.LiveSystem(authorization=None, grant=None, provider_grant=None, host_id="fixture", attempt="p1v-fixture",
                          manifest={}, provider_manifest={}, baseline={}, closure={}, journal=None, native=None,
                          processes=None, roles=None, bootstrap={})

    def test_all_topics_implemented_and_unknown_rejected(self):
        for topic in ls.TOPICS:
            self.assertTrue(callable(getattr(ls.LiveSystem, "_observe_" + topic)))
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj._check = lambda: None
        with self.assertRaises(InvalidEvidence):
            obj.observe("unknown")

    def test_every_provider_operation_refuses_during_parent(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.running, obj.failed = True, False
        for topic in ls.TOPICS:
            with self.subTest(topic=topic), self.assertRaisesRegex(InvalidEvidence, "parent"):
                obj.observe(topic)
        for name, args in (("acquire_exclusion", ()), ("raw_reports", ("p1v-fixture",)),
                           ("syncfs", (40,)), ("block_get", (40,)), ("release", ())):
            with self.subTest(name=name), self.assertRaisesRegex(InvalidEvidence, "parent"):
                getattr(obj, name)(*args)

    def test_reports_impossible_before_parent_return(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj._check, obj.returned = lambda: None, False
        with patch("os.open", side_effect=AssertionError("report opened")), self.assertRaises(InvalidEvidence):
            obj.raw_reports("p1v-fixture")

    def test_authority_has_no_launch_observed(self):
        tree = ast.parse(Path(ls.__file__).read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_observe_authority")
        strings = {n.value for n in ast.walk(function) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn("launch_observed", strings)

    def test_no_payload_operations_in_metadata(self):
        tree = ast.parse(Path(ls.__file__).read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_observe_metadata")
        forbidden = {"read", "pread", "mmap", "file_hash", "hash_content"}
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in ast.walk(method)))

    def test_no_forbidden_process_or_policy_api(self):
        for module in (ls, lb):
            tree = ast.parse(Path(module.__file__).read_text())
            calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertFalse(calls & {"system", "popen", "Popen", "run", "set_mempolicy", "mbind", "migrate_pages",
                                     "move_pages", "ptrace", "process_vm_writev", "pipe", "socketpair", "setrlimit"})
            self.assertNotIn("/numa_maps", Path(module.__file__).read_text())

    def test_direct_child_release_exec_and_guardian_wait(self):
        tree = ast.parse(Path(lb.__file__).read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeProcesses")
        for name, nextcall in (("stop_then_exec_parent", "execve"), ("resume_wait_once", "waitpid")):
            method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
            kill = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "kill")
            statement = method.body[kill + 1]
            call = statement.value
            self.assertEqual(call.func.attr, nextcall)

    def test_protected_hashes_and_package_witness(self):
        from .test_linux_backend import assert_package_scope
        root = Path(ls.__file__).parent
        assert_package_scope(self, root)
        for name, expected in ls.PROTECTED.items():
            self.assertEqual(digest((root / name).read_bytes()), expected)

    def test_raw_report_names_derived_from_frozen_source(self):
        source = Path(ls.__file__).parents[2] / "rust-engine/src/gpu_native_predictor_v2_critical_path_attribution.rs"
        text = source.read_text()
        self.assertIn('"{}.{}.json"', text)
        self.assertIn('path.with_extension("stdout.log")', text)
        self.assertIn('path.with_extension("stderr.log")', text)
        self.assertEqual(ls.REPORT_NAMES["control"], "p1r.json.control.json")
        self.assertEqual(ls.REPORT_NAMES["treatment"], "p1r.json.treatment.json")
        self.assertIn("p1r.json.control.stdout.log", ls.REPORT_LOGS)


def plan_fixture():
    from .supervisor import plan
    return plan("p1v-fixture")


class ProviderReceiptTests(unittest.TestCase):
    def base(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj._check = lambda: None
        obj._host = lambda: "fixture-boot"
        obj.attempt, obj.host_id = "p1v-fixture", "fixture-host"
        obj.baseline = lb.LaunchManifest.parse(numa_baseline())
        obj.exclusion, obj.running, obj.returned, obj.failed = True, False, False, False
        obj.snapshot_consumed = False
        obj.io = Mock()
        obj.worker_pid = 40
        obj._worker = Mock(return_value=snapshot_fixture()["worker"])
        obj.source_cgroup, obj.cgroup, obj.child = "/sys/fs/cgroup/fixture", "/sys/fs/cgroup/fixture/p1v-fixture", 41
        obj.child_identity = {"pid": 41, "starttime": 901, "threads": 1}
        obj.io.identity.return_value = obj.child_identity
        obj.io.text.return_value = ""
        obj.io.namespace.side_effect = lambda pid, kind: {"mnt": "mnt:private", "net": "net:private", "user": "user:initial"}[kind]
        obj.namespace_receipt = {"namespace_id": "mnt:private", "mount_namespace_id": "mnt:private",
                                 "network_namespace_id": "net:private"}
        obj.host_broker_inventory = []
        obj.io.text.side_effect = lambda path: ("Num RefCount Protocol Flags Type St Inode Path\n"
                                               if path.endswith("/net/unix") else obj.io.text.return_value)
        obj.journal = Mock()
        obj.initial_user_ns = "user:initial"
        obj._cgroup_path = Mock(return_value=obj.cgroup)
        obj._domain = Mock(return_value=numa_baseline()["resource_domain"])
        obj.snapshot = ls.InheritanceSnapshot.capture(**snapshot_fixture())
        obj._inventory = Mock(return_value={})
        obj._launch_observed = Mock(return_value=numa_baseline())
        obj.io.status.return_value = {"State": "T (stopped)", "PPid": "40", "Pid": "41", "Threads": "1",
            "Uid": "1000 1000 1000 1000", "Gid": "1000 1000 1000 1000", "Groups": "",
            "Cpus_allowed_list": "0-1", "Mems_allowed_list": "0", "NoNewPrivs": "1", "Seccomp": "0",
            "CapEff": "0", "CapPrm": "0", "CapInh": "0", "CapAmb": "0", "CapBnd": "0"}
        return obj

    def test_confinement_actual_child_receipt_and_consumption(self):
        obj = self.base()
        result = obj._observe_confinement()
        self.assertEqual(result["handoff_pid"], 41)
        self.assertEqual(result["launch_observed"]["numa"], numa_baseline()["numa"])
        self.assertTrue(obj.snapshot_consumed)
        with self.assertRaises(InvalidEvidence):
            obj._observe_confinement()

    def test_confinement_wrong_cgroup(self):
        obj = self.base()
        obj._cgroup_path.return_value = "/sys/fs/cgroup/probe"
        with self.assertRaisesRegex(InvalidEvidence, "cgroup"):
            obj._observe_confinement()
        obj._launch_observed.assert_not_called()

    def test_confinement_actual_mems_must_equal_approved_and_snapshot(self):
        obj = self.base()
        obj.io.status.return_value["Mems_allowed_list"] = "1"
        with self.assertRaisesRegex(InvalidEvidence, "memory-node"):
            obj._observe_confinement()
        obj._launch_observed.assert_not_called()

    def test_confinement_replaced_child(self):
        obj = self.base()
        obj.io.identity.return_value = {"pid": 41, "starttime": 999, "threads": 1}
        with self.assertRaisesRegex(InvalidEvidence, "identity"):
            obj._observe_confinement()

    def test_confinement_no_rebind_rechecked(self):
        obj = self.base()
        obj.io.text.return_value = "+cpuset"
        with self.assertRaisesRegex(InvalidEvidence, "controller mutation"):
            obj._observe_confinement()

    def test_confinement_full_launch_equality(self):
        for field in lb.LAUNCH_FIELDS:
            obj = self.base()
            obj._launch_observed.return_value[field] = "drift"
            with self.subTest(field=field), self.assertRaises((InvalidEvidence, TypeError)):
                obj._observe_confinement()

    def test_confinement_capabilities_rejected(self):
        for field in ("CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd"):
            obj = self.base()
            obj.io.status.return_value[field] = "1"
            with self.subTest(field=field), self.assertRaisesRegex(InvalidEvidence, "capability"):
                obj._observe_confinement()

    def test_confinement_all_credential_slots_rejected(self):
        for field in ("Uid", "Gid"):
            for index in range(4):
                obj = self.base()
                value = ["1000"] * 4
                value[index] = "0"
                obj.io.status.return_value[field] = " ".join(value)
                with self.subTest(field=field, index=index), self.assertRaises(InvalidEvidence):
                    obj._observe_confinement()

    def test_confinement_inherited_seccomp_rejected(self):
        obj = self.base()
        obj.io.status.return_value["Seccomp"] = "2"
        with self.assertRaisesRegex(InvalidEvidence, "seccomp"):
            obj._observe_confinement()

    def test_confinement_not_stopped_rejected(self):
        obj = self.base()
        obj.io.status.return_value["State"] = "S (sleeping)"
        with self.assertRaisesRegex(InvalidEvidence, "stopped"):
            obj._observe_confinement()

    def test_parent_wait_blocks_all_provider_operations(self):
        obj = self.base()
        obj.snapshot_consumed = True
        obj._check = lambda: ls.require(not obj.running, "parent running")
        calls = []
        def wait(pid):
            calls.append(pid)
            self.assertTrue(obj.running)
            for topic in ls.TOPICS:
                with self.assertRaisesRegex(InvalidEvidence, "parent running"):
                    obj.observe(topic)
            return 0
        obj.roles = SimpleNamespace(resume_wait_parent=wait)
        self.assertEqual(obj.resume_wait_parent(41), 0)
        self.assertEqual(calls, [41])
        self.assertTrue(obj.returned)
        with self.assertRaises(InvalidEvidence):
            obj.resume_wait_parent(41)

    def test_uncertain_wait_keeps_provider_closed(self):
        obj = self.base()
        obj.snapshot_consumed = True
        obj.roles = SimpleNamespace(resume_wait_parent=Mock(side_effect=OSError("wait uncertain")))
        with self.assertRaises(OSError):
            obj.resume_wait_parent(41)
        self.assertTrue(obj.running)
        self.assertFalse(obj.returned)

    def phase_object(self):
        obj = self.base()
        obj.sync_done, obj.ro_transition, obj.rw_transition = True, False, False
        obj.native = Mock(retained_fds={"device": 40, "filesystem": 41})
        obj.backing = {"filesystem_fd": 41, "device_fd": 40, "identity": {"fixture": True}}
        obj.block_get = Mock(return_value=0)
        value = {"uuid": "fixture-uuid", "healthy": True, "external_journal": False, "recovery_required": True}
        record = {"mount_id": 20, "mount_ro": False, "superblock_ro": False, "mount_options": ["rw"]}
        obj._health = Mock(return_value=(value, record))
        return obj, value, record

    def test_ext4_three_phase_certificate(self):
        obj, value, record = self.phase_object()
        pre = obj._observe_sync_health()
        self.assertNotIn("journal_clean", pre)
        self.assertTrue(pre["recovery_required"])
        obj.ro_transition = True
        value["recovery_required"] = False
        record.update(mount_ro=True, superblock_ro=True)
        ro = obj._observe_filesystem_ro()
        self.assertTrue(ro["journal_clean"])
        obj.rw_transition = True
        value["recovery_required"] = True
        record.update(mount_ro=False, superblock_ro=False)
        rw = obj._observe_filesystem_rw()
        self.assertNotIn("journal_clean", rw)
        self.assertTrue(rw["filesystem_rw"])

    def test_sync_success_not_journal_clean(self):
        obj, value, record = self.phase_object()
        obj._observe_sync_health()
        obj.ro_transition = True
        record.update(mount_ro=True, superblock_ro=True)
        with self.assertRaisesRegex(InvalidEvidence, "recovery-bit"):
            obj._observe_filesystem_ro()

    def test_mount_ro_alone_not_journal_clean(self):
        obj, value, record = self.phase_object()
        obj._observe_sync_health()
        obj.ro_transition = True
        value["recovery_required"] = False
        record["mount_ro"] = True
        with self.assertRaisesRegex(InvalidEvidence, "recovery-bit"):
            obj._observe_filesystem_ro()

    def test_changed_uuid_cannot_certify_clean_transition(self):
        obj, value, record = self.phase_object()
        obj._observe_sync_health()
        obj.ro_transition = True
        value.update(uuid="different", recovery_required=False)
        record.update(mount_ro=True, superblock_ro=True)
        with self.assertRaisesRegex(InvalidEvidence, "identity drift"):
            obj._observe_filesystem_ro()

    def test_clean_ro_without_pretransition_rejected(self):
        obj, value, record = self.phase_object()
        obj.pre_superblock = None
        obj.ro_transition = True
        value["recovery_required"] = False
        record.update(mount_ro=True, superblock_ro=True)
        with self.assertRaisesRegex(InvalidEvidence, "certificate absent"):
            obj._observe_filesystem_ro()

    def test_rw_restoration_requires_active_journal(self):
        obj, value, record = self.phase_object()
        obj.rw_transition = True
        value["recovery_required"] = False
        with self.assertRaisesRegex(InvalidEvidence, "RW_RESTORED_ACTIVE"):
            obj._observe_filesystem_rw()

    def test_child_groups_or_population_block_shutdown(self):
        obj = self.base()
        obj.returned = True
        obj.io.text.side_effect = ["populated 1\nfrozen 0\n", "41\n"]
        with self.assertRaisesRegex(InvalidEvidence, "populated"):
            obj._observe_shutdown()


class ExclusionTests(unittest.TestCase):
    def fixture(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj._check = lambda: None
        obj.exclusion, obj.lock_fd, obj.worker_pid = True, 10, 40
        obj.original_ns, obj.initial_user_ns = "mnt:original", "user:initial"
        obj.profile = {"namespace_services": {}}
        obj.backing = {"device_fd": 11}
        obj._raw_device_access = Mock(return_value={(stat.S_IFBLK, 999)})
        obj.io = Mock()
        obj.io.listdir.return_value = ["50"]
        obj.io.status.return_value = {"Uid": "0 0 0 0"}
        obj.io.identity.return_value = {"pid": 50, "starttime": 100, "threads": 1}
        obj.io.namespace.side_effect = lambda pid, kind: "mnt:original" if kind == "mnt" else "user:initial"
        obj.io.descriptors.return_value = []
        obj.io.text.return_value = ""
        obj._mounts = Mock(return_value=[{"major_minor": "259:4", "target": ls.MOUNT, "root": "/", "fstype": "ext4"}])
        return obj

    def test_observed_namespace_inventory(self):
        obj = self.fixture()
        result = obj._inventory()
        self.assertEqual(result["namespaces"], {"mnt:original": "user:initial"})
        self.assertEqual(result["processes"][0]["pid"], 50)

    def test_unexpected_mount_namespace_rejected(self):
        obj = self.fixture()
        obj.io.namespace.side_effect = lambda pid, kind: ("mnt:other" if pid == 50 else "mnt:original") if kind == "mnt" else "user:initial"
        obj.io.link.return_value = "/unapproved/service"
        with self.assertRaisesRegex(InvalidEvidence, "namespace class"):
            obj._inventory()

    def test_unexpected_user_namespace_rejected(self):
        obj = self.fixture()
        obj.io.namespace.side_effect = lambda pid, kind: "mnt:original" if kind == "mnt" else "user:other"
        with self.assertRaisesRegex(InvalidEvidence, "user namespace"):
            obj._inventory()

    def test_extra_device_alias_rejected(self):
        obj = self.fixture()
        obj._mounts.return_value.append({"major_minor": "259:4", "target": "/alias", "root": "/", "fstype": "ext4"})
        with self.assertRaisesRegex(InvalidEvidence, "mount alias"):
            obj._inventory()

    def test_protected_submount_rejected(self):
        obj = self.fixture()
        obj._mounts.return_value.append({"major_minor": "0:1", "target": ls.MOUNT + "/escape", "root": "/", "fstype": "tmpfs"})
        with self.assertRaisesRegex(InvalidEvidence, "submount"):
            obj._inventory()

    def test_raw_device_holder_rejected(self):
        obj = self.fixture()
        obj.io.descriptors.return_value = ["7"]
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFBLK | 0o600, st_rdev=999)
        with self.assertRaisesRegex(InvalidEvidence, "raw-device holder"):
            obj._inventory()

    def test_data_writer_rejected(self):
        obj = self.fixture()
        obj.io.descriptors.return_value = ["7"]
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFREG | 0o600)
        obj.io.text.return_value = "flags:\t02\n"
        with patch.object(ls, "major_minor", return_value="259:4"), self.assertRaisesRegex(InvalidEvidence, "writer"):
            obj._inventory()

    def test_shared_writable_mapping_rejected(self):
        obj = self.fixture()
        obj.io.text.return_value = "1000-2000 rw-s 00000000 103:04 123 /fixture\n"
        with self.assertRaisesRegex(InvalidEvidence, "writable.*mapping"):
            obj._inventory()

    def test_process_churn_rejected(self):
        obj = self.fixture()
        obj.io.identity.side_effect = [{"pid": 50, "starttime": 1}, {"pid": 50, "starttime": 2}]
        with self.assertRaisesRegex(InvalidEvidence, "churn"):
            obj._inventory()

    def test_unreadable_inventory_fails_closed(self):
        obj = self.fixture()
        obj.io.status.side_effect = PermissionError("fixture denial")
        with self.assertRaises(PermissionError):
            obj._inventory()


class BrokerIsolationTests(unittest.TestCase):
    header = "Num RefCount Protocol Flags Type St Inode Path\n"

    def inventory(self, *paths):
        return self.header + "".join(f"0000000000000000: 00000002 00000000 00010000 0001 01 {100 + i} {path}\n"
                                     for i, path in enumerate(paths))

    def fixture(self, inventory=None):
        obj = object.__new__(ls.LiveSystem)
        obj._check = Mock()
        obj.baseline = Mock()
        obj.baseline.values.return_value = {"credentials": {"uid": 1001, "gid": 1002, "groups": []}}
        obj.child = None
        obj.host_broker_inventory = None
        obj.io = Mock()
        obj.io.text.return_value = self.header if inventory is None else inventory
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFSOCK | 0o666, st_uid=0, st_gid=0)
        obj._no_acl = Mock()
        return obj

    def test_pathname_broker_selected_dac_write_and_acl(self):
        cases = [
            (1001, 1002, 0o444, [], "owner", "NO_CONNECT_AUTHORITY"),
            (0, 1002, 0o444, [], "group", "NO_CONNECT_AUTHORITY"),
            (0, 0, 0o444, [], "other", "NO_CONNECT_AUTHORITY"),
            (1001, 0, 0o200, [], "owner", "SAME_IDENTITY_IPC"),
            (0, 0, 0o002, [], "other", "AMBIENT_PUBLIC_IPC"),
            (0, 1002, 0o020, [], "group", "GROUP_PRIVILEGE_ROUTE"),
            (0, 119, 0o660, [119], "group", "GROUP_PRIVILEGE_ROUTE"),
            (1001, 1002, 0o066, [], "owner", "NO_CONNECT_AUTHORITY"),
            (0, 1002, 0o606, [], "group", "NO_CONNECT_AUTHORITY"),
        ]
        for uid, gid, mode, groups, selected, classification in cases:
            for path in ("/fixture/control.sock", "/var/snap/lxd/common/lxd/unix.socket"):
                with self.subTest(uid=uid, gid=gid, mode=oct(mode), path=path):
                    obj = self.fixture(self.inventory(path))
                    obj.baseline.values.return_value["credentials"]["groups"] = groups
                    obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFSOCK | mode, st_uid=uid, st_gid=gid)
                    value = ls.classify_pathname_broker(obj.io.stat.return_value, obj.baseline.values()["credentials"])
                    self.assertEqual(value, {"dac_class": selected, "connect_authority": classification != "NO_CONNECT_AUTHORITY",
                                             "classification": classification})
                    if classification == "GROUP_PRIVILEGE_ROUTE":
                        with self.assertRaisesRegex(InvalidEvidence, "GROUP_PRIVILEGE_ROUTE"):
                            obj._observe_brokers(private=False)
                    else:
                        receipt = obj._observe_brokers(private=False)
                        self.assertEqual(receipt["pathname_sockets"][0]["classification"], classification)
                    obj._no_acl.assert_called_once_with(path)
        obj = self.fixture(self.inventory("/fixture/control.sock"))
        obj._no_acl = ls.LiveSystem._no_acl.__get__(obj)
        with patch("os.getxattr", create=True, return_value=b"acl"), self.assertRaisesRegex(InvalidEvidence, "ACL"):
            obj._observe_brokers(private=False)

    def test_live_supplementary_groups_rejected_before_direct_access(self):
        def construct(bundle):
            return ls.LiveSystem(authorization=None, grant=None, provider_grant=SimpleNamespace(storage_identity=storage_fixture()),
                host_id="fixture-host", attempt="p1v-fixture", manifest=bundle["adapter"], provider_manifest=bundle["provider"],
                baseline=bundle["launch"], closure=bundle["closure"], journal=SimpleNamespace(pending=None),
                native=object.__new__(lb.NativePrimitives), processes=object.__new__(lb.NativeProcesses),
                roles=object.__new__(lb.ProcessRoles), bootstrap={})
        for groups in ([119], [1000], [119, 1000]):
            bundle = bundle_fixture()
            bundle["launch"]["credentials"]["groups"] = groups
            pin = lb.LaunchManifest.parse(bundle["launch"]).sha256
            bundle["adapter"]["launch_manifest_sha256"] = bundle["provider"]["launch_manifest_sha256"] = pin
            with self.subTest(groups=groups), patch.object(ls, "require_live"), \
                    patch.object(ls, "DirectLinux", side_effect=AssertionError("direct host construction")), \
                    patch("os.getpid", side_effect=AssertionError("host access")), \
                    patch("ctypes.CDLL", side_effect=AssertionError("library")), \
                    self.assertRaisesRegex(InvalidEvidence, "supplementary groups must be empty"):
                construct(bundle)
            from . import linux_gate as gate
            with patch.object(gate, "require_live"), \
                    patch.object(gate, "DirectLinux", side_effect=AssertionError("direct host construction")), \
                    self.assertRaisesRegex(InvalidEvidence, "supplementary groups must be empty"):
                gate.compose(bundle, authorization=None, execution_grant=None, provider_grant=None,
                             host_id="fixture-host", attempt="p1v-fixture")
        bundle = bundle_fixture()
        with patch.object(ls, "require_live"), \
                patch.object(ls.DirectLinux, "file_hash", side_effect=lambda path: {**ls.PROTECTED, **bundle["provider"]["modules"]}[Path(path).name]), \
                patch.object(ls.DirectLinux, "namespace", side_effect=lambda pid, kind: kind + ":original") as namespaces, \
                patch("os.getpid", return_value=40), patch("os.getppid", return_value=39), \
                patch("ctypes.CDLL", side_effect=AssertionError("library")):
            obj = construct(bundle)
        self.assertEqual(obj.baseline.values()["credentials"], {"uid": 1000, "gid": 1000, "groups": []})
        self.assertEqual(obj.original_network_ns, "net:original")
        namespaces.assert_any_call(40, "net")
        primary = SimpleNamespace(st_uid=0, st_gid=1000, st_mode=stat.S_IFSOCK | 0o020)
        self.assertEqual(ls.classify_pathname_broker(primary, obj.baseline.values()["credentials"])["classification"],
                         "GROUP_PRIVILEGE_ROUTE")
        self.assertEqual(launch_fixture()["credentials"]["groups"], [1000])

    def test_abstract_inventory_private_namespace_and_actual_child_closure(self):
        io = object.__new__(ls.DirectLinux)
        with patch.object(io, "stat", return_value=SimpleNamespace(st_dev=4, st_ino=5)) as observed:
            self.assertEqual(io.namespace(40, "net"), "4:5")
        observed.assert_called_once_with("/proc/40/ns/net", True)
        host_text = self.inventory("/fixture/ambient.sock", "@fixture-control", "@second abstract name")
        obj = self.fixture(host_text)
        obj._inventory = Mock(return_value={})
        obj.exclusion = True
        receipt = obj._observe_exclusion()
        self.assertTrue(receipt["brokers"]["pending_private_network_isolation"])
        self.assertEqual(receipt["brokers"]["abstract_socket_count"], 2)
        self.assertEqual([e["path"] for e in receipt["brokers"]["abstract_sockets"]],
                         ["@fixture-control", "@second abstract name"])
        self.assertNotIn("brokers_excluded", receipt)
        self.assertNotIn("brokers_excluded", receipt["brokers"])
        self.assertEqual(len(receipt["brokers"]["inventory"]), 3)
        for malformed in ("", "bad header\n", self.header + "bad row\n", host_text.rstrip("\n"),
                          self.inventory("relative.sock"), host_text.replace("00010000", "bad"),
                          host_text.replace("0001 01", "zzzz 01"), host_text.replace(" 100 ", " nope "),
                          self.header + "\n", self.inventory("@bad\0name"), self.header + "x" * 1048576 + "\n"):
            with self.subTest(malformed=malformed[:70]), self.assertRaises(InvalidEvidence):
                ls.parse_unix_sockets(malformed)
        unnamed = self.header + "0000000000000000: 00000002 00000000 00000000 0001 01 555\n"
        self.assertEqual(ls.parse_unix_sockets(unnamed)[0]["path"], "")
        obj.worker_pid, obj.original_ns, obj.original_network_ns, obj.initial_user_ns = 40, "mnt:original", "net:original", "user:initial"
        obj._mounts = Mock(return_value=[{"target": ls.MOUNT, "mount_id": 200, "propagation": []}])
        obj.backing = {"host_mount_id": 100}
        obj.native = SimpleNamespace(retained_fds={"private_mount_ns": 9})
        obj.namespace_receipt = None
        actual = {"mnt": "mnt:private", "net": "net:private", "user": "user:initial"}
        obj.io.namespace.side_effect = lambda pid, kind: actual[kind]
        with self.assertRaisesRegex(InvalidEvidence, "abstract sockets remain"):
            obj._observe_namespace()
        self.assertIsNone(obj.namespace_receipt)
        obj.io.text.return_value = self.header
        for kind, drift in (("mnt", "mnt:original"), ("net", "net:original"), ("user", "user:other")):
            old = actual[kind]
            actual[kind] = drift
            with self.subTest(kind=kind), self.assertRaises(InvalidEvidence):
                obj._observe_namespace()
            actual[kind] = old
        obj._mounts.return_value[0]["propagation"] = ["shared:1"]
        with self.assertRaisesRegex(InvalidEvidence, "recursively private"):
            obj._observe_namespace()
        obj._mounts.return_value[0]["propagation"] = []
        receipt = obj._observe_namespace()
        self.assertTrue(receipt["private_network_namespace"])
        self.assertTrue(receipt["private_mount_namespace"])
        self.assertEqual(receipt["abstract_socket_count"], 0)
        self.assertEqual(receipt["original_network_namespace_id"], "net:original")
        self.assertEqual(receipt["network_namespace_id"], "net:private")
        self.assertEqual(receipt["mount_namespace_id"], receipt["namespace_id"])
        self.assertEqual(receipt["original_mount_namespace_id"], receipt["original_namespace_id"])
        self.assertEqual(receipt["brokers"]["pathname_sockets"][0]["path"], "/fixture/ambient.sock")
        # Host pathname authority must still reject in an otherwise empty netns.
        obj.io.stat.return_value.st_gid = 1002
        with self.assertRaisesRegex(InvalidEvidence, "GROUP_PRIVILEGE_ROUTE"):
            obj._observe_namespace()
        for kind in ("net", "mnt", "user"):
            child = ProviderReceiptTests().base()
            child.io.namespace.side_effect = lambda pid, which: ("wrong" if which == kind else
                {"mnt": "mnt:private", "net": "net:private", "user": "user:initial"}[which])
            with self.subTest(child_namespace=kind), self.assertRaisesRegex(InvalidEvidence, "child namespace binding"):
                child._observe_confinement()
            child.journal.publish.assert_not_called()
            child._launch_observed.assert_not_called()
        child = ProviderReceiptTests().base()
        child.io.text.side_effect = lambda path: self.inventory("@private-escape") if path.endswith("/net/unix") else ""
        with self.assertRaisesRegex(InvalidEvidence, "abstract sockets remain"):
            child._observe_confinement()
        child.journal.publish.assert_not_called()
        child = ProviderReceiptTests().base()
        child.host_broker_inventory = ls.parse_unix_sockets(self.inventory("/fixture/group.sock"))
        child.io.stat.return_value = SimpleNamespace(st_uid=0, st_gid=1000, st_mode=stat.S_IFSOCK | 0o020)
        child._no_acl = Mock()
        with self.assertRaisesRegex(InvalidEvidence, "GROUP_PRIVILEGE_ROUTE"):
            child._observe_confinement()
        child.journal.publish.assert_not_called()
        for drift in (None, {"private_network_namespace": False},
                      {"private_network_namespace": True, "abstract_socket_count": 1},
                      {"private_network_namespace": True, "abstract_socket_count": 0, "network_namespace_id": "wrong"}):
            prechild, _ = CgroupTests().fixture()
            prechild._observe_cgroup_preflight()
            prechild.create_accounting_cgroup(prechild.attempt)
            prechild.namespace_receipt = drift
            with self.subTest(prechild=drift), self.assertRaisesRegex(InvalidEvidence, "before child creation"):
                prechild.prepare_stopped_child((), prechild.baseline.encoded, prechild.cgroup)
            prechild.roles.prepare_stopped_child.assert_not_called()
        child = ProviderReceiptTests().base()
        result = child._observe_confinement()
        self.assertIs(result["private_network_namespace"], True)
        self.assertIs(result["brokers_excluded"], True)
        child.io.text.assert_any_call("/proc/41/net/unix")
        name, data, action = child.journal.publish.call_args.args
        proof = strict_json(data)
        self.assertEqual((name, action), ("confinement-brokers.json", "CONFINEMENT"))
        self.assertEqual(proof["network_namespace_id"], "net:private")
        self.assertEqual(proof["handoff_child"], child.child_identity)
        self.assertEqual(proof["brokers"]["abstract_socket_count"], 0)
        self.assertFalse(proof["brokers"]["pending_private_network_isolation"])


class RawDeviceSysfsTests(unittest.TestCase):
    controller = "/sys/devices/pci0000:00/nvme/nvme1"
    registrations = "Character devices:\n240 nvme\n236 unrelated-driver\n\nBlock devices:\n259 blkext\n"

    def node(self, major=236, minor=255, mode=stat.S_IFCHR | 0o600):
        return SimpleNamespace(st_mode=mode, st_rdev=os.makedev(major, minor), st_uid=0, st_gid=0)

    def fixture(self, entries, registrations=None, registry=b""):
        obj = object.__new__(ls.LiveSystem)
        obj._check = Mock()
        obj._device = Mock(return_value={"sysfs_identity": self.controller + "/nvme1n1"})
        obj.baseline = Mock()
        obj.baseline.values.return_value = {"credentials": {"uid": 1001, "gid": 1002, "groups": []}}
        obj._no_acl = Mock()
        obj.io = Mock()
        obj.profile = {"dependency_files": {ls.STATIC_DEVNAME_REGISTRY: digest(registry)}}
        obj.io.read.side_effect = lambda path, *, regular: {ls.STATIC_DEVNAME_REGISTRY: registry}[path]
        obj.fixture_nodes = {"/dev/nvme1": self.node(240, 0), **entries}
        obj.io.stat.side_effect = obj.fixture_nodes.__getitem__
        texts = {"/proc/devices": self.registrations if registrations is None else registrations,
                 "/proc/net/unix": "Num RefCount Protocol Flags Type St Inode Path\n"}
        obj.io.text.side_effect = texts.__getitem__
        return obj

    def scan(self, obj, resolutions):
        def resolve(path, *, strict):
            self.assertTrue(strict)
            self.assertTrue(path.startswith("/sys/dev/char/"))
            rdev = path.rsplit("/", 1)[1]
            value = resolutions.get(rdev, self.controller if rdev == "240:0" else "/sys/devices/unrelated")
            if isinstance(value, list):
                value = value.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        directories = {}
        for path in obj.fixture_nodes:
            directories.setdefault(str(Path(path).parent), []).append(Path(path).name)
        with patch("os.walk", return_value=[(root, [], files) for root, files in directories.items()]) as walk, \
                patch("os.path.realpath", side_effect=resolve), \
                patch("os.open", side_effect=AssertionError("host access")), \
                patch("os.getxattr", create=True, side_effect=AssertionError("host ACL access")):
            result = obj._raw_device_access()
        walk.assert_called_once_with("/dev", followlinks=False, onerror=unittest.mock.ANY)
        obj.io.text.assert_any_call("/proc/devices")
        return result

    def test_unrelated_registered_missing_sysfs_character_is_tolerated(self):
        for name, driver in (("unrelated-node", "unrelated-driver"), ("nvidia-nvswitchctl", "nvidia-nvswitch")):
            with self.subTest(name=name, driver=driver):
                unrelated = self.node(mode=stat.S_IFCHR | 0o666)
                entries = {"/dev/" + name: unrelated,
                           "/dev/data": self.node(259, 4, stat.S_IFBLK | 0o600),
                           "/dev/controller-child": self.node(241, 0),
                           "/dev/nvme-other": self.node(242, 0)}
                obj = self.fixture(entries, self.registrations.replace("unrelated-driver", driver))
                result = self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing"),
                                         "241:0": self.controller + "/child"})
                self.assertEqual(result, {(stat.S_IFCHR, os.makedev(240, 0)),
                                          (stat.S_IFBLK, os.makedev(259, 4)),
                                          (stat.S_IFCHR, os.makedev(241, 0)),
                                          (stat.S_IFCHR, os.makedev(242, 0))})
                self.assertEqual([call.args[0] for call in obj._no_acl.call_args_list],
                                 ["/dev/nvme1", "/dev/data", "/dev/controller-child", "/dev/nvme-other"])

    def test_missing_sysfs_protected_basename_classes_reject(self):
        for name in ("nvme", "nvme42", "ng", "ng42n1", "sg", "sg42"):
            obj = self.fixture({"/dev/" + name: self.node()})
            with self.subTest(name=name), self.assertRaisesRegex(InvalidEvidence, "negative proof"):
                self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing")})

    def test_missing_sysfs_protected_registered_driver_classes_reject(self):
        for driver in ("nvme", "nvme-generic", "sg"):
            obj = self.fixture({"/dev/arbitrary": self.node()}, self.registrations.replace("unrelated-driver", driver))
            with self.subTest(driver=driver), self.assertRaisesRegex(InvalidEvidence, "negative proof"):
                self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing")})

    def test_missing_sysfs_same_rdev_alias_remains_protected(self):
        # The seeded controller identity rejects even an unrelated driver/name.
        obj = self.fixture({"/dev/arbitrary": self.node(240, 0)}, self.registrations.replace("240 nvme", "240 unrelated"))
        with self.assertRaisesRegex(InvalidEvidence, "negative proof"):
            self.scan(obj, {"240:0": [self.controller, FileNotFoundError(errno.ENOENT, "missing alias")]})
        # Classification by an existing route also protects a later missing alias.
        table = self.registrations.replace("236 unrelated-driver", "236 unrelated-driver\n241 unrelated")
        obj = self.fixture({"/dev/related": self.node(241, 0), "/dev/later-alias": self.node(241, 0)}, table)
        with self.assertRaisesRegex(InvalidEvidence, "negative proof"):
            self.scan(obj, {"241:0": [self.controller + "/child", FileNotFoundError(errno.ENOENT, "missing alias")]})
        # An alias observed before classification must remain in the second pass.
        for mode, denied in ((0o600, False), (0o666, True)):
            obj = self.fixture({"/dev/first-alias": self.node(241, 0, stat.S_IFCHR | mode),
                                "/dev/related": self.node(241, 0)}, table)
            resolutions = {"241:0": [FileNotFoundError(errno.ENOENT, "missing alias"), self.controller + "/child"]}
            with self.subTest(mode=mode):
                if denied:
                    with self.assertRaisesRegex(InvalidEvidence, "runtime raw/passthrough device exposure"):
                        self.scan(obj, resolutions)
                else:
                    self.assertIn((stat.S_IFCHR, os.makedev(241, 0)), self.scan(obj, resolutions))
                obj._no_acl.assert_any_call("/dev/first-alias")

    def test_missing_sysfs_unregistered_character_major_rejects(self):
        # A block registration of the same major cannot supply character proof.
        for block in ("", "236 unrelated-driver\n"):
            table = "Character devices:\n240 nvme\n\nBlock devices:\n" + block
            obj = self.fixture({"/dev/arbitrary": self.node()}, table)
            with self.subTest(block=block), self.assertRaisesRegex(InvalidEvidence, "negative proof"):
                self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing")})

    def test_malformed_or_conflicting_character_registration_rejects(self):
        rows = ("236", "236 driver extra", "-1 driver", "+236 driver", "0xec driver",
                "236.0 driver", "\u0662\u0663\u0666 driver", "4294967296 driver", "driver 236")
        tables = ["Character devices:\n" + row + "\n\nBlock devices:\n" for row in rows]
        tables += ["", "236 driver\n", "Block devices:\n236 driver\n",
                   "Character devices:\n236 driver\nCharacter devices:\n",
                   "Character devices:\n236 driver\nBlock devices:\nCharacter devices:\n",
                   "Character devices:\n236 driver\nBlock devices:\nBlock devices:\n"]
        for table in tables:
            obj = self.fixture({"/dev/arbitrary": self.node()}, table)
            with self.subTest(table=table), self.assertRaises(InvalidEvidence):
                self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing")})
        valid = ("Character devices:\n0 zero\n4294967295 maximum\n236 unrelated-driver\n"
                 "0236 unrelated-driver\n236 first\n236 second\n0236 second\n"
                 "\nBlock devices:\n236 nvme\nblock rows are not character data\n")
        self.assertEqual(ls.parse_character_devices(valid),
                         {0: ("zero",), 4294967295: ("maximum",), 236: ("first", "second", "unrelated-driver")})
        obj = self.fixture({"/dev/arbitrary": self.node()}, valid)
        self.assertNotIn((stat.S_IFCHR, os.makedev(236, 255)),
                         self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing")}))

    def test_multiple_character_registrations_are_canonical_sorted_unique(self):
        expected = {4: ("/dev/vc/0", "tty", "ttyS"),
                    5: ("/dev/console", "/dev/ptmx", "/dev/tty", "ttyprintk"),
                    195: ("nvidia", "nvidia-modeset", "nvidiactl")}
        rows = [str(major) + " " + name for major, names in expected.items() for name in names]
        rows += ["4 tty", "5 /dev/ptmx", "195 nvidia", "0195 nvidiactl"]
        for order in (rows, list(reversed(rows)), rows[5:] + rows[:5]):
            table = "Character devices:\n" + "\n".join(order) + "\n\nBlock devices:\n195 nvme\n8 sd\n"
            with self.subTest(rows=order):
                result = ls.parse_character_devices(table)
                self.assertEqual(result, expected)
                self.assertTrue(all(type(major) is int for major in result))
                for names in result.values():
                    self.assertIs(type(names), tuple)
                    self.assertTrue(names)
                    self.assertEqual(names, tuple(sorted(set(names))))

    def test_missing_sysfs_multiple_unprotected_registrations_are_tolerated(self):
        for major, basename, names in ((195, "nvidia0", ("nvidia", "nvidia-modeset", "nvidiactl")),
                                       (236, "arbitrary", ("unrelated", "another-driver", "third-driver"))):
            table = "Character devices:\n240 nvme\n" + "".join(str(major) + " " + name + "\n" for name in names)
            obj = self.fixture({"/dev/" + basename: self.node(major, 255, stat.S_IFCHR | 0o666)}, table)
            with self.subTest(major=major, names=names):
                result = self.scan(obj, {str(major) + ":255": FileNotFoundError(errno.ENOENT, "missing")})
                self.assertEqual(result, {(stat.S_IFCHR, os.makedev(240, 0))})
                obj._no_acl.assert_called_once_with("/dev/nvme1")

    def test_missing_sysfs_any_protected_registration_rejects(self):
        for protected in ("nvme", "nvme-generic", "sg"):
            names = ("aaa-unrelated", protected, "zzz-unrelated")
            for order in (names, tuple(reversed(names)), names[1:] + names[:1]):
                table = "Character devices:\n240 nvme\n" + "".join("236 " + name + "\n" for name in order)
                obj = self.fixture({"/dev/arbitrary": self.node()}, table)
                with self.subTest(protected=protected, names=order), \
                        self.assertRaisesRegex(InvalidEvidence, "negative proof"):
                    self.scan(obj, {"236:255": FileNotFoundError(errno.ENOENT, "missing")})

    def test_non_enoent_sysfs_resolution_errors_remain_fatal(self):
        errors = [OSError(code, "resolution failed") for code in
                  (errno.EACCES, errno.EPERM, errno.ELOOP, errno.EINVAL, errno.ENOTDIR, errno.EIO)]
        errors += [ValueError("malformed path"), FileNotFoundError("no ENOENT proof")]
        for error in errors:
            obj = self.fixture({"/dev/arbitrary": self.node()})
            with self.subTest(error=repr(error)), self.assertRaises(type(error)) as caught:
                self.scan(obj, {"236:255": error})
            self.assertIs(caught.exception, error)

    def test_static_devname_parser_canonical_and_fail_closed(self):
        rows = ["snd_seq snd/seq c116:1", "snd_timer snd/timer c116:33",
                "snd_seq snd/seq c116:1", "z-owner snd/seq c0116:01",
                "a.owner snd/seq c116:1", "a.owner snd/seq c116:1",
                "generic_1 disk/example b0:0", "maximum max c4294967295:4294967295"]
        expected = {"/dev/snd/seq": (stat.S_IFCHR, 116, 1, "116:1", ("a.owner", "snd_seq", "z-owner")),
                    "/dev/snd/timer": (stat.S_IFCHR, 116, 33, "116:33", ("snd_timer",)),
                    "/dev/disk/example": (stat.S_IFBLK, 0, 0, "0:0", ("generic_1",)),
                    "/dev/max": (stat.S_IFCHR, 4294967295, 4294967295,
                                 "4294967295:4294967295", ("maximum",))}
        for order in (rows, list(reversed(rows))):
            self.assertEqual(ls.parse_static_devnames("\n \t# comment\n" + "\n".join(order) + "\n\t\n"), expected)
        self.assertEqual(ls.parse_static_devnames("# empty\n\n"), {})
        bad_rows = ["owner", "owner node", "owner node c1:2 extra", "owner node c1:2 # inline"]
        bad_rows += [owner + " node c1:2" for owner in ("bad/name", "nönascii", "bad!", "bad\x00name", "+bad", "")]
        bad_rows += ["owner " + path + " c1:2" for path in
                     ("", "/node", "node/", "node//child", ".", "..", "./node", "../node",
                      "node/./child", "node/../child", "node\x00child", "node\nchild")]
        bad_rows += ["owner node " + device for device in
                     ("C1:2", "p1:2", "c:2", "c1:", "c1", "cc1:2", "c1:2:3", "c-1:2",
                      "c1:-2", "c+1:2", "c1:+2", "c0x1:2", "c1:2.0", "c١:2", "c1:٢",
                      "c4294967296:0", "c0:4294967296")]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(InvalidEvidence):
                ls.parse_static_devnames("good kept c2:3\n" + row)
        for conflict in ("b116:1", "c117:1", "c116:2"):
            with self.subTest(conflict=conflict), self.assertRaisesRegex(InvalidEvidence, "conflicting"):
                ls.parse_static_devnames("snd_seq snd/seq c116:1\nother snd/seq " + conflict)

    def test_missing_sysfs_exact_pinned_static_character_nodes_are_tolerated(self):
        table = "Character devices:\n240 nvme\n\nBlock devices:\n116 block-only\n"
        registry = b"snd_seq snd/seq c116:1\nsnd_timer snd/timer c116:33\ngeneric_module other/node c117:2\n"
        entries = {"/dev/snd/seq": self.node(116, 1, stat.S_IFCHR | 0o666),
                   "/dev/snd/timer": self.node(116, 33, stat.S_IFCHR | 0o666),
                   "/dev/other/node": self.node(117, 2, stat.S_IFCHR | 0o666)}
        obj = self.fixture(entries, table, registry)
        missing = {rdev: FileNotFoundError(errno.ENOENT, "missing") for rdev in ("116:1", "116:33", "117:2")}
        self.assertEqual(self.scan(obj, missing), {(stat.S_IFCHR, os.makedev(240, 0))})
        self.assertEqual(obj.io.read.call_count, 3)
        for call in obj.io.read.call_args_list:
            self.assertEqual(call, unittest.mock.call(ls.STATIC_DEVNAME_REGISTRY, regular=True))
        obj._no_acl.assert_called_once_with("/dev/nvme1")
        obj.io.file_hash.assert_not_called()  # Hash the very bytes parsed, with no second file read.
        # Neither successful sysfs routes nor active unrelated registrations consult the registry.
        for active in (False, True):
            obj = self.fixture(entries, table)
            if active:
                obj.io.text.side_effect = {"/proc/devices": table.replace("240 nvme", "240 nvme\n116 generic\n117 generic"),
                                           "/proc/net/unix": "header\n"}.__getitem__
            obj.profile["dependency_files"] = {}
            obj.io.read.side_effect = AssertionError("irrelevant static registry accessed")
            self.assertEqual(self.scan(obj, missing if active else {}), {(stat.S_IFCHR, os.makedev(240, 0))})
            obj.io.read.assert_not_called()

    def test_missing_sysfs_static_fallback_fail_closed_matrix(self):
        table = "Character devices:\n240 nvme\n\nBlock devices:\n"
        benign = b"generic_module arbitrary c116:1\n"
        cases = [("not-pinned", benign), ("hash-mismatch", benign), ("bad-pin", benign),
                 ("no-declaration", b""), ("path-mismatch", b"generic other/arbitrary c116:1\n"),
                 ("type-mismatch", b"generic arbitrary b116:1\n"),
                 ("major-mismatch", b"generic arbitrary c117:1\n"),
                 ("minor-mismatch", b"generic arbitrary c116:2\n"),
                 ("decode-failure", b"\xff"), ("malformed", b"generic arbitrary c116:1\nbad row\n"),
                 ("missing-file", benign), ("read-failure", benign),
                 ("already-protected", b"generic arbitrary c240:0\n"),
                 ("non-enoent", benign)]
        for owner in ("sg", "nvme", "nvme-core", "nvme-generic", "nvme-fabrics", "nvme-tcp", "nvme_core", "nvme_tcp"):
            cases.append(("owner-" + owner, benign + (owner + " arbitrary c116:1\n").encode()))
        for basename in ("nvme", "nvme42", "ng", "ng42n1", "sg", "sg42"):
            cases.append(("basename-" + basename, ("generic " + basename + " c116:1\n").encode()))
        for registration in ("nvme", "nvme-generic", "sg"):
            cases.append(("active-" + registration, benign))
        for case, registry in cases:
            basename = case.removeprefix("basename-") if case.startswith("basename-") else "arbitrary"
            major = 240 if case == "already-protected" else 116
            active = table
            if case == "already-protected":
                active = table.replace("240 nvme\n", "")
            elif case.startswith("active-"):
                active = table.replace("240 nvme", "240 nvme\n116 unrelated\n116 " + case.removeprefix("active-"))
            obj = self.fixture({"/dev/" + basename: self.node(major, 1 if major == 116 else 0)}, active, registry)
            error = FileNotFoundError(errno.ENOENT, "missing")
            expected = InvalidEvidence
            if case == "not-pinned":
                del obj.profile["dependency_files"][ls.STATIC_DEVNAME_REGISTRY]
            elif case == "hash-mismatch":
                obj.profile["dependency_files"][ls.STATIC_DEVNAME_REGISTRY] = "0" * 64
            elif case == "bad-pin":
                obj.profile["dependency_files"][ls.STATIC_DEVNAME_REGISTRY] = "bad"
            elif case == "decode-failure":
                expected = UnicodeDecodeError
            elif case in ("missing-file", "read-failure"):
                expected = FileNotFoundError if case == "missing-file" else PermissionError
                obj.io.read.side_effect = expected("registry read denied")
            elif case == "non-enoent":
                error, expected = PermissionError(errno.EACCES, "sysfs denied"), PermissionError
            resolutions = {"116:1": error} if major == 116 else {"240:0": [self.controller, error]}
            with self.subTest(case=case), self.assertRaises(expected):
                self.scan(obj, resolutions)
            if case.startswith("active-") or case in ("not-pinned", "bad-pin", "non-enoent"):
                obj.io.read.assert_not_called()
        # A static alias encountered before a protected route still reaches the unchanged second pass.
        for mode in (0o600, 0o666):
            obj = self.fixture({"/dev/arbitrary": self.node(116, 1, stat.S_IFCHR | mode),
                                "/dev/related": self.node(116, 1)}, table, benign)
            routes = {"116:1": [FileNotFoundError(errno.ENOENT, "missing"), self.controller + "/child"]}
            with self.subTest(alias_mode=mode):
                if mode == 0o666:
                    with self.assertRaisesRegex(InvalidEvidence, "runtime raw/passthrough device exposure"):
                        self.scan(obj, routes)
                else:
                    self.assertIn((stat.S_IFCHR, os.makedev(116, 1)), self.scan(obj, routes))
                obj._no_acl.assert_any_call("/dev/arbitrary")
        # The direct reader checks regular-file type on the opened descriptor, before reading bytes.
        io = object.__new__(ls.DirectLinux)
        io.check = Mock()
        io._storage_identity = canonical_json(storage_fixture())
        for mode in (stat.S_IFDIR, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFLNK):
            with self.subTest(nonregular=mode), patch("os.open", return_value=40) as opened, \
                    patch("os.fstat", return_value=SimpleNamespace(st_mode=mode)), patch("os.close") as closed, \
                    patch("os.read", side_effect=AssertionError("nonregular read")):
                with self.assertRaisesRegex(InvalidEvidence, "regular dependency"):
                    io.read(ls.STATIC_DEVNAME_REGISTRY, regular=True)
                opened.assert_called_once_with(ls.STATIC_DEVNAME_REGISTRY,
                                               os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
                closed.assert_called_once_with(40)


class NamespaceServiceTests(unittest.TestCase):
    executable = "/fixture/namespace-service"

    def entry(self):
        return {"sha256": "c" * 64, "cgroup": "0::/system.slice/fixture.service",
                "uid": [1201] * 4, "gid": [1202] * 4}

    def fixture(self):
        obj = ExclusionTests.fixture(self)
        allowed = self.entry()
        obj.profile = bundle_fixture()["provider"]["host_profile"]
        obj.profile["namespace_services"] = {self.executable: allowed}
        obj.io.namespace.side_effect = lambda pid, kind: (
            "mnt:service" if pid == 50 else "mnt:original") if kind == "mnt" else "user:initial"
        obj.io.link.return_value = self.executable
        obj.io.file_hash.return_value = allowed["sha256"]
        obj.io.text.side_effect = lambda path: allowed["cgroup"] + "\n" if path == "/proc/50/cgroup" else ""
        obj.io.status.return_value = {"Uid": "1201 1201 1201 1201", "Gid": "1202 1202 1202 1202"}
        return obj

    def test_profile_accepts_exact_root_and_nonroot_identities(self):
        profile = bundle_fixture()["provider"]["host_profile"]
        ls.validate_host_profile(profile)
        for uid, gid in (([0] * 4, [0] * 4), ([1201] * 4, [1202] * 4),
                         ([0, 1, 2, 2**32 - 1], [2**32 - 1, 2, 1, 0])):
            profile["namespace_services"] = {self.executable: dict(self.entry(), uid=uid, gid=gid)}
            with self.subTest(uid=uid, gid=gid):
                ls.validate_host_profile(profile)

    def test_malformed_profile_entries_reject(self):
        cases = [("non-dict", []), ("old-schema", {k: self.entry()[k] for k in ("sha256", "cgroup")}),
                 ("extra-key", dict(self.entry(), allow_nonroot=True))]
        for key in self.entry():
            entry = self.entry()
            del entry[key]
            cases.append(("missing-" + key, entry))
        for invalid in ("bad", "A" * 64, "a" * 63, None):
            cases.append(("sha256", dict(self.entry(), sha256=invalid)))
        for invalid in (None, "", "0::", "1::/service", "0::/bad\0service", "0::/bad\nservice"):
            cases.append(("cgroup", dict(self.entry(), cgroup=invalid)))
        for key in ("uid", "gid"):
            for invalid in ((1201,) * 4, "1201 1201 1201 1201", None, [], [0] * 3, [0] * 5,
                            [0, 0, 0, True], [0, 0, 0, "1"], [0, 0, 0, 1.0],
                            [0, 0, 0, -1], [0, 0, 0, 2**32]):
                cases.append((key, dict(self.entry(), **{key: invalid})))
        for label, entry in cases:
            profile = bundle_fixture()["provider"]["host_profile"]
            profile["namespace_services"] = {self.executable: entry}
            with self.subTest(case=label, entry=entry), self.assertRaises(InvalidEvidence):
                ls.validate_host_profile(profile)
        for executable in ("relative/service", "/fixture/../service", "/fixture/invalid\0service"):
            profile = bundle_fixture()["provider"]["host_profile"]
            profile["namespace_services"] = {executable: self.entry()}
            with self.subTest(executable=executable), self.assertRaises(InvalidEvidence):
                ls.validate_host_profile(profile)

    def test_inventory_accepts_exact_nonroot_with_namespace_and_mount_gates(self):
        obj = self.fixture()
        ls.validate_host_profile(obj.profile)
        self.assertEqual(obj._inventory(), {"namespaces": {"mnt:service": "user:initial"},
                                           "processes": [{"pid": 50, "starttime": 100, "threads": 1}]})
        obj.io.link.assert_called_once_with("/proc/50/exe")
        obj.io.file_hash.assert_called_once_with(self.executable)
        obj.io.text.assert_any_call("/proc/50/cgroup")
        for case, message in (("unknown-executable", "namespace class"), ("user-namespace", "user namespace"),
                              ("extra-mount", "mount alias"), ("target", "mount alias"),
                              ("root", "mount alias"), ("fstype", "mount alias"), ("submount", "submount")):
            obj = self.fixture()
            if case == "unknown-executable":
                obj.io.link.return_value = "/fixture/unapproved-service"
            elif case == "user-namespace":
                obj.io.namespace.side_effect = lambda pid, kind: "mnt:service" if kind == "mnt" else "user:other"
            elif case == "extra-mount":
                obj._mounts.return_value.append(dict(obj._mounts.return_value[0], target="/fixture/alias"))
            elif case == "submount":
                obj._mounts.return_value.append({"major_minor": "0:1", "target": ls.MOUNT + "/escape",
                                                "root": "/", "fstype": "tmpfs"})
            else:
                obj._mounts.return_value[0][case] = {"target": "/fixture/alias", "root": "/subdir", "fstype": "tmpfs"}[case]
            with self.subTest(case=case), self.assertRaisesRegex(InvalidEvidence, message):
                obj._inventory()

    def test_inventory_exact_uid_tuple_drift_rejects(self):
        for slot in range(4):
            obj = self.fixture()
            values = [1201] * 4
            values[slot] = 0
            obj.io.status.return_value["Uid"] = " ".join(map(str, values))
            with self.subTest(slot=slot), self.assertRaisesRegex(InvalidEvidence, "service identity drift"):
                obj._inventory()
        for value, error in ((None, KeyError), ("1201 invalid 1201 1201", ValueError),
                             ("", InvalidEvidence), ("1201 " * 3, InvalidEvidence), ("1201 " * 5, InvalidEvidence)):
            obj = self.fixture()
            if value is None:
                del obj.io.status.return_value["Uid"]
            else:
                obj.io.status.return_value["Uid"] = value
            with self.subTest(value=value), self.assertRaises(error):
                obj._inventory()

    def test_inventory_exact_gid_tuple_drift_rejects(self):
        for slot in range(4):
            obj = self.fixture()
            values = [1202] * 4
            values[slot] = 0
            obj.io.status.return_value["Gid"] = " ".join(map(str, values))
            with self.subTest(slot=slot), self.assertRaisesRegex(InvalidEvidence, "service identity drift"):
                obj._inventory()
        for value, error in ((None, KeyError), ("1202 invalid 1202 1202", ValueError),
                             ("", InvalidEvidence), ("1202 " * 3, InvalidEvidence), ("1202 " * 5, InvalidEvidence)):
            obj = self.fixture()
            if value is None:
                del obj.io.status.return_value["Gid"]
            else:
                obj.io.status.return_value["Gid"] = value
            with self.subTest(value=value), self.assertRaises(error):
                obj._inventory()

    def test_inventory_hash_and_cgroup_drift_reject_with_matching_ids(self):
        for field in ("sha256", "cgroup"):
            obj = self.fixture()
            if field == "sha256":
                obj.io.file_hash.return_value = "d" * 64
            else:
                obj.io.text.side_effect = lambda path: "0::/system.slice/other.service\n" if path == "/proc/50/cgroup" else ""
            with self.subTest(field=field), self.assertRaisesRegex(InvalidEvidence, "service identity drift"):
                obj._inventory()


class MetadataTests(unittest.TestCase):
    def metadata(self, variant=None, storage=None, file_storage=None):
        storage = storage_fixture() if storage is None else storage
        file_storage = storage if file_storage is None else file_storage
        major, minor = map(int, file_storage["major_minor"].split(":"))
        from contextlib import ExitStack
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage)
        obj._check = lambda: None
        obj.baseline = lb.LaunchManifest.parse(numa_baseline())
        obj.io = Mock()
        obj.io.dio_alignment.return_value = {"memory": 512, "offset": 512, "source": "STATX_DIOALIGN"}
        info = SimpleNamespace(st_dev=1, st_ino=10, st_size=2658304, st_mode=stat.S_IFREG | 0o400,
                               st_nlink=1, st_mtime_ns=1, st_ctime_ns=1)
        if variant == "symlink":
            info.st_mode = stat.S_IFLNK | 0o777
        if variant == "wrong_size":
            info.st_size = 2
        if variant == "hardlink":
            info.st_nlink = 2
        def lstat(path):
            if str(path).endswith(".bin"):
                if variant == "fallback" and str(path).endswith("expert_6027.bin"):
                    raise FileNotFoundError()
                if variant == "permission" and str(path).endswith("expert_6027.bin"):
                    raise PermissionError()
                return info
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755)
        with ExitStack() as stack:
            for name, result in (("geteuid", 0), ("getegid", 0), ("getgroups", [0]),
                                  ("setgroups", None), ("setegid", None), ("seteuid", None),
                                  ("open", 100), ("close", None), ("major", major), ("minor", minor)):
                stack.enter_context(patch.object(os, name, return_value=result))
            stack.enter_context(patch.object(os, "O_DIRECT", 0x4000, create=True))
            stack.enter_context(patch.object(os, "lstat", side_effect=lstat))
            stack.enter_context(patch.object(os, "fstat", return_value=info))
            stack.enter_context(patch("os.path.realpath", side_effect=lambda p, **kw: ls.CANONICAL + "/" + str(p).rsplit("/", 1)[1]))
            stack.enter_context(patch("fcntl.fcntl", return_value=0 if variant == "no_direct" else 0x4000))
            stack.enter_context(patch("fcntl.ioctl", return_value=0))
            for name in ("read", "pread"):
                stack.enter_context(patch.object(os, name, create=True, side_effect=AssertionError("expert bytes read")))
            return obj._observe_metadata()

    def test_metadata_exact_ids_without_payload_read(self):
        records = self.metadata()["records"]
        self.assertEqual([r["global_expert_id"] for r in records], list(ls.STARTUP_IDS + ls.IDS))
        self.assertTrue(all(r["payload_bytes_read"] == 0 and r["o_direct_retained"] for r in records))
        from .supervisor import metadata_gate
        metadata_gate(records, storage_fixture())

    def test_primary_enoent_permits_exact_fallback(self):
        records = self.metadata("fallback")["records"]
        self.assertEqual(records[8]["selected"], ls.alternatives(6027)[1])

    def test_permission_error_never_falls_back(self):
        with self.assertRaises(PermissionError):
            self.metadata("permission")

    def test_symlink_hardlink_wrongsize_and_missing_direct_reject(self):
        for variant in ("symlink", "hardlink", "wrong_size", "no_direct"):
            with self.subTest(variant=variant), self.assertRaises(InvalidEvidence):
                self.metadata(variant)

    def test_statx_dioalign_mock_abi(self):
        obj = object.__new__(ls.DirectLinux)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.check = lambda: None
        def statx(fd, path, flags, requested, out):
            self.assertEqual((fd, path, flags, requested), (100, b"", 0x1000, 0x2000))
            data = bytearray(256)
            struct.pack_into("<I", data, 0, 0x2000)
            struct.pack_into("<II", data, 152, 512, 512)
            ctypes.memmove(out, bytes(data), 256)
            return 0
        libc = SimpleNamespace(statx=Mock(side_effect=statx))
        with patch.object(obj, "libc", return_value=libc):
            self.assertEqual(obj.dio_alignment(100)["offset"], 512)

    def test_statx_missing_alignment_rejects(self):
        obj = object.__new__(ls.DirectLinux)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.check = lambda: None
        with patch.object(obj, "libc", return_value=SimpleNamespace(statx=Mock(return_value=0))), self.assertRaises(InvalidEvidence):
            obj.dio_alignment(100)


class BpfQueryTests(unittest.TestCase):
    KINDS = (0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22,
             29, 30, 31, 32, 34, 43, 49, 50, 51, 52, 53)

    class Query(ctypes.Structure):
        _fields_ = [("target_fd", ctypes.c_uint32), ("attach_type", ctypes.c_uint32),
                    ("query_flags", ctypes.c_uint32), ("attach_flags", ctypes.c_uint32),
                    ("prog_ids", ctypes.c_uint64), ("prog_cnt", ctypes.c_uint32),
                    ("padding", ctypes.c_uint32), ("prog_attach_flags", ctypes.c_uint64)]

    def query(self, respond, path="/sys/fs/cgroup"):
        obj = object.__new__(ls.DirectLinux)
        obj.check = Mock()
        self.queried = []

        def syscall(number, command, pointer, size):
            self.assertEqual((number.value, command.value, size), (321, 16, 40))
            query = ctypes.cast(pointer, ctypes.POINTER(self.Query)).contents
            self.queried.append(query.attach_type)
            self.assertEqual((query.target_fd, query.query_flags, query.prog_cnt), (101, 0, 256))
            self.assertNotEqual(query.prog_ids, 0)
            self.assertNotEqual(query.prog_attach_flags, 0)
            self.assertNotEqual(query.prog_ids, query.prog_attach_flags)
            self.assertEqual(list((ctypes.c_uint32 * 256).from_address(query.prog_ids)), [0] * 256)
            self.assertEqual(list((ctypes.c_uint32 * 256).from_address(query.prog_attach_flags)), [0] * 256)
            return respond(query)

        obj.libc = Mock(return_value=SimpleNamespace(syscall=Mock(side_effect=syscall)))
        with patch("os.open", return_value=101) as opened, patch("os.close") as closed:
            try:
                return obj.bpf_ids(path)
            finally:
                obj.check.assert_called_once_with()
                opened.assert_called_once_with(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
                closed.assert_called_once_with(101)

    def answer(self, query, programs):
        ids = (ctypes.c_uint32 * 256).from_address(query.prog_ids)
        flags = (ctypes.c_uint32 * 256).from_address(query.prog_attach_flags)
        query.attach_flags = 0x55
        query.prog_cnt = len(programs)
        for i, (program_id, flag) in enumerate(programs):
            ids[i], flags[i] = program_id, flag
        return 0

    def test_exact_attach_sequence_and_non_effective_flags_buffer_for_every_query(self):
        self.assertEqual(self.query(lambda query: self.answer(query, [])), [])
        self.assertEqual(self.queried, list(self.KINDS))

    def test_lsm_cgroup_retains_distinct_per_program_flags(self):
        programs = [(701, 1), (702, 0x80000002)]
        result = self.query(lambda query: self.answer(query, programs if query.attach_type == 43 else []))
        self.assertEqual(result, [{"type": 43, "id": 701, "flags": 1},
                                  {"type": 43, "id": 702, "flags": 0x80000002}])

    def test_ordinary_cgroup_retains_per_program_not_common_flags(self):
        programs = [(801, 0), (802, 2)]
        result = self.query(lambda query: self.answer(query, programs if query.attach_type == 1 else []))
        self.assertEqual(result, [{"type": 1, "id": 801, "flags": 0},
                                  {"type": 1, "id": 802, "flags": 2}])

    def test_syscall_errors_including_einval_stop_before_later_attach_types(self):
        for kind in self.KINDS:
            for error, ret in ((errno.EINVAL, -1), (errno.EPERM, -1), (errno.EACCES, -1),
                               (errno.EIO, -1), (0, 1)):
                def respond(query):
                    if query.attach_type == kind:
                        ctypes.set_errno(error)
                        return ret
                    return self.answer(query, [(901, 2)])
                with self.subTest(kind=kind, errno=error, ret=ret):
                    with self.assertRaisesRegex(InvalidEvidence, "cgroup BPF inventory unavailable/overflow"):
                        self.query(respond)
                    self.assertEqual(self.queried, list(self.KINDS[:self.KINDS.index(kind) + 1]))

    def test_program_count_overflow_fails_closed(self):
        for kind in self.KINDS:
            def respond(query):
                if query.attach_type == kind:
                    query.prog_cnt = 257
                    return 0
                return self.answer(query, [])
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(InvalidEvidence, "cgroup BPF inventory unavailable/overflow"):
                    self.query(respond)
                self.assertEqual(self.queried, list(self.KINDS[:self.KINDS.index(kind) + 1]))

    def test_domain_preserves_exact_ancestor_local_bpf_schema_and_flags(self):
        root = "/sys/fs/cgroup"
        paths = [root, root + "/parent", root + "/parent/leaf"]
        attached = {paths[0]: {1: [(1011, 2)], 43: [(1012, 4)]}, paths[1]: {},
                    paths[2]: {43: [(1013, 1), (1014, 0x80000002)]}}
        obj = object.__new__(ls.LiveSystem)
        obj.provider = {"nvml": {"uuid": "GPU-fixture"}}
        obj.gpu_management = {"fixture": True}
        obj.io = Mock()
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0)
        obj.io.listdir.side_effect = lambda path: ["cpuset.cpus.effective", "cpuset.mems.effective"] if path == root else []
        obj.io.text.side_effect = lambda path: {root + "/cpuset.cpus.effective": "0-1\n",
                                               root + "/cpuset.mems.effective": "0\n"}[path]
        obj.io.bpf_ids.side_effect = lambda path: self.query(
            lambda query: self.answer(query, attached[path].get(query.attach_type, [])), path)
        with patch.object(lb, "resource_domain", wraps=lb.resource_domain) as validate:
            result = obj._domain(paths[-1])
        expected = {name: {"domain": "cgroup2-root-disabled", "constraints": {}}
                    for name in ("cpu", "memory", "io", "cpuset", "pids", "hugetlb", "rdma", "misc")}
        expected.update({"gpu": {"domain": "GPU-fixture", "constraints": {"fixture": True}},
                         "effective_cpus": [0, 1], "effective_mems": [0], "ancestor_constraints": [],
                         "bpf_ancestry": [
                             {"path": paths[0], "programs": [{"type": 1, "id": 1011, "flags": 2},
                                                            {"type": 43, "id": 1012, "flags": 4}]},
                             {"path": paths[2], "programs": [{"type": 43, "id": 1013, "flags": 1},
                                                            {"type": 43, "id": 1014, "flags": 0x80000002}]}]})
        self.assertEqual(result, expected)
        self.assertEqual([call.args for call in obj.io.bpf_ids.call_args_list], [(path,) for path in paths])
        validate.assert_called_once_with(expected)


class CgroupTests(unittest.TestCase):
    def fixture(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj._check, obj._host = lambda: None, lambda: "fixture-boot"
        obj.attempt, obj.host_id, obj.worker_pid = "p1v-fixture", "fixture-host", 40
        obj.exclusion, obj.snapshot_consumed = True, False
        obj.snapshot = obj.child = obj.cgroup = None
        obj.namespace_receipt = {"private_network_namespace": True, "abstract_socket_count": 0,
                                 "network_namespace_id": "net:private"}
        obj.profile = {"cgroup_parent": "/sys/fs/cgroup"}
        obj.provider = {"nvml": {"uuid": "GPU-fixture"}}
        obj.gpu_management = {"fixture": True}
        obj.nvml = SimpleNamespace(management=Mock(return_value=obj.gpu_management))
        obj._worker = Mock(return_value=snapshot_fixture()["worker"])
        obj.io = Mock()
        obj.io.namespace.side_effect = lambda pid, kind: (
            "net:private"
            if (pid, kind) == (obj.worker_pid, "net") else obj.io.namespace.return_value)
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0)
        obj.io.listdir.side_effect = lambda path: ["cpuset.cpus.effective", "cpuset.mems.effective"] if path == "/sys/fs/cgroup" else []
        values = {"/proc/40/cgroup": "0::/\n", "/sys/fs/cgroup/cgroup.type": "domain\n",
                  "/sys/fs/cgroup/cgroup.controllers": "cpu cpuset memory io\n", "/sys/fs/cgroup/cgroup.subtree_control": "\n",
                  "/sys/fs/cgroup/cpuset.cpus.effective": "0-1\n", "/sys/fs/cgroup/cpuset.mems.effective": "0\n"}
        obj.io.text.side_effect = lambda path: values[path]
        obj.io.bpf_ids.return_value = []
        obj.io.status.return_value = {"Mems_allowed_list": "0"}
        obj.io.task_policy.return_value = numa_baseline()["numa"]["memory_policy"]
        obj._mounts = Mock(return_value=[{"target": "/sys/fs/cgroup", "fstype": "cgroup2", "root": "/"}])
        baseline = numa_baseline()
        baseline["resource_domain"] = obj._domain("/sys/fs/cgroup")
        obj.baseline = lb.LaunchManifest.parse(baseline)
        obj.processes = SimpleNamespace(create_accounting_cgroup=Mock(return_value="/sys/fs/cgroup/p1v-fixture"))
        obj.roles = SimpleNamespace(prepare_stopped_child=Mock(return_value=41))
        obj.io.identity.return_value = {"pid": 41, "starttime": 901, "threads": 1}
        return obj, values

    def test_preflight_then_create_then_exact_fork(self):
        obj, values = self.fixture()
        preflight = obj._observe_cgroup_preflight()
        self.assertEqual(preflight["baseline"], preflight["prospective"])
        self.assertEqual(preflight["settings_to_write"], [])
        group = obj.create_accounting_cgroup(obj.attempt)
        from .supervisor import parent_argv
        self.assertEqual(obj.prepare_stopped_child(parent_argv(obj.attempt), obj.baseline.encoded, group), 41)
        obj.roles.prepare_stopped_child.assert_called_once()
        self.assertEqual(obj.io.task_policy.call_count, 2)

    def test_create_before_preflight_forbidden(self):
        obj, _ = self.fixture()
        with self.assertRaises(InvalidEvidence):
            obj.create_accounting_cgroup(obj.attempt)
        obj.processes.create_accounting_cgroup.assert_not_called()

    def test_active_controller_before_query_forbidden(self):
        obj, values = self.fixture()
        values["/sys/fs/cgroup/cgroup.subtree_control"] = "cpuset cpu"
        with self.assertRaisesRegex(InvalidEvidence, "controller mutation"):
            obj._observe_cgroup_preflight()
        obj.io.task_policy.assert_not_called()

    def test_hybrid_hierarchy_rejected(self):
        obj, values = self.fixture()
        values["/proc/40/cgroup"] = "0::/\n3:cpuset:/\n"
        with self.assertRaisesRegex(InvalidEvidence, "unified"):
            obj._observe_cgroup_preflight()
        obj.io.task_policy.assert_not_called()

    def test_effective_mems_changed_before_fork_rejected(self):
        obj, values = self.fixture()
        obj._observe_cgroup_preflight()
        obj.create_accounting_cgroup(obj.attempt)
        values["/sys/fs/cgroup/cpuset.mems.effective"] = "1"
        with self.assertRaisesRegex(InvalidEvidence, "no-rebind"):
            obj.prepare_stopped_child((), obj.baseline.encoded, obj.cgroup)
        obj.roles.prepare_stopped_child.assert_not_called()

    def test_snapshot_cannot_be_captured_twice(self):
        obj, _ = self.fixture()
        obj._observe_cgroup_preflight()
        with self.assertRaisesRegex(InvalidEvidence, "one NUMA"):
            obj._observe_cgroup_preflight()

    def test_wrong_cgroup_attempt_rejected(self):
        obj, _ = self.fixture()
        obj._observe_cgroup_preflight()
        with self.assertRaises(InvalidEvidence):
            obj.create_accounting_cgroup("p1v-other")
        obj.processes.create_accounting_cgroup.assert_not_called()


class IntegrationAdmissionTests(unittest.TestCase):
    def test_constructing_provider_does_not_initialize_nvml_or_query_policy(self):
        bundle = bundle_fixture()
        journal = SimpleNamespace(pending=None)
        def code_hash(path):
            return {**ls.PROTECTED, **bundle["provider"]["modules"]}[Path(path).name]
        with patch.object(ls, "require_live"), patch.object(ls.DirectLinux, "file_hash", side_effect=code_hash), \
             patch.object(ls.DirectLinux, "namespace", return_value="fixture-ns"), \
             patch.object(ls.DirectLinux, "task_policy", side_effect=AssertionError("early task policy query")), \
             patch("os.getpid", return_value=40), patch("os.getppid", return_value=39), \
             patch("ctypes.CDLL", side_effect=AssertionError("NVML library initialized early")):
            obj = ls.LiveSystem(authorization=None, grant=None, provider_grant=SimpleNamespace(storage_identity=storage_fixture()), host_id="fixture-host",
                attempt="p1v-fixture", manifest=bundle["adapter"], provider_manifest=bundle["provider"],
                baseline=bundle["launch"], closure=bundle["closure"], journal=journal,
                native=object.__new__(lb.NativePrimitives), processes=object.__new__(lb.NativeProcesses),
                roles=object.__new__(lb.ProcessRoles), bootstrap={})
        self.assertIsNone(obj.nvml.library)
        self.assertIsNone(obj.snapshot)

    def test_direct_interface_constructor_cannot_use_callback_as_grant(self):
        context = (None, None, None, "fixture", "p1v-fixture", {}, {})
        guard = Mock(side_effect=AssertionError("guard must not run"))
        with self.assertRaises(InvalidEvidence):
            ls.DirectLinux(context, guard)
        guard.assert_not_called()

    def test_provider_code_mismatch_before_proc_identity(self):
        bundle = bundle_fixture()
        with patch.object(ls, "require_live"), patch.object(ls.DirectLinux, "file_hash", return_value="0" * 64), \
             patch.object(ls.DirectLinux, "namespace", side_effect=AssertionError("proc observation before code admission")), \
             patch("os.getpid", side_effect=AssertionError("host identity before code admission")), \
             self.assertRaisesRegex(InvalidEvidence, "module hash mismatch"):
            ls.LiveSystem(authorization=None, grant=None, provider_grant=SimpleNamespace(storage_identity=storage_fixture()), host_id="fixture-host",
                attempt="p1v-fixture", manifest=bundle["adapter"], provider_manifest=bundle["provider"],
                baseline=bundle["launch"], closure=bundle["closure"], journal=SimpleNamespace(pending=None),
                native=object.__new__(lb.NativePrimitives), processes=object.__new__(lb.NativeProcesses),
                roles=object.__new__(lb.ProcessRoles), bootstrap={})

    def test_future_scope_hash_mismatch_is_not_a_grant(self):
        scope = {"authority_issue": 245, "host_id": "fixture", "attempt": "p1v-fixture",
                 "operations": [a.name for a in plan_fixture()], "provider_manifest_sha256": "a" * 64, "storage_identity": storage_fixture()}
        with self.assertRaisesRegex(InvalidEvidence, "scope pin"):
            ls.ProviderGrant(scope, "0" * 64, b"x" * 32, digest(b"x" * 32))

    def test_future_token_mismatch_is_not_a_grant(self):
        scope = {"authority_issue": 245, "host_id": "fixture", "attempt": "p1v-fixture",
                 "operations": [a.name for a in plan_fixture()], "provider_manifest_sha256": "a" * 64, "storage_identity": storage_fixture()}
        with self.assertRaisesRegex(InvalidEvidence, "token"):
            ls.ProviderGrant(scope, digest(canonical_json(scope)), b"x" * 32, "0" * 64)

    def test_nvml_guard_rejects_parent_before_library_calls(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.running, obj.failed = True, False
        with self.assertRaisesRegex(InvalidEvidence, "parent"):
            obj._gpu_check()

    def report_object(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj._check = lambda: None
        obj.returned, obj.attempt = True, "p1v-fixture"
        obj.baseline = lb.LaunchManifest.parse(numa_baseline())
        obj.io = Mock()
        directory = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=50, st_uid=1000, st_gid=1000)
        obj.io.stat.return_value = directory
        obj.bootstrap = {"runtime_directory_identity": {"device": "259:1", "inode": 50, "uid": 1000, "gid": 1000, "mode": 0o700}}
        info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=1, st_ino=51, st_size=3,
                               st_uid=1000, st_gid=1000, st_nlink=1, st_mtime_ns=1, st_ctime_ns=1)
        return obj, info

    def test_raw_reports_immutable_exact_bytes_and_names(self):
        obj, info = self.report_object()
        with patch.object(ls, "major_minor", return_value="259:1"), patch("os.open", return_value=100) as opening, \
             patch("os.fstat", return_value=info), patch("os.lstat", return_value=info), \
             patch("os.read", return_value=b"{}\n"), patch("os.close"):
            result = obj.raw_reports(obj.attempt)
        self.assertEqual(result, {name: b"{}\n" for name in ("parent", "control", "treatment")})
        self.assertEqual([Path(c.args[0]).name for c in opening.call_args_list], list(ls.REPORT_NAMES.values()))
        self.assertTrue(all(c.args[1] & os.O_NOFOLLOW for c in opening.call_args_list))

    def test_report_owner_type_hardlink_rejected(self):
        for field, value in (("st_uid", 0), ("st_mode", stat.S_IFLNK | 0o777), ("st_nlink", 2)):
            obj, info = self.report_object()
            setattr(info, field, value)
            with self.subTest(field=field), patch.object(ls, "major_minor", return_value="259:1"), \
                 patch("os.open", return_value=100), patch("os.fstat", return_value=info), patch("os.close"), \
                 patch("os.read", side_effect=AssertionError("unadmitted report read")), self.assertRaises(InvalidEvidence):
                obj.raw_reports(obj.attempt)

    def test_report_attempt_directory_replacement_rejected(self):
        obj, _ = self.report_object()
        obj.io.stat.return_value.st_ino = 99
        with patch.object(ls, "major_minor", return_value="259:1"), \
             patch("os.open", side_effect=AssertionError("unadmitted report opened")), self.assertRaises(InvalidEvidence):
            obj.raw_reports(obj.attempt)

    def test_report_wrong_backing_rejected(self):
        obj, info = self.report_object()
        with patch.object(ls, "major_minor", side_effect=["259:1", "259:4"]), \
             patch("os.open", return_value=100), patch("os.fstat", return_value=info), patch("os.close"), \
             patch("os.read", side_effect=AssertionError("unadmitted report read")), self.assertRaises(InvalidEvidence):
            obj.raw_reports(obj.attempt)


class HostIdentityHermeticityTests(unittest.TestCase):
    boot = "11111111-2222-3333-4444-555555555555"
    fixture_host_id = "f85e708067ebddffffef7de70b872fceb3f2d0c86d7e386a2c50f666204bb884"

    def observe(self, boot, machine_id="a" * 32, host_id=None):
        host = Mock()
        host.text.side_effect = [boot + "\n", machine_id + "\n"]
        uname = SimpleNamespace(sysname="Linux", release="6.8.0-1069-gcp", machine="x86_64")
        if host_id is None:
            host_id = digest(canonical_json({"machine_id": machine_id, "boot_id": boot,
                                            "kernel": uname.release, "machine": uname.machine}))
        with patch("os.uname", return_value=uname):
            return ls.observe_host_identity(host, host_id)

    def test_canonical_lowercase_without_version_or_variant_restrictions(self):
        for boot in (self.boot, "abcdef01-2345-6789-abcd-ef0123456789",
                     "00000000-0000-0000-0000-000000000000",
                     "ffffffff-ffff-ffff-ffff-ffffffffffff"):
            with self.subTest(boot=boot):
                self.assertEqual(self.observe(boot), boot)

    def test_malformed_boot_ids_reject_before_machine_read(self):
        boot = "abcdef01-2345-6789-abcd-ef0123456789"
        cases = {"uppercase": boot.upper(), "missing_hyphen": boot.replace("-", "", 1),
                 "misplaced_hyphen": boot[:7] + "-" + boot[7] + boot[9:],
                 "extra_hyphen": boot[:8] + "-" + boot[8:],
                 "nonhex": "g" + boot[1:], "truncated": boot[:-1], "extended": boot + "0",
                 "braced": "{" + boot + "}", "internal_space": boot[:20] + " " + boot[21:],
                 "internal_tab": boot[:20] + "\t" + boot[21:],
                 "internal_newline": boot[:20] + "\n" + boot[21:], "empty": ""}
        for name, value in cases.items():
            host = Mock()
            host.text.return_value = value + "\n"
            with self.subTest(case=name), patch("os.uname", return_value=SimpleNamespace(
                    sysname="Linux", release="6.8.0-1069-gcp", machine="x86_64")), \
                    self.assertRaisesRegex(InvalidEvidence, "boot ID format"):
                ls.observe_host_identity(host, "fixture")
            host.text.assert_called_once_with("/proc/sys/kernel/random/boot_id")

    def test_every_position_requires_exact_hyphen_or_lowercase_hex(self):
        for index in range(36):
            replacements = ("0", "a", " ") if index in (8, 13, 18, 23) else ("-", "A", "G", "g", " ", "\u0661")
            for char in replacements:
                boot = self.boot[:index] + char + self.boot[index + 1:]
                with self.subTest(index=index, char=char), self.assertRaisesRegex(InvalidEvidence, "boot ID format"):
                    self.observe(boot)

    def test_machine_id_remains_exact_lowercase_hex(self):
        for machine_id in ("a" * 32, "0123456789abcdef" * 2):
            with self.subTest(machine_id=machine_id):
                self.assertEqual(self.observe(self.boot, machine_id), self.boot)
        for machine_id in ("", "a" * 31, "a" * 33, "A" * 32, "g" + "a" * 31,
                           "a" * 15 + "-" + "a" * 16, "a" * 15 + " " + "a" * 16,
                           "{" + "a" * 30 + "}", self.boot):
            with self.subTest(machine_id=machine_id), self.assertRaisesRegex(InvalidEvidence, "machine ID format"):
                self.observe(self.boot, machine_id)

    def test_existing_fixture_digest_is_unchanged(self):
        self.assertEqual(self.observe(self.boot, host_id=self.fixture_host_id), self.boot)
        self.assertEqual(self.observe(" \t" + self.boot + "\n", "\t" + "a" * 32 + " ",
                                      self.fixture_host_id), self.boot)

    def test_changed_identity_still_rejects_original_digest(self):
        for boot, machine_id in (("0" + self.boot[1:], "a" * 32), (self.boot, "b" * 32)):
            with self.subTest(boot=boot, machine_id=machine_id), \
                    self.assertRaisesRegex(InvalidEvidence, "host ID/grant mismatch"):
                self.observe(boot, machine_id, self.fixture_host_id)

    def test_denied_check_precedes_uname_and_text(self):
        host = Mock()
        host.check.side_effect = InvalidEvidence("denied")
        with patch("os.uname", side_effect=AssertionError("host observation before check")) as uname, \
                self.assertRaisesRegex(InvalidEvidence, "denied"):
            ls.observe_host_identity(host, self.fixture_host_id)
        uname.assert_not_called()
        host.text.assert_not_called()

    def test_check_uname_boot_machine_order(self):
        events = []
        host = Mock()
        host.check.side_effect = lambda: events.append("check")
        values = {"/proc/sys/kernel/random/boot_id": self.boot + "\n", "/etc/machine-id": "a" * 32 + "\n"}
        def text(path):
            events.append(path)
            return values[path]
        def uname():
            events.append("uname")
            return SimpleNamespace(sysname="Linux", release="6.8.0-1069-gcp", machine="x86_64")
        host.text.side_effect = text
        with patch("os.uname", side_effect=uname):
            self.assertEqual(ls.observe_host_identity(host, self.fixture_host_id), self.boot)
        self.assertEqual(events, ["check", "uname", "/proc/sys/kernel/random/boot_id", "/etc/machine-id"])

    def test_kernel_or_architecture_mismatch_precedes_text(self):
        for field, value in (("sysname", "Darwin"), ("release", "other"), ("machine", "aarch64")):
            host = Mock()
            uname = SimpleNamespace(sysname="Linux", release="6.8.0-1069-gcp", machine="x86_64")
            setattr(uname, field, value)
            with self.subTest(field=field), patch("os.uname", return_value=uname), \
                    self.assertRaisesRegex(InvalidEvidence, "target kernel/architecture mismatch"):
                ls.observe_host_identity(host, self.fixture_host_id)
            host.text.assert_not_called()

    def test_observation_needs_no_runtime_imports(self):
        for release in ("6.8.0-1069-gcp", "other"):
            host = Mock()
            host.text.side_effect = [self.boot + "\n", "a" * 32 + "\n"]
            uname = SimpleNamespace(sysname="Linux", release=release, machine="x86_64")
            with self.subTest(release=release), patch("os.uname", return_value=uname):
                if release == "other":
                    with self.assertRaisesRegex(InvalidEvidence, "target kernel/architecture mismatch"):
                        with patch("builtins.__import__", side_effect=AssertionError("incidental runtime import")):
                            ls.observe_host_identity(host, self.fixture_host_id)
                    host.text.assert_not_called()
                else:
                    with patch("builtins.__import__", side_effect=AssertionError("incidental runtime import")):
                        observed = ls.observe_host_identity(host, self.fixture_host_id)
                    self.assertEqual(observed, self.boot)

    def test_import_does_not_discover_host_or_import_uuid(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], ("uuid", "platform"))
            return original_import(name, *args, **kwargs)
        with patch("os.uname", side_effect=AssertionError("import-time host discovery")), \
                patch("builtins.__import__", side_effect=guarded_import):
            importlib.reload(ls)


class HostIdentityTests(unittest.TestCase):
    def test_observed_kernel_boot_machine_binding(self):
        host = Mock()
        boot = "11111111-2222-3333-4444-555555555555"
        host.text.side_effect = [boot + "\n", "a" * 32 + "\n"]
        uname = SimpleNamespace(sysname="Linux", release="6.8.0-1069-gcp", machine="x86_64")
        host_id = digest(canonical_json({"machine_id": "a" * 32, "boot_id": boot,
                                         "kernel": uname.release, "machine": uname.machine}))
        with patch("os.uname", return_value=uname):
            self.assertEqual(ls.observe_host_identity(host, host_id), boot)

    def test_wrong_target_kernel_stops_before_boot_read(self):
        host = Mock()
        with patch("os.uname", return_value=SimpleNamespace(sysname="Linux", release="other", machine="x86_64")), \
             self.assertRaisesRegex(InvalidEvidence, "kernel"):
            ls.observe_host_identity(host, "fixture")
        host.text.assert_not_called()

    def device_fixture(self):
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.io = Mock()
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFBLK | 0o600, st_rdev=999)
        obj.io.listdir.side_effect = lambda path: ["google-local-nvme-ssd-0"] if path == "/dev/disk/by-id" else []
        obj.io.text.side_effect = lambda path: "259:4\n" if path.endswith("/dev") else "1\n"
        return obj

    def test_exact_sysfs_device_uuid_link_sequence(self):
        obj = self.device_fixture()
        with patch.object(ls, "major_minor", return_value="259:4"), \
             patch("os.path.realpath", return_value="/sys/devices/pci/nvme/nvme0/nvme0n1"):
            value = obj._device()
        self.assertEqual(value["disk_sequence"], 1)
        self.assertEqual(value["uuid"], storage_fixture()["uuid"])
        self.assertEqual(value["device_path"], "/dev/nvme0n1")

    def test_stacked_device_rejected(self):
        obj = self.device_fixture()
        obj.io.listdir.side_effect = None
        obj.io.listdir.return_value = ["dm-0"]
        with patch.object(ls, "major_minor", return_value="259:4"), \
             patch("os.path.realpath", return_value="/sys/devices/pci/nvme/nvme0/nvme0n1"), \
             self.assertRaisesRegex(InvalidEvidence, "holders/slaves"):
            obj._device()

    def test_uuid_device_mismatch_rejected(self):
        obj = self.device_fixture()
        obj.io.stat.side_effect = [SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=999),
                                  SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=998)]
        with patch.object(ls, "major_minor", return_value="259:4"), \
             patch("os.path.realpath", return_value="/sys/devices/pci/nvme/nvme0/nvme0n1"), \
             self.assertRaisesRegex(InvalidEvidence, "UUID"):
            obj._device()

    def test_direct_numa_query_impossible_during_parent(self):
        provider = object.__new__(ls.LiveSystem)
        provider.running, provider.failed = True, False
        obj = object.__new__(ls.DirectLinux)
        obj._storage_identity = canonical_json(storage_fixture())
        obj.check = provider._check
        with patch.object(obj, "libc", side_effect=AssertionError("task syscall during parent")), \
             self.assertRaisesRegex(InvalidEvidence, "parent"):
            obj.task_policy()


class DataFilesystemProfileTests(unittest.TestCase):
    """Prospective identity only; historical P1V1 payload continuity is unproven.

    Future continuity begins at the newly qualified current filesystem baseline.
    The historical UUID below is used only as an explicitly rejected fixture.
    """

    CURRENT = "b362df64-42eb-4603-a685-c8e5595038ee"
    HISTORICAL_REJECTED = "5bd529dc-bb64-47f2-9d27-82c3defb0d39"
    WRONG = "11111111-2222-3333-4444-555555555555"

    def test_exact_single_current_uuid_expectation(self):
        self.assertIs(type(storage_fixture()["uuid"]), str)
        self.assertEqual(storage_fixture()["uuid"], self.CURRENT)
        self.assertNotEqual(storage_fixture()["uuid"], self.HISTORICAL_REJECTED)
        self.assertFalse({"data_uuid", "data_device", "data_major_minor"} & set(ls.EXPECTATIONS))

    def test_ext4_fixture_contains_current_uuid_in_each_phase(self):
        for recovery in (True, False):
            with self.subTest(recovery=recovery):
                value = ls.ext4_superblock(superblock(recovery))
                self.assertEqual(value["uuid"], self.CURRENT)
                self.assertIs(value["recovery_required"], recovery)

    def device(self, obj):
        with patch.object(ls, "major_minor", return_value="259:4"), \
                patch("os.path.realpath", return_value="/sys/devices/pci/nvme/nvme0/nvme0n1"):
            return obj._device()

    def test_device_queries_only_exact_current_uuid_alias(self):
        obj = HostIdentityTests.device_fixture(self)
        self.assertEqual(self.device(obj)["uuid"], self.CURRENT)
        self.assertEqual([c.args for c in obj.io.stat.call_args_list],
                         [("/dev/nvme0n1",), ("/dev/disk/by-uuid/" + self.CURRENT, True),
                          (storage_fixture()["by_id_aliases"][0], True)])

    def test_uuid_alias_must_be_block_device(self):
        for mode in (stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK):
            obj = HostIdentityTests.device_fixture(self)
            obj.io.stat.side_effect = [SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=999),
                                      SimpleNamespace(st_mode=mode, st_rdev=999)]
            with self.subTest(mode=mode), self.assertRaisesRegex(InvalidEvidence, "UUID device link"):
                self.device(obj)

    def test_current_alias_must_resolve_to_exact_retained_device(self):
        obj = HostIdentityTests.device_fixture(self)
        obj.io.stat.side_effect = [SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=os.makedev(259, 4)),
                                  SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=os.makedev(259, 1))]
        with self.assertRaisesRegex(InvalidEvidence, "UUID device link"):
            self.device(obj)

    def test_historical_alias_cannot_substitute_for_missing_current_alias(self):
        obj = HostIdentityTests.device_fixture(self)
        def observed(path, follow=False):
            if path in ("/dev/nvme0n1", "/dev/disk/by-uuid/" + self.HISTORICAL_REJECTED):
                return SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=999)
            raise FileNotFoundError(path)
        obj.io.stat.side_effect = observed
        with self.assertRaises(FileNotFoundError):
            self.device(obj)
        self.assertEqual([c.args for c in obj.io.stat.call_args_list],
                         [("/dev/nvme0n1",), ("/dev/disk/by-uuid/" + self.CURRENT, True)])

    def health_fixture(self):
        obj = HostIdentityTests.device_fixture(self)
        identity = self.device(obj)
        obj._device = Mock(return_value=identity)
        obj.native = Mock(retained_fds={"device": 40, "filesystem": 41})
        obj.backing = {"filesystem_fd": 41, "device_fd": 40, "identity": copy.deepcopy(identity), "private_mount_id": 20}
        data = bytearray(superblock())
        obj.io.superblock.side_effect = lambda fd: ls.ext4_superblock(bytes(data))
        record = {"mount_id": 20, "target": ls.MOUNT, "fstype": "ext4", "major_minor": "259:4",
                  "mount_ro": False, "superblock_ro": False, "mount_options": ["rw"]}
        obj._mounts = Mock(return_value=[record])
        obj.sync_done, obj.ro_transition, obj.rw_transition = True, False, False
        obj.block_get = Mock(return_value=0)
        return obj, data, record

    def transition_ro(self, obj, data, record):
        obj._observe_sync_health()
        obj.ro_transition = True
        struct.pack_into("<I", data, 0x60, 0)
        record.update(mount_ro=True, superblock_ro=True, mount_options=["ro"])

    def test_historical_and_wrong_superblock_uuid_reject(self):
        for rejected in (self.HISTORICAL_REJECTED, self.WRONG):
            obj, data, _ = self.health_fixture()
            data[0x68:0x78] = bytes.fromhex(rejected.replace("-", ""))
            with self.subTest(uuid=rejected), self.assertRaisesRegex(InvalidEvidence, "same retained ext4 device/UUID"):
                obj._observe_sync_health()

    def test_current_uuid_passes_all_three_phase_identities(self):
        obj, data, record = self.health_fixture()
        self.transition_ro(obj, data, record)
        self.assertEqual(strict_json(obj.pre_superblock)["uuid"], self.CURRENT)
        ro = obj._observe_filesystem_ro()
        self.assertTrue(ro["journal_clean"])
        self.assertEqual(ro["device_identity"]["uuid"], self.CURRENT)
        obj.rw_transition = True
        struct.pack_into("<I", data, 0x60, 4)
        record.update(mount_ro=False, superblock_ro=False, mount_options=["rw"])
        rw = obj._observe_filesystem_rw()
        self.assertEqual(rw["device_identity"], ro["device_identity"])
        self.assertTrue(rw["recovery_required"])
        self.assertNotIn("journal_clean", rw)

    def test_pre_ro_to_ro_uuid_drift_rejects(self):
        for rejected in (self.HISTORICAL_REJECTED, self.WRONG):
            obj, data, record = self.health_fixture()
            self.transition_ro(obj, data, record)
            data[0x68:0x78] = bytes.fromhex(rejected.replace("-", ""))
            with self.subTest(uuid=rejected), self.assertRaisesRegex(InvalidEvidence, "same retained ext4 device/UUID"):
                obj._observe_filesystem_ro()

    def test_ro_to_restored_rw_uuid_drift_rejects(self):
        for rejected in (self.HISTORICAL_REJECTED, self.WRONG):
            obj, data, record = self.health_fixture()
            self.transition_ro(obj, data, record)
            obj._observe_filesystem_ro()
            obj.rw_transition = True
            struct.pack_into("<I", data, 0x60, 4)
            record.update(mount_ro=False, superblock_ro=False)
            data[0x68:0x78] = bytes.fromhex(rejected.replace("-", ""))
            with self.subTest(uuid=rejected), self.assertRaisesRegex(InvalidEvidence, "same retained ext4 device/UUID"):
                obj._observe_filesystem_rw()

    def test_same_device_and_private_mount_equality_still_exact(self):
        for field in ("device_path", "major_minor", "disk_sequence", "sysfs_identity", "by_id_aliases"):
            obj, _, _ = self.health_fixture()
            obj._device.return_value[field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(InvalidEvidence, "same retained ext4 device/UUID"):
                obj._health()
        obj, _, record = self.health_fixture()
        record["mount_id"] += 1
        with self.assertRaisesRegex(InvalidEvidence, "private ext4 mount identity"):
            obj._health()

    def test_all_phase_receipts_require_current_uuid(self):
        from .supervisor import receipt_gate
        from .test_p1v import encode_raw, fake_content, raw_fixture, receipts
        fixture = receipts(encode_raw(raw_fixture()), fake_content())
        for action in ("FILESYSTEM_RO", "DEVICE_RO", "PROTECTED", "REVALIDATE", "DEVICE_RW", "FILESYSTEM_RW"):
            self.assertEqual(fixture[action]["device_identity"]["uuid"], self.CURRENT)
            receipt_gate(action, fixture[action], {"receipts": fixture})
            for rejected in (self.HISTORICAL_REJECTED, self.WRONG):
                bad = copy.deepcopy(fixture[action])
                bad["device_identity"]["uuid"] = rejected
                with self.subTest(action=action, uuid=rejected), self.assertRaises(InvalidEvidence):
                    receipt_gate(action, bad, {"receipts": fixture})

    def test_refreshed_hash_propagation_and_other_protected_pins(self):
        from .test_linux_backend import PROTECTED_P1V_FILES
        root = Path(ls.__file__).parent
        expected_init = "a936eec1bd452981180dd06ad236caf84040ba3d3275a5ad56de5a570228b1be"
        self.assertEqual(digest((root / "__init__.py").read_bytes()), expected_init)
        self.assertEqual(PROTECTED_P1V_FILES["__init__.py"], expected_init)
        self.assertEqual(ls.PROTECTED, {**PROTECTED_P1V_FILES,
            "test_linux_backend.py": digest((root / "test_linux_backend.py").read_bytes())})

    def test_package_witness_rejects_restoring_historical_uuid(self):
        from .test_linux_backend import PackageScopeWitnessTests, assert_package_scope
        root = PackageScopeWitnessTests.package_fixture(self)
        path = root / "__init__.py"
        self.assertNotIn(self.CURRENT, path.read_text())
        path.write_text(path.read_text() + "\n# forbidden fallback " + self.HISTORICAL_REJECTED)
        with self.assertRaisesRegex(AssertionError, "protected SHA256: __init__.py"):
            assert_package_scope(self, root)

    def test_no_historical_uuid_fallback_or_uuid_repair_in_production(self):
        root = Path(ls.__file__).parent
        for name in ("__init__.py", "live_system.py", "linux_gate.py", "linux_backend.py",
                     "content_hasher.py", "offline_verify.py", "supervisor.py"):
            source = (root / name).read_text()
            with self.subTest(module=name):
                for forbidden in (self.HISTORICAL_REJECTED, "tune2fs", "e2fsck", "udevadm"):
                    self.assertNotIn(forbidden, source)


class CpuPolicyAbsenceTests(unittest.TestCase):
    ROOT = "/sys/devices/system/cpu/cpufreq"
    CATEGORIES = ("governors", "frequency_limits", "boost", "power")
    LIMITS = ("scaling_min_freq", "scaling_max_freq", "cpuinfo_min_freq", "cpuinfo_max_freq")

    def profile(self, absent=False):
        profile = bundle_fixture()["provider"]["host_profile"]
        if absent:
            profile["policy_files"]["cpu_policy"] = {name: [] for name in self.CATEGORIES}
        return profile

    def provider(self, absent=False):
        obj = object.__new__(ls.LiveSystem)
        obj.profile = self.profile(absent)
        obj.io = Mock()
        obj.io.listdir.return_value = [] if absent else ["policy0", "policy2"]
        obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755)
        obj.io.text.side_effect = lambda path: "observed:" + path + "\n"
        if not absent:
            paths = obj.profile["policy_files"]["cpu_policy"]
            paths["governors"] = [self.ROOT + "/" + name + "/scaling_governor" for name in ("policy0", "policy2")]
            paths["frequency_limits"] = sorted(self.ROOT + "/" + name + "/" + field
                                               for name in ("policy0", "policy2") for field in self.LIMITS)
        ls.validate_host_profile(obj.profile)
        return obj

    def empty_observation(self):
        return {name: {} for name in self.CATEGORIES}

    def test_profile_accepts_all_four_empty_lists_and_existing_exposed_form(self):
        for absent in (False, True):
            profile = self.profile(absent)
            before = copy.deepcopy(profile)
            with self.subTest(absent=absent):
                ls.validate_host_profile(profile)
                self.assertEqual(profile, before)

    def test_every_partial_empty_profile_rejects(self):
        for bits in range(1, 15):
            profile = self.profile()
            for index, name in enumerate(self.CATEGORIES):
                if bits & (1 << index):
                    profile["policy_files"]["cpu_policy"][name] = []
            with self.subTest(bits=bits), self.assertRaisesRegex(InvalidEvidence, "fully EXPOSED or fully ABSENT"):
                ls.validate_host_profile(profile)

    def test_profile_rejects_nonlists_duplicates_unsorted_and_nonsys_paths(self):
        for name in self.CATEGORIES:
            for invalid in (None, {}, (), "", ["/sys/a", "/sys/a"], ["/sys/b", "/sys/a"],
                            ["/proc/sys/fixture"], ["/fixture/policy"]):
                profile = self.profile()
                profile["policy_files"]["cpu_policy"][name] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(InvalidEvidence):
                    ls.validate_host_profile(profile)

    def test_thp_and_numa_inventories_remain_nonempty(self):
        for absent in (False, True):
            for group in ("thp", "numa_balancing"):
                for empty in ({}, {"enabled": []}):
                    profile = self.profile(absent)
                    profile["policy_files"][group] = empty
                    with self.subTest(absent=absent, group=group, empty=empty), self.assertRaises(InvalidEvidence):
                        ls.validate_host_profile(profile)

    def test_absent_missing_root_returns_exact_empty_maps(self):
        obj = self.provider(True)
        obj.io.listdir.side_effect = FileNotFoundError(self.ROOT)
        before = copy.deepcopy(obj.profile)
        self.assertEqual(obj._policy_files("cpu_policy"), self.empty_observation())
        self.assertEqual(obj.profile, before)
        obj.io.listdir.assert_called_once_with(self.ROOT)
        obj.io.stat.assert_not_called()
        obj.io.text.assert_not_called()

    def test_absent_existing_root_without_policy_directories_returns_exact_empty_maps(self):
        for entries in ([], ["boost", "unrelated"]):
            obj = self.provider(True)
            obj.io.listdir.return_value = entries
            before = copy.deepcopy(obj.profile)
            with self.subTest(entries=entries):
                self.assertEqual(obj._policy_files("cpu_policy"), self.empty_observation())
                self.assertEqual(obj.profile, before)
                obj.io.listdir.assert_called_once_with(self.ROOT)
                obj.io.text.assert_not_called()

    def test_absent_policy_directory_appearance_rejects_without_rewriting_profile(self):
        for name in ("policy0", "policy19", "policy-extra"):
            obj = self.provider(True)
            obj.io.listdir.return_value = [name]
            before = copy.deepcopy(obj.profile)
            with self.subTest(name=name), self.assertRaisesRegex(InvalidEvidence, "ABSENT CPU policy interface drift"):
                obj._policy_files("cpu_policy")
            self.assertEqual(obj.profile, before)
            obj.io.text.assert_not_called()

    def test_exposed_missing_or_empty_root_rejects_before_policy_reads(self):
        for missing in (False, True):
            obj = self.provider()
            obj.io.listdir.return_value = []
            if missing:
                obj.io.listdir.side_effect = FileNotFoundError(self.ROOT)
            with self.subTest(missing=missing), self.assertRaisesRegex(InvalidEvidence, "interface unavailable"):
                obj._policy_files("cpu_policy")
            obj.io.text.assert_not_called()

    def test_root_errors_other_than_missing_remain_hard_failures(self):
        for absent in (False, True):
            for error in (PermissionError, OSError, NotADirectoryError, RuntimeError):
                obj = self.provider(absent)
                obj.io.listdir.side_effect = error("root observation failed")
                with self.subTest(absent=absent, error=error), self.assertRaises(error):
                    obj._policy_files("cpu_policy")
                obj.io.text.assert_not_called()

    def test_policy_entry_malformed_or_unreadable_remains_failure(self):
        for absent in (False, True):
            for mode in (stat.S_IFREG, stat.S_IFLNK):
                obj = self.provider(absent)
                obj.io.listdir.return_value = ["policy0"]
                obj.io.stat.return_value = SimpleNamespace(st_mode=mode)
                with self.subTest(absent=absent, mode=mode), self.assertRaisesRegex(InvalidEvidence, "not a directory"):
                    obj._policy_files("cpu_policy")
            for error in (FileNotFoundError, PermissionError, OSError):
                obj = self.provider(absent)
                obj.io.listdir.return_value = ["policy0"]
                obj.io.stat.side_effect = error("policy observation failed")
                with self.subTest(absent=absent, error=error), self.assertRaises(error):
                    obj._policy_files("cpu_policy")

    def test_exposed_exact_governor_frequency_boost_and_power_values(self):
        obj = self.provider()
        paths = obj.profile["policy_files"]["cpu_policy"]
        before = copy.deepcopy(obj.profile)
        expected = {name: {path: "observed:" + path for path in files} for name, files in paths.items()}
        self.assertEqual(obj._policy_files("cpu_policy"), expected)
        self.assertEqual(obj.profile, before)
        self.assertEqual(obj.io.mock_calls[0].args, (self.ROOT,))
        self.assertEqual(obj.io.mock_calls[0][0], "listdir")
        self.assertEqual([call.args[0] for call in obj.io.text.call_args_list],
                         [path for files in paths.values() for path in files])

    def test_exposed_governor_and_every_frequency_path_coverage_remains_exact(self):
        for category in ("governors", "frequency_limits"):
            for missing in self.provider().profile["policy_files"]["cpu_policy"][category]:
                obj = self.provider()
                obj.profile["policy_files"]["cpu_policy"][category].remove(missing)
                with self.subTest(category=category, missing=missing), self.assertRaisesRegex(InvalidEvidence, "coverage"):
                    obj._policy_files("cpu_policy")
            obj = self.provider()
            obj.profile["policy_files"]["cpu_policy"][category].append("/sys/extra")
            with self.subTest(category=category, extra=True), self.assertRaisesRegex(InvalidEvidence, "coverage"):
                obj._policy_files("cpu_policy")

    def test_exposed_policy_directory_set_drift_rejects(self):
        for policies in (["policy0"], ["policy0", "policy2", "policy3"]):
            obj = self.provider()
            obj.io.listdir.return_value = policies
            with self.subTest(policies=policies), self.assertRaisesRegex(InvalidEvidence, "coverage"):
                obj._policy_files("cpu_policy")

    def test_exposed_configured_path_read_failure_cannot_fall_back(self):
        for category in self.CATEGORIES:
            for error in (FileNotFoundError, PermissionError, OSError):
                obj = self.provider()
                failed_path = obj.profile["policy_files"]["cpu_policy"][category][0]
                def read(path):
                    if path == failed_path:
                        raise error("configured policy path unavailable")
                    return "observed"
                obj.io.text.side_effect = read
                with self.subTest(category=category, error=error), self.assertRaises(error):
                    obj._policy_files("cpu_policy")

    def test_empty_launch_cpu_policy_is_accepted_by_existing_parser(self):
        value = numa_baseline()
        value["cpu_policy"] = self.empty_observation()
        manifest = lb.LaunchManifest.parse(value)
        self.assertEqual(manifest.values()["cpu_policy"], self.empty_observation())
        manifest.compare(copy.deepcopy(value))

    def test_absent_confinement_matches_only_absent_authority(self):
        obj = ProviderReceiptTests().base()
        baseline = numa_baseline()
        baseline["cpu_policy"] = self.provider(True)._policy_files("cpu_policy")
        obj.baseline = lb.LaunchManifest.parse(baseline)
        obj._launch_observed.return_value = copy.deepcopy(baseline)
        self.assertEqual(obj._observe_confinement()["launch_observed"]["cpu_policy"], self.empty_observation())

    def test_confinement_rejects_both_form_transitions(self):
        for absent in (False, True):
            obj = ProviderReceiptTests().base()
            baseline = numa_baseline()
            baseline["cpu_policy"] = self.provider(absent)._policy_files("cpu_policy")
            obj.baseline = lb.LaunchManifest.parse(baseline)
            observed = copy.deepcopy(baseline)
            observed["cpu_policy"] = self.provider(not absent)._policy_files("cpu_policy")
            obj._launch_observed.return_value = observed
            with self.subTest(absent=absent), self.assertRaisesRegex(InvalidEvidence, "frozen launch profile drift"):
                obj._observe_confinement()

    def test_confinement_rejects_exposed_path_and_value_drift_in_every_category(self):
        for category in self.CATEGORIES:
            for change in ("path", "value"):
                obj = ProviderReceiptTests().base()
                baseline = numa_baseline()
                baseline["cpu_policy"] = self.provider()._policy_files("cpu_policy")
                obj.baseline = lb.LaunchManifest.parse(baseline)
                observed = copy.deepcopy(baseline)
                values = observed["cpu_policy"][category]
                path = next(iter(values))
                if change == "path":
                    values["/sys/changed"] = values.pop(path)
                else:
                    values[path] = "changed"
                obj._launch_observed.return_value = observed
                with self.subTest(category=category, change=change), self.assertRaisesRegex(InvalidEvidence, "frozen launch profile drift"):
                    obj._observe_confinement()


class RefreshedAuthorityTests(unittest.TestCase):
    """Pure synthetic grant checks; no complete external live grant is issued."""

    def scope(self):
        return {"authority_issue": 245, "host_id": "fixture", "attempt": "p1v-fixture",
                "operations": [a.name for a in plan_fixture()], "provider_manifest_sha256": "a" * 64, "storage_identity": storage_fixture()}

    def test_exact_floor(self):
        self.assertEqual(ls.FLOOR, 244)

    def test_every_nonnegative_authority_through_242_rejects_before_host_access(self):
        for issue in range(245):
            scope = dict(self.scope(), authority_issue=issue)
            authorization = object.__new__(ls.FutureAuthorization)
            authorization.issue = issue
            with self.subTest(issue=issue), patch.object(lb, "require_adapter_grant", side_effect=AssertionError("adapter")), \
                    patch("os.open", side_effect=AssertionError("host access")), \
                    patch("os.listdir", side_effect=AssertionError("host access")), \
                    patch("os.getpid", side_effect=AssertionError("host access")), \
                    patch("ctypes.CDLL", side_effect=AssertionError("library")):
                with self.assertRaisesRegex(InvalidEvidence, "follow #244"):
                    ls.ProviderGrant(scope, digest(canonical_json(scope)), b"x" * 32, digest(b"x" * 32))
                with self.assertRaisesRegex(InvalidEvidence, "follow #244"):
                    ls.require_live(authorization, None, None, "fixture", "p1v-fixture", {}, {}, "AUTHORITY")

    def test_synthetic_243_complete_scope_passes_only_pure_constructor_check(self):
        receiver, scope = SimpleNamespace(), self.scope()
        ls.ProviderGrant.__init__(receiver, scope, digest(canonical_json(scope)), b"x" * 32, digest(b"x" * 32))
        self.assertEqual(strict_json(receiver.encoded), scope)
        # This fake receiver cannot satisfy the exact ProviderGrant type gate.
        authorization = ls.FutureAuthorization(b"x" * 32, digest(b"x" * 32), 245)
        with self.assertRaisesRegex(InvalidEvidence, "provider-specific grant required"):
            ls.require_live(authorization, None, receiver, "fixture", "p1v-fixture", {}, {}, "AUTHORITY")

    def test_synthetic_243_requires_complete_valid_scope(self):
        for field in self.scope():
            scope = self.scope()
            del scope[field]
            with self.subTest(missing=field), self.assertRaises(InvalidEvidence):
                ls.ProviderGrant.__init__(SimpleNamespace(), scope, digest(canonical_json(scope)), b"x" * 32, digest(b"x" * 32))
        for field, invalid in (("authority_issue", True), ("host_id", ""), ("attempt", "../escape"),
                               ("operations", ["AUTHORITY"]), ("provider_manifest_sha256", "bad")):
            scope = dict(self.scope(), **{field: invalid})
            with self.subTest(field=field), self.assertRaises(InvalidEvidence):
                ls.ProviderGrant.__init__(SimpleNamespace(), scope, digest(canonical_json(scope)), b"x" * 32, digest(b"x" * 32))

    def test_synthetic_243_provider_bindings_and_adapter_grant_still_required(self):
        provider = bundle_fixture()["provider"]
        scope = dict(self.scope(), provider_manifest_sha256=digest(canonical_json(provider)))
        authorization = ls.FutureAuthorization(b"x" * 32, digest(b"x" * 32), 245)
        # Uninitialized provider receiver only: no adapter grant or live System.
        receiver = object.__new__(ls.ProviderGrant)
        object.__setattr__(receiver, "encoded", canonical_json(scope))
        with self.assertRaisesRegex(InvalidEvidence, "adapter-specific external execution grant required"):
            ls.require_live(authorization, None, receiver, "fixture", "p1v-fixture", {}, provider, "AUTHORITY")
        for field, mismatch in (("authority_issue", 244), ("host_id", "other"), ("attempt", "p1v-other"),
                                ("operations", []), ("provider_manifest_sha256", "0" * 64)):
            object.__setattr__(receiver, "encoded", canonical_json(dict(scope, **{field: mismatch})))
            with self.subTest(field=field), self.assertRaisesRegex(InvalidEvidence, "provider grant mismatch"):
                ls.require_live(authorization, None, receiver, "fixture", "p1v-fixture", {}, provider, "AUTHORITY")



class ExternalStorageProviderTests(unittest.TestCase):
    def grants(self, storage, provider_storage=None):
        adapter = adapter_fixture()
        provider = {"synthetic": True}
        execution_scope = {"authority_issue": 245, "host_id": "synthetic-host", "attempt": "p1v-fixture",
                           "operations": [a.name for a in plan_fixture()],
                           "adapter_manifest_sha256": digest(canonical_json(adapter)),
                           "storage_identity": storage}
        execution_token = b"synthetic pure validation only!!!" * 2
        authorization = ls.FutureAuthorization(execution_token, digest(execution_token), 245)
        execution = lb.ExecutionGrant(execution_scope, digest(canonical_json(execution_scope)),
                                      execution_token, digest(execution_token))
        scope = {"authority_issue": 245, "host_id": "synthetic-host", "attempt": "p1v-fixture",
                 "operations": [a.name for a in plan_fixture()], "provider_manifest_sha256": digest(canonical_json(provider)),
                 "storage_identity": storage if provider_storage is None else provider_storage}
        token = b"synthetic provider storage proof!!" * 2
        grant = ls.ProviderGrant(scope, digest(canonical_json(scope)), token, digest(token))
        return authorization, execution, grant, adapter, provider, scope

    def test_cross_grants_equal_before_any_host_access_and_freeze_scope(self):
        from dataclasses import FrozenInstanceError
        for second in (False, True):
            storage = storage_fixture(second)
            authorization, execution, grant, adapter, provider, scope = self.grants(storage)
            with patch("os.open", side_effect=AssertionError("host access")), patch("ctypes.CDLL", side_effect=AssertionError("library")):
                ls.require_live(authorization, execution, grant, "synthetic-host", "p1v-fixture", adapter, provider, "AUTHORITY")
                context = (authorization, execution, grant, "synthetic-host", "p1v-fixture", adapter, provider)
                io = ls.DirectLinux(context, lambda: None)
            scope["storage_identity"]["by_id_aliases"].append("changed")
            grant.storage_identity["major_minor"] = "1:2"
            self.assertEqual(execution.storage_identity, grant.storage_identity)
            self.assertEqual(io.storage_identity, storage_fixture(second))
            with self.assertRaises(FrozenInstanceError): grant.encoded = b"{}"
            authorization, execution, grant, adapter, provider, _ = self.grants(storage_fixture(second), storage_fixture(not second))
            with patch("os.open", side_effect=AssertionError("host access")), patch("ctypes.CDLL", side_effect=AssertionError("library")):
                with self.assertRaisesRegex(InvalidEvidence, "independent storage grants"):
                    ls.require_live(authorization, execution, grant, "synthetic-host", "p1v-fixture", adapter, provider, "AUTHORITY")
                with self.assertRaisesRegex(InvalidEvidence, "independent storage grants"):
                    ls.LiveSystem(authorization=authorization, grant=execution, provider_grant=grant,
                                  host_id="synthetic-host", attempt="p1v-fixture", manifest=adapter,
                                  provider_manifest=provider, baseline={}, closure={}, journal=None,
                                  native=None, processes=None, roles=None, bootstrap=None)

    def test_provider_constructor_rejects_noncanonical_storage(self):
        for field in storage_fixture():
            storage = storage_fixture(); del storage[field]
            with self.subTest(field=field), self.assertRaises(InvalidEvidence):
                self.grants(storage_fixture(), storage)

    def device(self, second, variant=None):
        storage = storage_fixture(second)
        obj = object.__new__(ls.LiveSystem)
        obj._storage_identity = canonical_json(storage)
        block = SimpleNamespace(st_mode=stat.S_IFBLK | 0o600, st_rdev=999, mm=storage["major_minor"])
        obj.io = Mock()
        def observed(path, follow=False):
            if path.endswith("/partition"):
                if variant == "partition": return SimpleNamespace(st_mode=stat.S_IFREG)
                raise FileNotFoundError(path)
            info = copy.copy(block)
            if path == storage["device_path"]:
                if variant == "rdev": info.mm = "1:2"
                if variant == "symlink": info.st_mode = stat.S_IFLNK
            elif "/by-uuid/" in path and variant == "uuid_alias": info.st_rdev += 1
            elif "/by-id/" in path and variant == "by_id_rdev": info.st_rdev += 1
            return info
        def listing(path):
            if path == "/dev/disk/by-id":
                return [storage["by_id_aliases"][0].rsplit("/", 1)[1]] + (["google-local-ssd-1"] if variant == "extra_alias" else [])
            if path.endswith("/holders"): return ["holder"] if variant == "holders" else []
            if path.endswith("/slaves"): return ["slave"] if variant == "slaves" else []
            return ["arbitrary-device-child"] if variant == "partition" else (["partition"] if variant == "is_partition" else [])
        obj.io.stat.side_effect = observed
        obj.io.listdir.side_effect = listing
        obj.io.text.side_effect = lambda path: ("1:2" if variant == "sysfs_dev" else storage["major_minor"]) if path.endswith("/dev") else str(storage["disk_sequence"] + int(variant == "diskseq"))
        with patch.object(ls, "major_minor", side_effect=lambda info, device=False: info.mm), \
             patch("os.path.realpath", return_value=storage["sysfs_identity"] + ("-changed" if variant == "sysfs" else "")):
            return obj._device()

    def test_device_checks_every_granted_coordinate_for_both_storage(self):
        for second in (False, True):
            self.assertEqual(self.device(second), storage_fixture(second))
            for variant in ("rdev", "symlink", "sysfs", "sysfs_dev", "diskseq", "uuid_alias", "by_id_rdev", "extra_alias", "holders", "slaves", "partition", "is_partition"):
                with self.subTest(second=second, variant=variant), self.assertRaises(InvalidEvidence):
                    self.device(second, variant)

    def test_metadata_checks_provider_storage_for_both_storage_and_cross_rejects(self):
        from .supervisor import metadata_gate
        for second in (False, True):
            storage = storage_fixture(second)
            records = MetadataTests.metadata(self, storage=storage)["records"]
            metadata_gate(records, storage)
            with self.assertRaisesRegex(InvalidEvidence, "wrong expert device"):
                MetadataTests.metadata(self, storage=storage, file_storage=storage_fixture(not second))

    def test_dynamic_read_dependency_and_gpu_backing_exclusions(self):
        for second in (False, True):
            storage = storage_fixture(second)
            io = object.__new__(ls.DirectLinux)
            io._storage_identity = canonical_json(storage)
            io.check = lambda: None
            with patch("os.open", return_value=40), patch("os.close"), \
                 patch("os.fstat", return_value=SimpleNamespace(st_mode=stat.S_IFREG)), \
                 patch.object(ls, "major_minor", return_value=storage["major_minor"]), \
                 patch("os.read", side_effect=AssertionError("protected payload read")):
                for method in (io.read, io.file_hash):
                    with self.subTest(method=method.__name__), self.assertRaisesRegex(InvalidEvidence, "expert backing"):
                        method("/synthetic/dependency")
            obj = object.__new__(ls.LiveSystem)
            obj._storage_identity = canonical_json(storage)
            obj.io = Mock()
            obj.io.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR)
            obj.profile = {"gpu_inventory_roots": {g: ["/synthetic/" + g] for g in ("nodes", "libraries", "icds", "layers", "cache")}}
            with patch.object(ls, "major_minor", return_value=storage["major_minor"]), self.assertRaisesRegex(InvalidEvidence, "expert backing"):
                obj._gpu_inventory()
            obj.io.listdir.assert_not_called()
            obj.io.file_hash.assert_not_called()

    def test_mount_source_spelling_cannot_replace_granted_major_minor(self):
        for second in (False, True):
            storage = storage_fixture(second)
            for source in (storage["device_path"], storage["by_id_aliases"][0], "UUID=" + storage["uuid"]):
                obj = object.__new__(ls.LiveSystem)
                obj._storage_identity = canonical_json(storage)
                obj._host = Mock(return_value="synthetic-boot")
                obj._device = Mock(return_value=storage)
                base = {"root": "/", "mount_ro": False, "superblock_ro": False, "mount_options": ["rw", "relatime"], "super_options": ["rw"]}
                data = dict(base, target=ls.MOUNT, major_minor=storage["major_minor"], fstype="ext4", source=source, mount_id=10)
                output = dict(base, target="/", major_minor="259:1", fstype="ext4", source="synthetic-root", mount_id=1)
                obj._mounts = Mock(return_value=[data, output])
                obj.io = Mock(); obj.io.link.return_value = ls.CANONICAL
                obj.native = Mock(); obj.native.retain_objects.return_value = {"device": 40, "filesystem": 41}
                obj._topology_expectations = Mock(return_value={})
                obj.backing = None
                self.assertEqual(obj._observe_topology()["device_identity"], storage)
                obj.backing = None
                data["major_minor"] = storage_fixture(not second)["major_minor"]
                with self.assertRaisesRegex(InvalidEvidence, "original ext4 RW mount"):
                    obj._observe_topology()

    def test_both_storage_ext4_uuid_continuity_pre_ro_through_restored_rw(self):
        for second in (False, True):
            storage = storage_fixture(second)
            obj, data, record = DataFilesystemProfileTests.health_fixture(DataFilesystemProfileTests())
            obj._storage_identity = canonical_json(storage)
            obj.backing["identity"] = copy.deepcopy(storage)
            obj._device.return_value = copy.deepcopy(storage)
            record["major_minor"] = storage["major_minor"]
            data[0x68:0x78] = bytes.fromhex(storage["uuid"].replace("-", ""))
            self.assertTrue(obj._observe_sync_health()["syncfs_ok"])
            obj.ro_transition = True
            struct.pack_into("<I", data, 0x60, 0)
            record.update(mount_ro=True, superblock_ro=True, mount_options=["ro"])
            self.assertTrue(obj._observe_filesystem_ro()["journal_clean"])
            obj.rw_transition = True
            struct.pack_into("<I", data, 0x60, 4)
            record.update(mount_ro=False, superblock_ro=False, mount_options=["rw"])
            self.assertTrue(obj._observe_filesystem_rw()["recovery_required"])
            data[0x68:0x78] = bytes.fromhex(storage_fixture(not second)["uuid"].replace("-", ""))
            with self.assertRaisesRegex(InvalidEvidence, "same retained ext4 device/UUID"):
                obj._observe_filesystem_rw()


if __name__ == "__main__":
    unittest.main()
