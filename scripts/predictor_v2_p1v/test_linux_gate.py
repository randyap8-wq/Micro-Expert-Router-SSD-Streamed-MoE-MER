"""Portable plan/grant/bootstrap validation; all host interfaces are mocked."""

import copy
import importlib
import io
import os
import stat
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import InvalidEvidence, canonical_json, digest
from . import linux_gate as gate
from . import live_system as ls
from .test_live_system import bundle_fixture


class GateTests(unittest.TestCase):
    def test_plan_side_effect_free(self):
        with patch("os.open", side_effect=AssertionError("host access")), patch("ctypes.CDLL", side_effect=AssertionError("library")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("file")):
            result = gate.planning("p1v-fixture")
        self.assertEqual(result["live_authority_must_exceed"], 248)
        self.assertEqual(result["implementation_issue"], 217)
        self.assertEqual(result["execution"], "NOT_AUTHORIZED")
        self.assertEqual(result["hot_path_interposition"], "NONE")
        self.assertEqual(result["runtime_report_root"], "/home/randyap8/predictor-v2-p1v")

    def test_import_has_no_host_effect(self):
        with patch("os.open", side_effect=AssertionError("host")), patch("ctypes.CDLL", side_effect=AssertionError("library")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")):
            importlib.reload(gate)

    def test_execute_denied_before_even_manifest_read(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("manifest opened")), \
             patch("os.open", side_effect=AssertionError("host opened")), patch("ctypes.CDLL", side_effect=AssertionError("library")), \
             redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as caught:
            gate.main(["--execute", "--validate-manifest", "/must-not-open"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("NOT_AUTHORIZED", error.getvalue())
        self.assertIn("issue >248", error.getvalue())

    def test_execute_still_denied_with_plan_and_future_looking_arguments(self):
        with patch.object(gate, "planning", side_effect=AssertionError("plan")), \
             patch.object(gate, "compose", side_effect=AssertionError("composition")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("manifest")), \
             patch("os.open", side_effect=AssertionError("host")), \
             patch("ctypes.CDLL", side_effect=AssertionError("library")), \
             redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as caught:
            gate.main(["--execute", "--plan", "--attempt", "p1v-issue-229", "--validate-manifest", "/must-not-open"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("issue >248", error.getvalue())

    def test_compose_requires_future_grants_before_io(self):
        with patch.object(gate, "DirectLinux", side_effect=AssertionError("host constructor")), self.assertRaises(InvalidEvidence):
            gate.compose(bundle_fixture(), authorization=None, execution_grant=None, provider_grant=None,
                         host_id="fixture", attempt="p1v-fixture")

    def test_manifest_validated_without_host_access(self):
        bundle = bundle_fixture()
        with patch("os.open", side_effect=AssertionError("host access")), patch("ctypes.CDLL", side_effect=AssertionError("library")):
            self.assertEqual(gate.validate_bundle(bundle), digest(canonical_json(bundle["provider"])))

    def test_manifest_missing_fields_rejected(self):
        bundle = bundle_fixture()
        for field in list(bundle["provider"]):
            value = copy.deepcopy(bundle)
            del value["provider"][field]
            with self.subTest(field=field), self.assertRaises(InvalidEvidence):
                gate.validate_bundle(value)

    def test_manifest_extra_field_rejected(self):
        value = bundle_fixture()
        value["provider"]["extra"] = True
        with self.assertRaises(InvalidEvidence):
            gate.validate_bundle(value)

    def test_bad_module_hash_rejected(self):
        for module in ("live_system.py", "linux_gate.py", "linux_backend.py"):
            value = bundle_fixture()
            value["provider"]["modules"][module] = "bad"
            with self.subTest(module=module), self.assertRaises(InvalidEvidence):
                gate.validate_bundle(value)

    def test_reviewed_adapter_pin_mandatory(self):
        value = bundle_fixture()
        value["provider"]["modules"]["linux_backend.py"] = "0" * 64
        with self.assertRaises(InvalidEvidence):
            gate.validate_bundle(value)

    def test_launch_and_closure_hash_bindings(self):
        for field in ("launch_manifest_sha256", "execution_closure_sha256"):
            value = bundle_fixture()
            value["provider"][field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(InvalidEvidence):
                gate.validate_bundle(value)

    def test_numa_abi_mismatch(self):
        value = bundle_fixture()
        value["provider"]["numa_abi"]["syscall"] = 0
        with self.assertRaises(InvalidEvidence):
            gate.validate_bundle(value)

    def test_gpu_normalization_explicit_dedicated_authority(self):
        for field in ("dedicated_host", "normalize_default_unlocked"):
            value = bundle_fixture()
            value["provider"]["nvml"][field] = False
            with self.subTest(field=field), self.assertRaises(InvalidEvidence):
                gate.validate_bundle(value)

    def test_evidence_root_separation(self):
        for path in (gate.OUTPUT_ROOT, gate.OUTPUT_ROOT + "/evidence", "/home/randyap8"):
            value = bundle_fixture()
            value["provider"]["host_profile"]["trusted_evidence_root"] = path
            with self.subTest(path=path), self.assertRaises(InvalidEvidence):
                gate.validate_bundle(value)

    def test_plan_cli(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(gate.main(["--plan", "--attempt", "p1v-fixture"]), 0)
        self.assertIn('"live_authority_must_exceed":248', output.getvalue())

    def test_no_current_external_grant_created(self):
        source = Path(gate.__file__).read_text()
        self.assertNotIn("ProviderGrant(", source)
        self.assertNotIn("FutureAuthorization(", source)
        self.assertNotIn("ExecutionGrant(", source)

    def test_bootstrap_denial_precedes_directory_creation(self):
        host = Mock()
        host.check.side_effect = InvalidEvidence("denied")
        with patch.object(Path, "mkdir", side_effect=AssertionError("directory")), self.assertRaises(InvalidEvidence):
            gate.bootstrap_outputs(host, "p1v-fixture", {}, {})

    def test_bootstrap_no_clobber(self):
        host = Mock()
        host.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0, st_gid=0,
                                                st_dev=os.makedev(259, 1))
        bundle = bundle_fixture()
        with patch.object(gate, "major_minor", return_value="259:1"), \
             patch.object(Path, "mkdir", side_effect=FileExistsError("reserved")) as mkdir, \
             patch("os.chown", side_effect=AssertionError("must not chown reused attempt")), self.assertRaises(FileExistsError):
            gate.bootstrap_outputs(host, "p1v-fixture", bundle["launch"], bundle["provider"]["host_profile"])
        mkdir.assert_called_once_with(mode=0o700)

    def test_bootstrap_wrong_backing(self):
        host = Mock()
        host.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0, st_gid=0,
                                                st_dev=os.makedev(259, 4))
        bundle = bundle_fixture()
        with patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), self.assertRaises(InvalidEvidence):
            gate.bootstrap_outputs(host, "p1v-fixture", bundle["launch"], bundle["provider"]["host_profile"])

    def test_evidence_ancestor_runtime_writable_rejected(self):
        host = Mock()
        host.stat.return_value = SimpleNamespace(st_mode=stat.S_IFDIR | 0o777, st_uid=0, st_gid=0,
                                                st_dev=os.makedev(259, 1))
        bundle = bundle_fixture()
        with patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), self.assertRaises(InvalidEvidence):
            gate.bootstrap_outputs(host, "p1v-fixture", bundle["launch"], bundle["provider"]["host_profile"])


if __name__ == "__main__":
    unittest.main()
