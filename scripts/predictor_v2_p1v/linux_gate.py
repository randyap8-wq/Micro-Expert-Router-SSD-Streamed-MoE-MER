"""P1V5 plan/manifest CLI and externally authorized future Linux composition.

No live operation occurs on import. --execute under all current authorities
refuses before opening even a manifest, /proc, /sys, or a management library.
A future reviewed caller supplies all three external grants to compose().
"""

import argparse
import copy
import os
import stat
from pathlib import Path

from . import FutureAuthorization, InvalidEvidence, canonical_json, digest, require, strict_json
from . import linux_backend as lb
from .live_system import (FLOOR, NUMA_ABI, TOPICS, PROTECTED, REPORT_NAMES, REPORT_LOGS,
                          DirectLinux, LiveSystem, ProviderGrant, controlled_directory,
                          major_minor, observe_host_identity, require_live, validate_provider_manifest)
from .supervisor import OUTPUT_ROOT, Supervisor, parent_argv, plan, validate_attempt


def planning(attempt):
    validate_attempt(attempt)
    return {"schema": "mer.p1v.linux-gate-plan.v1", "attempt": attempt,
            "implementation_issue": 217, "live_authority_must_exceed": FLOOR,
            "execution": "NOT_AUTHORIZED", "runtime_report_root": OUTPUT_ROOT,
            "trusted_evidence_root": "REQUIRED_DISTINCT_ROOT_CONTROLLED_OUTPUT_BACKING",
            "output_major_minor": "259:1", "parent_argv": list(parent_argv(attempt)),
            "report_names": copy.deepcopy(REPORT_NAMES), "report_logs": list(REPORT_LOGS),
            "topics": list(TOPICS), "numa_abi": copy.deepcopy(NUMA_ABI),
            "actions": [a.name for a in plan(attempt)],
            "ext4_phases": ["PRE_RO_SYNCED_ACTIVE", "RO_QUIESCED_CLEAN", "RW_RESTORED_ACTIVE"],
            "hot_path_interposition": "NONE", "seccomp": "NOT_PRESENT",
            "cgroup": "PASSIVE_TREE_ONLY", "cgroup_device_bpf": "NOT_PRESENT"}


def validate_bundle(bundle):
    lb.exact_keys(bundle, ("adapter", "provider", "launch", "closure"), "gate manifest bundle")
    return validate_provider_manifest(bundle["provider"], bundle["adapter"], bundle["launch"], bundle["closure"])


def bootstrap_outputs(io, attempt, launch, profile):
    """Future only; no-clobber runtime attempt separate from root evidence."""
    io.check()
    validate_attempt(attempt)
    runtime, evidence = OUTPUT_ROOT, profile["trusted_evidence_root"]
    require(runtime != evidence and not evidence.startswith(runtime + "/")
            and not runtime.startswith(evidence + "/"), "runtime/evidence roots must be separate")
    ids = launch["credentials"]
    evidence_stat = controlled_directory(io, evidence, ids["uid"])
    require(major_minor(evidence_stat) == "259:1", "trusted evidence on wrong backing")
    # The approved runtime root is pre-provisioned, with root-controlled parents.
    for part in (Path(runtime), *Path(runtime).parents):
        value = io.stat(str(part))
        require(stat.S_ISDIR(value.st_mode) and value.st_uid in (0, ids["uid"])
                and not value.st_mode & 0o022, "runtime root ancestor ownership/mode")
    runtime_stat = io.stat(runtime)
    require(major_minor(runtime_stat) == "259:1", "runtime reports on wrong backing")
    root = Path(runtime) / attempt
    root.mkdir(mode=0o700)
    os.chown(root, ids["uid"], ids["gid"], follow_symlinks=False)
    os.chmod(root, 0o700, follow_symlinks=False)
    info = io.stat(str(root))
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == ids["uid"] and info.st_gid == ids["gid"]
            and stat.S_IMODE(info.st_mode) == 0o700 and major_minor(info) == "259:1", "runtime attempt ownership/backing")
    require(io.listdir(str(root)) == [], "new runtime reports must be absent")
    lb.DurableJournal._sync_dir(root.parent)
    values = {"attempt": attempt, "runtime_report_root": runtime, "runtime_attempt": str(root),
              "trusted_evidence_root": evidence, "major_minor": "259:1",
              "runtime_directory_identity": {"device": major_minor(info), "inode": info.st_ino,
                    "uid": info.st_uid, "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode)},
              "runtime_owner": ids["uid"], "runtime_group": ids["gid"], "runtime_mode": 0o700,
              "evidence_owner": evidence_stat.st_uid, "evidence_mode": stat.S_IMODE(evidence_stat.st_mode),
              "expected_absent": list(REPORT_NAMES.values()) + list(REPORT_LOGS),
              "runtime_cannot_write_evidence": True}
    return values | {"bootstrap_sha256": digest(canonical_json(values))}


def compose(bundle, *, authorization, execution_grant, provider_grant, host_id, attempt):
    """Future-only exact composition. Never creates or infers an authorization."""
    require_live(authorization, execution_grant, provider_grant, host_id, attempt,
                 bundle["adapter"], bundle["provider"], "AUTHORITY")
    validate_bundle(bundle)
    context = (authorization, execution_grant, provider_grant, host_id, attempt,
               bundle["adapter"], bundle["provider"])
    check = lambda: require_live(*context, "AUTHORITY")
    io = DirectLinux(context, check)
    # Code bindings must precede host discovery and output mutations.
    for name, expected in bundle["provider"]["modules"].items():
        require(io.file_hash(str(Path(__file__).parent / name)) == expected, "provider code hash mismatch")
    for name, expected in PROTECTED.items():
        require(io.file_hash(str(Path(__file__).parent / name)) == expected, "protected source hash mismatch")
    for path, expected in bundle["provider"]["host_profile"]["dependency_files"].items():
        require(io.file_hash(path) == expected, "direct dependency hash mismatch")
    observe_host_identity(io, host_id)
    # Bootstrap needs admission before DurableJournal can create trusted output.
    bootstrap = bootstrap_outputs(io, attempt, bundle["launch"], bundle["provider"]["host_profile"])
    journal = lb.DurableJournal(bundle["provider"]["host_profile"]["trusted_evidence_root"], attempt,
        authorization=authorization, grant=execution_grant, host_id=host_id, manifest=bundle["adapter"],
        backing={"root_owned": True, "controlled": True, "major_minor": "259:1",
                 "runtime_cannot_write": True, "no_symlink_components": True})
    current_action = lambda: journal.pending[1] if journal.pending else "AUTHORITY"
    native = lb.NativePrimitives(authorization, execution_grant, host_id, attempt, bundle["adapter"], current_action)
    processes = lb.NativeProcesses(authorization, execution_grant, host_id, attempt, bundle["adapter"])
    roles = lb.ProcessRoles(processes, authorization=authorization, grant=execution_grant,
                            host_id=host_id, attempt=attempt, manifest=bundle["adapter"])

    def worker():
        system = LiveSystem(authorization=authorization, grant=execution_grant,
            provider_grant=provider_grant, host_id=host_id, attempt=attempt, manifest=bundle["adapter"],
            provider_manifest=bundle["provider"], baseline=bundle["launch"], closure=bundle["closure"],
            journal=journal, native=native, processes=processes, roles=roles, bootstrap=bootstrap)
        backend = lb.LinuxBackend(system, journal, bundle["launch"], bundle["adapter"], bundle["closure"],
                                 host_id=host_id, attempt=attempt, authorization=authorization, grant=execution_grant)
        return Supervisor(backend, attempt).run()

    return roles.guardian(worker)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="p1v-plan")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--validate-manifest")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        # No CLI switch/token can grant execution. A future reviewed entry point
        # must call compose with all externally pinned matching grants >242.
        parser.error("P1V5 execution NOT_AUTHORIZED; external future grants for an issue >242 required")
    try:
        if args.validate_manifest:
            value = strict_json(Path(args.validate_manifest).read_bytes())
            print(canonical_json({"valid": True, "provider_manifest_sha256": validate_bundle(value)}).decode(), end="")
        else:
            print(canonical_json(planning(args.attempt)).decode(), end="")
    except (InvalidEvidence, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
