"""Fail-closed P1V control machine and side-effect-free --plan.

Host operations are an explicit adapter boundary: #214 implements and tests
their ordered contract, not a deployable privileged launcher. FutureBackend
requires separate authorization and a reviewed executor; no shell executor,
sudo, automatic recovery, or live backend is installed by this package.

An executor must durably append each intent/result under supervisor ownership,
implement kernel transitions and inspect their results (not echo expectations),
hold administrative exclusion/control references, and run the parent once.
FakeBackend is explicitly labelled SIMULATED and never qualifies real evidence.
"""

import argparse
import copy
import re
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from . import (ALIAS_PARENT, AUTHORITY, CANONICAL, CONFIGURED, DEVICE, EXPECTATIONS,
               IDS, MOUNT, STARTUP_IDS, TOOLS, InvalidEvidence, canonical_json,
               digest, require, require_future, sha256, tool_hashes, uint)
from .content_hasher import Identity, alternatives, verify_content

BIN = "/home/randyap8/predictor-v2-p1r-lineage-fix-linux-gate-20260926T184626Z/cargo-target/release/micro-expert-router"
CONFIG = "/home/randyap8/slice11-qwen3-coder-gpu-native.toml"
REQUEST = "/home/randyap8/predictor-v2-p1r-critical-path-corrected-candidate-gcp-l4-20260926T190511Z/evidence/request.json"
OUTPUT_ROOT = "/home/randyap8/predictor-v2-p1v"
HASH_WINDOW = {"phase": "POST_SHUTDOWN", "runtime_tree_empty": True,
               "filesystem_ro": True, "device_ro": True,
               "alias_locked": True, "exclusion_held": True}


@dataclass(frozen=True)
class Action:
    name: str
    phase: str
    operation: str
    argv: tuple = ()


def validate_attempt(attempt):
    require(isinstance(attempt, str) and re.fullmatch(r"p1v-[a-z0-9][a-z0-9-]{0,63}", attempt),
            "new P1V attempt ID required")
    return attempt


def parent_argv(attempt):
    validate_attempt(attempt)
    return (BIN, "diagnose-gpu-native-predictor-v2-critical-path-attribution",
            "--config", CONFIG, "--request-json", REQUEST,
            "--expected-adapter-name", "NVIDIA L4", "--diagnostic-id", attempt,
            "--report-out", f"{OUTPUT_ROOT}/{attempt}/p1r.json")


def plan(attempt="p1v-plan"):
    validate_attempt(attempt)
    return (
        Action("AUTHORITY", "PRE", "verify exact source/input hashes, helper pins, new output paths, limits/environment"),
        Action("EXCLUSION", "PRE", "acquire trusted-host administrative exclusion; reject uncontrolled writers/raw holders/aliases"),
        Action("TOPOLOGY", "PRE", "verify frozen kernel/ext4/device/UUID, output backing and original healthy rw state"),
        Action("NAMESPACE", "PRE", "unshare(CLONE_NEWNS) in initial host user namespace; pin references; recursively private",
               ("mount", "--make-rprivate", "/")),
        Action("ALIAS_BIND", "PRE", "bind configured alias parent onto itself in private namespace",
               ("mount", "--bind", ALIAS_PARENT, ALIAS_PARENT)),
        Action("ALIAS_RO", "PRE", "lock alias parent; verify exact symlink target and no writable runtime alias",
               ("mount", "-o", "remount,bind,ro", ALIAS_PARENT)),
        Action("METADATA", "PRE", "bounded metadata/DIO-feature checks under runtime credentials; zero payload reads"),
        Action("FILESYSTEM_RO", "PRE", "normal filesystem-wide ext4 remount; finish journal/writeback while device is writable",
               ("mount", "-o", "remount,ro", MOUNT)),
        Action("DEVICE_RO", "PRE", "only after verified successful filesystem-ro transition",
               ("blockdev", "--setro", DEVICE)),
        Action("PROTECTED", "PRE", "verify superblock/device ro, alias lock, pinned namespace and continuous exclusion"),
        Action("CONFINEMENT", "PRE", "prepare unprivileged child handoff: all UID/GID slots, groups, zero capabilities, no_new_privs, seccomp, closed setup FDs, nondelegated cgroup"),
        Action("MARKER", "PRE", "consume new exclusive one-shot marker; fsync file and parent directory"),
        Action("P1R_PARENT", "PAIR", "one unchanged parent, serial CONTROL then TREATMENT; no external helper between arms", parent_argv(attempt)),
        Action("SHUTDOWN", "POST", "wait for parent exit and empty nondelegated runtime cgroup; reject any uncertain descendant"),
        Action("REVALIDATE", "POST", "revalidate protected backing, alias, namespace and held exclusion"),
        Action("HASH", "POST_SHUTDOWN", "direct-only six full files in fixed order; descriptor/path and PRE identity checks"),
        Action("VERIFY", "POST_SHUTDOWN", "verify immutable report hashes, raw evidence and exact 13 launches; offline derive"),
        Action("SEAL", "POST_SHUTDOWN", "exclusive durable content/derived/enforcement envelope; retain trusted root before restoration"),
        Action("DEVICE_RW", "RESTORE", "only after durable seal; leave filesystem ro",
               ("blockdev", "--setrw", DEVICE)),
        Action("FILESYSTEM_RW", "RESTORE", "only after verified device-rw and filesystem still healthy/ro; restore exact original options",
               ("mount", "-o", "remount,rw", MOUNT)),
        Action("RESTORATION_SEAL", "RESTORE", "append durable restoration record referencing original seal; never rewrite the seal"),
        Action("RELEASE", "RESTORE", "release private namespace/control references and administrative exclusion"),
    )


class Backend(Protocol):
    simulated: bool

    def perform(self, action: Action, evidence: dict) -> dict:
        """Execute once, return observed receipt, or raise; never retry."""

    def retain(self, record: dict) -> None:
        """Durably append no-clobber journal record, even on failure."""


class FutureBackend:
    """Gated integration seam for a separately reviewed, privileged host adapter.

    The adapter receives argv arrays, never shell text. No real executor ships
    under P1V2; --execute therefore refuses before any host or payload access.
    Authorization must be external, cannot be inferred from a plan or VM facts.
    """
    simulated = False

    def __init__(self, authorization, executor, journal):
        require_future(authorization)
        require(callable(executor) and callable(journal), "reviewed host adapter required")
        self.authorization = authorization
        self.executor, self.journal = executor, journal

    def perform(self, action, evidence):
        return self.executor(action, copy.deepcopy(evidence))

    def retain(self, record):
        self.journal(copy.deepcopy(record))


class FakeBackend:
    """In-memory operations only; supplied receipts are synthetic, never live."""
    simulated = True

    def __init__(self, receipts, fail_at=None, control_ok=True):
        self.receipts = copy.deepcopy(receipts)
        self.fail_at, self.control_ok = fail_at, control_ok
        self.trace, self.journal = [], []
        self.marker = False

    def retain(self, record):
        self.journal.append(copy.deepcopy(record))

    def perform(self, action, evidence):
        self.trace.append(action.name)
        if action.name == self.fail_at:
            raise InvalidEvidence("injected uncertain failure: " + action.name)
        if action.name == "MARKER":
            require(not self.marker, "attempt marker already exists")
            self.marker = True
        if action.name == "P1R_PARENT":
            self.trace.append("CONTROL")
            require(self.control_ok, "CONTROL failed; unchanged parent forbids TREATMENT")
            self.trace.append("TREATMENT")
        receipt = copy.deepcopy(self.receipts[action.name])
        if action.name == "SEAL":
            self.sealed_bytes = canonical_json(evidence)
            receipt["sha256"] = digest(self.sealed_bytes)
        return receipt


def protection(receipt):
    for key in ("filesystem_ro", "device_ro", "alias_locked", "exclusion_held",
                "namespace_pinned", "healthy", "no_uncontrolled_aliases"):
        require(receipt[key] is True, "protected interval uncertain: " + key)
    require(receipt["expectations"] == EXPECTATIONS, "backing/path drift")


def metadata_gate(records):
    require([r["global_expert_id"] for r in records] == list(STARTUP_IDS + IDS), "bounded metadata coverage")
    for record in records:
        g = record["global_expert_id"]
        require(record["selected"] in alternatives(g), "metadata selected path")
        require(record["canonical"] == CANONICAL + "/" + record["selected"].rsplit("/", 1)[1], "metadata escape")
        Identity(**record["identity"]).check()
        require(record["dio_eligible"] is True and record["o_direct_retained"] is True
                and record["payload_bytes_read"] == 0 and record["redirects"] == []
                and record["disallowed_features"] == [], "DIO/namespace metadata gate")


def receipt_gate(name, r, evidence):
    require(type(r) is dict and r["ok"] is True, "operation failed: " + name)
    previous = evidence["receipts"]
    if name == "AUTHORITY":
        require(r["authority"] == AUTHORITY and r["tools"] == evidence["tools"], "authority/tool mismatch")
        require(set(r["tools"]) == set(TOOLS), "tool scope")
        for value in r["tools"].values():
            sha256(value)
        for field in ("inputs_verified", "outputs_absent", "marker_absent", "frozen_environment", "frozen_limits"):
            require(r[field] is True, "preflight gate: " + field)
        require(r["argv"] == list(parent_argv(evidence["attempt"])), "parent invocation mismatch")
    elif name == "EXCLUSION":
        require(r["held"] is True and r["trusted_host"] is True
                and r["uncontrolled_writers"] == [] and r["raw_holders"] == []
                and r["uncontrolled_mount_namespaces"] == [], "trusted-host exclusion unavailable")
    elif name == "TOPOLOGY":
        require(r["expectations"] == EXPECTATIONS, "frozen host expectation mismatch")
        require(r["filesystem_rw"] is True and r["device_rw"] is True and r["healthy"] is True,
                "unknown original filesystem/device state")
        require(r["external_journal"] is False and r["stacked_or_overlapping_devices"] == []
                and r["uncontrolled_aliases"] == [] and r["writes_on_data_backing"] == [], "unsupported topology")
        require(r["boot_id"] and r["disk_sequence"] and r["mount_id"] and r["original_mount_options"], "host identity missing")
        require(r["output_major_minor"] == "259:1", "evidence/cache backing mismatch")
    elif name == "NAMESPACE":
        require(r["private_recursive"] is True and r["initial_user_namespace"] is True
                and r["references_pinned"] is True and r["namespace_id"], "namespace preparation failed")
    elif name == "ALIAS_BIND":
        require(r["path"] == ALIAS_PARENT and r["private_bind"] is True, "alias bind failed")
    elif name == "ALIAS_RO":
        require(r["alias_parent_ro"] is True and r["configured"] == CONFIGURED
                and r["symlink_target"] == CANONICAL and r["runtime_cannot_replace"] is True,
                "configured alias lockdown failed")
    elif name == "METADATA":
        metadata_gate(r["records"])
    elif name == "FILESYSTEM_RO":
        require(r["superblock_ro"] is True and r["normal_unforced_transition"] is True
                and r["journal_writeback_complete"] is True and r["device_still_rw"] is True
                and r["healthy"] is True, "filesystem-ro transition uncertain")
    elif name == "DEVICE_RO":
        require(r["device_ro"] is True and r["filesystem_still_ro"] is True, "device-ro transition uncertain")
    elif name in ("PROTECTED", "REVALIDATE"):
        protection(r)
        require(r["namespace_id"] == previous["NAMESPACE"]["namespace_id"], "namespace drift")
        for key in ("boot_id", "disk_sequence", "mount_id"):
            require(r[key] == previous["TOPOLOGY"][key], "host identity drift: " + key)
    elif name == "CONFINEMENT":
        require(len(r["uids"]) == 3 and len(set(r["uids"])) == 1 and uint(r["uids"][0]) > 0,
                "runtime privileged UID")
        require(len(r["gids"]) == 3 and len(set(r["gids"])) == 1 and uint(r["gids"][0]) > 0,
                "runtime privileged GID")
        require(r["capabilities"] == {k: 0 for k in ("effective", "permitted", "inheritable", "ambient", "bounding")}, "runtime capabilities")
        for key in ("no_new_privs", "groups_reviewed", "brokers_excluded", "setup_fds_closed",
                    "seccomp_enforced", "cgroup_nondelegated", "namespace_apis_denied", "raw_device_access_denied"):
            require(r[key] is True, "runtime confinement: " + key)
        sha256(r["policy_sha256"])
        require(r["inherited_fds"] == [0, 1, 2] and r["cgroup_id"], "runtime FD/cgroup boundary")
    elif name == "MARKER":
        require(r["created_exclusive"] is True and r["durable"] is True
                and r["attempt"] == evidence["attempt"], "one-shot marker failed")
    elif name == "P1R_PARENT":
        require(r["invocations"] == 1 and r["argv"] == previous["AUTHORITY"]["argv"]
                and r["external_actions_between_arms"] == [] and r["exit_code"] == 0,
                "unchanged one-pair parent contract failed")
        require(r["report_origin"] == "created_by_this_invocation" and r["attempt"] == evidence["attempt"], "new-run binding absent")
        require(set(r["raw_report_hashes"]) == {"parent", "control", "treatment"}, "report coverage")
        for h in r["raw_report_hashes"].values():
            sha256(h)
    elif name == "SHUTDOWN":
        require(r["parent_reaped"] is True and r["runtime_tree_empty"] is True
                and r["uncertain_descendants"] == [] and r["cgroup_id"] == previous["CONFINEMENT"]["cgroup_id"],
                "whole runtime tree shutdown uncertain")
    elif name == "HASH":
        require(r["window"] == HASH_WINDOW and r["hasher_stopped"] is True, "hasher lifecycle uncertain")
        verify_content(r["content"])
        selected = previous["METADATA"]["records"][-6:]
        for pre, post in zip(selected, r["content"]["records"]):
            require(pre["selected"] == post["authoritative_selected_path"]
                    and pre["canonical"] == post["canonical_path"] and pre["identity"] == post["identity"], "pre/hash identity drift")
    elif name == "VERIFY":
        require(r["raw_report_hashes"] == previous["P1R_PARENT"]["raw_report_hashes"]
                and r["derived"]["semantic_payload_equivalent"] is True
                and r["derived"]["exact_launch_gate"] == "PASS"
                and r["derived"]["performance_verdict"] == "NOT_AUTHORIZED", "offline verification failed")
    elif name == "SEAL":
        require(r["durable"] is True and r["exclusive"] is True and r["runtime_cannot_write"] is True,
                "durable evidence seal unavailable")
        require(r["sha256"] == digest(canonical_json(evidence)), "seal bytes mismatch")
    elif name == "DEVICE_RW":
        require(r["device_rw"] is True and r["filesystem_still_ro"] is True
                and r["runtime_tree_empty"] is True and r["hasher_stopped"] is True,
                "device restoration uncertain")
    elif name == "FILESYSTEM_RW":
        require(r["filesystem_rw"] is True and r["device_rw"] is True and r["healthy"] is True
                and r["mount_options"] == previous["TOPOLOGY"]["original_mount_options"], "filesystem restoration uncertain")
    elif name == "RESTORATION_SEAL":
        require(r["durable"] is True and r["exclusive"] is True
                and r["evidence_seal_sha256"] == previous["SEAL"]["sha256"], "restoration record not sealed")
    elif name == "RELEASE":
        require(r["released"] is True, "exclusion release uncertain")
    else:
        raise InvalidEvidence("unknown action")


class Supervisor:
    def __init__(self, backend: Backend, attempt, clock=time.monotonic_ns):
        if not backend.simulated:
            require(isinstance(backend, FutureBackend), "live backend must pass future authorization interlock")
            require_future(backend.authorization)
        self.backend, self.attempt, self.clock = backend, validate_attempt(attempt), clock
        self.used = False

    def run(self):
        require(not self.used, "one-shot supervisor cannot be retried")
        self.used = True
        evidence = {"schema": "mer.p1v.enforcement.v1", "attempt": self.attempt,
                    "mode": "SIMULATED" if self.backend.simulated else "LIVE",
                    "authority": copy.deepcopy(AUTHORITY), "tools": tool_hashes(),
                    "events": [], "receipts": {}}
        previous_time = -1
        for step, action in enumerate(plan(self.attempt)):
            now = uint(self.clock())
            require(now >= previous_time, "monotonic clock regressed")
            previous_time = now
            intent = {"step": step, "action": action.name, "phase": action.phase,
                      "monotonic_ns": now, "status": "INTENT"}
            try:
                self.backend.retain(intent)
                receipt = self.backend.perform(action, copy.deepcopy(evidence))
                receipt_gate(action.name, receipt, evidence)
                event = dict(intent, status="PASS", receipt_sha256=digest(canonical_json(receipt)))
                self.backend.retain(event)
                evidence["events"].append(event)
                evidence["receipts"][action.name] = receipt
            except Exception as error:
                failure = dict(intent, status="FAIL", error=str(error), restoration="NOT_ATTEMPTED_FURTHER")
                # A failed journal itself is a stop. Never run a finally-rw trap.
                try:
                    self.backend.retain(failure)
                except Exception as journal_error:
                    failure["journal_error"] = str(journal_error)
                evidence["failure"] = failure
                evidence["status"] = "FAIL"
                return evidence
        evidence["status"] = "PASS_SIMULATION" if self.backend.simulated else "PASS_FOR_REVIEW"
        return evidence


def validate_enforcement(envelope, allow_simulated=False, through_hash=False):
    """Validate the pre-restoration sealed prefix, not just endpoint ro booleans."""
    require(envelope["schema"] == "mer.p1v.enforcement.v1", "enforcement schema")
    require(envelope["mode"] == "LIVE" or (allow_simulated and envelope["mode"] == "SIMULATED"),
            "synthetic evidence cannot qualify a live run")
    require(envelope["authority"] == AUTHORITY, "enforcement authority mismatch")
    require(set(envelope["tools"]) == set(TOOLS), "enforcement tool coverage")
    for value in envelope["tools"].values():
        sha256(value)
    actions = plan(envelope["attempt"])
    prefix = actions[:16 if through_hash else 17]  # strictly before seal/restoration
    require([e["action"] for e in envelope["events"]] == [a.name for a in prefix], "continuity event order/coverage")
    require(set(envelope["receipts"]) == {a.name for a in prefix}, "continuity receipt coverage")
    replay = {k: copy.deepcopy(envelope[k]) for k in ("schema", "attempt", "mode", "authority", "tools")}
    replay.update(events=[], receipts={})
    last = -1
    for step, (action, event) in enumerate(zip(prefix, envelope["events"])):
        require(event["step"] == step and event["phase"] == action.phase
                and event["status"] == "PASS" and uint(event["monotonic_ns"]) >= last, "transition chronology")
        last = event["monotonic_ns"]
        receipt = envelope["receipts"][action.name]
        require(digest(canonical_json(receipt)) == event["receipt_sha256"], "receipt hash mismatch")
        receipt_gate(action.name, receipt, replay)
        replay["events"].append(event)
        replay["receipts"][action.name] = receipt
    return envelope["receipts"]["HASH"]["content"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--attempt", default="p1v-plan")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute or not args.plan:
        parser.error("NOT_AUTHORIZED: #214 supplies no live host adapter or future authorization token")
    output = {"mode": "PLAN_ONLY", "expectations_not_observations": EXPECTATIONS,
              "authority": AUTHORITY, "future_authorization_required": True,
              "startup_metadata_ids": STARTUP_IDS, "post_shutdown_hash_ids": IDS,
              "actions": [asdict(a) for a in plan(args.attempt)]}
    print(canonical_json(output).decode("ascii"), end="")


if __name__ == "__main__":
    main()
