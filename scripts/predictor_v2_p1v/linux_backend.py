"""P1V4 Linux adapter, with injected host observations and syscall interfaces.

#216/#218/#220/#222 authorize implementation, never execution. Nothing happens on import.
Live integration needs an externally reviewed >230 grant, a complete baseline,
root-controlled evidence backing and pinned inspection/process dependencies.
The trusted runtime premise excludes hostile namespace substitution; it does
not assert that unprivileged namespace syscalls are universally denied.
"""

import copy
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import (ALIAS_PARENT, AUTHORITY, CANONICAL, CONFIGURED,
               EXPECTATIONS, MOUNT, InvalidEvidence, canonical_json, digest,
               path_bytes, require, require_future, sha256, uint, strict_json,
               validate_storage_identity, observed_local_ssd_aliases)
from .content_hasher import Identity, hash_content
from .supervisor import (HASH_WINDOW, FutureBackend, parent_argv, plan,
                         protection, receipt_gate, validate_attempt)

SCHEMA = "mer.p1v.enforcement.v2"
PROFILE = "trusted-runtime-minimal-hot-path"
CLONE_NEWNS = 0x00020000
CLONE_NEWNET = 0x40000000
MS_RDONLY, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 32, 4096, 16384, 1 << 18
BLKROSET, BLKROGET = 0x125D, 0x125E
RLIMITS = ("CPU", "FSIZE", "DATA", "STACK", "CORE", "RSS", "NPROC", "NOFILE",
           "MEMLOCK", "AS", "LOCKS", "SIGPENDING", "MSGQUEUE", "NICE", "RTPRIO", "RTTIME")
CONTROLLERS = ("cpu", "memory", "io", "cpuset", "pids", "hugetlb", "rdma", "misc", "gpu")
DOMAIN_FIELDS = (*CONTROLLERS, "effective_cpus", "effective_mems", "ancestor_constraints", "bpf_ancestry")
DEPENDENCIES = ("python", "loader", "libraries", "inspection", "process_control", "system_interface")
TRUST_PREMISE = {
    "exact_source_has_no_intended_escape": True,
    "unreviewed_injection_plugins_brokers": False,
    "hostile_namespace_substitution": "OUTSIDE_TRUST_MODEL",
    "administrative_noninterference": "TRUSTED_HOST_PREMISE",
}
LAUNCH_FIELDS = ("environment", "rlimits", "umask", "stdio", "scheduler", "nice",
                 "cpu_affinity", "io_priority", "cpu_policy", "numa", "thp",
                 "numa_balancing", "resource_domain", "gpu", "cwd", "signals",
                 "credentials", "adapter_pins", "session", "security_context")
POLICY_FIELDS = {
    "scheduler": ("class", "attributes"),
    "io_priority": ("class", "priority"),
    "cpu_policy": ("governors", "frequency_limits", "boost", "power"),
    "numa": ("allowed_nodes", "memory_policy"),
    "thp": ("policy", "evidence_sha256"),
    "numa_balancing": ("policy", "evidence_sha256"),
    "gpu": ("device", "driver", "nodes", "libraries", "icds", "layers", "cache",
            "persistence", "compute", "power", "clocks", "mps", "mig"),
    "signals": ("mask", "dispositions"),
    "session": ("session_id", "process_group", "autogroup"),
    "security_context": ("lsm", "labels", "inherited_policy_sha256"),
}
SESSION_POLICY = {"policy": "INHERIT_WORKER", "child_session_equals_worker": True,
                  "child_process_group_equals_worker": True,
                  "child_autogroup_equals_worker": True, "autogroup_nice": 0}
HOLD_TRUE = ("exclusion_held", "filesystem_ro", "device_ro", "alias_locked",
             "private_mount_namespace", "private_network_namespace", "handoff_child_stopped",
             "brokers_excluded", "parent_nvml_cold", "parent_single_thread")


def validate_stdio(entries):
    """Linux x86-64 status flags: read-only stdin; append-only output opens."""
    require(type(entries) is list and len(entries) == 3, "exact stdio inventory")
    for fd, entry in enumerate(entries):
        exact_keys(entry, ("fd", "type", "backing", "flags"), "stdio type/backing/flags")
        require(type(entry["fd"]) is int and entry["fd"] == fd and entry["type"] == "regular", "stdio descriptor/type")
        exact_keys(entry["backing"], ("device", "inode", "path"), "stdio backing identity")
        backing = entry["backing"]
        require(backing["device"] == "259:1" and uint(backing["inode"]) > 0, "stdio output backing")
        path_bytes(backing["path"])
        require(type(entry["flags"]) is int and entry["flags"] == (0o100000 if fd == 0 else 0o102001),
                "stdio exact Linux access/append/status flags")
    require(len({e["backing"]["path"] for e in entries}) == 3
            and len({e["backing"]["inode"] for e in entries}) == 3, "distinct reviewed stdio targets")


def validate_hold_proof(value):
    exact_keys(value, (*HOLD_TRUE, "private_network_abstract_sockets", "nvml_helper_active"), "strict confinement hold proof")
    require(all(value[k] is True for k in HOLD_TRUE), "confinement hold readiness")
    require(type(value["private_network_abstract_sockets"]) is int
            and value["private_network_abstract_sockets"] == 0
            and value["nvml_helper_active"] is False, "confinement hold isolation")


def linux_rlimit_id(resource_module, name, platform_name):
    """Resolve exported limits, with only the reviewed Linux LOCKS ABI fallback."""
    require(name in RLIMITS, "unknown frozen RLIMIT name")
    try:
        identifier = getattr(resource_module, "RLIMIT_" + name)
    except AttributeError:
        require(name == "LOCKS" and platform_name == "linux", "missing RLIMIT symbol: " + name)
        # Linux v6.8 include/uapi/asm-generic/resource.h: RLIMIT_LOCKS = 10.
        # Historical architecture differences in IDs 5-9 use exports above.
        identifier = 10
    require(type(identifier) is int, "RLIMIT identifier must be integer: " + name)
    if name == "LOCKS":
        require(identifier == 10, "RLIMIT_LOCKS must match Linux ABI id 10")
    return identifier


def exact_keys(value, keys, message):
    require(type(value) is dict and set(value) == set(keys), message)


def known(value):
    """No unknown/default sentinel; absence must be represented explicitly."""
    require(value is not None, "unknown required manifest value")
    if isinstance(value, dict):
        for key, child in value.items():
            require(type(key) is str and key, "manifest key")
            known(child)
    elif isinstance(value, list):
        for child in value:
            known(child)
    elif isinstance(value, str):
        require(value.upper() not in ("UNKNOWN", "UNAVAILABLE", "TBD"), "unknown manifest value")
    else:
        require(type(value) in (int, bool), "manifest value type")


def resource_domain(value):
    exact_keys(value, DOMAIN_FIELDS, "complete resource domain required")
    known(value)
    for name in CONTROLLERS:
        exact_keys(value[name], ("domain", "constraints"), "controller domain/constraints: " + name)
        require(type(value[name]["domain"]) is str, "resource domain identity")
        require(type(value[name]["constraints"]) is dict, "resource constraint map")
    require(type(value["bpf_ancestry"]) is list, "BPF ancestry inventory required")
    for key in ("effective_cpus", "effective_mems"):
        numeric_set(value[key], nonempty=True)
    require(type(value["ancestor_constraints"]) is list, "ancestor inventory")
    return value


def numeric_set(values, nonempty=False):
    require(type(values) is list and (values or not nonempty), "numeric set required")
    for value in values:
        uint(value, 32)
    require(values == sorted(set(values)), "set must be exact sorted unique numeric values")


@dataclass(frozen=True)
class LaunchManifest:
    """Canonical bytes prevent caller mutation after baseline approval."""
    encoded: bytes

    @classmethod
    def parse(cls, value, *, authority_issue=216):
        exact_keys(value, LAUNCH_FIELDS, "complete launch manifest required")
        known(value)
        env = value["environment"]
        exact_keys(env, ("complete", "values", "absent"), "environment absence/presence contract")
        require(env["complete"] is True and type(env["values"]) is dict and type(env["absent"]) is list,
                "complete environment map required")
        require(all(type(k) is str and k and "=" not in k and "\0" not in k
                    and type(v) is str and "\0" not in v for k, v in env["values"].items()), "environment entry")
        require(all(type(k) is str and k and "=" not in k and "\0" not in k for k in env["absent"])
                and env["absent"] == sorted(set(env["absent"]))
                and not set(env["absent"]) & set(env["values"]), "environment absence conflict")
        exact_keys(value["rlimits"], RLIMITS, "all Linux RLIMITs required")
        for pair in value["rlimits"].values():
            require(type(pair) is list and len(pair) == 2, "RLIMIT soft/hard")
            require(all(v == "INFINITY" or type(v) is int and v >= 0 for v in pair), "RLIMIT unknown")
            require(pair[1] == "INFINITY" or pair[0] != "INFINITY" and pair[0] <= pair[1], "RLIMIT ordering")
        require(uint(value["umask"]) <= 0o777, "umask")
        require(type(value["stdio"]) is list and len(value["stdio"]) == 3, "exact stdio inventory")
        for fd, entry in enumerate(value["stdio"]):
            exact_keys(entry, ("fd", "type", "backing", "flags"), "stdio type/backing/flags")
            require(type(entry["fd"]) is int and entry["fd"] == fd, "stdio descriptor")
            uint(entry["flags"])
        require(type(value["session"]) is dict, "session policy object required")
        relational = "policy" in value["session"]
        require(type(authority_issue) is int and authority_issue >= 216, "launch authority")
        require(authority_issue <= 251 or relational, "new authority requires relational INHERIT_WORKER session")
        for name, fields in POLICY_FIELDS.items():
            if name == "session" and relational:
                fields = SESSION_POLICY
            exact_keys(value[name], fields, "launch policy coverage: " + name)
        if relational:
            require(canonical_json(value["session"]) == canonical_json(SESSION_POLICY), "session inheritance policy drift")
            validate_stdio(value["stdio"])
        require(type(value["nice"]) is int and -20 <= value["nice"] <= 19, "nice")
        numeric_set(value["cpu_affinity"], nonempty=True)
        numeric_set(value["numa"]["allowed_nodes"], nonempty=True)
        path_bytes(value["cwd"])
        resource_domain(value["resource_domain"])
        exact_keys(value["credentials"], ("uid", "gid", "groups"), "credentials")
        require(uint(value["credentials"]["uid"], 32) > 0
                and uint(value["credentials"]["gid"], 32) > 0, "fixed nonroot IDs")
        numeric_set(value["credentials"]["groups"])
        exact_keys(value["adapter_pins"], DEPENDENCIES, "adapter dependency pins")
        for pin in value["adapter_pins"].values():
            sha256(pin)
        return cls(canonical_json(value))

    @property
    def sha256(self):
        return digest(self.encoded)

    def values(self):
        from . import strict_json
        return strict_json(self.encoded)

    def compare(self, observed):
        require(self.encoded == self.parse(observed).encoded, "frozen launch profile drift")


def validate_adapter_manifest(manifest, baseline, closure):
    exact_keys(manifest, ("schema", "implementation_issue", "module_sha256", "dependencies",
                          "launch_manifest_sha256", "execution_closure_sha256"), "adapter manifest coverage")
    require(manifest["schema"] == "mer.p1v.adapter.v1"
            and type(manifest["implementation_issue"]) is int and manifest["implementation_issue"] == 216,
            "adapter implementation authority")
    sha256(manifest["module_sha256"])
    launch = LaunchManifest.parse(baseline)
    require(manifest["launch_manifest_sha256"] == launch.sha256, "launch manifest binding")
    require(manifest["dependencies"] == baseline["adapter_pins"], "dependency manifest binding")
    exact_keys(closure, ("premise", "source_commit", "binary_sha256", "config_sha256", "request_sha256",
                        "loader", "libraries", "git", "driver_modules", "environment_sha256"), "execution closure coverage")
    require(closure["premise"] == TRUST_PREMISE, "trusted runtime premise required")
    require(closure["source_commit"] == AUTHORITY["commit"], "trusted source")
    for name in ("binary_sha256", "config_sha256", "request_sha256"):
        require(closure[name] == AUTHORITY[name], "trusted input: " + name)
    for name in ("loader", "libraries", "git", "driver_modules"):
        require(type(closure[name]) is dict and closure[name], "execution closure pins: " + name)
        for path, pin in closure[name].items():
            path_bytes(path)
            sha256(pin)
    require(closure["environment_sha256"] == digest(canonical_json(baseline["environment"])), "trusted environment")
    require(manifest["execution_closure_sha256"] == digest(canonical_json(closure)), "execution closure binding")
    return launch


@dataclass(frozen=True, init=False)
class ExecutionGrant:
    """External interlock, not a signature service. Never issued by this module.

    A later operator supplies independently pinned canonical scope bytes and
    token digest. Secrets are checked then discarded, never serialized. Scope
    pins the adapter/closure/baseline transitively via adapter_manifest_sha256.
    """
    _scope: bytes

    def __init__(self, scope, expected_scope_sha256, token, expected_token_sha256):
        exact_keys(scope, ("authority_issue", "host_id", "attempt", "operations", "adapter_manifest_sha256", "storage_identity"), "grant scope")
        require(type(scope["authority_issue"]) is int and scope["authority_issue"] > 230, "execution authority must follow #230")
        require(type(scope["host_id"]) is str and scope["host_id"], "grant host")
        validate_attempt(scope["attempt"])
        require(scope["operations"] == [a.name for a in plan(scope["attempt"])], "explicit full operation scope")
        sha256(scope["adapter_manifest_sha256"])
        validate_storage_identity(scope["storage_identity"])
        require(digest(canonical_json(scope)) == sha256(expected_scope_sha256), "external scope pin mismatch")
        require(type(token) is bytes and len(token) >= 32
                and digest(token) == sha256(expected_token_sha256), "external adapter grant required")
        object.__setattr__(self, "_scope", canonical_json(scope))

    @property
    def storage_identity(self):
        return self.public_scope()["storage_identity"]

    def public_scope(self):
        from . import strict_json
        return strict_json(self._scope)

    def check(self, authorization, host_id, attempt, manifest, operation):
        from . import strict_json
        require_future(authorization)
        scope = strict_json(self._scope)
        require(authorization.issue == scope["authority_issue"] and authorization.issue > 230,
                "both authorities must identify the same future issue")
        require(scope["host_id"] == host_id and scope["attempt"] == attempt
                and scope["adapter_manifest_sha256"] == digest(canonical_json(manifest))
                and operation in scope["operations"], "adapter execution grant scope mismatch")


def require_adapter_grant(authorization, grant, host_id, attempt, manifest, operation):
    require_future(authorization)
    require(type(grant) is ExecutionGrant, "adapter-specific external execution grant required")
    grant.check(authorization, host_id, attempt, manifest, operation)


class DurableJournal:
    """Exclusive records on an already admitted output backing; failures poison it.

    Fixture mode is restricted to a caller-supplied TemporaryDirectory object.
    Live creation requires the composite authorization before mkdir/open/fsync.
    A bootstrap reservation is separate from the one-shot runtime marker.
    """
    def __init__(self, root, attempt, *, fixture=None, authorization=None, grant=None,
                 host_id=None, manifest=None, backing=None):
        self.failed = False
        self.sequence, self.previous = 0, "0" * 64
        self.pending = None
        self.next_step = 0
        self.attempt = validate_attempt(attempt)
        if fixture is not None:
            import tempfile
            require(type(fixture) is tempfile.TemporaryDirectory, "temporary fixture required")
            require(Path(root).resolve() == Path(fixture.name).resolve(), "fixture backing mismatch")
            self.authorize = lambda operation: None
        else:
            def authorize(operation):
                require_adapter_grant(authorization, grant, host_id, attempt, manifest, operation)
            authorize("AUTHORITY")
            require(backing == {"root_owned": True, "controlled": True, "major_minor": "259:1",
                                "runtime_cannot_write": True, "no_symlink_components": True}, "evidence backing admission")
            self.authorize = authorize
            root_path = Path(root)
            require(root_path.is_absolute(), "absolute evidence root")
            for part in (root_path, *root_path.parents):
                info = part.lstat()
                require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0
                        and info.st_mode & 0o022 == 0, "uncontrolled evidence ancestor")
            info = root_path.stat()
            require((os.major(info.st_dev), os.minor(info.st_dev)) == (259, 1), "journal output backing")
        self.fixture = fixture is not None
        self.root = Path(root) / attempt
        self.authorize("AUTHORITY")
        self.root.mkdir(mode=0o700)  # no parents/exist_ok: no attempt reuse
        self._sync_dir(self.root.parent)
        self.publish("bootstrap.json", canonical_json({"attempt": attempt, "kind": "RESERVATION"}), "AUTHORITY")

    @staticmethod
    def _sync_dir(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def publish(self, name, data, operation):
        require(not self.failed and Path(name).name == name and name not in (".", ".."), "journal stopped/path")
        self.authorize(operation)
        try:
            fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o400)
            try:
                require(stat.S_ISREG(os.fstat(fd).st_mode), "journal file type")
                require(os.write(fd, data) == len(data), "short journal write; no retry")
                os.fsync(fd)
            finally:
                os.close(fd)
            self._sync_dir(self.root)
        except BaseException:
            self.failed = True
            raise
        return digest(data)

    def retain(self, record):
        require(not self.failed, "journal durability uncertain")
        step, name, status = record["step"], record["action"], record["status"]
        actions = plan(self.attempt)
        require(type(step) is int and 0 <= step < len(actions) and actions[step].name == name, "journal action sequence")
        if status == "INTENT":
            require(self.pending is None and step == self.next_step, "journal intent order/no reuse")
        else:
            require(status in ("PASS", "FAIL") and self.pending == (step, name), "journal result without intent")
        wrapped = {"sequence": self.sequence, "previous_sha256": self.previous, "record": record}
        self.previous = self.publish(f"{self.sequence:04d}-{name}-{status}.json", canonical_json(wrapped), name)
        self.sequence += 1
        self.pending = (step, name) if status == "INTENT" else None
        if status == "PASS":
            self.next_step += 1
        elif status == "FAIL":
            self.failed = True


class System(Protocol):
    """Injected, hash-pinned Linux interfaces; observation methods read metadata.

    The process implementation owns guardian(original ns), single-threaded
    worker(private ns), and the stopped handoff child. No method may read expert
    payloads. resume_wait_parent must release once and do one blocking wait/reap,
    with no callbacks/timers/polling. observe('...') cannot synthesize assertions:
    the provider must read actual kernel/host state and reject unknown values.
    """
    simulated: bool

    def observe(self, topic: str) -> dict: ...
    def acquire_exclusion(self) -> None: ...
    def unshare(self, flags: int) -> None: ...
    def mount(self, source, target: str, fstype, flags: int, data) -> None: ...
    def syncfs(self, fd: int) -> None: ...
    def retain_backing(self) -> dict: ...
    def block_ro(self, fd: int, read_only: bool) -> None: ...
    def block_get(self, fd: int) -> int: ...
    def create_accounting_cgroup(self, attempt: str) -> str: ...
    def prepare_stopped_child(self, argv: tuple, manifest: bytes, cgroup: str) -> int: ...
    def resume_wait_parent(self, pid: int) -> int: ...
    def raw_reports(self, attempt: str) -> dict: ...
    def release(self) -> None: ...


class LinuxBackend(FutureBackend):
    """Concrete ordered adapter. Every operation has an observed readback gate.

    No default live system, grant, baseline, or process/inspection provider is
    supplied. Tests inject a pure in-memory System and fixture journal. The
    adapter never interprets plan argv as commands and never reads payloads
    through System. Only HASH constructs a reader.
    """
    def __init__(self, system, journal, baseline, manifest, closure, *, host_id,
                 attempt, authorization=None, grant=None, fixture_reader_factory=None):
        self.system, self.journal = system, journal
        self.baseline = validate_adapter_manifest(manifest, baseline, closure)
        self.manifest, self.closure = copy.deepcopy(manifest), copy.deepcopy(closure)
        self.host_id, self.attempt = host_id, validate_attempt(attempt)
        self.authorization, self.grant = authorization, grant
        self.authorize("AUTHORITY")
        LaunchManifest.parse(baseline, authority_issue=authorization.issue)
        self._storage_identity = canonical_json(grant.storage_identity)
        self.simulated = system.simulated is True
        require(journal.attempt == attempt and journal.fixture == self.simulated, "journal attempt/mode mismatch")
        if not self.simulated:
            self.authorize("AUTHORITY")
            require(fixture_reader_factory is None, "live reader cannot be overridden")
            require(manifest["module_sha256"] == digest(Path(__file__).read_bytes()), "adapter bytes differ from grant")
        else:
            require(fixture_reader_factory is not None and journal.authorize is not None, "synthetic reader required")
        self.reader_factory = fixture_reader_factory
        self.done, self.running, self.failed = [], False, False
        self.parent_calls = 0
        self.backing = self.child = self.cgroup = None
        self.raw = None

    @property
    def storage_identity(self):
        return strict_json(self._storage_identity)

    def authorize(self, operation):
        require_adapter_grant(self.authorization, self.grant, self.host_id, self.attempt, self.manifest, operation)

    def retain(self, record):
        require(not self.running, "adapter journal forbidden while P1R parent runs")
        if not self.simulated:
            self.authorize(record["action"])
        self.journal.retain(record)

    def _observe(self, topic):
        require(not self.running, "adapter observation forbidden while P1R parent runs")
        return copy.deepcopy(self.system.observe(topic))

    def _identity_health_check(self, snapshot):
        require(snapshot["device_identity"] == self.backing["identity"] == self.storage_identity, "retained device identity drift")
        require(snapshot["private_mount_id"] == self.backing["private_mount_id"], "private mount identity drift")
        require(snapshot["healthy"] is True and snapshot["external_journal"] is False,
                "filesystem health/internal journal uncertain")

    def _quiesced_ro_check(self, snapshot):
        """RO_QUIESCED_CLEAN exists only after the normal ext4 RO remount."""
        self._identity_health_check(snapshot)
        require(snapshot["recovery_required"] is False and snapshot["journal_clean"] is True,
                "RO quiesced journal uncertain")
        require(snapshot["mount_ro"] is True and snapshot["superblock_ro"] is True,
                "mount and superblock RO required")

    def _restored_rw_check(self, snapshot):
        """RW_RESTORED_ACTIVE has an active internal journal, not an RO certificate."""
        self._identity_health_check(snapshot)
        require(snapshot["recovery_required"] is True and "journal_clean" not in snapshot,
                "restored active RW journal uncertain")

    def _protected(self):
        r = self._observe("protected")
        protection(r)
        self._quiesced_ro_check(r)
        require(r["mount_ro"] is True and r["superblock_ro"] is True, "mount and superblock RO required")
        require(self.system.block_get(self.backing["device_fd"]) == 1, "protected device readback")
        return r

    def perform(self, action, evidence):
        require(not self.running and not self.failed, "adapter stopped/running")
        expected = plan(self.attempt)
        require(len(self.done) < len(expected) and action == expected[len(self.done)], "adapter order/reuse")
        require(evidence["attempt"] == self.attempt and list(evidence["receipts"]) == self.done, "adapter evidence prefix")
        if not self.simulated:
            self.authorize(action.name)
        try:
            result = self._perform(action.name, evidence)
            receipt_gate(action.name, result, evidence)
        except BaseException:
            self.failed = True
            raise
        self.done.append(action.name)
        return result

    def _perform(self, name, evidence):
        s, before = self.system, evidence["receipts"]
        if name == "AUTHORITY":
            r = self._observe("authority")
            require("launch_observed" not in r, "AUTHORITY forbids child launch observation")
            # The approved manifest is prospective; only CONFINEMENT observes
            # the exact stopped handoff child and compares its full launch state.
            require(r.pop("execution_closure") == self.closure, "observed trusted execution closure drift")
            require(r.pop("adapter_manifest") == self.manifest, "observed adapter/dependency pins drift")
            require(r["host_id"] == self.host_id, "grant/observed host mismatch")
            r.update(adapter_manifest=self.manifest, launch_manifest=self.baseline.values(), execution_closure=self.closure,
                     execution_grant={"status": "NOT_GRANTED", "storage_identity": self.storage_identity}
                     if self.simulated else self.grant.public_scope())
            return r
        if name == "EXCLUSION":
            s.acquire_exclusion()
            return self._observe("exclusion")
        if name == "TOPOLOGY":
            r = self._observe("topology")
            # Retain exact objects once; no reopen through a configurable path.
            self.backing = copy.deepcopy(s.retain_backing())
            b = self.backing
            exact_keys(b, ("device_fd", "filesystem_fd", "identity", "host_mount_id", "private_mount_id",
                           "mount_flags", "mount_data", "alias_flags"), "retained backing coverage")
            validate_storage_identity(b["identity"])
            require(b["identity"] == self.storage_identity
                    and b["identity"]["disk_sequence"] == r["disk_sequence"], "retained exact grant storage identity")
            require(r["device_identity"] == self.storage_identity, "observed topology storage grant mismatch")
            require(b["host_mount_id"] == r["mount_id"], "original mount identity")
            for key in ("device_fd", "filesystem_fd", "mount_flags", "alias_flags"):
                uint(b[key])
            require(b["device_fd"] > 2 and b["filesystem_fd"] > 2
                    and b["device_fd"] != b["filesystem_fd"], "retained control descriptors")
            require(b["mount_flags"] & (MS_RDONLY | MS_BIND | MS_REMOUNT) == 0, "original mount flags")
            require(s.block_get(b["device_fd"]) == 0, "original device not rw")
            r.update(device_identity=copy.deepcopy(b["identity"]), original_mount_flags=b["mount_flags"],
                     original_mount_data=b["mount_data"], original_alias_flags=b["alias_flags"])
            return r
        if name == "NAMESPACE":
            roles = self._observe("roles")
            require(roles["guardian_original_namespace"] is True and roles["worker_single_threaded"] is True
                    and roles["initial_user_namespace"] is True and roles["worker_privileged"] is True,
                    "guardian/worker architecture unavailable")
            s.unshare(CLONE_NEWNS | CLONE_NEWNET)
            s.mount(None, "/", None, MS_REC | MS_PRIVATE, None)
            r = self._observe("namespace")
            require(r["private_mount_namespace"] is True and r["private_network_namespace"] is True,
                    "private mount/network namespaces required")
            require(r["original_mount_namespace_id"] == r["original_namespace_id"]
                    and r["mount_namespace_id"] == r["namespace_id"]
                    and r["original_namespace_id"] != r["namespace_id"], "mount namespace not distinct")
            require(type(r["original_network_namespace_id"]) is str and r["original_network_namespace_id"]
                    and type(r["network_namespace_id"]) is str and r["network_namespace_id"]
                    and r["original_network_namespace_id"] != r["network_namespace_id"],
                    "network namespace not distinct")
            require(r["original_mount_id"] == self.backing["host_mount_id"]
                    and r["private_mount_id"] != r["original_mount_id"], "mount identities not distinct")
            self.backing["private_mount_id"] = r["private_mount_id"]
            return r
        if name == "ALIAS_BIND":
            s.mount(ALIAS_PARENT, ALIAS_PARENT, None, MS_BIND, None)
            return self._observe("alias_bind")
        if name == "ALIAS_RO":
            s.mount(None, ALIAS_PARENT, None, self.backing["alias_flags"] | MS_BIND | MS_REMOUNT | MS_RDONLY, None)
            r = self._observe("alias_ro")
            require(r["preserved_flags"] == self.backing["alias_flags"]
                    and r["canonical_target"] == CANONICAL and r["output_still_rw"] is True,
                    "alias flags/target/output backing drift")
            return r
        if name == "METADATA":
            return self._observe("metadata")
        if name == "FILESYSTEM_RO":
            s.syncfs(self.backing["filesystem_fd"])
            health = self._observe("sync_health")
            # PRE_RO_SYNCED_ACTIVE: syncfs is synchronization, not journal-empty proof.
            exact_keys(health, ("syncfs_ok", "healthy", "external_journal", "recovery_required"),
                       "pre-RO sync receipt must not assert journal_clean")
            require(health["syncfs_ok"] is True and health["healthy"] is True
                    and health["external_journal"] is False and health["recovery_required"] is True,
                    "pre-RO synced active journal uncertain")
            s.mount(None, MOUNT, "ext4", self.backing["mount_flags"] | MS_REMOUNT | MS_RDONLY, self.backing["mount_data"])
            r = self._observe("filesystem_ro")
            self._quiesced_ro_check(r)
            require(r["mount_ro"] is True and s.block_get(self.backing["device_fd"]) == 0, "FS/device transition order")
            return r
        if name == "DEVICE_RO":
            r = self._observe("filesystem_ro")
            self._quiesced_ro_check(r)
            require(r["superblock_ro"] is True and r["mount_ro"] is True, "superblock RO prerequisite")
            s.block_ro(self.backing["device_fd"], True)
            require(s.block_get(self.backing["device_fd"]) == 1, "BLKROSET RO readback failed")
            r = self._observe("device_ro")
            self._quiesced_ro_check(r)
            return r
        if name in ("PROTECTED", "REVALIDATE"):
            return self._protected()
        if name == "CONFINEMENT":
            placement = self._observe("cgroup_preflight")
            baseline = self.baseline.values()["resource_domain"]
            for key in ("baseline", "prospective"):
                resource_domain(placement[key])
                require(placement[key] == baseline, "resource-neutral accounting placement not proven")
            require(placement["nondelegated"] is True and placement["controllers_to_enable"] == []
                    and placement["settings_to_write"] == [] and placement["bpf_to_attach"] == [], "active cgroup policy forbidden")
            self.cgroup = s.create_accounting_cgroup(self.attempt)
            self.child = s.prepare_stopped_child(parent_argv(self.attempt), self.baseline.encoded, self.cgroup)
            require(uint(self.child, 32) > 0, "stopped handoff child required")
            r = self._observe("confinement")
            require(r["handoff_pid"] == self.child and r["handoff_stopped"] is True
                    and r["namespace_id"] == before["NAMESPACE"]["namespace_id"]
                    and r["cgroup_id"] == self.cgroup, "actual stopped handoff child binding")
            self.baseline.compare(r["launch_observed"])
            require(r["resource_domain_observed"] == baseline, "actual cgroup domain drift")
            return r
        if name == "MARKER":
            self.journal.publish("marker.json", canonical_json({"attempt": self.attempt, "kind": "ONE_SHOT"}), name)
            return {"ok": True, "created_exclusive": True, "durable": True, "attempt": self.attempt}
        if name == "P1R_PARENT":
            require(self.parent_calls == 0 and self.child is not None, "one parent invocation only")
            self.parent_calls += 1
            self.running = True
            # No adapter call exists between release and the single blocking reap.
            # If the wait fails/interrupts, running remains true: no journal/query/RW.
            code = s.resume_wait_parent(self.child)
            self.running = False
            require(type(code) is int and code == 0, "P1R parent exit failure")
            self.raw = s.raw_reports(self.attempt)
            require(set(self.raw) == {"parent", "control", "treatment"}
                    and all(type(v) is bytes for v in self.raw.values()), "raw immutable reports")
            return {"ok": True, "invocations": self.parent_calls, "argv": list(parent_argv(self.attempt)),
                    "external_actions_between_arms": [], "exit_code": code,
                    "report_origin": "created_by_this_invocation", "attempt": self.attempt,
                    "raw_report_hashes": {k: digest(v) for k, v in self.raw.items()}}
        if name == "SHUTDOWN":
            r = self._observe("shutdown")
            require(r["recursive_populated"] == 0 and type(r["recursive_populated"]) is int
                    and r["unexpected_child_groups"] == [], "recursive cgroup emptiness uncertain")
            return r
        if name == "HASH":
            require(self.done[-3:] == ["P1R_PARENT", "SHUTDOWN", "REVALIDATE"], "hash lifecycle prerequisite")
            self._protected()
            storage_identity = self.storage_identity
            if self.simulated:
                reader = self.reader_factory()
            else:
                self.authorize(name)
                from .content_hasher import LinuxDirectReader
                reader = LinuxDirectReader(self.authorization, HASH_WINDOW, storage_identity)
            pre = {r["global_expert_id"]: (r["selected"], r["canonical"], Identity(**r["identity"]))
                   for r in before["METADATA"]["records"][-6:]}
            content = hash_content(reader, pre, HASH_WINDOW, storage_identity)
            return {"ok": True, "window": copy.deepcopy(HASH_WINDOW), "hasher_stopped": True, "content": content}
        if name == "VERIFY":
            from .offline_verify import verify_reports
            # Re-read immutable report bytes and compare against launch-bound hashes.
            raw = s.raw_reports(self.attempt)
            require(raw == self.raw, "raw report mutation")
            derived = verify_reports(raw, evidence, allow_simulated=self.simulated, sealed=False)
            return {"ok": True, "raw_report_hashes": {k: digest(v) for k, v in raw.items()}, "derived": derived}
        if name == "SEAL":
            pin = self.journal.publish("enforcement.json", canonical_json(evidence), name)
            return {"ok": True, "durable": True, "exclusive": True, "runtime_cannot_write": True, "sha256": pin}
        if name == "DEVICE_RW":
            self._protected()
            r = self._observe("shutdown")
            require(r["runtime_tree_empty"] is True and r["recursive_populated"] == 0
                    and r["uncertain_descendants"] == [], "restore tree uncertainty")
            s.block_ro(self.backing["device_fd"], False)
            require(s.block_get(self.backing["device_fd"]) == 0, "BLKROSET RW readback failed")
            r = self._observe("device_rw")
            self._quiesced_ro_check(r)
            return r
        if name == "FILESYSTEM_RW":
            r = self._observe("device_rw")
            self._quiesced_ro_check(r)
            require(r["filesystem_still_ro"] is True and s.block_get(self.backing["device_fd"]) == 0,
                    "device-rw/FS-ro prerequisite")
            s.mount(None, MOUNT, "ext4", self.backing["mount_flags"] | MS_REMOUNT, self.backing["mount_data"])
            r = self._observe("filesystem_rw")
            self._restored_rw_check(r)
            return r
        if name == "RESTORATION_SEAL":
            data = {"evidence_seal_sha256": before["SEAL"]["sha256"],
                    "device": before["DEVICE_RW"], "filesystem": before["FILESYSTEM_RW"]}
            self.journal.publish("restoration.json", canonical_json(data), name)
            return {"ok": True, "durable": True, "exclusive": True, "evidence_seal_sha256": data["evidence_seal_sha256"]}
        if name == "RELEASE":
            s.release()
            return self._observe("release")
        raise InvalidEvidence("unknown adapter action")


class NativePrimitives:
    """Narrow native syscall layer for an externally granted injected System.

    Not constructed by the CLI or portable tests. The authorize callback must
    recheck the composite grant for the current action before every native call.
    No retries, force, subprocesses, controller writes, filters or device BPF.
    Host inspection and process ownership stay in the pinned System provider.
    """
    def __init__(self, authorization, grant, host_id, attempt, manifest, current_action):
        require_adapter_grant(authorization, grant, host_id, attempt, manifest, "AUTHORITY")
        self.context = (authorization, grant, host_id, attempt, copy.deepcopy(manifest))
        self._storage_identity = canonical_json(grant.storage_identity)
        self.current_action = current_action
        self.retained_fds = {}

    @property
    def storage_identity(self):
        return strict_json(self._storage_identity)

    def _check(self, permitted):
        action = self.current_action()
        require(action in permitted, "native operation outside phase")
        require_adapter_grant(*self.context, action)
        require(os.geteuid() == 0, "future native operations require root")
        return action

    @staticmethod
    def _libc():
        import ctypes
        import sys
        require(sys.platform == "linux", "Linux required")
        libc = ctypes.CDLL(None, use_errno=True)
        libc.unshare.argtypes, libc.unshare.restype = [ctypes.c_int], ctypes.c_int
        libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
        libc.mount.restype = ctypes.c_int
        libc.syncfs.argtypes, libc.syncfs.restype = [ctypes.c_int], ctypes.c_int
        return libc

    @staticmethod
    def _result(result):
        import ctypes
        if result != 0:
            raise OSError(ctypes.get_errno(), "Linux operation failed; no retry")

    def retain_objects(self):
        """Retain only metadata/control descriptions, never read payload bytes.

        Partial setup failures deliberately retain already opened objects for
        the owning guardian/worker; there is no reopen or automatic recovery.
        """
        self._check(("TOPOLOGY",))
        require(not self.retained_fds, "retained object reuse")
        identity = self.storage_identity
        for name, path, directory in (("device", identity["device_path"], False), ("filesystem", MOUNT, True),
                                      ("host_mount_ns", "/proc/self/ns/mnt", False)):
            flags = os.O_RDONLY | os.O_CLOEXEC
            if name != "host_mount_ns":
                flags |= os.O_NOFOLLOW
            if directory:
                flags |= os.O_DIRECTORY
            self.retained_fds[name] = os.open(path, flags)
        device = self._device(self.retained_fds["device"])
        path_info = os.lstat(identity["device_path"])
        require(stat.S_ISBLK(path_info.st_mode) and path_info.st_rdev == device.st_rdev,
                "retained non-symlink device path binding")
        info = os.fstat(self.retained_fds["filesystem"])
        require(stat.S_ISDIR(info.st_mode) and f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}" == identity["major_minor"],
                "retained filesystem device")
        for alias in ["/dev/disk/by-uuid/" + identity["uuid"], *identity["by_id_aliases"]]:
            expected = os.stat(alias)
            require(stat.S_ISBLK(expected.st_mode) and expected.st_rdev == device.st_rdev, "granted alias device binding")
        require(observed_local_ssd_aliases(device.st_rdev, os.listdir, os.stat) == identity["by_id_aliases"],
                "complete retained Local SSD alias set")
        records = self.mount_snapshot()
        matches = [r for r in records if r["target"] == MOUNT]
        require(len(matches) == 1 and matches[0]["major_minor"] == identity["major_minor"]
                and matches[0]["fstype"] == "ext4" and matches[0]["root"] == "/", "retained exact mount backing")
        return dict(self.retained_fds)

    def mount_snapshot(self):
        self._check(("TOPOLOGY", "NAMESPACE", "ALIAS_BIND", "ALIAS_RO", "FILESYSTEM_RO", "DEVICE_RO",
                     "PROTECTED", "REVALIDATE", "HASH", "DEVICE_RW", "FILESYSTEM_RW"))
        return parse_mountinfo(Path("/proc/self/mountinfo").read_text())

    def pin_private_namespace(self):
        self._check(("NAMESPACE",))
        require("host_mount_ns" in self.retained_fds and "private_mount_ns" not in self.retained_fds, "namespace reference lifecycle")
        private = os.open("/proc/self/ns/mnt", os.O_RDONLY | os.O_CLOEXEC)
        self.retained_fds["private_mount_ns"] = private
        original = os.fstat(self.retained_fds["host_mount_ns"])
        copied = os.fstat(private)
        require((original.st_dev, original.st_ino) != (copied.st_dev, copied.st_ino), "namespace not private")
        return {"original_namespace_id": original.st_ino, "private_namespace_id": copied.st_ino}

    def unshare(self, flags):
        self._check(("NAMESPACE",))
        require(type(flags) is int and flags == CLONE_NEWNS | CLONE_NEWNET,
                "only combined CLONE_NEWNS | CLONE_NEWNET allowed")
        self._result(self._libc().unshare(flags))

    def mount(self, source, target, fstype, flags, data):
        action = self._check(("NAMESPACE", "ALIAS_BIND", "ALIAS_RO", "FILESYSTEM_RO", "FILESYSTEM_RW"))
        if action == "NAMESPACE":
            require((source, target, fstype, flags, data) == (None, "/", None, MS_REC | MS_PRIVATE, None), "private propagation only")
        elif action == "ALIAS_BIND":
            require((source, target, fstype, flags, data) == (ALIAS_PARENT, ALIAS_PARENT, None, MS_BIND, None), "self bind only")
        elif action == "ALIAS_RO":
            require(source is None and target == ALIAS_PARENT and fstype is None and data is None
                    and flags & (MS_BIND | MS_REMOUNT | MS_RDONLY) == MS_BIND | MS_REMOUNT | MS_RDONLY, "alias remount only")
        else:
            require(source is None and target == MOUNT and fstype == "ext4" and flags & MS_REMOUNT
                    and not flags & MS_BIND and bool(flags & MS_RDONLY) == (action == "FILESYSTEM_RO"), "normal ext4 remount only")
        encoded = [None if v is None else v.encode() for v in (source, target, fstype, data)]
        self._result(self._libc().mount(encoded[0], encoded[1], encoded[2], flags, encoded[3]))

    def syncfs(self, fd):
        self._check(("FILESYSTEM_RO",))
        self._result(self._libc().syncfs(fd))

    def _device(self, fd):
        require(fd == self.retained_fds.get("device"), "retained block descriptor only")
        value = os.fstat(fd)
        require(stat.S_ISBLK(value.st_mode)
                and f"{os.major(value.st_rdev)}:{os.minor(value.st_rdev)}" == self.storage_identity["major_minor"],
                "exact block descriptor")
        return value

    def block_ro(self, fd, read_only):
        action = self._check(("DEVICE_RO", "DEVICE_RW"))
        require(type(read_only) is bool and read_only == (action == "DEVICE_RO"), "block transition phase")
        self._device(fd)
        import array
        import fcntl
        fcntl.ioctl(fd, BLKROSET, array.array("i", [int(read_only)]), True)

    def block_get(self, fd):
        self._check(("TOPOLOGY", "FILESYSTEM_RO", "DEVICE_RO", "PROTECTED", "CONFINEMENT", "REVALIDATE", "HASH", "DEVICE_RW", "FILESYSTEM_RW"))
        self._device(fd)
        import array
        import fcntl
        value = array.array("i", [0])
        fcntl.ioctl(fd, BLKROGET, value, True)
        require(value[0] in (0, 1), "block RO readback invalid")
        return value[0]


def validate_confinement(r, baseline, namespace):
    # Historical sealed evidence retains its old schema. New live admission
    # requires SESSION_POLICY before any host access; it cannot enter this arm.
    lifecycle_fields = ("hold_proof", "session_observations") if "policy" in baseline["session"] else ()
    true_fields = ("no_new_privs", "groups_exact", "brokers_excluded", "setup_fds_closed",
                   "initial_user_namespace", "private_mount_namespace", "private_network_namespace",
                   "trusted_execution_closure_verified",
                   "cgroup_nondelegated", "resource_domain_equal", "launch_profile_equal", "raw_device_access_denied",
                   "handoff_stopped")
    empty_fields = ("raw_or_privileged_fds", "writable_alias_routes", "cgroup_controllers_changed",
                    "cgroup_limits_changed", "cgroup_bpf_added")
    exact_keys(r, ("ok", "uids", "gids", "groups", "capabilities", *true_fields, *empty_fields,
                   "seccomp_mode", "p1v_device_bpf_attached", "inherited_fds", "cgroup_id", "cgroup_mode",
                   "cgroup_active_monitoring_during_pair", "launch_observed", "resource_domain_observed",
                   "namespace_id", "handoff_pid", *lifecycle_fields), "strict v2 confinement fields")
    if lifecycle_fields:
        validate_hold_proof(r["hold_proof"])
        require(session_relationship(r["session_observations"]) == baseline["session"], "session relation drift")
    expected = baseline["credentials"]
    for field, key in (("uids", "uid"), ("gids", "gid")):
        require(type(r[field]) is list and len(r[field]) == 3, "credential slots")
        for value in r[field]:
            require(uint(value, 32) > 0 and value == expected[key], "fixed reviewed credential mismatch")
    numeric_set(r["groups"])
    require(r["groups"] == expected["groups"], "exact reviewed supplementary groups")
    exact_keys(r["capabilities"], ("effective", "permitted", "inheritable", "ambient", "bounding"), "all capability sets required")
    for value in r["capabilities"].values():
        require(uint(value) == 0, "nonzero capability")
    for name in true_fields:
        require(r[name] is True, "runtime confinement: " + name)
    for name in empty_fields:
        require(type(r[name]) is list and r[name] == [], "runtime exclusion: " + name)
    require(type(r["seccomp_mode"]) is int and r["seccomp_mode"] == 0, "inherited seccomp rejected")
    require(r["p1v_device_bpf_attached"] is False and r["cgroup_active_monitoring_during_pair"] is False,
            "added BPF/active cgroup monitoring forbidden")
    require(type(r["inherited_fds"]) is list and all(type(v) is int for v in r["inherited_fds"])
            and r["inherited_fds"] == [0, 1, 2], "setup/control FD leak")
    require(type(r["cgroup_id"]) is str and r["cgroup_id"] and r["cgroup_mode"] == "PASSIVE_TREE_ONLY", "passive cgroup boundary")
    require(r["namespace_id"] == namespace["namespace_id"] and uint(r["handoff_pid"], 32) > 0, "handoff namespace/process binding")
    LaunchManifest.parse(baseline).compare(r["launch_observed"])
    require(resource_domain(r["resource_domain_observed"]) == baseline["resource_domain"], "resource domain/constraint drift")


def session_relationship(observations):
    import re
    exact_keys(observations, ("worker", "child"), "raw worker/child sessions")
    for value in observations.values():
        exact_keys(value, ("session_id", "process_group", "autogroup"), "raw session observation")
        require(uint(value["session_id"], 32) > 0 and uint(value["process_group"], 32) > 0, "session numeric observation")
        require(type(value["autogroup"]) is str
                and re.fullmatch(r"/autogroup-[0-9]+ nice 0", value["autogroup"]), "autogroup nice must be zero")
    worker, child = observations["worker"], observations["child"]
    result = {"policy": "INHERIT_WORKER", "child_session_equals_worker": child["session_id"] == worker["session_id"],
              "child_process_group_equals_worker": child["process_group"] == worker["process_group"],
              "child_autogroup_equals_worker": child["autogroup"] == worker["autogroup"], "autogroup_nice": 0}
    require(result == SESSION_POLICY, "session inheritance relation drift")
    return result


class ProcessInterfaces(Protocol):
    """Hash-pinned process primitives, supplied with the reviewed host System.

    Methods execute in the indicated process; handoff setup must preserve the
    exact manifest, drop bounding/ambient/capability sets and credentials, set
    no_new_privs, close all but 0/1/2, and stop before exec. No filter is installed.
    A stopped child's observation is made by its owner before marker consumption.
    """
    simulated: bool

    def fork(self) -> int: ...
    def wait_once(self, pid: int, stopped: bool) -> dict: ...
    def resume_wait_once(self, pid: int) -> dict: ...
    def resume_wait_worker(self, pid: int) -> dict: ...
    def failed_hold_forever(self) -> None: ...
    def enter_cgroup(self, cgroup: str) -> None: ...
    def apply_launch_credentials_and_close_fds(self, manifest: bytes) -> None: ...
    def stop_self(self) -> None: ...
    def stop_then_exec_parent(self, argv: tuple, manifest: bytes) -> None: ...
    def exit_worker(self, code: int) -> None: ...


class ProcessRoles:
    """One original-ns guardian, one worker and one stopped runtime child.

    Injected primitives make role/failure ordering portable-testable. The live
    provider is part of the externally pinned dependency closure. There are no
    retry, timeout, event-loop, signal-cleanup or automatic restoration paths.
    """
    def __init__(self, processes, *, authorization=None, grant=None, host_id=None,
                 attempt, manifest):
        self.processes = processes
        self.context = (authorization, grant, host_id, attempt, copy.deepcopy(manifest))
        self.attempt = validate_attempt(attempt)
        self.worker_used = self.child_used = self.wait_used = False
        self.child = None
        self.failure_owner = None
        self._authorize("AUTHORITY")

    def bind_failure_owner(self, owner):
        require(self.failure_owner is None, "one protected failure owner")
        self.failure_owner = owner

    def _authorize(self, operation):
        if self.processes.simulated is not True:
            require_adapter_grant(*self.context, operation)

    def guardian(self, worker_main):
        self._authorize("AUTHORITY")
        require(not self.worker_used, "worker reuse forbidden")
        self.worker_used = True
        import mmap
        # A failure journal may itself be poisoned. This bounded anonymous
        # channel carries diagnostics only and never transfers control FDs.
        with mmap.mmap(-1, 65536, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE) as shared:
            return self._guardian(worker_main, shared)

    def _guardian(self, worker_main, shared):
        pid = self.processes.fork()
        if pid == 0:
            # Establish guardian wait readiness before worker setup can begin.
            self.processes.stop_self()
            code = 1
            try:
                code = 0 if worker_main()["status"] in ("PASS_FOR_REVIEW", "PASS_SIMULATION") else 1
            except BaseException:
                pass
            owner = self.failure_owner
            if code and owner is not None and owner.protection_owned():
                # Observation/publication failure must never turn hold into exit.
                try:
                    receipt = owner.failure_receipt()
                    payload = canonical_json(receipt)
                    require(len(payload) <= 65532, "failed hold receipt size")
                    shared[4:4 + len(payload)] = payload
                    shared[:4] = len(payload).to_bytes(4, "big")
                except BaseException:
                    shared[:4] = bytes(4)
                self.processes.failed_hold_forever()
                raise InvalidEvidence("failed hold unexpectedly returned")
            self.processes.exit_worker(code)
            raise InvalidEvidence("worker exit unexpectedly returned")
        require(uint(pid, 32) > 0, "worker fork failure")
        require(self.processes.wait_once(pid, True) == {"pid": pid, "stopped": True}, "worker bootstrap stop failed")
        result = self.processes.resume_wait_worker(pid)
        if result == {"pid": pid, "stopped": True}:
            length = int.from_bytes(shared[:4], "big")
            receipt = None
            try:
                require(0 < length <= 65532, "failed hold observation unavailable")
                payload = shared[4:4 + length]
                receipt = strict_json(payload)
                require(canonical_json(receipt) == payload and receipt["worker_pid"] == pid,
                        "failed hold owner mismatch")
            except Exception:
                receipt = None
            return {"status": "FAIL_FOR_REVIEW_HOLDING", "worker_pid": pid, "worker_stopped": True,
                    "ownership_observation": receipt, "ownership_observation_available": receipt is not None,
                    "automatic_restoration": False, "attempt_reusable": False}
        require(result == {"pid": pid, "exited": True, "exit_code": 0}, "guardian worker exited with failure")
        return result

    def prepare_stopped_child(self, argv, manifest, cgroup):
        self._authorize("CONFINEMENT")
        require(not self.child_used and argv == parent_argv(self.attempt), "one unchanged parent only")
        self.child_used = True
        LaunchManifest.parse(LaunchManifest(manifest).values())
        pid = self.processes.fork()
        if pid == 0:
            try:
                self.processes.handoff_child_pid = os.getpid()
                self.processes.enter_cgroup(cgroup)
                self.processes.apply_launch_credentials_and_close_fds(manifest)
                # SIGCONT returns directly to execve inside this boundary.
                self.processes.stop_then_exec_parent(argv, manifest)
            except BaseException:
                os._exit(127)
            os._exit(127)  # exec returning is also terminal, never supervisor work.
        require(uint(pid, 32) > 0, "handoff fork failure")
        self.child = pid  # retain exact ownership even when the stop wait fails
        require(self.processes.wait_once(pid, True) == {"pid": pid, "stopped": True}, "handoff child did not stop")
        return pid

    def resume_wait_parent(self, pid):
        self._authorize("P1R_PARENT")
        require(not self.wait_used and pid == self.child and self.child_used, "parent wait scope/reuse")
        self.wait_used = True
        # The primitive authorizes before release, then calls blocking wait
        # directly. In particular it must not hash a grant after release.
        result = self.processes.resume_wait_once(pid)
        require(result["pid"] == pid and result["exited"] is True, "parent reap uncertain")
        require(type(result["exit_code"]) is int, "parent exit code")
        return result["exit_code"]


class NativeProcesses:
    """Future-gated os/ctypes process primitives for ProcessRoles.

    The owner must inspect the actual stopped child against the full manifest;
    setup success alone is never a confinement receipt. No process is spawned
    on construction. Kernel observations/DAC/broker/cgroup-domain admission are
    supplied by the independently pinned host inspection interface.
    """
    simulated = False

    def __init__(self, authorization, grant, host_id, attempt, manifest):
        require_adapter_grant(authorization, grant, host_id, attempt, manifest, "AUTHORITY")
        self.context = (authorization, grant, host_id, attempt, copy.deepcopy(manifest))
        self.attempt = attempt
        self.cgroups = set()
        self.nvml_helper = self.nvml_helper_child = None
        self.handoff_child_pid = None

    def _check(self, action):
        import sys
        require_adapter_grant(*self.context, action)
        require(sys.platform == "linux", "Linux process interfaces required")

    def fork(self):
        self._check("CONFINEMENT")
        require(self.nvml_helper is None and self.nvml_helper_child is None, "NVML helper active at handoff fork")
        require(os.geteuid() == 0 and len(os.listdir("/proc/self/task")) == 1, "single-threaded privileged fork required")
        return os.fork()

    def resume_wait_worker(self, pid):
        self._check("AUTHORITY")
        require(uint(pid, 32) > 0, "owned worker pid required")
        import signal
        os.kill(pid, signal.SIGCONT)
        found, status = os.waitpid(pid, os.WUNTRACED)
        require(found == pid, "exact worker wait required")
        if os.WIFSTOPPED(status):
            require(os.WSTOPSIG(status) == signal.SIGSTOP, "failed worker stop mismatch")
            return {"pid": pid, "stopped": True}
        require(os.WIFEXITED(status), "worker exit uncertainty")
        return {"pid": pid, "exited": True, "exit_code": os.WEXITSTATUS(status)}

    @staticmethod
    def failed_hold_forever():
        # SIGCONT is not recovery authority. Its sole continuation is SIGSTOP.
        import signal
        stop, pid, sig = os.kill, os.getpid(), signal.SIGSTOP
        signal.pthread_sigmask(signal.SIG_SETMASK, set(signal.valid_signals()) - {signal.SIGKILL, signal.SIGSTOP})
        while True:
            stop(pid, sig)

    def install_child_stdio(self, manifest):
        self._check("CONFINEMENT")
        value = LaunchManifest.parse(LaunchManifest(manifest).values()).values()
        validate_stdio(value["stdio"])
        require(os.geteuid() == 0 and self.handoff_child_pid == os.getpid(), "stdio requires privileged handoff child")
        import fcntl
        # Reserve standard slots: every setup open must be a temporary >=3.
        for fd in range(3):
            os.fstat(fd)

        def identity(fd, entry, flags=False):
            info = os.fstat(fd)
            backing = entry["backing"]
            require(stat.S_ISREG(info.st_mode) and info.st_ino == backing["inode"]
                    and f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}" == backing["device"]
                    and os.readlink(f"/proc/self/fd/{fd}") == backing["path"], "stdio exact regular object/path")
            if flags:
                require(fcntl.fcntl(fd, fcntl.F_GETFL) == entry["flags"], "stdio final status flags")

        for entry in value["stdio"]:
            target, temporary = None, None
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                parts = Path(entry["backing"]["path"]).parts[1:]
                for part in parts[:-1]:
                    info = os.fstat(directory)
                    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                            "uncontrolled stdio ancestor")
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
                    os.close(directory)
                    directory = child
                info = os.fstat(directory)
                require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                        "uncontrolled stdio parent")
                # O_PATH pins without opening a device/FIFO. NOFOLLOW returns a
                # symlink object, which the regular-file check rejects. Reopen
                # only that pinned regular object, without persisting NOFOLLOW
                # in its Linux F_GETFL status flags. No create/truncate anywhere.
                target = os.open(parts[-1], os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
                identity(target, entry)
                mode = os.O_RDONLY if entry["fd"] == 0 else os.O_WRONLY | os.O_APPEND
                temporary = os.open(f"/proc/self/fd/{target}", mode | os.O_CLOEXEC)
                identity(temporary, entry, True)
                os.dup2(temporary, entry["fd"], inheritable=True)
            finally:
                for descriptor in (temporary, target, directory):
                    if descriptor is not None:
                        os.close(descriptor)
        descriptors = [int(n) for n in os.listdir("/proc/self/fd")]
        os.closerange(3, max([3, *descriptors]) + 1)
        for entry in value["stdio"]:
            identity(entry["fd"], entry, True)
            require(os.get_inheritable(entry["fd"]), "stdio must survive exec")

    def _nvml_action(self, action):
        require(action in ("EXCLUSION", "CONFINEMENT"), "NVML helper action scope")
        self._check(action)
        require(os.geteuid() == 0, "privileged NVML helper required")

    def fork_nvml_helper(self, action):
        self._nvml_action(action)
        require(self.nvml_helper is None and self.nvml_helper_child is None, "NVML helper already active")
        threads = len(os.listdir("/proc/self/task"))
        require(threads == 1, "single-threaded NVML helper fork required")
        owner = os.getpid()
        pid = os.fork()
        require(type(pid) is int and 0 <= pid < 2 ** 32, "NVML helper fork PID")
        if pid == 0:
            self.nvml_helper_child = (os.getpid(), action)
        else:
            require(pid != owner, "NVML helper must be a child")
            self.nvml_helper = (pid, action, owner, threads)
        return pid

    def wait_nvml_helper(self, pid, action):
        self._nvml_action(action)
        require(uint(pid, 32) > 0 and self.nvml_helper is not None
                and self.nvml_helper[:3] == (pid, action, os.getpid())
                and self.nvml_helper_child is None, "owned NVML helper PID/action required")
        before = self.nvml_helper[3]
        found, status = os.waitpid(pid, 0)
        require(found == pid, "exact NVML helper reap required")
        self.nvml_helper = None
        after = len(os.listdir("/proc/self/task"))
        require(after == 1, "single-threaded worker after NVML helper reap required")
        require(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, "NVML helper failed or signaled")
        return {"pid": pid, "exit_code": os.WEXITSTATUS(status),
                "parent_threads_before": before, "parent_threads_after": after}

    def exit_nvml_helper(self, code, action):
        self._nvml_action(action)
        require(self.nvml_helper is None and self.nvml_helper_child == (os.getpid(), action),
                "only owned NVML helper may exit")
        require(type(code) is int and code in (0, 1), "NVML helper exit status")
        os._exit(code)

    def wait_once(self, pid, stopped):
        self._check("CONFINEMENT" if stopped else "P1R_PARENT")
        require(uint(pid, 32) > 0, "owned child pid required")
        found, status = os.waitpid(pid, os.WUNTRACED if stopped else 0)
        require(found == pid, "owned child reap mismatch")
        if stopped:
            import signal
            require(os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGSTOP, "handoff stop mismatch")
            return {"pid": pid, "stopped": True}
        require(os.WIFEXITED(status), "child signal/exit uncertainty")
        return {"pid": pid, "exited": True, "exit_code": os.WEXITSTATUS(status)}

    def resume_wait_once(self, pid):
        self._check("P1R_PARENT")
        require(uint(pid, 32) > 0, "owned stopped child pid required")
        import signal
        # All checks/imports/hashes precede release. Nothing except the one
        # blocking wait/reap runs until the child has returned.
        os.kill(pid, signal.SIGCONT)
        found, status = os.waitpid(pid, 0)
        require(found == pid and os.WIFEXITED(status), "child reap uncertainty")
        return {"pid": pid, "exited": True, "exit_code": os.WEXITSTATUS(status)}

    def create_accounting_cgroup(self, parent, attempt):
        self._check("CONFINEMENT")
        require(attempt == self.attempt, "cgroup attempt scope")
        # System must prove domain neutrality before this call; only membership
        # objects are created. No controller, limit, weight or BPF is written.
        path_bytes(parent)
        require(parent == "/sys/fs/cgroup" or parent.startswith("/sys/fs/cgroup/"), "cgroup backing")
        for ancestor in (Path(parent), *Path(parent).parents):
            info = ancestor.lstat()
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and info.st_mode & 0o022 == 0,
                    "cgroup ancestor delegated/uncontrolled")
        child = Path(parent) / attempt
        child.mkdir(mode=0o700)
        self.cgroups.add(str(child))
        return str(child)

    def enter_cgroup(self, cgroup):
        self._check("CONFINEMENT")
        require(cgroup in self.cgroups, "unowned accounting cgroup")
        fd = os.open(Path(cgroup) / "cgroup.procs", os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            value = str(os.getpid()).encode("ascii")
            require(os.write(fd, value) == len(value), "cgroup membership short write")
        finally:
            os.close(fd)

    @staticmethod
    def _prctl(libc, option, a=0, b=0):
        NativePrimitives._result(libc.prctl(option, a, b, 0, 0))

    def apply_launch_credentials_and_close_fds(self, manifest):
        self._check("CONFINEMENT")
        import ctypes
        import resource
        import signal
        import sys
        value = LaunchManifest(manifest).values()
        LaunchManifest.parse(value)
        require(os.geteuid() == 0, "privileged handoff setup required")
        for name in RLIMITS:
            expected = tuple(resource.RLIM_INFINITY if v == "INFINITY" else v for v in value["rlimits"][name])
            require(resource.getrlimit(linux_rlimit_id(resource, name, sys.platform)) == expected, "inherited RLIMIT drift")
        require(sorted(os.sched_getaffinity(0)) == value["cpu_affinity"]
                and os.getpriority(os.PRIO_PROCESS, 0) == value["nice"], "inherited scheduling drift")
        self.install_child_stdio(manifest)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        libc.prctl.restype = ctypes.c_int
        # Read inherited mode only. There is no filter installation interface.
        require(libc.prctl(21, 0, 0, 0, 0) == 0, "inherited seccomp rejected")
        self._prctl(libc, 38, 1)  # PR_SET_NO_NEW_PRIVS
        self._prctl(libc, 47, 4)  # PR_CAP_AMBIENT / CLEAR_ALL
        last_cap = int(Path("/proc/sys/kernel/cap_last_cap").read_text().strip())
        require(0 <= last_cap <= 63, "unsupported capability ABI")
        for cap in range(last_cap + 1):
            self._prctl(libc, 24, cap)  # PR_CAPBSET_DROP; bounded, no retry
        self._prctl(libc, 8, 0)  # PR_SET_KEEPCAPS off
        ids = value["credentials"]
        os.setgroups(ids["groups"])
        os.setresgid(ids["gid"], ids["gid"], ids["gid"])
        os.setresuid(ids["uid"], ids["uid"], ids["uid"])

        class Header(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

        class Data(ctypes.Structure):
            _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]

        header, data = Header(0x20080522, 0), (Data * 2)()
        libc.capset.argtypes, libc.capset.restype = [ctypes.POINTER(Header), ctypes.POINTER(Data)], ctypes.c_int
        NativePrimitives._result(libc.capset(ctypes.byref(header), data))
        os.umask(value["umask"])
        os.chdir(value["cwd"])
        dispositions = value["signals"]["dispositions"]
        valid = {int(n) for n in signal.valid_signals()} - {int(signal.SIGKILL), int(signal.SIGSTOP)}
        require(set(dispositions) == {str(n) for n in valid}, "complete exec signal dispositions required")
        for number in sorted(valid):
            disposition = dispositions[str(number)]
            require(disposition in ("DEFAULT", "IGNORE"), "unreviewed signal handler")
            signal.signal(number, signal.SIG_DFL if disposition == "DEFAULT" else signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, value["signals"]["mask"])
        descriptors = [int(n) for n in os.listdir("/proc/self/fd")]
        os.closerange(3, max([3, *descriptors]) + 1)

    def stop_self(self):
        self._check("CONFINEMENT")
        import signal
        os.kill(os.getpid(), signal.SIGSTOP)

    def stop_then_exec_parent(self, argv, manifest):
        """Prepare everything before SIGSTOP; SIGCONT returns directly to exec.

        The canonical launch manifest supplies a private environment dictionary
        and must match the externally granted adapter manifest. The guardian
        independently rechecks P1R_PARENT authority before releasing this child.
        """
        self._check("P1R_PARENT")
        import signal
        require(type(argv) is tuple and argv == parent_argv(self.attempt), "only unchanged P1R parent may execute")
        require(type(manifest) is bytes, "canonical launch manifest bytes required")
        launch = LaunchManifest.parse(LaunchManifest(manifest).values())
        require(launch.encoded == manifest and launch.sha256 == self.context[4]["launch_manifest_sha256"],
                "exec launch manifest differs from grant")
        environment = launch.values()["environment"]["values"]
        executable = argv[0]
        pid = os.getpid()
        os.kill(pid, signal.SIGSTOP)
        os.execve(executable, argv, environment)

    def exit_worker(self, code):
        self._check("AUTHORITY")
        require(type(code) is int and code in (0, 1), "worker exit status")
        os._exit(code)


def backing_identity(value, topology, storage_identity):
    validate_storage_identity(storage_identity)
    validate_storage_identity(value)
    require(value == storage_identity, "topology differs from external storage grant")
    require(value["disk_sequence"] == topology["disk_sequence"], "retained storage topology drift")


def validate_execution_scope(receipt, evidence):
    require(type(receipt["host_id"]) is str and receipt["host_id"], "observed host identity required")
    scope = receipt["execution_grant"]
    if evidence["mode"] == "SIMULATED":
        exact_keys(scope, ("status", "storage_identity"), "synthetic scope coverage")
        require(scope["status"] == "NOT_GRANTED", "portable evidence must not possess live grant")
        validate_storage_identity(scope["storage_identity"])
        return
    exact_keys(scope, ("authority_issue", "host_id", "attempt", "operations", "adapter_manifest_sha256", "storage_identity"), "live scope coverage")
    require(type(scope["authority_issue"]) is int and scope["authority_issue"] > 230, "live grant must postdate implementation")
    LaunchManifest.parse(receipt["launch_manifest"], authority_issue=scope["authority_issue"])
    validate_storage_identity(scope["storage_identity"])
    require(scope["host_id"] == receipt["host_id"] and scope["attempt"] == evidence["attempt"]
            and scope["operations"] == [a.name for a in plan(evidence["attempt"])]
            and scope["adapter_manifest_sha256"] == digest(canonical_json(receipt["adapter_manifest"])), "live grant evidence binding")


def parse_mountinfo(text):
    """Pure parser keeping per-mount and superblock RO evidence independent.

    A read-only bind has ro before ' - ' and can still have rw after it. It
    cannot satisfy filesystem-wide protection. Ambiguous/missing records stop.
    """
    import re

    def unescape(value):
        return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match.group(1), 8)), value)

    records = []
    seen = set()
    for line in text.splitlines():
        sides = line.split(" - ")
        require(len(sides) == 2, "malformed mountinfo separator")
        left, right = sides[0].split(), sides[1].split()
        require(len(left) >= 6 and len(right) == 3, "malformed mountinfo fields")
        mount_id, parent_id = int(left[0]), int(left[1])
        require(mount_id > 0 and parent_id > 0 and mount_id not in seen, "ambiguous mount identity")
        seen.add(mount_id)
        mount_options, super_options = left[5].split(","), right[2].split(",")
        require(len(set(mount_options) & {"rw", "ro"}) == 1
                and len(set(super_options) & {"rw", "ro"}) == 1, "unknown mount/superblock state")
        records.append({"mount_id": mount_id, "parent_id": parent_id, "major_minor": left[2],
                        "root": unescape(left[3]), "target": unescape(left[4]), "fstype": right[0],
                        "source": unescape(right[1]), "mount_options": mount_options,
                        "super_options": super_options, "mount_ro": "ro" in mount_options,
                        "superblock_ro": "ro" in super_options, "propagation": left[6:]})
    require(records, "empty mountinfo")
    return records
