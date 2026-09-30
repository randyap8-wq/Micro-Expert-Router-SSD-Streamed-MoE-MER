"""P1V5 direct observations. Live execution requires a future issue >230.

Import performs no host operation. Live construction needs three matching
external authorities. Observations fail closed; expected state is never an
observation. Linux v6.8 NUMA inheritance and clock normalization are explicit
transition certificates, not target-task or opaque-clock readbacks.
"""

import copy
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

from . import (AUTHORITY, EXPECTATIONS, ALIAS_PARENT, CANONICAL, CONFIGURED,
               MOUNT, IDS, STARTUP_IDS, SOURCE_FILES, TOOLS,
               FutureAuthorization, InvalidEvidence, canonical_json, digest,
               path_bytes, require, sha256, strict_json, uint, validate_storage_identity,
               observed_local_ssd_aliases)
from . import linux_backend as lb
from .content_hasher import Identity, alternatives
from .supervisor import BIN, CONFIG, REQUEST, OUTPUT_ROOT, parent_argv, plan, validate_attempt

FLOOR = 230
TOPICS = ("authority", "exclusion", "topology", "roles", "namespace", "alias_bind",
          "alias_ro", "metadata", "sync_health", "filesystem_ro", "device_ro",
          "protected", "cgroup_preflight", "confinement", "shutdown", "device_rw",
          "filesystem_rw", "release")
PROTECTED = {
    "__init__.py": "a936eec1bd452981180dd06ad236caf84040ba3d3275a5ad56de5a570228b1be",
    "content_hasher.py": "dec979b29cb55f0e61afd08173267d338bafe120ebffa98b4305cbaca42f3085",
    "offline_verify.py": "6008eceb36210b2755ff4ba3d5f0970f0656a919bdcfc6151a65f52562b86537",
    "supervisor.py": "c0f57a524814ae12677af8d42a65fa08b900a10ba183093d6006bdbf5df354f8",
    "test_p1v.py": "0035021c0532c56d2934ede1e5db976876fd07024d60744039ef6555005752e5",
    "linux_backend.py": "b67cd701c721157e96c56ffad25ac1abde40dd5508138f0408811531b5c1b484",
    "test_linux_backend.py": "542dd08c07525070e579f57971d8fe4ce3f669c1abec96aaec53429588cbc535",
}
NUMA_ABI = {"schema": "mer.p1v.numa-abi.v1", "architecture": "x86_64",
            "kernel": "6.8.0-1069-gcp", "source_profile": "Linux-v6.8",
            "syscall": 239, "word_bits": 64, "maxnode": 1024, "query_flags": 0}
MPOL_NAMES = ("MPOL_DEFAULT", "MPOL_PREFERRED", "MPOL_BIND", "MPOL_INTERLEAVE",
              "MPOL_LOCAL", "MPOL_PREFERRED_MANY")
STATIC, RELATIVE, BALANCING = 1 << 15, 1 << 14, 1 << 13
MODE_FLAGS = STATIC | RELATIVE | BALANCING
REPORT_NAMES = {"parent": "p1r.json", "control": "p1r.json.control.json",
                "treatment": "p1r.json.treatment.json"}
REPORT_LOGS = tuple("p1r.json." + arm + "." + stream + ".log"
                    for arm in ("control", "treatment") for stream in ("stdout", "stderr"))


def integer_list(text, limit=65536):
    """Strict Linux list ABI; no duplicate/overlapping or reverse ranges."""
    import re
    require(type(text) is str and text.strip(), "empty CPU/node list")
    result = []
    for item in text.strip().split(","):
        require(re.fullmatch(r"[0-9]+(?:-[0-9]+)?", item), "malformed CPU/node list")
        bounds = [int(n) for n in item.split("-")]
        lo, hi = bounds[0], bounds[-1]
        require(0 <= lo <= hi < limit, "CPU/node bounds")
        result.extend(range(lo, hi + 1))
    require(result == sorted(set(result)), "CPU/node list overlap/order")
    return result


def task_policy(mode, nodemask, maxnode=1024):
    """Exact returned user ABI; never substitute effective allowed mems."""
    require(type(mode) is int and mode >= 0, "malformed MPOL mode")
    base, flags = mode & ~MODE_FLAGS, mode & MODE_FLAGS
    require(base < len(MPOL_NAMES), "unsupported MPOL mode or flags")
    require(not (flags & STATIC and flags & RELATIVE), "incompatible MPOL flags")
    require(not flags & BALANCING or base == 2, "NUMA_BALANCING requires BIND")
    require(type(maxnode) is int and maxnode == NUMA_ABI["maxnode"], "bounded MPOL ABI size")
    require(type(nodemask) is bytes and len(nodemask) == maxnode // 8,
            "malformed/truncated MPOL nodemask")
    nodes = [n for n in range(maxnode) if nodemask[n // 8] & (1 << (n % 8))]
    require(base not in (0, 4) or not nodes, "DEFAULT/LOCAL nodemask must be empty")
    require(base not in (2, 3, 5) or nodes, "policy requires nodemask")
    require(base != 4 or not flags, "LOCAL mode flags")
    value = {"schema": "mer.p1v.numa-task-policy.v1", "mode": base,
             "mode_name": MPOL_NAMES[base], "mode_flags": flags, "nodemask": nodes,
             "maxnode": maxnode, "source": "GET_MEMPOLICY_CURRENT_TASK",
             "inheritance": "FORK_EXEC"}
    return canonical_json(value).decode("ascii").rstrip("\n")


def parse_status(text):
    fields = {}
    for line in text.splitlines():
        require(":" in line, "malformed proc status")
        key, value = line.split(":", 1)
        require(key not in fields, "duplicate proc status field")
        fields[key] = value.strip()
    needed = ("Pid", "PPid", "Tgid", "State", "Threads", "Uid", "Gid", "Groups",
              "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs",
              "Seccomp", "Cpus_allowed_list", "Mems_allowed_list", "Umask",
              "SigBlk", "SigIgn", "SigCgt")
    require(set(needed) <= set(fields), "incomplete proc status")
    return fields


def parse_stat(text):
    end = text.rfind(")")
    require(end > 0 and text.count("(") >= 1, "malformed proc stat")
    pid = int(text.split("(", 1)[0])
    tail = text[end + 1:].split()
    require(len(tail) >= 20, "truncated proc stat")
    return {"pid": pid, "state": tail[0], "ppid": int(tail[1]),
            "pgrp": int(tail[2]), "session": int(tail[3]), "starttime": int(tail[19])}


def ext4_superblock(data):
    """Only the 1024-byte primary superblock at device offset 1024."""
    import uuid
    require(type(data) is bytes and len(data) == 1024, "truncated ext4 superblock")
    u16 = lambda offset: struct.unpack_from("<H", data, offset)[0]
    u32 = lambda offset: struct.unpack_from("<I", data, offset)[0]
    require(u16(0x38) == 0xEF53, "ext4 magic")
    compat, incompat, ro_compat = u32(0x5C), u32(0x60), u32(0x64)
    require(compat & 4, "ext4 internal journal required")
    external = bool(incompat & 8 or u32(0xE4) or any(data[0xD0:0xE0]))
    require(not external, "external journal forbidden")
    require(u32(0xE0) > 0, "journal inode absent")
    state = u16(0x3A)
    require(state & ~7 == 0 and not state & 6 and u32(0x194) == 0,
            "ext4 recorded errors/orphan recovery uncertainty")
    require(u32(0x18) <= 2 and u32(0x4C) in (0, 1), "unsupported ext4 geometry/revision")
    if ro_compat & 0x400:
        require(data[0x175] == 1, "unsupported ext4 checksum type")
        crc = 0xFFFFFFFF
        for byte in data[:0x3FC]:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        require(crc == u32(0x3FC), "ext4 superblock checksum")
    return {"uuid": str(uuid.UUID(bytes=data[0x68:0x78])), "healthy": True,
            "external_journal": False, "recovery_required": bool(incompat & 4),
            "features": [compat, incompat, ro_compat], "state": state}


def validate_provider_manifest(value, adapter, launch, closure):
    lb.exact_keys(value, ("schema", "implementation_issue", "modules", "enforcement_schema",
                         "enforcement_profile", "runtime", "dependencies", "launch_manifest_sha256",
                         "execution_closure_sha256", "numa_abi", "nvml", "host_profile"),
                  "provider manifest coverage")
    require(value["schema"] == "mer.p1v.provider.v1" and value["implementation_issue"] == 217,
            "provider schema/implementation issue")
    lb.exact_keys(value["modules"], ("live_system.py", "linux_gate.py", "linux_backend.py"), "provider modules")
    for pin in value["modules"].values():
        sha256(pin)
    require(value["modules"]["linux_backend.py"] == PROTECTED["linux_backend.py"]
            and adapter["module_sha256"] == PROTECTED["linux_backend.py"], "reviewed adapter pin")
    baseline = lb.validate_adapter_manifest(adapter, launch, closure)
    require(value["launch_manifest_sha256"] == baseline.sha256
            and value["execution_closure_sha256"] == digest(canonical_json(closure))
            and value["dependencies"] == adapter["dependencies"], "provider closure pins")
    require(value["enforcement_schema"] == lb.SCHEMA and value["enforcement_profile"] == lb.PROFILE,
            "provider enforcement profile")
    require(value["numa_abi"] == NUMA_ABI, "reviewed NUMA ABI/profile")
    lb.exact_keys(value["runtime"], ("implementation", "version", "cache_tag", "machine", "byteorder"),
                  "runtime ABI coverage")
    require(value["runtime"]["implementation"] == "cpython"
            and value["runtime"]["machine"] == "x86_64" and value["runtime"]["byteorder"] == "little",
            "supported runtime ABI")
    lb.known(value)
    validate_host_profile(value["host_profile"])
    lb.exact_keys(value["nvml"], ("path", "sha256", "uuid", "pci_bus_id", "driver", "version",
                                "dedicated_host", "normalize_default_unlocked"), "NVML closure")
    path_bytes(value["nvml"]["path"])
    sha256(value["nvml"]["sha256"])
    require(value["nvml"]["driver"] == EXPECTATIONS["driver"]
            and value["nvml"]["dedicated_host"] is True
            and value["nvml"]["normalize_default_unlocked"] is True, "dedicated-host normalization authority")
    return digest(canonical_json(value))


def validate_host_profile(value):
    lb.exact_keys(value, ("source_root", "dependency_files", "cgroup_parent", "lock_path",
                         "trusted_evidence_root", "namespace_services", "policy_files",
                         "gpu_inventory_roots", "kernel_source_profile", "administrative_noninterference"),
                  "host profile coverage")
    for key in ("source_root", "cgroup_parent", "lock_path", "trusted_evidence_root"):
        path_bytes(value[key])
    require(value["kernel_source_profile"] == "Linux-v6.8"
            and value["administrative_noninterference"] == "TRUSTED_HOST_PREMISE", "host inheritance premise")
    require(value["trusted_evidence_root"] != OUTPUT_ROOT
            and not value["trusted_evidence_root"].startswith(OUTPUT_ROOT + "/")
            and not OUTPUT_ROOT.startswith(value["trusted_evidence_root"] + "/"), "report/evidence root separation")
    require(type(value["namespace_services"]) is dict and type(value["policy_files"]) is dict,
            "explicit inspection inventory")
    for path, pin in value["dependency_files"].items():
        path_bytes(path)
        sha256(pin)
    require(value["dependency_files"], "complete direct dependency pins required")
    lb.exact_keys(value["policy_files"], ("cpu_policy", "thp", "numa_balancing"), "kernel policy coverage")
    lb.exact_keys(value["policy_files"]["cpu_policy"], ("governors", "frequency_limits", "boost", "power"), "CPU policy coverage")
    for group in value["policy_files"].values():
        require(type(group) is dict and group, "policy inventory cannot be empty")
        for paths in group.values():
            require(type(paths) is list and paths and paths == sorted(set(paths)), "ordered complete policy path set")
            for path in paths:
                path_bytes(path)
                require(path.startswith(("/sys/", "/proc/sys/")), "direct kernel policy path")
    lb.exact_keys(value["gpu_inventory_roots"], ("nodes", "libraries", "icds", "layers", "cache"), "GPU inventory coverage")
    for paths in value["gpu_inventory_roots"].values():
        require(type(paths) is list and paths and paths == sorted(set(paths)), "GPU inventory paths required")
        for path in paths:
            path_bytes(path)


@dataclass(frozen=True, init=False)
class ProviderGrant:
    """External scope interlock; this implementation issues no grants."""
    encoded: bytes

    def __init__(self, scope, expected_scope_sha256, token, expected_token_sha256):
        lb.exact_keys(scope, ("authority_issue", "host_id", "attempt", "operations",
                             "provider_manifest_sha256", "storage_identity"), "provider grant scope")
        require(type(scope["authority_issue"]) is int and scope["authority_issue"] > FLOOR,
                "execution authority must follow #230")
        require(type(scope["host_id"]) is str and scope["host_id"], "provider host ID")
        validate_attempt(scope["attempt"])
        require(scope["operations"] == [a.name for a in plan(scope["attempt"])], "full provider operation scope")
        sha256(scope["provider_manifest_sha256"])
        validate_storage_identity(scope["storage_identity"])
        require(digest(canonical_json(scope)) == sha256(expected_scope_sha256), "external provider scope pin")
        require(type(token) is bytes and len(token) >= 32
                and digest(token) == sha256(expected_token_sha256), "external provider token")
        object.__setattr__(self, "encoded", canonical_json(scope))

    @property
    def storage_identity(self):
        return self.public_scope()["storage_identity"]

    def public_scope(self):
        return strict_json(self.encoded)


def require_live(authorization, adapter_grant, provider_grant, host_id, attempt, adapter, provider, operation):
    require(type(authorization) is FutureAuthorization and type(authorization.issue) is int
            and authorization.issue > FLOOR, "P1V5 live authority must follow #230")
    require(type(provider_grant) is ProviderGrant, "external provider-specific grant required")
    scope = provider_grant.public_scope()
    require(scope["authority_issue"] == authorization.issue and scope["host_id"] == host_id
            and scope["attempt"] == attempt and operation in scope["operations"]
            and scope["provider_manifest_sha256"] == digest(canonical_json(provider)), "provider grant mismatch")
    lb.require_adapter_grant(authorization, adapter_grant, host_id, attempt, adapter, operation)
    require(adapter_grant.storage_identity == provider_grant.storage_identity, "independent storage grants mismatch")


@dataclass(frozen=True)
class InheritanceSnapshot:
    encoded: bytes

    @classmethod
    def capture(cls, *, attempt, host_id, worker, kernel, policy, allowed, source, target,
                hierarchy, exclusion, baseline):
        require(exclusion is True and kernel == NUMA_ABI and hierarchy == "cgroup2",
                "NUMA kernel/hierarchy/exclusion premise")
        lb.exact_keys(worker, ("pid", "starttime", "threads"), "worker identity")
        require(uint(worker["pid"], 32) > 0 and worker["threads"] == 1, "single-threaded exact worker")
        require(policy == baseline["numa"]["memory_policy"], "prospective task-policy mismatch")
        require(allowed == baseline["numa"]["allowed_nodes"], "prospective allowed-node mismatch")
        lb.resource_domain(source)
        lb.resource_domain(target)
        require(source == target == baseline["resource_domain"], "complete resource domain/no-rebind mismatch")
        return cls(canonical_json({"attempt": validate_attempt(attempt), "host_id": host_id,
                                  "worker": worker, "kernel": kernel, "policy": policy,
                                  "abi_fields": strict_json(policy), "allowed": allowed,
                                  "source": source, "target": target, "hierarchy": hierarchy,
                                  "administrative_noninterference": "TRUSTED_HOST_PREMISE"}))

    def bind_child(self, *, attempt, host_id, worker, returned_pid, child, source, target, baseline):
        value = strict_json(self.encoded)
        require(value["attempt"] == attempt and value["host_id"] == host_id
                and value["worker"] == worker, "NUMA attempt/exact worker binding")
        require(child["pid"] == returned_pid and child["ppid"] == worker["pid"]
                and child["stopped"] is True and child["threads"] == 1, "exact stopped fork child required")
        require(source == value["source"] and target == value["target"], "NUMA no-rebind drift")
        require(child["allowed_mems"] == value["allowed"] == baseline["numa"]["allowed_nodes"],
                "actual child allowed memory-node drift")
        require(child["allowed_cpus"] == baseline["cpu_affinity"], "actual child allowed CPU drift")
        require(value["policy"] == baseline["numa"]["memory_policy"], "NUMA approved policy drift")
        return {"allowed_nodes": child["allowed_mems"], "memory_policy": value["policy"]}


class DirectLinux:
    """Pinned direct host ABI. Every entry is guarded before any host access."""
    def __init__(self, context, guard):
        require_live(*context, "AUTHORITY")
        require(callable(guard), "provider phase guard required")
        self.context, self.guard = context, guard
        self._storage_identity = canonical_json(context[2].storage_identity)

    @property
    def storage_identity(self):
        return strict_json(self._storage_identity)

    def authorize(self):
        require_live(*self.context, "AUTHORITY")

    def check(self):
        self.authorize()
        self.guard()

    def read(self, path, limit=1048576):
        self.check()
        require(not str(path).startswith((CANONICAL + "/", CONFIGURED + "/")), "expert payload read forbidden")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            require(major_minor(os.fstat(fd)) != self.storage_identity["major_minor"], "expert backing read forbidden")
            chunks, size = [], 0
            while True:
                part = os.read(fd, min(65536, limit + 1 - size))
                if not part:
                    break
                chunks.append(part)
                size += len(part)
                require(size <= limit, "bounded host observation overflow")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def text(self, path, limit=1048576):
        return self.read(path, limit).decode("utf-8", "strict")

    def listdir(self, path):
        self.check()
        return sorted(os.listdir(path))

    def descriptors(self, pid):
        self.check()
        with os.scandir(f"/proc/{pid}/fd") as directory:
            for entry in directory:
                self.check()
                yield entry.name

    def stat(self, path, follow=False):
        self.check()
        return os.stat(path, follow_symlinks=follow)

    def link(self, path):
        self.check()
        return os.readlink(path)

    def libc(self):
        self.check()
        import ctypes
        import sys
        require(sys.platform == "linux" and os.uname().machine == "x86_64", "pinned Linux x86-64 ABI")
        return ctypes.CDLL(None, use_errno=True)

    def task_policy(self):
        self.check()
        import ctypes
        libc = self.libc()
        libc.syscall.restype = ctypes.c_long
        mode = ctypes.c_int(-1)
        mask = (ctypes.c_ulong * (NUMA_ABI["maxnode"] // 64))()
        result = libc.syscall(ctypes.c_long(239), ctypes.byref(mode), mask,
                              ctypes.c_ulong(NUMA_ABI["maxnode"]), ctypes.c_void_p(), ctypes.c_ulong(0))
        require(result == 0, "get_mempolicy task-policy syscall failure")
        return task_policy(mode.value, bytes(mask))

    def dio_alignment(self, fd):
        self.check()
        import ctypes
        libc = self.libc()
        libc.statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_uint, ctypes.c_void_p]
        libc.statx.restype = ctypes.c_int
        buffer = (ctypes.c_ubyte * 256)()
        require(libc.statx(fd, b"", 0x1000, 0x2000, buffer) == 0, "STATX_DIOALIGN failed")
        data = bytes(buffer)
        require(struct.unpack_from("<I", data, 0)[0] & 0x2000, "STATX_DIOALIGN unavailable")
        memory, offset = struct.unpack_from("<II", data, 152)
        require(memory > 0 and offset > 0 and 4096 % memory == 0 and 4096 % offset == 0,
                "unsupported direct I/O alignment")
        return {"memory": memory, "offset": offset, "source": "STATX_DIOALIGN"}

    def superblock(self, fd):
        self.check()
        require(stat.S_ISBLK(os.fstat(fd).st_mode), "bounded metadata requires retained block descriptor")
        return ext4_superblock(os.pread(fd, 1024, 1024))

    def namespace(self, pid, kind):
        require(kind in ("mnt", "user"), "namespace kind")
        info = self.stat(f"/proc/{pid}/ns/{kind}", True)
        return f"{info.st_dev}:{info.st_ino}"

    def status(self, pid):
        return parse_status(self.text(f"/proc/{pid}/status"))

    def identity(self, pid):
        s, p = self.status(pid), parse_stat(self.text(f"/proc/{pid}/stat"))
        require(int(s["Pid"]) == pid == p["pid"] == int(s["Tgid"]), "process identity mismatch")
        return {"pid": pid, "starttime": p["starttime"], "threads": int(s["Threads"])}

    def file_hash(self, path):
        self.check()
        import hashlib
        require(not str(path).startswith((CANONICAL + "/", CONFIGURED + "/", "/dev/")), "payload hash forbidden")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and major_minor(before) != self.storage_identity["major_minor"], "dependency must be regular and off expert backing")
            result = hashlib.sha256()
            while True:
                data = os.read(fd, 1048576)
                if not data:
                    break
                result.update(data)
            after = os.fstat(fd)
            require(stable_stat(before) == stable_stat(after), "dependency changed during hash")
            return result.hexdigest()
        finally:
            os.close(fd)

    def bpf_ids(self, path):
        """Read-only BPF_PROG_QUERY; no program load/attach operation exists."""
        self.check()
        import ctypes
        libc = self.libc()
        libc.syscall.restype = ctypes.c_long

        class Query(ctypes.Structure):
            _fields_ = [("target_fd", ctypes.c_uint32), ("attach_type", ctypes.c_uint32),
                        ("query_flags", ctypes.c_uint32), ("attach_flags", ctypes.c_uint32),
                        ("prog_ids", ctypes.c_uint64), ("prog_cnt", ctypes.c_uint32),
                        ("padding", ctypes.c_uint32), ("prog_attach_flags", ctypes.c_uint64)]

        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            result = []
            # v6.8 cgroup attachment types from include/uapi/linux/bpf.h.
            for kind in (0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22,
                         29, 30, 31, 32, 34, 43, 49, 50, 51, 52, 53):
                ids = (ctypes.c_uint32 * 256)()
                query = Query(fd, kind, 0, 0, ctypes.addressof(ids), 256, 0, 0)
                ret = libc.syscall(ctypes.c_long(321), ctypes.c_int(16), ctypes.byref(query), ctypes.sizeof(query))
                require(ret == 0 and query.prog_cnt <= 256, "cgroup BPF inventory unavailable/overflow")
                result.extend({"type": kind, "id": int(ids[i]), "flags": query.attach_flags}
                              for i in range(query.prog_cnt))
            return result
        finally:
            os.close(fd)

    def scheduler(self, pid):
        self.check()
        import ctypes

        class Attr(ctypes.Structure):
            _fields_ = [("size", ctypes.c_uint32), ("policy", ctypes.c_uint32),
                        ("flags", ctypes.c_uint64), ("nice", ctypes.c_int32),
                        ("priority", ctypes.c_uint32), ("runtime", ctypes.c_uint64),
                        ("deadline", ctypes.c_uint64), ("period", ctypes.c_uint64),
                        ("util_min", ctypes.c_uint32), ("util_max", ctypes.c_uint32)]

        a = Attr()
        a.size = ctypes.sizeof(a)
        libc = self.libc()
        libc.syscall.restype = ctypes.c_long
        require(libc.syscall(ctypes.c_long(315), ctypes.c_int(pid), ctypes.byref(a),
                             ctypes.c_uint(a.size), ctypes.c_uint(0)) == 0, "sched_getattr failed")
        io = libc.syscall(ctypes.c_long(252), ctypes.c_int(1), ctypes.c_int(pid))
        require(io >= 0, "ioprio_get failed")
        return ({"class": int(a.policy), "attributes": {k: int(getattr(a, k)) for k, _ in a._fields_}},
                {"class": int(io) >> 13, "priority": int(io) & 0x1FFF})


class PinnedNvml:
    """Management-only direct ABI; initialized only after exclusion admission."""
    def __init__(self, io, spec, check):
        require(type(io) is DirectLinux, "authenticated direct Linux interface required")
        io.authorize()
        require(callable(check), "NVML phase guard required")
        self.io, self.spec, self.check = io, copy.deepcopy(spec), check
        self.library = self.handle = self.certificate = None

    def _call(self, name, args, types, optional=False):
        self.check()
        import ctypes
        require(self.library is not None, "NVML not initialized")
        function = getattr(self.library, name)
        function.argtypes, function.restype = types, ctypes.c_int
        status = function(*args)
        require(status == 0 or optional and status == 3, "NVML operation failed: " + name)
        return status

    def _string(self, name, device=True):
        import ctypes
        data = ctypes.create_string_buffer(256)
        args = [self.handle] if device else []
        types = [ctypes.c_void_p] if device else []
        self._call(name, [*args, data, 256], [*types, ctypes.c_char_p, ctypes.c_uint])
        require(data.value and len(data.value) < 255, "NVML string bounds")
        return data.value.decode("ascii")

    def _uint(self, name, selectors=(), optional=False):
        import ctypes
        value = ctypes.c_uint()
        status = self._call(name, [self.handle, *selectors, ctypes.byref(value)],
                            [ctypes.c_void_p, *([ctypes.c_uint] * len(selectors)), ctypes.POINTER(ctypes.c_uint)], optional)
        return {"status": "NVML_ERROR_NOT_SUPPORTED"} if status == 3 else {"status": "NVML_SUCCESS", "value": value.value}

    def identity(self):
        import ctypes

        class PCI(ctypes.Structure):
            _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint),
                        ("bus", ctypes.c_uint), ("device", ctypes.c_uint),
                        ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                        ("busId", ctypes.c_char * 32)]

        pci = PCI()
        self._call("nvmlDeviceGetPciInfo_v3", [self.handle, ctypes.byref(pci)],
                   [ctypes.c_void_p, ctypes.POINTER(PCI)])
        return {"device_uuid": self._string("nvmlDeviceGetUUID"),
                "pci_bus_id": bytes(pci.busId).decode("ascii"),
                "driver_version": self._string("nvmlSystemGetDriverVersion", False),
                "nvml_version": self._string("nvmlSystemGetNVMLVersion", False),
                "device_name": self._string("nvmlDeviceGetName"),
                "nvml_library_sha256": self.io.file_hash(self.spec["path"])}

    def query_policy(self):
        import ctypes
        application = {}
        for domain in (0, 2):
            target = self._uint("nvmlDeviceGetApplicationsClock", (domain,), True)
            default = self._uint("nvmlDeviceGetDefaultApplicationsClock", (domain,), True)
            require(target == default, "legacy application/default clock mismatch")
            application[str(domain)] = {"application": target, "default": default}

        class Offset(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_uint),
                        ("pstate", ctypes.c_uint), ("clockOffsetMHz", ctypes.c_int),
                        ("minClockOffsetMHz", ctypes.c_int), ("maxClockOffsetMHz", ctypes.c_int)]

        offsets = []
        for domain in (0, 2):
            for pstate in range(16):
                entry = Offset(ctypes.sizeof(Offset) | (1 << 24), domain, pstate, 0, 0, 0)
                status = self._call("nvmlDeviceGetClockOffsets", [self.handle, ctypes.byref(entry)],
                                    [ctypes.c_void_p, ctypes.POINTER(Offset)], True)
                value = {"domain": domain, "pstate": pstate,
                         "status": "NVML_ERROR_NOT_SUPPORTED" if status == 3 else "NVML_SUCCESS"}
                if status == 0:
                    value.update(offset=entry.clockOffsetMHz, minimum=entry.minClockOffsetMHz,
                                 maximum=entry.maxClockOffsetMHz)
                offsets.append(value)
        return {"application_clocks": application, "offsets": offsets}

    def normalize(self):
        self.check()
        require(self.library is None and self.certificate is None, "NVML normalization reuse")
        require(self.io.file_hash(self.spec["path"]) == self.spec["sha256"], "pinned NVML library mismatch")
        import ctypes
        self.library = ctypes.CDLL(self.spec["path"], mode=os.RTLD_LOCAL | os.RTLD_NOW)
        self._call("nvmlInit_v2", [], [])
        self.handle = ctypes.c_void_p()
        self._call("nvmlDeviceGetHandleByUUID", [self.spec["uuid"].encode("ascii"), ctypes.byref(self.handle)],
                   [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)])
        identity = self.identity()
        require(identity == {"device_uuid": self.spec["uuid"], "pci_bus_id": self.spec["pci_bus_id"],
                             "driver_version": self.spec["driver"], "nvml_version": self.spec["version"],
                             "device_name": "NVIDIA L4", "nvml_library_sha256": self.spec["sha256"]},
                "GPU/PCI/driver/NVML identity mismatch")
        self._call("nvmlDeviceResetGpuLockedClocks", [self.handle], [ctypes.c_void_p])
        self._call("nvmlDeviceResetMemoryLockedClocks", [self.handle], [ctypes.c_void_p])
        result = {"schema": "mer.p1v.gpu-clock-policy.v1", **identity,
                  "normalization": "DEFAULT_UNLOCKED", "gpu_locked_clock_reset": "NVML_SUCCESS",
                  "memory_locked_clock_reset": "NVML_SUCCESS", "normalization_after_exclusion": True,
                  "normalization_before_launch_baseline": True, "normalization_before_stopped_child": True,
                  "no_clock_policy_action_during_parent": True, **self.query_policy()}
        self.certificate = canonical_json(result)
        return strict_json(self.certificate)

    def observed(self):
        self.check()
        require(self.certificate is not None, "normalization transition absent")
        result = strict_json(self.certificate)
        require(all(result[k] == v for k, v in self.identity().items()), "GPU identity drift")
        require(all(result[k] == v for k, v in self.query_policy().items()), "GPU clock policy drift")
        return result

    def management(self):
        import ctypes
        current, pending = ctypes.c_uint(), ctypes.c_uint()
        status = self._call("nvmlDeviceGetMigMode", [self.handle, ctypes.byref(current), ctypes.byref(pending)],
                            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)], True)
        mig = {"status": "NVML_ERROR_NOT_SUPPORTED"} if status == 3 else {
            "status": "NVML_SUCCESS", "current": current.value, "pending": pending.value}
        require(status == 3 or current.value == pending.value == 0, "MIG active/pending")
        return {"persistence": self._uint("nvmlDeviceGetPersistenceMode"),
                "compute": self._uint("nvmlDeviceGetComputeMode"), "mig": mig,
                "power": {"limit": self._uint("nvmlDeviceGetPowerManagementLimit"),
                          "default": self._uint("nvmlDeviceGetPowerManagementDefaultLimit"),
                          "enforced": self._uint("nvmlDeviceGetEnforcedPowerLimit")}}


def observe_host_identity(io, host_id):
    io.check()
    uname = os.uname()
    require(uname.sysname == "Linux" and uname.release == NUMA_ABI["kernel"]
            and uname.machine == "x86_64", "target kernel/architecture mismatch")
    boot = io.text("/proc/sys/kernel/random/boot_id").strip()
    require(len(boot) == 36 and all(c == "-" if i in (8, 13, 18, 23)
                                  else c in "0123456789abcdef" for i, c in enumerate(boot)),
            "boot ID format")
    machine_id = io.text("/etc/machine-id").strip()
    require(len(machine_id) == 32 and all(c in "0123456789abcdef" for c in machine_id), "machine ID format")
    observed = digest(canonical_json({"machine_id": machine_id, "boot_id": boot,
                                      "kernel": uname.release, "machine": uname.machine}))
    require(observed == host_id, "observed host ID/grant mismatch")
    return boot


def controlled_directory(io, path, runtime_uid=None):
    path_bytes(path)
    for part in reversed((Path(path), *Path(path).parents)):
        info = io.stat(str(part))
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                "root-controlled nonsymlink directory required")
    if runtime_uid is not None:
        require(runtime_uid > 0, "nonroot runtime required")
    return io.stat(path)


def stable_stat(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_nlink,
            info.st_uid, info.st_gid, info.st_mtime_ns, info.st_ctime_ns)


def major_minor(info, device=False):
    value = info.st_rdev if device else info.st_dev
    return f"{os.major(value)}:{os.minor(value)}"


def mount_at(records, path):
    candidates = [r for r in records if path == r["target"] or path.startswith(r["target"].rstrip("/") + "/")]
    require(candidates, "mount backing absent")
    longest = max(len(r["target"]) for r in candidates)
    candidates = [r for r in candidates if len(r["target"]) == longest]
    require(len(candidates) == 1, "ambiguous overmount")
    return candidates[0]


def mount_recipe(record):
    flags = 0
    bits = {"nosuid": 2, "nodev": 4, "noexec": 8, "sync": 16, "dirsync": 128,
            "mand": 64, "noatime": 1024, "nodiratime": 2048, "relatime": 1 << 21,
            "strictatime": 1 << 24, "lazytime": 1 << 25}
    for name in record["mount_options"]:
        require(name in bits or name in ("ro", "rw"), "unsupported mount flag")
        flags |= bits.get(name, 0)
    data = ",".join(v for v in record["super_options"] if v not in ("ro", "rw") and v not in bits)
    return flags, data


class LiveSystem:
    """Complete host provider; the protected adapter owns all phase sequencing."""
    simulated = False

    def __init__(self, *, authorization, grant, provider_grant, host_id, attempt,
                 manifest, provider_manifest, baseline, closure, journal, native, processes,
                 roles, bootstrap):
        # This rejection precedes /proc, /sys, library load, stat and native calls.
        require_live(authorization, grant, provider_grant, host_id, attempt, manifest,
                     provider_manifest, "AUTHORITY")
        validate_provider_manifest(provider_manifest, manifest, baseline, closure)
        require(type(native) is lb.NativePrimitives and type(processes) is lb.NativeProcesses
                and type(roles) is lb.ProcessRoles, "reviewed direct process/native implementations required")
        self.context = (authorization, grant, provider_grant, host_id, attempt,
                        copy.deepcopy(manifest), copy.deepcopy(provider_manifest))
        self._storage_identity = canonical_json(provider_grant.storage_identity)
        self.host_id, self.attempt = host_id, validate_attempt(attempt)
        self.manifest, self.provider, self.closure = map(copy.deepcopy, (manifest, provider_manifest, closure))
        self.baseline = lb.LaunchManifest.parse(baseline)
        self.journal, self.native, self.processes, self.roles = journal, native, processes, roles
        self.bootstrap = copy.deepcopy(bootstrap)
        self.profile = self.provider["host_profile"]
        self.running = self.returned = self.released = self.failed = False
        self.exclusion = False
        self.lock_fd = self.backing = self.topology = self.namespace_receipt = None
        self.child = self.cgroup = self.snapshot = self.numa_receipt = None
        self.snapshot_consumed = False
        self.sync_done = self.ro_transition = self.rw_transition = False
        self.pre_superblock = None
        self.io = DirectLinux(self.context, self._check)
        self.nvml = PinnedNvml(self.io, self.provider["nvml"], self._gpu_check)
        for name, pin in self.provider["modules"].items():
            require(self.io.file_hash(str(Path(__file__).parent / name)) == pin, "provider module hash mismatch")
        for name, pin in PROTECTED.items():
            require(self.io.file_hash(str(Path(__file__).parent / name)) == pin, "protected source hash mismatch")
        self.worker_pid = os.getpid()
        self.guardian_pid = os.getppid()
        self.original_ns = self.io.namespace(self.worker_pid, "mnt")
        self.initial_user_ns = self.io.namespace(1, "user")
        require(self.io.namespace(self.worker_pid, "user") == self.initial_user_ns, "initial user namespace required")
    @property
    def storage_identity(self):
        return strict_json(self._storage_identity)

    def _action(self):
        return self.journal.pending[1] if self.journal.pending else "AUTHORITY"

    def _check(self):
        require(not self.running and not self.failed, "provider forbidden while P1R parent runs or after failure")
        require_live(*self.context, self._action())

    def _gpu_check(self):
        self._check()
        require(self.exclusion and not self.released, "GPU management requires held exclusion")
        require(self._action() in ("EXCLUSION", "CONFINEMENT"), "NVML outside pre-run policy closure")

    def _mounts(self, pid="self"):
        return lb.parse_mountinfo(self.io.text(f"/proc/{pid}/mountinfo"))

    def _worker(self):
        require(os.getpid() == self.worker_pid, "provider moved to another worker")
        return self.io.identity(self.worker_pid)

    def _host(self):
        self._check()
        return observe_host_identity(self.io, self.host_id)

    def _device(self):
        identity = self.storage_identity
        info = self.io.stat(identity["device_path"])
        require(stat.S_ISBLK(info.st_mode) and major_minor(info, True) == identity["major_minor"],
                "exact granted block device")
        sysfs = os.path.realpath("/sys/dev/block/" + identity["major_minor"], strict=True)
        require(sysfs == identity["sysfs_identity"], "exact granted sysfs identity")
        mm = self.io.text(sysfs + "/dev").strip()
        require(mm == identity["major_minor"], "sysfs major/minor mismatch")
        require(self.io.listdir(sysfs + "/holders") == [] and self.io.listdir(sysfs + "/slaves") == [],
                "stacked block-device holders/slaves")
        entries = self.io.listdir(sysfs)
        require("partition" not in entries, "granted device is a partition")
        for name in entries:
            try:
                self.io.stat(sysfs + "/" + name + "/partition")
            except (FileNotFoundError, NotADirectoryError):
                continue
            raise InvalidEvidence("overlapping partitions")
        sequence = int(self.io.text(sysfs + "/diskseq").strip())
        require(sequence == identity["disk_sequence"], "exact granted disk sequence")
        link = self.io.stat("/dev/disk/by-uuid/" + identity["uuid"], True)
        require(stat.S_ISBLK(link.st_mode) and link.st_rdev == info.st_rdev, "ext4 UUID device link")
        aliases = observed_local_ssd_aliases(info.st_rdev, self.io.listdir, lambda p: self.io.stat(p, True))
        observed = {"device_path": identity["device_path"], "major_minor": major_minor(info, True),
                    "uuid": identity["uuid"], "disk_sequence": sequence,
                    "sysfs_identity": sysfs, "by_id_aliases": aliases}
        require(observed == identity, "observed storage differs from external grant")
        return observed

    def _health(self):
        require(self.backing is not None, "retained backing absent")
        device = self._device()
        require(self.backing["device_fd"] == self.native.retained_fds["device"]
                and self.backing["filesystem_fd"] == self.native.retained_fds["filesystem"], "retained descriptor drift")
        self.native._device(self.backing["device_fd"])
        value = self.io.superblock(self.backing["device_fd"])
        require(value["uuid"] == device["uuid"] and device == self.backing["identity"] == self.storage_identity, "same retained ext4 device/UUID")
        record = mount_at(self._mounts(), MOUNT)
        require(record["target"] == MOUNT and record["fstype"] == "ext4"
                and record["major_minor"] == device["major_minor"]
                and record["mount_id"] == self.backing["private_mount_id"], "private ext4 mount identity")
        return value, record

    def observe(self, topic):
        self._check()
        require(topic in TOPICS, "unknown live System observation topic")
        try:
            return copy.deepcopy(getattr(self, "_observe_" + topic)())
        except BaseException:
            self.failed = True
            raise

    def _observe_authority(self):
        self._host()
        root = Path(self.profile["source_root"])
        for path, expected in SOURCE_FILES.items():
            require(self.io.file_hash(str(root / "rust-engine/src" / path)) == expected, "frozen source/input drift")
        for path, expected in ((BIN, AUTHORITY["binary_sha256"]), (CONFIG, AUTHORITY["config_sha256"]),
                               (REQUEST, AUTHORITY["request_sha256"])):
            require(self.io.file_hash(path) == expected, "frozen binary/config/request drift")
        for group in ("loader", "libraries", "git", "driver_modules"):
            for path, pin in self.closure[group].items():
                require(self.io.file_hash(path) == pin, "execution dependency drift")
        for path, pin in self.profile["dependency_files"].items():
            require(self.io.file_hash(path) == pin, "provider direct dependency drift")
        import sys
        runtime = {"implementation": sys.implementation.name,
                   "version": list(sys.version_info[:3]), "cache_tag": sys.implementation.cache_tag,
                   "machine": os.uname().machine, "byteorder": sys.byteorder}
        require(runtime == self.provider["runtime"], "Python/runtime ABI mismatch")
        self._loaded_code()
        self._outputs_absent()
        # These are prospective approvals and validated inputs, never child observations.
        return {"ok": True, "authority": copy.deepcopy(AUTHORITY), "host_id": self.host_id,
                "tools": {n: self.io.file_hash(str(Path(__file__).parent / n)) for n in TOOLS},
                "inputs_verified": True, "outputs_absent": True, "marker_absent": True,
                "frozen_environment": True, "frozen_limits": True,
                "argv": list(parent_argv(self.attempt)), "execution_closure": self.closure,
                "adapter_manifest": self.manifest, "provider_manifest": self.provider,
                "provider_grant": self.context[2].public_scope(), "bootstrap": self.bootstrap}

    def _loaded_code(self):
        pins = dict(self.profile["dependency_files"])
        for group in ("loader", "libraries"):
            pins.update(self.closure[group])
        pins[self.provider["nvml"]["path"]] = self.provider["nvml"]["sha256"]
        for line in self.io.text(f"/proc/{self.worker_pid}/maps").splitlines():
            fields = line.split(None, 5)
            require(len(fields) >= 5, "malformed executable-map inventory")
            if "x" not in fields[1]:
                continue
            require(len(fields) == 6, "anonymous executable mapping")
            path = fields[5]
            if path in ("[vdso]", "[vsyscall]"):
                continue
            require(path in pins and self.io.file_hash(path) == pins[path], "unbound executable dependency")

    def _outputs_absent(self):
        root = Path(OUTPUT_ROOT) / self.attempt
        require(self.io.listdir(str(root)) == [], "runtime output attempt not empty")
        try:
            self.io.stat(str(self.journal.root / "marker.json"))
        except FileNotFoundError:
            pass
        else:
            raise InvalidEvidence("marker already exists")

    def acquire_exclusion(self):
        self._check()
        require(not self.exclusion and self.lock_fd is None, "exclusion reuse")
        import fcntl
        path = self.profile["lock_path"]
        controlled_directory(self.io, str(Path(path).parent))
        self.lock_fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        info = os.fstat(self.lock_fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o077,
                "uncontrolled exclusion lock")
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.exclusion = True
        self._inventory()
        require(self.child is None, "clock normalization must precede handoff child")
        self.nvml.normalize()

    def _inventory(self):
        self._check()
        require(self.exclusion and self.lock_fd is not None, "exclusion lock absent")
        raw_nodes = self._raw_device_access()
        namespaces, processes = {}, []
        for name in self.io.listdir("/proc"):
            if not name.isdecimal():
                continue
            pid = int(name)
            status = self.io.status(pid)
            identity = self.io.identity(pid)
            if status.get("Kthread") == "1":
                require(self.io.listdir(f"/proc/{pid}/fd") == []
                        and self.io.text(f"/proc/{pid}/maps") == "", "kernel task has userspace objects")
                processes.append(identity)
                continue
            ns = self.io.namespace(pid, "mnt")
            userns = self.io.namespace(pid, "user")
            records = self._mounts(pid)
            exposing = [r for r in records if r["major_minor"] == self.storage_identity["major_minor"]]
            if exposing:
                require(userns == self.initial_user_ns, "unexpected user namespace exposing protected data")
                if ns not in (self.original_ns, self.io.namespace(self.worker_pid, "mnt")):
                    executable = self.io.link(f"/proc/{pid}/exe")
                    allowed = self.profile["namespace_services"].get(executable)
                    require(type(allowed) is dict and set(allowed) == {"sha256", "cgroup"},
                            "unexpected mount namespace class")
                    require(self.io.file_hash(executable) == allowed["sha256"]
                            and self.io.text(f"/proc/{pid}/cgroup").strip() == allowed["cgroup"]
                            and all(int(v) == 0 for v in status["Uid"].split()), "namespace service identity drift")
                require(len(exposing) == 1 and exposing[0]["target"] == MOUNT
                        and exposing[0]["root"] == "/" and exposing[0]["fstype"] == "ext4",
                        "alternate protected-device mount alias")
                require(not any(r["target"].startswith(MOUNT + "/") for r in records), "protected submount escape")
            for fdname in self.io.descriptors(pid):
                require(fdname.isdecimal(), "unknown descriptor")
                path = f"/proc/{pid}/fd/{fdname}"
                info = self.io.stat(path, True)
                own = pid == self.worker_pid
                if (stat.S_ISBLK(info.st_mode) or stat.S_ISCHR(info.st_mode)) and (stat.S_IFMT(info.st_mode), info.st_rdev) in raw_nodes:
                    require(own and self.backing is not None and int(fdname) == self.backing["device_fd"],
                            "unexpected raw-device holder")
                if major_minor(info) == self.storage_identity["major_minor"]:
                    fields = dict(line.split(":", 1) for line in self.io.text(f"/proc/{pid}/fdinfo/{fdname}").splitlines()
                                  if ":" in line)
                    flags = int(fields["flags"].strip(), 8)
                    require(flags & os.O_ACCMODE == os.O_RDONLY, "unexpected protected-data writer")
            maps = self.io.text(f"/proc/{pid}/maps")
            for line in maps.splitlines():
                parts = line.split(None, 5)
                require(len(parts) >= 5, "malformed process maps")
                require(not (tuple(int(n, 16) for n in parts[3].split(":")) == tuple(int(n) for n in self.storage_identity["major_minor"].split(":")) and "w" in parts[1] and parts[1].endswith("s")),
                        "shared writable protected-data mapping")
            require(identity == self.io.identity(pid), "process churn during exclusion inventory")
            namespaces[ns] = userns
            processes.append(identity)
        return {"namespaces": namespaces, "processes": processes}

    def _raw_device_access(self):
        ids = self.baseline.values()["credentials"]
        def accessible(info):
            bits = (info.st_mode >> 6) if info.st_uid == ids["uid"] else (
                info.st_mode >> 3 if info.st_gid in {ids["gid"], *ids["groups"]} else info.st_mode)
            return bool(bits & 6)
        # Derive the controller route from the observed protected namespace.
        controller = str(Path(self._device()["sysfs_identity"]).parent)
        controller_node = self.io.stat("/dev/" + Path(controller).name)
        require(stat.S_ISCHR(controller_node.st_mode), "NVMe controller character identity")
        nodes = {(stat.S_IFCHR, controller_node.st_rdev)}
        observed = []
        def walk_error(error):
            raise error
        for root, dirs, files in os.walk("/dev", followlinks=False, onerror=walk_error):
            self._check()
            dirs[:] = [d for d in dirs if not stat.S_ISLNK(self.io.stat(str(Path(root) / d)).st_mode)]
            for name in files:
                path = str(Path(root) / name)
                info = self.io.stat(path)
                if not (stat.S_ISBLK(info.st_mode) or stat.S_ISCHR(info.st_mode)):
                    continue
                observed.append((path, info))
                if stat.S_ISBLK(info.st_mode):
                    nodes.add((stat.S_IFBLK, info.st_rdev))
                else:
                    sysfs = os.path.realpath("/sys/dev/char/" + major_minor(info, True), strict=True)
                    if sysfs == controller or sysfs.startswith(controller + "/") or name.startswith(("nvme", "ng", "sg")):
                        nodes.add((stat.S_IFCHR, info.st_rdev))
        # The second pass includes aliases with arbitrary names and identical rdev.
        for path, info in observed:
            if (stat.S_IFMT(info.st_mode), info.st_rdev) in nodes:
                self._no_acl(path)
                require(not accessible(info), "runtime raw/passthrough device exposure")
        # Conservative broker admission: writable Unix control sockets are rejected.
        for line in self.io.text("/proc/net/unix").splitlines()[1:]:
            fields = line.split()
            require(len(fields) in (7, 8), "malformed Unix socket inventory")
            if len(fields) == 8:
                path = fields[7]
                require(not path.startswith("@"), "abstract control broker cannot be excluded")
                if path.startswith("/"):
                    info = self.io.stat(path)
                    self._no_acl(path)
                    require(not accessible(info), "runtime privileged/control broker exposure")
        return nodes

    def _no_acl(self, path):
        self._check()
        import errno
        try:
            os.getxattr(path, "system.posix_acl_access", follow_symlinks=False)
        except OSError as exc:
            require(exc.errno in (errno.ENODATA, errno.ENOTSUP), "ACL observation failed")
        else:
            raise InvalidEvidence("unreviewed ACL access route")

    def _observe_exclusion(self):
        inventory = self._inventory()
        return {"ok": True, "held": self.exclusion, "trusted_host": True,
                "uncontrolled_writers": [], "raw_holders": [], "uncontrolled_mount_namespaces": [],
                "inventory": inventory, "administrative_noninterference": "TRUSTED_HOST_PREMISE"}

    def _observe_topology(self):
        boot, device = self._host(), self._device()
        mounts = self._mounts()
        data = mount_at(mounts, MOUNT)
        output = mount_at(mounts, OUTPUT_ROOT)
        require(data["root"] == "/" and data["target"] == MOUNT and data["major_minor"] == self.storage_identity["major_minor"] and data["fstype"] == "ext4"
                and not data["mount_ro"] and not data["superblock_ro"], "original ext4 RW mount")
        require(output["major_minor"] == "259:1" and not output["mount_ro"] and not output["superblock_ro"], "output backing")
        require(self.io.link(CONFIGURED) == CANONICAL, "configured alias target")
        require(self.backing is None, "retained topology reuse")
        refs = self.native.retain_objects()
        flags, mount_data = mount_recipe(data)
        alias_flags, _ = mount_recipe(mount_at(mounts, ALIAS_PARENT))
        self.backing = {"device_fd": refs["device"], "filesystem_fd": refs["filesystem"],
                        "identity": device, "host_mount_id": data["mount_id"], "private_mount_id": data["mount_id"],
                        "mount_flags": flags, "mount_data": mount_data, "alias_flags": alias_flags}
        self.backing_delivered = False
        self.topology = {"ok": True, "boot_id": boot, "disk_sequence": device["disk_sequence"],
                         "mount_id": data["mount_id"], "original_mount_options": data["mount_options"],
                         "output_major_minor": output["major_minor"], "device_identity": device}
        self.topology.update(self._topology_expectations(data, device))
        return self.topology.copy()

    def _topology_expectations(self, record, device):
        # Every expected constant below is checked against a direct observation.
        require(self.nvml.certificate is not None, "GPU identity transition required")
        gpu = strict_json(self.nvml.certificate)
        observed = {"kernel": os.uname().release, "data_mount": record["target"],
                    "data_fstype": record["fstype"],
                    "configured_data_path": CONFIGURED, "canonical_data_path": self.io.link(CONFIGURED),
                    "configured_path_alias": "symlink_to_canonical_data_path",
                    "root_output_major_minor": major_minor(self.io.stat(OUTPUT_ROOT)),
                    "expert_size": 2658304, "block_align": 4096, "no_direct": False,
                    "packed_blob": None, "packed_manifest": None,
                    "gpu": gpu["device_name"], "driver": gpu["driver_version"]}
        # Storage layout values are source/config closure claims; verify the pinned
        # TOML explicitly, and validate file sizes/O_DIRECT at METADATA.
        import tomllib
        config = tomllib.loads(self.io.text(CONFIG))
        storage = config["storage"]
        require(config["model"]["expert_size"] == observed["expert_size"] and storage["block_align"] == 4096
                and storage["no_direct"] is False and storage.get("packed_blob") is None
                and storage.get("packed_manifest") is None, "pinned storage configuration mismatch")
        require(observed == EXPECTATIONS, "observed host topology mismatch")
        # Actual ext4 health is populated from the retained descriptor below.
        return {"expectations": observed, "filesystem_rw": True, "device_rw": self.io.text(device["sysfs_identity"] + "/ro").strip() == "0",
                "healthy": self._preliminary_health(device), "external_journal": False,
                "stacked_or_overlapping_devices": [], "uncontrolled_aliases": [], "writes_on_data_backing": []}

    def _preliminary_health(self, device):
        value = self.io.superblock(self.backing["device_fd"])
        require(device == self.storage_identity and value["uuid"] == device["uuid"] and value["recovery_required"], "original ext4 active journal")
        return value["healthy"]

    def retain_backing(self):
        self._check()
        require(self.backing is not None and not self.backing_delivered, "one retained backing")
        self.backing_delivered = True
        return copy.deepcopy(self.backing)

    def _observe_roles(self):
        worker = self._worker()
        require(worker["threads"] == 1 and os.geteuid() == 0, "privileged single-threaded worker")
        require(self.io.namespace(self.guardian_pid, "mnt") == self.original_ns
                and self.io.namespace(self.worker_pid, "user") == self.initial_user_ns, "guardian/worker namespaces")
        return {"guardian_original_namespace": True, "worker_single_threaded": True,
                "initial_user_namespace": True, "worker_privileged": True}

    def unshare(self, flags):
        self._check()
        self.native.unshare(flags)
        self.native.pin_private_namespace()

    def mount(self, source, target, fstype, flags, data):
        self._check()
        self.native.mount(source, target, fstype, flags, data)
        if self._action() == "FILESYSTEM_RO":
            self.ro_transition = True
        if self._action() == "FILESYSTEM_RW":
            self.rw_transition = True

    def _observe_namespace(self):
        records = self._mounts()
        require(all(not r["propagation"] for r in records), "mount propagation not recursively private")
        ns = self.io.namespace(self.worker_pid, "mnt")
        mount = mount_at(records, MOUNT)
        require(ns != self.original_ns and mount["mount_id"] != self.backing["host_mount_id"], "private namespace/mount identity")
        self.backing["private_mount_id"] = mount["mount_id"]
        self.namespace_receipt = {"ok": True, "original_namespace_id": self.original_ns, "namespace_id": ns,
                                  "original_mount_id": self.backing["host_mount_id"], "private_mount_id": mount["mount_id"],
                                  "private_recursive": True, "initial_user_namespace": self.io.namespace(self.worker_pid, "user") == self.initial_user_ns,
                                  "references_pinned": "private_mount_ns" in self.native.retained_fds}
        return self.namespace_receipt

    def _observe_alias_bind(self):
        record = mount_at(self._mounts(), ALIAS_PARENT)
        require(record["target"] == ALIAS_PARENT and not record["propagation"], "private alias bind absent")
        return {"ok": True, "path": record["target"], "private_bind": True}

    def _observe_alias_ro(self):
        records = self._mounts()
        alias = mount_at(records, ALIAS_PARENT)
        output = mount_at(records, OUTPUT_ROOT)
        flags, _ = mount_recipe(alias)
        require(alias["target"] == ALIAS_PARENT and alias["mount_ro"]
                and flags == self.backing["alias_flags"], "alias RO/original flags")
        require(self.io.link(CONFIGURED) == CANONICAL and not output["mount_ro"]
                and not output["superblock_ro"] and output["major_minor"] == "259:1", "alias/output drift")
        require(not any(r["target"].startswith(ALIAS_PARENT + "/") for r in records), "alias submount")
        return {"ok": True, "canonical_target": CANONICAL, "output_still_rw": True,
                "preserved_flags": flags, "alias_parent_ro": True, "configured": CONFIGURED,
                "symlink_target": self.io.link(CONFIGURED), "runtime_cannot_replace": True}

    def _observe_metadata(self):
        self._check()
        import array
        import fcntl
        from dataclasses import asdict
        ids = self.baseline.values()["credentials"]
        old_uid, old_gid, old_groups = os.geteuid(), os.getegid(), os.getgroups()
        require(old_uid == 0, "metadata credential switch requires privileged worker")
        records = []
        try:
            os.setgroups(ids["groups"])
            os.setegid(ids["gid"])
            os.seteuid(ids["uid"])
            for global_id in STARTUP_IDS + IDS:
                primary, fallback = alternatives(global_id)
                try:
                    info = os.lstat(primary)
                    selected = primary
                except FileNotFoundError:
                    selected = fallback
                    info = os.lstat(selected)
                identity = Identity.from_stat(info)
                identity.check(self.storage_identity)
                canonical = os.path.realpath(selected, strict=True)
                require(canonical == CANONICAL + "/" + selected.rsplit("/", 1)[1], "metadata path escape")
                for root in (str(Path(CONFIGURED).parent), CANONICAL):
                    for ancestor in (Path(root), *Path(root).parents):
                        require(stat.S_ISDIR(os.lstat(ancestor).st_mode), "metadata symlink ancestor")
                fd = os.open(selected, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC | os.O_NOFOLLOW)
                try:
                    require(Identity.from_stat(os.fstat(fd)) == identity, "metadata descriptor/path mismatch")
                    require(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT, "O_DIRECT not retained")
                    flags = array.array("L", [0])
                    fcntl.ioctl(fd, 0x80086601, flags, True)  # FS_IOC_GETFLAGS, x86-64.
                    require(flags[0] & (0x4 | 0x800 | 0x100000 | 0x10000000) == 0,
                            "compressed/encrypted/verity/inline expert feature")
                    alignment = self.io.dio_alignment(fd)
                    require(identity.file_length % 4096 == 0, "expert DIO length alignment")
                finally:
                    os.close(fd)
                records.append({"global_expert_id": global_id, "selected": selected, "canonical": canonical,
                                "identity": asdict(identity), "dio_eligible": True, "o_direct_retained": True,
                                "payload_bytes_read": 0, "redirects": [], "disallowed_features": [],
                                "dio_alignment": alignment})
        finally:
            os.seteuid(old_uid)
            os.setegid(old_gid)
            os.setgroups(old_groups)
        return {"ok": True, "records": records}

    def syncfs(self, fd):
        self._check()
        require(self.backing is not None and fd == self.backing["filesystem_fd"], "retained filesystem sync")
        self.native.syncfs(fd)
        self.sync_done = True

    def _observe_sync_health(self):
        require(self.sync_done and not self.ro_transition, "PRE_RO_SYNCED_ACTIVE phase")
        value, record = self._health()
        require(value["recovery_required"] and not record["mount_ro"] and not record["superblock_ro"], "pre-RO active journal")
        self.pre_superblock = canonical_json({"device": self.backing["identity"], "uuid": value["uuid"],
                                             "mount": record["mount_id"], "recovery_required": True})
        return {"syncfs_ok": self.sync_done, "healthy": value["healthy"],
                "external_journal": value["external_journal"], "recovery_required": value["recovery_required"]}

    def _ro_health(self):
        value, record = self._health()
        require(self.pre_superblock is not None and self.sync_done and self.ro_transition and not self.rw_transition,
                "RO transition certificate absent")
        before = strict_json(self.pre_superblock)
        require(before == {"device": self.backing["identity"], "uuid": value["uuid"],
                           "mount": record["mount_id"], "recovery_required": True}, "ext4 transition identity drift")
        require(not value["recovery_required"] and record["mount_ro"] and record["superblock_ro"],
                "RO_QUIESCED_CLEAN recovery-bit certificate")
        return {"ok": True, "device_identity": self.backing["identity"], "private_mount_id": record["mount_id"],
                "healthy": value["healthy"], "external_journal": value["external_journal"],
                "recovery_required": value["recovery_required"], "journal_clean": True,
                "mount_ro": record["mount_ro"], "superblock_ro": record["superblock_ro"]}

    def _observe_filesystem_ro(self):
        result = self._ro_health()
        return result | {"syncfs_checked": self.sync_done, "normal_unforced_transition": self.ro_transition,
                         "journal_writeback_complete": True, "device_still_rw": self.block_get(self.backing["device_fd"]) == 0}

    def block_ro(self, fd, read_only):
        self._check()
        require(fd == self.backing["device_fd"], "retained block mutation only")
        self.native.block_ro(fd, read_only)

    def block_get(self, fd):
        self._check()
        require(fd == self.backing["device_fd"], "retained block observation only")
        return self.native.block_get(fd)

    def _observe_device_ro(self):
        return self._ro_health() | {"device_ro": self.block_get(self.backing["device_fd"]) == 1, "filesystem_still_ro": True}

    def _observe_protected(self):
        result = self._ro_health()
        self._inventory()
        alias = self._observe_alias_ro()
        boot = self._host()
        ns = self.io.namespace(self.worker_pid, "mnt")
        require(ns == self.namespace_receipt["namespace_id"], "protected namespace drift")
        return result | {"filesystem_ro": True, "device_ro": self.block_get(self.backing["device_fd"]) == 1,
                         "alias_locked": alias["alias_parent_ro"], "exclusion_held": self.exclusion,
                         "namespace_pinned": "private_mount_ns" in self.native.retained_fds,
                         "no_uncontrolled_aliases": True, "expectations": self.topology["expectations"],
                         "namespace_id": ns, "boot_id": boot, "disk_sequence": self.backing["identity"]["disk_sequence"],
                         "mount_id": self.backing["host_mount_id"]}

    def _cgroup_path(self, pid):
        text = self.io.text(f"/proc/{pid}/cgroup").strip()
        require(text.startswith("0::/") and "\n" not in text, "unified cgroup2/default hierarchy required")
        path = "/sys/fs/cgroup" + text[3:].rstrip("/")
        record = mount_at(self._mounts(pid), path)
        require(record["fstype"] == "cgroup2" and record["target"] == "/sys/fs/cgroup"
                and record["root"] == "/", "cgroup2 mount root/hierarchy")
        return path

    def _domain(self, path):
        path_bytes(path)
        require(path == "/sys/fs/cgroup" or path.startswith("/sys/fs/cgroup/"), "cgroup domain path")
        prefixes = ("cpu", "memory", "io", "cpuset", "pids", "hugetlb", "rdma", "misc")
        knobs = {"cpu": ("cpu.weight", "cpu.weight.nice", "cpu.max", "cpu.max.burst", "cpu.idle", "cpu.uclamp.min", "cpu.uclamp.max"),
                 "memory": ("memory.min", "memory.low", "memory.high", "memory.max", "memory.oom.group", "memory.swap.high", "memory.swap.max", "memory.zswap.max", "memory.zswap.writeback"),
                 "io": ("io.weight", "io.max", "io.latency", "io.cost.qos", "io.cost.model", "io.prio.class"),
                 "cpuset": ("cpuset.cpus", "cpuset.mems", "cpuset.cpus.partition", "cpuset.cpus.exclusive", "cpuset.cpus.exclusive.effective"),
                 "pids": ("pids.max",), "rdma": ("rdma.max",), "misc": ("misc.max",)}
        domains = {name: {"domain": "cgroup2-root-disabled", "constraints": {}} for name in prefixes}
        ancestors, bpf, effective = [], [], {}
        parents = []
        current = Path(path)
        while str(current).startswith("/sys/fs/cgroup"):
            parents.append(str(current))
            if str(current) == "/sys/fs/cgroup":
                break
            current = current.parent
        for parent in reversed(parents):
            controlled_directory(self.io, parent)
            files = self.io.listdir(parent)
            values = {}
            for controller in prefixes:
                names = tuple(n for n in files if n.startswith("hugetlb.") and n.endswith((".max", ".rsvd.max"))) if controller == "hugetlb" else knobs[controller]
                found = {name: self.io.text(parent + "/" + name).strip() for name in names if name in files}
                if found:
                    domains[controller] = {"domain": parent, "constraints": found}
                    values[controller] = found
            for name in ("cpus", "mems"):
                file = "cpuset." + name + ".effective"
                if file in files:
                    effective[name] = integer_list(self.io.text(parent + "/" + file))
            attached = self.io.bpf_ids(parent)
            if attached:
                bpf.append({"path": parent, "programs": attached})
            if values:
                ancestors.append({"path": parent, "constraints": values})
        require(set(effective) == {"cpus", "mems"}, "effective cgroup CPUs/mems unavailable")
        result = {**domains, "gpu": {"domain": self.provider["nvml"]["uuid"],
                                     "constraints": copy.deepcopy(self.gpu_management)},
                  "effective_cpus": effective["cpus"], "effective_mems": effective["mems"],
                  "ancestor_constraints": ancestors, "bpf_ancestry": bpf}
        return lb.resource_domain(result)

    def _observe_cgroup_preflight(self):
        require(self.snapshot is None and self.child is None and self.exclusion, "one NUMA pre-fork snapshot under exclusion")
        self._host()
        self.gpu_management = self.nvml.management()
        source = self._cgroup_path(self.worker_pid)
        require(source == self.profile["cgroup_parent"], "worker/source accounting parent mismatch")
        require(self.io.text(source + "/cgroup.type").strip() == "domain", "non-domain cgroup")
        require(self.io.text(source + "/cgroup.subtree_control").strip() == "", "new resource domain/controller mutation required")
        controllers = self.io.text(source + "/cgroup.controllers").split()
        require(set(controllers) <= set(lb.CONTROLLERS), "unknown resource controller")
        require(self.attempt not in self.io.listdir(source), "accounting cgroup already exists")
        domain = self._domain(source)
        worker = self._worker()
        status = self.io.status(self.worker_pid)
        policy = self.io.task_policy()
        allowed = integer_list(status["Mems_allowed_list"], 1024)
        # No enabled subtree controllers means the prospective child inherits
        # the source's effective controller CSS and adds no resource constraint.
        self.snapshot = InheritanceSnapshot.capture(attempt=self.attempt, host_id=self.host_id,
            worker=worker, kernel=NUMA_ABI, policy=policy, allowed=allowed, source=domain,
            target=copy.deepcopy(domain), hierarchy="cgroup2", exclusion=self.exclusion,
            baseline=self.baseline.values())
        self.source_cgroup = source
        return {"baseline": domain, "prospective": copy.deepcopy(domain), "nondelegated": True,
                "controllers_to_enable": [], "settings_to_write": [], "bpf_to_attach": []}

    def create_accounting_cgroup(self, attempt):
        self._check()
        require(attempt == self.attempt and self.snapshot is not None and self.cgroup is None, "admitted cgroup attempt")
        self.cgroup = self.processes.create_accounting_cgroup(self.source_cgroup, attempt)
        require(self._domain(self.cgroup) == strict_json(self.snapshot.encoded)["target"], "actual passive cgroup domain drift")
        return self.cgroup

    def prepare_stopped_child(self, argv, manifest, cgroup):
        self._check()
        require(self.snapshot is not None and not self.snapshot_consumed and self.child is None
                and cgroup == self.cgroup and manifest == self.baseline.encoded, "one approved inheritance handoff")
        snapshot = strict_json(self.snapshot.encoded)
        require(self._worker() == snapshot["worker"] and self._domain(self.source_cgroup) == snapshot["source"]
                and self._domain(cgroup) == snapshot["target"], "pre-fork worker/no-rebind drift")
        require(self.io.task_policy() == snapshot["policy"], "pre-fork task-policy drift")
        self.child = self.roles.prepare_stopped_child(argv, manifest, cgroup)
        self.child_identity = self.io.identity(self.child)
        return self.child

    def _policy_files(self, group):
        paths = self.profile["policy_files"][group]
        require(type(paths) is dict and paths, "explicit policy observation files")
        values = {}
        for name, files in paths.items():
            require(type(files) is list and files, "policy file inventory required")
            values[name] = {path: self.io.text(path).strip() for path in files}
        if group == "cpu_policy":
            root = "/sys/devices/system/cpu/cpufreq"
            policies = [name for name in self.io.listdir(root) if name.startswith("policy")]
            require(policies, "CPU frequency/governor interface unavailable")
            expected = [root + "/" + name + "/scaling_governor" for name in policies]
            require(sorted(values["governors"]) == sorted(expected), "incomplete CPU governor coverage")
            limits = [root + "/" + name + "/" + field for name in policies
                      for field in ("scaling_min_freq", "scaling_max_freq", "cpuinfo_min_freq", "cpuinfo_max_freq")]
            require(sorted(values["frequency_limits"]) == sorted(limits), "incomplete CPU frequency limit coverage")
        return values

    def _gpu_inventory(self):
        result = {}
        roots = self.profile["gpu_inventory_roots"]
        lb.exact_keys(roots, ("nodes", "libraries", "icds", "layers", "cache"), "GPU filesystem closure")
        for group, paths in roots.items():
            require(type(paths) is list and paths, "GPU inventory roots required")
            entries = {}
            for path in paths:
                path_bytes(path)
                info = self.io.stat(path)
                if stat.S_ISREG(info.st_mode):
                    entries[path] = {"sha256": self.io.file_hash(path)}
                elif stat.S_ISDIR(info.st_mode):
                    require(major_minor(info) != self.storage_identity["major_minor"], "GPU inventory cannot read expert backing")
                    contents = {}
                    pending = [path]
                    while pending:
                        directory = pending.pop()
                        names = self.io.listdir(directory)
                        require(len(names) + len(contents) < 4096, "bounded GPU inventory exceeded")
                        for name in names:
                            child = str(Path(directory) / name)
                            obj = self.io.stat(child)
                            if stat.S_ISREG(obj.st_mode):
                                contents[child] = {"sha256": self.io.file_hash(child)}
                            elif stat.S_ISDIR(obj.st_mode):
                                pending.append(child)
                                contents[child] = {"directory": True, "mode": stat.S_IMODE(obj.st_mode)}
                            else:
                                raise InvalidEvidence("GPU directory contains unreviewed symlink/object")
                    entries[path] = {"owner": info.st_uid, "group": info.st_gid, "mode": stat.S_IMODE(info.st_mode),
                                     "device": major_minor(info), "contents": contents}
                elif stat.S_ISCHR(info.st_mode):
                    entries[path] = {"rdev": major_minor(info, True), "owner": info.st_uid,
                                     "group": info.st_gid, "mode": stat.S_IMODE(info.st_mode)}
                else:
                    raise InvalidEvidence("unreviewed GPU inventory object")
            result[group] = entries
        return result

    def _launch_observed(self, pid, numa, status, domain):
        import resource
        import signal
        import sys
        environ = self.io.read(f"/proc/{pid}/environ")
        require(not environ or environ.endswith(b"\0"), "truncated process environment")
        values = {}
        for entry in environ.rstrip(b"\0").split(b"\0") if environ else []:
            key, value = entry.decode("utf-8").split("=", 1)
            require(key not in values, "duplicate actual environment")
            values[key] = value
        absent = self.baseline.values()["environment"]["absent"]
        require(not set(absent) & set(values), "forbidden environment present")
        limits = {}
        for name in lb.RLIMITS:
            pair = resource.prlimit(pid, lb.linux_rlimit_id(resource, name, sys.platform))
            limits[name] = ["INFINITY" if n == resource.RLIM_INFINITY else n for n in pair]
        fds = self.io.listdir(f"/proc/{pid}/fd")
        require(fds == ["0", "1", "2"], "actual inherited descriptors must be 0/1/2")
        stdio = []
        for fd in range(3):
            path = f"/proc/{pid}/fd/{fd}"
            info = self.io.stat(path, True)
            fields = dict(line.split(":", 1) for line in self.io.text(f"/proc/{pid}/fdinfo/{fd}").splitlines() if ":" in line)
            require(stat.S_ISREG(info.st_mode) and major_minor(info) == "259:1", "stdio regular output backing required")
            stdio.append({"fd": fd, "type": "regular", "backing": {"device": major_minor(info), "inode": info.st_ino,
                          "path": self.io.link(path)}, "flags": int(fields["flags"], 8)})
        scheduler, priority = self.io.scheduler(pid)
        pstat = parse_stat(self.io.text(f"/proc/{pid}/stat"))
        blocked, ignored, caught = (int(status[n], 16) for n in ("SigBlk", "SigIgn", "SigCgt"))
        require(caught == 0, "unreviewed child signal handler")
        valid = {int(n) for n in signal.valid_signals()} - {int(signal.SIGKILL), int(signal.SIGSTOP)}
        dispositions = {str(n): "IGNORE" if ignored & (1 << (n - 1)) else "DEFAULT" for n in sorted(valid)}
        thp, balancing = self._policy_files("thp"), self._policy_files("numa_balancing")
        require(status.get("THP_enabled") in ("0", "1"), "actual child THP policy unavailable")
        thp = {"global": thp, "task_enabled": int(status["THP_enabled"])}
        gpu = self._gpu_inventory() | self.nvml.management()
        certificate = self.nvml.observed()
        gpu.update(device={k: certificate[k] for k in ("device_uuid", "pci_bus_id", "device_name")},
                   driver=certificate["driver_version"], clocks=certificate,
                   mps=self._mps_observation())
        lsm = self.io.text("/sys/kernel/security/lsm").strip().split(",")
        label = self.io.text(f"/proc/{pid}/attr/current").strip()
        security = {"lsm": lsm, "labels": {"current": label},
                    "inherited_policy_sha256": digest(canonical_json({"lsm": lsm, "label": label,
                      "seccomp": int(status["Seccomp"]), "no_new_privs": int(status["NoNewPrivs"])}))}
        return {"environment": {"complete": True, "values": values, "absent": absent}, "rlimits": limits,
                "umask": int(status["Umask"], 8), "stdio": stdio, "scheduler": scheduler,
                "nice": os.getpriority(os.PRIO_PROCESS, pid), "cpu_affinity": sorted(os.sched_getaffinity(pid)),
                "io_priority": priority, "cpu_policy": self._policy_files("cpu_policy"), "numa": numa,
                "thp": {"policy": thp, "evidence_sha256": digest(canonical_json(thp))},
                "numa_balancing": {"policy": balancing, "evidence_sha256": digest(canonical_json(balancing))},
                "resource_domain": domain, "gpu": gpu, "cwd": self.io.link(f"/proc/{pid}/cwd"),
                "signals": {"mask": [n for n in range(1, 65) if blocked & (1 << (n - 1))], "dispositions": dispositions},
                "credentials": {"uid": int(status["Uid"].split()[0]), "gid": int(status["Gid"].split()[0]),
                                "groups": sorted(int(v) for v in status["Groups"].split())},
                "adapter_pins": self._dependency_observation(),
                "session": {"session_id": pstat["session"], "process_group": pstat["pgrp"],
                            "autogroup": self.io.text(f"/proc/{pid}/autogroup").strip()}, "security_context": security}

    def _dependency_observation(self):
        self._loaded_code()
        values = {}
        for name, pin in self.manifest["dependencies"].items():
            matches = [path for path, expected in self.profile["dependency_files"].items() if expected == pin]
            require(matches, "dependency pin has no observed file binding: " + name)
            for path in matches:
                require(self.io.file_hash(path) == pin, "dependency bytes drift")
            values[name] = pin
        return values

    def _mps_observation(self):
        forbidden = []
        for name in self.io.listdir("/proc"):
            if name.isdecimal():
                comm = self.io.text(f"/proc/{name}/comm").strip()
                if comm.startswith("nvidia-cuda-mps"):
                    forbidden.append(int(name))
        require(not forbidden, "MPS broker present")
        return {"processes": forbidden, "status": "NO_MPS_PROCESS_OBSERVED"}

    def _observe_confinement(self):
        require(self.child is not None and self.snapshot is not None and not self.snapshot_consumed, "one exact child inheritance receipt")
        self._host()
        worker = self._worker()
        require(self.io.identity(self.child) == self.child_identity, "returned child identity drift")
        status = self.io.status(self.child)
        require(status["State"].split()[0] == "T" and int(status["PPid"]) == self.worker_pid,
                "handoff must be exact stopped child")
        require(self._cgroup_path(self.child) == self.cgroup, "exact child cgroup binding")
        require(self.io.namespace(self.child, "mnt") == self.namespace_receipt["namespace_id"]
                and self.io.namespace(self.child, "user") == self.initial_user_ns, "child namespace binding")
        require(self.io.text(self.source_cgroup + "/cgroup.subtree_control").strip() == ""
                and self.io.text(self.cgroup + "/cgroup.subtree_control").strip() == "", "controller mutation after snapshot")
        source, target = self._domain(self.source_cgroup), self._domain(self.cgroup)
        child = {"pid": int(status["Pid"]), "ppid": int(status["PPid"]), "stopped": True,
                 "threads": int(status["Threads"]), "allowed_cpus": integer_list(status["Cpus_allowed_list"]),
                 "allowed_mems": integer_list(status["Mems_allowed_list"], 1024)}
        numa = self.snapshot.bind_child(attempt=self.attempt, host_id=self.host_id, worker=worker,
            returned_pid=self.child, child=child, source=source, target=target, baseline=self.baseline.values())
        self.snapshot_consumed = True
        self.numa_receipt = numa
        observed = self._launch_observed(self.child, numa, status, target)
        self.baseline.compare(observed)
        self._inventory()
        ids = observed["credentials"]
        uids, gids = ([int(v) for v in status[n].split()] for n in ("Uid", "Gid"))
        require(uids == [ids["uid"]] * 4 and gids == [ids["gid"]] * 4, "all child credential slots")
        result = {"ok": True, "uids": uids[:3], "gids": gids[:3], "groups": ids["groups"],
                  "capabilities": {name: int(status[field], 16) for name, field in (
                    ("effective", "CapEff"), ("permitted", "CapPrm"), ("inheritable", "CapInh"),
                    ("ambient", "CapAmb"), ("bounding", "CapBnd"))},
                  "no_new_privs": status["NoNewPrivs"] == "1", "seccomp_mode": int(status["Seccomp"]),
                  "groups_exact": True, "brokers_excluded": True, "setup_fds_closed": True,
                  "initial_user_namespace": True, "private_mount_namespace": True,
                  "trusted_execution_closure_verified": True, "cgroup_nondelegated": True,
                  "resource_domain_equal": source == target, "launch_profile_equal": True,
                  "raw_device_access_denied": True, "handoff_stopped": True,
                  "raw_or_privileged_fds": [], "writable_alias_routes": [], "cgroup_controllers_changed": [],
                  "cgroup_limits_changed": [], "cgroup_bpf_added": [], "p1v_device_bpf_attached": False,
                  "inherited_fds": [0, 1, 2], "cgroup_id": self.cgroup, "cgroup_mode": "PASSIVE_TREE_ONLY",
                  "cgroup_active_monitoring_during_pair": False, "launch_observed": observed,
                  "resource_domain_observed": target, "namespace_id": self.io.namespace(self.child, "mnt"),
                  "handoff_pid": self.child}
        lb.validate_confinement(result, self.baseline.values(), self.namespace_receipt)
        return result

    def resume_wait_parent(self, pid):
        self._check()
        require(pid == self.child and self.snapshot_consumed and not self.returned, "one certified parent invocation")
        self.running = True
        # All provider checks precede the protected release-and-one-wait call.
        # On uncertainty running remains true; no observation/restoration follows.
        code = self.roles.resume_wait_parent(pid)
        self.running = False
        self.returned = True
        return code

    def raw_reports(self, attempt):
        self._check()
        require(self.returned and attempt == self.attempt, "report bytes only after exact parent return")
        root = Path(OUTPUT_ROOT) / attempt
        for part in (root, *root.parents):
            require(stat.S_ISDIR(self.io.stat(str(part)).st_mode), "report symlink ancestor")
        directory = self.io.stat(str(root))
        require({"device": major_minor(directory), "inode": directory.st_ino,
                 "uid": directory.st_uid, "gid": directory.st_gid,
                 "mode": stat.S_IMODE(directory.st_mode)} == self.bootstrap["runtime_directory_identity"],
                "report attempt directory replaced")
        result = {}
        for key, name in REPORT_NAMES.items():
            path = root / name
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                        and major_minor(before) == "259:1" and before.st_uid == self.baseline.values()["credentials"]["uid"],
                        "report object ownership/type/backing")
                require(0 < before.st_size <= 64 * 1024 * 1024, "bounded raw report size")
                parts, remaining = [], before.st_size
                while remaining:
                    data = os.read(fd, min(1048576, remaining))
                    require(data, "truncated raw report")
                    parts.append(data)
                    remaining -= len(data)
                require(stable_stat(os.fstat(fd)) == stable_stat(before) == stable_stat(os.lstat(path)), "raw report object drift")
                result[key] = b"".join(parts)
            finally:
                os.close(fd)
        return result

    def _observe_shutdown(self):
        require(self.returned and not self.running, "shutdown only after parent reap")
        events = dict(line.split() for line in self.io.text(self.cgroup + "/cgroup.events").splitlines())
        require(events["populated"] == "0" and self.io.text(self.cgroup + "/cgroup.procs").strip() == "",
                "recursive runtime tree remains populated")
        children = [n for n in self.io.listdir(self.cgroup)
                    if stat.S_ISDIR(self.io.stat(self.cgroup + "/" + n).st_mode)]
        require(children == [], "unexpected child cgroups")
        return {"ok": True, "recursive_populated": 0, "unexpected_child_groups": [], "parent_reaped": True,
                "runtime_tree_empty": True, "uncertain_descendants": [], "cgroup_id": self.cgroup}

    def _observe_device_rw(self):
        self._observe_shutdown()
        return self._ro_health() | {"device_rw": self.block_get(self.backing["device_fd"]) == 0,
                                   "filesystem_still_ro": True, "runtime_tree_empty": True, "hasher_stopped": True}

    def _observe_filesystem_rw(self):
        require(self.rw_transition, "RW restoration transition absent")
        value, record = self._health()
        require(value["recovery_required"] and not record["mount_ro"] and not record["superblock_ro"],
                "RW_RESTORED_ACTIVE state required")
        return {"ok": True, "device_identity": self.backing["identity"], "private_mount_id": record["mount_id"],
                "healthy": value["healthy"], "external_journal": value["external_journal"],
                "recovery_required": value["recovery_required"], "filesystem_rw": True,
                "device_rw": self.block_get(self.backing["device_fd"]) == 0, "mount_options": record["mount_options"]}

    def release(self):
        self._check()
        require(self.exclusion and self.rw_transition and not self.released, "release lifecycle")
        for fd in self.native.retained_fds.values():
            os.close(fd)
        self.native.retained_fds.clear()
        os.close(self.lock_fd)
        self.lock_fd = None
        self.exclusion, self.released = False, True

    def _observe_release(self):
        require(self.released and self.lock_fd is None and not self.native.retained_fds, "release incomplete")
        return {"ok": True, "released": True}
