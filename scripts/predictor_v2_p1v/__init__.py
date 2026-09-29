"""P1V2 software-only qualification tools; #214 grants no execution authority.

Only synthetic backends are runnable here. A future reviewed host adapter must
implement the supervisor's receipt contract and supply separate authorization.
Raw P1R reports and logical generation checks are never changed.
"""

import hashlib
import json
import os
import re
import stat
from decimal import Decimal
from pathlib import Path

BASE = "ed60f78256d35fdecd3c18d9b3b75ee11403d9e8"
TREE = "e26fd53ed8878e451c68092969deaca6ce338972"
PARENT = "142436798cecc22e5a6ea7dce46c3aa59c5d20e1"
BRANCH = "diag/predictor-v2-p1v2-content-bound-supervisor"
AUTHORITY = {
    "issue": 214,
    "commit": BASE,
    "tree": TREE,
    "parent": PARENT,
    "binary_sha256": "88ad438272f345dbc496d4935481cfd9ff9198a250ea9701e5887202c2b16c83",
    "config_sha256": "33d7cf96328d9c68b0ff45448d91d597d2e3a757cb99e6e61c72998ceabdd056",
    "request_sha256": "e850fd8acfbf4265beadbd891babc0d48da2d187f0ec52b9150e173c496581d7",
    "profile": "GCP-L4-580.178.04",
}
CANONICAL = "/mnt/localssd/data/qwen3-coder-q4"
CONFIGURED = "/mnt/mer-local-ssd/mer/qwen3-coder-q4"
ALIAS_PARENT = "/mnt/mer-local-ssd/mer"
DEVICE = "/dev/nvme0n1"
MOUNT = "/mnt/localssd"
EXPERT_SIZE = 2658304
ALIGN = 4096
IDS = (6027, 6030, 6038, 6075, 6093, 6142)
STARTUP_IDS = tuple(range(8))
LAUNCHES = (
    (25, 29, 11, 6027), (41, 46, 14, 6030), (73, 78, 126, 6142),
    (77, 82, 77, 6093), (79, 84, 126, 6142), (80, 85, 126, 6142),
    (81, 86, 126, 6142), (91, 96, 126, 6142), (92, 97, 126, 6142),
    (95, 100, 126, 6142), (102, 107, 11, 6027), (104, 109, 59, 6075),
    (111, 116, 22, 6038),
)
EXPECTATIONS = {
    "kernel": "6.8.0-1069-gcp", "data_device": DEVICE,
    "data_major_minor": "259:4", "data_mount": MOUNT, "data_fstype": "ext4",
    "data_uuid": "b362df64-42eb-4603-a685-c8e5595038ee",
    "configured_data_path": CONFIGURED, "canonical_data_path": CANONICAL,
    "configured_path_alias": "symlink_to_canonical_data_path",
    "root_output_major_minor": "259:1", "expert_size": EXPERT_SIZE,
    "block_align": ALIGN, "no_direct": False, "packed_blob": None,
    "packed_manifest": None, "gpu": "NVIDIA L4", "driver": "580.178.04",
}
TOOLS = ("__init__.py", "supervisor.py", "content_hasher.py", "offline_verify.py", "test_p1v.py")
SOURCE_FILES = {
    "gpu_native_predictor_v2_critical_path_attribution.rs": "187ed89936955aafe67021d380181f786780663fba574632a1699f58496528e3",
    "gpu_native_token_loop.rs": "98bdd025d992dbb00ee05d8e47c5f0512a6d3fdcd84b1f9fc7be47fd1c58be19",
    "gpu_native_residency.rs": "9b7861c6cab3b515877e909d2e86c228705844ac5f7a41b0a8189dc51b70c0a9",
    "backend/gpu_native.rs": "0f22b3c2d878f9638fbc2693d29256bea6296df3b02b4750558dcc7aee93c956",
    "engine.rs": "33a9352ffc758f56bbebb3b6ff87a12c97290f425e0c6c5e6268bea03c88182a",
    "main.rs": "233db5c065e014cb23362510e38d8831b819943b2d3e94691513eebe7d148955",
    "gpu_native_predictor_v2_observation.rs": "5d68d484823a4bae0847d9c48407fe47da4a4464056d2d9f4d2b10e96fcbd7bf",
    "gpu_native_q4_route_parallel.rs": "5702df709aaa6ab9d14ca5a55b60100239ab0417f22b8c0281eea40c2dd879be",
    "gpu_native_predictor_v2_sidecar_performance.rs": "376a325ba49c83d836e1cedd923d88ec8309843a4e9cb391e59c1896e2767981",
}
MODEL = {"architecture": "qwen3_moe", "num_layers": 48, "num_experts_per_layer": 128,
         "total_experts": 6144, "top_k": 8, "d_model": 2048, "d_ff": 768, "routed_expert_dtype": "q4_0"}


class InvalidEvidence(ValueError):
    """A missing, ambiguous or contradictory gate. Never retry automatically."""


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def uint(value, bits=64):
    require(type(value) is int and 0 <= value < 2 ** bits, "invalid unsigned integer")
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha256(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    def bad_constant(value):
        raise InvalidEvidence("nonfinite JSON number: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=Decimal)


def tool_hashes():
    root = Path(__file__).parent
    return {name: digest((root / name).read_bytes()) for name in TOOLS}


def path_bytes(value):
    require(isinstance(value, str) and value.startswith("/"), "absolute path required")
    raw = os.fsencode(value)
    require(b"\0" not in raw and os.path.normpath(value) == value
            and not value.startswith("//"), "noncanonical path")
    return raw


def publish_new(path, data, forbidden=()):
    """Exclusive, fsynced publication. Partial failures are deliberately retained.

    Never rename over a target, follow a symlink, remove failed evidence, or
    retry. The containing directory must be controlled by the caller; future
    host integration must keep seals outside the runtime UID's write authority.
    """
    path = Path(path)
    resolved = path.resolve()
    require(all(resolved != Path(p).resolve() for p in forbidden), "immutable report output target")
    require(not path.is_symlink(), "symlink output target")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "nonregular output")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return digest(data)


class FutureAuthorization:
    """Explicit future grant, supplied externally, never issued by this package.

    This is an interlock, not a signature or authorization service. An approved
    operator must pin the token digest and independently review the host adapter.
    #214 and portable tests cannot authorize live operations.
    """

    def __init__(self, token, expected_sha256, approved_issue):
        require(type(approved_issue) is int and approved_issue > 214, "future authority required")
        require(isinstance(token, bytes) and len(token) >= 32, "future token required")
        require(digest(token) == sha256(expected_sha256), "future token mismatch")
        self.issue = approved_issue


def require_future(authorization):
    require(isinstance(authorization, FutureAuthorization), "live execution NOT_AUTHORIZED under #214")
