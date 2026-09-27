"""Post-shutdown direct-only full-file SHA256 and frozen P1V binary ledger.

LinuxDirectReader uses an aligned libc pread buffer, not Python buffered file
I/O or a shell. Construction requires a future grant. Tests inject fake readers.
"""

import ctypes
import fcntl
import hashlib
import os
import stat
import struct
import sys
from dataclasses import asdict, dataclass

from . import (ALIGN, CANONICAL, CONFIGURED, EXPERT_SIZE, IDS, digest,
               path_bytes, require, require_future, sha256, uint)


@dataclass(frozen=True)
class Identity:
    device_major: int
    device_minor: int
    inode: int
    file_length: int
    mode: int
    links: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value):
        return cls(os.major(value.st_dev), os.minor(value.st_dev), value.st_ino,
                   value.st_size, value.st_mode, value.st_nlink,
                   value.st_mtime_ns, value.st_ctime_ns)

    def check(self):
        require(stat.S_ISREG(self.mode) and self.links == 1, "selected file must be regular with one link")
        require((self.device_major, self.device_minor) == (259, 4), "wrong expert device")
        require(self.file_length == EXPERT_SIZE, "wrong expert size")
        uint(self.inode)


def alternatives(global_id):
    uint(global_id, 32)
    return (f"{CONFIGURED}/expert_{global_id}.bin",
            f"{CONFIGURED}/expert_{global_id // 128}_{global_id % 128}.bin")


def resolve(reader, global_id):
    """Same primary preference as NvmeStorage; only ENOENT permits fallback.

    Frozen source falls back on *any* metadata error. Permission/I/O errors are
    ambiguous here and fail instead. A dangling symlink is rejected by lstat.
    """
    primary, fallback = alternatives(global_id)
    try:
        identity = reader.metadata(primary)
        selected = primary
    except FileNotFoundError:
        selected = fallback
        identity = reader.metadata(fallback)
    identity.check()
    canonical = reader.canonical(selected)
    require(canonical == CANONICAL + "/" + selected.rsplit("/", 1)[1], "resolver escape")
    return selected, canonical, identity


def encode_ledger(records):
    require(isinstance(records, list) and len(records) == 6, "exact six records required")
    require([r["global_expert_id"] for r in records] == list(IDS), "six-ID ordering mismatch")
    ledger = bytearray(b"MER-P1V-CONTENT-v1\0" + struct.pack("<I", 6))
    for record in records:
        g = uint(record["global_expert_id"], 32)
        selected = record["authoritative_selected_path"]
        canonical = record["canonical_path"]
        require(selected in alternatives(g), "unexpected selected path")
        require(canonical == CANONICAL + "/" + selected.rsplit("/", 1)[1], "unexpected canonical path")
        body = bytearray(struct.pack("<I", g))
        for path in (selected, canonical):
            raw = path_bytes(path)
            body.extend(struct.pack("<I", uint(len(raw), 32)) + raw)
        require((record["device_major"], record["device_minor"]) == (259, 4), "ledger device")
        require(record["file_length"] == EXPERT_SIZE, "ledger length")
        body.extend(struct.pack("<IIQQ", uint(record["device_major"], 32),
                                uint(record["device_minor"], 32), uint(record["inode"]),
                                uint(record["file_length"])))
        body.extend(bytes.fromhex(sha256(record["sha256"])))
        ledger.extend(struct.pack("<I", len(body)) + body)
    return bytes(ledger)


def check_hash_window(window):
    require(window == {"phase": "POST_SHUTDOWN", "runtime_tree_empty": True,
                       "filesystem_ro": True, "device_ro": True,
                       "alias_locked": True, "exclusion_held": True},
            "hashing requires stopped runtime and intact protected interval")


def hash_content(reader, before, window):
    check_hash_window(window)
    require(list(before) == list(IDS), "pre-metadata ordering/coverage")
    records = []
    for g in IDS:
        selected, canonical, identity = resolve(reader, g)
        require(before[g] == (selected, canonical, identity), "pre/post resolver identity drift")
        handle = reader.open_direct(selected)
        try:
            require(reader.is_direct(handle), "O_DIRECT missing; fallback forbidden")
            require(reader.identity(handle) == identity, "path/descriptor identity mismatch")
            hasher = hashlib.sha256()
            count = 0
            while count < EXPERT_SIZE:
                data = reader.read_direct(handle, count, ALIGN)
                require(type(data) is bytes and len(data) == ALIGN, "short/error direct read")
                hasher.update(data)
                count += len(data)
            require(count == EXPERT_SIZE and reader.identity(handle) == identity,
                    "byte count/descriptor drift")
            require(resolve(reader, g) == (selected, canonical, identity), "post-hash path drift")
            records.append({"global_expert_id": g, "authoritative_selected_path": selected,
                            "canonical_path": canonical, "device_major": identity.device_major,
                            "device_minor": identity.device_minor, "inode": identity.inode,
                            "file_length": count, "sha256": hasher.hexdigest(),
                            "direct": True, "identity": asdict(identity)})
        finally:
            reader.close(handle)
    ledger = encode_ledger(records)
    return {"schema": "mer.p1v.content.v1", "status": "PASS", "records": records,
            "ledger_hex": ledger.hex(), "content_root": digest(ledger),
            "read_size": ALIGN, "full_stored_bytes": True, "buffered_fallback": False}


def verify_content(content):
    require(content["schema"] == "mer.p1v.content.v1" and content["status"] == "PASS", "content status")
    require(content["read_size"] == ALIGN and content["full_stored_bytes"] is True
            and content["buffered_fallback"] is False, "direct content contract")
    for r in content["records"]:
        identity = Identity(**r["identity"])
        identity.check()
        require(r["direct"] is True, "non-direct content record")
        for key in ("device_major", "device_minor", "inode", "file_length"):
            require(r[key] == getattr(identity, key), "ledger/identity mismatch")
    ledger = encode_ledger(content["records"])
    require(ledger.hex() == content["ledger_hex"] and digest(ledger) == content["content_root"], "content root mismatch")
    return content["content_root"]


class LinuxDirectReader:
    """Future-only reader. Host adapter must establish/retain the hash window.

    All directory components are checked. The sole permitted symlink is the
    frozen configured alias itself. Trusted-host exclusion plus alias lockdown
    is still essential; these endpoint checks alone do not prove continuity.
    """

    def __init__(self, authorization, window):
        require_future(authorization)
        check_hash_window(window)
        require(sys.platform == "linux" and hasattr(os, "O_DIRECT"), "Linux O_DIRECT required")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
        self.libc.posix_memalign.restype = ctypes.c_int
        self.libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
        self.libc.pread.restype = ctypes.c_ssize_t
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None

    def canonical(self, path):
        require(path in [p for g in IDS for p in alternatives(g)], "unapproved expert path")
        require(os.path.islink(CONFIGURED) and os.readlink(CONFIGURED) == CANONICAL, "configured alias drift")
        for root in (os.path.dirname(CONFIGURED), CANONICAL):
            current = "/"
            for component in root.strip("/").split("/"):
                current = os.path.join(current, component)
                require(stat.S_ISDIR(os.lstat(current).st_mode), "symlink/non-directory ancestor")
        return os.path.realpath(path, strict=True)

    def metadata(self, path):
        # lstat must precede realpath so dangling primary symlinks cannot select fallback.
        value = Identity.from_stat(os.lstat(path))
        value.check()
        self.canonical(path)
        return value

    def open_direct(self, path):
        self.metadata(path)
        return os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW | os.O_CLOEXEC)

    def is_direct(self, handle):
        return bool(fcntl.fcntl(handle, fcntl.F_GETFL) & os.O_DIRECT)

    def identity(self, handle):
        return Identity.from_stat(os.fstat(handle))

    def read_direct(self, handle, offset, length):
        require(length == ALIGN and offset % ALIGN == 0 and 0 <= offset < EXPERT_SIZE,
                "unaligned/out-of-range direct read")
        buffer = ctypes.c_void_p()
        require(self.libc.posix_memalign(ctypes.byref(buffer), ALIGN, length) == 0, "aligned allocation failed")
        try:
            got = self.libc.pread(handle, buffer, length, offset)
            if got < 0:
                raise OSError(ctypes.get_errno(), "direct pread failed; no retry/fallback")
            require(got == length, "short direct pread")
            return ctypes.string_at(buffer, got)
        finally:
            self.libc.free(buffer)

    def close(self, handle):
        os.close(handle)
