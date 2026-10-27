#!/usr/bin/env bash
# MER2 S2-R3: PRIVATE artifact transfer + static integrity verification ONLY.
# No AWS CLI/credentials, no code build, no candidate run, no marker, no GPU.
set -euo pipefail
umask 077
export LC_ALL=C LANG=C TZ=UTC
stop() { printf 'STOP_S2_R3_PRIVATE_STAGE: %s\n' "$*" >&2; exit 1; }
[[ "$(id -un)" == ubuntu ]] || stop 'expected ubuntu user'
[[ "$(uname -srm)" == 'Linux 6.8.0-1066-aws x86_64' ]] || stop 'host kernel/arch drift'
[[ -x /usr/bin/python3.11 ]] || stop 'Python 3.11 is missing'
[[ -d "$HOME/mer-tools" ]] || stop 'mer-tools directory is missing'
command -v curl >/dev/null || stop 'curl is missing'
printf '%s\n' 'MER2 S2-R3 private S3 artifact delivery (NO MER TESTS)'
printf '%s\n' 'The URL is hidden and will not appear in the terminal command or success output.'
IFS= read -r -s -p 'Paste the 1-hour S3 presigned HTTPS URL, then press Enter: ' delivery_url || stop 'no URL supplied'
printf '\n'
[[ -n "$delivery_url" && "$delivery_url" == https://* ]] || stop 'expected HTTPS URL'
case "$delivery_url" in
  *'"'*|*'\'*) stop 'unsupported special character in URL' ;;
esac
host="$(printf '%s' "$delivery_url" | cut -d/ -f3)"
case "$host" in
  s3.amazonaws.com|s3.*.amazonaws.com|*.s3.amazonaws.com|*.s3.*.amazonaws.com|s3-*.amazonaws.com|*.s3-*.amazonaws.com) ;;
  *) stop 'expected an Amazon S3 HTTPS hostname' ;;
esac
stage="$(mktemp -d "$HOME/mer-tools/mer2-s2-r3-inputs.XXXXXXXX")" || stop 'could not allocate fresh staging directory'
zip_path="$stage/MER2-G3-P6A-A3-S2-R3-R1-Verified-Staging-Package.zip"
# Bearer URL is supplied via curl config on stdin, not through argv.
# No non-HTTPS redirects and no automatic retries.
if ! printf 'url = "%s"\n' "$delivery_url" | curl --config - --proto '=https' --proto-redir '=https' --location --fail --silent --show-error --retry 0 --connect-timeout 15 --max-time 180 --output "$zip_path"; then
  unset delivery_url
  stop 'transfer failed; inspect before attempting another delivery'
fi
unset delivery_url
/usr/bin/python3.11 -B - "$zip_path" "$stage" <<'PY'
import hashlib
import io
import json
import pathlib
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
expected_outer_sha = "a62778c8c658f19f17d9f3774a5371910d8921380eddc451379133a735cafeb0"
expected = {
    "MER2-G3-P6A-A3-S2-R2-Evidence.zip": "a0695dcd8198d720002fd37e20033c61b25a61344ad4988f89da6d73fa17cc89",
    "MER2-G3-P6A-A3-S2-R2-Independent-Review.md": "60a2e80078a1be6ca5f17f449270187f03807afe7860e6450520badcd910f1b2",
    "MER2-G3-P6A-A3-S2-R3-R1-execution-spec.json": "7e32a9354b78ec5da3771521c574b9492c5eed1a428be6f49dbd9c5fc10ec74b",
    "MER2-G3-P6A-A3-S2-R3-R1-linux-runner.py": "2593a3054e490378f052d115611a885c66fc0a298cb3a389140963f43137b8a2",
    "MER2-G3-P6A-A3-S2-R3-R1-preparation-verification.json": "b9412a164ac54425db0c31818e21b081135916b2d4797fe897526303705f0624",
    "STAGING-README.txt": "fe33a2a8775f536c1b3a2727f7d2bd61b731c6721500eb2a958fc8d1329c3338",
}
def require(condition, message):
    if not condition:
        raise SystemExit("STOP_S2_R3_PRIVATE_STAGE: " + message)
def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()
def parse_manifest(text):
    records = [line.split("  ", 1) for line in text.splitlines()]
    require(all(len(parts) == 2 for parts in records), "malformed manifest")
    result = {name: digest for digest, name in records}
    require(len(result) == len(records), "duplicate manifest entries")
    return result

require(archive.stat().st_size == 13336850, "unexpected outer package size")
with archive.open("rb") as f:
    require(hashlib.file_digest(f, "sha256").hexdigest() == expected_outer_sha, "outer SHA256 mismatch")
with zipfile.ZipFile(archive) as outer:
    infos = outer.infolist()
    names = [i.filename for i in infos]
    require(len(names) == len(set(names)), "duplicate outer ZIP entry")
    require(set(names) == set(expected) | {"SHA256SUMS"}, "unexpected outer ZIP members")
    require(all("/" not in name and "\\" not in name for name in names), "invalid outer member name")
    require(outer.testzip() is None, "outer ZIP CRC failure")
    manifest = parse_manifest(outer.read("SHA256SUMS").decode("ascii"))
    require(manifest == expected, "outer manifest mismatch")
    for name, sha in expected.items():
        require(hash_bytes(outer.read(name)) == sha, "outer payload hash: " + name)
    r2_bytes = outer.read("MER2-G3-P6A-A3-S2-R2-Evidence.zip")
    spec = json.loads(outer.read("MER2-G3-P6A-A3-S2-R3-R1-execution-spec.json"))
    require(spec["source_sha256"] and len(spec["source_sha256"]) == 8, "source map missing")
    require(spec["zip_sha256"] == expected["MER2-G3-P6A-A3-S2-R2-Evidence.zip"], "R2 reference drift")
    with zipfile.ZipFile(io.BytesIO(r2_bytes)) as inner:
        inner_names = inner.namelist()
        require(len(inner_names) == len(set(inner_names)), "duplicate R2 entries")
        inner_manifest = parse_manifest(inner.read("SHA256SUMS").decode("ascii"))
        require(len(inner_manifest) == 122 and set(inner_manifest) == set(inner_names) - {"SHA256SUMS"}, "R2 manifest incomplete")
        require(inner.testzip() is None, "R2 ZIP CRC failure")
        for name, digest in inner_manifest.items():
            require(hash_bytes(inner.read(name)) == digest, "R2 payload mismatch: " + name)
        for rel, sha in spec["source_sha256"].items():
            require(hash_bytes(inner.read("corrected-source/" + rel)) == sha, "corrected source mismatch: " + rel)
    # Extract ONLY seven verified top-level staging members.
    # Do NOT extract R2 historical markers or execute source code.
    for name in names:
        destination = root / name
        with destination.open("xb") as out:
            out.write(outer.read(name))
        destination.chmod(0o600)
print("S2_R3_PRIVATE_STAGING=PASS")
print("OUTER_SHA256=" + expected_outer_sha)
print("R2_SHA256=" + expected["MER2-G3-P6A-A3-S2-R2-Evidence.zip"])
print("R2_MANIFEST=122_OF_122")
print("CORRECTED_SOURCE_HASHES=8_OF_8")
print("RUNNER_SHA256=" + expected["MER2-G3-P6A-A3-S2-R3-R1-linux-runner.py"])
print("SPEC_SHA256=" + expected["MER2-G3-P6A-A3-S2-R3-R1-execution-spec.json"])
print("STAGED_ROOT=" + str(root))
print("MER_BUILDS=0 MER_TESTS=0 ONE_USE_MARKERS=0")
print("EXECUTION_AUTHORITY=NOT_PRESENT")
PY
