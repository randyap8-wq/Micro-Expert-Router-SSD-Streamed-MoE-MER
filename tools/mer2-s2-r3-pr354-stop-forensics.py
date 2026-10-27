#!/usr/bin/env python3
"""One-read forensic inspection of a frozen S2-R3 authority review STOP.

This NEVER invokes the failed independent review again; no writes, tests, Cargo,
marker, source edits, run root, or authorization. Compare the already-collected
PR #354 receipt with ONE fresh GitHub GET, report only differing JSON key paths.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path("/home/ubuntu/mer-tools/mer2-s2-r3-authority.a872b5b9cb5e81")
REQUEST_SHA = "0ebd0032e0228ce39dfabbc159c48a3fdb284aa856355174a7ff91e15ebe6457"
EXPECTED_HEAD = "6150f4bb047b18679ddd48e468b819d7f39ad2ac"
EXPECTED_BASE = "1b72de2fdb72fd10c8125d766a56b8876eb081af"
URL = "https://api.github.com/repos/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/pulls/354"

class Stop(Exception):
    pass

def require(ok, message):
    if not ok:
        raise Stop(message)

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def delta(a, b, prefix="", output=None):
    if output is None:
        output = []
    if type(a) is not type(b):
        output.append(prefix or "(root)")
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            label = prefix + "/" + key
            if key not in a or key not in b:
                output.append(label + (" (only live)" if key not in a else " (only stored)"))
            else:
                delta(a[key], b[key], label, output)
    elif isinstance(a, list):
        if len(a) != len(b):
            output.append(prefix + "/length")
        for index in range(min(len(a), len(b))):
            delta(a[index], b[index], prefix + "/" + str(index), output)
    elif a != b:
        output.append(prefix or "(root)")
    return output

def expected_state(record):
    return (record.get("number") == 354
        and record.get("head", {}).get("sha") == EXPECTED_HEAD
        and record.get("base", {}).get("sha") == EXPECTED_BASE
        and record.get("state") == "open"
        and record.get("draft") is True
        and record.get("merged") is False
        and record.get("merged_at") is None)

def main():
    require(len(sys.argv) == 1, "NO_ARGUMENTS")
    req_path = ROOT / "review-request.json"
    receipt = ROOT / "receipts" / "pr-354.json"
    require(req_path.is_file() and receipt.is_file(), "RECEIPTS_MISSING")
    require(digest(req_path) == REQUEST_SHA, "FROZEN_REVIEW_REQUEST_HASH_DRIFT")
    request = json.loads(req_path.read_text())
    pinned = request["receipts"]["pr-354"]["sha256"]
    require(digest(receipt) == pinned, "FROZEN_PR354_RECEIPT_HASH_DRIFT")
    stored = json.loads(receipt.read_text())
    require(expected_state(stored), "STORED_FROZEN_PR354_IDENTITY_DRIFT")
    webrequest = urllib.request.Request(URL, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "MER-S2-R3-pr354-STOP-forensics/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    # One read only. No retries, mutation, redirects, or fallback route.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise Stop("UNEXPECTED_API_REDIRECT")
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(webrequest, timeout=20) as response:
        require(response.status == 200 and response.geturl() == URL, "UNEXPECTED_API_RESPONSE")
        raw = response.read(2_000_001)
    require(len(raw) < 2_000_001, "API_RESPONSE_TOO_LARGE")
    live = json.loads(raw)
    if not expected_state(live):
        print("PR354_FROZEN_IDENTITY=STOP_MATERIAL_DRIFT")
        print("EXECUTION_AUTHORIZED=FALSE")
        raise Stop("LIVE_PR354_FROZEN_IDENTITY_CHANGED")
    changed = delta(stored, live)
    print("S2_R3_PR354_STOP_FORENSICS=READ_ONLY_COMPLETE")
    print("PR354_FROZEN_IDENTITY=PASS_EXPECTED_HEAD_BASE_OPEN_DRAFT_UNMERGED")
    print("STORED_PR354_RECEIPT_SHA256=" + pinned)
    print("LIVE_JSON_EQUALS_STORED=" + ("YES" if not changed else "NO"))
    print("CHANGED_JSON_PATH_COUNT=" + str(len(changed)))
    for field in changed[:60]:
        print("DIFFERING_FIELD=" + field)
    if len(changed) > 60:
        print("ADDITIONAL_FIELDS_SUPPRESSED=" + str(len(changed) - 60))
    for key in ("mergeable", "mergeable_state", "rebaseable", "comments",
                "review_comments", "updated_at"):
        if stored.get(key) != live.get(key):
            print("NON_SOURCE_METADATA_CHANGE=" + key + " stored=" +
                  repr(stored.get(key)) + " live=" + repr(live.get(key)))
    print("FAILED_REVIEW_REEXECUTED=NO")
    print("EXECUTION_AUTHORIZED=FALSE MER_BUILDS=0 MER_TESTS=0 MARKERS=0")
    print("NEXT=ASSESS_ROOT_CAUSE_AND_PREPARE_SEPARATE_NEW_REVIEW_ATTEMPT")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOP_S2_R3_PR354_FORENSICS: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr)
        sys.exit(1)
