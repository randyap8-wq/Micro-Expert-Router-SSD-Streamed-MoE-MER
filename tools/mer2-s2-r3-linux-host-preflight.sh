#!/usr/bin/env bash
# MER2 G3 P6A A3 S2 R3: read-only Linux host preflight.
# This is NOT an execution-authority receipt or a test launcher.
# No sudo, disk mutations, Cargo builds/tests, GPU/CUDA access, model operations,
# archive extraction, source modification, marker creation, or publication.
set -u
set -o pipefail
export LC_ALL=C LANG=C TZ=UTC RUSTUP_AUTO_INSTALL=0 CARGO_NET_OFFLINE=true

section() { printf '\n== %s ==\n' "$1"; }
section 'MER2_S2_R3_READ_ONLY_HOST_PREFLIGHT'
printf 'timestamp_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'hostname='; hostname 2>/dev/null || printf 'UNAVAILABLE\n'
printf 'uname='; uname -srm
if [[ -r /etc/os-release ]]; then
  grep -E '^(NAME|VERSION|ID|VERSION_ID)=' /etc/os-release || true
fi
printf 'architecture='; uname -m
printf 'proc_self_stat='; [[ -r /proc/self/stat ]] && printf 'READABLE\n' || printf 'UNAVAILABLE\n'

section 'PYTHON (no bytecode)'
for py in python3 python3.11 python3.12 python3.13 python3.14; do
  if command -v "$py" >/dev/null 2>&1; then
    "$py" -B -c 'import sys; print("python_executable="+sys.executable+" version="+sys.version.split()[0]+" meets_3_11="+str(sys.version_info >= (3,11)))' 2>&1 || true
  fi
done

section 'RUST TOOLS (identity only; no build)'
for tool in cargo rustc rustfmt cargo-fmt; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'tool=%s state=NOT_FOUND\n' "$tool"
    continue
  fi
  declared=$(command -v "$tool")
  resolved=$(readlink -f -- "$declared" 2>/dev/null || printf '%s' "$declared")
  printf 'tool=%s declared=%s resolved=%s\n' "$tool" "$declared" "$resolved"
  if [[ -f "$resolved" ]]; then
    printf 'tool_sha256='; sha256sum -- "$resolved" 2>/dev/null || printf 'UNAVAILABLE\n'
    magic=$(od -An -tx1 -N4 -- "$resolved" 2>/dev/null | tr -d '[:space:]')
    [[ "$magic" == '7f454c46' ]] && printf 'tool_elf=YES\n' || printf 'tool_elf=NO\n'
  fi
done
printf 'rustc_version='; rustc -vV 2>&1 || true
printf 'cargo_version='; cargo --version 2>&1 || true
printf 'rustfmt_version='; rustfmt --version 2>&1 || true
printf 'cargo_fmt_version='; cargo fmt --version 2>&1 || true

section 'INSTALLED NATIVE TOOLCHAIN CANDIDATES (listing only)'
found=0
for bindir in "$HOME"/.rustup/toolchains/*/bin; do
  if [[ -f "$bindir/cargo" && -f "$bindir/rustc" && -f "$bindir/rustfmt" && -f "$bindir/cargo-fmt" ]]; then
    printf 'toolchain_bin=%s\n' "$bindir"
    found=$((found+1))
  fi
done
printf 'toolchain_directories_found=%d\n' "$found"

section 'LOCAL SSD (metadata only)'
if [[ ! -d /mnt/localssd ]]; then
  printf 'localssd=PATH_MISSING\n'
else
  if command -v mountpoint >/dev/null 2>&1 && mountpoint -q /mnt/localssd; then
    printf 'localssd=MOUNTED\n'
  else
    printf 'localssd=NOT_A_SEPARATE_MOUNTPOINT\n'
  fi
  printf 'localssd_permissions='; stat -c '%A %U:%G' /mnt/localssd 2>/dev/null || printf 'UNAVAILABLE\n'
  printf 'localssd_writable_by_current_user='; [[ -w /mnt/localssd ]] && printf 'YES\n' || printf 'NO\n'
  if command -v findmnt >/dev/null 2>&1; then
    findmnt --mountpoint /mnt/localssd -o TARGET,SOURCE,FSTYPE,OPTIONS --noheadings 2>&1 || true
  fi
  df -hP /mnt/localssd 2>&1 || true
fi

section 'OPTIONAL OFFLINE CARGO CACHE (existence only)'
for path in "$HOME/.cargo/registry/src" "$HOME/.cargo/registry/cache" "$HOME/.cargo/git/checkouts"; do
  [[ -d "$path" ]] && printf 'cache=%s PRESENT\n' "$path" || printf 'cache=%s ABSENT\n' "$path"
done

section 'RESULT'
printf 'MER2_S2_R3_HOST_PREFLIGHT=COMPLETE_INFORMATION_ONLY\n'
printf 'TESTS_EXECUTED=0 BUILDS_EXECUTED=0 EXECUTION_MARKERS_CREATED=0\n'
printf 'NO_EXECUTION_AUTHORIZATION_IS_GRANTED_BY_THIS_SCRIPT\n'