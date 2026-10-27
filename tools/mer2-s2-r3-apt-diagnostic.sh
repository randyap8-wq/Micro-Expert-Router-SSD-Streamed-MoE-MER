#!/usr/bin/env bash
# Read-only diagnostic after S2-R3 host remediation exited before reporting PASS.
# No sudo, no apt install/update, no file writes, no MER build/test or marker.
set -u
set -o pipefail
export LC_ALL=C LANG=C TZ=UTC
printf '%s\n' '== MER2_S2_R3_APT_DIAGNOSTIC_READ_ONLY =='
printf 'time_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'user='; id -un
printf 'kernel_arch='; uname -srm

printf '\n%s\n' '== PYTHON STATE =='
printf 'system_python='
/usr/bin/python3 -B -c 'import sys; print(sys.version.split()[0])' 2>&1 || true
for p in /usr/bin/python3.11 /usr/local/bin/python3.11; do
  if [[ -x "$p" ]]; then
    "$p" -B -c 'import sys,hashlib; print("path="+sys.executable+" version="+sys.version.split()[0]+" file_digest="+str(hasattr(hashlib,"file_digest")))' 2>&1 || true
  else
    printf 'python_path=%s state=ABSENT\n' "$p"
  fi
done

printf '\n%s\n' '== APT POLICY (capture return code; do not use early-exit pipeline) =='
if command -v apt-cache >/dev/null 2>&1; then
  policy="$(apt-cache policy python3.11 2>&1)"
  rc=$?
  printf 'apt_cache_exit=%d\n' "$rc"
  printf '%s\n' "$policy"
  candidate='NOT_REPORTED'
  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*Candidate:[[:space:]]*(.*)$ ]]; then
      candidate="${BASH_REMATCH[1]}"
    fi
  done <<< "$policy"
  printf 'apt_candidate=%s\n' "$candidate"
else
  printf '%s\n' 'apt_cache=NOT_FOUND'
fi

printf '\n%s\n' '== PACKAGE DATABASE (observation only) =='
if command -v dpkg-query >/dev/null 2>&1; then
  status="$(dpkg-query -W python3.11 2>&1)"
  rc=$?
  printf 'dpkg_query_exit=%d\n' "$rc"
  printf '%s\n' "$status"
fi
if [[ -d /var/lib/apt/lists ]]; then
  count="$(find /var/lib/apt/lists -maxdepth 1 -type f 2>/dev/null | wc -l)"
  printf 'apt_index_regular_files=%s\n' "$count"
fi

printf '\n%s\n' '== SSD DIRECTORY (observation only) =='
for p in /mnt/localssd /mnt/localssd/mer2-s2-r3-runs; do
  if [[ -d "$p" ]]; then
    stat -c 'dir=%n mode=%a owner=%U group=%G' "$p" 2>&1 || true
    if [[ -w "$p" ]]; then
      printf 'dir_writable_by_user=YES path=%s\n' "$p"
    else
      printf 'dir_writable_by_user=NO path=%s\n' "$p"
    fi
  else
    printf 'dir_absent=%s\n' "$p"
  fi
done

printf '\n%s\n' 'DIAGNOSTIC_COMPLETE=READ_ONLY'
printf '%s\n' 'CHANGES=0 MER_BUILDS=0 MER_TESTS=0 EXECUTION_MARKERS=0'
