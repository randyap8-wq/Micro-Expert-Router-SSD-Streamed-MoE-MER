#!/usr/bin/env bash
# Read-only prerequisite inventory for MER2 S2-R3. NO installation, mutation, tests, or marker.
set -u
set -o pipefail
export LC_ALL=C LANG=C TZ=UTC RUSTUP_AUTO_INSTALL=0 CARGO_NET_OFFLINE=true
shopt -s nullglob
printf '%s\n' '== MER2_S2_R3_PREREQUISITE_INVENTORY =='
printf 'timestamp_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'kernel_arch='; uname -srm
printf '\n%s\n' '== EXISTING PYTHON >=3.11 (no install) =='
py_count=0
for py in /usr/bin/python3.1[1-9] /usr/local/bin/python3.1[1-9] /opt/*/bin/python3.1[1-9] "$HOME"/.pyenv/versions/*/bin/python3.1[1-9] "$HOME"/miniconda*/bin/python3.1[1-9] "$HOME"/.local/share/uv/python/*/bin/python3.1[1-9]; do
  [[ -x "$py" ]] || continue
  "$py" -B -c 'import hashlib,platform,sys; print("PYTHON="+sys.executable+" VERSION="+platform.python_version()+" ELIGIBLE="+str(sys.version_info>=(3,11))+" FILE_DIGEST="+str(hasattr(hashlib,"file_digest")))' 2>&1 || true
  py_count=$((py_count+1))
done
printf 'python_candidates_found=%d\n' "$py_count"
printf '\n%s\n' '== DIRECT NATIVE TOOLCHAIN BINARIES =='
toolbin="$HOME/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin"
printf 'toolbin=%s\n' "$toolbin"
for tool in cargo rustc rustfmt cargo-fmt; do
  p="$toolbin/$tool"
  if [[ ! -x "$p" ]]; then printf 'missing=%s\n' "$p"; continue; fi
  printf 'tool=%s path=%s\n' "$tool" "$p"
  sha256sum -- "$p" || true
  magic=$(od -An -tx1 -N4 -- "$p" 2>/dev/null | tr -d '[:space:]')
  [[ "$magic" == 7f454c46 ]] && printf 'native_elf=YES\n' || printf 'native_elf=NO\n'
  if [[ "$tool" == rustc ]]; then "$p" -vV 2>&1 || true; else "$p" --version 2>&1 || true; fi
done
printf '\n%s\n' '== SSD RUN ROOT PARENT (no creation) =='
for p in /mnt/localssd /mnt/localssd/mer2-s2-r3-runs; do
  if [[ -d "$p" ]]; then
    stat -c 'dir=%n mode=%a owner=%U group=%G' "$p" 2>&1 || true
    [[ -w "$p" ]] && printf 'writable=YES path=%s\n' "$p" || printf 'writable=NO path=%s\n' "$p"
  else
    printf 'not_present=%s\n' "$p"
  fi
done
df -hP /mnt/localssd 2>&1 || true
printf '\n%s\n' 'MER2_S2_R3_PREREQUISITE_INVENTORY=COMPLETE_READ_ONLY'
printf '%s\n' 'INSTALLATIONS=0 SOURCE_EDITS=0 CARGO_BUILDS=0 TESTS=0 EXECUTION_MARKERS=0'
printf '%s\n' 'EXECUTION_AUTHORIZATION=NOT_GRANTED'
