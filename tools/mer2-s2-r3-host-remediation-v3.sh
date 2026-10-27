#!/usr/bin/env bash
# MER2 S2-R3: explicit opt-in host-only remediation, not a candidate test.
# Installs PINNED Ubuntu Jammy Python 3.11 RC1 ONLY for the isolated CPU harness,
# and creates one ubuntu-owned directory. Does not alter system python3.
# No apt update, third-party repo, Rust builds, GPU access, models, or gate markers.
set -euo pipefail
export LC_ALL=C LANG=C TZ=UTC RUSTUP_AUTO_INSTALL=0 CARGO_NET_OFFLINE=true
stop() { printf 'STOP_HOST_REMEDIATION_V3: %s\n' "$*" >&2; exit 1; }
printf '%s\n' '== MER2_S2_R3_HOST_REMEDIATION_V3 =='
printf 'timestamp_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
[[ "$(id -un)" == ubuntu ]] || stop 'expected ubuntu operator'
[[ -r /etc/os-release ]] || stop 'cannot read OS release'
. /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || stop 'unexpected Ubuntu release'
[[ "$(uname -srm)" == 'Linux 6.8.0-1066-aws x86_64' ]] || stop 'unexpected kernel/architecture'
command -v findmnt >/dev/null 2>&1 || stop 'findmnt not installed'
mountpoint -q /mnt/localssd || stop 'local SSD not mounted'
mount_device="$(findmnt -n -o SOURCE --mountpoint /mnt/localssd)" || stop 'cannot read mount source'
mount_fstype="$(findmnt -n -o FSTYPE --mountpoint /mnt/localssd)" || stop 'cannot read mount fs'
[[ "$mount_device" == /dev/nvme1n1 && "$mount_fstype" == ext4 ]] || stop 'SSD mount identity drift'
[[ "$(stat -c '%U:%G:%a' /mnt/localssd)" == root:root:755 ]] || stop 'mount permissions drift'
toolbin="$HOME/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin"
while read -r name expected; do
  [[ -x "$toolbin/$name" ]] || stop "native Rust tool missing: $name"
  printf '%s  %s\n' "$expected" "$toolbin/$name" | sha256sum -c - >/dev/null || stop "native Rust hash drift: $name"
done <<'EXPECTED_RUST'
cargo 828980723df339d62434390e9fb8ef8831036583343ae2316b7ab5646b5c1953
rustc d3a664c970a9fd8361b64194861bebc1ae37b9054e5ee3400dc1c9e691797eea
rustfmt 30de9e1efcd8f8fe7750e00d0c45ff8f4c480608ef1be5baf9ab6f1b4556e8f8
cargo-fmt 07ce638a5be73e97defbf0864f32a64ed40a0b7682b85625e68790f9e68d19e9
EXPECTED_RUST
parent=/mnt/localssd/mer2-s2-r3-runs
[[ ! -L "$parent" ]] || stop 'run-root parent is a symlink'
if [[ -e "$parent" ]]; then
  [[ -d "$parent" ]] || stop 'run-root parent is not directory'
  [[ "$(stat -c '%U:%G:%a' "$parent")" == ubuntu:ubuntu:700 ]] || stop 'existing run-root parent ownership/mode drift'
else
  [[ ! -e /mnt/localssd/.git ]] || stop 'unexpected Git marker at mount root'
fi

printf '\n%s\n' '== Python 3.11 via pinned official Ubuntu archive (not production qualification) =='
if [[ -x /usr/bin/python3.11 ]]; then
  printf '%s\n' 'python3.11 already installed; no APT mutation'
else
  policy=''
  if ! policy="$(apt-cache policy python3.11 2>&1)"; then
    printf '%s\n' "$policy" >&2
    stop 'apt-cache policy failed; no installation attempted'
  fi
  printf '%s\n' "$policy"
  candidate=''
  while IFS= read -r policyline; do
    if [[ "$policyline" =~ ^[[:space:]]*Candidate:[[:space:]]*(.*)$ ]]; then
      candidate="${BASH_REMATCH[1]}"
    fi
  done <<< "$policy"
  [[ "$candidate" == '3.11.0~rc1-1~22.04.1' ]] || stop "unexpected apt candidate: $candidate"
  printf '%s\n' 'WARNING: This official Jammy package is Python 3.11 RC1; use it only for this frozen local test harness, not production workloads.'
  simulation=''
  if ! simulation="$(apt-get -s --no-install-recommends --no-upgrade install "python3.11=$candidate" 2>&1)"; then
    printf '%s\n' "$simulation" >&2
    stop 'APT simulation failed; nothing installed'
  fi
  printf '%s\n' "$simulation"
  printf '%s\n' "$simulation" | grep -Eq '^0 upgraded, [0-9]+ newly installed, 0 to remove' || stop 'APT simulation plans an upgrade or removal'
  new_packages=0
  while IFS= read -r aptline; do
    if [[ "$aptline" =~ ^Remv[[:space:]] ]]; then stop "APT planned removal: $aptline"; fi
    if [[ "$aptline" =~ ^Inst[[:space:]]+([^[:space:]]+) ]]; then
      name="${BASH_REMATCH[1]}"
      # APT's [amd64] after a NEW package is architecture metadata, not an upgrade.
      case "$name" in
        python3.11|python3.11-minimal|libpython3.11-minimal|libpython3.11-stdlib|media-types|libmpdec3) ;;
        *) stop "unexpected APT package in plan: $name" ;;
      esac
      new_packages=$((new_packages+1))
    fi
  done <<< "$simulation"
  (( new_packages >= 4 )) || stop 'APT simulation did not show expected Python components'
  printf '\n%s\n' 'Proceeding with the bounded APT install; no apt update, no repository changes.'
  if ! sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get -y --no-install-recommends --no-upgrade install "python3.11=$candidate"; then
    stop 'APT installation failed; inspect package state, do not blindly retry'
  fi
fi
[[ -x /usr/bin/python3.11 ]] || stop 'Python executable still absent'
/usr/bin/python3.11 -B -c 'import hashlib,io,sys; assert sys.version_info >= (3,11); assert hasattr(hashlib,"file_digest"); assert hashlib.file_digest(io.BytesIO(b"abc"),"sha256").hexdigest()=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; print("PYTHON_TEST_INTERPRETER="+sys.executable+" VERSION="+sys.version.split()[0])' || stop 'Python 3.11 basic standard-library check failed'
[[ "$(readlink -f /usr/bin/python3)" == /usr/bin/python3.10 ]] || stop 'system python3 link changed unexpectedly'

printf '\n%s\n' '== Dedicated SSD parent (no mount-level chmod or chown) =='
if [[ ! -e "$parent" ]]; then
  sudo -n install -d -m 0700 -o ubuntu -g ubuntu -- "$parent" || stop 'could not create dedicated SSD parent'
fi
[[ -d "$parent" && -w "$parent" ]] || stop 'dedicated SSD directory missing or unwritable'
[[ "$(stat -c '%U:%G:%a' "$parent")" == ubuntu:ubuntu:700 ]] || stop 'dedicated SSD directory identity mismatch'
printf 'SSD_RUN_PARENT=%s\n' "$parent"
printf '%s\n' 'HOST_PREPARATION_V3=PASS'
printf '%s\n' 'MER_BUILDS=0 MER_TESTS=0 EXECUTION_MARKERS=0'
printf '%s\n' 'CORRECTED_SOURCE_EXECUTION_AUTHORITY=NOT_GRANTED'
