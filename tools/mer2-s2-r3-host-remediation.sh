#!/usr/bin/env bash
# Explicit opt-in host remediation, separate from MER2 S2-R3 test execution.
# Installs Ubuntu's python3.11 package (only if absent) and creates one owned SSD parent.
# Never runs the frozen MER candidate, Cargo, a GPU, CUDA, or any one-shot marker.
set -euo pipefail
export LC_ALL=C LANG=C TZ=UTC RUSTUP_AUTO_INSTALL=0 CARGO_NET_OFFLINE=true
stop() { printf 'STOP_HOST_PREPARATION: %s\n' "$*" >&2; exit 1; }
[[ "$(id -un)" == "ubuntu" ]] || stop 'expected ubuntu user'
[[ -r /etc/os-release ]] || stop 'missing OS identity'
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] || stop 'unexpected Ubuntu release'
[[ "$(uname -srm)" == "Linux 6.8.0-1066-aws x86_64" ]] || stop 'unexpected Linux kernel or architecture'
mountpoint -q /mnt/localssd || stop 'SSD is not mounted'
source_device=$(findmnt -n -o SOURCE --mountpoint /mnt/localssd)
source_type=$(findmnt -n -o FSTYPE --mountpoint /mnt/localssd)
[[ "$source_device" == "/dev/nvme1n1" && "$source_type" == "ext4" ]] || stop 'unexpected SSD source or filesystem'
[[ -w /mnt/localssd ]] && stop 'SSD root permissions differ from read-only preflight'
toolbin="$HOME/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin"
while read -r tool sha; do
  [[ -x "$toolbin/$tool" ]] || stop "missing native $tool"
  printf '%s  %s\n' "$sha" "$toolbin/$tool" | sha256sum -c - >/dev/null || stop "tool drift: $tool"
done <<'TOOL_HASHES'
cargo 828980723df339d62434390e9fb8ef8831036583343ae2316b7ab5646b5c1953
rustc d3a664c970a9fd8361b64194861bebc1ae37b9054e5ee3400dc1c9e691797eea
rustfmt 30de9e1efcd8f8fe7750e00d0c45ff8f4c480608ef1be5baf9ab6f1b4556e8f8
cargo-fmt 07ce638a5be73e97defbf0864f32a64ed40a0b7682b85625e68790f9e68d19e9
TOOL_HASHES
parent=/mnt/localssd/mer2-s2-r3-runs
[[ ! -L "$parent" ]] || stop 'refuse symlink parent'
if [[ -e "$parent" ]]; then
  [[ -d "$parent" && "$(stat -c %U "$parent")" == "ubuntu" && "$(stat -c %a "$parent")" == "700" ]] || stop 'existing run parent has unexpected ownership or mode'
else
  [[ ! -e /mnt/localssd/.git ]] || stop 'SSD contains .git marker'
fi

printf '\n== Python 3.11 host preparation (NOT a MER test) ==\n'
if [[ ! -x /usr/bin/python3.11 ]]; then
  candidate=$(apt-cache policy python3.11 | awk '/Candidate:/{print $2; exit}')
  [[ -n "$candidate" && "$candidate" != "(none)" ]] || stop 'Ubuntu python3.11 package unavailable; do not add repositories automatically'
  printf 'Ubuntu python3.11 apt candidate: %s\n' "$candidate"
  simulation=$(sudo -n apt-get -s install --no-install-recommends --no-upgrade python3.11) || stop 'apt simulation failed'
  printf '%s\n' "$simulation"
  if printf '%s\n' "$simulation" | grep -Eq '^Remv |^Inst [^ ]+ \['; then
    stop 'apt simulation would remove/upgrade an existing package'
  fi
  printf 'Installing python3.11 alongside system python3.10 (no default interpreter change).\n'
  sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y --no-install-recommends --no-upgrade python3.11 || stop 'python installation failed'
fi
/usr/bin/python3.11 -B -c 'import hashlib,sys; assert sys.version_info >= (3,11); assert hasattr(hashlib,"file_digest"); print("PYTHON_3_11_READY="+sys.version.split()[0]+" "+sys.executable)' || stop 'Python 3.11 validation failed'

printf '\n== Creating only one dedicated SSD parent ==\n'
if [[ ! -e "$parent" ]]; then
  sudo -n install -d -m 0700 -o ubuntu -g ubuntu -- "$parent" || stop 'could not create dedicated parent'
fi
[[ -d "$parent" && -w "$parent" && "$(stat -c %U "$parent")" == "ubuntu" && "$(stat -c %a "$parent")" == "700" ]] || stop 'parent verification failed'
printf 'SSD_RUN_PARENT=%s\n' "$parent"
printf 'HOST_PREPARATION=PASS\n'
printf 'MER_CANDIDATE_BUILDS=0 MER_TESTS=0 EXECUTION_MARKERS=0\n'
printf 'EXECUTION_AUTHORIZATION=NOT_GRANTED\n'
