#!/usr/bin/env bash
#
# setup_cuda_env.sh — export the environment needed to build the engine
# with `--features cuda`.
#
# Why this exists:
#   cudarc (pulled in transitively by `candle-core/cuda`) detects the
#   installed CUDA toolkit version by running `nvcc --version` from the
#   PATH, and locates the CUDA libraries via the CUDA_HOME / CUDA_PATH /
#   CUDA_ROOT / CUDA_TOOLKIT_ROOT_DIR environment variables.
#
#   On GCP Deep Learning VMs (Ubuntu 24.04, CUDA 12.x) the toolkit is
#   installed under /usr/local/cuda but `nvcc` is *not* on the PATH, so
#   the cudarc build script fails with:
#
#       `nvcc --version` failed.
#       Err(Os { code: 2, kind: NotFound, message: "No such file or directory" })
#
#   Sourcing this script puts `<cuda>/bin` on the PATH and exports the
#   CUDA_* variables so the build can find both nvcc and the libraries.
#
# Usage (note the leading `source` — it must run in your current shell):
#
#       source scripts/setup_cuda_env.sh
#       cd rust-engine
#       cargo build --release --features "cuda,avx512,tokenizer"
#
# You may override the toolkit location by exporting CUDA_HOME first:
#
#       CUDA_HOME=/usr/local/cuda-12.9 source scripts/setup_cuda_env.sh
#

# Pick the toolkit root: honour an existing CUDA_HOME/CUDA_PATH, then
# fall back to the canonical symlink, then to the newest /usr/local/cuda-*.
_mer_cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
if [ -z "${_mer_cuda_home}" ]; then
    if [ -d /usr/local/cuda ]; then
        _mer_cuda_home=/usr/local/cuda
    else
        # Newest versioned install, if any (e.g. /usr/local/cuda-12.9).
        _mer_cuda_home="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1)"
    fi
fi

if [ -z "${_mer_cuda_home}" ] || [ ! -d "${_mer_cuda_home}" ]; then
    echo "setup_cuda_env.sh: could not find a CUDA toolkit under /usr/local." >&2
    echo "  Install CUDA 12.x or export CUDA_HOME to the toolkit root and re-run." >&2
    unset _mer_cuda_home
    return 1 2>/dev/null || exit 1
fi

if [ ! -x "${_mer_cuda_home}/bin/nvcc" ]; then
    echo "setup_cuda_env.sh: nvcc not found at ${_mer_cuda_home}/bin/nvcc." >&2
    echo "  Make sure the full CUDA toolkit (not just the runtime) is installed." >&2
    unset _mer_cuda_home
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="${_mer_cuda_home}"
export CUDA_PATH="${_mer_cuda_home}"
export CUDA_ROOT="${_mer_cuda_home}"
export CUDA_TOOLKIT_ROOT_DIR="${_mer_cuda_home}"

# Put nvcc on the PATH so cudarc's `nvcc --version` version probe works.
case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) : ;;                       # already present
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
esac

# Help the dynamic linker find libcudart/libcublas at run time.
case ":${LD_LIBRARY_PATH-}:" in
    *":${CUDA_HOME}/lib64:"*) : ;;
    *) export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

echo "CUDA environment ready:"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  nvcc:     $(command -v nvcc)"
"${CUDA_HOME}/bin/nvcc" --version | grep -i release || true

unset _mer_cuda_home
