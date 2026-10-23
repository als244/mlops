#!/usr/bin/env bash
# Install cuLA's pinned Hopper training kernels beside MLOps' tested CuTe stack.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable=python
build_root="${MLOPS_BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/mlops/build}"
source_dir="${CULA_SOURCE:-}"
mode=training
revision=bee137a998f822e8fa84f26ffd7f7103a3bbdcba
while (($#)); do
    case "$1" in
        --python) python_executable="${2:?Specify a Python executable}"; shift 2 ;;
        --build-dir) build_root="${2:?Specify a build directory}"; shift 2 ;;
        --source) source_dir="${2:?Specify a pinned cuLA checkout}"; shift 2 ;;
        training|wy-only) mode="$1"; shift ;;
        --help|-h)
            echo 'Usage: setup_cula_sm90.sh [--python PATH] [--build-dir DIR] [--source DIR] [training|wy-only]'
            echo "Builds cuLA $revision for SM90; wy-only skips the native forward extension."
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
# Upstream's CuTe <4.7 constraint is for private SM100 bindings. Our SM90
# adapters import the public CuTe kernel directly and are validated at 4.7.1.
"$python_executable" -c 'import cutlass, torch, fla; assert cutlass.__version__ == "4.7.1", cutlass.__version__'
install_packages() {
    if command -v uv >/dev/null; then
        uv pip install --python "$python_executable" "$@"
    else
        "$python_executable" -m pip install "$@"
    fi
}
if [[ -z "$source_dir" ]]; then
    source_dir="$build_root/cuLA-$revision"
    mkdir -p "$build_root"
    if [[ ! -d "$source_dir/.git" ]]; then
        git clone --quiet https://github.com/inclusionAI/cuLA.git "$source_dir"
    fi
    git -C "$source_dir" cat-file -e "$revision^{commit}" 2>/dev/null ||
        git -C "$source_dir" fetch --quiet origin "$revision"
    git -C "$source_dir" checkout --quiet "$revision"
fi
actual=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
    echo "Expected cuLA $revision, found $actual" >&2
    exit 1
fi
# cuLA's setup queries nvcc even when all native extensions are disabled.
cuda_root=$("$python_executable" - <<'PYCUDA'
from pathlib import Path
import sys
from torch.utils.cpp_extension import CUDA_HOME
candidates = [Path(CUDA_HOME)] if CUDA_HOME else []
# Reuse a toolkit already downloaded by setup_flashmla.sh, without requiring
# that subprocess's CUDA_HOME export to survive in its parent shell.
candidates += sorted((Path(sys.prefix) / "share/mlops/cuda").glob("*/nvidia/cu13"), reverse=True)
root = next((p for p in candidates if (p / "bin/nvcc").is_file()), None)
if root is None:
    raise SystemExit("cuLA setup requires nvcc; set CUDA_HOME to a matching CUDA toolkit")
print(root)
PYCUDA
)
export CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root" PATH="$cuda_root/bin:$PATH"
# Torch's CUDA wheels supply cuBLAS headers on nodes with a compiler-only
# system toolkit. Include these without forcing Transformer Engine's newer
# cuBLAS runtime or downloading a second complete toolkit.
cuda_headers=$("$python_executable" - <<'PYHEADERS'
from pathlib import Path
import torch
root = Path(torch.__file__).resolve().parent.parent / "nvidia"
print(":".join(str(p) for p in sorted(root.glob("*/include"))))
PYHEADERS
)
export CPATH="$cuda_root/include:$cuda_root/include/cccl:$cuda_headers${CPATH:+:$CPATH}"
native_disabled=1
if [[ "$mode" == training ]]; then
    native_disabled=0
    git -C "$source_dir" submodule update --init --recursive --quiet csrc/cutlass
    install_packages ninja packaging setuptools wheel
fi
export CULA_BUILD_ALL_ARCHS=0
export CULA_DISABLE_SM90="$native_disabled" CULA_DISABLE_SM100=1 CULA_DISABLE_SM103=1
export MAX_JOBS="${MAX_JOBS:-2}" NVCC_THREADS="${NVCC_THREADS:-4}"
install_packages --no-build-isolation --no-deps --force-reinstall "$source_dir"
"$python_executable" -c 'from cula.ops.kda.sm90.bwd_wy_dqkg import chunk_kda_bwd_wy_dqkg_fused'
if [[ "$mode" == training ]]; then
    "$python_executable" -c 'import torch; import cula._cudac_sm90'
fi
echo "Installed cuLA ${revision:0:7} for SM90 ($mode)."
