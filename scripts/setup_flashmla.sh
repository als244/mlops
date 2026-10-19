#!/usr/bin/env bash
# FlashMLA's sparse-attention forward for Hopper, built from source at a pinned
# commit and installed into the caller's Torch environment.
#
# DeepSeek's FlashMLA `main` dropped Hopper on 2026-09-30; its README sends
# Hopper users to the last `main` commit that carries the sm90a sparse prefill
# kernel, ba89a34 (2026-09-15), which is what mlops' Hopper sparse latent
# attention provider uses for its forward (cuDNN Frontend supplies the
# backward). There is no wheel, so this builds that commit with the same
# complete CUDA toolkit selection the
# Transformer Engine step uses: a toolkit on the machine whose headers and
# libraries are all present, else NVIDIA's toolkit wheels downloaded into the
# environment. A partial toolkit (one without cuBLAS, cuRAND or CCCL headers,
# as on nodes that install only the compiler) is what makes the build fail
# otherwise.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable=python
build_root="${MLOPS_BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/mlops/build}"
archs="${FLASHMLA_ARCHS:-90a}"
force=0
flashmla_repository="https://github.com/deepseek-ai/FlashMLA.git"
flashmla_commit="ba89a3466e9470ad08ab39738d4e7bb66989e1e7"
while (($#)); do
 case "$1" in
  --python) python_executable="${2:?Specify Python executable}"; shift 2 ;;
  --build-dir) build_root="${2:?Specify a build directory}"; shift 2 ;;
  --archs) archs="${2:?Specify 90a, 100f, or 90a,100f}"; shift 2 ;;
  --force) force=1; shift ;;
  --help|-h)
   echo 'Usage: setup_flashmla.sh [--python PATH] [--build-dir DIR] [--archs 90a|100f|90a,100f] [--force]'
   echo "Builds FlashMLA main @ ${flashmla_commit:0:7} (default archs: 90a) and installs it."
   exit 0 ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
 esac
done
"$python_executable" -c 'import torch; assert torch.__version__.split("+")[0].startswith("2.13.") and (torch.version.cuda or "").startswith("13."), "FlashMLA here requires PyTorch 2.13 with CUDA 13"'
install_packages() {
 if command -v uv >/dev/null; then
  uv pip install --python "$python_executable" "$@"
 else
  "$python_executable" -m pip install "$@"
 fi
}
source_dir="$build_root/FlashMLA"
marker="$source_dir/.mlops-installed-$flashmla_commit-$archs"
if [[ "$force" == 0 && -f "$marker" ]] && "$python_executable" -c 'import flash_mla.cuda; from flash_mla import flash_mla_sparse_fwd' 2>/dev/null; then
 echo "FlashMLA ${flashmla_commit:0:7} ($archs) is already installed; --force rebuilds it."
 exit 0
fi

# 1. A complete toolkit, found or downloaded; the compiler check catches stray
#    include paths from another CUDA before the long build starts.
setup_helper="$project_root/scripts/expert_parallel_setup.py"
cuda_root="$("$python_executable" "$setup_helper" toolkit)"
echo "Using CUDA toolkit: $cuda_root"
export CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root" PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for include_variable in CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
 include_value="$("$python_executable" "$setup_helper" include-path "$include_variable")"
 if [[ -n "$include_value" ]]; then
  export "$include_variable=$include_value"
 else
  unset "$include_variable"
 fi
done
# The host compiler sees the toolkit's CCCL headers only through this path;
# nvcc adds it by itself. The C++ API file includes CUTLASS, which needs them.
export CPATH="$cuda_root/include:$cuda_root/include/cccl${CPATH:+:$CPATH}"
install_packages ninja packaging setuptools wheel
"$python_executable" "$setup_helper" compiler

# 2. The pinned source, with its CUTLASS submodule.
mkdir -p "$build_root"
if [[ ! -d "$source_dir/.git" ]]; then
 git clone --quiet "$flashmla_repository" "$source_dir"
fi
git -C "$source_dir" fetch --quiet origin "$flashmla_commit" 2>/dev/null || git -C "$source_dir" fetch --quiet origin
git -C "$source_dir" checkout --quiet "$flashmla_commit"
git -C "$source_dir" submodule update --init --recursive --quiet csrc/cutlass

# 3. The build. Only the requested architectures are compiled: sm100f needs
#    nvcc 12.9 or newer and doubles the build, and the Hopper kernel is the one
#    mlops dispatches to.
export FLASH_MLA_DISABLE_SM100=1 FLASH_MLA_DISABLE_SM90=1
[[ ",$archs," == *",90a,"* ]] && unset FLASH_MLA_DISABLE_SM90
[[ ",$archs," == *",100f,"* ]] && unset FLASH_MLA_DISABLE_SM100
export MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-4}"
rm -f "$source_dir"/.mlops-installed-*
install_packages --no-build-isolation --no-deps "$source_dir"

# 4. Verify in a fresh interpreter: the extension loads beside this torch, and
#    the sparse forward entry point mlops uses exists. A Hopper GPU, when
#    visible, runs one tiny call.
"$python_executable" - <<'PY'
import torch
import flash_mla.cuda  # noqa: F401  -- the CUDA extension
from flash_mla import flash_mla_sparse_fwd
print(f"FlashMLA imports beside torch {torch.__version__} (CUDA {torch.version.cuda})")
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9:
    q = torch.randn(64, 64, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, 1, 576, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(64, device="cuda", dtype=torch.int32)[:, None, None].expand(64, 1, 64).contiguous()
    ids = torch.where(ids <= torch.arange(64, device="cuda", dtype=torch.int32)[:, None, None], ids, -1)
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, ids, 576 ** -0.5, d_v=512)
    torch.cuda.synchronize()
    assert out.shape == (64, 64, 512) and torch.isfinite(out.float()).all(), "FlashMLA smoke test produced non-finite output"
    print("FlashMLA sparse forward ran on", torch.cuda.get_device_name(0))
else:
    print("No Hopper GPU visible here; FlashMLA is installed for Hopper nodes")
PY
touch "$marker"
echo "Installed FlashMLA ${flashmla_commit:0:7} for $archs."
