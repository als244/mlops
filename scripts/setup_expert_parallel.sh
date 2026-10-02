#!/usr/bin/env bash
# Optional Hopper expert-parallel backends; leave the default install unchanged.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable=python
backend=quack
while (($#)); do
 case "$1" in
  --python) python_executable="${2:?Specify Python executable}"; shift 2 ;;
  --backend) backend="${2:?Specify quack, te, or both}"; shift 2 ;;
  --help|-h) echo 'Usage: setup_expert_parallel.sh [--python PATH] [--backend quack|te|both]'; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
 esac
done
case "$backend" in
 quack) extras=ep-quack ;;
 te) extras=ep-te ;;
 both) extras=ep-quack,ep-te ;;
 *) echo '--backend must be quack, te, or both' >&2; exit 2 ;;
esac
# Use the caller's Torch environment. Optional CUDA tools/runtime are managed below.
"$python_executable" -c 'import torch; assert torch.__version__.split("+")[0].startswith("2.13.") and (torch.version.cuda or "").startswith("13."), "This backend stack requires PyTorch 2.13 with CUDA 13"'
install_packages() {
 if command -v uv >/dev/null; then
  uv pip install --python "$python_executable" "$@"
 else
  "$python_executable" -m pip install "$@"
 fi
}
if [[ "$backend" == te || "$backend" == both ]]; then
 # Select a complete, compatible toolkit, downloading NVIDIA's wheels if needed.
 setup_helper="$project_root/scripts/expert_parallel_setup.py"
 te_cuda_root="$("$python_executable" "$setup_helper" toolkit)"
 echo "Using CUDA toolkit: $te_cuda_root"
 export CUDA_HOME="$te_cuda_root" CUDA_PATH="$te_cuda_root" PATH="$te_cuda_root/bin:$PATH"
 export LD_LIBRARY_PATH="$te_cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
 # These must exist before TE runs its non-isolated build/metadata hooks.
 install_packages 'nvidia-cudnn-frontend==1.30.0' 'pybind11>=2.13' cmake ninja packaging setuptools wheel
 export NVTE_FRAMEWORK=pytorch
 # MoonEP handles communication. TE's separate NCCL EP backend is unnecessary
 # and depends on newer NCCL headers than many Torch installations provide.
 export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
 export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"
 export MAX_JOBS="${MAX_JOBS:-8}" NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-2}"
 # CMake marks toolkit headers as system headers. Merely prepending CPATH does
 # not override an older non-system CUDA include; remove those conflicts first.
 for include_variable in CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
  include_value="$("$python_executable" "$setup_helper" include-path "$include_variable")"
  if [[ -n "$include_value" ]]; then
   export "$include_variable=$include_value"
  else
   unset "$include_variable"
  fi
 done
 # Torch's CUDA wheels ship cuDNN/NCCL headers outside the toolkit prefix.
 # TE's CMake build finds them, but its Torch C++ extension needs the compiler
 # search path too. Discover them in the selected interpreter, not a fixed env.
 te_include_dirs="$("$python_executable" - <<'PY'
from pathlib import Path
import site
print(":".join(str(p) for root in site.getsitepackages()
               for name in ("cudnn", "nccl")
               if (p := Path(root) / "nvidia" / name / "include").is_dir()))
PY
)"
 if [[ -n "$te_include_dirs" ]]; then
  export CPATH="$te_cuda_root/include:$te_include_dirs${CPATH:+:$CPATH}"
 else
  export CPATH="$te_cuda_root/include${CPATH:+:$CPATH}"
 fi
 "$python_executable" "$setup_helper" compiler
fi
install_packages --no-build-isolation -e "$project_root[$extras]"
install_packages --no-deps --no-build-isolation \
 'moonep @ git+https://github.com/moonshotAI/moonep.git@2bd860b4dd083df62b79d5e916fca71ec5742228'
# MoonEP's declared DSL==4.4.2 conflicts with the tested DSL==4.7.1 stack.
# MLOps applies an isolated planner-only PTXAS level-2 compatibility fix at
# runtime. Installed MoonEP source stays untouched; Quack keeps normal settings.
echo "Installed $backend expert-parallel dependencies with Cutlass DSL 4.7.1."
echo 'MoonEP metadata still declares DSL 4.4.2; pip check reports that known mismatch.'
echo 'MLOps applies the MoonEP planner compiler compatibility fix for DSL 4.7.1.'
if [[ "$backend" == te || "$backend" == both ]]; then
 # Resolve Torch's ordinary dependencies first, then install the newer cuBLAS
 # needed by TE. Do not change Torch or the system CUDA/driver installation.
 cublas_version="$("$python_executable" "$setup_helper" runtime-version)"
 install_packages --no-deps "nvidia-cublas==$cublas_version"
 if ! "$python_executable" "$setup_helper" verify; then
  echo 'Rebuilding the pinned TE extension for the detected toolkit.'
  te_requirement="$("$python_executable" - "$project_root/pyproject.toml" <<'PYTE'
import sys
import tomllib
from pathlib import Path
extras = tomllib.loads(Path(sys.argv[1]).read_text())["project"]["optional-dependencies"]
print(next(item for item in extras["ep-te"] if item.startswith("transformer_engine")))
PYTE
)"
  if command -v uv >/dev/null; then
   uv pip install --python "$python_executable" --no-cache --reinstall \
    --no-deps --no-build-isolation "$te_requirement"
  else
   "$python_executable" -m pip install --no-cache-dir --force-reinstall \
    --no-deps --no-build-isolation "$te_requirement"
  fi
  "$python_executable" "$setup_helper" verify
 fi
 echo "TE runtime verified without CUDA/library-path overrides (cuBLAS $cublas_version)."
 echo 'Torch CUDA-wheel metadata may pin an older cuBLAS; this optional stack overrides that pin.'
fi
