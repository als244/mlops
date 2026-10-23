#!/usr/bin/env bash
# Pinned DeepGEMM host extension and JIT headers, built beside the caller's torch.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable=python
build_root="${MLOPS_BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/mlops/build}"
revision=5081f8513af57e61696de4f4f04b0e707ffb503e
force=0
while (($#)); do
  case "$1" in
    --python) python_executable="${2:?Specify Python executable}"; shift 2 ;;
    --build-dir) build_root="${2:?Specify a build directory}"; shift 2 ;;
    --force) force=1; shift ;;
    --help|-h)
      echo 'Usage: setup_deepgemm.sh [--python PATH] [--build-dir DIR] [--force]'
      echo 'Requires a CUDA >=12.9 toolkit and a C++20 compiler with <format> (e.g. GCC 13+). Set CXX explicitly if needed.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
source_dir="$build_root/DeepGEMM-$revision"
if [[ "$force" == 0 ]] && "$python_executable" - "$revision" <<'PY'
import sys
try:
    import deep_gemm
    from importlib.metadata import version
    assert sys.argv[1][:7] in version("deep_gemm")
    assert callable(deep_gemm.tf32_hc_prenorm_gemm)
except (ImportError, AssertionError, AttributeError):
    raise SystemExit(1)
PY
then
  echo "DeepGEMM ${revision:0:7} is already installed; --force rebuilds it."
  exit 0
fi
setup_helper="$project_root/scripts/expert_parallel_setup.py"
cuda_root="$("$python_executable" "$setup_helper" toolkit)"
export CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root" PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for include_variable in CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
  include_value="$("$python_executable" "$setup_helper" include-path "$include_variable")"
  if [[ -n "$include_value" ]]; then export "$include_variable=$include_value"; else unset "$include_variable"; fi
done
export CPATH="$cuda_root/include:$cuda_root/include/cccl${CPATH:+:$CPATH}"
"$python_executable" - <<'PY'
import os, subprocess, tempfile
from pathlib import Path
compiler=os.environ.get('CXX','c++')
with tempfile.TemporaryDirectory(prefix='mlops-deepgemm-compiler-') as directory:
    source=Path(directory)/'format.cpp'
    source.write_text('#include <format>\nint main() { return std::format("{}", 1) != "1"; }\n')
    subprocess.run([compiler,'-std=c++20',str(source),'-o',str(Path(directory)/'format')],check=True)
PY
install_packages() {
  if command -v uv >/dev/null; then uv pip install --python "$python_executable" "$@";
  else "$python_executable" -m pip install "$@"; fi
}
# DeepJIT needs elfutils headers for symbolized exceptions. Use system headers,
# an explicit prefix, or an isolated Conda build-dependency environment.
elfutils_root="${MLOPS_ELFUTILS_ROOT:-}"
if [[ -n "$elfutils_root" ]]; then
  export CPATH="$elfutils_root/include:$CPATH"
fi
if ! printf '#include <elfutils/libdwfl.h>\n' | "${CXX:-c++}" -x c++ -E - >/dev/null 2>&1; then
  if [[ -z "$elfutils_root" ]] && command -v conda >/dev/null; then
    elfutils_root="$build_root/deepgemm-build-deps"
    if [[ ! -f "$elfutils_root/include/elfutils/libdwfl.h" ]]; then
      conda create --yes --prefix "$elfutils_root" --override-channels -c conda-forge 'elfutils>=0.190,<0.200'
    fi
    export CPATH="$elfutils_root/include:$CPATH"
  else
    echo 'Install elfutils-devel/libdw-dev and libelf development headers, or set MLOPS_ELFUTILS_ROOT to their prefix.' >&2
    exit 1
  fi
fi
printf '#include <elfutils/libdwfl.h>\n' | "${CXX:-c++}" -x c++ -E - >/dev/null
install_packages ninja packaging setuptools wheel
mkdir -p "$build_root"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --quiet --filter=blob:none https://github.com/deepseek-ai/DeepGEMM.git "$source_dir"
fi
git -C "$source_dir" fetch --quiet origin "$revision"
git -C "$source_dir" checkout --quiet "$revision"
git -C "$source_dir" submodule update --init --recursive --quiet
export DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=1 MAX_JOBS="${MAX_JOBS:-4}"
install_packages --no-build-isolation --no-deps "$source_dir"
"$python_executable" - <<'PY'
import deep_gemm
assert callable(deep_gemm.tf32_hc_prenorm_gemm)
print(f'DeepGEMM {deep_gemm.__version__}: HC prenorm API installed')
print('Kernel compilation and numerical tests run separately on SM90/SM100.')
PY
