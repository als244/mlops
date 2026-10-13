#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly default_environment="${project_root}/.venv"

python_executable=""
torch_backend="auto"
expert_parallel=""

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--python PATH] [--torch-backend BACKEND]
                          [--expert-parallel quack|te|both]

Create a complete mlops development environment.  By default the script
creates .venv with Python 3.12 and automatically selects the installed GPU's
PyTorch backend.  --python installs into an existing virtual or Conda
environment instead.  Every optional implementation provider is installed:
FlashAttention-3, flash-linear-attention, Quack, SonicMoE, and tilelang.
FlashAttention-3 is installed whatever the local GPU, and mlops activates it
automatically on Hopper GPUs; Quack and SonicMoE serve Hopper and Blackwell.  --expert-parallel also installs
the optional expert-parallel backends -- Quack, Transformer Engine, or both,
each with MoonEP -- through scripts/setup_expert_parallel.sh, which also adds
them to an existing environment; see docs/EXPERT_PARALLEL.md for their
Hopper dependencies.
EOF
}

while (($#)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a path" >&2; exit 2; }
      python_executable="$2"
      shift 2
      ;;
    --torch-backend)
      [[ $# -ge 2 ]] || { echo "--torch-backend requires a value" >&2; exit 2; }
      torch_backend="$2"
      shift 2
      ;;
    --expert-parallel)
      [[ $# -ge 2 ]] || { echo "--expert-parallel requires quack, te, or both" >&2; exit 2; }
      case "$2" in
        quack|te|both) expert_parallel="$2" ;;
        *) echo "--expert-parallel must be quack, te, or both" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

uv_executable="$(command -v uv || true)"
bootstrap_dir=""
if [[ -z "${uv_executable}" ]]; then
  bootstrap_python=""
  bootstrap_python="$(command -v python3 || command -v python || true)"
  if [[ -z "${bootstrap_python}" ]]; then
    echo "setup requires uv or a Python interpreter with venv support" >&2
    exit 1
  fi

  bootstrap_dir="$(mktemp -d)"
  "${bootstrap_python}" -m venv "${bootstrap_dir}"
  "${bootstrap_dir}/bin/python" -m pip install --quiet "uv>=0.9,<1" >&2
  uv_executable="${bootstrap_dir}/bin/uv"
  trap 'rm -rf -- "${bootstrap_dir}"' EXIT
fi
readonly uv_executable

if [[ -z "${python_executable}" ]]; then
  "${uv_executable}" venv --allow-existing --python 3.12 "${default_environment}"
  python_executable="${default_environment}/bin/python"
fi

if [[ ! -x "${python_executable}" ]]; then
  echo "Python interpreter is not executable: ${python_executable}" >&2
  exit 1
fi

echo "[1/4] Installing PyTorch 2.13 with backend '${torch_backend}'"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  "torch>=2.13,<2.14"

echo "[2/4] Installing mlops with every implementation provider"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  --editable "${project_root}[providers,dev]"

if [[ -n "${expert_parallel}" ]]; then
  echo "[3/4] Installing the '${expert_parallel}' expert-parallel backends"
  # Last of the installs: installing torch restores the cuBLAS its wheel pins,
  # and Transformer Engine needs the newer one this step puts back.
  "${project_root}/scripts/setup_expert_parallel.sh" \
    --python "${python_executable}" \
    --backend "${expert_parallel}"
else
  echo "[3/4] Skipping the expert-parallel backends; --expert-parallel adds them"
fi

echo "[4/4] Verifying PyTorch, the accelerator, and the providers"
"${python_executable}" - <<'PY'
from importlib import import_module
from importlib.metadata import version

import torch

release = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if release != (2, 13):
    raise RuntimeError(f"expected PyTorch 2.13, found {torch.__version__}")
if torch.version.cuda is None:
    raise RuntimeError(f"expected a CUDA-enabled PyTorch build, found {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch has a CUDA backend but cannot access a GPU; check the NVIDIA driver"
    )

print(f"PyTorch: {torch.__version__}")
print(f"CUDA backend: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

import mlops  # noqa: F401  -- registers every provider adapter

for module_name, package_name in (
    ("fla", "flash-linear-attention"),
    ("quack", "quack-kernels"),
    ("sonicmoe", "sonic-moe"),
    ("tilelang", "tilelang"),
):
    import_module(module_name)
    print(f"{package_name}: {version(package_name)}")

# Importing FA3 loads its CUDA extension without running a kernel, so on any
# GPU it shows whether Hopper could use this install. The wheel is the CUDA 13
# build, which a CUDA 12 torch cannot load; mlops then keeps PyTorch's own
# attention kernels, a working install rather than a broken one.
try:
    import_module("flash_attn_interface")
except ImportError as error:
    print(f"flash-attn-3: not loadable beside this torch, so unused ({error})")
else:
    print(f"flash-attn-3: {version('flash-attn-3')}")

PY

echo "mlops setup is complete."
if [[ "${python_executable}" == "${default_environment}/bin/python" ]]; then
  echo "Activate it with: source .venv/bin/activate"
fi
