#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly default_environment="${project_root}/.venv"

python_executable=""
torch_backend="auto"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--python PATH] [--torch-backend BACKEND]

Create a complete mlops development environment.  By default the script
creates .venv with Python 3.12 and automatically selects the installed GPU's
PyTorch backend.  --python installs into an existing virtual or Conda
environment instead.  Every optional implementation provider is installed:
flash-linear-attention, liger-kernel, scattermoe, and tilelang.
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

echo "[1/3] Installing PyTorch 2.13 with backend '${torch_backend}'"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  "torch>=2.13,<2.14"

echo "[2/3] Installing mlops with every implementation provider"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  --editable "${project_root}[providers,dev]"

echo "[3/3] Verifying PyTorch, the accelerator, and the providers"
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
    ("liger_kernel", "liger-kernel"),
    ("scattermoe", "scattermoe"),
    ("tilelang", "tilelang"),
):
    import_module(module_name)
    print(f"{package_name}: {version(package_name)}")
PY

echo "mlops setup is complete."
if [[ "${python_executable}" == "${default_environment}/bin/python" ]]; then
  echo "Activate it with: source .venv/bin/activate"
fi
