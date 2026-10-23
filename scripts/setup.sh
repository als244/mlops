#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly default_environment="${project_root}/.venv"

python_executable=""
torch_backend="auto"
expert_parallel=""
flashmla=1
cula=1

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--python PATH] [--torch-backend BACKEND]
                          [--expert-parallel quack|te|both] [--no-flashmla] [--no-cula]

Create a complete mlops development environment.  By default the script
creates .venv with Python 3.12 and automatically selects the installed GPU's
PyTorch backend.  --python installs into an existing virtual or Conda
environment instead.  Every optional implementation provider is installed:
cuDNN Frontend, FlashAttention-3, flash-linear-attention, Quack, SonicMoE,
TileLang and TileKernels. On Hopper, cuLA's native KDA forward and fused WY backward
are built at a pinned commit by scripts/setup_cula_sm90.sh; --no-cula skips
that build. Set MLOPS_BUILD_DIR to keep build files outside a quota-limited home.
FlashAttention-3 is installed whatever the local GPU, and mlops activates it
automatically on Hopper GPUs; Quack and SonicMoE serve Hopper and Blackwell.  FlashMLA's
Hopper sparse-attention forward is built from source at a pinned commit by
scripts/setup_flashmla.sh (it has no wheel); mlops' Hopper sparse latent
attention provider uses it with cuDNN Frontend's backward.  --no-flashmla skips
that build.  --expert-parallel also installs
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
    --no-cula)
      cula=0
      shift
      ;;
    --no-flashmla)
      flashmla=0
      shift
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

echo "[1/6] Installing PyTorch 2.13 with backend '${torch_backend}'"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  "torch>=2.13,<2.14"

echo "[2/6] Installing mlops with every implementation provider"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  --editable "${project_root}[providers,dev]"

if [[ "${flashmla}" == 1 ]]; then
  echo "[3/6] Building FlashMLA's Hopper sparse-attention forward"
  "${project_root}/scripts/setup_flashmla.sh" --python "${python_executable}"
else
  echo "[3/6] Skipping FlashMLA; mlops' Hopper sparse attention then keeps its TileLang forward"
fi

if [[ "${cula}" == 1 ]] && "${python_executable}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0) else 1)'; then
  echo "[4/6] Building cuLA's Hopper KDA training provider"
  "${project_root}/scripts/setup_cula_sm90.sh" --python "${python_executable}"
else
  echo "[4/6] Skipping cuLA's SM90 build; fast FLA remains available"
fi

if [[ -n "${expert_parallel}" ]]; then
  echo "[5/6] Installing the '${expert_parallel}' expert-parallel backends"
  # Last of the installs: installing torch restores the cuBLAS its wheel pins,
  # and Transformer Engine needs the newer one this step puts back.
  "${project_root}/scripts/setup_expert_parallel.sh" \
    --python "${python_executable}" \
    --backend "${expert_parallel}"
else
  echo "[5/6] Skipping the expert-parallel backends; --expert-parallel adds them"
fi

echo "[6/6] Verifying PyTorch, the accelerator, and the providers"
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
    ("cudnn.deepseek_sparse_attention", "nvidia-cudnn-frontend"),
    ("fla", "flash-linear-attention"),
    ("quack", "quack-kernels"),
    ("sonicmoe", "sonic-moe"),
    ("tilelang", "tilelang"),
):
    import_module(module_name)
    print(f"{package_name}: {version(package_name)}")

# cuDNN 1.30 supplies the preferred SM90 pooled indexer, including FP32 weights.
import_module("cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm90")
print("cuDNN SM90 pooled indexer: installed")
try:
    import_module("cula.ops.kda.sm90.bwd_wy_dqkg")
except ImportError:
    print("cuLA SM90 KDA: not installed; fast FLA is available")
else:
    try:
        import_module("cula._cudac_sm90")
    except ImportError:
        print("cuLA SM90 KDA: FLA forward + fused WY backward")
    else:
        print("cuLA SM90 KDA: native forward + recomputed cuLA/FLA backward")

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

# FlashMLA is built by scripts/setup_flashmla.sh; without it the Hopper sparse
# latent attention provider keeps its TileLang forward.
try:
    import_module("flash_mla.cuda")
except ImportError as error:
    print(f"flash-mla: not installed, so the TileLang sparse forward is used ({error})")
else:
    print("flash-mla: installed (Hopper sparse-attention forward)")

PY

echo "mlops setup is complete."
if [[ "${python_executable}" == "${default_environment}/bin/python" ]]; then
  echo "Activate it with: source .venv/bin/activate"
fi
