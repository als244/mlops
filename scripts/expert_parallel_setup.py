"""Installation-only discovery/checks for the pinned Transformer Engine stack."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CUDA_TOOLKIT = "13.3.1"
CUBLAS_RUNTIME = "13.6.0.2"
TOOLKIT_COMPONENTS = "nvcc,cccl,cublas,curand,cusolver,cusparse,nvml,nvtx,profiler"
TOOLKIT_HEADERS = (
    "cuda_runtime.h",
    "cub/cub.cuh",
    "cublas_api.h",
    "curand_kernel.h",
    "nvml.h",
    "nvtx3/nvToolsExt.h",
    "cuda_profiler_api.h",
    "cusparse.h",
    "cusolverDn.h",
)


def without_cuda_headers(paths: str) -> str:
    """Keep user include paths, excluding headers from another CUDA toolkit."""
    markers = (
        "cuda.h",
        "cuda_runtime.h",
        "cuda_runtime_api.h",
        "cub/cub.cuh",
        "cuda/std",
    )
    return ":".join(
        value
        for value in paths.split(os.pathsep)
        if value and not any((Path(value) / name).exists() for name in markers)
    )


def cublas_header_version(root: Path) -> tuple[int, int, int] | None:
    """Read cuBLAS headers only from a complete compiler/link toolkit."""
    header = root / "include/cublas_api.h"
    if not os.access(root / "bin/nvcc", os.X_OK) or not header.is_file():
        return None
    if not all(
        any((root / "include" / extra / name).is_file() for extra in ("", "cccl"))
        for name in TOOLKIT_HEADERS
    ):
        return None
    if not any((root / d / "libcublasLt.so").exists() for d in ("lib64", "lib")):
        return None
    source = header.read_text()
    parts = [
        re.search(r"#define CUBLAS_VER_" + name + r"\s+(\d+)", source)
        for name in ("MAJOR", "MINOR", "PATCH")
    ]
    if not all(parts):
        return None
    return tuple(int(part[1]) for part in parts)


def discover_toolkit(candidates: list[Path]) -> Path | None:
    # Match the tested runtime instead of mixing newer headers with older DSOs.
    for root in dict.fromkeys(p.resolve() for p in candidates):
        if cublas_header_version(root) == (13, 6, 0):
            return root
    return None


def prepare_wheel_toolkit(root: Path) -> None:
    """Supply ordinary linker names omitted by NVIDIA's runtime wheels."""
    lib = root / "lib"
    for library in sorted(lib.glob("*.so.*")):
        link = lib / (library.name.split(".so.", 1)[0] + ".so")
        if not link.exists():
            link.symlink_to(library.name)
    if not (root / "lib64").exists():
        (root / "lib64").symlink_to("lib")


def select_toolkit() -> Path:
    cache = Path(sys.prefix) / "share/mlops/cuda" / CUDA_TOOLKIT
    candidates = [
        Path(os.environ[key])
        for key in ("CUDA_HOME", "CUDA_PATH")
        if os.environ.get(key)
    ]
    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates += [Path(sys.prefix), cache / "nvidia/cu13"]
    for parent in (Path("/usr/local"), Path("/opt")):
        candidates += sorted(parent.glob("cuda*"), reverse=True)
    if root := discover_toolkit(candidates):
        return root

    # NVIDIA publishes the compiler, headers and libraries as wheels. Keep the
    # build toolkit separate from Torch's runtime packages; no root access or
    # visible GPU is required on a head/build node.
    print(
        f"No matching CUDA toolkit found; downloading CUDA {CUDA_TOOLKIT} "
        f"into {cache}.",
        file=sys.stderr,
        flush=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache.parent) as temporary:
        target = Path(temporary) / "toolkit"
        if uv := shutil.which("uv"):
            command = [uv, "pip", "install", "--python", sys.executable]
        else:
            command = [sys.executable, "-m", "pip", "install"]
        subprocess.run(
            command
            + [
                "--target",
                str(target),
                f"cuda-toolkit[{TOOLKIT_COMPONENTS}]=={CUDA_TOOLKIT}",
            ],
            check=True,
            stdout=sys.stderr,
        )
        prepare_wheel_toolkit(target / "nvidia/cu13")
        if discover_toolkit([target / "nvidia/cu13"]) is None:
            raise RuntimeError("Downloaded NVIDIA toolkit has an unsupported layout")
        # An incomplete prior download must not be mistaken for a working cache.
        if cache.exists():
            shutil.rmtree(cache)
        target.rename(cache)
    return cache / "nvidia/cu13"


def verify_runtime() -> None:
    # Starting a new interpreter is essential: cuBLAS may already be loaded by
    # Torch, and changing LD_LIBRARY_PATH after that cannot repair its bindings.
    env = dict(os.environ)
    for key in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(key, None)
    code = """
import json
from pathlib import Path
import torch
import transformer_engine.pytorch
import transformer_engine_torch as tex
version = int(tex.get_cublasLt_version())
assert version >= 130600, f'Expected cuBLASLt >= 13.6, loaded {version}'
# Also rejects an old cached TE binary built without grouped-GEMM support.
workspace_bytes = tex.get_grouped_gemm_setup_workspace_size(1)
libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if 'libcublas' in line})
print(json.dumps({'torch': torch.__version__, 'cublaslt': version,
                  'grouped_setup_bytes': workspace_bytes, 'libraries': libraries}, indent=2))
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def verify_compiler() -> None:
    """Catch inherited CUDA include paths before starting TE's long build."""
    root = Path(os.environ["CUDA_HOME"])
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "check.cu"
        source.write_text(
            "#include <cuda_runtime.h>\n#include <cub/cub.cuh>\n"
            "#include <cublasLt.h>\n"
            "#include <nvtx3/nvToolsExt.h>\n#include <nvml.h>\n"
            "#include <curand_kernel.h>\n#include <cuda_profiler_api.h>\n"
            "static_assert(CUBLAS_VERSION >= 130600);\n"
            "__global__ void check(float* x) { x[threadIdx.x] = 1; }\n"
        )
        subprocess.run(
            [
                str(root / "bin/nvcc"),
                "-isystem",
                str(root / "include"),
                "-std=c++17",
                "-arch=sm_90",
                "-c",
                str(source),
                "-o",
                str(source.with_suffix(".o")),
            ],
            check=True,
        )
        import sysconfig

        import torch

        torch_include = Path(torch.__file__).parent / "include"
        source = Path(temporary) / "torch.cpp"
        source.write_text(
            "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n"
        )
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++20",
                "-fsyntax-only",
                "-I" + str(torch_include),
                "-I" + str(torch_include / "torch/csrc/api/include"),
                "-I" + sysconfig.get_path("include"),
                "-I" + str(root / "include"),
                str(source),
            ],
            check=True,
        )
    print("CUDA compiler, headers and cuBLAS preflight passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("toolkit", "verify", "compiler", "runtime-version", "include-path"),
    )
    parser.add_argument("variable", nargs="?")
    args = parser.parse_args()
    if args.action == "toolkit":
        print(select_toolkit())
    elif args.action == "runtime-version":
        print(CUBLAS_RUNTIME)
    elif args.action == "compiler":
        verify_compiler()
    elif args.action == "include-path":
        print(without_cuda_headers(os.environ.get(args.variable, "")))
    else:
        verify_runtime()


if __name__ == "__main__":
    main()
