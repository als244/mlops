"""CUDA discovery must reject incomplete/old toolkits before a long TE build."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def ep_setup():
    path = Path(__file__).resolve().parents[2] / "scripts/expert_parallel_setup.py"
    spec = importlib.util.spec_from_file_location("ep_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_toolkit(root, minor, *, wheel_layout=False):
    for name in ("bin", "include", "lib"):
        (root / name).mkdir(parents=True)
    nvcc = root / "bin/nvcc"
    nvcc.write_text("#!/bin/sh\nexit 0\n")
    nvcc.chmod(0o755)
    (root / "include/cublas_api.h").write_text(
        f"#define CUBLAS_VER_MAJOR 13\n#define CUBLAS_VER_MINOR {minor}\n"
        "#define CUBLAS_VER_PATCH 0\n"
    )
    for name in (
        "cuda_runtime.h",
        "cub/cub.cuh",
        "curand_kernel.h",
        "nvml.h",
        "nvtx3/nvToolsExt.h",
        "cuda_profiler_api.h",
        "cusparse.h",
        "cusolverDn.h",
    ):
        header = root / "include" / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.touch()
    name = "libcublasLt.so.13" if wheel_layout else "libcublasLt.so"
    (root / "lib" / name).touch()
    return root


def test_skips_old_toolkit_and_keeps_compatible_preference(ep_setup, tmp_path):
    old = make_toolkit(tmp_path / "old", 4)
    first = make_toolkit(tmp_path / "selected", 6)
    second = make_toolkit(tmp_path / "system", 6)
    assert ep_setup.discover_toolkit([old, first, second]) == first


def test_incomplete_or_unmatched_toolkit_is_not_selected(ep_setup, tmp_path):
    root = make_toolkit(tmp_path / "incomplete", 6)
    (root / "bin/nvcc").unlink()
    newer = make_toolkit(tmp_path / "different-runtime", 8)
    assert ep_setup.discover_toolkit([root, newer]) is None


def test_wheel_link_names_preserve_payloads_and_are_reusable(ep_setup, tmp_path):
    root = make_toolkit(tmp_path / "wheel", 6, wheel_layout=True)
    ep_setup.prepare_wheel_toolkit(root)
    ep_setup.prepare_wheel_toolkit(root)
    assert ep_setup.discover_toolkit([root]) == root
    assert (root / "lib64/libcublasLt.so").resolve() == root / "lib/libcublasLt.so.13"


def test_malformed_header_is_not_selected(ep_setup, tmp_path):
    root = make_toolkit(tmp_path / "broken", 6)
    (root / "include/cublas_api.h").write_text("unrelated header")
    assert ep_setup.discover_toolkit([root]) is None


def test_old_cuda_headers_are_removed_but_other_includes_are_preserved(
    ep_setup, tmp_path
):
    cuda = tmp_path / "cuda/include"
    cuda.mkdir(parents=True)
    (cuda / "cuda_runtime.h").touch()
    cccl = tmp_path / "cccl"
    (cccl / "cuda/std").mkdir(parents=True)
    project = tmp_path / "project/include"
    project.mkdir(parents=True)
    paths = f"{cuda}:{project}:{cccl}"
    assert ep_setup.without_cuda_headers(paths) == str(project)
