"""Shared local expert math must not depend on an EP transport installation."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "old_package,names",
    [
        (
            "experts",
            ("bf16", "block_fp8_gemm", "fp8", "policy", "swiglu_limit", "wgrad"),
        ),
        (
            "kernels",
            ("pointwise", "quantize_rows", "quantize_transpose", "swiglu_rows"),
        ),
    ],
)
def test_legacy_paths_alias_the_shared_module(old_package, names):
    pytest.importorskip("quack")
    for name in names:
        shared = importlib.import_module(f"mlops.kernels.quack_moe.{name}")
        old = importlib.import_module(
            f"mlops.expert_parallel.quack.{old_package}.{name}"
        )
        assert old is shared
    kernel = importlib.import_module("mlops.kernels.quack_moe.block_fp8_gemm")
    assert kernel.GemmSm90BlockFP8T.__module__.startswith("mlops.kernels.quack_moe.")


def test_local_math_imports_without_moonep():
    pytest.importorskip("quack")
    script = """
import importlib.abc
import sys
class NoTransport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'moonep' or fullname.startswith('moonep.'):
            raise ImportError('Transport is intentionally unavailable')
sys.meta_path.insert(0, NoTransport())
from mlops.kernels.quack_moe import bf16, block_fp8_gemm, fp8
assert not any(n == 'moonep' or n.startswith('moonep.') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=90)
