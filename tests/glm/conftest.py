"""Capability gates for the optional GLM accelerator tests."""

from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).parent
_KDA_FILES = {"test_kda.py", "test_block.py"}
_SPARSE_FILES = {"test_attention.py", "test_lora.py", "test_mla.py"}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.parent != _ROOT or item.get_closest_marker("gpu") is None:
            continue
        reason = None
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            reason = "GLM GPU comparisons require CUDA with BF16 support (SM80+)"
        name = item.path.name
        needs_kda = name in _KDA_FILES or (
            name == "test_contracts.py" and "kda" in item.name
        )
        needs_sparse = name in _SPARSE_FILES or (
            name == "test_contracts.py" and "sparse_attention" in item.name
        )
        if needs_kda and find_spec("fla") is None:
            reason = "KDA checks require the mlops[glm] optional dependencies"
        if needs_sparse and find_spec("tilelang") is None:
            reason = "Sparse MLA checks require the mlops[glm] optional dependencies"
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
