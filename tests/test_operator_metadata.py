"""Compiler facts are usable without optional kernels or any execution engine."""

import subprocess
import sys

import pytest
import torch

from mlops.operator_metadata import (
    ChunkLocalOutput,
    mhc_chunking,
    operator_chunking,
    operator_preparations,
    sonicmoe_chunking,
    sonicmoe_gradient_accumulation,
    sonicmoe_preparation,
)


def test_declarations_do_not_load_external_packages_or_initialize_cuda():
    script = """
import importlib.abc, sys, torch, mlops
class NoExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root not in sys.stdlib_module_names and root not in {'torch', 'mlops'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, NoExternal())
from mlops import operator_metadata
for factory in (operator_metadata.operator_chunking, operator_metadata.sonicmoe_chunking,
                operator_metadata.mhc_chunking, operator_metadata.sonicmoe_gradient_accumulation,
                operator_metadata.fla_input_layouts, operator_metadata.operator_preparations,
                operator_metadata.sonicmoe_preparation):
    assert factory()
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_precision_domains_and_local_state_are_explicit():
    bf16 = {c.target: c for c in sonicmoe_chunking()}
    fp8 = sonicmoe_chunking(compute_precision="fp8_block")
    assert len(bf16) == 7 and len(fp8) == 5
    for c in fp8:
        assert "accumulate" not in c.target
        assert c.output_axes == bf16[c.target].output_axes
        assert c.reductions == bf16[c.target].reductions
        assert c.version != bf16[c.target].version
    metadata = bf16["mlops.sonicmoe_routing_metadata.default"]
    assert metadata.output_axes == (
        None,
        ChunkLocalOutput(0),
        ChunkLocalOutput(0),
        ChunkLocalOutput(0),
    )
    assert metadata.reductions == (0,)
    assert len(sonicmoe_gradient_accumulation()) == 2
    with pytest.raises(ValueError, match="frozen fp8_block"):
        sonicmoe_chunking(compute_precision="fp8_current")
    for factory in (operator_chunking, mhc_chunking):
        with pytest.raises(ValueError, match="positive"):
            factory(batch_size=0)


def test_preparation_options_are_owned_and_declarations_preserve_dispatch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description queried a device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    explicit = sonicmoe_preparation(cache_root="/tmp/build-only")
    assert explicit.factory == "mlops.preparation.sonic:factory"
    assert dict(explicit.options) == {"cache_root": "/tmp/build-only"}
    with pytest.raises(TypeError):
        explicit.options["cache_root"] = "/tmp/changed"
    assert len(operator_preparations()) == 2
    assert operator_preparations(dtype=torch.float32) == ()
    assert len(mhc_chunking()) == 4
