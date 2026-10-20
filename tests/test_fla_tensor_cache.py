"""The FLA adapters turn FLA's identity-keyed tensor cache off."""

import importlib
import sys

import pytest
import torch


def test_loading_the_fla_adapters_turns_flas_tensor_cache_off():
    pytest.importorskip("fla")
    importlib.import_module("mlops.providers.fla")
    from fla.ops.utils.index import prepare_chunk_offsets

    assert sys.modules["fla.utils"].FLA_DISABLE_TENSOR_CACHE
    cumulative = torch.tensor([0, 5, 12, 16], dtype=torch.int64)
    first = prepare_chunk_offsets(cumulative, 4)
    second = prepare_chunk_offsets(cumulative, 4)
    # Cached, the same arguments would give back the same tensor object.
    assert first is not second
    assert torch.equal(first, second)
