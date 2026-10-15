"""CPU ownership checks for shared TE communication scratch, not GPU numerics."""

import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mlops.expert_parallel.transformer_engine.bank_pool import projection_banks
from mlops.expert_parallel.transformer_engine.config import MoEConfig


class Bank:
    pass


def test_shared_banks_are_projection_distinct_and_weakly_owned():
    config = MoEConfig(
        ep_size=2,
        num_experts=4,
        top_k=2,
        model_dim=128,
        expert_hidden_dim=128,
        share_expert_banks=True,
    )
    buffer = SimpleNamespace()
    runtime = SimpleNamespace(
        cfg=config, buffer=buffer, _bank_shapes=lambda c: [(128, 128)] * 3
    )
    first = projection_banks(runtime, lambda *shape: Bank())
    second = projection_banks(runtime, lambda *shape: Bank())
    assert all(a is b for a, b in zip(first, second))
    assert len({id(bank) for bank in first}) == 3
    ref = weakref.ref(first[0])
    del first
    gc.collect()
    assert ref() is second[0]
    del second
    gc.collect()
    assert ref() is None
    assert not buffer._te_expert_banks
    runtime.cfg = replace(config, share_expert_banks=False)
    first = projection_banks(runtime, lambda *shape: Bank())
    second = projection_banks(runtime, lambda *shape: Bank())
    assert all(a is not b for a, b in zip(first, second))


def test_unqualified_fp8_sharing_fails_immediately():
    with pytest.raises(ValueError, match="require BF16"):
        MoEConfig(
            ep_size=2,
            num_experts=4,
            top_k=2,
            model_dim=128,
            expert_hidden_dim=128,
            share_expert_banks=True,
            compute_precision="fp8_current",
        )
