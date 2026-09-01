"""FA3 routing for the builtin FlashAttention implementation on Hopper."""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch

import mlops
from mlops.providers.builtin import flash_attention as builtin_flash_attention


def _fa3_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability()[0] != 9:
        return "requires a Hopper GPU"
    if find_spec("flash_attn_interface") is None:
        return "requires the flash-attn-3 wheel"
    return None


_UNAVAILABLE = _fa3_unavailable_reason()


@pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or "available")
def test_builtin_flash_attention_activates_fa3_on_hopper():
    lengths = (64, 96, 32)
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(sum(lengths), 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    output = mlops.flash_attention(q, k, v, lengths, causal=True)
    assert builtin_flash_attention._FA3_ACTIVE

    references = []
    start = 0
    for length in lengths:
        stop = start + length
        qs, ks, vs = (
            tensor[start:stop].float().transpose(0, 1) for tensor in (q, k, v)
        )
        references.append(
            torch.nn.functional.scaled_dot_product_attention(
                qs, ks, vs, is_causal=True
            ).transpose(0, 1)
        )
        start = stop
    torch.testing.assert_close(
        output.float(), torch.cat(references), rtol=2e-2, atol=2e-2
    )
