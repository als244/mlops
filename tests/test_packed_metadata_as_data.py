"""Metadata derived from a lengths tensor drives the FLA kernels exactly as the
metadata derived from the same lengths as integers.

The tensor form pads to a fixed size -- empty sequences, and chunk rows that
belong to an empty sequence -- so these check that padding costs nothing but
work: the same outputs and the same gradients.
"""

import pytest
import torch

import mlops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

LENGTHS = (73, 38, 17, 128)
TOKENS = sum(LENGTHS)


def _metadata(lengths: object) -> tuple[torch.Tensor, torch.Tensor]:
    like = torch.empty(TOKENS, device="cuda")
    return mlops.prepare_packed_sequence_metadata(lengths, like)


def _padded_lengths(empty_slots: int) -> torch.Tensor:
    return torch.tensor(
        (*LENGTHS, *((0,) * empty_slots)), dtype=torch.int32, device="cuda"
    )


def _leaf(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype).requires_grad_(True)


def _run(function, inputs, metadata):
    for tensor in inputs:
        tensor.grad = None
    output = function(*inputs, *metadata)
    output.float().square().sum().backward()
    return output.detach(), *(tensor.grad.clone() for tensor in inputs)


def _assert_same(actual, expected) -> None:
    assert torch.equal(actual[0], expected[0])
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("empty_slots", [3, 8192])
def test_padded_metadata_convolves_as_exact_metadata(empty_slots: int) -> None:
    torch.manual_seed(0)
    inputs = (_leaf(TOKENS, 64), _leaf(64, 1, 4))
    expected = _run(mlops.causal_conv_silu, inputs, _metadata(LENGTHS))
    actual = _run(
        mlops.causal_conv_silu, inputs, _metadata(_padded_lengths(empty_slots))
    )
    _assert_same(actual, expected)


@pytest.mark.parametrize("empty_slots", [3, 8192])
def test_padded_metadata_attends_as_exact_metadata(empty_slots: int) -> None:
    torch.manual_seed(1)
    key_heads, value_heads, width = 2, 4, 64

    def attend(q, k, v, beta, decay, a_log, dt_bias, cumulative, chunks):
        return mlops.linear_attention(
            mlops.l2_norm(q),
            mlops.l2_norm(k),
            v,
            torch.sigmoid(beta.float()).to(v.dtype),
            decay,
            a_log,
            dt_bias,
            cumulative,
            chunks,
        )

    inputs = (
        _leaf(TOKENS, key_heads, width),
        _leaf(TOKENS, key_heads, width),
        _leaf(TOKENS, value_heads, width),
        _leaf(TOKENS, value_heads),
        _leaf(TOKENS, value_heads),
        _leaf(value_heads, dtype=torch.float32),
        _leaf(value_heads, dtype=torch.float32),
    )
    expected = _run(attend, inputs, _metadata(LENGTHS))
    actual = _run(attend, inputs, _metadata(_padded_lengths(empty_slots)))
    _assert_same(actual, expected)


@pytest.mark.parametrize("empty_slots", [3, 8192])
def test_padded_metadata_kda_matches_exact_metadata(empty_slots: int) -> None:
    torch.manual_seed(2)
    heads, width = 2, 64
    inputs = (
        torch.nn.functional.normalize(
            _leaf(TOKENS, heads, width).detach(), dim=-1
        ).requires_grad_(),
        torch.nn.functional.normalize(
            _leaf(TOKENS, heads, width).detach(), dim=-1
        ).requires_grad_(),
        _leaf(TOKENS, heads, width),
        (
            -torch.rand(TOKENS, heads, width, device="cuda") * 0.3 - 0.01
        ).requires_grad_(),
        torch.rand(TOKENS, heads, device="cuda", dtype=torch.bfloat16).requires_grad_(),
    )
    expected = _run(mlops.kimi_delta_attention, inputs, _metadata(LENGTHS))
    actual = _run(
        mlops.kimi_delta_attention, inputs, _metadata(_padded_lengths(empty_slots))
    )
    _assert_same(actual, expected)
