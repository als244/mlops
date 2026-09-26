"""Sequence offsets are data: repeats pad them, and new values reuse one graph."""

import mlops
import pytest
import torch
from mlops.dispatch import use_implementations

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOKENS = 228


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def tensor(heads: int) -> torch.Tensor:
        value = torch.randn(
            TOKENS, heads, 64, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        return value.requires_grad_(True)

    return tensor(4), tensor(2), tensor(2)


def _offsets(*offsets: int) -> torch.Tensor:
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _attend(cu_seqlens: torch.Tensor, max_seqlen: int) -> tuple[torch.Tensor, ...]:
    q, k, v = _inputs(0)
    output = mlops.flash_attention(q, k, v, cu_seqlens, max_seqlen)
    output.float().square().sum().backward()
    return output.detach(), q.grad, k.grad, v.grad


@pytest.mark.parametrize("max_seqlen", [100, 128, TOKENS])
def test_repeated_offsets_are_empty_sequences(max_seqlen: int) -> None:
    # Two repeats of the last offset pad the tensor, as a fixed-size input is.
    expected = _attend(_offsets(0, 37, 137, 228), 100)
    actual = _attend(_offsets(0, 37, 137, 228, 228, 228), max_seqlen)

    assert torch.equal(actual[0], expected[0])
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_the_reference_reads_offsets_as_the_kernel_does() -> None:
    q, k, v = (tensor.detach() for tensor in _inputs(2))
    cu_seqlens = _offsets(0, 37, 137, 228, 228)
    with use_implementations({"flash_attention": "native_torch.flash_attention"}):
        reference = mlops.flash_attention(q, k, v, cu_seqlens, 100)
    actual = mlops.flash_attention(q, k, v, cu_seqlens, 100)

    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)


def test_new_offsets_reuse_one_captured_graph() -> None:
    # The offsets are an input to what is captured, so a new packing of the
    # same tokens is the same graph given a different value.
    q, k, v = (tensor.detach() for tensor in _inputs(1))
    captures = 0

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> object:
        nonlocal captures
        captures += 1
        return graph_module.forward

    attend = torch.compile(mlops.flash_attention, backend=backend, fullgraph=True)
    with use_implementations({"flash_attention": "builtin.flash_attention.aten"}):
        for packing in ((0, 37, 137, 228, 228), (0, 228, 228, 228, 228), (0, 1, 3, 6, 228)):
            cu_seqlens = _offsets(*packing)
            expected = mlops.flash_attention(q, k, v, cu_seqlens, TOKENS)
            assert torch.equal(attend(q, k, v, cu_seqlens, TOKENS), expected)
    assert captures == 1
