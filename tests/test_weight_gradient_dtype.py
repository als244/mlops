"""Operations give their weights' gradients at the dtype asked for.

An operation sums a weight's gradient at fp32 and rounds it to the weight's
dtype as it returns it; ``weight_gradients_at(dtype)`` has it return the sum at
``dtype`` instead. The default leaves every operation as it was.
"""

import importlib

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import mlops
from mlops.dispatch import (
    set_weight_gradient_dtype,
    use_implementations,
    weight_gradient_dtype,
    weight_gradients_at,
)
from mlops.kernels.cross_entropy import cross_entropy_fwd_bwd
from mlops.kernels.embedding import embedding_backward
from mlops.kernels.layer_norm import layer_norm_backward, layer_norm_forward
from mlops.kernels.matmul import add_product_, product_at
from mlops.kernels.moe_grouped_gemm import grouped_mm_wgrad
from mlops.kernels.rms_norm import rms_norm_backward, rms_norm_forward

head = importlib.import_module("mlops.providers.builtin.head")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DEVICES = [
    pytest.param("cpu"),
    pytest.param("cuda", marks=[pytest.mark.gpu, cuda]),
]


def _bf16(*shape, device):
    return torch.randn(*shape, device=device, dtype=torch.bfloat16)


def _assert_rounds_to(wide, rounded):
    """On CUDA the one kernel summed both, so rounding the wide sum gives the
    narrow one exactly; the CPU's reference sums may add in another order."""
    exact = rounded.is_cuda
    torch.testing.assert_close(
        wide.to(rounded.dtype),
        rounded,
        rtol=0 if exact else 2**-7,
        atol=0 if exact else 2**-7 * rounded.abs().max().item(),
    )


def test_the_setting_is_scoped_and_refuses_non_floating_dtypes():
    assert weight_gradient_dtype() is None
    with weight_gradients_at(torch.float32):
        assert weight_gradient_dtype() is torch.float32
    assert weight_gradient_dtype() is None
    with pytest.raises(ValueError, match="floating dtype"):
        set_weight_gradient_dtype(torch.int32)


@pytest.mark.parametrize("device", DEVICES)
def test_a_norm_returns_its_weight_gradient_sum_unrounded(device):
    torch.manual_seed(0)
    x, weight = _bf16(64, 96, device=device), _bf16(96, device=device)
    grad = _bf16(64, 96, device=device)
    _output, rstd = rms_norm_forward(x, weight, 1e-5)
    grad_x, rounded = rms_norm_backward(grad, x, weight, rstd)
    wide_x, wide = rms_norm_backward(grad, x, weight, rstd, torch.float32)

    assert rounded.dtype == torch.bfloat16 and wide.dtype == torch.float32
    assert torch.equal(wide_x, grad_x)
    _assert_rounds_to(wide, rounded)

    affine = _bf16(96, device=device)
    _output, mean, rstd = layer_norm_forward(x, weight, affine, 1e-5)
    rounded = layer_norm_backward(grad, x, weight, mean, rstd)
    wide = layer_norm_backward(grad, x, weight, mean, rstd, torch.float32)
    assert torch.equal(wide[0], rounded[0])
    for kept, narrow in zip(wide[1:], rounded[1:], strict=True):
        assert kept.dtype == torch.float32
        _assert_rounds_to(kept, narrow)


@pytest.mark.parametrize("device", DEVICES)
def test_an_embedding_returns_its_table_gradient_sum_unrounded(device):
    torch.manual_seed(0)
    tokens = torch.randint(0, 8, (4, 32), device=device)
    grad = _bf16(4, 32, 16, device=device)
    exact = torch.zeros(8, 16, dtype=torch.float64, device=device)
    exact.index_add_(0, tokens.reshape(-1), grad.reshape(-1, 16).double())

    rounded = embedding_backward(tokens, grad, 8)
    wide = embedding_backward(tokens, grad, 8, torch.float32)

    assert rounded.dtype == torch.bfloat16 and wide.dtype == torch.float32
    torch.testing.assert_close(wide.double(), exact, rtol=1e-6, atol=1e-6)
    _assert_rounds_to(wide, rounded)


@pytest.mark.parametrize("device", DEVICES)
def test_an_experts_weight_gradient_is_their_sum_unrounded(device):
    torch.manual_seed(0)
    x, grad = _bf16(24, 16, device=device), _bf16(24, 8, device=device)
    offsets = torch.tensor([0, 10, 10, 24], dtype=torch.int32, device=device)
    exact = torch.stack(
        [
            x[start:stop].double().T @ grad[start:stop].double()
            for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist())
        ]
    )

    rounded = grouped_mm_wgrad(x, grad, offsets, (3, 16, 8))
    wide = grouped_mm_wgrad(x, grad, offsets, (3, 16, 8), torch.float32)

    assert rounded.dtype == torch.bfloat16 and wide.dtype == torch.float32
    torch.testing.assert_close(wide.double(), exact, rtol=1e-5, atol=1e-5)
    _assert_rounds_to(wide, rounded)


@pytest.mark.parametrize("device", DEVICES)
def test_products_are_summed_at_the_dtype_asked_for(device):
    torch.manual_seed(0)
    left, right = _bf16(32, 256, device=device), _bf16(256, 16, device=device)
    exact = left.double() @ right.double()

    product = product_at(left, right, torch.float32)
    accumulator = torch.ones(32, 16, device=device)
    add_product_(accumulator, left, right)

    assert product.dtype == torch.float32
    torch.testing.assert_close(product.double(), exact, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(accumulator.double(), exact + 1, rtol=1e-5, atol=1e-5)
    # A dtype the multiply does not write -- fp64 from bf16 -- is exact too.
    torch.testing.assert_close(
        product_at(left, right, torch.float64), exact, rtol=1e-12, atol=1e-12
    )


@pytest.mark.gpu
@cuda
def test_the_head_sums_its_chunks_at_the_dtype_asked_for():
    """Kept at fp32, the head's gradient over its chunks is within fp32's
    error of the exact sum of the chunks' products; kept at bf16, each chunk
    is rounded twice."""

    torch.manual_seed(0)
    rows, width, vocabulary, chunk = 64, 32, 48, 8
    hidden = _bf16(rows, width, device="cuda")
    weight = _bf16(vocabulary, width, device="cuda")
    targets = torch.randint(0, vocabulary, (rows,), device="cuda")
    exact = torch.zeros(vocabulary, width, dtype=torch.float64, device="cuda")
    for start in range(0, rows, chunk):
        piece = hidden[start : start + chunk]
        _loss, grad_logits = cross_entropy_fwd_bwd(
            piece @ weight.T, targets[start : start + chunk], total_rows=rows
        )
        exact += grad_logits.double().T @ piece.double()

    _loss, _grad_hidden, rounded = head.forward(
        hidden, weight, targets, chunk_size=chunk
    )
    _loss, _grad_hidden, wide = head.forward(
        hidden, weight, targets, chunk_size=chunk, weight_grad_dtype=torch.float32
    )

    assert rounded.dtype == torch.bfloat16 and wide.dtype == torch.float32
    scale = exact.abs().max()
    wide_error = (wide.double() - exact).abs().max() / scale
    rounded_error = (rounded.double() - exact).abs().max() / scale
    assert wide_error < 1e-5 < rounded_error


def _backward_dtype(graph, name):
    (node,) = (
        node
        for node in graph.graph.nodes
        if node.op == "call_function" and name in str(node.target)
    )
    return node.args[-1]


@pytest.mark.gpu
@cuda
@pytest.mark.parametrize("dtype", [None, torch.float32])
def test_the_setting_reaches_the_backward_the_forward_captured(dtype):
    """Read when the operation is called, the dtype is an argument of the
    forward and so of the backward a trace captures."""

    weight = torch.randn(16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)

    def gradient(weight, x):
        return torch.autograd.grad(mlops.rms_norm(x, weight).sum(), weight)

    with (
        use_implementations({"rms_norm": "builtin.rms_norm.triton"}),
        weight_gradients_at(dtype),
    ):
        graph = make_fx(gradient)(weight, x)

    assert _backward_dtype(graph, "rms_norm_builtin_triton_bwd") == dtype


@pytest.mark.gpu
@cuda
def test_eager_training_runs_with_weight_gradients_kept_at_fp32():
    """Autograd gives a parameter its gradient at the parameter's dtype, so
    eager training rounds what the operation returned; it runs either way."""

    torch.manual_seed(0)
    weight = torch.randn(16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    table = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.randint(0, 8, (2, 4), device="cuda")

    gradients = []
    for dtype in (None, torch.float32):
        with (
            use_implementations(
                {
                    "rms_norm": "builtin.rms_norm.triton",
                    "embedding": "builtin.embedding.deterministic",
                }
            ),
            weight_gradients_at(dtype),
        ):
            loss = mlops.rms_norm(mlops.embedding(tokens, table), weight).float().sum()
            gradients.append(torch.autograd.grad(loss, (weight, table)))

    for rounded, kept in zip(*gradients, strict=True):
        assert kept.dtype == rounded.dtype == torch.bfloat16
        assert torch.equal(kept, rounded)
