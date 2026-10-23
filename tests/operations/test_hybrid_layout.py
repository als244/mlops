"""Packed layouts and frozen-weight convolution must preserve values and VJPs."""

import pytest
import torch

import mlops
from mlops.dispatch import use_implementation

pytestmark = pytest.mark.gpu


def conv_reference(dy, x, weight, lengths):
    """Independent FP32 oracle with FLA's preactivation dtype boundary."""
    w = weight[:, 0].float()
    dw = torch.zeros_like(w)
    outputs = []
    start = 0
    for length in lengths:
        xx = x[start : start + length].float()
        pre = torch.zeros_like(xx)
        for tap in range(w.shape[-1]):
            shift = w.shape[-1] - 1 - tap
            if shift < length:
                pre[shift:] += xx[: length - shift] * w[:, tap]
        pre = pre.to(x.dtype).float()
        sig = pre.sigmoid()
        dz = dy[start : start + length].float() * sig * (1 + pre * (1 - sig))
        dx = torch.zeros_like(xx)
        for tap in range(w.shape[-1]):
            shift = w.shape[-1] - 1 - tap
            if shift < length:
                dx[: length - shift] += dz[shift:] * w[:, tap]
                dw[:, tap] += (dz[shift:] * xx[: length - shift]).sum(0)
        outputs.append(dx.to(x.dtype))
        start += length
    return torch.cat(outputs), dw[:, None].to(weight.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [127, 128, 256])
@pytest.mark.parametrize("layout", ["packed", "transpose", "feature_stride"])
def test_strided_l2_values_gradients_and_layout(dtype, width, layout):
    from mlops.providers.fla.hybrid import _l2_forward_op

    torch.manual_seed(527)
    if layout == "packed":
        x = torch.randn(137, 3, 4, width, device="cuda", dtype=dtype)[:, 1]
    elif layout == "transpose":
        x = torch.randn(4, 137, width, device="cuda", dtype=dtype).transpose(0, 1)
    else:
        x = torch.randn(137, 4, 2 * width, device="cuda", dtype=dtype)[..., ::2]
    x.requires_grad_()
    xf = x.detach().float()
    rs = 1 / (xf.square().sum(-1) + 1e-6).sqrt()
    expected = (xf * rs.unsqueeze(-1)).to(x.dtype)
    dy = torch.randn_like(expected)
    y = expected.float()
    target_dx = (
        (dy.float() - (dy.float() * y).sum(-1, keepdim=True) * y) * rs.unsqueeze(-1)
    ).to(x.dtype)
    with use_implementation("l2_norm", "fla.l2_norm"):
        result = mlops.l2_norm(x)
        if width == 128 and dtype == torch.bfloat16 and layout == "packed":
            (dx,) = torch.autograd.grad(result, x, dy)
        else:
            dx = None
    assert result.is_contiguous()
    tolerance = 0.008 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
    if dx is not None:
        torch.testing.assert_close(dx, target_dx, atol=tolerance, rtol=tolerance)
    torch.library.opcheck(
        _l2_forward_op,
        (x.detach(), 1e-6),
        test_utils=("test_faketensor", "test_schema"),
    )


@pytest.mark.parametrize("taps", [1, 3, 4, 7])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_frozen_conv_input_gradient_matches_fla(taps, weight_dtype, packed):
    from mlops.providers.fla.hybrid import _causal_input_backward_op

    torch.manual_seed(823)
    lengths = (1, 2, 63, 65, 70) if packed else (201,)
    t, channels = sum(lengths), 67
    x = torch.randn(t, channels * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    dy = torch.randn(t, channels * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    weight = torch.randn(channels, 1, taps, device="cuda", dtype=weight_dtype) * 0.2
    cumulative = (
        torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int64,
        )
        if packed
        else torch.empty(0, device="cuda", dtype=torch.int64)
    )
    chunks = (
        torch.tensor(
            [(i, j) for i, n in enumerate(lengths) for j in range((n + 63) // 64)],
            device="cuda",
            dtype=torch.int64,
        )
        if packed
        else None
    )
    expected, _ = conv_reference(dy, x, weight, lengths)
    actual = _causal_input_backward_op(dy, x, weight, cumulative, chunks)
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    torch.library.opcheck(
        _causal_input_backward_op,
        (dy, x, weight, cumulative, chunks),
        test_utils=("test_faketensor", "test_schema"),
    )


@pytest.mark.parametrize("train_weight", [False, True])
def test_conv_autograd_selects_input_only_or_full_backward(train_weight):
    from torch._functorch.aot_autograd import aot_function

    torch.manual_seed(989)
    x = torch.randn(135, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = (
        torch.randn(64, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_(train_weight)
    dy = torch.randn_like(x)
    cu = torch.tensor([0, 65, 135], device="cuda", dtype=torch.int64)
    chunks = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], device="cuda", dtype=torch.int64
    )
    graphs = []

    def compiler(gm, _args):
        graphs.append(gm)
        from functorch.compile import make_boxed_func

        return make_boxed_func(gm.forward)

    def call(a, b, cumulative, indices):
        return mlops.causal_conv_silu(a, b, cumulative, indices)

    with use_implementation("causal_conv_silu", "fla.causal_conv_silu"):
        result = aot_function(call, fw_compiler=compiler, bw_compiler=compiler)(
            x, weight, cu, chunks
        )
        requested = (x, weight) if train_weight else (x,)
        grads = torch.autograd.grad(result, requested, dy)
    expected = conv_reference(dy, x, weight, (65, 70))
    for actual, target in zip(grads, expected):
        torch.testing.assert_close(actual, target, atol=0.008, rtol=0.008)
    backward_targets = [str(n.target) for n in graphs[-1].graph.nodes]
    name = (
        "mlops.causal_conv_silu_fla_bwd.default"
        if train_weight
        else "mlops.causal_conv_silu_fla_input_bwd.default"
    )
    assert name in backward_targets
