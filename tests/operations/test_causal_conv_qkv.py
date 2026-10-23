"""Split convolution keeps packed boundaries and the previous rounding contract."""

import pytest
import torch

import mlops
from mlops.dispatch import use_implementation

from .test_hybrid_layout import conv_reference

pytestmark = pytest.mark.gpu


def inputs(channels, packed, weight_dtype, taps):
    lengths = (1, 2, 65, 69) if packed else (137,)
    torch.manual_seed(834)
    parts = [
        torch.randn(sum(lengths), c * 2, device="cuda", dtype=torch.bfloat16)[
            :, ::2
        ].requires_grad_()
        for c in channels
    ]
    weight = (
        torch.randn(sum(channels), 1, taps, device="cuda", dtype=weight_dtype) * 0.1
    )
    cu = (
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
    return parts, weight, cu, chunks, lengths


@pytest.mark.parametrize("channels", [(32, 32, 32), (35, 67, 19)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("taps", [4, 7])
def test_split_values_gradients_and_fake(channels, packed, weight_dtype, taps):
    from mlops.providers.fla.causal_conv_qkv import _input_backward

    parts, weight, cu, chunks, lengths = inputs(channels, packed, weight_dtype, taps)
    grads = [
        torch.randn(p.shape[0], p.shape[1] * 2, device=p.device, dtype=p.dtype)[:, ::2]
        for p in parts
    ]
    with use_implementation("causal_conv_silu_qkv", "fla.causal_conv_silu_qkv"):
        out = mlops.causal_conv_silu_qkv(*parts, weight, cu, chunks)
        actual = torch.autograd.grad(out, parts, grads)
    packed_x = torch.cat(parts, dim=-1).detach()
    with use_implementation(
        "causal_conv_silu_qkv", "builtin.causal_conv_silu_qkv.composed"
    ):
        reference_out = mlops.causal_conv_silu_qkv(*parts, weight, cu, chunks)
    expected, _ = conv_reference(torch.cat(grads, dim=-1), packed_x, weight, lengths)
    for a, b in zip(out, reference_out):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(actual, expected.split(channels, dim=-1)):
        assert a.is_contiguous()
        torch.testing.assert_close(a, b, atol=0.008, rtol=0.008)
    torch.library.opcheck(
        _input_backward,
        (*grads, packed_x, weight, cu, chunks),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_projection_path_and_trainable_weight_fallback(compiled):
    from mlops.dispatch import explain_implementation

    parts, weight, cu, chunks, _ = inputs((32, 32, 32), True, torch.bfloat16, 4)
    weight.requires_grad_()
    args = (*parts, weight, cu, chunks)
    assert (
        explain_implementation("causal_conv_silu_qkv", *args).selected
        == "builtin.causal_conv_silu_qkv.composed"
    )
    fn = (
        torch.compile(mlops.causal_conv_silu_qkv, backend="aot_eager", fullgraph=True)
        if compiled
        else mlops.causal_conv_silu_qkv
    )
    out = fn(*args)
    grads = torch.autograd.grad(
        sum(t.float().square().mean() for t in out), (*parts, weight)
    )
    assert all(torch.isfinite(g).all() for g in grads)


def test_aot_backward_has_no_gradient_concatenation():
    from functorch.compile import make_boxed_func
    from torch._functorch.aot_autograd import aot_function

    parts, weight, cu, chunks, _ = inputs((32, 32, 32), True, torch.bfloat16, 4)
    graphs = []

    def compiler(gm, _):
        graphs.append(gm)
        return make_boxed_func(gm.forward)

    fn = aot_function(
        mlops.causal_conv_silu_qkv, fw_compiler=compiler, bw_compiler=compiler
    )
    out = fn(*parts, weight, cu, chunks)
    torch.autograd.grad(sum(t.float().square().sum() for t in out), parts)
    targets = [str(n.target) for n in graphs[-1].graph.nodes]
    assert "mlops.causal_conv_silu_qkv_input_bwd.default" in targets
    assert "aten.cat.default" not in targets


def test_inductor_split_forward_backward():
    parts, weight, cu, chunks, _ = inputs((32, 32, 32), True, torch.bfloat16, 4)
    fn = torch.compile(mlops.causal_conv_silu_qkv, fullgraph=True)
    out = fn(*parts, weight, cu, chunks)
    dy = tuple(torch.randn_like(t) for t in out)
    actual = torch.autograd.grad(out, parts, dy)
    expected = torch.autograd.grad(
        mlops.causal_conv_silu_qkv(*parts, weight, cu, chunks), parts, dy
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_export_preserves_split_gradient_boundary():
    from functorch.compile import make_boxed_func
    from torch._functorch.aot_autograd import aot_function

    class Convolution(torch.nn.Module):
        def forward(self, q, k, v, weight, cu, chunks):
            return mlops.causal_conv_silu_qkv(q, k, v, weight, cu, chunks)

    parts, weight, cu, chunks, _ = inputs((32, 32, 32), True, torch.bfloat16, 4)
    args = (*parts, weight, cu, chunks)
    exported = torch.export.export(Convolution(), args, strict=True).module()
    graphs = []

    def compiler(gm, _):
        graphs.append(gm)
        return make_boxed_func(gm.forward)

    fn = aot_function(exported, fw_compiler=compiler, bw_compiler=compiler)
    out = fn(*args)
    torch.autograd.grad(sum(t.float().square().sum() for t in out), parts)
    targets = [str(n.target) for n in graphs[-1].graph.nodes]
    assert "mlops.causal_conv_silu_qkv_input_bwd.default" in targets
    assert "aten.cat.default" not in targets


def test_split_forward_custom_op_contract():
    from mlops.providers.fla.causal_conv_qkv import _forward

    parts, weight, cu, chunks, _ = inputs((35, 67, 19), True, torch.bfloat16, 4)
    torch.library.opcheck(_forward, (*parts, weight, cu, chunks))
