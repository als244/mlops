"""Relayout the certified operands and check values, gradients and immutability."""

import pytest
import torch

from mlops.operator_metadata import fla_input_layouts


def test_layout_description_factory_needs_no_device_or_scheduling_policy():
    initialized = torch.cuda.is_initialized()
    descriptions = fla_input_layouts()
    assert len(descriptions) == 9
    assert torch.cuda.is_initialized() == initialized
    description = next(
        c for c in descriptions if c.target == "mlops.l2_norm_fla_bwd.default"
    )
    assert description.inputs == (0,)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.no_grad()
def test_certified_value_operands_preserve_packed_forward_and_vjp(dtype):
    pytest.importorskip("fla")
    import mlops
    from mlops.providers.fla import hybrid  # noqa: F401

    torch.manual_seed(911)
    tokens, hq, hv, width = 105, 2, 4, 64

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    x = rand(tokens, 256)
    weight = rand(256, 1, 4).mul_(0.1)
    cumulative, chunks = mlops.prepare_packed_sequence_metadata((31, 65, 9), x)
    q, k = (
        rand(tokens, hq, width).mul_(width**-0.5),
        rand(tokens, hq, width).mul_(width**-0.5),
    )
    v, beta, a = (
        rand(tokens, hv, width),
        rand(tokens, hv).sigmoid_(),
        rand(tokens, hv).sub_(1),
    )
    a_log = torch.zeros(hv, device="cuda")
    bias = torch.zeros_like(a_log)
    ops = torch.ops.mlops
    linear = (q, k, v, beta, a, a_log, bias, cumulative, chunks, width**-0.5)
    output, gate, matrix = ops.linear_attention_fla_gated_delta_rule_fwd.default(
        *linear
    )
    norm, rstd = ops.l2_norm_fla_fwd.default(q, 1e-6)
    gated_x, gated_gate, gated_weight = (
        rand(tokens, hv, width),
        rand(tokens, hv, width),
        rand(width),
    )
    _, gated_rstd = ops.gated_rms_norm_fla_fwd.default(
        gated_x, gated_gate, gated_weight, 1e-5
    )
    cases = {
        "causal_conv_silu_fla_fwd": (x, weight, cumulative, chunks),
        "causal_conv_silu_fla_bwd": (rand(*x.shape), x, weight, cumulative, chunks),
        "causal_conv_silu_fla_input_bwd": (
            rand(*x.shape),
            x,
            weight,
            cumulative,
            chunks,
        ),
        "l2_norm_fla_fwd": (q, 1e-6),
        "l2_norm_fla_bwd": (rand(*q.shape), norm, rstd, 1e-6),
        "gated_rms_norm_fla_fwd": (gated_x, gated_gate, gated_weight, 1e-5),
        "gated_rms_norm_fla_bwd": (
            rand(*gated_x.shape),
            gated_x,
            gated_gate,
            gated_weight,
            gated_rstd,
            1e-5,
        ),
        "linear_attention_fla_gated_delta_rule_fwd": linear,
        "linear_attention_fla_gated_delta_rule_bwd": (
            rand(*output.shape),
            *linear[:7],
            gate,
            matrix,
            cumulative,
            chunks,
            width**-0.5,
        ),
    }
    for description in fla_input_layouts():
        name = description.target.split(".")[1]
        op = getattr(ops, name).default
        args = cases[name]
        expected = op(*args)
        changed = list(args)
        for index in description.inputs:
            value = args[index]
            backing = torch.empty(
                *value.shape[:-1],
                value.shape[-1] * 2 + 1,
                device=value.device,
                dtype=value.dtype,
            )
            changed[index] = backing[..., 1::2].copy_(value)
            assert not changed[index].is_contiguous()
        immutable = [
            (value, value.clone())
            for value in changed
            if isinstance(value, torch.Tensor)
        ]
        actual = op(*changed)
        actual = (actual,) if isinstance(actual, torch.Tensor) else actual
        expected = (expected,) if isinstance(expected, torch.Tensor) else expected
        for got, want in zip(actual, expected, strict=True):
            assert torch.isfinite(got).all(), name
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3, msg=name)
        for value, snapshot in immutable:
            torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
