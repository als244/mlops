"""The EP operator's row math runs as row kernels and compiles nothing.

The references are the eager formulas these kernels replace: the routed rows'
score scaling and the E4M3 rows' dequantization, the routed down projection's
derivative, and the shared expert's SwiGLU and its gradient. Each kernel does
the same FP32 arithmetic and rounds once, so scaling and dequantization match
bit for bit and the SwiGLU forms to BF16 rounding.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

LIMIT = 7.0
BF16 = {"rtol": 1e-2, "atol": 1e-2}


def _generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _bf16(*shape, seed, scale=3.0):
    values = torch.randn(*shape, device="cuda", generator=_generator(seed)) * scale
    return values.to(torch.bfloat16)


def _with_limits(value):
    # Exercise both sides of the clip, and the clip itself.
    edges = [LIMIT - 0.5, LIMIT, LIMIT + 0.5, -LIMIT - 0.5, -LIMIT, 20.0, -20.0]
    value[0, : len(edges)] = torch.tensor(edges, device="cuda", dtype=value.dtype)
    return value


def _reference_down_derivative(preact, r, probabilities, swiglu_limit=None):
    gate, up = preact[:, 0::2].float(), preact[:, 1::2].float()
    raw_gate, raw_up = gate, up
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    s = gate.sigmoid()
    activation = F.silu(gate) * up
    dh = r.float() * probabilities[:, None]
    dgate = dh * up * s * (1 + gate * (1 - s))
    dup = dh * F.silu(gate)
    if swiglu_limit is not None:
        dgate = torch.where(raw_gate <= swiglu_limit, dgate, 0.0)
        dup = torch.where(raw_up.abs() <= swiglu_limit, dup, 0.0)
    dpreact = torch.stack((dgate, dup), dim=-1).flatten(1).to(torch.bfloat16)
    return (
        dpreact,
        (activation * probabilities[:, None]).to(torch.bfloat16),
        (r.float() * activation).sum(-1),
    )


def _reference_shared_forward(x, gate_weight, up_weight, down_weight, limit=None):
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    g, u = gate.float(), up.float()
    if limit is not None:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    hidden = (F.silu(g) * u).to(x.dtype)
    return F.linear(hidden, down_weight), gate, up, hidden


def _reference_shared_backward(dy, x, gate, up, hidden, gate_w, up_w, down_w):
    dh = (dy @ down_w).float()
    g = gate.float()
    sigmoid = torch.sigmoid(g)
    dgate = (dh * up.float() * sigmoid * (1 + g * (1 - sigmoid))).to(x.dtype)
    dup = (dh * F.silu(g)).to(x.dtype)
    return dgate @ gate_w + dup @ up_w, dgate.T @ x, dup.T @ x, dy.T @ hidden


def _reference_frozen_backward(dy, x, gate, up, hidden, gate_w, up_w, down_w, limit):
    dh = (dy @ down_w).float()
    g, u = gate.float(), up.float()
    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    sigmoid = g.sigmoid()
    dg = torch.where(gate <= limit, dh * u * sigmoid * (1 + g * (1 - sigmoid)), 0.0)
    du = torch.where(up.abs() <= limit, dh * g * sigmoid, 0.0)
    return (dg.to(x.dtype) @ gate_w + du.to(x.dtype) @ up_w,)


def test_routed_rows_scale_and_e4m3_rows_dequantize_exactly():
    from mlops.expert_parallel.quack.lora.math import bf16_rows, scaled_rows
    from mlops.kernels.quack_moe.rows import FP8Rows

    rows, width = 37, 1000
    probabilities = torch.rand(rows, device="cuda", generator=_generator(2))
    # Pitched rows, as a communication slot or a padded E4M3 buffer holds them.
    value = _bf16(rows, width + 24, seed=1)[:, :width]
    out = torch.empty(rows, width + 8, device="cuda", dtype=torch.bfloat16)[:, :width]
    scaled_rows(value, probabilities, out)
    expected = (value.float() * probabilities[:, None]).to(torch.bfloat16)
    assert torch.equal(out, expected)

    data = (_bf16(rows, width + 16, seed=3)).to(torch.float8_e4m3fn)[:, :width]
    scales = torch.rand(rows, device="cuda", generator=_generator(4)) + 0.5
    rows_bf16 = bf16_rows(FP8Rows(data, scales))
    assert rows_bf16.dtype == torch.bfloat16
    assert torch.equal(rows_bf16, (data.float() * scales[:, None]).to(torch.bfloat16))
    assert bf16_rows(value) is value


@pytest.mark.parametrize("limit", [None, LIMIT])
def test_routed_down_derivative_matches_its_eager_formula(limit):
    from mlops.expert_parallel.quack.lora.math import down_derivative

    rows, half = 41, 768
    preact = _with_limits(_bf16(rows, 2 * half, seed=5))
    r = _bf16(rows, half, seed=6, scale=1.0)
    probabilities = torch.rand(rows, device="cuda", generator=_generator(7))
    actual = down_derivative(preact, r, probabilities, limit)
    expected = _reference_down_derivative(preact, r, probabilities, limit)
    for got, want in zip(actual[:2], expected[:2], strict=True):
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got.float(), want.float(), **BF16)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("limit", [None, LIMIT])
def test_shared_expert_matches_its_eager_formulas(limit):
    from mlops.expert_parallel.quack.shared_expert import (
        frozen_shared_backward,
        shared_backward,
        shared_forward,
    )

    tokens, hidden, inner = 96, 256, 384
    x = _bf16(tokens, hidden, seed=8, scale=1.0)
    gate_w = _bf16(inner, hidden, seed=9, scale=0.2)
    up_w = _bf16(inner, hidden, seed=10, scale=0.2)
    down_w = _bf16(hidden, inner, seed=11, scale=0.2)
    dy = _bf16(tokens, hidden, seed=12, scale=1.0)

    actual = shared_forward(x, gate_w, up_w, down_w, swiglu_limit=limit)
    expected = _reference_shared_forward(x, gate_w, up_w, down_w, limit)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.float(), want.float(), **BF16)

    _, gate, up, hidden_state = expected
    state = (gate, up, hidden_state, gate_w, up_w, down_w)
    if limit is None:
        pairs = zip(
            shared_backward(dy, x, *state),
            _reference_shared_backward(dy, x, *state),
            strict=True,
        )
    else:
        pairs = zip(
            frozen_shared_backward(dy, x, *state, swiglu_limit=limit),
            _reference_frozen_backward(dy, x, *state, limit),
            strict=True,
        )
    for got, want in pairs:
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


def test_the_row_math_launches_no_inductor_kernel():
    # A kernel Inductor compiles inside the operator is invisible to the
    # build that captured the operator, and is compiled again in every worker.
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    from mlops.expert_parallel.quack.lora.math import (
        bf16_rows,
        down_derivative,
        scaled_rows,
    )
    from mlops.expert_parallel.quack.shared_expert import (
        frozen_shared_math,
        shared_math,
    )
    from mlops.kernels.quack_moe.rows import FP8Rows

    launched = []
    original = CachingAutotuner.run

    def run(tuner, *args, **kwargs):
        launched.append(tuner.inductor_meta.get("kernel_name"))
        return original(tuner, *args, **kwargs)

    rows, half, hidden = 16, 128, 64
    preact = _bf16(rows, 2 * half, seed=13)
    probabilities = torch.rand(rows, device="cuda", generator=_generator(14))
    value = _bf16(rows, hidden, seed=15)
    x = _bf16(rows, hidden, seed=16, scale=1.0)
    weights = (
        _bf16(half, hidden, seed=17, scale=0.2),
        _bf16(half, hidden, seed=18, scale=0.2),
        _bf16(hidden, half, seed=19, scale=0.2),
    )
    with patch.object(CachingAutotuner, "run", run):
        scaled_rows(value, probabilities, torch.empty_like(value))
        bf16_rows(FP8Rows(value.to(torch.float8_e4m3fn), probabilities))
        down_derivative(preact, _bf16(rows, half, seed=20), probabilities, LIMIT)
        for forward, backward in (shared_math(), frozen_shared_math(LIMIT)):
            _, *state = forward(x, *weights)
            backward(x, x, *state, *weights)
        torch.cuda.synchronize()
    assert launched == []
