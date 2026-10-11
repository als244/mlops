"""FLA KDA registration with call-local metadata and explicit VJPs."""

from importlib.util import find_spec

import torch

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kda import _validate


@torch.library.custom_op("mlops::kimi_delta_attention_fla_fwd", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from ...kernels.kda.execution import _define_unused_triangle, forward

    o, gates, aqk, akk = forward(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        cumulative,
        chunks,
        scale,
    )
    _define_unused_triangle[(chunks.shape[0], q.shape[1])](
        aqk, cumulative, chunks, q.shape[1]
    )
    return o.squeeze(0), gates.squeeze(0), aqk.squeeze(0), akk.squeeze(0)


@_forward.register_fake
def _fake(q, k, v, g, beta, cumulative, chunks, scale):
    matrices = (*beta.shape, 64)
    return (
        torch.empty_like(v),
        torch.empty_like(g, dtype=torch.float32),
        q.new_empty(matrices),
        q.new_empty(matrices),
    )


@torch.library.custom_op("mlops::kimi_delta_attention_fla_bwd", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gates: torch.Tensor,
    aqk: torch.Tensor,
    akk: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from ...kernels.kda.execution import backward

    dq, dk, dv, dg, db = backward(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        beta.unsqueeze(0),
        gates.unsqueeze(0),
        aqk.unsqueeze(0),
        akk.unsqueeze(0),
        dy.contiguous().unsqueeze(0),
        cumulative,
        chunks,
        scale,
    )
    return (
        dq.squeeze(0).to(q),
        dk.squeeze(0).to(k),
        dv.squeeze(0).to(v),
        dg.squeeze(0).to(g),
        db.squeeze(0).to(beta),
    )


@_backward.register_fake
def _fake_bwd(dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v, g, beta))


def _setup(ctx, inputs, output):
    q, k, v, g, beta, cumulative, chunks, ctx.scale = inputs
    _, gates, aqk, akk = output
    ctx.save_for_backward(q, k, v, g, beta, gates, aqk, akk, cumulative, chunks)
    ctx.mark_non_differentiable(gates, aqk, akk)


def _autograd(ctx, dy, *_):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


def _supports(q, k, v, g, beta, cumulative, chunks, *, surface, **_kwargs):
    try:
        _validate(q, k, v, g, beta, cumulative, chunks)
    except ValueError as error:
        return SupportResult.no(str(error))
    if find_spec("fla") is None:
        return SupportResult.no("KDA requires the mlops[fla] optional dependencies")
    return SupportResult.yes()


def forward(q, k, v, g, beta, cumulative, chunks):
    return _forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        cumulative,
        chunks,
        q.shape[-1] ** -0.5,
    )


def backward(dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks):
    return _backward(
        dy,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g,
        beta.contiguous(),
        gates,
        aqk,
        akk,
        cumulative,
        chunks,
        q.shape[-1] ** -0.5,
    )


def apply(q, k, v, g, beta, cumulative, chunks):
    return forward(q, k, v, g, beta, cumulative, chunks)[0]


@flop_formula(_forward)
def _forward_flops(q, k, v, g, beta, cumulative, chunks, scale, **_kwargs):
    return costs.kimi_delta_attention(
        q, k, v, g, beta, cumulative, chunks
    ).logical_flops


@flop_formula(_backward)
def _backward_flops(
    dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks, scale, **_kwargs
):
    return costs.kimi_delta_attention(
        q, k, v, g, beta, cumulative, chunks, entrypoint="backward"
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "kimi_delta_attention",
        "fla.kimi_delta_attention.chunk",
        "fla",
        100,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)
