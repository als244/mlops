"""cuLA SM90 WY backward with FLA forward and call-local sequence metadata.

cuLA's Hopper modular training path uses FLA for the other stages. Import its
CuTe WGMMA backward kernel directly: this avoids unrelated SM100 MLIR imports
and the upstream wrapper's dependency on a different FLA private API revision.
"""

from importlib.util import find_spec

import torch

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kda import _validate


@torch.library.custom_op("mlops::kimi_delta_attention_cula_sm90_fwd", mutates_args=())
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
        cumulative.to(torch.int32),
        chunks.to(torch.int32),
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


@torch.library.custom_op("mlops::kimi_delta_attention_cula_sm90_bwd", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gates: torch.Tensor,
    aqk: torch.Tensor,
    akk: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from cula.ops.kda.sm90.bwd_wy_dqkg import chunk_kda_bwd_wy_dqkg_fused

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
        cumulative.to(torch.int32),
        chunks.to(torch.int32),
        scale,
        wy_kernel=chunk_kda_bwd_wy_dqkg_fused,
    )
    return (
        dq.squeeze(0).to(q),
        dk.squeeze(0).to(k),
        dv.squeeze(0).to(v),
        dg.squeeze(0).float(),
        db.squeeze(0).to(beta),
    )


@_backward.register_fake
def _fake_bwd(dy, q, k, v, beta, gates, aqk, akk, cumulative, chunks, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v, gates, beta))


def _setup(ctx, inputs, output):
    q, k, v, _g, beta, cumulative, chunks, ctx.scale = inputs
    _, gates, aqk, akk = output
    ctx.save_for_backward(q, k, v, beta, gates, aqk, akk, cumulative, chunks)
    ctx.mark_non_differentiable(gates, aqk, akk)


def _autograd(ctx, dy, *_):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


def _supports(q, k, v, g, beta, cumulative, chunks, *, surface, **_kwargs):
    try:
        _validate(q, k, v, g, beta, cumulative, chunks)
    except ValueError as error:
        return SupportResult.no(str(error))
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128 or v.shape[-1] != 128:
        return SupportResult.no("cuLA SM90 KDA requires BF16 Q/K/V with D=128")
    if beta.dtype not in (torch.bfloat16, torch.float32):
        return SupportResult.no("cuLA SM90 beta requires BF16 or FP32")
    if q.shape[0] >= 2**31:
        return SupportResult.no("cuLA SM90 sequence metadata uses int32")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        return SupportResult.no("cuLA SM90 WY kernel requires Hopper SM90")
    if any(find_spec(name) is None for name in ("fla", "cula", "cutlass", "tvm_ffi")):
        return SupportResult.no("cuLA requires its pinned optional installation")
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
    dy, q, k, v, beta, gates, aqk, akk, cumulative, chunks, scale, **_kwargs
):
    return costs.kimi_delta_attention(
        q, k, v, gates, beta, cumulative, chunks, entrypoint="backward"
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "kimi_delta_attention",
        "cula.kimi_delta_attention.sm90",
        "cula",
        110,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)
