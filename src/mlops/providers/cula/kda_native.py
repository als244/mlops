"""Native cuLA forward plus cuLA/FLA backward with intra-chunk recomputation.

The C++ forward does not return Aqk/Akk. Rebuild those matrices during backward
and reuse the resulting WY factors for the gradient computation. Gates are
saved; all metadata and native workspace remain call-local.
"""

from importlib.util import find_spec

import torch

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation


@torch.library.custom_op(
    "mlops::kimi_delta_attention_cula_sm90_native_fwd", mutates_args=()
)
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
    from cula.cudac import kda_fwd_prefill
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    cumulative32 = cumulative.to(torch.int32)
    gates = chunk_local_cumsum(
        g=g.unsqueeze(0),
        scale=RCP_LN2,
        chunk_size=64,
        cu_seqlens=cumulative32,
        chunk_indices=chunks,
    ).squeeze(0)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    workspace = torch.empty(sm_count * 128, device=q.device, dtype=torch.uint8)
    with torch.cuda.device(q.device):
        out, _ = kda_fwd_prefill(
            None,
            None,
            q,
            k,
            v,
            None,
            gates,
            beta,
            cumulative32,
            workspace,
            scale,
            False,
            True,
        )
    return out, gates, q.new_empty(0), q.new_empty(0)


@_forward.register_fake
def _fake(q, k, v, g, beta, cumulative, chunks, scale):
    return (
        torch.empty_like(v),
        torch.empty_like(g, dtype=torch.float32),
        q.new_empty(0),
        q.new_empty(0),
    )


@torch.library.custom_op(
    "mlops::kimi_delta_attention_cula_sm90_native_bwd", mutates_args=()
)
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gates: torch.Tensor,
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
        q.new_empty(0),
        q.new_empty(0),
        dy.contiguous().unsqueeze(0),
        cumulative.to(torch.int32),
        chunks.to(torch.int32),
        scale,
        wy_kernel=chunk_kda_bwd_wy_dqkg_fused,
        recompute_intra=True,
    )
    return (
        dq.squeeze(0).to(q),
        dk.squeeze(0).to(k),
        dv.squeeze(0).to(v),
        dg.squeeze(0).float(),
        db.squeeze(0).to(beta),
    )


@_backward.register_fake
def _fake_bwd(dy, q, k, v, beta, gates, cumulative, chunks, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v, gates, beta))


def _setup(ctx, inputs, output):
    q, k, v, _g, beta, cumulative, chunks, ctx.scale = inputs
    _, gates, aqk, akk = output
    ctx.save_for_backward(q, k, v, beta, gates, cumulative, chunks)
    ctx.mark_non_differentiable(gates, aqk, akk)


def _autograd(ctx, dy, *_):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


def _supports(q, k, v, g, beta, cumulative, chunks, *, surface, **kwargs):
    from .kda import _supports as supports_modular

    result = supports_modular(
        q, k, v, g, beta, cumulative, chunks, surface=surface, **kwargs
    )
    if not result:
        return result
    if find_spec("cula._cudac_sm90") is None:
        return SupportResult.no("Native cuLA forward requires its SM90 C++ extension")
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
def _backward_flops(dy, q, k, v, beta, gates, cumulative, chunks, scale, **_kwargs):
    return costs.kimi_delta_attention(
        q, k, v, gates, beta, cumulative, chunks, entrypoint="backward"
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "kimi_delta_attention",
        "cula.kimi_delta_attention.sm90.recompute",
        "cula",
        120,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)
