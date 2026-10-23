"""Split-QKV convolution with an input-only packed-sequence VJP.

Forward reuses FLA for each projection. The opaque multi-output boundary
preserves the separate-gradient VJP through torch.export and later AOT capture.
"""

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from .hybrid import _causal_forward_op, _causal_support


@torch.library.custom_op("mlops::causal_conv_silu_qkv_input_bwd", mutates_args=())
def _input_backward(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    packed: torch.Tensor,
    weight: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ...kernels.hybrid_layout import causal_silu_qkv_input_backward

    return causal_silu_qkv_input_backward(
        dq, dk, dv, packed, weight, cumulative, chunks
    )


@_input_backward.register_fake
def _input_backward_fake(dq, dk, dv, packed, weight, cumulative, chunks=None):
    return tuple(
        torch.empty_like(g, memory_format=torch.contiguous_format) for g in (dq, dk, dv)
    )


@flop_formula(_input_backward)
def _input_backward_flops(
    dq, dk, dv, packed, weight, cumulative, chunks=None, **_kwargs
):
    return logical.causal_conv_silu(
        packed, weight, cumulative, chunks, entrypoint="backward", weight_gradient=False
    ).logical_flops


@torch.library.custom_op("mlops::causal_conv_silu_qkv_fla_fwd", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Separate allocations make the custom-op outputs functional and non-aliasing.
    # Reuse FLA rather than maintaining a second convolution forward kernel.
    outputs = []
    offset = 0
    for part in (q, k, v):
        stop = offset + part.shape[-1]
        outputs.append(
            _causal_forward_op(part, weight[offset:stop], cumulative, chunks)
        )
        offset = stop
    return (*outputs, torch.cat((q, k, v), dim=-1))


@_forward.register_fake
def _forward_fake(q, k, v, weight, cumulative, chunks=None):
    return (
        *(
            torch.empty_like(t, memory_format=torch.contiguous_format)
            for t in (q, k, v)
        ),
        q.new_empty((q.shape[0], q.shape[1] + k.shape[1] + v.shape[1])),
    )


@flop_formula(_forward)
def _forward_flops(q, k, v, weight, cumulative, chunks=None, **_kwargs):
    return logical.causal_conv_silu_qkv(
        q, k, v, weight, cumulative, chunks
    ).logical_flops


def _setup(ctx, inputs, output):
    _q, _k, _v, weight, cumulative, chunks = inputs
    packed = output[-1]
    ctx.has_chunks = chunks is not None
    if chunks is None:
        chunks = cumulative.new_empty((0, 2))
    ctx.save_for_backward(packed, weight, cumulative, chunks)
    ctx.mark_non_differentiable(packed)


def _backward(ctx, dq, dk, dv, _packed_grad):
    packed, weight, cumulative, chunks = ctx.saved_tensors
    return (
        *_input_backward(
            dq, dk, dv, packed, weight, cumulative, chunks if ctx.has_chunks else None
        ),
        None,
        None,
        None,
    )


_forward.register_autograd(_backward, setup_context=_setup)


def _supports(q, k, v, weight, cumulative, chunks=None, *, surface, **_kwargs):
    if surface != "semantic":
        return SupportResult.no("split-QKV convolution is a semantic operation")
    supported = _causal_support(q, weight, cumulative, chunks, surface=surface)
    if not supported:
        return supported
    if weight.requires_grad or weight.shape[-1] > 8:
        return SupportResult.no(
            "input-only fusion requires frozen weights and width <= 8"
        )
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return SupportResult.no("requires BF16, FP16 or FP32 activations")
    return SupportResult.yes()


def apply(q, k, v, weight, cumulative, chunks=None):
    return _forward(q, k, v, weight, cumulative, chunks)[:3]


IMPLEMENTATION = register_implementation(
    Implementation(
        "causal_conv_silu_qkv",
        "fla.causal_conv_silu_qkv",
        "fla",
        100,
        True,
        _supports,
        apply,
    )
)
