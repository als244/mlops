"""FLA-backed causal-convolution, normalization, and delta-rule adapters."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.causal_conv import fla_causal_conv_backward, fla_causal_conv_forward
from ...kernels.gated_rms_norm import (
    fla_gated_rms_norm_backward,
    fla_gated_rms_norm_forward,
)
from ...kernels.l2_norm import fla_l2_norm_backward, fla_l2_norm_forward
from ...kernels.linear_attention import (
    fla_linear_attention_backward,
    fla_linear_attention_forward,
)


def _fla_available():
    try:
        import fla  # noqa: F401
    except ImportError as error:
        return SupportResult.no(f"flash-linear-attention is unavailable: {error}")
    return SupportResult.yes()


def _cuda_support(*tensors):
    available = _fla_available()
    if not available:
        return available
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return SupportResult.no("all tensor arguments must be tensors")
    if not all(value.is_cuda for value in tensors):
        return SupportResult.no("FLA implementations require CUDA tensors")
    return SupportResult.yes()


def _causal_support(
    x, weight, lengths, cumulative, chunk_indices=None, *, surface, **_kwargs
):
    del surface
    supported = _cuda_support(x, weight, cumulative)
    if not supported:
        return supported
    if cumulative.dtype != torch.int64 or cumulative.ndim != 1:
        return SupportResult.no("cumulative must be a one-dimensional INT64 tensor")
    expected = 0 if len(lengths) == 1 else len(lengths) + 1
    if lengths and cumulative.numel() != expected:
        return SupportResult.no(
            f"cumulative must contain {expected} values for {len(lengths)} sequences"
        )
    if chunk_indices is not None and (
        not isinstance(chunk_indices, torch.Tensor)
        or chunk_indices.dtype != torch.int64
        or chunk_indices.ndim != 2
        or chunk_indices.shape[1] != 2
        or chunk_indices.device != cumulative.device
    ):
        return SupportResult.no(
            "chunk_indices must be a two-column INT64 tensor on the metadata device"
        )
    return SupportResult.yes()


def causal_forward(x, weight, lengths, cumulative, chunk_indices=None):
    with torch.no_grad():
        del lengths
        return fla_causal_conv_forward(
            x, weight, cumulative, chunk_indices
        )


def causal_backward(
    grad_output, x, weight, cumulative, chunk_indices=None
):
    with torch.no_grad():
        return fla_causal_conv_backward(
            grad_output, x, weight, cumulative, chunk_indices
        )


def causal_apply(x, weight, lengths, cumulative, chunk_indices=None):
    flat = x.reshape(-1, x.shape[-1])
    output = _causal_forward_op(
        flat, weight, list(lengths), cumulative, chunk_indices
    )
    return output.reshape_as(x)


@torch.library.custom_op(
    "mlops::causal_conv_silu_fla_fwd", mutates_args=()
)
def _causal_forward_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    lengths: list[int],
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    return causal_forward(x, weight, lengths, cumulative, chunk_indices)


@_causal_forward_op.register_fake
def _causal_forward_fake(x, weight, lengths, cumulative, chunk_indices=None):
    del weight, lengths, cumulative, chunk_indices
    return torch.empty_like(x)


@torch.library.custom_op(
    "mlops::causal_conv_silu_fla_bwd", mutates_args=()
)
def _causal_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return causal_backward(
        grad_output, x, weight, cumulative, chunk_indices
    )


@_causal_backward_op.register_fake
def _causal_backward_fake(
    grad_output, x, weight, cumulative, chunk_indices=None
):
    del grad_output, cumulative, chunk_indices
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_causal_context(ctx, inputs, output):
    x, weight, _lengths, cumulative, chunk_indices = inputs
    if chunk_indices is None:
        chunk_indices = cumulative.new_empty((0, 2))
    ctx.save_for_backward(x, weight, cumulative, chunk_indices)
    ctx.has_chunk_indices = inputs[4] is not None


def _causal_autograd_backward(ctx, grad_output):
    x, weight, cumulative, chunk_indices = ctx.saved_tensors
    grad_x, grad_weight = _causal_backward_op(
        grad_output,
        x,
        weight,
        cumulative,
        chunk_indices if ctx.has_chunk_indices else None,
    )
    return grad_x, grad_weight, None, None, None


_causal_forward_op.register_autograd(
    _causal_autograd_backward, setup_context=_setup_causal_context
)


def _l2_support(x, _eps=1e-6, *, surface, **_kwargs):
    del surface
    return _cuda_support(x)


def l2_forward(x, eps=1e-6):
    with torch.no_grad():
        return fla_l2_norm_forward(x, float(eps))


def l2_backward(grad_output, output, rstd, eps=1e-6):
    with torch.no_grad():
        return fla_l2_norm_backward(grad_output, output, rstd, float(eps))


def l2_apply(x, eps=1e-6):
    output, _rstd = _l2_forward_op(x, float(eps))
    return output


@torch.library.custom_op(
    "mlops::l2_norm_fla_fwd", mutates_args=()
)
def _l2_forward_op(
    x: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return l2_forward(x, float(eps))


@_l2_forward_op.register_fake
def _l2_forward_fake(x, eps):
    del eps
    return (
        torch.empty_like(x),
        torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(
    "mlops::l2_norm_fla_bwd", mutates_args=()
)
def _l2_backward_op(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return l2_backward(grad_output, output, rstd, float(eps))


@_l2_backward_op.register_fake
def _l2_backward_fake(grad_output, output, rstd, eps):
    del grad_output, rstd, eps
    return torch.empty_like(output)


def _setup_l2_context(ctx, inputs, output):
    _x, eps = inputs
    normalized, rstd = output
    ctx.save_for_backward(normalized, rstd)
    ctx.eps = eps
    ctx.mark_non_differentiable(rstd)


def _l2_autograd_backward(ctx, grad_output, _grad_rstd):
    output, rstd = ctx.saved_tensors
    return _l2_backward_op(grad_output, output, rstd, ctx.eps), None


_l2_forward_op.register_autograd(
    _l2_autograd_backward, setup_context=_setup_l2_context
)


def _gated_support(x, gate, weight, _eps=1e-5, *, surface, **_kwargs):
    del surface
    return _cuda_support(x, gate, weight)


def gated_forward(x, gate, weight, eps=1e-5):
    with torch.no_grad():
        return fla_gated_rms_norm_forward(x, gate, weight, float(eps))


def gated_backward(grad_output, x, gate, weight, rstd, eps=1e-5):
    with torch.no_grad():
        return fla_gated_rms_norm_backward(
            grad_output, x, gate, weight, rstd, float(eps)
        )


def gated_apply(x, gate, weight, eps=1e-5):
    output, _rstd = _gated_forward_op(x, gate, weight, float(eps))
    return output


@torch.library.custom_op(
    "mlops::gated_rms_norm_fla_fwd", mutates_args=()
)
def _gated_forward_op(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gated_forward(x, gate, weight, float(eps))


@_gated_forward_op.register_fake
def _gated_forward_fake(x, gate, weight, eps):
    del gate, weight, eps
    rows = x.numel() // x.shape[-1]
    return torch.empty_like(x), torch.empty(
        rows, dtype=torch.float32, device=x.device
    )


@torch.library.custom_op(
    "mlops::gated_rms_norm_fla_bwd", mutates_args=()
)
def _gated_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return gated_backward(grad_output, x, gate, weight, rstd, float(eps))


@_gated_backward_op.register_fake
def _gated_backward_fake(grad_output, x, gate, weight, rstd, eps):
    del grad_output, rstd, eps
    return torch.empty_like(x), torch.empty_like(gate), torch.empty_like(weight)


def _setup_gated_context(ctx, inputs, output):
    x, gate, weight, eps = inputs
    _normalized, rstd = output
    ctx.save_for_backward(x, gate, weight, rstd)
    ctx.eps = eps
    ctx.mark_non_differentiable(rstd)


def _gated_autograd_backward(ctx, grad_output, _grad_rstd):
    x, gate, weight, rstd = ctx.saved_tensors
    gradients = _gated_backward_op(
        grad_output, x, gate, weight, rstd, ctx.eps
    )
    return (*gradients, None)


_gated_forward_op.register_autograd(
    _gated_autograd_backward, setup_context=_setup_gated_context
)


def _linear_support(
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    lengths,
    cumulative,
    chunk_indices,
    *,
    surface,
    scale=None,
    **_kwargs,
):
    del surface, scale
    supported = _cuda_support(
        q, k, v, beta, a, a_log, dt_bias, cumulative, chunk_indices
    )
    if not supported:
        return supported
    if cumulative.dtype != torch.int64 or cumulative.ndim != 1:
        return SupportResult.no("cumulative must be a one-dimensional INT64 tensor")
    if chunk_indices.dtype != torch.int64 or (
        chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2
    ):
        return SupportResult.no("chunk_indices must have shape [chunks, 2] and dtype INT64")
    if len(lengths) == 1:
        if cumulative.numel() or chunk_indices.numel():
            return SupportResult.no("single-sequence metadata must use empty tensors")
    elif lengths and cumulative.numel() != len(lengths) + 1:
        return SupportResult.no("cumulative size must equal len(lengths) + 1")
    return SupportResult.yes()


def linear_forward(
    q, k, v, beta, a, a_log, dt_bias, lengths, cumulative, chunk_indices, *, scale
):
    with torch.no_grad():
        del lengths
        output, gate, matrix = (
            fla_linear_attention_forward(
                q,
                k,
                v,
                beta,
                a,
                a_log,
                dt_bias,
                cumulative,
                chunk_indices,
                float(scale),
            )
        )
        return output, gate, matrix


def linear_backward(
    grad_output, q, k, v, beta, a, a_log, dt_bias, gate, matrix,
    cumulative, chunk_indices, *, scale,
):
    with torch.no_grad():
        sequence_metadata = None if cumulative.numel() == 0 else cumulative
        chunk_metadata = None if chunk_indices.numel() == 0 else chunk_indices
        return fla_linear_attention_backward(
            grad_output,
            q,
            k,
            v,
            beta,
            a,
            a_log,
            dt_bias,
            gate,
            matrix,
            sequence_metadata,
            chunk_metadata,
            float(scale),
        )


def linear_apply(
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    lengths,
    cumulative,
    chunk_indices,
    *,
    scale=None,
):
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    output, *_residuals = _linear_forward_op(
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        cumulative,
        chunk_indices,
        resolved_scale,
    )
    return output


@torch.library.custom_op(
    "mlops::linear_attention_fla_gated_delta_rule_fwd",
    mutates_args=(),
)
def _linear_forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return linear_forward(
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        (),
        cumulative,
        chunk_indices,
        scale=float(scale),
    )


@_linear_forward_op.register_fake
def _linear_forward_fake(
    q, k, v, beta, a, a_log, dt_bias, cumulative, chunk_indices, scale
):
    del k, beta, a, a_log, dt_bias, scale
    tokens = q.shape[0]
    value_heads = v.shape[1]
    del cumulative, chunk_indices
    return (
        torch.empty_like(v),
        torch.empty((1, tokens, value_heads), dtype=torch.float32, device=q.device),
        torch.empty((1, tokens, value_heads, 64), dtype=v.dtype, device=q.device),
    )


@torch.library.custom_op(
    "mlops::linear_attention_fla_gated_delta_rule_bwd",
    mutates_args=(),
)
def _linear_backward_op(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gate: torch.Tensor,
    matrix: torch.Tensor,
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    return linear_backward(
        grad_output,
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        gate,
        matrix,
        cumulative,
        chunk_indices,
        scale=float(scale),
    )


@_linear_backward_op.register_fake
def _linear_backward_fake(
    grad_output, q, k, v, beta, a, a_log, dt_bias, gate, matrix,
    cumulative, chunk_indices, scale,
):
    del grad_output, gate, matrix, cumulative, chunk_indices, scale
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(beta),
        torch.empty_like(a),
        torch.empty_like(a_log, dtype=torch.float32),
        torch.empty_like(dt_bias, dtype=torch.float32),
    )


def _setup_linear_context(ctx, inputs, output):
    q, k, v, beta, a, a_log, dt_bias, cumulative, chunk_indices, scale = inputs
    _result, gate, matrix = output
    ctx.save_for_backward(
        q, k, v, beta, a, a_log, dt_bias, gate, matrix, cumulative, chunk_indices
    )
    ctx.scale = scale
    ctx.mark_non_differentiable(gate, matrix)


def _linear_autograd_backward(ctx, grad_output, _grad_gate, _grad_matrix):
    gradients = _linear_backward_op(
        grad_output, *ctx.saved_tensors, ctx.scale
    )
    return (*gradients, None, None, None)


_linear_forward_op.register_autograd(
    _linear_autograd_backward, setup_context=_setup_linear_context
)


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "causal_conv_silu", "fla.causal_conv_silu", "fla", 100, True,
            _causal_support, causal_apply, causal_forward, causal_backward,
        ),
        Implementation(
            "l2_norm", "fla.l2_norm", "fla", 100, True,
            _l2_support, l2_apply, l2_forward, l2_backward,
        ),
        Implementation(
            "gated_rms_norm", "fla.gated_rms_norm", "fla", 100, True,
            _gated_support, gated_apply, gated_forward, gated_backward,
        ),
        Implementation(
            "linear_attention", "fla.linear_attention.gated_delta_rule", "fla", 100, True,
            _linear_support, linear_apply, linear_forward, linear_backward,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
