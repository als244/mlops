"""Builtin variable-length native FlashAttention implementation."""

from __future__ import annotations

from functools import lru_cache

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.flash_attention import (
    flash_attention_backward,
    flash_attention_forward,
    native_flash_attention_supported,
)


@lru_cache(maxsize=1)
def _flash_attention_3_activated(capability_major: int) -> bool:
    """Whether the aten flash primitive now routes to FlashAttention-3.

    PyTorch's FA3 registration replaces the CUDA implementations of the same
    aten operations this provider calls, so activation changes the kernels
    without changing this implementation's identity or contract.

    FA3 is a Hopper kernel shipped as a compiled extension, which makes
    "installed" and "loadable" different questions: a wheel built against
    another CUDA runtime is present on disk and still raises on import. So
    the import is the test. A negative answer of any kind leaves this
    provider on PyTorch's own kernels, which is a working configuration
    rather than a failure.

    The answer is cached rather than recorded in module state, and it is
    asked on the first CUDA call rather than at import, because reading a
    device capability initializes CUDA and importing mlops must not.
    """

    if capability_major != 9:
        return False
    try:
        import flash_attn_interface  # noqa: F401
        from torch.nn.attention import activate_flash_attention_impl
    except ImportError:
        return False
    try:
        activate_flash_attention_impl("FA3")
    except (RuntimeError, ValueError):
        return False
    return True


def _maybe_activate_fa3(device: torch.device) -> bool:
    """Activate FA3 once for this process, if this device can run it."""

    if device.type != "cuda":
        return False
    return _flash_attention_3_activated(torch.cuda.get_device_capability(device)[0])


def _supports(
    q, k, v, cu_seqlens, max_seqlen, *, surface, causal=True, softmax_scale=None, **_kwargs
):
    del surface, max_seqlen, causal, softmax_scale
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v, cu_seqlens)):
        return SupportResult.no("q, k, v, and cu_seqlens must be tensors")
    _maybe_activate_fa3(q.device)
    # The offsets are data, so nothing here may read their values.
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        return SupportResult.no("cu_seqlens must be one-dimensional int32")
    if cu_seqlens.device != q.device:
        return SupportResult.no("cu_seqlens must be on the queries' device")
    if not native_flash_attention_supported(q, k, v):
        return SupportResult.no(
            "PyTorch variable-length FlashAttention requires contiguous FP16/BF16 "
            "CUDA tensors and compute capability 8.0 or newer"
        )
    return SupportResult.yes()


def forward(
    q, k, v, cu_seqlens, max_seqlen, lengths, *, causal=True, softmax_scale=None
):
    # Execution-only processes replay compiled artifacts without ever
    # resolving an implementation, so the activation attempt must also sit on
    # the call path itself, not just in the support gate.
    _maybe_activate_fa3(q.device)
    normalized = tuple(int(length) for length in lengths)
    with torch.no_grad():
        output, lse, used_native = flash_attention_forward(
            q,
            k,
            v,
            cu_seqlens,
            int(max_seqlen),
            normalized,
            bool(causal),
            softmax_scale,
        )
        if not used_native:
            raise RuntimeError(
                "builtin FlashAttention implementation became unsupported"
            )
        if lse is None:
            raise RuntimeError("native FlashAttention did not return log-sum-exp state")
        return output, lse


def backward(
    grad_output,
    q,
    k,
    v,
    output,
    lse,
    cu_seqlens,
    max_seqlen,
    lengths,
    *,
    causal=True,
    softmax_scale=None,
    deterministic=False,
):
    _maybe_activate_fa3(q.device)
    with torch.no_grad():
        return flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            output,
            lse,
            cu_seqlens,
            int(max_seqlen),
            tuple(int(length) for length in lengths),
            True,
            bool(causal),
            softmax_scale,
            bool(deterministic),
        )


@torch.library.custom_op("mlops::flash_attention_builtin_aten_fwd", mutates_args=())
def _forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``deterministic`` selects nothing in the forward; it rides this
    # operation's inputs because a custom autograd operation's saved context
    # is the only channel that reaches its backward.
    del deterministic
    # The offsets are data: a captured graph takes them as an input, so one
    # graph serves every packing of the same tokens.
    return forward(
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        (),
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )


@_forward_op.register_fake
def _forward_fake(q, k, v, cu_seqlens, max_seqlen, causal, softmax_scale, deterministic):
    del k, v, cu_seqlens, max_seqlen, causal, softmax_scale, deterministic
    lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
    return torch.empty_like(q), lse


@torch.library.custom_op("mlops::flash_attention_builtin_aten_bwd", mutates_args=())
def _backward_op(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    saved_lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The native backward reads the offsets alone, never lengths.
    return backward(
        grad_output,
        q,
        k,
        v,
        output,
        saved_lse,
        cu_seqlens,
        int(max_seqlen),
        (),
        causal=bool(causal),
        softmax_scale=softmax_scale,
        deterministic=bool(deterministic),
    )


@_backward_op.register_fake
def _backward_fake(
    grad_output,
    q,
    k,
    v,
    output,
    saved_lse,
    cu_seqlens,
    max_seqlen,
    causal,
    softmax_scale,
    deterministic,
):
    del grad_output, output, saved_lse, cu_seqlens, max_seqlen
    del causal, softmax_scale, deterministic
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


@flop_formula(_forward_op)
def _forward_flops(
    q, k, v, cu_seqlens, max_seqlen, causal=True, *_rest, out_val=None, **_kwargs
):
    del out_val
    return logical.flash_attention(
        q, k, v, cu_seqlens, max_seqlen, causal=causal, entrypoint="forward"
    ).logical_flops


@flop_formula(_backward_op)
def _backward_flops(
    grad_output,
    q,
    k,
    v,
    output,
    saved_lse,
    cu_seqlens,
    max_seqlen,
    causal=True,
    *_rest,
    out_val=None,
    **_kwargs,
):
    del grad_output, output, saved_lse, out_val
    return logical.flash_attention(
        q, k, v, cu_seqlens, max_seqlen, causal=causal, entrypoint="backward"
    ).logical_flops


def _setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen, causal, softmax_scale, deterministic = inputs
    result, saved_lse = output
    ctx.save_for_backward(q, k, v, result, saved_lse, cu_seqlens)
    ctx.max_seqlen = int(max_seqlen)
    ctx.causal = bool(causal)
    ctx.softmax_scale = softmax_scale
    ctx.deterministic = bool(deterministic)
    ctx.mark_non_differentiable(saved_lse)


def _autograd_backward(ctx, grad_output, _grad_lse):
    q, k, v, output, saved_lse, cu_seqlens = ctx.saved_tensors
    gradients = _backward_op(
        grad_output,
        q,
        k,
        v,
        output,
        saved_lse,
        cu_seqlens,
        ctx.max_seqlen,
        ctx.causal,
        ctx.softmax_scale,
        ctx.deterministic,
    )
    return (*gradients, None, None, None, None, None)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    *,
    causal=True,
    softmax_scale=None,
    deterministic=False,
):
    output, _lse = _forward_op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens,
        int(max_seqlen),
        bool(causal),
        softmax_scale,
        bool(deterministic),
    )
    return output


IMPLEMENTATION = register_implementation(
    Implementation(
        "flash_attention",
        "builtin.flash_attention.aten",
        "builtin",
        100,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
