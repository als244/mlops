"""Builtin variable-length native FlashAttention implementation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib.util import find_spec

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.flash_attention import (
    flash_attention_backward,
    flash_attention_forward,
    native_flash_attention_supported,
)

_FA3_ATTEMPTED = False
_FA3_ACTIVE = False


def _maybe_activate_fa3(device: torch.device) -> bool:
    """Route the aten flash primitive to FlashAttention-3 where it exists.

    PyTorch's FA3 registration replaces the CUDA implementations of the same
    aten operations this provider calls, so activation changes the kernels
    without changing this implementation's identity or contract. The attempt
    happens on the first CUDA call rather than at import: probing the device
    capability initializes CUDA, and importing mlops must never do that.
    """
    global _FA3_ATTEMPTED, _FA3_ACTIVE
    if _FA3_ATTEMPTED or device.type != "cuda":
        return _FA3_ACTIVE
    _FA3_ATTEMPTED = True
    if find_spec("flash_attn_interface") is None:
        return False
    if torch.cuda.get_device_capability(device)[0] != 9:
        return False
    try:
        from torch.nn.attention import activate_flash_attention_impl

        activate_flash_attention_impl("FA3")
    except Exception:
        return False
    _FA3_ACTIVE = True
    return True


def _supports(q, k, v, lengths, *, surface, causal=True, softmax_scale=None, **_kwargs):
    del surface, causal, softmax_scale
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v)):
        return SupportResult.no("q, k, and v must be tensors")
    _maybe_activate_fa3(q.device)
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        return SupportResult.no("lengths must contain positive values")
    if sum(normalized) != q.shape[0]:
        return SupportResult.no("lengths must cover the query token axis")
    if not native_flash_attention_supported(q, k, v):
        return SupportResult.no("PyTorch native variable-length flash is unsupported")
    return SupportResult.yes()


@lru_cache(maxsize=128)
def _cumulative_host_metadata(
    normalized: tuple[int, ...],
    *,
    pinned: bool,
) -> torch.Tensor:
    cumulative = [0]
    for length in normalized:
        cumulative.append(cumulative[-1] + length)
    cumulative_host = torch.tensor(cumulative, dtype=torch.int32)
    if pinned:
        # A direct pageable-host ``torch.tensor(..., device="cuda")`` copy is
        # synchronous: PyTorch follows cudaMemcpyAsync with a compute-stream
        # synchronize so the temporary host storage can be reclaimed.  This
        # metadata is tiny but is rebuilt by every attention/recompute task,
        # and those host barriers prevent the planner from dispatching future
        # prefetches.  Pinned staging gives the tensor's ordinary same-stream
        # dependency without blocking the launching thread.
        cumulative_host = cumulative_host.pin_memory()
    return cumulative_host


def _metadata(q, lengths: Sequence[int]):
    normalized = tuple(int(length) for length in lengths)
    cumulative_host = _cumulative_host_metadata(
        normalized,
        pinned=q.device.type == "cuda",
    )
    return (
        cumulative_host.to(
            q.device,
            non_blocking=q.device.type == "cuda",
        ),
        max(normalized),
        normalized,
    )


def forward(q, k, v, cu_seqlens, max_seqlen, lengths, *, causal=True, softmax_scale=None):
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
            raise RuntimeError("builtin FlashAttention implementation became unsupported")
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


@torch.library.custom_op(
    "mlops::flash_attention_builtin_aten_fwd", mutates_args=()
)
def _forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: list[int],
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ``deterministic`` selects nothing in the forward; it rides this
    # operation's inputs because a custom autograd operation's saved context
    # is the only channel that reaches its backward.
    del deterministic
    # ``lengths`` is the semantic metadata.  Build the provider's compact
    # device offsets inside this opaque operation so FakeTensor capture never
    # mistakes a data-bearing, call-derived tensor for a serializable constant.
    cu_seqlens, max_seqlen, normalized = _metadata(q, lengths)
    output, lse = forward(
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        normalized,
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )
    return output, lse, cu_seqlens


@_forward_op.register_fake
def _forward_fake(q, k, v, lengths, causal, softmax_scale, deterministic):
    del k, v, causal, softmax_scale, deterministic
    lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
    cu_seqlens = torch.empty(
        (len(lengths) + 1,), dtype=torch.int32, device=q.device
    )
    return torch.empty_like(q), lse, cu_seqlens


@torch.library.custom_op(
    "mlops::flash_attention_builtin_aten_bwd", mutates_args=()
)
def _backward_op(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    saved_lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    lengths: list[int],
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return backward(
        grad_output,
        q,
        k,
        v,
        output,
        saved_lse,
        cu_seqlens,
        int(max_seqlen),
        lengths,
        causal=bool(causal),
        softmax_scale=softmax_scale,
        deterministic=bool(deterministic),
    )


@_backward_op.register_fake
def _backward_fake(
    grad_output, q, k, v, output, saved_lse, cu_seqlens, max_seqlen,
    lengths, causal, softmax_scale, deterministic,
):
    del grad_output, output, saved_lse, cu_seqlens, max_seqlen
    del lengths, causal, softmax_scale, deterministic
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, lengths, causal, softmax_scale, deterministic = inputs
    result, saved_lse, cu_seqlens = output
    ctx.save_for_backward(q, k, v, result, saved_lse, cu_seqlens)
    ctx.max_seqlen = max(int(length) for length in lengths)
    ctx.lengths = list(lengths)
    ctx.causal = bool(causal)
    ctx.softmax_scale = softmax_scale
    ctx.deterministic = bool(deterministic)
    ctx.mark_non_differentiable(saved_lse, cu_seqlens)


def _autograd_backward(ctx, grad_output, _grad_lse, _grad_cu_seqlens):
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
        ctx.lengths,
        ctx.causal,
        ctx.softmax_scale,
        ctx.deterministic,
    )
    return (*gradients, None, None, None, None)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(q, k, v, lengths, *, causal=True, softmax_scale=None, deterministic=False):
    normalized = tuple(int(length) for length in lengths)
    output, _lse, _cu_seqlens = _forward_op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        list(normalized),
        bool(causal),
        softmax_scale,
        bool(deterministic),
    )
    return output


IMPLEMENTATION = register_implementation(
    Implementation(
        "flash_attention", "builtin.flash_attention.aten", "builtin", 100, False,
        _supports, apply, forward, backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
