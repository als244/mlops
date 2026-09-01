"""Liger Kernel 0.8.1 RMSNorm adapter with functional custom ops."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch

from ...dispatch.costs import CostHints
from ...dispatch.registry import Implementation, SupportResult, register_implementation


try:
    from liger_kernel.ops.rms_norm import rms_norm_backward as _liger_backward
    from liger_kernel.ops.rms_norm import rms_norm_forward as _liger_forward
    from liger_kernel.ops.utils import calculate_settings as _calculate_settings

    _VERSION = version("liger-kernel")
    _IMPORT_ERROR = None
except (ImportError, PackageNotFoundError) as error:  # optional dependency
    _liger_backward = None
    _liger_forward = None
    _calculate_settings = None
    _VERSION = None
    _IMPORT_ERROR = error


_CASTING_MODE_LLAMA = 0


def _supports(x, weight, _eps=1e-5, *, surface, **_kwargs) -> SupportResult:
    del surface
    if _IMPORT_ERROR is not None:
        return SupportResult.no(f"liger-kernel is unavailable: {_IMPORT_ERROR}")
    if _VERSION != "0.8.1":
        return SupportResult.no(f"requires liger-kernel 0.8.1, found {_VERSION}")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return SupportResult.no("x and weight must be tensors")
    if not x.is_cuda or not weight.is_cuda:
        return SupportResult.no("Liger RMSNorm requires CUDA tensors")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        return SupportResult.no("Liger RMSNorm supports float16 or bfloat16 activations")
    if weight.dtype != x.dtype:
        return SupportResult.no("weight dtype must equal activation dtype")
    if not x.is_contiguous() or not weight.is_contiguous():
        return SupportResult.no("x and weight must be contiguous")
    if weight.ndim != 1 or x.ndim < 1 or x.shape[-1] != weight.numel():
        return SupportResult.no("weight must be rank one and match x.shape[-1]")
    if x.shape[-1] > 65536:
        return SupportResult.no("hidden width exceeds Liger's 65536-element limit")
    return SupportResult.yes()


def _estimate(x, weight, _eps=1e-5, *, entrypoint, **_kwargs) -> CostHints:
    """Report known Liger workspace without guessing at physical traffic."""
    if entrypoint == "forward":
        return CostHints(
            workspace_bytes=0,
            notes=(
                "Liger physical FLOPs/bytes are undefined pending kernel-level accounting",
            ),
        )
    multiprocessors = torch.cuda.get_device_properties(
        x.device
    ).multi_processor_count
    return CostHints(
        workspace_bytes=multiprocessors * x.shape[-1] * 4,
        notes=(
            "workspace is Liger's fp32 per-SM dweight partial tensor",
            "Liger physical FLOPs/bytes are undefined pending kernel-level accounting",
        ),
    )


def forward(x, weight, eps=1e-5):
    """Return Liger ``(output, rstd)`` using Llama casting and offset zero."""
    if _liger_forward is None:
        raise RuntimeError(f"liger-kernel is unavailable: {_IMPORT_ERROR}")
    output, _x_rows, rstd, _block, _warps, _casting = _liger_forward(
        x, weight, float(eps), 0.0, "llama", False
    )
    return output, rstd


def backward(grad_output, x, weight, rstd):
    """Return Liger's non-mutating explicit ``(grad_x, grad_weight)`` VJP."""
    if _liger_backward is None or _calculate_settings is None:
        raise RuntimeError(f"liger-kernel is unavailable: {_IMPORT_ERROR}")
    block_size, num_warps = _calculate_settings(x.shape[-1])
    return _liger_backward(
        grad_output.contiguous(),
        x.reshape(-1, x.shape[-1]),
        weight,
        rstd,
        0.0,
        _CASTING_MODE_LLAMA,
        block_size,
        num_warps,
        False,
        False,
    )


@torch.library.custom_op(
    "mlops::rms_norm_liger_fwd", mutates_args=()
)
def _forward_op(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return forward(x, weight, float(eps))


@_forward_op.register_fake
def _forward_fake(x, weight, eps):
    del weight, eps
    rows = x.numel() // x.shape[-1]
    return torch.empty_like(x), torch.empty(rows, dtype=torch.float32, device=x.device)


@torch.library.custom_op(
    "mlops::rms_norm_liger_bwd", mutates_args=()
)
def _backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return backward(grad_output, x, weight, rstd)


@_backward_op.register_fake
def _backward_fake(grad_output, x, weight, rstd):
    del grad_output, rstd
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, weight, _eps = inputs
    _normalized, rstd = output
    ctx.save_for_backward(x, weight, rstd)
    ctx.mark_non_differentiable(rstd)


def _autograd_backward(ctx, grad_output, _grad_rstd):
    x, weight, rstd = ctx.saved_tensors
    grad_x, grad_weight = _backward_op(grad_output, x, weight, rstd)
    return grad_x, grad_weight, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(x, weight, eps=1e-5):
    """Apply the autograd-enabled Liger RMSNorm boundary."""
    output, _rstd = _forward_op(x, weight, float(eps))
    return output


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="rms_norm",
        implementation_id="liger.rms_norm",
        provider="liger",
        priority=50,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
        estimate=_estimate,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
