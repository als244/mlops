"""Builtin Triton RMSNorm implementation and opaque PyTorch boundary."""

from __future__ import annotations

import torch

from ...dispatch.costs import CostHints, register_operation_estimator
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.rms_norm import (
    rms_norm_backward_triton,
    rms_norm_forward_triton,
    rms_norm_triton_available,
)


def _supports(x, weight, _eps=1e-5, *, surface, **_kwargs) -> SupportResult:
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return SupportResult.no("x and weight must be tensors")
    if x.ndim < 1 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        return SupportResult.no("weight must be rank one and match x.shape[-1]")
    if not x.is_cuda or not weight.is_cuda:
        return SupportResult.no("requires CUDA x and weight tensors")
    if not rms_norm_triton_available():
        return SupportResult.no("Triton is unavailable")
    return SupportResult.yes()


def _logical_cost(x, weight, _eps=1e-5, *, entrypoint, **_kwargs) -> CostHints:
    """Estimate ideal RMSNorm work, independent of a concrete kernel."""
    elements = x.numel()
    rows = elements // x.shape[-1]
    activation_bytes = elements * x.element_size()
    weight_bytes = weight.numel() * weight.element_size()
    if entrypoint == "forward":
        return CostHints(
            logical_flops=5 * elements + 3 * rows,
            logical_bytes_accessed=2 * activation_bytes + weight_bytes,
            notes=(
                "logical FLOPs count square, reduction, scale, rsqrt, and two multiplies",
                "logical bytes are minimum tensor traffic and exclude saved residuals",
            ),
        )
    return CostHints(
        logical_flops=10 * elements + rows,
        logical_bytes_accessed=(
            3 * activation_bytes + 2 * weight_bytes + 4 * rows
        ),
        notes=(
            "backward logical bytes include dy/x/weight/rstd reads and dx/dweight writes",
        ),
    )


register_operation_estimator("rms_norm", _logical_cost)


def _estimate(x, weight, _eps=1e-5, *, entrypoint, **_kwargs) -> CostHints:
    """Estimate source-level traffic for the package-owned Triton kernels."""
    elements = x.numel()
    rows = elements // x.shape[-1]
    width = x.shape[-1]
    activation_bytes = elements * x.element_size()
    weight_bytes = weight.numel() * weight.element_size()
    layout_workspace = (
        (0 if x.is_contiguous() else x.numel() * x.element_size())
        + (0 if weight.is_contiguous() else weight.numel() * weight.element_size())
    )
    if entrypoint == "forward":
        return CostHints(
            implementation_flops=5 * elements + 3 * rows,
            implementation_bytes_accessed=4 * activation_bytes + 4 * rows,
            workspace_bytes=layout_workspace,
            notes=(
                "physical bytes count source-level global loads/stores, not cache effects",
            ),
        )
    programs = (rows + 63) // 64
    partial_bytes = programs * width * 4
    return CostHints(
        implementation_flops=10 * elements + rows,
        implementation_bytes_accessed=(
            7 * activation_bytes
            + weight_bytes
            + 8 * elements
            + 4 * rows
            + partial_bytes
        ),
        workspace_bytes=partial_bytes + layout_workspace,
        notes=(
            "backward traffic is a source-level estimate including fp32 dweight partials",
            "allocator overhead and reduction-library temporaries are excluded",
        ),
    )


def forward(x, weight, eps=1e-5):
    """Return the builtin Triton ``(output, rstd)`` pair."""
    return rms_norm_forward_triton(x.contiguous(), weight.contiguous(), float(eps))


def backward(grad_output, x, weight, rstd):
    """Return the builtin Triton ``(grad_x, grad_weight)`` VJP."""
    return rms_norm_backward_triton(
        grad_output.contiguous(),
        x.contiguous(),
        weight.contiguous(),
        rstd.contiguous(),
    )


@torch.library.custom_op(
    "mlops::rms_norm_builtin_triton_fwd", mutates_args=()
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
    "mlops::rms_norm_builtin_triton_bwd", mutates_args=()
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
    """Apply the autograd-enabled builtin Triton RMSNorm boundary."""
    output, _rstd = _forward_op(x, weight, float(eps))
    return output


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="rms_norm",
        implementation_id="builtin.rms_norm.triton",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
        estimate=_estimate,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
