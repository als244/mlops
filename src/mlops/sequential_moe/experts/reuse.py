"""Expert VJPs that accumulate into explicit caller-owned gradient tensors.

Fixed tuples expose two base gradients or four LoRA gradients to compilation.
The caller initializes dx/dp once and keeps them live across expert calls.
"""

import torch

from .execution import backward_expert_into
from .operators import _backward_fake


def _validate_destinations(x, probs, dx, dp):
    if (
        dx.shape != x.shape
        or dx.dtype != x.dtype
        or dp.shape != probs.shape
        or dp.dtype != probs.dtype
    ):
        raise ValueError("Gradient destinations must match input/probability geometry")


@torch.library.custom_op(
    "mlops_seqmoe::expert_backward_into", mutates_args=("dx", "dp")
)
def expert_backward_into(
    dy: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
    grad_dtype: str,
    train_base: bool,
    dx: torch.Tensor,
    dp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add this expert's VJP into caller-owned, initialized gradient buffers."""
    _validate_destinations(x, probs, dx, dp)
    if train_base:
        raise ValueError("Use expert_backward_base_into for trainable base experts")
    return tuple(
        backward_expert_into(
            dy,
            x,
            probs,
            order,
            bounds,
            weights,
            factors,
            hidden,
            capacity,
            multiple,
            scale,
            clip,
            precision,
            specs,
            grad_dtype,
            train_base,
            dx,
            dp,
            True,
        )
    )


@torch.library.custom_op(
    "mlops_seqmoe::expert_backward_base_into", mutates_args=("dx", "dp")
)
def expert_backward_base_into(
    dy: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
    grad_dtype: str,
    train_base: bool,
    dx: torch.Tensor,
    dp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add this expert's VJP into caller-owned, initialized gradient buffers."""
    _validate_destinations(x, probs, dx, dp)
    if not train_base or factors:
        raise ValueError(
            "Base destination backward expects two trainable base matrices"
        )
    return tuple(
        backward_expert_into(
            dy,
            x,
            probs,
            order,
            bounds,
            weights,
            factors,
            hidden,
            capacity,
            multiple,
            scale,
            clip,
            precision,
            specs,
            grad_dtype,
            train_base,
            dx,
            dp,
            True,
        )
    )


def _backward_into_fake(*args):
    # The final two operands are destinations; parameter-gradient shapes are
    # identical to the functional VJP. No new dx/dp are returned.
    return tuple(_backward_fake(*args[:-2])[2])


expert_backward_into.register_fake(_backward_into_fake)
expert_backward_base_into.register_fake(_backward_into_fake)
