"""FX-based eligibility for the Sonic destination-writing backward operators."""

from __future__ import annotations

import math

import torch

from .types import (
    GradientContext,
    GradientContract,
    GradientDestination,
    GradientMapping,
)


def _path(
    item: GradientDestination, raw: torch.Tensor, *, up: bool
) -> tuple[float, bool] | None:
    remaining, scale = [], 1.0
    scale_seen = False
    neutral = {
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    }
    for node in reversed(item.operations):
        if node.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
            if (
                scale_seen
                or len(node.args) != 2
                or type(node.args[1]) not in (int, float)
            ):
                return None
            scale = float(node.args[1])
            if not math.isfinite(scale):
                return None
            scale_seen = True
        elif node.target in neutral:
            before, after = node.args[0].meta.get("val"), node.meta.get("val")
            if not isinstance(before, torch.Tensor) or not isinstance(
                after, torch.Tensor
            ):
                return None
            if (
                before.dtype != after.dtype
                or before.device != after.device
                or before.shape != after.shape
            ):
                return None
        else:
            remaining.append(node)
    if not remaining:
        return scale, False
    if not up or item.output_index != 3 or raw.ndim != 3 or raw.shape[1] % 2:
        return None
    views = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}
    if len(remaining) not in (3, 4):
        return None
    first, permutation, *tail = remaining
    if first.target not in views or permutation.target != torch.ops.aten.transpose.int:
        return None
    if set(permutation.args[1:]) not in ({1, 2}, {-3, -2}):
        return None
    if len(tail) == 2:
        if tail[0].target != torch.ops.aten.clone.default:
            return None
        tail = tail[1:]
    if tail[0].target not in views:
        return None
    experts, packed, rank = raw.shape
    if tuple(first.meta["val"].shape) != (experts, packed // 2, 2, rank):
        return None
    if tuple(permutation.meta["val"].shape) != (experts, 2, packed // 2, rank):
        return None
    if tuple(tail[0].meta["val"].shape) != tuple(raw.shape):
        return None
    return scale, True


def _mapping(context: GradientContext, *, up: bool) -> GradientMapping | None:

    if context.producer.args[-1]:
        return None  # FP8 containers require a separate contract.
    indices, scales, canonical_b = [], [], False
    outputs = context.producer.meta["val"]
    for item in context.destinations:
        raw, destination = outputs[item.output_index], item.destination.meta["val"]
        if not isinstance(raw, torch.Tensor) or not isinstance(
            destination, torch.Tensor
        ):
            continue
        if (
            raw.shape != destination.shape
            or raw.dtype != destination.dtype
            or raw.device != destination.device
        ):
            continue
        if not destination.is_contiguous() or raw.dtype not in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ):
            continue
        if raw.dtype != torch.float32 and not context.fuse_narrow_gradient_accumulation:
            continue
        resolved = _path(item, raw, up=up)
        if resolved is None:
            continue
        scale, canonical = resolved
        indices.append(item.output_index)
        scales.append(scale)
        canonical_b |= canonical
    if not indices:
        return None
    return GradientMapping(
        tuple(indices),
        {
            "gradient_scales": scales,
            "canonical_b": canonical_b,
            "fuse_narrow_gradient_accumulation": context.fuse_narrow_gradient_accumulation,
        },
    )


def _up_mapping(
    context: GradientContext,
) -> GradientMapping | None:
    return _mapping(context, up=True)


def _down_mapping(
    context: GradientContext,
) -> GradientMapping | None:
    return _mapping(context, up=False)


def sonicmoe_gradient_contracts() -> tuple[GradientContract, ...]:
    """Fuse eligible later Sonic parameter products into existing gradients.

    A compiler adapter consumes these descriptions. Contract v1 handles complete, aligned,
    contiguous same-dtype destinations, one literal scale, and canonical up-B
    layout. It declines intermediate dtype changes, crops and FP8 containers.
    Narrow gradients require ``fuse_narrow_gradient_accumulation`` permission.
    This is a build input; saved execution needs no mapper callback.
    """

    from ..providers.sonicmoe import lora_destinations  # noqa: F401

    return tuple(
        GradientContract(
            f"mlops.lora_sonicmoe_{direction}_bwd.default",
            f"mlops.lora_sonicmoe_{direction}_bwd_accumulate.default",
            (1, 2, 3),
            1,
            mapper,
        )
        for direction, mapper in (("up", _up_mapping), ("down", _down_mapping))
    )
