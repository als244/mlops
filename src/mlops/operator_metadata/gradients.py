"""FX-based eligibility for the Sonic destination-writing backward operators."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.fx import Node


class GradientDestination(Protocol):
    """FX output and destination presented to a provider's fusion resolver.

    ``operations`` is ordered from final gradient back toward the producer;
    the producer tuple's getitem is excluded. Tensor metadata is in ``val``.
    The resolver inspects the graph but must not modify it.
    """

    output_index: int
    destination: Node
    operations: tuple[Node, ...]


class GradientAccumulationContext(Protocol):
    producer: Node
    destinations: tuple[GradientDestination, ...]
    fuse_narrow_gradient_accumulation: bool


@dataclass(frozen=True)
class GradientAccumulationSelection:
    """Selected outputs and literal keywords for the accumulating overload."""

    output_indices: tuple[int, ...]
    keywords: Mapping[str, Any]


@dataclass(frozen=True)
class GradientAccumulation:
    """Replace one backward with its destination-writing overload once.

    The caller passes original arguments plus ``destinations``,
    ``output_indices``, ``contract_version`` and the resolved keywords. Selected
    outputs use their destinations; other outputs retain their original values.
    Returning None from the resolver declines the replacement.
    """

    target: str
    replacement: str
    output_indices: tuple[int, ...]
    version: int
    resolve: Callable[
        [GradientAccumulationContext], GradientAccumulationSelection | None
    ]


def _supported_gradient_transforms(
    destination: GradientDestination,
    raw_gradient: torch.Tensor,
    *,
    is_up_projection: bool,
) -> tuple[float, bool] | None:
    """Return (scale, canonical_up_b) when the kernel can perform every transform.

    The graph may scale, cast or reorder a raw backward output before adding it
    to the parameter gradient. Fusing the add is only valid if the accumulating
    kernel performs those same transformations. None keeps the ordinary path.
    """
    layout_operations = []
    scale = 1.0
    scale_seen = False
    value_preserving_operations = {
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    }
    # The recorded path goes from the final gradient back to the producer.
    # Inspect it in execution order, starting at the raw backward output.
    for node in reversed(destination.operations):
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
        elif node.target in value_preserving_operations:
            before = node.args[0].meta.get("val")
            after = node.meta.get("val")
            if not isinstance(before, torch.Tensor) or not isinstance(
                after, torch.Tensor
            ):
                return None
            # A cast through another dtype changes rounding, even if a later
            # cast returns to the original dtype. Do not fuse that path.
            if (
                before.dtype != after.dtype
                or before.device != after.device
                or before.shape != after.shape
            ):
                return None
        else:
            layout_operations.append(node)

    if not layout_operations:
        return scale, False
    if not is_up_projection or destination.output_index != 3:
        return None
    if not _is_canonical_up_b_layout(raw_gradient, layout_operations):
        return None
    return scale, True


def _is_canonical_up_b_layout(
    raw_gradient: torch.Tensor, operations: list[Node]
) -> bool:
    """Recognize the supported interleaved-to-gate/up LoRA B gradient reorder.

    [experts, 2 * hidden, rank] -> [experts, hidden, 2, rank]
    -> [experts, 2, hidden, rank] -> [experts, 2 * hidden, rank].
    A clone between the transpose and final view is optional.
    """
    if raw_gradient.ndim != 3 or raw_gradient.shape[1] % 2:
        return False
    if len(operations) not in (3, 4):
        return False
    view_operations = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}
    unpack, transpose, *tail = operations
    if (
        unpack.target not in view_operations
        or transpose.target != torch.ops.aten.transpose.int
    ):
        return False
    if set(transpose.args[1:]) not in ({1, 2}, {-3, -2}):
        return False
    if len(tail) == 2:
        if tail[0].target != torch.ops.aten.clone.default:
            return False
        tail = tail[1:]
    repack = tail[0]
    if repack.target not in view_operations:
        return False
    experts, packed_hidden, rank = raw_gradient.shape
    return (
        tuple(unpack.meta["val"].shape) == (experts, packed_hidden // 2, 2, rank)
        and tuple(transpose.meta["val"].shape) == (experts, 2, packed_hidden // 2, rank)
        and tuple(repack.meta["val"].shape) == tuple(raw_gradient.shape)
    )


def _select_accumulating_outputs(
    context: GradientAccumulationContext, *, is_up_projection: bool
) -> GradientAccumulationSelection | None:
    """Select supported parameter outputs together for one backward call."""
    # The final schema argument identifies a frozen FP8 base container.
    # Its backward does not support destination fusion.
    if context.producer.args[-1]:
        return None

    output_indices = []
    gradient_scales = []
    canonical_up_b = False
    outputs = context.producer.meta["val"]
    for candidate in context.destinations:
        raw_gradient = outputs[candidate.output_index]
        buffer = candidate.destination.meta["val"]
        if not isinstance(raw_gradient, torch.Tensor) or not isinstance(
            buffer, torch.Tensor
        ):
            continue
        if (
            raw_gradient.shape != buffer.shape
            or raw_gradient.dtype != buffer.dtype
            or raw_gradient.device != buffer.device
        ):
            continue
        if not buffer.is_contiguous() or raw_gradient.dtype not in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ):
            continue
        if (
            raw_gradient.dtype != torch.float32
            and not context.fuse_narrow_gradient_accumulation
        ):
            continue

        transforms = _supported_gradient_transforms(
            candidate, raw_gradient, is_up_projection=is_up_projection
        )
        if transforms is None:
            continue
        scale, reorder_up_b = transforms
        output_indices.append(candidate.output_index)
        gradient_scales.append(scale)
        canonical_up_b |= reorder_up_b

    if not output_indices:
        return None
    return GradientAccumulationSelection(
        output_indices=tuple(output_indices),
        keywords={
            "gradient_scales": gradient_scales,
            "canonical_b": canonical_up_b,
            "fuse_narrow_gradient_accumulation": context.fuse_narrow_gradient_accumulation,
        },
    )


def _select_up_projection_outputs(
    context: GradientAccumulationContext,
) -> GradientAccumulationSelection | None:
    return _select_accumulating_outputs(context, is_up_projection=True)


def _select_down_projection_outputs(
    context: GradientAccumulationContext,
) -> GradientAccumulationSelection | None:
    return _select_accumulating_outputs(context, is_up_projection=False)


def sonicmoe_gradient_accumulation() -> tuple[GradientAccumulation, ...]:
    """Describe Sonic backwards that can add products directly into gradients.

    Version 1 supports complete, aligned, contiguous same-dtype buffers, one
    literal scale, and canonical up-B layout. Intermediate dtype changes, crops
    and FP8 containers keep the ordinary backward and addition. Narrow gradients
    require explicit permission. Saved execution needs no selection callback.
    """
    # Register the accumulating overloads; no kernel compilation or CUDA access.
    from ..providers.sonicmoe import lora_destinations  # noqa: F401

    return (
        GradientAccumulation(
            target="mlops.lora_sonicmoe_up_bwd.default",
            replacement="mlops.lora_sonicmoe_up_bwd_accumulate.default",
            output_indices=(1, 2, 3),
            version=1,
            resolve=_select_up_projection_outputs,
        ),
        GradientAccumulation(
            target="mlops.lora_sonicmoe_down_bwd.default",
            replacement="mlops.lora_sonicmoe_down_bwd_accumulate.default",
            output_indices=(1, 2, 3),
            version=1,
            resolve=_select_down_projection_outputs,
        ),
    )
