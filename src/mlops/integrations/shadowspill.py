"""Caller-owned MLOps contracts for ShadowSpill's generic build APIs.

ShadowSpill is optional and imported only when a factory or mapper is called.
These contracts do not select a model, checkpoint cut, execution region, chunk
size, retention policy or device. Unsupported gradient paths keep the ordinary
accumulation path. The factories neither initialize CUDA nor prepare kernels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .fla_lifetimes import fla_execution_lifetimes

if TYPE_CHECKING:
    from shadowspill import OperatorPreparation
    from shadowspill.pytorch import (
        GradientDestination,
        GradientDestinationContext,
        GradientDestinationMapping,
        GradientDestinationRule,
        OperatorRule,
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


def _mapping(
    context: GradientDestinationContext, *, up: bool
) -> GradientDestinationMapping | None:
    from shadowspill.pytorch import GradientDestinationMapping

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
    return GradientDestinationMapping(
        tuple(indices),
        ((),) * len(indices),
        {
            "gradient_scales": scales,
            "canonical_b": canonical_b,
            "fuse_narrow_gradient_accumulation": context.fuse_narrow_gradient_accumulation,
        },
    )


def _up_mapping(
    context: GradientDestinationContext,
) -> GradientDestinationMapping | None:
    return _mapping(context, up=True)


def _down_mapping(
    context: GradientDestinationContext,
) -> GradientDestinationMapping | None:
    return _mapping(context, up=False)


def sonicmoe_gradient_destination_rules() -> tuple[GradientDestinationRule, ...]:
    """Fuse eligible later Sonic parameter products into existing gradients.

    Pass this tuple as ``gradient_destination_rules`` to ``build_step`` or
    ``build_forward_backward``. Contract v1 handles complete, aligned,
    contiguous same-dtype destinations, one literal scale, and canonical up-B
    layout. It declines intermediate dtype changes, crops and FP8 containers.
    Narrow gradients require ``fuse_narrow_gradient_accumulation`` permission.
    This is a build input; saved execution needs no mapper callback.
    """
    from shadowspill.pytorch import GradientDestinationRule

    from ..providers.sonicmoe import lora_destinations  # noqa: F401

    return tuple(
        GradientDestinationRule(
            f"mlops.lora_sonicmoe_{direction}_bwd.default",
            f"mlops.lora_sonicmoe_{direction}_bwd_accumulate.default",
            (1, 2, 3),
            1,
            mapper,
        )
        for direction, mapper in (("up", _up_mapping), ("down", _down_mapping))
    )


def sonicmoe_chunk_rules(
    *, compute_precision: str = "bf16"
) -> tuple[OperatorRule, ...]:
    """Describe chunk-local Sonic rows and additive parameter reductions.

    Combine with rules for the other operators in a selected region. The
    caller chooses the logical input axis and chunk size. Sorted rows and
    routing maps are chunk-local under these rules; expert counts reduce across
    chunks. Retaining local state across the enclosing boundary would require a
    segmented representation with each chunk's metadata, which is not supplied
    by this contract.
    ``compute_precision="fp8_block"`` describes frozen block-FP8 experts with
    BF16 activations/LoRA factors. Quantization reduces features within each
    row, so token slicing preserves that domain. FP8 weight components remain
    shared. FP8 destination fusion is deliberately excluded; ordinary additive
    gradient accumulation remains available. These rules do not certify replay,
    RNG, trainable FP8 weights, current-scaled FP8 or distributed expert routing.
    """
    from shadowspill.pytorch import AxisFrom, ChunkLocal, OperatorRule

    if compute_precision not in {"bf16", "fp8_block"}:
        raise ValueError("Sonic chunk contracts support bf16 or frozen fp8_block")
    local = ChunkLocal(0)
    specifications = {
        "sonicmoe_routing_metadata": ((None, local, local, local), (0,)),
        "lora_sonicmoe_up_fwd": ((local, local, local), ()),
        "lora_sonicmoe_up_bwd": ((AxisFrom(1), None, None, None), (1, 2, 3)),
        "lora_sonicmoe_down_fwd": ((AxisFrom(10), local), ()),
        "lora_sonicmoe_down_bwd": (
            (local, None, None, None, AxisFrom(9)),
            (1, 2, 3),
        ),
    }
    original = tuple(
        OperatorRule(
            f"mlops.{name}.default",
            "bf16-lora-candidate-1"
            if compute_precision == "bf16"
            else "fp8-block-lora-1",
            axes,
            reductions,
        )
        for name, (axes, reductions) in specifications.items()
    )
    if compute_precision == "fp8_block":
        return original
    accumulating = tuple(
        OperatorRule(
            f"mlops.{name}_accumulate.default", "bf16-destinations-v1", axes, reductions
        )
        for name, (axes, reductions) in specifications.items()
        if name.endswith("_bwd")
    )
    return (*original, *accumulating)


def mhc_chunk_rules(*, batch_size: int = 1) -> tuple[OperatorRule, ...]:
    """Token-axis contracts for fused DeepGEMM pre and TileKernels post.

    Pre takes packed ``[batch * sequence, 4, hidden]`` streams. Its invocation
    (including each tail) must satisfy the provider's 32-token/128-hidden
    alignment. Post preserves its ordinary ``[1, tokens, ...]`` coordinates.
    Pre parameter gradients sum across chunks; no coefficient normalization
    crosses tokens. This does not certify arbitrary stream-axis splitting.
    """
    from shadowspill.pytorch import AxisFrom, ChunkAxis, OperatorRule

    flat = ChunkAxis(0, before=batch_size)
    return (
        OperatorRule("mlops.mhc_pre_deepgemm.default", "tokens-1", (flat,) * 5),
        OperatorRule(
            "mlops.mhc_pre_deepgemm_backward.default",
            "tokens-1",
            (AxisFrom(0), None, None, None),
            (1, 2, 3),
        ),
        OperatorRule("mlops.mhc_post_tilekernels.default", "tokens-1", (AxisFrom(1),)),
        OperatorRule(
            "mlops.mhc_post_tilekernels_backward.default",
            "tokens-1",
            tuple(AxisFrom(i) for i in range(1, 5)),
        ),
    )


def sonicmoe_preparation(*, cache_root: str | None = None) -> OperatorPreparation:
    """Package BF16 Sonic kernels and their selected Quack configurations.

    Pass this description through ``operator_preparation`` on the build API or
    shared search build options. Construction is CPU-only. The provider owns
    dispatch, source validation and task-local host caches; the engine treats
    its manifests as opaque. FP8 and expert parallelism are not covered.
    """
    from shadowspill import OperatorPreparation

    return OperatorPreparation(
        "mlops.preparation.sonic:factory",
        {} if cache_root is None else {"cache_root": cache_root},
    )


def operator_preparations(
    *, dtype: torch.dtype = torch.bfloat16, cache_root: str | None = None
) -> tuple[OperatorPreparation, ...]:
    """Descriptions for the supported runtimes in ordinary MLOps dispatch.

    Covers BF16 Sonic and the Hopper sparse-forward backend/LSE selection.
    Other MLOps providers retain their ordinary behavior; this is not a claim
    that every external compiler is prepared. Device selection is deferred to build;
    constructing descriptions neither imports optional kernels nor queries CUDA.
    For a caller explicitly selecting Sonic, use ``sonicmoe_preparation``.
    """
    from shadowspill import OperatorPreparation

    if dtype != torch.bfloat16:
        return ()
    return (
        OperatorPreparation(
            "mlops.preparation.available:factory",
            {} if cache_root is None else {"cache_root": cache_root},
        ),
        OperatorPreparation(
            "mlops.providers.builtin.sparse_latent_preparation:factory", {}
        ),
    )


__all__ = [
    "fla_execution_lifetimes",
    "mhc_chunk_rules",
    "operator_preparations",
    "sonicmoe_chunk_rules",
    "sonicmoe_gradient_destination_rules",
    "sonicmoe_preparation",
]
