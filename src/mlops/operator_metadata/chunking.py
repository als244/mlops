"""Independent axes, local coordinates and reductions owned by MLOps operators."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AxisFromInput:
    """Inherit an input's logical axis after removing trailing feature dims."""

    input_index: int
    trailing_dims: int = 0

    def __post_init__(self) -> None:
        if any(
            type(x) is not int or x < 0 for x in (self.input_index, self.trailing_dims)
        ):
            raise ValueError(
                "input indices and trailing dimensions must be nonnegative integers"
            )


@dataclass(frozen=True)
class FactoredAxis:
    """Physical dimension ordered as ``before * logical_extent * after``."""

    dim: int
    before: int = 1
    after: int = 1

    def __post_init__(self) -> None:
        if any(type(x) is not int for x in (self.dim, self.before, self.after)):
            raise TypeError("axis dimensions and factors must be integers")
        if self.before < 1 or self.after < 1:
            raise ValueError("axis factors must be positive")


@dataclass(frozen=True)
class ChunkLocalOutput:
    """Invocation-local coordinates; cannot escape as an ordinary full output."""

    dim: int | None = None

    def __post_init__(self) -> None:
        if self.dim is not None and type(self.dim) is not int:
            raise TypeError("a local dimension must be an integer or None")


@dataclass(frozen=True)
class OperatorChunking:
    """Output-axis and additive-reduction semantics for one dispatcher target.

    ``None`` denotes a shared output, or an additive result when its position is
    listed in ``reductions``. These facts do not choose axes or chunk sizes for
    the caller's graph and do not implement an execution loop.
    """

    target: str
    version: str
    output_axes: tuple[AxisFromInput | FactoredAxis | ChunkLocalOutput | None, ...]
    reductions: tuple[int, ...] = ()


def operator_chunking(batch_size: int = 1) -> tuple[OperatorChunking, ...]:
    """Axis/reduction facts for normalization, rotary, mHC and LoRA dispatch ops.

    ``batch_size`` is the leading factor when an operator flattens batch and
    logical positions. Combine this tuple with explicit SonicMoE/fused-mHC
    descriptions when those operators occur. Select input axes and chunk sizes in
    the caller; these declarations neither select nor implement chunk loops.
    """

    same = AxisFromInput(0)
    flat = FactoredAxis(0, before=batch_size)
    local = ChunkLocalOutput(0)
    # Each entry is (output axes in return order, indices of summed outputs).
    specifications = {
        "rms_norm_builtin_triton_fwd": ((same, AxisFromInput(0, 1)), ()),
        "rms_norm_builtin_triton_bwd": ((same, None), (1,)),
        "swiglu_builtin_triton_fwd": ((same,), ()),
        "swiglu_builtin_triton_bwd": ((same, same), ()),
        "packed_swiglu_builtin_triton_fwd": ((flat,), ()),
        "packed_swiglu_builtin_triton_bwd": ((flat,), ()),
        "partial_rope_builtin_triton_fwd": ((same,), ()),
        "partial_rope_builtin_triton_bwd": ((same,), ()),
        "rope_builtin_triton_table_fwd": ((same,), ()),
        "rope_builtin_triton_table_bwd": ((same,), ()),
        "rope_builtin_triton_analytic_fwd": ((same,), ()),
        "rope_builtin_triton_analytic_bwd": ((same,), ()),
        "l2_norm_fla_fwd": ((same, AxisFromInput(0, 1)), ()),
        "l2_norm_fla_bwd": ((same,), ()),
        "gated_rms_norm_fla_fwd": ((same, AxisFromInput(0, 1)), ()),
        "gated_rms_norm_fla_bwd": ((same, same, None), (2,)),
        "mhc_mix_four": ((flat,), ()),
        "mhc_post_four": ((flat,), ()),
        "mhc_post_four_input_grads": ((flat, flat), ()),
        "mhc_post_four_coefficient_grads": ((flat, flat), ()),
        "lora_router_logits": ((flat,), ()),
        "lora_router_logits_bwd": ((flat, None), (1,)),
        "lora_route": ((flat, flat), ()),
        "lora_route_bwd": ((flat,), ()),
        # Prefix counts add across chunks; sort/slot indices stay chunk-local.
        "lora_assignment_layout": ((local, None, local), (1,)),
        "lora_dispatch": ((local,), ()),
        "lora_dispatch_bwd": ((flat,), ()),
        "lora_grouped_linear_fwd": ((local,), ()),
        "lora_grouped_linear_bwd": ((local, None), (1,)),
        "lora_combine": ((flat,), ()),
        "lora_combine_bwd": ((local, flat), ()),
    }
    return tuple(
        OperatorChunking(
            target=f"mlops.{name}.default",
            version="1",
            output_axes=axes,
            reductions=reductions,
        )
        for name, (axes, reductions) in specifications.items()
    )


def sonicmoe_chunking(
    *, compute_precision: str = "bf16"
) -> tuple[OperatorChunking, ...]:
    """Describe chunk-local Sonic rows and additive parameter reductions.

    Combine with rules for the other operators in a selected region. The
    caller chooses the logical input axis and chunk size. Sorted rows and
    routing maps are chunk-local under these rules; expert counts reduce across
    chunks. Retaining local state across the enclosing boundary would require a
    segmented representation with each chunk's metadata, which is not supplied
    by these output descriptions.
    ``compute_precision="fp8_block"`` describes frozen block-FP8 experts with
    BF16 activations/LoRA factors. Quantization reduces features within each
    row, so token slicing preserves that domain. FP8 weight components remain
    shared. FP8 destination fusion is deliberately excluded; ordinary additive
    gradient accumulation remains available. These rules do not certify replay,
    RNG, trainable FP8 weights, current-scaled FP8 or distributed expert routing.
    """

    if compute_precision not in {"bf16", "fp8_block"}:
        raise ValueError("Sonic chunking supports bf16 or frozen fp8_block")
    local = ChunkLocalOutput(0)
    specifications = {
        "sonicmoe_routing_metadata": ((None, local, local, local), (0,)),
        "lora_sonicmoe_up_fwd": ((local, local, local), ()),
        "lora_sonicmoe_up_bwd": ((AxisFromInput(1), None, None, None), (1, 2, 3)),
        "lora_sonicmoe_down_fwd": ((AxisFromInput(10), local), ()),
        "lora_sonicmoe_down_bwd": (
            (local, None, None, None, AxisFromInput(9)),
            (1, 2, 3),
        ),
    }
    original = tuple(
        OperatorChunking(
            target=f"mlops.{name}.default",
            version="bf16-lora-candidate-1"
            if compute_precision == "bf16"
            else "fp8-block-lora-1",
            output_axes=axes,
            reductions=reductions,
        )
        for name, (axes, reductions) in specifications.items()
    )
    if compute_precision == "fp8_block":
        return original
    accumulating = tuple(
        OperatorChunking(
            target=f"mlops.{name}_accumulate.default",
            version="bf16-destinations-v1",
            output_axes=axes,
            reductions=reductions,
        )
        for name, (axes, reductions) in specifications.items()
        if name.endswith("_bwd")
    )
    return (*original, *accumulating)


def mhc_chunking(*, batch_size: int = 1) -> tuple[OperatorChunking, ...]:
    """Token-axis descriptions for fused DeepGEMM pre and TileKernels post.

    Pre takes packed ``[batch * sequence, 4, hidden]`` streams. Its invocation
    (including each tail) must satisfy the provider's 32-token/128-hidden
    alignment. Post preserves its ordinary ``[1, tokens, ...]`` coordinates.
    Pre parameter gradients sum across chunks; no coefficient normalization
    crosses tokens. This does not certify arbitrary stream-axis splitting.
    """

    flat = FactoredAxis(0, before=batch_size)
    return (
        OperatorChunking(
            target="mlops.mhc_pre_deepgemm.default",
            version="tokens-1",
            output_axes=(flat,) * 5,
        ),
        OperatorChunking(
            target="mlops.mhc_pre_deepgemm_backward.default",
            version="tokens-1",
            output_axes=(AxisFromInput(0), None, None, None),
            reductions=(1, 2, 3),
        ),
        OperatorChunking(
            target="mlops.mhc_post_tilekernels.default",
            version="tokens-1",
            output_axes=(AxisFromInput(1),),
        ),
        OperatorChunking(
            target="mlops.mhc_post_tilekernels_backward.default",
            version="tokens-1",
            output_axes=tuple(AxisFromInput(i) for i in range(1, 5)),
        ),
    )
