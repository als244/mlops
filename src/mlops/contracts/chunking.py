"""Independent axes, local coordinates and reductions owned by MLOps operators."""

from .types import AxisFrom, ChunkAxis, ChunkContract, ChunkLocal


def operator_chunk_contracts(batch_size: int = 1) -> tuple[ChunkContract, ...]:
    """Axis/reduction facts for normalization, rotary, mHC and LoRA dispatch ops.

    ``batch_size`` is the leading factor when an operator flattens batch and
    logical positions. Combine this tuple with explicit SonicMoE/fused-mHC
    contracts when those operators occur. Select input axes and chunk sizes in
    the caller; these declarations neither select nor implement chunk loops.
    """

    same = AxisFrom(0)
    flat = ChunkAxis(0, before=batch_size)
    local = ChunkLocal(0)
    specs = {
        "rms_norm_builtin_triton_fwd": ((same, AxisFrom(0, 1)), ()),
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
        "l2_norm_fla_fwd": ((same, AxisFrom(0, 1)), ()),
        "l2_norm_fla_bwd": ((same,), ()),
        "gated_rms_norm_fla_fwd": ((same, AxisFrom(0, 1)), ()),
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
        ChunkContract(f"mlops.{name}.default", "1", axes, reductions)
        for name, (axes, reductions) in specs.items()
    )


def sonicmoe_chunk_contracts(
    *, compute_precision: str = "bf16"
) -> tuple[ChunkContract, ...]:
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
        ChunkContract(
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
        ChunkContract(
            f"mlops.{name}_accumulate.default", "bf16-destinations-v1", axes, reductions
        )
        for name, (axes, reductions) in specifications.items()
        if name.endswith("_bwd")
    )
    return (*original, *accumulating)


def mhc_chunk_contracts(*, batch_size: int = 1) -> tuple[ChunkContract, ...]:
    """Token-axis contracts for fused DeepGEMM pre and TileKernels post.

    Pre takes packed ``[batch * sequence, 4, hidden]`` streams. Its invocation
    (including each tail) must satisfy the provider's 32-token/128-hidden
    alignment. Post preserves its ordinary ``[1, tokens, ...]`` coordinates.
    Pre parameter gradients sum across chunks; no coefficient normalization
    crosses tokens. This does not certify arbitrary stream-axis splitting.
    """

    flat = ChunkAxis(0, before=batch_size)
    return (
        ChunkContract("mlops.mhc_pre_deepgemm.default", "tokens-1", (flat,) * 5),
        ChunkContract(
            "mlops.mhc_pre_deepgemm_backward.default",
            "tokens-1",
            (AxisFrom(0), None, None, None),
            (1, 2, 3),
        ),
        ChunkContract("mlops.mhc_post_tilekernels.default", "tokens-1", (AxisFrom(1),)),
        ChunkContract(
            "mlops.mhc_post_tilekernels_backward.default",
            "tokens-1",
            tuple(AxisFrom(i) for i in range(1, 5)),
        ),
    )
