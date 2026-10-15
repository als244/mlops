"""Block-scaled FP8 expert math for the SonicMoE implementations
(``compute_precision="fp8_block"``).

The expert weights are `BlockFP8Weight` containers (``mlops/block_fp8.py``):
E4M3 ``[E, out, in]`` with one FP32 scale per 128x128 block, stored once and
in ``nn.Linear`` row order. Inside the custom operators a container is its
components, a `BlockWeight`. The GEMMs (``expert_parallel/quack/experts/
block_fp8_gemm.py``) read the same bytes as ``W`` for a projection and, for an
input gradient, as ``Wᵀ``, transposing each E4M3 tile in shared memory; both
promote each 128-wide K chunk by its block scale. Routed rows and cotangents
are quantized per row per call by the expert-parallel path's quantizer,
gathered in the same pass (for an input gradient, in that GEMM's K order).
As in the two-copy form, the up projection writes ``h`` with gate and up
interleaved and the SwiGLU in its epilogue, and the down projection's input
gradient runs the SwiGLU backward in its epilogue. Where those kernels'
conditions fail (rows other than bf16; gate/up halves not whole 64-row
groups) the grouped Triton kernel (``kernels/block_fp8.py``) and the row
kernels serve instead.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ...dispatch.registry import SupportResult

CAPABILITY = 9  # validated on SM90
ALIGNMENT = 16  # elements per 16-byte row pitch of E4M3 operands


class BlockWeight(NamedTuple):
    """A `BlockFP8Weight`'s components: the operands of one expert weight."""

    data: torch.Tensor  # E4M3 [E, out, in], rows in nn.Linear order
    scale: torch.Tensor  # FP32 [E, ceil(out / 128), ceil(in / 128)]

    @property
    def shape(self):
        return self.data.shape


def supports(h2, w13_experts, w2_experts) -> SupportResult:
    """Whether block-scaled FP8 math applies, after the bf16 form's checks:
    SM90, Triton, and 16-element widths."""
    from importlib.util import find_spec

    if not h2.is_cuda or torch.cuda.get_device_capability(h2.device)[0] != CAPABILITY:
        return SupportResult.no(
            "fp8_block expert math is served on compute capability 9.x"
        )
    if find_spec("triton") is None:
        return SupportResult.no("fp8_block needs triton")
    if h2.shape[-1] % ALIGNMENT or w2_experts.shape[2] % ALIGNMENT:
        return SupportResult.no(
            "fp8_block needs model and expert widths that are multiples of 16"
        )
    return SupportResult.yes()


def _rows():
    from ...expert_parallel.quack.kernels.quantize_rows import quantize_rows_fp8

    return quantize_rows_fp8


def _gemm(rows, scales, weight, offsets, dtype, **options):
    from ...kernels.block_fp8 import grouped_block_gemm

    return grouped_block_gemm(
        rows,
        weight.data,
        weight.scale,
        offsets,
        row_scale=scales,
        out_dtype=dtype,
        **options,
    )


def _kernels():
    from ...expert_parallel.quack.experts import block_fp8_gemm

    return block_fp8_gemm


def _bf16(*tensors):
    return all(tensor.dtype == torch.bfloat16 for tensor in tensors)


def _fused_up(x, w13: BlockWeight):
    """Whether the up projection's GEMM serves ``x``: bf16 rows, and gate and
    up halves of whole 64-row groups (each output tile then meets one scale
    block of each half)."""
    return _bf16(x) and w13.shape[1] % 128 == 0


def _project(rows, scales, weight: BlockWeight, offsets, dtype, c=None):
    """``rows @ Wᵀ (+ c)`` per expert, the rows' scales applied."""
    if dtype == torch.bfloat16 and (c is None or _bf16(c)):
        return _kernels().block_fp8_gemm(
            rows,
            scales,
            weight.data,
            weight.scale,
            offsets=offsets,
            c=None if c is None else c.contiguous(),
        )
    options = {} if c is None else {"addend": c.contiguous()}
    return _gemm(rows, scales, weight, offsets, dtype, **options)


def up_projection(h2_flat, w13: BlockWeight, offsets, gather, limit=None):
    """Gather and quantize the routed rows, project through ``w13`` and apply
    SwiGLU (clamped at ``limit`` when given): ``(h, a)``, ``h`` interleaved."""
    from ...expert_parallel.quack.kernels.swiglu_rows import swiglu_rows

    rows, scales = _rows()(h2_flat, gather=gather)
    if _fused_up(h2_flat, w13):
        return _kernels().block_fp8_gemm_swiglu(
            rows, scales, w13.data, w13.scale, offsets=offsets, limit=limit
        )
    h = _gemm(rows, scales, w13, offsets, h2_flat.dtype, interleave_out=True)
    return h, swiglu_rows(h, limit=limit)


def project_down(a, w2: BlockWeight, offsets):
    """``y = a @ w2ᵀ`` per expert from the activation rows."""
    rows, scales = _rows()(a)
    return _project(rows, scales, w2, offsets, a.dtype)


def project_down_from_preactivation(h, w2: BlockWeight, offsets, limit=None):
    """The same from the interleaved pre-activation: SwiGLU (clamped at
    ``limit`` when given) and the row quantization in one pass."""
    from ...expert_parallel.quack.kernels.swiglu_rows import swiglu_quantize_rows_fp8

    rows, scales = swiglu_quantize_rows_fp8(h, limit=limit)
    return _project(rows, scales, w2, offsets, h.dtype)


def input_gradient_rows(
    cotangent, weight: BlockWeight, offsets, gather=None, gate_up=False
):
    """``rows @ W`` per expert: ``dx = dh @ w13`` or ``da = dy @ w2``, the rows
    gathered from ``cotangent`` by ``gather`` when given.

    The weight is read MN-major, which Hopper's E4M3 tensor cores do not take,
    so the GEMM (`block_fp8_transposed_gemm`) transposes each E4M3 tile in
    shared memory before multiplying, promoting each 128-K chunk by its block
    scale. Its E4M3 rows come from the row quantizer in the GEMM's K order,
    gathered as they are read; ``gate_up`` names ``w13``, whose interleaved
    ``dh`` the quantizer reads gate columns first, matching ``w13``'s
    gate-then-up rows. The two calls belong together: the K order is a
    contract between them."""
    from ...expert_parallel.quack.experts.block_fp8_gemm import (
        block_fp8_transposed_gemm,
    )

    rows, scales = _rows()(
        cotangent, gather=gather, transposed_gemm=True, gate_up=gate_up
    )
    return block_fp8_transposed_gemm(
        rows, scales, weight.data, weight.scale, offsets=offsets
    )


def down_backward_activations(
    grad_output_flat, h, w2: BlockWeight, score_rows, offsets, gather, limit=None
):
    """``(dh, a', dscore_rows)``: the rows' gradient through ``w2`` with the
    SwiGLU backward (clamped at ``limit``) in the GEMM's epilogue -- the score
    gradient, the interleaved pre-activation gradient and the route-scaled
    activation; for rows other than bf16, the GEMM and then one row pass."""
    from ...expert_parallel.quack.kernels.swiglu_rows import (
        routed_swiglu_rows_backward,
    )

    if _bf16(grad_output_flat, h):
        rows, scales = _rows()(grad_output_flat, gather=gather, transposed_gemm=True)
        return _kernels().block_fp8_transposed_gemm_dswiglu(
            rows,
            scales,
            w2.data,
            w2.scale,
            h,
            score_rows.float().contiguous(),
            offsets=offsets,
            limit=limit,
        )
    r = input_gradient_rows(grad_output_flat, w2, offsets, gather=gather)
    dscore, dh, a_prime = routed_swiglu_rows_backward(
        r, h, score_rows, scaled_activation=True, limit=limit
    )
    return dh, a_prime, dscore


def lora_up_projection(x, w13: BlockWeight, c, offsets, gather=None, limit=None):
    """``h = x @ w13ᵀ + c`` per expert, ``c`` the LoRA term in the interleaved
    column order, then SwiGLU (clamped at ``limit``): ``(h, a)``."""
    from ...expert_parallel.quack.kernels.swiglu_rows import swiglu_rows

    rows, scales = _rows()(x, gather=gather)
    if _fused_up(x, w13) and _bf16(c):
        return _kernels().block_fp8_gemm_swiglu(
            rows,
            scales,
            w13.data,
            w13.scale,
            offsets=offsets,
            c=c.contiguous(),
            limit=limit,
        )
    h = _gemm(
        rows, scales, w13, offsets, x.dtype, addend=c.contiguous(), interleave_out=True
    )
    return h, swiglu_rows(h, limit=limit)


def lora_down_projection(a, w2: BlockWeight, c, offsets):
    """``y = a @ w2ᵀ + c`` per expert, ``c`` the LoRA term."""
    rows, scales = _rows()(a)
    return _project(rows, scales, w2, offsets, a.dtype, c=c)


__all__ = [
    "BlockWeight",
    "down_backward_activations",
    "input_gradient_rows",
    "lora_down_projection",
    "lora_up_projection",
    "project_down",
    "project_down_from_preactivation",
    "supports",
    "up_projection",
]
