"""Exact-K per-expert FP8 weight-gradient implementation for QuACK MoE.

One ordinary QuACK GEMM per expert bypasses the varlen-K layout restriction.
Token counts remain exact. Only the physical row pitch is aligned to 16 bytes.
The fused quantizer implements QuACK's per-row math; no TE quantizer is used.
"""

import torch
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import ColVecLoad, RowVecLoad

from mlops.expert_parallel.quack.experts.policy import FP8_SM90_CONFIG
from mlops.expert_parallel.quack.kernels.quantize_transpose import (
    quantize_transpose_fp8,
)


@gemm_epilogue(ops={"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws")})
def scaled_fp8(acc, xs, ws):
    return {"D": acc * xs * ws}


def quantize_transposed_tokens(value, *, poison_padding=False, row_scales=None):
    """Quantize token-major BF16 directly to feature-major FP8 and feature scales."""
    return quantize_transpose_fp8(
        value, poison_padding=poison_padding, row_scales=row_scales
    )


def expert_weight_gradient(
    x,
    dy,
    *,
    out=None,
    poison_padding=False,
    tuned=False,
    config=None,
    x_scales=None,
    dy_scales=None,
):
    """Return one FP32 dW = dy.T @ x using FP8 activations, with exact K."""
    if x.shape[0] != dy.shape[0]:
        raise ValueError("Mismatched expert token counts")
    if out is None:
        out = torch.empty(
            (dy.shape[1], x.shape[1]), device=x.device, dtype=torch.float32
        )
    if not x.shape[0]:
        out.zero_()
        return out
    qdy, sdy = quantize_transposed_tokens(
        dy, poison_padding=poison_padding, row_scales=dy_scales
    )
    qx, sx = quantize_transposed_tokens(
        x, poison_padding=poison_padding, row_scales=x_scales
    )
    scaled_fp8(
        qdy,
        qx.T,
        out={"D": out},
        xs=sdy,
        ws=sx,
        tuned=tuned,
        config=None if tuned else (config or FP8_SM90_CONFIG),
    )
    return out
