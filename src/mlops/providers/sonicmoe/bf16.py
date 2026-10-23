"""BF16 compute payloads whose operator boundaries retain FP32 gradients."""

import torch
from torch.nn import functional

from ...expert_parallel.parameters import BF16ComputeWeight


def compute_dtype(value):
    return value._data.dtype if isinstance(value, BF16ComputeWeight) else value.dtype


def for_compute(value, dtype):
    if isinstance(value, BF16ComputeWeight) and dtype == torch.bfloat16:
        return value
    return value.to(dtype)


class _Transform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale, padding):
        ctx.shape, ctx.scale = value.shape, scale
        data = value._data
        if padding:
            data = functional.pad(data, padding)
        if scale != 1:
            data = data * scale
        return BF16ComputeWeight(data, dtype=value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        # Padding is only on the high end of each axis. Undo it in the logical
        # gradient dtype; no BF16 conversion belongs on this return path.
        result = gradient[tuple(slice(0, size) for size in ctx.shape)]
        if ctx.scale != 1:
            result = result * ctx.scale
        return result, None, None


def scaled(value, scale):
    if isinstance(value, BF16ComputeWeight):
        return value if scale == 1 else _Transform.apply(value, scale, ())
    return value * scale


def padded(value, padding):
    if isinstance(value, BF16ComputeWeight):
        return _Transform.apply(value, 1.0, tuple(padding))
    return functional.pad(value, padding)
