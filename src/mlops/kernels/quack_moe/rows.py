"""FP8 row payloads and saved-state helpers, independent of transport."""

from dataclasses import dataclass

import torch

from .quantize_rows import quantize_rows_fp8


@dataclass
class FP8Rows:
    data: torch.Tensor
    scales: torch.Tensor


def row_data(value):
    return value.data if isinstance(value, FP8Rows) else value


def quantized_rows(value):
    if isinstance(value, FP8Rows):
        return value.data, value.scales
    return quantize_rows_fp8(value)


def gradient_rows(value):
    if isinstance(value, FP8Rows):
        return value.data, value.scales
    return value, None


def mask_rows_tail(pointwise, value, end):
    pointwise.mask_tail(row_data(value), end)
    if isinstance(value, FP8Rows):
        pointwise.mask_tail(value.scales, end)


def save_rows(value):
    """Return only ordinary tensors for the custom operator's saved-state ABI."""
    data, scales = gradient_rows(value)
    return data, [] if scales is None else [scales]


def restore_rows(data, extra):
    return FP8Rows(data, extra[0]) if extra else data


def empty_bf16_rows(value):
    data = row_data(value)
    return torch.empty(data.shape, device=data.device, dtype=torch.bfloat16)
