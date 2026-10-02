"""Explicit QuACK compute-weight components exposed to graph capture."""

import torch

from ...parameters import BF16ComputeWeight
from .fp8 import COMPONENTS, QuackFP8Weight

FP8_TYPES = (QuackFP8Weight,)
COMPUTE_WEIGHT_TYPES = (QuackFP8Weight, BF16ComputeWeight)


def component_names(precision):
    if precision != "fp8_current":
        raise ValueError(f"Unsupported QuACK FP8 precision: {precision}")
    return COMPONENTS


for cls in COMPUTE_WEIGHT_TYPES:
    for name in ("shape", "is_cuda", "is_cpu", "device"):
        setattr(cls, name, getattr(torch.Tensor, name))


def rowwise_payload(value):
    return value._data if isinstance(value, COMPUTE_WEIGHT_TYPES) else value


def flatten_compute_inputs(values):
    result = []
    for value in values:
        if isinstance(value, COMPUTE_WEIGHT_TYPES):
            result.extend(
                getattr(value, name) for name in value.__tensor_flatten__()[0]
            )
        else:
            result.append(value)
    return tuple(result)


flatten_fp8_inputs = flatten_compute_inputs
