"""MoE weight-representation compatibility and explicit graph-input flattening."""

import torch
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockwiseQTensor,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

from mlops.expert_parallel.transformer_engine.parameters.bf16 import BF16ComputeWeight

FP8_TYPES = (Float8Tensor, Float8BlockwiseQTensor)
COMPUTE_WEIGHT_TYPES = (*FP8_TYPES, BF16ComputeWeight)

# This order is the physical custom-op ABI. AOT may flatten the outer wrapper
# in a different order; accessing named components preserves that mapping.
_COMPONENT_NAMES = {
    "fp8_current": ("_data", "_transpose", "_scale_inv"),
    "fp8_block": (
        "_rowwise_data",
        "_columnwise_data",
        "_rowwise_scale_inv",
        "_columnwise_scale_inv",
    ),
}


def component_names(precision):
    """Every TE weight component consumed by the configured compute format."""
    try:
        return _COMPONENT_NAMES[precision]
    except KeyError as exc:
        raise ValueError(f"No FP8 component contract for {precision}") from exc


def explicit_weight_components(weights, precision):
    """Expose real component tensors without dequantizing or allocating payloads.

    Logical autograd remains attached to the outer weight. These physical bytes
    are the lower-level operator inputs, not separately trainable parameters.
    """
    import transformer_engine_torch as tex

    expected_type = (
        Float8Tensor if precision == "fp8_current" else Float8BlockwiseQTensor
    )
    names = component_names(precision)
    result = []
    for index, weight in enumerate(weights):
        if not isinstance(weight, expected_type):
            raise TypeError(f"FP8 weight {index} must be {expected_type.__name__}")
        if weight._fp8_dtype != tex.DType.kFloat8E4M3:
            raise ValueError(
                "This MoE integration requires the configured E4M3 compute format"
            )
        if precision == "fp8_current" and weight._transpose_invalid:
            raise ValueError("FP8 weight requires a valid explicit transpose")
        if precision == "fp8_block" and weight._is_2D_scaled:
            raise ValueError(
                "The configured block FP8 format uses one-dimensional scaling"
            )
        present, _ = weight.__tensor_flatten__()
        if set(present) != set(names):
            raise ValueError(
                f"FP8 weight {index} component set differs: {present} versus {names}"
            )
        result.extend(getattr(weight, name) for name in names)
    return result


# The MoE integration has fixed weight geometry. These descriptors expose the wrapper's
# own metadata without consulting an as-yet-untracked inner tensor during export.
for cls in COMPUTE_WEIGHT_TYPES:
    for name in ("shape", "is_cuda", "is_cpu", "device"):
        setattr(cls, name, getattr(torch.Tensor, name))
if not getattr(Float8Tensor, "_moon_metadata_complete", False):
    _original_metadata = Float8Tensor.get_metadata

    def _metadata(self):
        return {
            **_original_metadata(self),
            "transpose_invalid": self._transpose_invalid,
        }

    Float8Tensor.get_metadata = _metadata
    Float8Tensor._moon_metadata_complete = True


def rowwise_payload(value):
    if isinstance(value, BF16ComputeWeight):
        return value._data
    if isinstance(value, Float8Tensor):
        return value._data
    if isinstance(value, Float8BlockwiseQTensor):
        return value._rowwise_data
    return value


def flatten_compute_inputs(values):
    """Match AOT's fixed-shape wrapper flattening, without copying any storage."""
    result = []
    for value in values:
        if isinstance(value, COMPUTE_WEIGHT_TYPES):
            result.extend(
                getattr(value, name) for name in value.__tensor_flatten__()[0]
            )
        else:
            result.append(value)
    return tuple(result)


# Historical probe imports remain valid; this now handles every compute wrapper.
flatten_fp8_inputs = flatten_compute_inputs
