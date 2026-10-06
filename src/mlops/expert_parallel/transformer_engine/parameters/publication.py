"""Traceable optimizer publication into Transformer Engine weight components.

GPU publication calls TE's quantizer. CPU publication supports initialization
and checkpoint restore using the same current/block scaling formulas. These
quantizers are attached only to this layer's weights; TE's installed classes
and activation quantizers are unchanged.
"""

from __future__ import annotations

import torch
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
)

from ..experts import _make_quantizer
from .components import component_names


def _cpu_quantize(source, row, column, row_scale, column_scale, precision):
    from transformer_engine.pytorch.custom_recipes.reference_current_scaling import (
        _scale_from_amax_tensor,
    )

    def quantize(value, data, scales, block):
        values = value.float()
        if block:
            # MoE expert widths are multiples of 128. TE stores vector scales
            # transposed, with padding on the untiled dimension for GEMM.
            if values.shape[-1] % 128:
                raise ValueError("Block FP8 parameter width must be divisible by 128")
            values = values.reshape(values.shape[0], -1, 128)
            amax = values.abs().amax(-1, keepdim=True)
        else:
            amax = values.abs().amax().reshape(1)
        scale, inverse, _ = _scale_from_amax_tensor(
            source.dtype,
            amax,
            torch.float8_e4m3fn,
            eps=0.0,
            pow_2_scales=block,
        )
        quantized = (values * scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        data.copy_(quantized.reshape(value.shape).view(torch.uint8))
        if block:
            scales.zero_()
            scales[:, : value.shape[0]].copy_(inverse.squeeze(-1).t())
        else:
            scales.copy_(inverse)

    quantize(source, row, row_scale, precision == "fp8_block")
    if precision == "fp8_current":
        column.copy_(row.t())
    else:
        quantize(source.t(), column, column_scale, True)


@torch.library.custom_op(
    "mlops_ep::te_publish_weight",
    mutates_args={"row", "column", "row_scale", "column_scale"},
)
def publish_weight(
    source: torch.Tensor,
    row: torch.Tensor,
    column: torch.Tensor,
    row_scale: torch.Tensor,
    column_scale: torch.Tensor | None,
    precision: str,
    noop_flag: torch.Tensor | None = None,
) -> None:
    if source.device.type == "cpu":
        if noop_flag is not None and bool(noop_flag.item()):
            return
        _cpu_quantize(source, row, column, row_scale, column_scale, precision)
        return
    quantizer = _make_quantizer(precision)
    fields = (
        (row, column, row_scale)
        if column_scale is None
        else (row, column, row_scale, column_scale)
    )
    components = dict(zip(component_names(precision), fields, strict=True))
    metadata = quantizer.create_metadata(source.shape, dtype=source.dtype)
    destination = metadata["cls"].__tensor_unflatten__(
        components,
        metadata,
        source.shape,
        source.stride(),
    )
    quantizer.update_quantized(source, destination, noop_flag=noop_flag)


@publish_weight.register_fake
def _fake_publish(
    source, row, column, row_scale, column_scale, precision, noop_flag=None
):
    return None


class _WeightPublication:
    @property
    def precision(self):
        return (
            "fp8_current"
            if isinstance(self, Float8CurrentScalingQuantizer)
            else "fp8_block"
        )

    def copy(self):
        return _wrap_quantizer(self)

    def make_empty(self, shape, **kwargs):
        # TE's C++ converter accepts its exact quantizer types. Use its
        # allocation path, then attach the weight publication behavior.
        result = _make_quantizer(self.precision).make_empty(shape, **kwargs)
        result._quantizer = self.copy()
        return result

    def quantize_impl(self, source):
        destination = self.make_empty(
            source.shape, dtype=source.dtype, device=source.device
        )
        return self.update_quantized(source, destination)

    def update_quantized(self, src, dst, *, noop_flag=None):
        precision = self.precision
        # The weight format is deliberately fixed; delayed scaling and global
        # amax reductions require their own state/collective contracts.
        if self.amax_epsilon != 0 or getattr(self, "with_amax_reduction", False):
            raise ValueError(
                "MoE weight publication requires its configured local scaling recipe"
            )
        if precision == "fp8_current" and self.force_pow_2_scales:
            raise ValueError("Current-scaled MoE weights use unrestricted scales")
        if precision == "fp8_block" and (
            self.block_scaling_dim != 1 or not self.force_pow_2_scales
        ):
            raise ValueError("Block-scaled MoE weights use 1D power-of-two scales")
        fields = [getattr(dst, name) for name in component_names(precision)]
        if len(fields) == 3:
            fields.append(None)
        publish_weight(src, *fields, precision, noop_flag)
        if precision == "fp8_current":
            dst._transpose_invalid = False
        return dst


class CurrentWeightQuantizer(_WeightPublication, Float8CurrentScalingQuantizer):
    pass


class BlockWeightQuantizer(_WeightPublication, Float8BlockQuantizer):
    pass


from transformer_engine.pytorch.dynamo.quantizer_opaque import (
    register_value_opaque_quantizer,
)

register_value_opaque_quantizer(CurrentWeightQuantizer)
register_value_opaque_quantizer(BlockWeightQuantizer)


def weight_quantizer(precision):
    return _wrap_quantizer(_make_quantizer(precision))


def _wrap_quantizer(source):
    options = {
        "rowwise": source.rowwise_usage,
        "columnwise": source.columnwise_usage,
        "amax_epsilon": source.amax_epsilon,
        "force_pow_2_scales": source.force_pow_2_scales,
    }
    if isinstance(source, Float8CurrentScalingQuantizer):
        result = CurrentWeightQuantizer(source.dtype, device="cuda", **options)
    else:
        result = BlockWeightQuantizer(
            source.dtype, block_scaling_dim=source.block_scaling_dim, **options
        )
    result.internal = source.internal
    result.optimize_for_gemm = source.optimize_for_gemm
    return result
