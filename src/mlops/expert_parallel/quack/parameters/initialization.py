"""Initialize only this rank's expert shards and publish their compute storage.

Temporary FP32 initialization and FP8 quantization run on CPU. The returned
parameter wraps the bank's storage unless banks are shared between layers. Shared
banks require independent parameter storage; no master weights remain here.
"""

import torch
from torch import nn

from ...parameters import BF16ComputeWeight
from .fp8 import QuackFP8Weight


def compute_parameter(data, gradient_dtype):
    """Wrap BF16 compute storage when its logical gradient dtype differs."""
    return nn.Parameter(
        BF16ComputeWeight(data, dtype=gradient_dtype)
        if gradient_dtype != torch.bfloat16
        else data
    )


@torch.no_grad()
def initialize_expert_parameter(config, bank):
    shape = (config.local_experts, bank.out_features, bank.in_features)
    initial = (
        torch.randn(shape, device="cpu", dtype=torch.float32) * config.init_std
    ).bfloat16()
    if config.compute_precision == "bf16":
        data = bank.weight_state.parameter_data
        if config.share_expert_banks:
            data = initial.to(data.device)
        else:
            data.copy_(initial)
        return compute_parameter(data, config.weight_grad_dtype)

    from quack.gemm_w4 import quantize_act_per_token_fp8

    rows, row_scales = quantize_act_per_token_fp8(initial.flatten(0, 1))
    columns, column_scales = quantize_act_per_token_fp8(
        initial.transpose(-1, -2).reshape(-1, shape[-2])
    )
    values = (
        rows.reshape(shape),
        columns.reshape(shape[0], shape[2], shape[1]),
        row_scales.reshape(shape[:2]),
        column_scales.reshape(shape[0], shape[2]),
    )
    components = bank.weight_state.parameter_components
    if config.share_expert_banks:
        components = tuple(
            value.to(destination.device)
            for destination, value in zip(components, values)
        )
    else:
        for destination, value in zip(components, values):
            destination.copy_(value)
    return nn.Parameter(QuackFP8Weight(*components, dtype=config.weight_grad_dtype))
