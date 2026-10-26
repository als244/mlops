"""Ordinary parameters, independent of live communication resources."""

import torch
from torch import nn

from ...parameters import BF16ComputeWeight
from .fp8 import QuackFP8Weight
from .initialization import compute_parameter


def local_parameters(layer, config, device):
    c = config
    router = torch.empty(
        c.num_experts, c.model_dim, device=device, dtype=c.router_dtype
    )
    router.normal_(0, c.init_std)
    layer.router_weight = nn.Parameter(
        BF16ComputeWeight(router, dtype=c.router_weight_grad_dtype)
        if c.router_dtype != c.router_weight_grad_dtype
        else router
    )
    for name, shape in (
        ("shared_gate_weight", (c.shared_width, c.model_dim)),
        ("shared_up_weight", (c.shared_width, c.model_dim)),
        ("shared_down_weight", (c.model_dim, c.shared_width)),
    ):
        value = None
        if c.shared_width:
            data = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_(
                0, c.init_std
            )
            value = compute_parameter(data, c.weight_grad_dtype)
        layer.register_parameter(name, value)


def independent_experts(layer, config, device):
    """CPU/meta construction for a module bound to a separately created runtime."""
    c = config
    for name, shape in (
        ("gate_up_weight", (c.local_experts, 2 * c.expert_hidden_dim, c.feature_dim)),
        ("down_weight", (c.local_experts, c.feature_dim, c.expert_hidden_dim)),
    ):
        value = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_(
            0, c.init_std
        )
        parameter = (
            nn.Parameter(QuackFP8Weight.quantize(value, dtype=c.weight_grad_dtype))
            if c.compute_precision == "fp8_current"
            else compute_parameter(value, c.weight_grad_dtype)
        )
        layer.register_parameter(name, parameter)
