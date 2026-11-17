"""LoRA factors owned by a module, independently of live expert banks."""

import torch

from ..parameters.initialization import compute_parameter


@torch.no_grad()
def initialize(a, b, lora):
    initial = torch.empty(
        a.shape,
        dtype=lora.initialization_dtype,
        device="meta" if a.is_meta else "cpu",
    )
    for expert in initial:
        torch.nn.init.xavier_uniform_(expert)
    a.copy_(initial)
    b.zero_()


def independent_factors(layer, config, lora, device):
    for prefix, out_features, in_features in (
        ("gate_up", 2 * config.expert_hidden_dim, config.feature_dim),
        ("down", config.feature_dim, config.expert_hidden_dim),
    ):
        a = torch.empty(
            (config.local_experts, lora.rank, in_features),
            dtype=lora.compute_dtype,
            device=device,
        )
        b = torch.empty(
            (config.local_experts, out_features, lora.rank),
            dtype=lora.compute_dtype,
            device=device,
        )
        initialize(a, b, lora)
        for suffix, value in (("a", a), ("b", b)):
            layer.register_parameter(
                f"lora_{prefix}_{suffix}", compute_parameter(value, lora.gradient_dtype)
            )
