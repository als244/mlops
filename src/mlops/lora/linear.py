"""Low-rank dense projections without constructing a dense weight update."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..lora_head import lora_head_loss


class LoRALinear(nn.Module):
    """Reuse a Linear's parameter objects and names; add A and zero-initialized B.

    apply_lora controls base trainability. Factors are stored at factor_dtype
    and compute at the activation dtype, with ordinary autograd through casts.
    """

    def __init__(self, base, config):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.lora_config = config
        self._base_reset = type(base).reset_parameters
        if hasattr(base, "initializer_range"):
            self.initializer_range = base.initializer_range
        options = {"device": base.weight.device, "dtype": config.factor_dtype}
        self.lora_a = nn.Parameter(
            torch.empty(config.rank, self.in_features, **options)
        )
        self.lora_b = nn.Parameter(
            torch.empty(self.out_features, config.rank, **options)
        )
        self.reset_lora_parameters()
        self.train(base.training)

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def reset_parameters(self):
        # Used only when a caller explicitly initializes/materializes a model.
        # Conversion itself never resets or copies the existing base weights.
        self._base_reset(self)
        self.reset_lora_parameters()

    def forward(self, inputs):
        inner = F.linear(inputs, self.lora_a.to(inputs.dtype))
        update = F.linear(inner, self.lora_b.to(inputs.dtype))
        return (
            F.linear(inputs, self.weight, self.bias) + self.lora_config.scale * update
        )


class LoRAHead(LoRALinear):
    """Logits and bounded loss use the same low-rank update."""

    def loss(
        self, hidden, targets, *, chunk_size=None, valid_rows=None, reduction="mean"
    ):
        return lora_head_loss(
            hidden,
            self.weight,
            self.lora_a,
            self.lora_b,
            targets,
            scale=self.lora_config.scale,
            chunk_size=chunk_size,
            valid_rows=valid_rows,
            reduction=reduction,
        )
