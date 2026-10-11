"""Frozen compressed projections with independently selected GEMM precision."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .storage import DENSE, WeightSpec, materialize


@torch.library.custom_op("mlops_seqmoe::decode_weight", mutates_args=())
def decode_weight(
    parts: list[torch.Tensor], spec: str, dtype: torch.dtype
) -> torch.Tensor:
    return materialize(parts, WeightSpec.parse(spec), dtype)


@decode_weight.register_fake
def _decode_fake(parts, spec, dtype):
    spec = WeightSpec.parse(spec)
    return torch.empty((spec.rows, spec.cols), dtype=dtype, device=parts[0].device)


class FrozenLinear(nn.Module):
    """Preserve checkpoint components; decode one projection when it executes.

    These weights are frozen. Ordinary autograd still computes input gradients,
    allowing a trainable LoRA branch to be composed around this module.
    """

    def __init__(self, parts, spec, *, gemm_precision="bf16"):
        super().__init__()
        if spec.format in DENSE:
            raise ValueError("Use nn.Linear for ordinary floating-point weights")
        if gemm_precision not in ("bf16", "fp16"):
            raise ValueError("FrozenLinear currently supports BF16/FP16 GEMMs")
        spec.validate(parts)
        self.spec = spec
        self.gemm_precision = gemm_precision
        self.in_features, self.out_features = spec.cols, spec.rows
        self.components = nn.ParameterList(
            nn.Parameter(value, requires_grad=False) for value in parts
        )

    def forward(self, x):
        dtype = DENSE[self.gemm_precision]
        if x.dtype != dtype:
            raise ValueError(
                f"Expected {dtype} activations for {self.gemm_precision} GEMMs"
            )
        weight = decode_weight(list(self.components), self.spec.encode(), dtype)
        return F.linear(x, weight)


class FrozenLoRALinear(FrozenLinear):
    """Train a low-rank branch around a compressed, frozen projection.

    Physical checkpoint components remain separate inputs. The base matrix is
    decoded only while its projection executes; no persistent dense copy or
    dense low-rank weight update is constructed.
    """

    def __init__(self, base, config):
        super().__init__(
            list(base.components), base.spec, gemm_precision=base.gemm_precision
        )
        self.lora_config = config
        options = {"device": self.components[0].device, "dtype": config.factor_dtype}
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

    def forward(self, x):
        inner = F.linear(x, self.lora_a.to(x.dtype))
        update = F.linear(inner, self.lora_b.to(x.dtype))
        return super().forward(x) + self.lora_config.scale * update
