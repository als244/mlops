"""One expert's parameters, checkpoint import and normal/LoRA invocation."""

import math

import torch
from torch import nn

from ..config import SequentialMoEConfig
from ..precision import validate_weights
from ..precision.parameters import parameter, payload
from ..precision.quantized_parameter import QuantizedWeight
from ..precision.storage import DENSE, WeightSpec, pack
from .autograd import expert as expert_op


class SequentialExpert(nn.Module):
    def __init__(self, config: SequentialMoEConfig, device, *, lora: bool):
        super().__init__()
        self.config, self.lora = config, lora
        self.specs = {}
        d, h, r = config.hidden_size, config.expert_size, config.rank
        dtype = config.init_dtype or config.activation_dtype
        for name, shape in (("gate_up", (2 * h, d)), ("down", (d, h))):
            value = torch.empty(shape, device=device, dtype=dtype)
            nn.init.normal_(
                value, std=config.initializer_range or 1 / math.sqrt(shape[1])
            )
            parts, spec = pack(
                value, config.weight_storage, nvfp4_block_rows=config.nvfp4_block_rows
            )
            self._install(name, parts, spec)
        for name, shape in {
            "gate_up_a": (r, d),
            "gate_up_b": (2 * h, r),
            "down_a": (r, h),
            "down_b": (d, r),
        }.items():
            if not lora:
                self.register_parameter(name, None)
                continue
            data = torch.empty(shape, device=device, dtype=config.factor_dtype)
            if name.endswith("_a"):
                nn.init.kaiming_uniform_(data, a=math.sqrt(5))
            else:
                nn.init.zeros_(data)
            setattr(self, name, parameter(data, config.gradient_dtype))

    def _install(self, name, parts, spec):
        self.specs[name] = spec
        if spec.format in DENSE:
            p = (
                nn.Parameter(parts[0], requires_grad=False)
                if self.lora
                else parameter(parts[0], self.config.gradient_dtype)
            )
            setattr(self, name, p)
        elif self.lora:
            for suffix, value in zip(
                ("", "_block_scale", "_global_scale"), parts, strict=False
            ):
                setattr(self, name + suffix, nn.Parameter(value, requires_grad=False))
        else:
            setattr(
                self,
                name,
                nn.Parameter(
                    QuantizedWeight(
                        parts, spec.format, self.config.gradient_dtype, spec=spec
                    )
                ),
            )

    def base_tensors(self):
        result = []
        for name in ("gate_up", "down"):
            value = getattr(self, name)
            if isinstance(value, QuantizedWeight):
                result.extend(value.components())
            else:
                result.append(payload(value))
                if self.specs[name].count > 1:
                    result.append(getattr(self, name + "_block_scale"))
                if self.specs[name].count > 2:
                    result.append(getattr(self, name + "_global_scale"))
        return result

    @torch.no_grad()
    def load_weights_(self, *, gate_up, down):
        """Install (physical components, WeightSpec) pairs before optimizer creation.

        Validate both matrices before allocating device copies or replacing state.
        Packed bytes/scales are copied without requantization.
        """
        groups, specs = zip(gate_up, down, strict=True)
        validate_weights(self.config, specs, groups)
        device = self.gate_up.device
        copies = [[p.to(device=device).clone() for p in parts] for parts in groups]
        for name, parts, spec in zip(("gate_up", "down"), copies, specs, strict=True):
            self._install(name, parts, spec)
        return self

    @torch.no_grad()
    def load_fp8_(self, *, gate_up, down):
        """Install DeepSeek-style E4M3 plus 128x128 dequantization multipliers."""
        d, h = self.config.hidden_size, self.config.expert_size
        return self.load_weights_(
            gate_up=(gate_up, WeightSpec("fp8", 2 * h, d, 128, 128)),
            down=(down, WeightSpec("fp8", d, h, 128, 128)),
        )

    def forward(self, x, accumulator, weights, order, bounds):
        c = self.config
        options = (
            c.expert_size,
            c.chunk_capacity,
            c.row_multiple,
            c.scale,
            float(c.swiglu_limit or 0.0),
            c.gemm_precision,
            "|".join(self.specs[n].encode() for n in ("gate_up", "down")),
        )
        factors = (
            [self.gate_up_a, self.gate_up_b, self.down_a, self.down_b]
            if self.lora
            else []
        )
        base = self.base_tensors() if self.lora else [self.gate_up, self.down]
        return expert_op(
            x,
            accumulator,
            weights,
            order,
            bounds,
            base,
            factors,
            *options,
            str(c.gradient_dtype).removeprefix("torch."),
            not self.lora,
        )
