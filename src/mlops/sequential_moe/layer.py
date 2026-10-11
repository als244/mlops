"""Public SequentialMoE and SequentialMoELoRA layers sharing one execution path."""

import math

import torch
from torch import nn

from .config import SequentialMoEConfig
from .experts.layer import SequentialExpert
from .precision import validate_device
from .precision.parameters import parameter
from .router import router
from .routing import layout
from .shared import shared


class OutputCast(nn.Module):
    """Convert the completed accumulator back to the layer's activation dtype."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, accumulator):
        if accumulator.dtype == self.dtype:
            return accumulator
        # A dtype conversion allocates. Make that explicit in the exported
        # schema rather than using to(), which may also return an alias.
        return torch.ops.aten._to_copy.default(accumulator, dtype=self.dtype)


class _SequentialMoE(nn.Module):
    def __init__(self, config: SequentialMoEConfig, *, device="cuda", lora):
        super().__init__()
        validate_device(config, device)
        self.config, self.lora = config, lora
        self.output_cast = (
            nn.Identity()
            if config.activation_accum_dtype == config.activation_dtype
            else OutputCast(config.activation_dtype)
        )
        d = config.hidden_size
        router = torch.empty(
            config.num_experts, d, device=device, dtype=config.router_dtype
        )
        nn.init.normal_(router, std=config.initializer_range or 1 / math.sqrt(d))
        self.router = (
            parameter(router, config.gradient_dtype)
            if not lora and config.train_router
            else nn.Parameter(router, requires_grad=False)
        )
        self.experts = nn.ModuleList(
            SequentialExpert(config, device, lora=lora)
            for _ in range(config.num_experts)
        )
        h = config.shared_experts * (config.shared_expert_size or config.expert_size)
        for name, shape in (("shared_gate_up", (2 * h, d)), ("shared_down", (d, h))):
            if not h:
                self.register_parameter(name, None)
                continue
            value = torch.empty(shape, device=device, dtype=config.activation_dtype)
            nn.init.normal_(
                value, std=config.initializer_range or 1 / math.sqrt(shape[1])
            )
            p = (
                parameter(value, config.gradient_dtype)
                if not lora and config.train_shared
                else nn.Parameter(value, requires_grad=False)
            )
            setattr(self, name, p)

    def _validate_input(self, x):
        if x.ndim != 2 or x.shape[1] != self.config.hidden_size:
            raise ValueError("Expected X with shape [tokens, hidden_size]")
        if x.dtype != self.config.activation_dtype or x.device != self.router.device:
            raise ValueError(
                f"X must be on {self.router.device} at activation_dtype={self.config.activation_dtype}"
            )

    def route(self, x):
        self._validate_input(x)
        logits = router(x, self.router, self.config.chunk_capacity)
        selected, ids = torch.topk(logits, self.config.top_k, dim=-1)
        return ids, torch.softmax(selected.float(), dim=-1)

    def forward_routed(self, x, ids, weights, accumulator=None):
        c = self.config
        self._validate_input(x)
        if ids.shape != (x.shape[0], c.top_k) or weights.shape != ids.shape:
            raise ValueError("Expected routes and weights with shape [tokens, top_k]")
        if (
            ids.dtype != torch.int64
            or ids.device != x.device
            or weights.device != x.device
        ):
            raise ValueError("Routes must be int64 and on the input device")
        if weights.dtype != torch.float32:
            raise ValueError("Routing weights must be FP32, as produced by route()")
        x, ids, weights = x.contiguous(), ids.contiguous(), weights.contiguous()
        order, expert_bounds = layout(ids, c.num_experts)
        if accumulator is not None:
            if self.shared_gate_up is not None:
                raise ValueError(
                    "An explicit accumulator requires no internal shared expert"
                )
            if (
                accumulator.shape != x.shape
                or accumulator.dtype != c.activation_accum_dtype
                or accumulator.device != x.device
            ):
                raise ValueError(
                    "Accumulator must match the input shape/device and activation_accum_dtype"
                )
            acc = accumulator
        elif self.shared_gate_up is None:
            acc = torch.zeros_like(x, dtype=c.activation_accum_dtype)
        else:
            acc = shared(
                x,
                self.shared_gate_up,
                self.shared_down,
                c.chunk_capacity,
                c.row_multiple,
                float(c.swiglu_limit or 0.0),
                str(c.activation_accum_dtype).removeprefix("torch."),
            )
        for expert, bounds in zip(self.experts, expert_bounds, strict=True):
            acc = expert(x, acc, weights, order, bounds)
        return self.output_cast(acc)

    def forward(self, x):
        return self.forward_routed(x, *self.route(x))


class SequentialMoE(_SequentialMoE):
    """Train routed experts and, by default, router/shared experts."""

    def __init__(self, config: SequentialMoEConfig, *, device="cuda"):
        super().__init__(config, device=device, lora=False)


class SequentialMoELoRA(_SequentialMoE):
    """Freeze original weights and train only routed expert LoRA factors."""

    def __init__(self, config: SequentialMoEConfig, *, device="cuda"):
        super().__init__(config, device=device, lora=True)
