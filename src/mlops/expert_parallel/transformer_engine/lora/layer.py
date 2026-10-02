"""Frozen TE experts with independent per-expert joint gate/up LoRA factors."""

import torch
from torch import nn

from ...lora import LoRAConfig, signature
from ...parameters import BF16ComputeWeight
from ..config import config_signature
from ..layer import TEMoE
from ..registry import _runtime
from .operators import forward
from .runtime import LoRARuntime
from .shared import forward as shared_forward


class TEMoELoRA(TEMoE):
    def __init__(self, config, ep_group=None, *, buffer, lora=None, device=None):
        self.lora_config = lora or LoRAConfig()
        super().__init__(config, ep_group, buffer=buffer, device=device)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for prefix, bank in zip(("gate_up", "down"), _runtime(self._handle).factors):
            bank.initialize()
            for suffix, value in zip(("a", "b"), bank.parameter_data):
                logical = (
                    BF16ComputeWeight(value, dtype=self.lora_config.gradient_dtype)
                    if self.lora_config.gradient_dtype != torch.bfloat16
                    else value
                )
                self.register_parameter(
                    f"lora_{prefix}_{suffix}", nn.Parameter(logical)
                )
        self._spec = signature(config_signature(self.config), self.lora_config)
        self.shared_gate_up_weight = None
        if self.shared_gate_weight is not None:
            with torch.no_grad():
                self.shared_gate_up_weight = nn.Parameter(
                    torch.cat(
                        (
                            self.shared_gate_weight.to(torch.bfloat16),
                            self.shared_up_weight.to(torch.bfloat16),
                        )
                    ),
                    requires_grad=False,
                )
                self.shared_down_weight = nn.Parameter(
                    self.shared_down_weight.to(torch.bfloat16), requires_grad=False
                )
            self.shared_gate_weight = self.shared_up_weight = None

    def _create_runtime(self, config, group, device, buffer):
        return LoRARuntime(config, self.lora_config, group, device, buffer)

    def _expert_names(self):
        return (
            ("gate_up_weight", "down_weight")
            if self.config.compute_precision == "bf16"
            else ("gate_up_experts", "down_experts")
        )

    def expert_parameters(self):
        for name in self._expert_names():
            value = getattr(self, name)
            if self.config.compute_precision == "bf16":
                yield value
            else:
                yield from value

    def lora_parameters(self):
        for name in ("lora_gate_up_a", "lora_gate_up_b", "lora_down_a", "lora_down_b"):
            yield getattr(self, name)

    def _call_experts(self, x, p, ids):
        hist = torch.zeros(self.config.num_experts, device=x.device, dtype=torch.int32)
        hist.scatter_add_(0, ids.flatten().long(), torch.ones_like(ids.flatten()))
        y, _ = forward(
            x,
            p,
            ids,
            list(self.expert_parameters()),
            list(self.lora_parameters()),
            hist,
            [],
            self._handle,
            self._spec,
        )
        return y

    def _add_shared(self, flat, y):
        if self.shared_gate_up_weight is not None:
            y = (
                y
                + shared_forward(
                    flat, self.shared_gate_up_weight, self.shared_down_weight
                )[0]
            )
        return y

    def replicated_parameters(self):
        yield from super().replicated_parameters()
        if getattr(self, "shared_gate_up_weight", None) is not None:
            yield self.shared_gate_up_weight
            yield self.shared_down_weight

    def synchronize_replicated_gradients(self):
        """Frozen replicated weights have no parameter gradients to reduce."""

    def compute_state_report(self):
        return {
            "backend": "te",
            "training": "lora",
            "precision": self.config.compute_precision,
            "lora_rank": self.lora_config.rank,
            "lora_scale": self.lora_config.scale,
            "lora_gradient_dtype": str(self.lora_config.gradient_dtype),
            "frozen_base_gradients": False,
            "gpu_fp32_master_bytes": 0,
            "activation": "Transformer Engine SwiGLU",
            "shared_expert": "Transformer Engine GEMM/SwiGLU with explicit custom operators",
        }
