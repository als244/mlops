"""Frozen QuackMoE base with independent joint gate/up and down LoRA factors."""

import torch

from ..config import config_signature
from ..layer import MoELayer
from ..parameters.initialization import compute_parameter
from ..registry import _runtime
from .config import LoRAConfig, signature
from .operators import forward
from .runtime import LoRARuntime


class QuackMoELoRA(MoELayer):
    def __init__(self, config, ep_group=None, *, buffer, lora=None, device=None):
        self.lora_config = lora or LoRAConfig()
        super().__init__(config, ep_group, buffer=buffer, device=device)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for prefix, bank in zip(("gate_up", "down"), _runtime(self._handle).banks):
            bank.factors.initialize()
            for suffix, value in zip(("a", "b"), bank.factors.parameter_data):
                self.register_parameter(
                    f"lora_{prefix}_{suffix}",
                    compute_parameter(value, self.lora_config.gradient_dtype),
                )
        self._spec = signature(config_signature(self.config), self.lora_config)

    def _create_runtime(self, config, group, device, buffer):
        return LoRARuntime(config, self.lora_config, group, device, buffer)

    def lora_parameters(self):
        for name in ("lora_gate_up_a", "lora_gate_up_b", "lora_down_a", "lora_down_b"):
            yield getattr(self, name)

    def _call_experts(self, flat, p, ids, hist):
        shared = (
            []
            if self.shared_gate_weight is None
            else [
                w.to(torch.bfloat16)
                for w in (
                    self.shared_gate_weight,
                    self.shared_up_weight,
                    self.shared_down_weight,
                )
            ]
        )
        y, _ = forward(
            flat,
            p,
            ids,
            list(self.expert_parameters()),
            list(self.lora_parameters()),
            hist,
            shared,
            self._handle,
            self._spec,
        )
        return y

    def synchronize_replicated_gradients(self):
        """Frozen replicated weights have no parameter gradients to reduce."""

    def compute_state_report(self):
        return {
            "backend": "quack",
            "training": "lora",
            "precision": self.config.compute_precision,
            "lora_rank": self.lora_config.rank,
            "lora_scale": self.lora_config.scale,
            "lora_gradient_dtype": str(self.lora_config.gradient_dtype),
            "frozen_base_gradients": False,
            "gpu_fp32_master_bytes": 0,
            "token_chunks": self.config.num_chunks,
            "activation_transport": self.config.activation_transport,
        }
