"""LoRA specialization of the existing chunk/stream scheduler."""

from functools import lru_cache

import torch
import torch.distributed as dist

from ...lora import signature
from ..config import config_signature
from ..runtime import _Runtime
from ..shared_expert import shared_forward
from ..weights import _Bank
from .math import LoRAExperts
from .storage import FactorBank


class ProjectionBank(_Bank):
    def __init__(self, config, lora, rank, group, out_features, in_features, device):
        super().__init__(
            config, rank, group, out_features, in_features, device, trainable=False
        )
        self.factors = FactorBank(
            config, lora, rank, group, out_features, in_features, device
        )

    def publish(self, weights):
        self.weight_state.publish(weights[0])
        self.factors.publish(weights[1:])

    def prepare_grad_scratch(self):
        self.factors.prepare_grad_scratch()

    def reduce(self, plan, ctx):
        self.factors.reduce(plan, ctx)

    def grad_result(self, weights):
        return self.factors.grad_result()

    def external_tensors(self):
        return [*super().external_tensors(), *self.factors.external_tensors()]


def frozen_shared_backward(
    dy, x, gate, up, hidden, gate_weight, up_weight, down_weight
):
    dh = (dy @ down_weight).float()
    g, u = gate.float(), up.float()
    sigmoid = g.sigmoid()
    dg = (dh * u * sigmoid * (1 + g * (1 - sigmoid))).to(x.dtype)
    du = (dh * g * sigmoid).to(x.dtype)
    return (dg @ gate_weight + du @ up_weight,)


@lru_cache(None)
def frozen_shared_math():
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
    return (
        torch.compile(shared_forward, fullgraph=True, options=options),
        torch.compile(frozen_shared_backward, fullgraph=True, options=options),
    )


class LoRARuntime(_Runtime):
    def __init__(self, config, lora, group, device, buffer):
        self.lora = lora
        specs = [None] * config.ep_size
        dist.all_gather_object(
            specs, signature(config_signature(config), lora), group=group
        )
        if len(set(specs)) != 1:
            raise ValueError("Every EP rank must use the same LoRA configuration")
        super().__init__(config, group, device, buffer)
        self.math = LoRAExperts(self.chunk_cfg, lora, self.banks)

    def _make_bank(self, out_features, in_features):
        return ProjectionBank(
            self.cfg,
            self.lora,
            self.rank,
            self.group,
            out_features,
            in_features,
            self.device,
        )

    def _check_inputs(self, x, p, ids, params):
        super()._check_inputs(x, p, ids, [w[0] for w in params])

    def _prefetch(self, streams, chunk, phase):
        with streams.range(f"{phase}.{chunk.index}.prefetch", communication=True):
            for bank in self.banks:
                bank.weight_state.prefetch_slots(chunk.slots)
                bank.factors.prefetch_slots(chunk.slots)
            chunk.ready = streams.comm.record_event()

    def _shared_math(self):
        return frozen_shared_math()

    def _parameters(self, base, factors):
        weights = base if self.cfg.compute_precision == "bf16" else (base[:4], base[4:])
        return [(w, *factors[2 * i : 2 * i + 2]) for i, w in enumerate(weights)]

    def lora_forward(self, x, p, ids, base, factors, hist, shared):
        return self.forward(
            x,
            p,
            ids,
            hist,
            *self._parameters(base, factors),
            shared_weights=shared or None,
        )

    def lora_backward(self, dy, base, factors, state, x, shared):
        result = self.backward(
            dy,
            *self._parameters(base, factors),
            state,
            x=x,
            shared_weights=shared or None,
        )
        dx, dp, gradients = result[:3]
        return dx, dp, [g for pair in gradients for g in pair]
