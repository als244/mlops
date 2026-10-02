"""CPU-owned demo masters and explicit initialization outside MoELayer execution."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ExpertMasterSpec:
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    pin_memory: bool = False


class ExpertMasterWeights:
    def __init__(self, cfg, *, spec=None, init_std=None):
        spec = ExpertMasterSpec() if spec is None else spec
        if torch.device(spec.device).type != "cpu":
            raise ValueError("Demo optimizer masters must stay on CPU")
        self.cfg, self.spec = cfg, spec
        q, d, h = cfg.local_experts, cfg.feature_dim, cfg.expert_hidden_dim
        for name, shape in [
            ("gate", (q, h, d)),
            ("up", (q, h, d)),
            ("down", (q, d, h)),
        ]:
            value = torch.empty(
                shape, device="cpu", dtype=spec.dtype, pin_memory=spec.pin_memory
            )
            setattr(
                self,
                name,
                nn.Parameter(
                    value.normal_(0, cfg.init_std if init_std is None else init_std)
                ),
            )

    def tensors(self):
        return self.gate, self.up, self.down

    def state_dict(self):
        return {
            name: getattr(self, name).detach().clone()
            for name in ("gate", "up", "down")
        }

    @torch.no_grad()
    def load_into(self, layer):
        """Initialization/update between accumulation windows, never in FWD/BWD.

        Stage only one expert projection on GPU at a time. The persistent GPU
        parameters themselves contain only their BF16/FP8 compute representation.
        """
        if self.cfg != layer.config:
            raise ValueError("Master/layer configuration mismatch")
        from mlops.expert_parallel.transformer_engine.registry import _runtime

        runtime = _runtime(layer._handle)
        with runtime.execution():
            for bank, master in zip(runtime.banks, self.tensors()):
                before = bank.pointers()
                for index in range(self.cfg.local_experts):
                    if self.cfg.compute_precision == "bf16":
                        # Cast on CPU before upload, once during explicit setup.
                        bank.weight_views[index].copy_(master[index].to(torch.bfloat16))
                    else:
                        staging = master[index].to(layer._device)
                        bank.quantizer.update_quantized(
                            staging, bank.weight_views[index]
                        )
                        del staging
                if bank.pointers() != before:
                    raise RuntimeError(
                        "Weight initialization replaced preallocated storage"
                    )
        torch.cuda.synchronize(layer._device)


def make_demo_layer(cfg, **kwargs):
    from mlops.expert_parallel import TEMoE

    masters = ExpertMasterWeights(cfg)
    from mlops.expert_parallel.buffers import create_buffer

    buffer = create_buffer(cfg, cfg.tokens_per_rank, kwargs.get("ep_group"))
    layer = TEMoE(cfg, buffer=buffer, **kwargs)
    _DEMO_BUFFERS[id(layer)] = buffer
    masters.load_into(layer)
    return layer, masters


_DEMO_BUFFERS = {}


def close_demo_layer(layer):
    """Caller-side cleanup; production layer.close() never destroys a buffer."""
    try:
        layer.close()
    finally:
        buffer = _DEMO_BUFFERS.pop(id(layer), None)
        if buffer is not None:
            buffer.destroy()
