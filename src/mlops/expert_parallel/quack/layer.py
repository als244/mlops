"""Public QuackMoE: SonicMoE routing, MoonEP transport and fused QuACK expert math."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from ...kernels.router import routed_logits
from ..parameters import BF16ComputeWeight
from .buffers import bind_buffer
from .config import config_signature
from .operators import _forward
from .parameters import components  # noqa: F401
from .parameters.construction import independent_experts, local_parameters
from .parameters.initialization import initialize_expert_parameter
from .registry import (
    _REGISTRY_LOCK,
    _RUNTIMES,
    _register_runtime,
    _runtime,
    resource_handle,
)
from .router import route_op
from .shared_operators import _forward as _shared_forward


class QuackMoE(nn.Module):
    def __init__(self, config, ep_group=None, *, buffer, device=None):
        from . import _initialize

        _initialize()
        nn.Module.__init__(self)
        device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        config, ep_group = bind_buffer(config, buffer, ep_group, device)
        self.config, self.ep_group, self._device = config, ep_group, device
        self._spec, self._closed = config_signature(config), False
        c = config
        self._portable = False
        local_parameters(self, c, device)
        # MoonEP exchanges CPU handle tensors. A surrounding model may set a
        # CUDA default device; resource factories select GPU devices explicitly.
        with torch.device("cpu"):
            self._handle = _register_runtime(
                self._create_runtime(c, ep_group, device, buffer)
            )
        for name, bank in zip(
            ("gate_up_weight", "down_weight"), _runtime(self._handle).banks
        ):
            self.register_parameter(name, initialize_expert_parameter(c, bank))
        self.synchronize_replicated_parameters()

    @classmethod
    def from_resource(cls, config, name, *, device="meta"):
        """Construct tensor geometry without a group or live communication buffer.

        The caller initializes a matching named runtime separately before execution.
        Parameters remain ordinary module state and may be moved or initialized by
        the training engine. No broadcast or CUDA allocation occurs on meta/CPU.
        """
        if cls is not QuackMoE:
            raise NotImplementedError(
                "portable construction currently supports QuackMoE"
            )
        if config.tokens_per_rank is None:
            raise ValueError("portable construction requires tokens_per_rank")
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.config, layer.ep_group, layer._device = config, None, None
        layer._spec, layer._closed, layer._portable = (
            config_signature(config),
            False,
            True,
        )
        layer._handle = resource_handle(name)
        local_parameters(layer, config, device)
        independent_experts(layer, config, device)
        return layer

    def _create_runtime(self, config, group, device, buffer):
        from .runtime import _Runtime

        return _Runtime(config, group, device, buffer)

    def expert_parameters(self):
        yield self.gate_up_weight
        yield self.down_weight

    def route(self, x):
        # The router GEMM at the trainers' precision (mlops.kernels.router):
        # operands at x's dtype, fp32 sums and logits; the weight's gradient
        # at router_weight_grad_dtype, the activation's at x's dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = routed_logits(
                x,
                self.router_weight,
                weight_grad_dtype=self.config.router_weight_grad_dtype,
            )
        p, ids, hist = route_op(logits, self.config.top_k, self.config.renormalize_topk)
        return ids, p, hist

    @property
    def communication_buffer(self):
        """The borrowed MoonEP buffer; its caller owns destruction."""
        return _runtime(self._handle).caller_buffer

    def forward(self, x, expert_ids=None, routing_weights=None):
        if self._closed:
            raise RuntimeError("Layer is closed")
        if (
            x.dtype != torch.bfloat16
            or (x.device.type != "cuda" if self._portable else x.device != self._device)
            or x.shape[-1] != self.config.model_dim
        ):
            raise ValueError("Expected BF16 input on the configured CUDA device")
        flat = x.reshape(-1, self.config.model_dim)
        if flat.shape[0] != self.config.tokens_per_rank:
            raise ValueError(
                "Input token count does not match the supplied MoonEP buffer"
            )
        if (expert_ids is None) != (routing_weights is None):
            raise ValueError("Supply both routing tensors or neither")
        if expert_ids is None:
            ids, p, hist = self.route(flat)
        else:
            ids = expert_ids.to(torch.int32).contiguous()
            p = routing_weights.float().contiguous()
            hist = torch.zeros(
                self.config.num_experts, device=x.device, dtype=torch.int32
            )
            hist.scatter_add_(0, ids.flatten().long(), torch.ones_like(ids.flatten()))
        y = self._call_experts(flat.contiguous(), p, ids, hist)
        return y.reshape(x.shape)

    def _call_experts(self, flat, p, ids, hist):
        if self.shared_gate_weight is not None:
            shared_weights = [
                w.to(torch.bfloat16)
                for w in (
                    self.shared_gate_weight,
                    self.shared_up_weight,
                    self.shared_down_weight,
                )
            ]
            y, _ = _shared_forward(
                flat.contiguous(),
                p,
                ids,
                list(self.expert_parameters()),
                hist,
                shared_weights,
                self._handle,
                self._spec,
            )
        else:
            y, _ = _forward(
                flat.contiguous(),
                p,
                ids,
                list(self.expert_parameters()),
                hist,
                self._handle,
                self._spec,
            )
        return y

    def compute_state_report(self):
        runtime = _runtime(self._handle)
        return {
            "backend": "quack",
            "shared_expert_after_first_dispatch": True,
            "experimental_gemm_sms": self.config.experimental_gemm_sms,
            "experimental_local_comm_sms": self.config.experimental_local_comm_sms,
            "token_chunks": self.config.num_chunks,
            "communication_buffers": self.config.num_buffers,
            "share_expert_banks": self.config.share_expert_banks,
            "chunk_fp8_training": "experimental: feature scales reduce per chunk and expert; not equivalent to unchunked FP8 rounding",
            "precision": self.config.compute_precision,
            "activation_transport": self.config.activation_transport,
            "fp8_recipe": "E4M3 per-row/per-output-channel descales; independently quantized orientations",
            "fp8_wgrad_policy": "per-expert FP8 GEMMs over planned group rows with host offset readback",
            "gemm_policy": runtime.math.policy.as_dict(),
            "fp8_accumulation": "accurate per-K-tile FP32 promotion",
            "fp8_quantization": "fused QuACK per-row equations; matching scalar reciprocal and division rounding",
            "routing": "SonicMoE TC top-k plus full local metadata; MoonEP applies configured group padding",
            "saved_routed_output": False,
            "weight_grad_dtype": str(self.config.weight_grad_dtype),
            "token_padding": self.config.token_padding,
            "gemm_tuned": self.config.gemm_tuned,
            "expert_projection_tensors": 2,
            "compute_weight_copies_per_projection": 2,
            "gradient_banks_per_projection": 2,
            "pointers": [b.pointers() for b in runtime.banks],
            "gpu_fp32_master_bytes": 0,
        }

    def replicated_parameters(self):
        yield self.router_weight
        if self.shared_gate_weight is not None:
            yield self.shared_gate_weight
            yield self.shared_up_weight
            yield self.shared_down_weight

    @torch.no_grad()
    def synchronize_replicated_parameters(self):
        if self._portable:
            raise RuntimeError("portable parameters are initialized by their owner")
        src = 0 if self.ep_group is None else dist.get_global_rank(self.ep_group, 0)
        for p in self.replicated_parameters():
            dist.broadcast(
                p._data if isinstance(p, BF16ComputeWeight) else p,
                src=src,
                group=self.ep_group,
            )

    @torch.no_grad()
    def synchronize_replicated_gradients(self):
        """SUM once per accumulation window. Do NOT reduce expert shards over EP."""
        if self._portable:
            raise RuntimeError("portable gradients are reduced by their owner")
        for p in self.replicated_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.ep_group)

    def set_overlap_enabled(self, enabled: bool):
        """Collective benchmark control; call OUTSIDE compile, after pending work."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        r = _runtime(self._handle)
        torch.cuda.synchronize(r.device)
        values = [None] * self.config.ep_size
        dist.all_gather_object(values, enabled, group=r.group)
        if len(set(values)) != 1:
            raise ValueError("Every EP rank must select the same overlap mode")
        r.overlap_enabled = enabled

    def _apply(self, fn, recurse=True):
        if hasattr(self, "_handle") and not self._portable:
            raise RuntimeError(
                "Fixed runtime mappings: construct with the intended device/dtype, do not call .to() afterwards"
            )
        return super()._apply(fn, recurse=recurse)

    def close(self):
        if not self._closed:
            if self._portable:
                self._closed = True
                return
            _runtime(self._handle).close()
            with _REGISTRY_LOCK:
                _RUNTIMES.pop(self._handle, None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
