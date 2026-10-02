"""Public QuackMoE: SonicMoE routing, MoonEP transport and fused QuACK expert math."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .buffers import bind_buffer
from .config import config_signature
from .operators import _forward
from .parameters.bf16 import BF16ComputeWeight
from .parameters.initialization import compute_parameter, initialize_expert_parameter
from .registry import _REGISTRY_LOCK, _RUNTIMES, _register_runtime, _runtime
from .router import route_op
from .runtime import _Runtime
from .shared_operators import _forward as _shared_forward


class MoELayer(nn.Module):
    def __init__(self, config, ep_group=None, *, buffer, device=None):
        nn.Module.__init__(self)
        device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        config, ep_group = bind_buffer(config, buffer, ep_group, device)
        self.config, self.ep_group, self._device = config, ep_group, device
        self._spec, self._closed = config_signature(config), False
        c = config
        router_data = torch.empty(
            c.num_experts, c.model_dim, device=device, dtype=c.router_dtype
        ).normal_(0, c.init_std)
        self.router_weight = nn.Parameter(
            BF16ComputeWeight(router_data, dtype=c.router_weight_grad_dtype)
            if c.router_dtype != c.router_weight_grad_dtype
            else router_data
        )
        for name, shape in [
            ("shared_gate_weight", (c.shared_width, c.model_dim)),
            ("shared_up_weight", (c.shared_width, c.model_dim)),
            ("shared_down_weight", (c.model_dim, c.shared_width)),
        ]:
            if c.shared_width:
                data = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_(
                    0, c.init_std
                )
                self.register_parameter(
                    name, compute_parameter(data, c.weight_grad_dtype)
                )
            else:
                self.register_parameter(name, None)
        self._handle = _register_runtime(
            self._create_runtime(c, ep_group, device, buffer)
        )
        for name, bank in zip(
            ("gate_up_weight", "down_weight"), _runtime(self._handle).banks
        ):
            self.register_parameter(name, initialize_expert_parameter(c, bank))
        self.synchronize_replicated_parameters()

    def _create_runtime(self, config, group, device, buffer):
        return _Runtime(config, group, device, buffer)

    def expert_parameters(self):
        yield self.gate_up_weight
        yield self.down_weight

    def route(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(
                x.to(self.config.router_dtype),
                self.router_weight.to(self.config.router_dtype),
            ).float()
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
            or x.device != self._device
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
        for p in self.replicated_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.ep_group)

    def set_overlap_enabled(self, enabled: bool):
        """Collective benchmark control; call OUTSIDE compile, after pending work."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        r = _runtime(self._handle)
        torch.cuda.synchronize(self._device)
        values = [None] * self.config.ep_size
        dist.all_gather_object(values, enabled, group=self.ep_group)
        if len(set(values)) != 1:
            raise ValueError("Every EP rank must select the same overlap mode")
        r.overlap_enabled = enabled

    def _apply(self, fn, recurse=True):
        if hasattr(self, "_handle"):
            raise RuntimeError(
                "Fixed runtime mappings: construct with the intended device/dtype, do not call .to() afterwards"
            )
        return super()._apply(fn, recurse=recurse)

    def close(self):
        if not self._closed:
            _runtime(self._handle).close()
            with _REGISTRY_LOCK:
                _RUNTIMES.pop(self._handle, None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
