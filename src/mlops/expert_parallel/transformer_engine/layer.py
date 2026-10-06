"""Public MoE layer: router, expert call and optional shared expert."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..parameters import BF16ComputeWeight
from .buffers import bind_buffer
from .config import MoEConfig, config_signature
from .operators import _EFFECT_ERROR, _forward_op
from .registry import _REGISTRY_LOCK, _RUNTIMES, _register_runtime, _runtime
from .runtime import _MoonRuntime


class TEMoE(nn.Module):
    """Full first-order MoE module; strict group-wide GPU phase boundaries.

    x: BF16 [...,D] with S flattened tokens. Optional external routing is [S,K].
    All ranks must execute matching calls/backward orders and avoid parameter
    updates until all backward tasks in the accumulation window are complete.
    Replay/recomputation is managed by the caller's graph partitioner; forward
    invocations are not counted as training microbatches by this module.
    """

    def __init__(self, config: MoEConfig, ep_group=None, *, buffer, device=None):
        super().__init__()
        if _EFFECT_ERROR is not None:
            raise RuntimeError(
                "Ordered compiler-effect API unavailable"
            ) from _EFFECT_ERROR
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU; production MoonEP has no CPU fallback")
        device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        config, ep_group = bind_buffer(config, buffer, ep_group, device)
        self.config, self.ep_group, self._device = config, ep_group, device
        self._spec = config_signature(config)
        c = config

        def bf16_parameter(shape):
            data = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_(
                0, c.init_std
            )
            value = (
                BF16ComputeWeight(data, dtype=c.weight_grad_dtype)
                if c.weight_grad_dtype != torch.bfloat16
                else data
            )
            return nn.Parameter(value)

        for name in ("gate_weight", "up_weight", "down_weight"):
            self.register_parameter(name, None)
        router_data = torch.empty(
            c.num_experts, c.model_dim, device=device, dtype=c.router_dtype
        ).normal_(0, c.init_std)
        self.router_weight = nn.Parameter(
            BF16ComputeWeight(router_data, dtype=c.router_weight_grad_dtype)
            if c.router_dtype != c.router_weight_grad_dtype
            else router_data
        )
        if c.latent_dim is None:
            self.register_parameter("latent_down_weight", None)
            self.register_parameter("latent_up_weight", None)
        else:
            self.latent_down_weight = bf16_parameter((c.latent_dim, c.model_dim))
            self.latent_up_weight = bf16_parameter((c.model_dim, c.latent_dim))
        for name, shape in [
            ("shared_gate_weight", (c.shared_width, c.model_dim)),
            ("shared_up_weight", (c.shared_width, c.model_dim)),
            ("shared_down_weight", (c.model_dim, c.shared_width)),
        ]:
            self.register_parameter(
                name, bf16_parameter(shape) if c.shared_width else None
            )
        # MoonEP's exchanged handle tensors belong on CPU even when a caller
        # constructs its model under a CUDA default-device context.
        with torch.device("cpu"):
            self._handle = _register_runtime(
                self._create_runtime(c, ep_group, device, buffer)
            )
        self._closed = False
        self.synchronize_replicated_parameters()
        runtime = _runtime(self._handle)
        self._initialize_experts(runtime)

    def _initialize_experts(self, runtime):
        c, q, device = self.config, self.config.local_experts, self._device
        if c.compute_precision == "bf16":
            for name, bank in zip(self._expert_names(), runtime.banks):
                data = bank.weight_state.parameter_data
                with torch.no_grad():
                    data.normal_(0, c.init_std)
                value = (
                    BF16ComputeWeight(data, dtype=c.weight_grad_dtype)
                    if c.weight_grad_dtype != torch.bfloat16
                    else data
                )
                setattr(self, name, nn.Parameter(value))
        else:
            for name, bank in zip(self._expert_names(), runtime.banks):
                values = nn.ParameterList(
                    [nn.Parameter(w) for w in bank.weight_views[:q]]
                )
                setattr(self, name, values)
                bank.weight_views[:q] = list(values)
                # Ordinary initialization has only one temporary matrix resident.
                with torch.no_grad():
                    for value in values:
                        temporary = torch.empty(
                            value.shape, device=device, dtype=torch.float32
                        ).normal_(0, c.init_std)
                        bank.quantizer.update_quantized(temporary, value)
                        del temporary

    def _expert_names(self):
        return (
            ("gate_weight", "up_weight", "down_weight")
            if self.config.compute_precision == "bf16"
            else ("gate_experts", "up_experts", "down_experts")
        )

    def _create_runtime(self, config, group, device, buffer):
        return _MoonRuntime(config, group, device, buffer)

    def expert_parameters(self):
        if self.config.compute_precision == "bf16":
            yield self.gate_weight
            yield self.up_weight
            yield self.down_weight
        else:
            yield from self.gate_experts
            yield from self.up_experts
            yield from self.down_experts

    def replicated_parameters(self):
        yield self.router_weight
        if self.latent_down_weight is not None:
            yield self.latent_down_weight
            yield self.latent_up_weight
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

    def route(self, x):
        flat = x.reshape(-1, self.config.model_dim)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(
                flat.to(self.config.router_dtype),
                self.router_weight.to(self.config.router_dtype),
            ).float()
        values, ids = logits.topk(self.config.top_k, dim=-1)
        p = (
            values.softmax(-1)
            if self.config.renormalize_topk
            else logits.softmax(-1).gather(-1, ids)
        )
        return ids.to(torch.int32).contiguous(), p.float().contiguous()

    @property
    def communication_buffer(self):
        """The borrowed MoonEP buffer; its caller owns destruction."""
        return _runtime(self._handle).buffer

    def forward(self, x, expert_ids=None, routing_weights=None):
        if self._closed:
            raise RuntimeError("Layer is closed")
        if (
            x.dtype != torch.bfloat16
            or x.device != self._device
            or x.shape[-1] != self.config.model_dim
        ):
            raise ValueError(
                "Input must be BF16 on the configured device, with final dimension D"
            )
        flat = x.reshape(-1, self.config.model_dim)
        if flat.shape[0] != self.config.tokens_per_rank:
            raise ValueError(
                "Input token count does not match the supplied MoonEP buffer shape"
            )
        if (expert_ids is None) != (routing_weights is None):
            raise ValueError("Supply both routing tensors or neither")
        if expert_ids is None:
            ids, p = self.route(flat)
        else:
            ids = (
                expert_ids.reshape(flat.shape[0], self.config.top_k)
                .to(torch.int32)
                .contiguous()
            )
            p = (
                routing_weights.reshape(flat.shape[0], self.config.top_k)
                .float()
                .contiguous()
            )
        z = (
            flat.contiguous()
            if self.latent_down_weight is None
            else F.linear(flat, self.latent_down_weight.to(torch.bfloat16)).contiguous()
        )
        y = self._call_experts(z, p, ids)
        if self.latent_up_weight is not None:
            y = F.linear(y, self.latent_up_weight.to(torch.bfloat16))
        y = self._add_shared(flat, y)
        return y.reshape(x.shape)

    def _call_experts(self, x, p, ids):
        y, _ = _forward_op(
            x, p, ids, list(self.expert_parameters()), self._handle, self._spec
        )
        return y

    def _add_shared(self, flat, y):
        if self.shared_gate_weight is not None:
            # Replicated dense branch on the original model-width input. This
            # is compiler-visible BF16 math, including when routed GEMMs use FP8.
            gate = F.linear(flat, self.shared_gate_weight.to(torch.bfloat16))
            up = F.linear(flat, self.shared_up_weight.to(torch.bfloat16))
            hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            shared = F.linear(hidden, self.shared_down_weight.to(torch.bfloat16))
            y = y + shared
        return y

    def compute_state_report(self):
        """Inspect physical expert storage, excluding separate FP32 gradient buffers."""
        runtime = _runtime(self._handle)
        if self.config.compute_precision == "bf16":
            return {
                "precision": "bf16",
                "weight_grad_dtype": str(self.config.weight_grad_dtype),
                "pointers": [b.pointers() for b in runtime.banks],
                "components": [
                    {
                        "_data": {
                            "dtype": "torch.bfloat16",
                            "shape": tuple(b.weight_state.parameter_data.shape),
                            "transport_dtype": "torch.bfloat16",
                        }
                    }
                    for b in runtime.banks
                ],
            }
        return {
            "precision": self.config.compute_precision,
            "weight_grad_dtype": str(self.config.weight_grad_dtype),
            "pointers": [b.pointers() for b in runtime.banks],
            "components": [
                {
                    name: {
                        "dtype": str(bank.dtype),
                        "shape": bank.shape,
                        "transport_dtype": str(bank.remote.dtype),
                    }
                    for name, bank in b.components.items()
                }
                for b in runtime.banks
            ],
        }

    def inspect_intermediates(self):
        """Return latest call's phase tensors when retain_intermediates=True.

        Holding this mapping also holds its GPU storage. The runtime replaces it
        on the next forward/backward entry; clear it explicitly after inspection.
        """
        return _runtime(self._handle).retained_intermediates

    def clear_retained_intermediates(self):
        """Release the runtime's debug references; caller-held references remain."""
        _runtime(self._handle).retained_intermediates = None

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
