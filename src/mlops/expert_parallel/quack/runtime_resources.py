"""Common EP resources and validation; execution lives in runtime.py."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist

from .communication import _walk_tensors
from .config import config_signature
from .kernels.pointwise import _Pointwise
from .weights import _Bank


def local_group_counts(ends: torch.Tensor, rank: int, E: int, q: int) -> torch.Tensor:
    e = ends.to(torch.int64)
    starts = torch.cat((torch.zeros_like(e[:1]), e[:-1]))
    sizes = e - starts
    return torch.cat((sizes[rank * q : (rank + 1) * q], sizes[E : E + q])).contiguous()


class RuntimeResources:
    def __init__(self, c, group, device, buffer):
        if not dist.is_initialized() or dist.get_world_size(group) != c.ep_size:
            raise ValueError("Initialize distributed and pass the matching EP group")
        if dist.get_backend(group) != "nccl":
            raise ValueError("The MoonEP EP group must use NCCL")
        if device.index != torch.cuda.current_device():
            raise ValueError("Set the process CUDA device first")
        if torch.cuda.get_device_capability(device)[0] != 9:
            raise ValueError(
                "This prototype is scoped to H100/SM90 until other architectures are validated"
            )
        descriptors = [None] * c.ep_size
        dist.all_gather_object(descriptors, config_signature(c), group=group)
        if len(set(descriptors)) != 1:
            raise ValueError("All EP ranks must have identical configuration")
        from moonep.inter_rank_sync import launch_inter_rank_sync
        from moonep.planning import MoonEPCommPlan

        self.cfg, self.group, self.device = c, group, device
        self.rank = dist.get_rank(group)
        self.reuse_communication_buffers = c.reuse_communication_buffers
        self.overlap_enabled = c.overlap
        self.plan_type, self.rank_sync = MoonEPCommPlan, launch_inter_rank_sync
        self.closed = False
        self._caller_stream = None
        self.pw = _Pointwise()
        self.buffer = buffer
        self.ctx = self.buffer._require_ctx()
        self.banks = [
            self._make_bank(2 * c.expert_hidden_dim, c.feature_dim),
            self._make_bank(c.feature_dim, c.expert_hidden_dim),
        ]
        tensors = [self.buffer.hidden_nvsh_buffer_view]
        tensors.extend(t for b in self.banks for t in b.external_tensors())
        self.external_storage_extents = tuple(
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for t in tensors
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _make_bank(self, out_features, in_features):
        return _Bank(
            self.cfg, self.rank, self.group, out_features, in_features, self.device
        )

    def _check_fp8_components(self, params):
        for bank, components in zip(self.banks, params):
            bank.weight_state.validate_components(components)

    def _publish(self, params):
        for bank, w in zip(self.banks, params):
            bank.publish(w)

    @contextmanager
    def execution(self):
        if self.closed:
            raise RuntimeError("Runtime is closed")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("Wrong current CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graphs are not validated. Use options={'triton.cudagraphs': False}"
            )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self._caller_stream is None:
            self._caller_stream = stream
        elif self._caller_stream != stream:
            raise RuntimeError(
                "Use one caller/compute stream; the runtime owns its separate comm stream"
            )
        yield

    def _check_inputs(self, x, p, ids, params):
        c = self.cfg
        if x.shape != (c.tokens_per_rank, c.feature_dim) or x.dtype != torch.bfloat16:
            raise ValueError(
                "Expert input must have fixed [S, expert_width] BF16 shape"
            )
        if (
            p.shape != (c.tokens_per_rank, c.top_k)
            or p.dtype != torch.float32
            or ids.shape != p.shape
            or ids.dtype != torch.int32
        ):
            raise ValueError(
                "Routing requires FP32 probabilities and INT32 IDs of shape [S,K]"
            )
        flat_weights = tuple(_walk_tensors(params))
        if any(t.device != self.device or not t.is_contiguous() for t in (x, p, ids)):
            raise ValueError(
                "Activation and routing inputs must be contiguous on the runtime device"
            )
        if any(t.device != self.device for t in flat_weights):
            raise ValueError("Compute weight components must be on the runtime device")
        if c.compute_precision == "bf16":
            for w, b in zip(params, self.banks):
                if (
                    w.shape != (c.local_experts, b.out_features, b.in_features)
                    or w.dtype != torch.bfloat16
                ):
                    raise ValueError("BF16 compute component shape/dtype mismatch")
                if (
                    not w.is_contiguous()
                    and w.stride() != b.weight_state.parameter_data.stride()
                ):
                    raise ValueError(
                        "BF16 component must be contiguous or use the declared expert padding"
                    )
        else:
            self._check_fp8_components(params)

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.banks.clear()
        self.closed = True
