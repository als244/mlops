"""Packed per-expert A/B storage using MoonEP prefetch and gradient reduction.

Alignment applies to the entire rank allocation, not each skinny matrix.
Factor gradients are small owned outputs; frozen base gradients never exist.
"""

import torch

from ...kernels.grad_reduce import launch_grad_reduce
from ...lora import packed_pitch


class FactorBank:
    def __init__(self, config, lora, rank, group, out_features, in_features, device):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity
        from moonep.prefetch import launch_prefetch

        self.cfg, self.lora, self.rank, self.device = config, lora, rank, device
        self.out_features, self.in_features = out_features, in_features
        self.shapes = ((lora.rank, in_features), (out_features, lora.rank))
        self.a_elements = lora.rank * in_features
        self.elements = self.a_elements + out_features * lora.rank
        self.q = q = config.local_experts
        gran = int(get_vmm_granularity())
        pitch = packed_pitch(
            self.elements,
            element_size=2,
            experts=2 * q,
            granularity=gran,
            tile_elements=8192,
        )
        self.remote = create_nvl_dist_tensor(
            [2 * q, pitch], torch.bfloat16, rank, config.ep_size, group=group
        )
        local = self.remote[rank * 2 * q : (rank + 1) * 2 * q]
        local.zero_()
        self.parameter_data = self._views(local[:q])
        self.compute_weights = self._views(local)
        self.prefetch_source = self.remote.view(torch.uint8).view(
            -1, 128, pitch * 2 // 128
        )
        self.prefetch_destination = (
            local[q:].view(torch.uint8).view(q, 128, pitch * 2 // 128)
        )
        self.launch_prefetch = launch_prefetch
        grad_pitch = packed_pitch(
            self.elements,
            element_size=4,
            experts=q,
            granularity=gran,
            tile_elements=128 * 128,
        )
        self.grad_shape = (q, 128, grad_pitch // 128)
        self.replica_owner = create_nvl_dist_tensor(
            list(self.grad_shape), torch.float32, rank, config.ep_size, group=group
        )
        self.replica_grads = self.replica_owner.view(config.ep_size, *self.grad_shape)
        self.local_replica = self.replica_grads[rank]
        self.local_replica.zero_()
        self.home = None
        self.grad_views = None

    def _views(self, value):
        flat = value.view(value.shape[0], -1)
        return (
            flat[:, : self.a_elements].view(flat.shape[0], *self.shapes[0]),
            flat[:, self.a_elements : self.elements].view(
                flat.shape[0], *self.shapes[1]
            ),
        )

    @torch.no_grad()
    def initialize(self):
        a, b = self.parameter_data
        initial = torch.empty(
            a.shape, dtype=self.lora.initialization_dtype, device="cpu"
        )
        for expert in initial:
            torch.nn.init.xavier_uniform_(expert)
        a.copy_(initial)
        b.zero_()

    def publish(self, factors):
        if len(factors) != 2:
            raise ValueError("Expected per-expert A and B")
        for source, destination in zip(factors, self.parameter_data):
            if (
                source.shape != destination.shape
                or source.dtype != torch.bfloat16
                or source.device != self.device
            ):
                raise ValueError("LoRA factor shape, compute dtype or device mismatch")
            if source.data_ptr() != destination.data_ptr():
                destination.copy_(source)
            elif source.stride() != destination.stride():
                raise ValueError("LoRA factor aliases storage with different strides")

    def prefetch_slots(self, slots):
        self.launch_prefetch(
            self.prefetch_source,
            self.prefetch_destination,
            slots,
            self.cfg.num_comm_sms,
        )

    def prefetch(self, plan):
        ids = plan.experts_to_copy[self.rank]
        slots = torch.where(
            ids >= 0, (ids // self.q) * (2 * self.q) + ids % self.q, -1
        ).to(torch.int32)
        self.prefetch_slots(slots)

    def prepare_grad_scratch(self):
        if self.home is not None:
            raise RuntimeError(
                "Previous LoRA backward did not transfer its gradient results"
            )
        self.home = torch.zeros(
            self.grad_shape, device=self.device, dtype=torch.float32
        )
        self.grad_views = (self._views(self.home), self._views(self.local_replica))

    def reduce(self, plan, ctx):
        launch_grad_reduce(
            self.home,
            self.replica_grads,
            plan.experts_to_copy,
            rank=self.rank,
            num_sms=self.cfg.num_comm_sms,
            meta_buf=ctx["meta_buf"],
            meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]),
            grid_sync_bar=ctx["grid_sync_bar"],
        )

    def grad_result(self, _weights=None):
        # Outputs must not alias each other or persistent communication scratch.
        # These copies contain only LoRA factors, never a full base-weight matrix.
        result = [
            v.to(
                dtype=self.lora.gradient_dtype,
                copy=True,
                memory_format=torch.contiguous_format,
            )
            for v in self._views(self.home)
        ]
        self.home = self.grad_views = None
        return result

    def external_tensors(self):
        return [self.remote, self.replica_owner]
