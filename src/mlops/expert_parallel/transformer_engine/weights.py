"""MoonEP compute-weight publication and owned home/replica gradients."""

from __future__ import annotations

import math

import torch

from .experts import _make_quantizer


class _BF16WeightState:
    """BF16 compute weights and replicas; gradient ownership is separate."""

    def __init__(self, c, rank, group, out_features, in_features, device):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity
        from moonep.prefetch import launch_prefetch

        self.cfg, self.rank, self.out_features = c, rank, out_features
        self.prefetch_kernel = launch_prefetch
        gran = int(get_vmm_granularity())
        align = math.lcm(128, gran // math.gcd(gran, 2 * in_features))
        rows = math.ceil(out_features / align) * align
        q = c.local_experts
        shape = [q, rows, in_features]
        self.weights = create_nvl_dist_tensor(
            shape, torch.bfloat16, rank, c.ep_size, group=group
        )
        self.replicas = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        self.local_weight = self.weights[rank * q : (rank + 1) * q]
        self.local_weight.zero_()
        self.weight_views = [
            row[:out_features, :]
            for bank in (self.local_weight, self.replicas)
            for row in bank
        ]

    @property
    def parameter_data(self):
        return self.local_weight[:, : self.out_features, :]

    def publish(self, weights):
        if weights.dtype != torch.bfloat16:
            raise ValueError(
                "BF16 compute weights must be initialized before execution; no FP32 masters are accepted"
            )
        destination = self.parameter_data
        if weights.data_ptr() == destination.data_ptr():
            if weights.stride() != destination.stride():
                raise ValueError(
                    "BF16 component aliases its destination with incompatible strides"
                )
            return
        base = self.weights.untyped_storage().data_ptr()
        if base <= weights.data_ptr() < base + self.weights.untyped_storage().nbytes():
            raise ValueError(
                "BF16 component aliases a different publication destination"
            )
        # Relocated/reloaded graph inputs remain authoritative. This is a
        # BF16 byte copy only; normal bound parameters already alias destination.
        destination.copy_(weights)

    def pointers(self):
        return tuple(value.data_ptr() for value in self.weight_views)

    def prefetch(self, plan):
        self.prefetch_kernel(
            self.weights,
            self.replicas,
            plan.experts_to_copy[self.rank],
            self.cfg.num_comm_sms,
        )

    def external_tensors(self):
        return [self.weights]


class _ByteComponentBank:
    """One fixed TE component per expert, transported by unchanged MoonEP kernels."""

    def __init__(self, c, rank, group, shape, dtype, device):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity
        from moonep.prefetch import launch_prefetch

        self.cfg, self.rank, self.launch = c, rank, launch_prefetch
        self.shape, self.dtype = shape, dtype
        nbytes = math.prod(shape) * dtype.itemsize
        alignment = math.lcm(int(get_vmm_granularity()), 128 * 128)
        expert_bytes = math.ceil(nbytes / alignment) * alignment
        physical = [c.local_experts, 128, expert_bytes // 128]
        self.remote = create_nvl_dist_tensor(
            physical, torch.uint8, rank, c.ep_size, group=group
        )
        self.local = self.remote[rank * c.local_experts : (rank + 1) * c.local_experts]
        self.replicas = torch.zeros(physical, device=device, dtype=torch.uint8)
        self.local.zero_()
        self.views = [
            row.flatten()[:nbytes].view(dtype).view(shape)
            for bank in (self.local, self.replicas)
            for row in bank
        ]

    def prefetch(self, plan):
        self.launch(
            self.remote,
            self.replicas,
            plan.experts_to_copy[self.rank],
            self.cfg.num_comm_sms,
        )


class _FP8WeightState:
    """FP8 compute representation only: payloads, scales, and required orientations."""

    def __init__(self, c, rank, group, out_features, in_features, device):
        self.cfg, self.rank = c, rank
        self.out_features, self.in_features = out_features, in_features
        self.quantizer = _make_quantizer(c.compute_precision)
        specs = self.quantizer.inner_tensor_specs((out_features, in_features))
        self.components = {
            name: _ByteComponentBank(c, rank, group, shape, dtype, device)
            for name, (shape, dtype) in specs.items()
        }
        self.weight_views = []
        for index in range(2 * c.local_experts):
            value = self.quantizer.make_empty(
                (out_features, in_features), dtype=c.weight_grad_dtype, device=device
            )
            for name, bank in self.components.items():
                setattr(value, name, bank.views[index])
            if hasattr(value, "_transpose_invalid"):
                value._transpose_invalid = False
            self.weight_views.append(value)

    def publish(self, w):
        # Import the supplied representation into address-stable communication
        # storage. This is a byte copy, not quantization or an optimizer refresh.
        # The enclosing phase joins every EP rank before remote prefetch reads.
        from mlops.expert_parallel.transformer_engine.parameters.components import (
            component_names,
        )

        names = component_names(self.cfg.compute_precision)
        self.validate_components(w)
        for index, fields in enumerate(w):
            for name, source in zip(names, fields, strict=True):
                destination = self.components[name].views[index]
                if source.data_ptr() != destination.data_ptr():
                    destination.copy_(source)

    def validate_components(self, weights):
        from mlops.expert_parallel.transformer_engine.parameters.components import (
            component_names,
        )

        names = component_names(self.cfg.compute_precision)
        if len(weights) != self.cfg.local_experts:
            raise ValueError("Incorrect FP8 expert component group count")
        for index, fields in enumerate(weights):
            if len(fields) != len(names):
                raise ValueError("Incomplete FP8 weight representation")
            for name, source in zip(names, fields, strict=True):
                expected = self.components[name].views[index]
                if (
                    source.shape != expected.shape
                    or source.dtype != expected.dtype
                    or source.device != expected.device
                    or not source.is_contiguous()
                ):
                    raise ValueError(
                        f"Invalid FP8 component {index}/{name}: expected "
                        f"{tuple(expected.shape)} {expected.dtype} on {expected.device}"
                    )

    def pointers(self):
        return tuple(
            getattr(value, name).data_ptr()
            for value in self.weight_views
            for name in self.components
        )

    def prefetch(self, plan):
        for bank in self.components.values():
            bank.prefetch(plan)

    def external_tensors(self):
        return [bank.remote for bank in self.components.values()]


class _ExpertBank:
    """One gradient allocation/reduction/ownership path for every precision."""

    def __init__(
        self, c, rank, group, out_features, in_features, device, *, trainable=True
    ):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity

        from mlops.expert_parallel.transformer_engine.kernels.grad_reduce import (
            launch_grad_reduce,
        )

        self.cfg, self.rank, self.device = c, rank, device
        self.out_features, self.in_features = out_features, in_features
        self.reduce_kernel = launch_grad_reduce
        representation = (
            _BF16WeightState if c.compute_precision == "bf16" else _FP8WeightState
        )
        self.weight_state = representation(
            c, rank, group, out_features, in_features, device
        )
        self.trainable = trainable
        self.result_dtype = c.weight_grad_dtype
        if not trainable:
            return
        gran = int(get_vmm_granularity())
        alignment = math.lcm(128, gran // math.gcd(gran, 4 * in_features))
        rows = math.ceil(out_features / alignment) * alignment
        q, G = c.local_experts, c.ep_size
        self.shape = (q, rows, in_features)
        self._replica_grad_owner = create_nvl_dist_tensor(
            list(self.shape), torch.float32, rank, G, group=group
        )
        self.replica_grad = self._replica_grad_owner.view(G, q, rows, in_features)
        self.local_replica_grad = self.replica_grad[rank]
        self.local_replica_grad.zero_()
        self.local_home_grad = (
            torch.zeros(self.shape, device=device, dtype=torch.float32)
            if c.gradient_output_mode == "copy"
            else None
        )
        self.grad_views = []
        self.last_returned_pointer = None

    @property
    def weight_views(self):
        return self.weight_state.weight_views

    @property
    def components(self):
        return self.weight_state.components

    @property
    def quantizer(self):
        return self.weight_state.quantizer

    def pointers(self):
        return self.weight_state.pointers()

    def validate_components(self, weights):
        return self.weight_state.validate_components(weights)

    def publish(self, weights):
        self.weight_state.publish(weights)

    def prefetch(self, plan):
        self.weight_state.prefetch(plan)

    def _set_grad_views(self):
        self.grad_views = [
            row[: self.out_features, :]
            for bank in (self.local_home_grad, self.local_replica_grad)
            for row in bank
        ]

    def prepare_grad_scratch(self):
        if self.cfg.gradient_output_mode == "owned":
            if self.local_home_grad is not None:
                raise RuntimeError(
                    "Previous backward did not transfer its gradient result"
                )
            self.local_home_grad = torch.empty(
                self.shape, device=self.device, dtype=torch.float32
            )
            # Logical entries are overwritten by beta=0, even for empty groups.
            # Reduction also visits alignment padding, so initialize that suffix.
            if self.shape[1] != self.out_features:
                self.local_home_grad[:, self.out_features :, :].zero_()
        self._set_grad_views()

    def zero_grad_scratch(self):
        """Full initialization for diagnostic references, outside the fast path."""
        if self.local_home_grad is None:
            self.prepare_grad_scratch()
        else:
            self._set_grad_views()
        self.local_home_grad.zero_()
        self.local_replica_grad.zero_()

    def reduce(self, plan, ctx):
        self.reduce_kernel(
            self.local_home_grad,
            self.replica_grad,
            plan.experts_to_copy,
            rank=self.rank,
            num_sms=self.cfg.num_comm_sms,
            meta_buf=ctx["meta_buf"],
            meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]),
            grid_sync_bar=ctx["grid_sync_bar"],
        )

    def grad_result(self, weights):
        view = self.local_home_grad[:, : self.out_features, :]
        # A dtype conversion or removal of inter-expert padding must materialize
        # the public contiguous result. Unpadded FP32 owned results transfer as-is.
        result = view.to(
            dtype=self.result_dtype,
            copy=self.cfg.gradient_output_mode == "copy" or not view.is_contiguous(),
            memory_format=torch.contiguous_format,
        )
        self.last_returned_pointer = result.data_ptr()
        self.grad_views = []
        if self.cfg.gradient_output_mode == "owned":
            self.local_home_grad = None
        return result

    def external_tensors(self):
        gradients = [self._replica_grad_owner] if self.trainable else []
        return [*gradients, *self.weight_state.external_tensors()]
