"""Compute-weight banks and two-bank gradient ownership over MoonEP."""

from __future__ import annotations

import math

import torch


class _WeightState:
    """One local [home, replica] compute bank without a weight concat/copy.

    Each rank maps 2*q slots. The first q hold home weights, the next q replicas.
    Only slot indices passed to the unchanged MoonEP prefetch kernel are remapped.
    No additional physical model-weight bank is introduced.
    """

    def __init__(self, c, rank, group, out_features, in_features, device):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity
        from moonep.prefetch import launch_prefetch

        self.cfg, self.rank, self.out_features = c, rank, out_features
        self.prefetch_kernel = launch_prefetch
        gran = int(get_vmm_granularity())
        alignment = math.lcm(128, gran // math.gcd(gran, 2 * in_features))
        rows = math.ceil(out_features / alignment) * alignment
        q = c.local_experts
        self.weights = create_nvl_dist_tensor(
            [2 * q, rows, in_features], torch.bfloat16, rank, c.ep_size, group=group
        )
        local = self.weights[rank * 2 * q : (rank + 1) * 2 * q]
        self.local_weight, self.replicas = local[:q], local[q:]
        local.zero_()
        self.compute_weights = local[:, :out_features, :]
        self.weight_views = [row[:out_features, :] for row in local]

    def prefetch_slots(self, slots):
        self.prefetch_kernel(self.weights, self.replicas, slots, self.cfg.num_comm_sms)

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

    def external_tensors(self):
        return [self.weights]


class _FP8WeightState:
    """Four physical components with contiguous [home, replica] expert slots."""

    def __init__(self, c, rank, group, out_features, in_features, device):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity
        from moonep.prefetch import launch_prefetch

        self.cfg, self.rank, self.prefetch_kernel = c, rank, launch_prefetch
        self.remote, self.replicas, self.compute_weights = [], [], []
        q = c.local_experts
        shapes = (
            (out_features, in_features),
            (in_features, out_features),
            (out_features,),
            (in_features,),
        )
        dtypes = (
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float32,
            torch.float32,
        )
        for shape, dtype in zip(shapes, dtypes):
            nbytes = math.prod(shape) * dtype.itemsize
            alignment = math.lcm(int(get_vmm_granularity()), 128 * 128)
            pitch = math.ceil(nbytes / alignment) * alignment
            remote = create_nvl_dist_tensor(
                [2 * q, 128, pitch // 128], torch.uint8, rank, c.ep_size, group=group
            )
            local = remote[rank * 2 * q : (rank + 1) * 2 * q]
            local.zero_()
            data = local.view(2 * q, pitch)[:, :nbytes].view(dtype)
            data = data.view(2 * q, *shape)
            self.remote.append(remote)
            self.replicas.append(local[q:])
            self.compute_weights.append(data)
        self.compute_weights = tuple(self.compute_weights)
        self.parameter_components = tuple(v[:q] for v in self.compute_weights)

    def validate_components(self, weights):
        if len(weights) != 4:
            raise ValueError(
                "FP8 requires both payload orientations and both descale vectors"
            )
        for source, destination in zip(weights, self.parameter_components):
            if (
                source.shape != destination.shape
                or source.dtype != destination.dtype
                or source.device != destination.device
            ):
                raise ValueError("Invalid QuACK FP8 physical component")
            if source.data_ptr() == destination.data_ptr():
                if source.stride() != destination.stride():
                    raise ValueError(
                        "FP8 component aliases its destination with incompatible strides"
                    )
            elif any(
                remote.untyped_storage().data_ptr()
                <= source.data_ptr()
                < remote.untyped_storage().data_ptr()
                + remote.untyped_storage().nbytes()
                for remote in self.remote
            ):
                raise ValueError(
                    "FP8 component aliases a different publication destination"
                )

    def publish(self, weights):
        self.validate_components(weights)
        for source, destination in zip(weights, self.parameter_components):
            if source.data_ptr() != destination.data_ptr():
                destination.copy_(source)

    def prefetch_slots(self, slots):
        for remote, replica in zip(self.remote, self.replicas):
            self.prefetch_kernel(remote, replica, slots, self.cfg.num_comm_sms)

    def pointers(self):
        return tuple(v.data_ptr() for v in self.compute_weights)

    def external_tensors(self):
        return self.remote


class _Bank:
    def __init__(
        self, c, rank, group, out_features, in_features, device, *, trainable=True
    ):
        from moonep.buffer import create_nvl_dist_tensor, get_vmm_granularity

        from mlops.expert_parallel.quack.kernels.grad_reduce import launch_grad_reduce

        self.cfg, self.rank, self.device = c, rank, device
        self.out_features, self.in_features = out_features, in_features
        self.reduce_kernel = launch_grad_reduce
        representation = (
            _WeightState if c.compute_precision == "bf16" else _FP8WeightState
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
        # The reducer clears every consumed replica slot after its cross-rank
        # barrier. Unused slots stay zero from initialization. Empty replica
        # groups therefore need no additional write on the compute stream.
        self.replica_grad_cleared_after_reduce = True
        self.local_home_grad = (
            torch.zeros(self.shape, device=device, dtype=torch.float32)
            if c.gradient_output_mode == "copy"
            else None
        )
        self.grad_views = []
        self.last_returned_pointer = None

    def pointers(self):
        return self.weight_state.pointers()

    def publish(self, weights):
        self.weight_state.publish(weights)

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
