"""Private ProcessGroup runtime beneath :class:`mlops.optim.AdamW`.

The public optimizer owns ordinary Parameters and parameter groups. This
module owns deterministic flat buckets, collective submission, optimizer
state placement, and stream completion. It deliberately exposes no second
optimizer class.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ..providers.builtin.adamw import adamw_, master_adamw_
from ._adamw_common import resolve_dtype, validate_adamw_options

if TYPE_CHECKING:
    from .adamw_optimizer import AdamW


@dataclass(frozen=True)
class DistributedAdamWSpec:
    """Immutable semantics needed to identify one distributed deployment."""

    schema_version: str
    opt_state_strategy: str
    gradient_reduction: str
    bucket_bytes: int
    world_size: int
    rank: int
    collective_backend: str
    completion_mode: str
    manifest_sha256: str
    local_implementation_id: str


@dataclass(frozen=True)
class _ParameterRecord:
    parameter: torch.nn.Parameter
    arena_id: int
    parameter_element_offset: int
    bucket_element_offset: int
    arena_element_offset: int
    element_count: int


@dataclass
class _Bucket:
    bucket_id: int
    arena_id: int
    arena_element_offset: int
    group_index: int
    records: tuple[_ParameterRecord, ...]
    parameter_dtype: torch.dtype
    gradient_dtype: torch.dtype
    reduction_dtype: torch.dtype
    state_dtype: torch.dtype
    master_dtype: torch.dtype
    element_count: int
    padded_element_count: int
    shard_element_count: int
    flat_parameter: torch.Tensor | None = None
    update_parameter: torch.Tensor | None = None
    master_parameter: torch.Tensor | None = None
    exp_avg: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    step: torch.Tensor | None = None


@dataclass
class _Workspace:
    """Task-lifetime scratch shared by compatible serialized buckets."""

    local_gradient: torch.Tensor
    reduction_input: torch.Tensor
    reduced_gradient: torch.Tensor


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class DistributedAdamWRuntime:
    """Execute replicated or state-sharded AdamW behind one public façade."""

    def __init__(self, optimizer: AdamW) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "a group-backed mlops.optim.AdamW requires initialized "
                "torch.distributed"
            )
        self.optimizer = optimizer
        self.replica_group = optimizer.replica_group
        self.world_size = dist.get_world_size(self.replica_group)
        self.rank = dist.get_rank(self.replica_group)
        self.collective_backend = str(dist.get_backend(self.replica_group))
        # ProcessGroupGloo implements CUDA collectives but does not provide a
        # stream-safe future for every Work type (notably reduce-scatter in
        # PyTorch 2.13). A pure host wait is its portable completion boundary.
        self.completion_mode = (
            "stream_bridge"
            if self.collective_backend == "nccl"
            else "host_wait"
        )
        self._materialized = False
        self._pending_work: list[Any] = []
        self._completion_events: list[torch.cuda.Event] = []
        self._parameter_arenas: dict[int, torch.Tensor] = {}

        descriptors, parameters = self._describe_parameters()
        manifest_text = json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
        manifest_sha256 = hashlib.sha256(manifest_text.encode()).hexdigest()
        gathered: list[str | None] = [None] * self.world_size
        dist.all_gather_object(
            gathered,
            manifest_sha256,
            group=self.replica_group,
        )
        if any(value != manifest_sha256 for value in gathered):
            raise ValueError(
                "distributed AdamW parameter manifests differ across group "
                f"members: {gathered}"
            )
        self._manifest_sha256 = manifest_sha256
        self._parameters = parameters
        self.buckets = self._make_buckets()
        self.spec = DistributedAdamWSpec(
            schema_version="mlops.distributed_adamw.v3",
            opt_state_strategy=str(optimizer.opt_state_strategy),
            gradient_reduction=optimizer.gradient_reduction,
            bucket_bytes=optimizer.bucket_bytes,
            world_size=self.world_size,
            rank=self.rank,
            collective_backend=self.collective_backend,
            completion_mode=self.completion_mode,
            manifest_sha256=manifest_sha256,
            local_implementation_id=optimizer.implementation_id,
        )

    def _describe_parameters(
        self,
    ) -> tuple[list[dict[str, Any]], tuple[torch.nn.Parameter, ...]]:
        descriptors: list[dict[str, Any]] = []
        parameters: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        devices: set[torch.device] = set()
        for group_index, group in enumerate(self.optimizer.param_groups):
            validate_adamw_options(group)
            for parameter_index, parameter in enumerate(group["params"]):
                if not parameter.requires_grad:
                    continue
                identity = id(parameter)
                if identity in seen:
                    raise ValueError(
                        "distributed AdamW does not admit duplicate trainable "
                        "Parameter objects"
                    )
                seen.add(identity)
                if parameter.device.type != "cuda":
                    raise ValueError("distributed AdamW requires CUDA Parameters")
                if not parameter.is_contiguous():
                    raise ValueError(
                        "distributed AdamW requires contiguous Parameters"
                    )
                if parameter.dtype not in {
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                }:
                    raise ValueError(
                        "distributed AdamW requires BF16, FP16, or FP32 Parameters"
                    )
                devices.add(parameter.device)
                parameters.append(parameter)
                descriptors.append(
                    {
                        "group": group_index,
                        "parameter": parameter_index,
                        "shape": list(parameter.shape),
                        "parameter_dtype": _dtype_name(parameter.dtype),
                        "gradient_dtype": _dtype_name(
                            resolve_dtype(group["gradient_dtype"], parameter)
                        ),
                        "reduction_dtype": _dtype_name(
                            resolve_dtype(group["reduction_dtype"], parameter)
                        ),
                        "state_dtype": _dtype_name(
                            resolve_dtype(group["state_dtype"], parameter)
                        ),
                        "master_dtype": _dtype_name(
                            resolve_dtype(
                                group["master_parameter_dtype"], parameter
                            )
                        ),
                        "lr": float(group["lr"]),
                        "betas": [float(value) for value in group["betas"]],
                        "eps": float(group["eps"]),
                        "weight_decay": float(group["weight_decay"]),
                        "maximize": bool(group["maximize"]),
                    }
                )
        if not parameters:
            raise ValueError(
                "a group-backed AdamW requires at least one trainable Parameter"
            )
        if len(devices) != 1:
            raise ValueError(
                "one group-backed AdamW currently requires one local CUDA device"
            )
        self.device = next(iter(devices))
        return descriptors, tuple(parameters)

    def _make_buckets(self) -> list[_Bucket]:
        buckets: list[_Bucket] = []
        active: list[tuple[torch.nn.Parameter, int, int]] = []
        active_signature: tuple[Any, ...] | None = None
        active_elements = 0
        arena_id = -1
        arena_element_offset = 0
        capacity_elements = 0

        def flush() -> None:
            nonlocal active, active_elements, arena_element_offset
            if not active:
                return
            (
                group_index,
                parameter_dtype,
                gradient_dtype,
                reduction_dtype,
                state_dtype,
                master_dtype,
            ) = active_signature
            bucket_element_offset = 0
            records = []
            for (
                parameter,
                parameter_element_offset,
                element_count,
            ) in active:
                records.append(
                    _ParameterRecord(
                        parameter=parameter,
                        arena_id=arena_id,
                        parameter_element_offset=parameter_element_offset,
                        bucket_element_offset=bucket_element_offset,
                        arena_element_offset=(
                            arena_element_offset + bucket_element_offset
                        ),
                        element_count=element_count,
                    )
                )
                bucket_element_offset += element_count
            multiple = (
                self.world_size
                if self.optimizer.opt_state_strategy == "sharded"
                else 1
            )
            padded = (
                (active_elements + multiple - 1) // multiple
            ) * multiple
            buckets.append(
                _Bucket(
                    bucket_id=len(buckets),
                    arena_id=arena_id,
                    arena_element_offset=arena_element_offset,
                    group_index=group_index,
                    records=tuple(records),
                    parameter_dtype=parameter_dtype,
                    gradient_dtype=gradient_dtype,
                    reduction_dtype=reduction_dtype,
                    state_dtype=state_dtype,
                    master_dtype=master_dtype,
                    element_count=active_elements,
                    padded_element_count=padded,
                    shard_element_count=padded // multiple,
                )
            )
            arena_element_offset += padded
            active = []
            active_elements = 0

        for group_index, group in enumerate(self.optimizer.param_groups):
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                reduction_dtype = resolve_dtype(group["reduction_dtype"], parameter)
                signature = (
                    group_index,
                    parameter.dtype,
                    resolve_dtype(group["gradient_dtype"], parameter),
                    reduction_dtype,
                    resolve_dtype(group["state_dtype"], parameter),
                    resolve_dtype(group["master_parameter_dtype"], parameter),
                )
                if signature != active_signature:
                    flush()
                    active_signature = signature
                    arena_id += 1
                    arena_element_offset = 0
                    multiple = (
                        self.world_size
                        if self.optimizer.opt_state_strategy == "sharded"
                        else 1
                    )
                    raw_capacity = (
                        self.optimizer.bucket_bytes // reduction_dtype.itemsize
                    )
                    capacity_elements = (
                        raw_capacity // multiple
                    ) * multiple
                    if capacity_elements < multiple:
                        raise ValueError(
                            "bucket_bytes is too small for one collective shard: "
                            f"{self.optimizer.bucket_bytes} bytes for world size "
                            f"{self.world_size} and {_dtype_name(reduction_dtype)}"
                        )

                parameter_element_offset = 0
                while parameter_element_offset < parameter.numel():
                    take = min(
                        capacity_elements - active_elements,
                        parameter.numel() - parameter_element_offset,
                    )
                    active.append(
                        (
                            parameter,
                            parameter_element_offset,
                            take,
                        )
                    )
                    active_elements += take
                    parameter_element_offset += take
                    if active_elements == capacity_elements:
                        flush()
        flush()
        return buckets

    @torch.no_grad()
    def _materialize(self) -> None:
        if self._materialized:
            return

        arena_specs: dict[int, tuple[torch.dtype, int]] = {}
        for bucket in self.buckets:
            arena_end = (
                bucket.arena_element_offset + bucket.padded_element_count
            )
            previous = arena_specs.get(bucket.arena_id)
            if previous is None:
                arena_specs[bucket.arena_id] = (
                    bucket.parameter_dtype,
                    arena_end,
                )
            else:
                dtype, size = previous
                if dtype != bucket.parameter_dtype:
                    raise AssertionError("one parameter arena has multiple dtypes")
                arena_specs[bucket.arena_id] = (dtype, max(size, arena_end))

        self._parameter_arenas = {
            arena_id: torch.zeros(
                element_count,
                device=self.device,
                dtype=dtype,
            )
            for arena_id, (dtype, element_count) in arena_specs.items()
        }

        records_by_parameter: dict[int, list[_ParameterRecord]] = {}
        for bucket in self.buckets:
            arena = self._parameter_arenas[bucket.arena_id]
            for record in bucket.records:
                parameter_start = record.parameter_element_offset
                parameter_stop = parameter_start + record.element_count
                arena_start = record.arena_element_offset
                arena_stop = arena_start + record.element_count
                arena[arena_start:arena_stop].copy_(
                    record.parameter.detach().reshape(-1)[
                        parameter_start:parameter_stop
                    ]
                )
                records_by_parameter.setdefault(id(record.parameter), []).append(
                    record
                )

        for parameter in self._parameters:
            records = sorted(
                records_by_parameter[id(parameter)],
                key=lambda value: value.parameter_element_offset,
            )
            first = records[0]
            for record in records:
                if (
                    record.arena_element_offset
                    != first.arena_element_offset
                    + record.parameter_element_offset
                ):
                    raise AssertionError(
                        "parameter chunks are not contiguous in their arena"
                    )
            arena = self._parameter_arenas[first.arena_id]
            start = first.arena_element_offset
            parameter.data = arena[start : start + parameter.numel()].view(
                parameter.shape
            )

        for bucket in self.buckets:
            arena = self._parameter_arenas[bucket.arena_id]
            start = bucket.arena_element_offset
            bucket.flat_parameter = arena[
                start : start + bucket.padded_element_count
            ]

            state_count = (
                bucket.element_count
                if self.optimizer.opt_state_strategy == "replicated"
                else bucket.shard_element_count
            )
            if self.optimizer.opt_state_strategy == "replicated":
                bucket.update_parameter = bucket.flat_parameter[
                    : bucket.element_count
                ]
            else:
                shard_start = self.rank * bucket.shard_element_count
                shard_stop = shard_start + bucket.shard_element_count
                bucket.update_parameter = bucket.flat_parameter[
                    shard_start:shard_stop
                ].clone()
            bucket.exp_avg = torch.zeros(
                state_count,
                device=self.device,
                dtype=bucket.state_dtype,
            )
            bucket.exp_avg_sq = torch.zeros_like(bucket.exp_avg)
            bucket.step = torch.zeros((), device=self.device, dtype=torch.int64)
            if bucket.master_dtype != bucket.parameter_dtype:
                bucket.master_parameter = bucket.update_parameter.to(
                    bucket.master_dtype
                ).clone()
        self._materialized = True

    def _workspace_key(self, bucket: _Bucket) -> tuple[torch.dtype, torch.dtype]:
        return bucket.gradient_dtype, bucket.reduction_dtype

    def _allocate_workspaces(self) -> dict[tuple[torch.dtype, torch.dtype], _Workspace]:
        capacities: dict[tuple[torch.dtype, torch.dtype], tuple[int, int]] = {}
        for bucket in self.buckets:
            key = self._workspace_key(bucket)
            padded, shard = capacities.get(key, (0, 0))
            capacities[key] = (
                max(padded, bucket.padded_element_count),
                max(shard, bucket.shard_element_count),
            )
        workspaces = {}
        for (gradient_dtype, reduction_dtype), (padded, shard) in capacities.items():
            local_gradient = torch.empty(
                padded,
                device=self.device,
                dtype=gradient_dtype,
            )
            reduction_input = (
                local_gradient
                if gradient_dtype == reduction_dtype
                else torch.empty(
                    padded,
                    device=self.device,
                    dtype=reduction_dtype,
                )
            )
            reduced_count = (
                padded
                if self.optimizer.opt_state_strategy == "replicated"
                else shard
            )
            reduced_gradient = (
                reduction_input
                if self.optimizer.opt_state_strategy == "replicated"
                else torch.empty(
                    reduced_count,
                    device=self.device,
                    dtype=reduction_dtype,
                )
            )
            workspaces[(gradient_dtype, reduction_dtype)] = _Workspace(
                local_gradient=local_gradient,
                reduction_input=reduction_input,
                reduced_gradient=reduced_gradient,
            )
        return workspaces

    @torch.no_grad()
    def _pack(self, bucket: _Bucket, workspace: _Workspace) -> None:
        local_gradient = workspace.local_gradient[
            : bucket.padded_element_count
        ]
        local_gradient.zero_()
        for record in bucket.records:
            gradient = record.parameter.grad
            if gradient is None:
                raise RuntimeError(
                    "distributed AdamW requires one dense gradient for every "
                    "admitted trainable Parameter on every group member"
                )
            if gradient.is_sparse:
                raise RuntimeError(
                    "distributed AdamW does not support sparse gradients"
                )
            parameter_start = record.parameter_element_offset
            parameter_stop = parameter_start + record.element_count
            bucket_start = record.bucket_element_offset
            bucket_stop = bucket_start + record.element_count
            local_gradient[bucket_start:bucket_stop].copy_(
                gradient.detach().reshape(-1)[parameter_start:parameter_stop]
            )
        reduction_input = workspace.reduction_input[
            : bucket.padded_element_count
        ]
        if reduction_input.data_ptr() != local_gradient.data_ptr():
            reduction_input.copy_(local_gradient)

    def _complete_collective(self, work: Any) -> None:
        if self.completion_mode == "stream_bridge":
            work.block_current_stream()
            self._pending_work.append(work)
        else:
            work.wait()

    def _launch_reduction(
        self,
        bucket: _Bucket,
        workspace: _Workspace,
    ) -> torch.Tensor:
        reduction_input = workspace.reduction_input[
            : bucket.padded_element_count
        ]
        if self.optimizer.opt_state_strategy == "replicated":
            reduced_gradient = reduction_input[: bucket.element_count]
            work = dist.all_reduce(
                reduction_input,
                op=dist.ReduceOp.SUM,
                group=self.replica_group,
                async_op=True,
            )
        else:
            reduced_gradient = workspace.reduced_gradient[
                : bucket.shard_element_count
            ]
            work = dist.reduce_scatter_single(
                reduced_gradient,
                reduction_input,
                op=dist.ReduceOp.SUM,
                group=self.replica_group,
                async_op=True,
            )
        self._complete_collective(work)
        return reduced_gradient

    @torch.no_grad()
    def _apply_update(
        self,
        bucket: _Bucket,
        reduced_gradient: torch.Tensor,
    ) -> None:
        group = self.optimizer.param_groups[bucket.group_index]
        common = {
            "gradient_scale": (
                1.0 / self.world_size
                if self.optimizer.gradient_reduction == "mean"
                else 1.0
            ),
            "lr": float(group["lr"]),
            "betas": tuple(float(value) for value in group["betas"]),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "maximize": bool(group["maximize"]),
        }
        if bucket.master_parameter is None:
            adamw_(
                bucket.update_parameter,
                reduced_gradient,
                bucket.exp_avg,
                bucket.exp_avg_sq,
                bucket.step,
                **common,
            )
        else:
            master_adamw_(
                bucket.update_parameter,
                bucket.master_parameter,
                reduced_gradient,
                bucket.exp_avg,
                bucket.exp_avg_sq,
                bucket.step,
                **common,
            )

    def _launch_all_gather(self, bucket: _Bucket) -> None:
        work = dist.all_gather_single(
            bucket.flat_parameter,
            bucket.update_parameter,
            group=self.replica_group,
            async_op=True,
        )
        self._complete_collective(work)

    def _retire_completed_work(self) -> None:
        self._pending_work = [
            work for work in self._pending_work if not work.is_completed()
        ]
        self._completion_events = [
            event for event in self._completion_events if not event.query()
        ]

    @torch.no_grad()
    def step(self) -> None:
        self._retire_completed_work()
        self._materialize()
        workspaces = self._allocate_workspaces()
        for bucket in self.buckets:
            workspace = workspaces[self._workspace_key(bucket)]
            self._pack(bucket, workspace)
            reduced_gradient = self._launch_reduction(bucket, workspace)
            self._apply_update(bucket, reduced_gradient)
            if self.optimizer.opt_state_strategy == "sharded":
                self._launch_all_gather(bucket)
        complete = torch.cuda.Event()
        complete.record(torch.cuda.current_stream(self.device))
        self._completion_events.append(complete)

    def synchronize(self) -> None:
        """Host-wait for all explicitly outstanding collective work."""
        for work in self._pending_work:
            work.wait()
        for event in self._completion_events:
            event.synchronize()
        self._pending_work.clear()
        self._completion_events.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return a rank-local, same-topology optimizer checkpoint."""
        self._materialize()
        groups = torch.optim.Optimizer.state_dict(self.optimizer)["param_groups"]
        return {
            "state": {},
            "param_groups": groups,
            "mlops_distributed": {
                "spec": asdict(self.spec),
                "buckets": [
                    {
                        "bucket_id": bucket.bucket_id,
                        "exp_avg": bucket.exp_avg,
                        "exp_avg_sq": bucket.exp_avg_sq,
                        "step": bucket.step,
                        "master_parameter": bucket.master_parameter,
                    }
                    for bucket in self.buckets
                ],
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a checkpoint created by the same rank and topology."""
        payload = state_dict.get("mlops_distributed")
        if payload is None:
            raise ValueError("not an mlops distributed AdamW checkpoint")
        saved_spec = payload["spec"]
        for name in (
            "schema_version",
            "opt_state_strategy",
            "world_size",
            "rank",
            "manifest_sha256",
        ):
            if saved_spec[name] != getattr(self.spec, name):
                raise ValueError(
                    f"distributed AdamW checkpoint {name} mismatch: "
                    f"{saved_spec[name]!r} != {getattr(self.spec, name)!r}"
                )
        self._materialize()
        saved_buckets = payload["buckets"]
        if len(saved_buckets) != len(self.buckets):
            raise ValueError("distributed AdamW checkpoint bucket count differs")
        for bucket, saved in zip(self.buckets, saved_buckets, strict=True):
            if saved["bucket_id"] != bucket.bucket_id:
                raise ValueError("distributed AdamW checkpoint bucket order differs")
            bucket.exp_avg.copy_(saved["exp_avg"])
            bucket.exp_avg_sq.copy_(saved["exp_avg_sq"])
            bucket.step.copy_(saved["step"])
            if (bucket.master_parameter is None) != (
                saved["master_parameter"] is None
            ):
                raise ValueError("distributed AdamW checkpoint master policy differs")
            if bucket.master_parameter is not None:
                bucket.master_parameter.copy_(saved["master_parameter"])

    def execution_manifest(self) -> dict[str, Any]:
        """Describe local objects and internal workspace for lowering."""
        buckets = []
        workspace_capacities: dict[
            tuple[torch.dtype, torch.dtype], tuple[int, int]
        ] = {}
        for bucket in self.buckets:
            model_bytes = bucket.element_count * bucket.parameter_dtype.itemsize
            key = self._workspace_key(bucket)
            padded, shard = workspace_capacities.get(key, (0, 0))
            workspace_capacities[key] = (
                max(padded, bucket.padded_element_count),
                max(shard, bucket.shard_element_count),
            )
            state_elements = (
                bucket.element_count
                if self.optimizer.opt_state_strategy == "replicated"
                else bucket.shard_element_count
            )
            persistent_bytes = state_elements * (
                2 * bucket.state_dtype.itemsize
                + (
                    bucket.master_dtype.itemsize
                    if bucket.master_dtype != bucket.parameter_dtype
                    else 0
                )
            ) + torch.tensor([], dtype=torch.int64).element_size()
            parameter_shard_bytes = (
                bucket.shard_element_count * bucket.parameter_dtype.itemsize
                if self.optimizer.opt_state_strategy == "sharded"
                else 0
            )
            bucket_workspace_bytes = (
                bucket.padded_element_count * bucket.gradient_dtype.itemsize
            )
            if bucket.gradient_dtype != bucket.reduction_dtype:
                bucket_workspace_bytes += (
                    bucket.padded_element_count
                    * bucket.reduction_dtype.itemsize
                )
            if self.optimizer.opt_state_strategy == "sharded":
                bucket_workspace_bytes += (
                    bucket.shard_element_count
                    * bucket.reduction_dtype.itemsize
                )
            buckets.append(
                {
                    "bucket_id": bucket.bucket_id,
                    "group_index": bucket.group_index,
                    "parameter_bytes": model_bytes,
                    "optimizer_state_bytes": persistent_bytes,
                    "parameter_shard_bytes": parameter_shard_bytes,
                    "workspace_bytes": bucket_workspace_bytes,
                }
            )

        workspace_bytes = 0
        for (gradient_dtype, reduction_dtype), (
            padded,
            shard,
        ) in workspace_capacities.items():
            workspace_bytes += padded * gradient_dtype.itemsize
            if gradient_dtype != reduction_dtype:
                workspace_bytes += padded * reduction_dtype.itemsize
            if self.optimizer.opt_state_strategy == "sharded":
                workspace_bytes += shard * reduction_dtype.itemsize
        return {
            "spec": asdict(self.spec),
            "buckets": buckets,
            "workspace_bytes": workspace_bytes,
        }


__all__ = ["DistributedAdamWRuntime", "DistributedAdamWSpec"]
