"""One PyTorch AdamW façade for local and distributed execution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch

from ..providers.builtin.adamw import adamw_, master_adamw_
from ._adamw_common import (
    DTypePolicy,
    normalize_dtype_policy,
    resolve_dtype,
    validate_adamw_options,
)


class AdamW(torch.optim.Optimizer):
    """Mixed-dtype AdamW with optional replica-group synchronization."""

    implementation_id = "builtin.adamw.triton"

    def __init__(
        self,
        params: (
            Iterable[torch.Tensor]
            | Iterable[dict[str, Any]]
            | Iterable[tuple[str, torch.Tensor]]
        ),
        lr: float | torch.Tensor = 1e-3,
        betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        gradient_dtype: DTypePolicy = torch.bfloat16,
        reduction_dtype: DTypePolicy = torch.bfloat16,
        state_dtype: DTypePolicy = torch.bfloat16,
        master_parameter_dtype: DTypePolicy = torch.bfloat16,
        replica_group: Any | None = None,
        opt_state_strategy: Literal["replicated", "sharded"] = "sharded",
        gradient_reduction: Literal["sum", "mean"] = "mean",
        bucket_bytes: int = 64 << 20,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
            "gradient_dtype": normalize_dtype_policy(
                gradient_dtype, name="gradient_dtype"
            ),
            "reduction_dtype": normalize_dtype_policy(
                reduction_dtype, name="reduction_dtype"
            ),
            "state_dtype": normalize_dtype_policy(state_dtype, name="state_dtype"),
            "master_parameter_dtype": normalize_dtype_policy(
                master_parameter_dtype, name="master_parameter_dtype"
            ),
        }
        validate_adamw_options(defaults)
        if opt_state_strategy not in {"replicated", "sharded"}:
            raise ValueError(
                "opt_state_strategy must be 'replicated' or 'sharded'"
            )
        if gradient_reduction not in {"sum", "mean"}:
            raise ValueError("gradient_reduction must be 'sum' or 'mean'")
        if type(bucket_bytes) is not int or bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be one positive integer")
        super().__init__(params, defaults)
        for group in self.param_groups:
            validate_adamw_options(group)

        self.replica_group = replica_group
        self.opt_state_strategy = (
            None if replica_group is None else opt_state_strategy
        )
        self.gradient_reduction = gradient_reduction
        self.bucket_bytes = bucket_bytes
        self._distributed = None
        if replica_group is not None:
            # Imported lazily so the local optimizer has no distributed setup
            # or ProcessGroup dependency.
            from .distributed_adamw import DistributedAdamWRuntime

            self._distributed = DistributedAdamWRuntime(self)

    @property
    def distributed_spec(self):
        """Return immutable distributed semantics, or ``None`` for local use."""
        return None if self._distributed is None else self._distributed.spec

    @staticmethod
    def _initialize_parameter_state(
        parameter: torch.nn.Parameter,
        group: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        state_dtype = resolve_dtype(group["state_dtype"], parameter)
        master_dtype = resolve_dtype(group["master_parameter_dtype"], parameter)
        state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
        state["exp_avg"] = torch.zeros_like(parameter, dtype=state_dtype)
        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=state_dtype)
        if master_dtype != parameter.dtype:
            state["master_parameter"] = parameter.detach().to(master_dtype).clone()

    @torch.no_grad()
    def _local_step(self) -> None:
        for group in self.param_groups:
            validate_adamw_options(group)
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("mlops.optim.AdamW does not support sparse gradients")
                expected_gradient_dtype = resolve_dtype(
                    group["gradient_dtype"], parameter
                )
                update_gradient = (
                    gradient
                    if gradient.dtype == expected_gradient_dtype
                    else gradient.to(expected_gradient_dtype)
                )
                state = self.state[parameter]
                if not state:
                    self._initialize_parameter_state(parameter, group, state)
                common = {
                    "lr": float(group["lr"]),
                    "betas": tuple(float(value) for value in group["betas"]),
                    "eps": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "maximize": bool(group["maximize"]),
                }
                master = state.get("master_parameter")
                if master is None:
                    adamw_(
                        parameter,
                        update_gradient,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        **common,
                    )
                else:
                    master_adamw_(
                        parameter,
                        master,
                        update_gradient,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        **common,
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one local or distributed update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._distributed is None:
            self._local_step()
        else:
            self._distributed.step()
        return loss

    def synchronize(self) -> None:
        """Retire outstanding distributed work; local execution is a no-op."""
        if self._distributed is not None:
            self._distributed.synchronize()

    def state_dict(self):
        """Return standard local state or a rank-local distributed state."""
        if self._distributed is None:
            return super().state_dict()
        return self._distributed.state_dict()

    def load_state_dict(self, state_dict):
        """Restore standard local state or matching distributed state."""
        if self._distributed is None:
            return super().load_state_dict(state_dict)
        self._distributed.load_state_dict(state_dict)
        return None

    def execution_manifest(self):
        """Return distributed lowering hints, or ``None`` locally."""
        if self._distributed is None:
            return None
        return self._distributed.execution_manifest()


__all__ = ["AdamW"]
