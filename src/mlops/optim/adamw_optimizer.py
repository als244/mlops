"""Local mixed-dtype AdamW with explicit tensor state."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ..providers.builtin.adamw import Rounding, adamw_
from ._adamw_common import (
    DTypePolicy,
    hold_settings_on_host,
    normalize_dtype_policy,
    resolve_dtype,
    validate_adamw_options,
)


class AdamW(torch.optim.Optimizer):
    """Coordinate-wise AdamW; communication and sharding belong to the caller."""

    implementation_id = "builtin.adamw.triton"
    supports_flat_parameter_shards = True
    zero_lr_preserves_state = True

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
        opt_state_dtype: DTypePolicy = torch.bfloat16,
        parameter_rounding: Rounding = "nearest",
        opt_state_rounding: Rounding = "nearest",
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
            "opt_state_dtype": normalize_dtype_policy(
                opt_state_dtype, name="opt_state_dtype"
            ),
            "parameter_rounding": parameter_rounding,
            "opt_state_rounding": opt_state_rounding,
        }
        validate_adamw_options(defaults)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        offset = sum(len(group["params"]) for group in self.param_groups)
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        hold_settings_on_host(group)
        validate_adamw_options(group)
        # Independent host scalars are ordinary captured inputs. Constructing
        # them in step() would specialize each graph to a literal parameter
        # index; views into one vector would introduce shared storage instead.
        group["rounding_salts"] = tuple(
            torch.tensor(offset + index, dtype=torch.int64, device="cpu")
            for index in range(len(group["params"]))
        )

    def state_dict(self) -> dict[str, Any]:
        """Save update state; deterministic parameter salts are reconstructed."""
        state = super().state_dict()
        for group in state["param_groups"]:
            group.pop("rounding_salts", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # A group restored from a state dict that names no rounding was
        # rounded to nearest. One saved while these settings were named
        # state_dtype and state_rounding keeps its values under the current
        # names: defaulting them instead would quietly turn a stochastic
        # rounding of the moments into rounding to nearest.
        offset = 0
        for group in self.param_groups:
            if "rounding_salts" not in group:
                group["rounding_salts"] = tuple(
                    torch.tensor(offset + index, dtype=torch.int64, device="cpu")
                    for index in range(len(group["params"]))
                )
            offset += len(group["params"])
            for saved, name in (
                ("state_dtype", "opt_state_dtype"),
                ("state_rounding", "opt_state_rounding"),
            ):
                if saved in group:
                    group.setdefault(name, group.pop(saved))
            group.setdefault("parameter_rounding", "nearest")
            group.setdefault("opt_state_rounding", "nearest")

    @staticmethod
    def _initialize_parameter_state(
        parameter: torch.nn.Parameter,
        group: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        opt_state_dtype = resolve_dtype(group["opt_state_dtype"], parameter)
        state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
        state["exp_avg"] = torch.zeros_like(parameter, dtype=opt_state_dtype)
        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=opt_state_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        """Execute local AdamW, leaving tensor state unchanged when ``lr=0``.

        Zero LR still launches the update kernels. Missing state is initialized
        to zeros on first use; initialized moments and counters never advance.
        The closure, if supplied, retains its ordinary caller-defined effects.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Options are validated where they are set, not on every step. A
        # setting held in a tensor is a value the step reads, so comparing it
        # here would be a data-dependent branch inside the update -- which
        # anything capturing the step cannot resolve, and which turns a graph
        # it could have partitioned into one opaque task.
        # Each parameter's position is its stochastic rounding salt: the same
        # in every run over the same parameters, and its own stream of bits.
        for group in self.param_groups:
            for parameter, salt in zip(
                group["params"], group["rounding_salts"], strict=True
            ):
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
                    "lr": group["lr"],
                    "betas": tuple(group["betas"]),
                    "eps": group["eps"],
                    "weight_decay": group["weight_decay"],
                    "maximize": bool(group["maximize"]),
                    "parameter_rounding": group["parameter_rounding"],
                    "opt_state_rounding": group["opt_state_rounding"],
                    "rounding_salt": salt,
                }
                adamw_(
                    parameter,
                    update_gradient,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    **common,
                )

        return loss

    def load_state_dict(self, state_dict):
        """Restore state using the independently configured moment dtype."""
        # Optimizer.load_state_dict normally casts every floating state tensor
        # to the parameter dtype. Keep our separately typed state out of that
        # cast, and restore it before callers' post-hooks inspect the result.
        # The temporary hooks retain PyTorch's validation and caller pre-hooks.
        saved = {}

        def preserve(_optimizer, incoming):
            saved.update(incoming)
            parameter_keys = {
                key for group in incoming["param_groups"] for key in group["params"]
            }
            states = {
                key: (
                    {name: value for name, value in entries.items()
                     if name not in {"exp_avg", "exp_avg_sq", "step"}}
                    if key in parameter_keys else entries
                )
                for key, entries in incoming["state"].items()
            }
            return {**incoming, "state": states}

        def restore(_optimizer):
            for group, source in zip(self.param_groups, saved["param_groups"], strict=True):
                for parameter, key in zip(group["params"], source["params"], strict=True):
                    entries = saved["state"].get(key, {})
                    for name in ("exp_avg", "exp_avg_sq", "step"):
                        if name not in entries:
                            continue
                        dtype = (
                            torch.int64 if name == "step" else
                            resolve_dtype(group["opt_state_dtype"], parameter)
                        )
                        self.state[parameter][name] = entries[name].to(
                            device=parameter.device, dtype=dtype, copy=True
                        )

        pre = self.register_load_state_dict_pre_hook(preserve)
        post = self.register_load_state_dict_post_hook(restore, prepend=True)
        try:
            return super().load_state_dict(state_dict)
        finally:
            pre.remove()
            post.remove()


__all__ = ["AdamW"]
