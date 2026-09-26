"""Shared AdamW option and dtype-policy normalization."""

from __future__ import annotations

from typing import Any, Literal

import torch

type DTypePolicy = torch.dtype | Literal["parameter"]
_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def normalize_dtype_policy(value: Any, *, name: str) -> DTypePolicy:
    if value == "parameter":
        return "parameter"
    if isinstance(value, torch.dtype) and value in _FLOAT_DTYPES:
        return value
    raise ValueError(
        f"{name} must be 'parameter', torch.bfloat16, torch.float16, or "
        "torch.float32"
    )


def resolve_dtype(policy: DTypePolicy, parameter: torch.Tensor) -> torch.dtype:
    return parameter.dtype if policy == "parameter" else policy


def validate_adamw_options(group: dict[str, Any]) -> None:
    if bool(group.get("amsgrad", False)):
        raise ValueError("mlops.optim.AdamW does not support amsgrad=True")
    if bool(group.get("differentiable", False)):
        raise ValueError("mlops.optim.AdamW does not support differentiable=True")
    for name, value in (
        ("lr", group["lr"]),
        ("eps", group["eps"]),
        ("weight_decay", group["weight_decay"]),
        *(("betas", beta) for beta in group["betas"]),
    ):
        if isinstance(value, torch.Tensor) and (
            value.numel() != 1 or value.device.type != "cpu"
        ):
            raise ValueError(
                f"a tensor {name} must be one element on the host: it is read "
                "when the kernel is launched, and reading it there costs "
                "nothing only if it is already where the launch happens"
            )
    if float(group["lr"]) < 0 or float(group["eps"]) < 0:
        raise ValueError("lr, eps, and weight_decay must be non-negative")
    if float(group["weight_decay"]) < 0:
        raise ValueError("lr, eps, and weight_decay must be non-negative")
    if not 0 <= float(group["betas"][0]) < 1 or not 0 <= float(group["betas"][1]) < 1:
        raise ValueError("AdamW betas must be in [0, 1)")
    for name in (
        "gradient_dtype",
        "reduction_dtype",
        "state_dtype",
        "master_parameter_dtype",
    ):
        group[name] = normalize_dtype_policy(group[name], name=name)
    for name in ("parameter_rounding", "state_rounding"):
        if group.get(name, "nearest") not in {"nearest", "stochastic"}:
            raise ValueError(f"{name} must be 'nearest' or 'stochastic'")


#: The group settings the update reads, which a caller may hold in a tensor.
SETTING_NAMES = ("lr", "betas", "eps", "weight_decay")


def hold_settings_on_host(values: dict[str, Any]) -> None:
    """Hold this group's settings in host scalars, in place.

    Held once, where the optimizer is built, rather than each step. A tensor
    made inside ``step`` would be made inside anything capturing that step
    too -- an operation every parameter's update depends on, which reads as a
    dependency between parameters that share nothing, and collapses a per-stage
    update into a single task.

    On the host because that is where the value is read: the update passes it
    to the kernel as a launch argument, so nothing is copied to the device and
    nothing is synchronized, and a caller writing the next step's value writes
    host memory.
    """

    for name in SETTING_NAMES:
        value = values.get(name)
        if isinstance(value, tuple | list):
            values[name] = type(value)(
                item
                if isinstance(item, torch.Tensor)
                else torch.tensor(float(item), dtype=torch.float64)
                for item in value
            )
        elif value is not None and not isinstance(value, torch.Tensor):
            values[name] = torch.tensor(float(value), dtype=torch.float64)


__all__ = [
    "SETTING_NAMES",
    "DTypePolicy",
    "hold_settings_on_host",
    "normalize_dtype_policy",
    "resolve_dtype",
    "validate_adamw_options",
]
