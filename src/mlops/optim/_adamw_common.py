"""Shared AdamW option and dtype-policy normalization."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import torch


DTypePolicy: TypeAlias = torch.dtype | Literal["parameter"]
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
    if isinstance(group["lr"], torch.Tensor) or any(
        isinstance(beta, torch.Tensor) for beta in group["betas"]
    ):
        raise ValueError("mlops.optim.AdamW requires scalar lr and betas")
    if group["lr"] < 0 or group["eps"] < 0 or group["weight_decay"] < 0:
        raise ValueError("lr, eps, and weight_decay must be non-negative")
    if not 0 <= group["betas"][0] < 1 or not 0 <= group["betas"][1] < 1:
        raise ValueError("AdamW betas must be in [0, 1)")
    for name in (
        "gradient_dtype",
        "reduction_dtype",
        "state_dtype",
        "master_parameter_dtype",
    ):
        group[name] = normalize_dtype_policy(group[name], name=name)


__all__ = [
    "DTypePolicy",
    "normalize_dtype_policy",
    "resolve_dtype",
    "validate_adamw_options",
]
