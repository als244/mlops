"""Builtin mixed-dtype AdamW implementation and registered mutations."""

from __future__ import annotations

import torch

from ...dispatch.costs import CostHints
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.adamw import adamw_master_out_raw, adamw_out_raw


_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def _settings(
    gradient_scale,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    maximize,
):
    return {
        "gradient_scale": float(gradient_scale),
        "lr": float(lr),
        "betas": (float(beta1), float(beta2)),
        "eps": float(eps),
        "weight_decay": float(weight_decay),
        "maximize": bool(maximize),
    }


@torch.library.custom_op("mlops::adamw", mutates_args=())
def _functional_op(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    gradient_scale: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = tuple(
        torch.empty_like(value) for value in (parameter, exp_avg, exp_avg_sq, step)
    )
    return adamw_out_raw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **_settings(
            gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
        ),
        out=outputs,
    )


@_functional_op.register_fake
def _functional_fake(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    gradient_scale,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    maximize,
):
    del gradient, gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
    return tuple(
        torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device=value.device,
        )
        for value in (parameter, exp_avg, exp_avg_sq, step)
    )


@torch.library.custom_op(
    "mlops::adamw_out",
    mutates_args=("out_parameter", "out_exp_avg", "out_exp_avg_sq", "out_step"),
)
def _out_op(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    gradient_scale: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    out_parameter: torch.Tensor,
    out_exp_avg: torch.Tensor,
    out_exp_avg_sq: torch.Tensor,
    out_step: torch.Tensor,
) -> None:
    adamw_out_raw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **_settings(
            gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
        ),
        out=(out_parameter, out_exp_avg, out_exp_avg_sq, out_step),
    )


@_out_op.register_fake
def _out_fake(*args, **kwargs):
    del args, kwargs


@torch.library.custom_op(
    "mlops::adamw_",
    mutates_args=("parameter", "exp_avg", "exp_avg_sq", "step"),
)
def _in_place_op(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    gradient_scale: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    adamw_out_raw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **_settings(
            gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
        ),
        out=(parameter, exp_avg, exp_avg_sq, step),
    )


@_in_place_op.register_fake
def _in_place_fake(*args, **kwargs):
    del args, kwargs


@torch.library.custom_op("mlops::master_adamw", mutates_args=())
def _master_functional_op(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    gradient_scale: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = tuple(
        torch.empty_like(value)
        for value in (parameter, master_parameter, exp_avg, exp_avg_sq, step)
    )
    return adamw_master_out_raw(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **_settings(
            gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
        ),
        out=outputs,
    )


@_master_functional_op.register_fake
def _master_functional_fake(
    parameter,
    master_parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    gradient_scale,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    maximize,
):
    del gradient, gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
    return tuple(
        torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device=value.device,
        )
        for value in (parameter, master_parameter, exp_avg, exp_avg_sq, step)
    )


@torch.library.custom_op(
    "mlops::master_adamw_out",
    mutates_args=(
        "out_parameter",
        "out_master_parameter",
        "out_exp_avg",
        "out_exp_avg_sq",
        "out_step",
    ),
)
def _master_out_op(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    gradient_scale: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    out_parameter: torch.Tensor,
    out_master_parameter: torch.Tensor,
    out_exp_avg: torch.Tensor,
    out_exp_avg_sq: torch.Tensor,
    out_step: torch.Tensor,
) -> None:
    adamw_master_out_raw(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **_settings(
            gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
        ),
        out=(
            out_parameter,
            out_master_parameter,
            out_exp_avg,
            out_exp_avg_sq,
            out_step,
        ),
    )


@_master_out_op.register_fake
def _master_out_fake(*args, **kwargs):
    del args, kwargs


@torch.library.custom_op(
    "mlops::master_adamw_",
    mutates_args=(
        "parameter",
        "master_parameter",
        "exp_avg",
        "exp_avg_sq",
        "step",
    ),
)
def _master_in_place_op(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    gradient_scale: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    adamw_master_out_raw(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **_settings(
            gradient_scale, lr, beta1, beta2, eps, weight_decay, maximize
        ),
        out=(parameter, master_parameter, exp_avg, exp_avg_sq, step),
    )


@_master_in_place_op.register_fake
def _master_in_place_fake(*args, **kwargs):
    del args, kwargs


def _scalar_arguments(
    *,
    gradient_scale: float,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> tuple[float, float, float, float, float, float, bool]:
    return (
        float(gradient_scale),
        float(lr),
        float(betas[0]),
        float(betas[1]),
        float(eps),
        float(weight_decay),
        bool(maximize),
    )


def functional_adamw(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a new mixed-dtype AdamW state version."""
    return _functional_op(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        *_scalar_arguments(
            gradient_scale=gradient_scale,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
    )


def adamw(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool = False,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write one update to caller-owned storage-disjoint outputs."""
    _validate_disjoint(
        (parameter, exp_avg, exp_avg_sq, step),
        out,
    )
    _out_op(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        *_scalar_arguments(
            gradient_scale=gradient_scale,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
        *out,
    )
    return out


def adamw_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update a parameter-as-master and its optimizer state in place."""
    _in_place_op(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        *_scalar_arguments(
            gradient_scale=gradient_scale,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
    )
    return parameter, exp_avg, exp_avg_sq, step


def functional_master_adamw(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a new model/master/moment/step state version."""
    return _master_functional_op(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        *_scalar_arguments(
            gradient_scale=gradient_scale,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
    )


def master_adamw(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool = False,
    out: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Write one distinct-master update to caller-owned outputs."""
    _validate_disjoint(
        (parameter, master_parameter, exp_avg, exp_avg_sq, step),
        out,
    )
    _master_out_op(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        *_scalar_arguments(
            gradient_scale=gradient_scale,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
        *out,
    )
    return out


def master_adamw_(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update distinct model/master state in place."""
    _master_in_place_op(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        *_scalar_arguments(
            gradient_scale=gradient_scale,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
    )
    return parameter, master_parameter, exp_avg, exp_avg_sq, step


def _storage_key(value: torch.Tensor) -> tuple[str, int]:
    return str(value.device), value.untyped_storage().data_ptr()


def _validate_disjoint(
    sources: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
) -> None:
    source_keys = {_storage_key(value) for value in sources}
    output_keys = []
    for value in outputs:
        key = _storage_key(value)
        if key in source_keys or key in output_keys:
            raise ValueError("adamw out= tensors must be storage-disjoint")
        output_keys.append(key)


def _supports(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    surface,
    **_kwargs,
) -> SupportResult:
    del surface
    values = (parameter, gradient, exp_avg, exp_avg_sq)
    if not all(isinstance(value, torch.Tensor) for value in (*values, step)):
        return SupportResult.no("state and gradient must be tensors")
    if any(value.device.type != "cuda" for value in (*values, step)):
        return SupportResult.no("requires CUDA state and gradient tensors")
    if any(value.dtype not in _FLOAT_DTYPES for value in values):
        return SupportResult.no("requires BF16, FP16, or FP32 floating tensors")
    if step.dtype not in {torch.int64, torch.float32} or step.numel() != 1:
        return SupportResult.no("step must be one int64 or FP32 scalar")
    if not all(value.numel() == parameter.numel() and value.is_contiguous() for value in values):
        return SupportResult.no("tensor state must have equal contiguous geometry")
    return SupportResult.yes()


def _estimate(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    entrypoint,
    **_kwargs,
) -> CostHints:
    if entrypoint != "forward":
        return CostHints(notes=("optimizer update has no differentiable VJP",))
    read_bytes = sum(
        value.numel() * value.element_size()
        for value in (parameter, gradient, exp_avg, exp_avg_sq)
    )
    write_bytes = sum(
        value.numel() * value.element_size()
        for value in (parameter, exp_avg, exp_avg_sq)
    )
    return CostHints(
        implementation_bytes_accessed=read_bytes + write_bytes + 2 * step.element_size(),
        workspace_bytes=0,
        notes=(
            "traffic reflects each admitted tensor's actual dtype",
            "the separate scalar-step launch remains allocation-free",
        ),
    )


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="adamw",
        implementation_id="builtin.adamw.triton",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=functional_adamw,
        estimate=_estimate,
    )
)


__all__ = [
    "IMPLEMENTATION",
    "functional_adamw",
    "adamw",
    "adamw_",
    "functional_master_adamw",
    "master_adamw",
    "master_adamw_",
]
