"""Allocation-free mixed-dtype Triton AdamW kernels."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None
    libdevice = None


_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


if triton is not None:

    @triton.jit
    def _rounded(value, dtype: tl.constexpr, STOCHASTIC: tl.constexpr, bits):
        """``value`` at ``dtype``: to nearest, or stochastically.

        Stochastic rounding takes the representable neighbour on either side
        of the value -- toward zero, and one step away from it -- and picks the
        upper with probability equal to how far along the gap the value lies,
        so that in expectation nothing is lost: a small update to a weight or
        a moment kept at low precision accumulates instead of vanishing. It
        applies to a dtype narrower than FP32; FP32 itself is exact here. A
        value beyond the largest finite one has no finite neighbour above it,
        and is rounded to nearest, overflowing as it would there.
        """

        if STOCHASTIC and dtype.primitive_bitwidth == 16:
            down = value.to(dtype, fp_downcast_rounding="rtz")
            up = (down.to(tl.int16, bitcast=True) + 1).to(dtype, bitcast=True)
            low = down.to(tl.float32)
            high = up.to(tl.float32)
            fraction = (value - low) / (high - low)
            chosen = tl.where(tl.uint_to_uniform_float(bits) < fraction, up, down)
            return tl.where(tl.abs(high) == float("inf"), value.to(dtype), chosen)
        return value.to(dtype)

    @triton.jit
    def _adamw_torch_kernel(
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        out_parameter,
        out_master_parameter,
        out_exp_avg,
        out_exp_avg_sq,
        size,
        gradient_scale,
        lr,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        eps,
        weight_decay,
        decay_factor,
        salt,
        MAXIMIZE: tl.constexpr,
        PARAMETER_STOCHASTIC: tl.constexpr,
        STATE_STOCHASTIC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        gradient_f32 = tl.load(gradient + offsets, mask=mask, other=0).to(
            tl.float32
        )
        gradient_f32 *= gradient_scale
        if MAXIMIZE:
            gradient_f32 = -gradient_f32
        mean = tl.load(exp_avg + offsets, mask=mask, other=0).to(tl.float32)
        variance = tl.load(exp_avg_sq + offsets, mask=mask, other=0).to(
            tl.float32
        )
        master_f32 = tl.load(
            master_parameter + offsets, mask=mask, other=0
        ).to(tl.float32)
        if PARAMETER_STOCHASTIC or STATE_STOCHASTIC:
            # Counter-based bits, a function of the step, the tensor and the
            # element alone: every implementation of the update draws the same.
            seed = tl.load(step).to(tl.int64) * 1000003 + salt
            master_bits, mean_bits, variance_bits, _ = tl.randint4x(seed, offsets)
        else:
            master_bits = offsets
            mean_bits = offsets
            variance_bits = offsets

        # PyTorch's default AdamW is a sequence of in-place tensor primitives.
        # These casts reproduce the dtype-visible store between primitives --
        # rounded to nearest, which stochastic rounding replaces with the one
        # rounding of the stored result.
        master_f32 = master_f32 * decay_factor
        if not PARAMETER_STOCHASTIC:
            master_f32 = master_f32.to(out_master_parameter.dtype.element_ty).to(
                tl.float32
            )
        mean = libdevice.fma(one_minus_beta1, gradient_f32 - mean, mean)
        variance = variance * beta2
        if not STATE_STOCHASTIC:
            variance = variance.to(out_exp_avg_sq.dtype.element_ty).to(tl.float32)
        variance = libdevice.fma(
            one_minus_beta2,
            gradient_f32 * gradient_f32,
            variance,
        )
        rounded_mean = _rounded(
            mean, out_exp_avg.dtype.element_ty, STATE_STOCHASTIC, mean_bits
        )
        rounded_variance = _rounded(
            variance, out_exp_avg_sq.dtype.element_ty, STATE_STOCHASTIC, variance_bits
        )
        tl.store(out_exp_avg + offsets, rounded_mean, mask=mask)
        tl.store(out_exp_avg_sq + offsets, rounded_variance, mask=mask)

        next_step = tl.load(step).to(tl.float32) + 1.0
        correction1 = 1.0 - libdevice.pow(beta1, next_step)
        correction2 = 1.0 - libdevice.pow(beta2, next_step)
        step_size = lr / correction1
        correction2_sqrt = tl.sqrt(correction2)
        denominator = tl.sqrt(rounded_variance.to(tl.float32)).to(
            out_exp_avg_sq.dtype.element_ty
        ).to(tl.float32)
        denominator = (denominator / correction2_sqrt).to(
            out_exp_avg_sq.dtype.element_ty
        ).to(tl.float32)
        denominator = (denominator + eps).to(
            out_exp_avg_sq.dtype.element_ty
        ).to(tl.float32)
        master_f32 = libdevice.fma(
            -step_size,
            rounded_mean.to(tl.float32) / denominator,
            master_f32,
        )
        rounded_master = _rounded(
            master_f32,
            out_master_parameter.dtype.element_ty,
            PARAMETER_STOCHASTIC,
            master_bits,
        )
        tl.store(out_master_parameter + offsets, rounded_master, mask=mask)
        tl.store(
            out_parameter + offsets,
            rounded_master.to(out_parameter.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _adamw_internal_fp32_kernel(
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        out_parameter,
        out_master_parameter,
        out_exp_avg,
        out_exp_avg_sq,
        size,
        gradient_scale,
        lr,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        eps,
        weight_decay,
        decay_factor,
        salt,
        MAXIMIZE: tl.constexpr,
        PARAMETER_STOCHASTIC: tl.constexpr,
        STATE_STOCHASTIC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        gradient_f32 = tl.load(gradient + offsets, mask=mask, other=0).to(
            tl.float32
        )
        gradient_f32 *= gradient_scale
        if MAXIMIZE:
            gradient_f32 = -gradient_f32
        mean = tl.load(exp_avg + offsets, mask=mask, other=0).to(tl.float32)
        variance = tl.load(exp_avg_sq + offsets, mask=mask, other=0).to(
            tl.float32
        )
        master_f32 = tl.load(
            master_parameter + offsets, mask=mask, other=0
        ).to(tl.float32)
        if PARAMETER_STOCHASTIC or STATE_STOCHASTIC:
            # Counter-based bits, a function of the step, the tensor and the
            # element alone: every implementation of the update draws the same.
            seed = tl.load(step).to(tl.int64) * 1000003 + salt
            master_bits, mean_bits, variance_bits, _ = tl.randint4x(seed, offsets)
        else:
            master_bits = offsets
            mean_bits = offsets
            variance_bits = offsets

        mean = mean * beta1 + gradient_f32 * (1.0 - beta1)
        variance = variance * beta2 + gradient_f32 * gradient_f32 * (1.0 - beta2)
        rounded_mean = _rounded(
            mean, out_exp_avg.dtype.element_ty, STATE_STOCHASTIC, mean_bits
        )
        rounded_variance = _rounded(
            variance, out_exp_avg_sq.dtype.element_ty, STATE_STOCHASTIC, variance_bits
        )
        tl.store(out_exp_avg + offsets, rounded_mean, mask=mask)
        tl.store(out_exp_avg_sq + offsets, rounded_variance, mask=mask)

        next_step = tl.load(step).to(tl.float32) + 1.0
        correction1 = tl.where(
            next_step == 1.0,
            1.0 - beta1,
            1.0 - tl.exp(tl.log(beta1) * next_step),
        )
        correction2 = tl.where(
            next_step == 1.0,
            1.0 - beta2,
            1.0 - tl.exp(tl.log(beta2) * next_step),
        )
        corrected_mean = rounded_mean.to(tl.float32) / correction1
        corrected_variance = rounded_variance.to(tl.float32) / correction2
        master_f32 -= lr * (
            corrected_mean / (tl.sqrt(corrected_variance) + eps)
            + weight_decay * master_f32
        )
        rounded_master = _rounded(
            master_f32,
            out_master_parameter.dtype.element_ty,
            PARAMETER_STOCHASTIC,
            master_bits,
        )
        tl.store(out_master_parameter + offsets, rounded_master, mask=mask)
        tl.store(
            out_parameter + offsets,
            rounded_master.to(out_parameter.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _increment_scalar_kernel(source, destination):
        tl.store(destination, tl.load(source) + 1)


def _validate_state(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("mlops AdamW requires Triton")
    tensors = (parameter, master_parameter, gradient, exp_avg, exp_avg_sq)
    if any(value.device.type != "cuda" for value in (*tensors, step)):
        raise ValueError("mlops AdamW requires CUDA tensors")
    if any(value.dtype not in _FLOAT_DTYPES for value in tensors):
        raise ValueError("AdamW parameter, gradient, master, and moments must be floating")
    if step.dtype not in {torch.int64, torch.float32} or step.numel() != 1:
        raise ValueError("AdamW requires one int64 or FP32 step scalar")
    if not all(value.numel() == parameter.numel() for value in tensors):
        raise ValueError("AdamW tensor state must have equal element counts")
    if not all(value.is_contiguous() for value in tensors):
        raise ValueError("AdamW tensor state must be contiguous")


def _adamw_master_out_raw(
    parameter: torch.Tensor,
    master_parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    kernel,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool,
    parameter_stochastic: bool = False,
    opt_state_stochastic: bool = False,
    rounding_salt: int = 0,
    out: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_state(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
    )
    out_parameter, out_master, out_exp_avg, out_exp_avg_sq, out_step = out
    for source, destination in zip(
        (parameter, master_parameter, exp_avg, exp_avg_sq, step),
        out,
        strict=True,
    ):
        if (
            source.shape != destination.shape
            or source.stride() != destination.stride()
            or source.dtype != destination.dtype
            or source.device != destination.device
        ):
            raise ValueError("AdamW destination ABI differs from source")

    beta1, beta2 = betas
    block = 1024
    kernel[(triton.cdiv(parameter.numel(), block),)](
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        out_parameter,
        out_master,
        out_exp_avg,
        out_exp_avg_sq,
        parameter.numel(),
        float(gradient_scale),
        float(lr),
        float(beta1),
        float(beta2),
        float(1.0 - beta1),
        float(1.0 - beta2),
        float(eps),
        float(weight_decay),
        float(1.0 - lr * weight_decay),
        int(rounding_salt),
        MAXIMIZE=bool(maximize),
        PARAMETER_STOCHASTIC=bool(parameter_stochastic),
        STATE_STOCHASTIC=bool(opt_state_stochastic),
        BLOCK=block,
    )
    # Every main-grid block must read the old scalar before it is overwritten.
    _increment_scalar_kernel[(1,)](step, out_step)
    return out


def adamw_master_out_raw(
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
    maximize: bool,
    parameter_stochastic: bool = False,
    opt_state_stochastic: bool = False,
    rounding_salt: int = 0,
    out: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write one PyTorch-compatible mixed-dtype update."""
    return _adamw_master_out_raw(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        kernel=_adamw_torch_kernel,
        gradient_scale=gradient_scale,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        parameter_stochastic=parameter_stochastic,
        opt_state_stochastic=opt_state_stochastic,
        rounding_salt=rounding_salt,
        out=out,
    )


def adamw_master_out_internal_fp32_raw(
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
    maximize: bool,
    parameter_stochastic: bool = False,
    opt_state_stochastic: bool = False,
    rounding_salt: int = 0,
    out: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write one update using the original internal-FP32 arithmetic."""
    return _adamw_master_out_raw(
        parameter,
        master_parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        kernel=_adamw_internal_fp32_kernel,
        gradient_scale=gradient_scale,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        parameter_stochastic=parameter_stochastic,
        opt_state_stochastic=opt_state_stochastic,
        rounding_salt=rounding_salt,
        out=out,
    )


def _adamw_out_raw(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    *,
    master_update,
    gradient_scale: float = 1.0,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool,
    parameter_stochastic: bool = False,
    opt_state_stochastic: bool = False,
    rounding_salt: int = 0,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_parameter, out_exp_avg, out_exp_avg_sq, out_step = out
    master_update(
        parameter,
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        gradient_scale=gradient_scale,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        parameter_stochastic=parameter_stochastic,
        opt_state_stochastic=opt_state_stochastic,
        rounding_salt=rounding_salt,
        out=(
            out_parameter,
            out_parameter,
            out_exp_avg,
            out_exp_avg_sq,
            out_step,
        ),
    )
    return out


def adamw_out_raw(
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
    maximize: bool,
    parameter_stochastic: bool = False,
    opt_state_stochastic: bool = False,
    rounding_salt: int = 0,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write one PyTorch-compatible parameter-as-master update."""
    return _adamw_out_raw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        master_update=adamw_master_out_raw,
        gradient_scale=gradient_scale,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        parameter_stochastic=parameter_stochastic,
        opt_state_stochastic=opt_state_stochastic,
        rounding_salt=rounding_salt,
        out=out,
    )


def adamw_out_internal_fp32_raw(
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
    maximize: bool,
    parameter_stochastic: bool = False,
    opt_state_stochastic: bool = False,
    rounding_salt: int = 0,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write one parameter-as-master update with internal-FP32 arithmetic."""
    return _adamw_out_raw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        master_update=adamw_master_out_internal_fp32_raw,
        gradient_scale=gradient_scale,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        parameter_stochastic=parameter_stochastic,
        opt_state_stochastic=opt_state_stochastic,
        rounding_salt=rounding_salt,
        out=out,
    )


__all__ = [
    "adamw_master_out_internal_fp32_raw",
    "adamw_master_out_raw",
    "adamw_out_internal_fp32_raw",
    "adamw_out_raw",
]
