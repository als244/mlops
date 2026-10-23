"""Decomposable references for the two sides of an mHC branch."""

import torch
from torch.nn import functional as F

from ...dispatch.registry import Implementation, SupportResult, register_implementation


def supports_pre(
    streams,
    projection,
    bias,
    scale,
    *,
    eps=1e-5,
    sinkhorn_eps=1e-6,
    iterations=20,
    surface,
    **_kwargs,
):
    if surface != "semantic":
        return SupportResult.no("use ordinary autograd for the compositional VJP")
    if streams.ndim < 3 or min(streams.shape[-2:]) < 1:
        return SupportResult.no("streams must have shape [..., n, d]")
    n, d = streams.shape[-2:]
    width = n * (n + 2)
    if (
        projection.shape != (width, n * d)
        or bias.shape != (width,)
        or scale.shape != (3,)
    ):
        return SupportResult.no("projection/bias/scale do not match stream dimensions")
    if any(
        t.device != streams.device or not t.is_floating_point()
        for t in (streams, projection, bias, scale)
    ):
        return SupportResult.no("all inputs must be floating tensors on one device")
    if (
        not isinstance(iterations, int)
        or iterations < 1
        or eps <= 0
        or sinkhorn_eps <= 0
    ):
        return SupportResult.no(
            "positive epsilons and an integer iteration count required"
        )
    return SupportResult.yes()


def pre(
    streams, projection, bias, scale, *, eps=1e-5, sinkhorn_eps=1e-6, iterations=20
):
    from ...hyper_connection import mhc_coefficients, mhc_normalize_streams

    # Suppress ambient autocast: coefficient computation is FP32 even when
    # the checkpoint's projection is stored in BF16.
    with torch.autocast(device_type=streams.device.type, enabled=False):
        normalized = mhc_normalize_streams(streams, eps=eps)
        projected = F.linear(normalized, projection.float())
        return mhc_coefficients(
            streams,
            projected,
            bias,
            scale,
            sinkhorn_eps=sinkhorn_eps,
            iterations=iterations,
        )


def supports_combine(
    branch, streams, post, mixing, *, surface, coefficient_precision="fp32", **_kwargs
):
    if surface != "semantic":
        return SupportResult.no("use ordinary autograd for the compositional VJP")
    if streams.ndim < 3 or min(streams.shape[-2:]) < 1:
        return SupportResult.no("streams must have shape [..., n, d]")
    if coefficient_precision not in ("fp32", "activation"):
        return SupportResult.no("coefficient_precision must be fp32 or activation")
    prefix, n, d = streams.shape[:-2], streams.shape[-2], streams.shape[-1]
    if (
        branch.shape != (*prefix, d)
        or post.shape != (*prefix, n)
        or mixing.shape != (*prefix, n, n)
    ):
        return SupportResult.no("branch and coefficient shapes must match streams")
    if branch.dtype != streams.dtype or any(
        t.device != streams.device or not t.is_floating_point()
        for t in (branch, streams, post, mixing)
    ):
        return SupportResult.no(
            "floating inputs on one device and matching activation dtypes required"
        )
    return SupportResult.yes()


def combine(branch, streams, post, mixing, *, coefficient_precision="fp32"):
    dtype = streams.dtype
    if coefficient_precision == "fp32":
        with torch.autocast(device_type=streams.device.type, enabled=False):
            residual = mixing.float().transpose(-1, -2) @ streams.float()
            return (
                post.float().unsqueeze(-1) * branch.float().unsqueeze(-2) + residual
            ).to(dtype)
    residual = mixing.to(dtype).transpose(-1, -2) @ streams
    return post.to(dtype).unsqueeze(-1) * branch.unsqueeze(-2) + residual


PRE = register_implementation(
    Implementation(
        operation="mhc_pre",
        implementation_id="native_torch.mhc_pre",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=supports_pre,
        apply=pre,
    )
)
COMBINE = register_implementation(
    Implementation(
        operation="mhc_combine",
        implementation_id="native_torch.mhc_combine",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=supports_combine,
        apply=combine,
    )
)
