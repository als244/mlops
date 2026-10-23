"""mHC preprocessing and postprocessing with provider-dispatched autograd.

The branch (attention/MLP) is outside both operations. Implementations may fuse
within either boundary; the native graph remains decomposable for recomputation.
"""

import torch

from .dispatch import resolve_implementation


def mhc_normalize_streams(streams, *, eps=1e-5):
    """FP32 input for the caller-owned mHC coefficient projection."""
    x = streams.flatten(-2).float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def mhc_coefficients(
    streams, projected, bias, scale, *, sinkhorn_eps=1e-6, iterations=20
):
    """Convert externally projected FP32 logits into mHC coefficients.

    The coefficients are computed in FP32 whatever the bias and scale dtypes.
    """
    if streams.ndim < 3:
        raise ValueError("Expected [..., streams, hidden] input")
    count = streams.shape[-2]
    width = count * (count + 2)
    if (
        projected.shape != (*streams.shape[:-2], width)
        or bias.shape != (width,)
        or scale.shape != (3,)
    ):
        raise ValueError(
            "mHC projected logits/parameters do not match the stream count"
        )
    if projected.dtype != torch.float32:
        raise ValueError("mHC coefficient projection must be computed in FP32")
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")
    pre, post, mixing = projected.split((count, count, count * count), dim=-1)
    pre_bias, post_bias, mixing_bias = bias.float().split((count, count, count * count))
    pre_scale, post_scale, mixing_scale = scale.float().unbind()
    pre = (pre * pre_scale + pre_bias).sigmoid() + sinkhorn_eps
    post = 2 * (post * post_scale + post_bias).sigmoid()
    mixing = (mixing * mixing_scale + mixing_bias).unflatten(-1, (count, count))
    mixing = mixing.softmax(-1) + sinkhorn_eps
    mixing = mixing / (mixing.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iterations - 1):
        mixing = mixing / (mixing.sum(-1, keepdim=True) + sinkhorn_eps)
        mixing = mixing / (mixing.sum(-2, keepdim=True) + sinkhorn_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(-2).to(streams.dtype)
    return post, mixing, collapsed


def mhc_pre(
    streams: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float = 1e-5,
    sinkhorn_eps: float = 1e-6,
    iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(post, mixing, branch_input)`` before attention or the MLP.

    ``streams`` has shape [..., n, d]; ``projection`` is [n*(n+2), n*d],
    ``bias`` is [n*(n+2)], and ``scale`` is [3]. Parameters retain their storage
    dtypes. Normalization, projection, gates and Sinkhorn compute in FP32;
    the collapsed branch input has the streams' dtype. The ordering matches
    ``mhc_coefficients``. All four tensor inputs are differentiable.

    This is a semantic boundary, not a requirement to materialize normalized
    streams or to launch a separate kernel for each Sinkhorn iteration.
    """
    impl = resolve_implementation(
        "mhc_pre",
        streams,
        projection,
        bias,
        scale,
        eps=eps,
        sinkhorn_eps=sinkhorn_eps,
        iterations=iterations,
        surface="semantic",
    )
    return impl.apply(
        streams,
        projection,
        bias,
        scale,
        eps=eps,
        sinkhorn_eps=sinkhorn_eps,
        iterations=iterations,
    )


def mhc_combine(
    branch: torch.Tensor,
    streams: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor,
    *,
    coefficient_precision: str = "fp32",
) -> torch.Tensor:
    """Expand the branch, mix residual streams, and add: the mHC post operation.

    ``mixing[..., i, j]`` sends input stream i to output stream j. Coefficients
    and accumulation stay FP32 by default, with one cast of the final result
    to the activation dtype. Optimized providers do not materialize FP32
    activation copies. Use ``coefficient_precision="activation"`` for the
    checkpoint reference's reduced coefficients and intermediate rounding.
    All four inputs are differentiable. Sinkhorn belongs to ``mhc_pre``.
    """
    impl = resolve_implementation(
        "mhc_combine",
        branch,
        streams,
        post,
        mixing,
        surface="semantic",
        coefficient_precision=coefficient_precision,
    )
    return impl.apply(
        branch, streams, post, mixing, coefficient_precision=coefficient_precision
    )


__all__ = ["mhc_coefficients", "mhc_combine", "mhc_normalize_streams", "mhc_pre"]
