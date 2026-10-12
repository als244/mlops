"""Stateless mHC coefficients and residual mixing; ordinary differentiable math."""

import torch


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


def mhc_combine(branch, streams, post, mixing):
    dtype = streams.dtype
    return (
        post.to(dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        + mixing.to(dtype).transpose(-1, -2) @ streams
    )


__all__ = ["mhc_coefficients", "mhc_combine", "mhc_normalize_streams"]
