"""GLM decay and output gates, expressed as compilable PyTorch arithmetic."""

import torch


def kda_decay(raw, dt_bias, a_log, *, lower_bound=-5.0):
    """Map [T,H,D] projected gates to FP32 log-decays in [lower_bound, 0]."""
    if raw.ndim != 3 or dt_bias.numel() != raw.shape[1] * raw.shape[2]:
        raise ValueError(
            "Expected [tokens, heads, width] gates and one bias per channel"
        )
    if a_log.shape != (raw.shape[1],) or lower_bound >= 0:
        raise ValueError(
            "Expected one decay parameter per head and a negative lower bound"
        )
    biased = raw.float() + dt_bias.float().reshape(1, raw.shape[1], raw.shape[2])
    return lower_bound * (a_log.float().exp()[None, :, None] * biased).sigmoid()


def gated_rms_norm(x, gate, weight, *, eps=1e-6):
    """FP32 RMS normalization and sigmoid gating, then one output cast."""
    if x.shape != gate.shape or weight.shape != (x.shape[-1],):
        raise ValueError(
            "Gate must match input; norm weight must match its final dimension"
        )
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return (normalized * weight.float() * gate.float().sigmoid()).to(x.dtype)
