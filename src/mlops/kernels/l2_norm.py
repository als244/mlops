"""Last-dimension L2 normalization used by linear attention."""

from __future__ import annotations

import torch


def l2_norm_forward(x, eps):
    x_float = x.float()
    rstd = torch.rsqrt(x_float.square().sum(-1) + float(eps))
    return (x_float * rstd.unsqueeze(-1)).to(x.dtype), rstd


def fla_l2_norm_forward(x, eps):
    from fla.modules.l2norm import l2norm_fwd

    if not x.is_contiguous() and 0 < x.shape[-1] <= 512:
        from .hybrid_layout import strided_l2_forward

        return strided_l2_forward(x, eps)
    return l2norm_fwd(x.contiguous(), float(eps))


def fla_l2_norm_backward(grad_output, output, rstd, eps):
    from fla.modules.l2norm import l2norm_bwd

    return l2norm_bwd(output, rstd, grad_output.contiguous(), float(eps))
