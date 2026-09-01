"""Self-contained sparse-attention implementation for the DSA provider.

Selection indices are discrete inputs, work is isolated per packed sequence,
softmax reductions are fp32, and backward recomputes probabilities from the
saved log-sum-exp. A masked-flash/FlashMLA provider can replace this module
without changing the autograd or model-facing API.
"""

from __future__ import annotations

import torch

from .dsa_indexer import selection_mask, sequence_bounds


def sparse_attention_forward(q, k, v, indices, lengths):
    """Return ``(output, lse)`` for flat ``(tokens, heads, width)`` inputs."""
    tokens, heads, qk_dim = q.shape
    value_dim = v.shape[-1]
    scale = qk_dim**-0.5
    output = torch.empty((tokens, heads, value_dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((heads, tokens), dtype=torch.float32, device=q.device)
    for lo, hi in sequence_bounds(lengths):
        mask = selection_mask(indices, lo, hi, lo, hi)
        logits = (
            torch.einsum("thd,shd->hts", q[lo:hi].float(), k[lo:hi].float()) * scale
        )
        logits = logits + mask.unsqueeze(0)
        segment_lse = torch.logsumexp(logits, dim=-1)
        probability = torch.exp(logits - segment_lse.unsqueeze(-1))
        output[lo:hi] = torch.einsum("hts,shd->thd", probability, v[lo:hi].float()).to(
            output.dtype
        )
        lse[:, lo:hi] = segment_lse
    return output, lse


def sparse_attention_backward(grad_output, q, k, v, indices, lse, lengths):
    """Production-shaped deterministic recompute backward for sparse DSA."""
    scale = q.shape[-1] ** -0.5
    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    for lo, hi in sequence_bounds(lengths):
        mask = selection_mask(indices, lo, hi, lo, hi)
        query = q[lo:hi].float()
        key = k[lo:hi].float()
        value = v[lo:hi].float()
        grad = grad_output[lo:hi].float()
        logits = torch.einsum("thd,shd->hts", query, key) * scale
        probability = torch.exp(
            logits + mask.unsqueeze(0) - lse[:, lo:hi].unsqueeze(-1)
        )
        grad_probability = torch.einsum("thd,shd->hts", grad, value)
        centered = grad_probability - (grad_probability * probability).sum(
            -1, keepdim=True
        )
        grad_score = probability * centered * scale
        grad_q[lo:hi] = torch.einsum("hts,shd->thd", grad_score, key).to(grad_q.dtype)
        grad_k[lo:hi] = torch.einsum("hts,thd->shd", grad_score, query).to(grad_k.dtype)
        grad_v[lo:hi] = torch.einsum("hts,thd->shd", probability, grad).to(grad_v.dtype)
    return grad_q, grad_k, grad_v


__all__ = [
    "sparse_attention_backward",
    "sparse_attention_forward",
]
