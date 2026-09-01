"""Fused cross-entropy loss and logits gradient from the production path."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_BLOCK = 4096
_NATIVE_TORCH_CHUNK_ROWS = 1024


if triton is not None:

    @triton.jit
    def _standalone_cross_entropy_forward_kernel(
        logits_ptr,
        targets_ptr,
        losses_ptr,
        logsumexp_ptr,
        vocab,
        ignore_index,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        base = logits_ptr + row * vocab
        maximum = -float("inf")
        exponential_sum = 0.0
        for offset in range(0, vocab, BLOCK):
            mask = offset + columns < vocab
            logits = tl.load(
                base + offset + columns,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            block_maximum = tl.max(logits)
            new_maximum = tl.maximum(maximum, block_maximum)
            exponential_sum = exponential_sum * tl.exp(maximum - new_maximum)
            exponential_sum += tl.sum(
                tl.where(mask, tl.exp(logits - new_maximum), 0.0)
            )
            maximum = new_maximum
        logsumexp = maximum + tl.log(exponential_sum)
        target = tl.load(targets_ptr + row).to(tl.int64)
        in_range = (target >= 0) & (target < vocab)
        valid = (target != ignore_index) & in_range
        safe_target = tl.where(valid, target, 0)
        target_logit = tl.load(base + safe_target).to(tl.float32)
        tl.store(losses_ptr + row, tl.where(valid, logsumexp - target_logit, 0.0))
        tl.store(logsumexp_ptr + row, logsumexp)

    @triton.jit
    def _standalone_cross_entropy_backward_kernel(
        grad_losses_ptr,
        logits_ptr,
        targets_ptr,
        logsumexp_ptr,
        grad_logits_ptr,
        vocab,
        ignore_index,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        base = logits_ptr + row * vocab
        target = tl.load(targets_ptr + row).to(tl.int64)
        in_range = (target >= 0) & (target < vocab)
        valid = (target != ignore_index) & in_range
        safe_target = tl.where(valid, target, 0)
        logsumexp = tl.load(logsumexp_ptr + row).to(tl.float32)
        grad_loss = tl.load(grad_losses_ptr + row).to(tl.float32)
        for offset in range(0, vocab, BLOCK):
            mask = offset + columns < vocab
            logits = tl.load(
                base + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            probability = tl.exp(logits - logsumexp)
            one_hot = ((offset + columns).to(tl.int64) == safe_target).to(
                tl.float32
            )
            gradient = tl.where(
                valid,
                (probability - one_hot) * grad_loss,
                0.0,
            )
            tl.store(
                grad_logits_ptr + row * vocab + offset + columns,
                gradient.to(grad_logits_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def _cross_entropy_kernel(
        logits_ptr,
        targets_ptr,
        grad_logits_ptr,
        nll_ptr,
        total_rows,
        vocab,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        base = logits_ptr + row * vocab
        maximum = -float("inf")
        exponential_sum = 0.0
        for offset in range(0, vocab, BLOCK):
            mask = offset + columns < vocab
            logits = tl.load(
                base + offset + columns,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            block_maximum = tl.max(logits)
            new_maximum = tl.maximum(maximum, block_maximum)
            exponential_sum = exponential_sum * tl.exp(maximum - new_maximum)
            exponential_sum += tl.sum(
                tl.where(mask, tl.exp(logits - new_maximum), 0.0)
            )
            maximum = new_maximum
        logsumexp = maximum + tl.log(exponential_sum)
        target = tl.load(targets_ptr + row).to(tl.int64)
        valid = target >= 0
        safe_target = tl.where(valid, target, 0)
        target_logit = tl.load(base + safe_target).to(tl.float32)
        tl.store(nll_ptr + row, tl.where(valid, logsumexp - target_logit, 0.0))
        inverse_rows = 1.0 / total_rows
        for offset in range(0, vocab, BLOCK):
            mask = offset + columns < vocab
            logits = tl.load(
                base + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            probability = tl.exp(logits - logsumexp)
            one_hot = ((offset + columns).to(tl.int64) == safe_target).to(
                tl.float32
            )
            grad = tl.where(valid, (probability - one_hot) * inverse_rows, 0.0)
            tl.store(
                grad_logits_ptr + row * vocab + offset + columns,
                grad.to(grad_logits_ptr.dtype.element_ty),
                mask=mask,
            )


def standalone_cross_entropy_available() -> bool:
    """Whether the package-owned standalone Triton kernels can be launched."""
    return triton is not None


def cross_entropy_forward_triton(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row FP32 ``(losses, logsumexp)`` without autograd state."""
    if triton is None or not logits.is_cuda:
        raise RuntimeError("standalone cross entropy requires Triton and CUDA")
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected logits [rows, vocab] and targets [rows]")
    rows, vocab = logits.shape
    if targets.shape[0] != rows:
        raise ValueError("target row count must match logits")
    logits = logits.contiguous()
    targets = targets.contiguous()
    losses = torch.empty(rows, dtype=torch.float32, device=logits.device)
    logsumexp = torch.empty_like(losses)
    if rows:
        _standalone_cross_entropy_forward_kernel[(rows,)](
            logits,
            targets,
            losses,
            logsumexp,
            vocab,
            int(ignore_index),
            BLOCK=_BLOCK,
        )
    return losses, logsumexp


def cross_entropy_backward_triton(
    grad_losses: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    logsumexp: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return a logits VJP for arbitrary per-row loss cotangents."""
    if triton is None or not logits.is_cuda:
        raise RuntimeError("standalone cross entropy requires Triton and CUDA")
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected logits [rows, vocab] and targets [rows]")
    rows, vocab = logits.shape
    if (
        targets.shape != (rows,)
        or grad_losses.shape != (rows,)
        or logsumexp.shape != (rows,)
    ):
        raise ValueError("targets, grad_losses, and logsumexp must have shape [rows]")
    logits = logits.contiguous()
    targets = targets.contiguous()
    grad_losses = grad_losses.contiguous()
    logsumexp = logsumexp.contiguous()
    grad_logits = torch.empty_like(logits)
    if rows:
        _standalone_cross_entropy_backward_kernel[(rows,)](
            grad_losses,
            logits,
            targets,
            logsumexp,
            grad_logits,
            vocab,
            int(ignore_index),
            BLOCK=_BLOCK,
        )
    return grad_logits


def cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    total_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an fp32 partial loss and storage-dtype logits gradient."""
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got {tuple(logits.shape)}")
    rows, vocab = logits.shape
    total = rows if total_rows is None else int(total_rows)
    grad_logits = torch.empty_like(logits)
    if triton is not None and logits.is_cuda and logits.is_contiguous():
        nll = torch.empty(rows, dtype=torch.float32, device=logits.device)
        _cross_entropy_kernel[(rows,)](
            logits,
            targets.contiguous().int(),
            grad_logits,
            nll,
            total,
            vocab,
            BLOCK=_BLOCK,
        )
        return nll.sum() / total, grad_logits

    loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
    targets = targets.long()
    for start in range(0, rows, _NATIVE_TORCH_CHUNK_ROWS):
        stop = min(start + _NATIVE_TORCH_CHUNK_ROWS, rows)
        logits_float = logits[start:stop].float()
        logsumexp = torch.logsumexp(logits_float, dim=-1, keepdim=True)
        target = targets[start:stop]
        valid = target >= 0
        safe_target = target.clamp_min(0)
        nll = logsumexp.squeeze(-1) - logits_float.gather(
            1, safe_target.unsqueeze(1)
        ).squeeze(1)
        loss_sum += (nll * valid.float()).sum()
        probability = torch.exp(logits_float - logsumexp)
        probability.scatter_add_(
            1,
            safe_target.unsqueeze(1),
            torch.full(
                (stop - start, 1),
                -1.0,
                device=logits.device,
                dtype=torch.float32,
            ),
        )
        probability *= valid.unsqueeze(1)
        grad_logits[start:stop].copy_((probability / total).to(logits.dtype))
    return loss_sum / total, grad_logits
