"""A single row kernel for logprobs, objective values and logits gradients."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    triton = tl = libdevice = None


if triton is not None:

    @triton.jit
    def _kernel(
        logits_ptr,
        targets_ptr,
        valid_ptr,
        weights_ptr,
        old_ptr,
        advantages_ptr,
        reference_ptr,
        clip_ptr,
        beta_ptr,
        values_ptr,
        losses_ptr,
        grad_logits_ptr,
        vocab,
        POLICY: tl.constexpr,
        REFERENCE: tl.constexpr,
        GRADIENT: tl.constexpr,
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
                base + offset + columns, mask=mask, other=-float("inf")
            ).to(tl.float32)
            new_maximum = tl.maximum(maximum, tl.max(logits))
            exponential_sum = exponential_sum * tl.exp(maximum - new_maximum)
            exponential_sum += tl.sum(tl.where(mask, tl.exp(logits - new_maximum), 0.0))
            maximum = new_maximum
        logsumexp = maximum + tl.log(exponential_sum)
        target = tl.load(targets_ptr + row).to(tl.int64)
        label_valid = (target >= 0) & (target < vocab)
        safe_target = tl.where(label_valid, target, 0)
        target_logit = tl.load(base + safe_target).to(tl.float32)
        logprob = tl.where(label_valid, target_logit - logsumexp, 0.0)
        tl.store(values_ptr + row, logprob)

        valid = tl.load(valid_ptr + row)
        current = tl.where(valid, logprob, 0.0)
        weight = tl.load(weights_ptr + row, mask=valid, other=0.0).to(tl.float32)
        if POLICY:
            old = tl.load(old_ptr + row, mask=valid, other=0.0).to(tl.float32)
            advantage = tl.load(advantages_ptr + row, mask=valid, other=0.0).to(
                tl.float32
            )
            clip = tl.load(clip_ptr).to(tl.float32)
            upper, lower = libdevice.log1p(clip), libdevice.log1p(-clip)
            difference = current - old
            clipped = tl.where(advantage >= 0, difference > upper, difference < lower)
            bound = tl.where(advantage >= 0, upper, lower)
            term = -tl.exp(tl.where(clipped, bound, difference)) * advantage
            derivative = tl.where(clipped, 0.0, term)
            if REFERENCE:
                reference = tl.load(reference_ptr + row, mask=valid, other=0.0).to(
                    tl.float32
                )
                beta = tl.load(beta_ptr).to(tl.float32)
                delta = tl.where(beta != 0, reference - current, 0.0)
                penalty = libdevice.expm1(delta)
                term += beta * (penalty - delta)
                derivative -= beta * penalty
            objective = weight * term
            derivative = weight * derivative
        else:
            objective = -weight * current
            derivative = -weight
        tl.store(losses_ptr + row, tl.where(valid, objective, 0.0))

        if GRADIENT:
            for offset in range(0, vocab, BLOCK):
                mask = offset + columns < vocab
                logits = tl.load(base + offset + columns, mask=mask, other=0.0).to(
                    tl.float32
                )
                probability = tl.exp(logits - logsumexp)
                one_hot = ((offset + columns).to(tl.int64) == safe_target).to(
                    tl.float32
                )
                gradient = tl.where(
                    valid & label_valid, (one_hot - probability) * derivative, 0.0
                )
                tl.store(
                    grad_logits_ptr + row * vocab + offset + columns,
                    gradient.to(grad_logits_ptr.dtype.element_ty),
                    mask=mask,
                )


def fused_head_objective(
    logits, targets, inputs, controls, *, objective, need_gradient, logprobs, losses
):
    """Write row results and return optional logits gradients without row temporaries.

    All formula data, including clip ratio and beta, are device tensors. Only
    formula choice, geometry and trainability specialize the kernel.
    """
    if triton is None or not logits.is_cuda:
        raise RuntimeError("fused head objectives require Triton and CUDA")
    rows, vocab = logits.shape
    valid, weights = (value.contiguous() for value in inputs[:2])
    policy = objective == "clipped_policy"
    reference = policy and len(inputs) == 5
    old, advantages = (
        (value.contiguous() for value in inputs[2:4]) if policy else (None, None)
    )
    reference_values = inputs[4].contiguous() if reference else None
    gradient = torch.empty_like(logits) if need_gradient else None
    if rows:
        _kernel[(rows,)](
            logits,
            targets.contiguous(),
            valid,
            weights,
            old,
            advantages,
            reference_values,
            controls[0] if policy else None,
            controls[1] if reference else None,
            logprobs,
            losses,
            gradient,
            vocab,
            POLICY=policy,
            REFERENCE=reference,
            GRADIENT=need_gradient,
            BLOCK=4096,
        )
    return gradient
