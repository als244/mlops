"""Deterministic embedding-gradient accumulation."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _segment_kernel(
        sorted_tokens_ptr,
        order_ptr,
        grad_output_ptr,
        grad_weight_ptr,
        token_count,
        width,
        grad_stride,
        weight_stride,
        SEARCH_ITERS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0)
        token = tl.load(sorted_tokens_ptr + index)
        if index > 0:
            if tl.load(sorted_tokens_ptr + index - 1) == token:
                return

        low = index
        high = token_count
        for _ in range(SEARCH_ITERS):
            middle = (low + high) // 2
            value = tl.load(sorted_tokens_ptr + tl.minimum(middle, token_count - 1))
            ended = (middle >= token_count) | (value != token)
            high = tl.where(ended, middle, high)
            low = tl.where(ended, low, middle)
        segment_length = high - index

        for offset in range(0, width, BLOCK):
            columns = offset + tl.arange(0, BLOCK)
            mask = columns < width
            accumulator = tl.zeros((BLOCK,), tl.float32)
            for segment_index in range(0, segment_length):
                source = tl.load(order_ptr + index + segment_index)
                accumulator += tl.load(
                    grad_output_ptr + source * grad_stride + columns,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            tl.store(
                grad_weight_ptr + token * weight_stride + columns,
                accumulator.to(grad_weight_ptr.dtype.element_ty),
                mask=mask,
            )


def embedding_backward(
    tokens: torch.Tensor,
    grad_output: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    """Produce a deterministic dense embedding-table gradient."""
    flat_tokens = tokens.reshape(-1)
    flat_grad = grad_output.reshape(flat_tokens.numel(), -1).contiguous()
    grad_weight = torch.zeros(
        (int(num_embeddings), flat_grad.shape[1]),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    if triton is not None and grad_output.is_cuda:
        order = torch.argsort(flat_tokens.long(), stable=True)
        sorted_tokens = flat_tokens.long()[order].contiguous()
        iterations = max(1, (flat_tokens.numel() - 1).bit_length() + 1)
        _segment_kernel[(flat_tokens.numel(),)](
            sorted_tokens,
            order,
            flat_grad,
            grad_weight,
            flat_tokens.numel(),
            flat_grad.shape[1],
            flat_grad.stride(0),
            grad_weight.stride(0),
            SEARCH_ITERS=iterations,
            BLOCK=1024,
            num_warps=4,
            num_stages=2,
        )
        return grad_weight

    # CPU index_add is deterministic. Accumulate in fp32 and round once,
    # matching the single-writer CUDA kernel.
    accumulator = torch.zeros(
        grad_weight.shape, dtype=torch.float32, device=grad_output.device
    )
    accumulator.index_add_(0, flat_tokens.long(), flat_grad.float())
    return accumulator.to(grad_output.dtype)
