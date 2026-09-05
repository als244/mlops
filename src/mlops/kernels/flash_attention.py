"""Native varlen flash-attention forward/backward used by production.

The CUDA path calls PyTorch's low-level aten flash kernels directly, retaining
GQA and a single launch over all variable-length segments.  The fallback is a
plain per-segment SDPA implementation for CPU tests and unsupported dtypes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _block_causal_mask(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Build the packed causal mask without mutation.

    A slice-assignment implementation is mathematically straightforward but
    leaves ``aten.copy_`` in the pre-decomposition Export graph.  There is no
    persistent state mutation to model here, so construct the same mask as a
    pure expression.  This keeps the native-Torch correctness provider usable
    by strict Export/AOTAutograd as well as ordinary eager autograd.
    """
    total = sum(lengths)
    sequence_ids = torch.cat(
        [
            torch.full((length,), index, dtype=torch.int64, device=device)
            for index, length in enumerate(lengths)
        ]
    )
    positions = torch.arange(total, dtype=torch.int64, device=device)
    allowed = (sequence_ids[:, None] == sequence_ids[None, :]) & (
        positions[:, None] >= positions[None, :]
    )
    zeros = torch.zeros((total, total), device=device)
    blocked = torch.full((total, total), float("-inf"), device=device)
    return torch.where(allowed, zeros, blocked)


def native_torch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Match the raw models' single-call packed varlen SDPA expression."""
    lengths = tuple(int(length) for length in lengths)
    if not causal:
        raise ValueError("reference packed SDPA currently requires causal=True")
    repeat = q.shape[1] // k.shape[1]
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    value = v.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
    mask = _block_causal_mask(lengths, q.device).to(q.dtype)
    output = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        scale=softmax_scale,
    )
    return output.squeeze(0).transpose(0, 1)


def _native_torch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: tuple[int, ...],
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    outputs = []
    start = 0
    repeat = q.shape[1] // k.shape[1]
    for length in lengths:
        stop = start + int(length)
        q_segment = q[start:stop].transpose(0, 1).unsqueeze(0)
        k_segment = (
            k[start:stop].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
        )
        v_segment = (
            v[start:stop].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
        )
        output = functional.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
        start = stop
    return torch.cat(outputs, dim=0)


def native_flash_attention_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    return bool(
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    )


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    lengths: tuple[int, ...],
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Return ``(output, lse, used_native_kernel)`` for ``(tokens, H, D)``."""
    if native_flash_attention_supported(q, k, v):
        output, lse, _rng, _unused, _debug = torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            int(max_seqlen),
            int(max_seqlen),
            0.0,
            bool(causal),
            False,
            scale=softmax_scale,
        )
        return output, lse, True
    return (
        _native_torch_forward(q, k, v, lengths, causal, softmax_scale),
        None,
        False,
    )


def _deterministic_flash_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,
    softmax_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate dQ in a fixed order instead of with atomics.

    The aten primitive exposes no determinism control, and under FA3 its
    variable-length backward accumulates dQ across key blocks with atomics,
    so the sum order follows block completion and a rerun of one step can
    give different gradients.  FlashAttention-3's own entry point takes the
    flag that selects deterministic accumulation, and it accepts exactly the
    saved forward state the aten call is given, so no forward is recomputed.
    """
    try:
        import flash_attn_interface
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "deterministic variable-length flash attention needs "
            "flash_attn_interface (FlashAttention-3); install it or run "
            "without the determinism request"
        ) from error
    dq, dk, dv = (torch.empty_like(tensor) for tensor in (q, k, v))
    flash_attn_interface._flash_attn_backward(
        grad_output.contiguous(),
        q,
        k,
        v,
        output,
        lse.contiguous(),
        cu_seqlens,
        cu_seqlens,
        None,  # seqused_q: every query row in the range is present
        None,  # seqused_k
        max_seqlen,
        max_seqlen,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        deterministic=True,
    )
    return dq, dk, dv


def _deterministic_backward_available(device: torch.device) -> bool:
    """Whether this device can run FlashAttention-3's deterministic backward.

    FA3 ships Hopper kernels, so launching one anywhere else fails inside
    CUDA with "no kernel image is available for execution on the device",
    which says nothing about what was requested. Deciding here keeps the
    aten backward, which every supported device can run, as what a
    deterministic request falls back to.
    """

    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9


def flash_attention_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    lengths: tuple[int, ...],
    used_native: bool,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if used_native:
        if lse is None:  # pragma: no cover - guards an internal contract
            raise RuntimeError("native flash attention backward requires LSE")
        if deterministic and _deterministic_backward_available(q.device):
            return _deterministic_flash_backward(
                grad_output,
                q,
                k,
                v,
                output,
                lse,
                cu_seqlens,
                int(max_seqlen),
                bool(causal),
                softmax_scale,
            )
        philox = torch.zeros(2, dtype=torch.uint64, device=q.device)
        return torch.ops.aten._flash_attention_backward(
            grad_output.contiguous(),
            q,
            k,
            v,
            output,
            lse.contiguous(),
            cu_seqlens,
            cu_seqlens,
            int(max_seqlen),
            int(max_seqlen),
            0.0,
            bool(causal),
            philox,
            philox,
            scale=softmax_scale,
        )

    # A portable custom-backward fallback. Recompute is intentional: it
    # preserves the same saved-state contract as the native kernel.
    with torch.enable_grad():
        q_replay = q.detach().requires_grad_(True)
        k_replay = k.detach().requires_grad_(True)
        v_replay = v.detach().requires_grad_(True)
        replay = _native_torch_forward(
            q_replay,
            k_replay,
            v_replay,
            lengths,
            causal,
            softmax_scale,
        )
        grads = torch.autograd.grad(
            replay,
            (q_replay, k_replay, v_replay),
            grad_output,
            retain_graph=False,
            create_graph=False,
        )
    return grads


__all__ = [
    "flash_attention_backward",
    "flash_attention_forward",
    "native_flash_attention_supported",
    "native_torch_attention",
]
