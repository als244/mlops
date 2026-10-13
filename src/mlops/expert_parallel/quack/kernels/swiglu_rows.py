"""SwiGLU over QuACK's interleaved pre-activation rows.

The gated GEMM epilogues pair adjacent accumulator columns, so a saved
pre-activation ``h [rows, 2I]`` holds each ``gate_j, up_j`` side by side.
``a = silu(gate) * up`` is recomputed from it in FP32 and rounded once, as the
epilogue rounded it: to ``h``'s dtype, or straight into E4M3 rows with their
descales for an FP8 GEMM, one pass instead of an activation and a
quantization. The backward writes the interleaved pre-activation gradient
the same way, from ``h`` and the activation's cotangent, without splitting
``h`` into gate and up copies; its routed form takes the cotangent before
the route score, and folds the score's gradient ``<r, a>``, the scaling and
the backward into one pass over ``h`` and ``r``. Widths are runtime
arguments; only the block size and the flags are compiled in.
"""

import torch
import triton
import triton.language as tl

from .quantize_rows import quantize_row, warps


@triton.jit
def _swiglu_rows(H, OUT, S, HALF, QS, QUANTIZE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < HALF
    pair = H + row * (2 * HALF) + 2 * col
    gate = tl.load(pair, mask, other=0).to(tl.float32)
    up = tl.load(pair + 1, mask, other=0).to(tl.float32)
    value = gate * tl.sigmoid(gate) * up
    if QUANTIZE:
        quantized, scale = quantize_row(value)
        tl.store(OUT + row * QS + col, quantized, mask)
        tl.store(S + row, scale)
    else:
        tl.store(OUT + row * QS + col, value.to(OUT.dtype.element_ty), mask)


@triton.jit
def _swiglu_rows_backward(H, G, OUT, HALF, BLOCK: tl.constexpr):
    # dgate = g * up * silu'(gate), dup = g * silu(gate), stored interleaved
    # like ``h``: the pre-activation gradient the up projection's backward
    # reads, in one pass over ``h`` and ``g`` without splitting ``h``.
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < HALF
    pair = H + row * (2 * HALF) + 2 * col
    gate = tl.load(pair, mask, other=0).to(tl.float32)
    up = tl.load(pair + 1, mask, other=0).to(tl.float32)
    g = tl.load(G + row * HALF + col, mask, other=0).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    dgate = g * up * (sig + silu * (1.0 - sig))
    dup = g * silu
    out = OUT + row * (2 * HALF) + 2 * col
    tl.store(out, dgate.to(OUT.dtype.element_ty), mask)
    tl.store(out + 1, dup.to(OUT.dtype.element_ty), mask)


@triton.jit
def _routed_swiglu_rows_backward(
    H, R, S, DS, OUT, AP, HALF, WRITE_AP: tl.constexpr, BLOCK: tl.constexpr
):
    # A routed row's output is ``score * y`` with ``y`` linear in ``a``, so
    # with ``r`` the cotangent of the unscaled row (``dy @ W``, any LoRA
    # branch included): ``dscore = <r, a>``, the activation's cotangent is
    # ``score * r``, and ``a' = score * a`` is the operand of the down
    # weight's and factors' gradients. One pass over ``h`` and ``r`` yields
    # all of them with the pre-activation gradient.
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < HALF
    pair = H + row * (2 * HALF) + 2 * col
    gate = tl.load(pair, mask, other=0).to(tl.float32)
    up = tl.load(pair + 1, mask, other=0).to(tl.float32)
    r = tl.load(R + row * HALF + col, mask, other=0).to(tl.float32)
    score = tl.load(S + row).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    activation = silu * up
    tl.store(DS + row, tl.sum(r * activation, axis=0))
    g = r * score
    dgate = g * up * (sig + silu * (1.0 - sig))
    dup = g * silu
    out = OUT + row * (2 * HALF) + 2 * col
    tl.store(out, dgate.to(OUT.dtype.element_ty), mask)
    tl.store(out + 1, dup.to(OUT.dtype.element_ty), mask)
    if WRITE_AP:
        tl.store(
            AP + row * HALF + col, (activation * score).to(AP.dtype.element_ty), mask
        )


def _check(h):
    if h.ndim != 2 or not h.is_cuda or not h.is_contiguous():
        raise ValueError("Expected a contiguous CUDA [rows, 2I] pre-activation")
    if h.shape[1] == 0 or h.shape[1] % 2:
        raise ValueError("An interleaved pre-activation has an even, nonzero width")


def _launch(h, out, scales, quantize):
    rows, doubled = h.shape
    if rows:
        block = triton.next_power_of_2(doubled // 2)
        _swiglu_rows[(rows,)](
            h,
            out,
            scales,
            doubled // 2,
            out.stride(0),
            quantize,
            block,
            num_warps=warps(block),
        )


def swiglu_rows(h):
    """``silu(gate) * up`` at ``h``'s dtype from the interleaved ``h [rows, 2I]``."""
    _check(h)
    out = torch.empty(h.shape[0], h.shape[1] // 2, dtype=h.dtype, device=h.device)
    _launch(h, out, out, False)
    return out


def swiglu_quantize_rows_fp8(h):
    """E4M3 rows and FP32 descales of ``silu(gate) * up`` from the interleaved
    ``h [rows, 2I]``, pitched as `quantize_rows_fp8` pitches its rows."""
    _check(h)
    rows, doubled = h.shape
    half = doubled // 2
    pitch = triton.cdiv(half, 16) * 16
    data = torch.empty((rows, pitch), dtype=torch.float8_e4m3fn, device=h.device)
    scales = torch.empty(rows, dtype=torch.float32, device=h.device)
    _launch(h, data, scales, True)
    return data[:, :half], scales


def swiglu_rows_backward(grad_a, h):
    """``dh [rows, 2I]``, interleaved like ``h``, from the activation's
    cotangent ``grad_a [rows, I]`` and the interleaved pre-activation."""
    _check(h)
    rows, doubled = h.shape
    half = doubled // 2
    if grad_a.shape != (rows, half) or not grad_a.is_cuda:
        raise ValueError("Expected the activation's cotangent, [rows, I] on CUDA")
    source = grad_a.contiguous()
    dh = torch.empty_like(h)
    if rows:
        block = triton.next_power_of_2(half)
        _swiglu_rows_backward[(rows,)](
            h, source, dh, half, block, num_warps=warps(block)
        )
    return dh


def routed_swiglu_rows_backward(r, h, score, *, scaled_activation=False):
    """``(dscore, dh, a')`` from the cotangent ``r [rows, I]`` of each routed
    row's unscaled output, the interleaved ``h [rows, 2I]`` and the rows'
    ``score [rows]``: the score gradient ``<r, a>`` in FP32, the interleaved
    pre-activation gradient of ``score * r``, and ``a' = score * a`` at
    ``h``'s dtype, or ``None`` unless asked for."""
    _check(h)
    rows, doubled = h.shape
    half = doubled // 2
    if r.shape != (rows, half) or not r.is_cuda:
        raise ValueError("Expected the rows' cotangent, [rows, I] on CUDA")
    if score.shape != (rows,) or not score.is_cuda:
        raise ValueError("Expected one score per row")
    source = r.contiguous()
    dscore = torch.empty(rows, dtype=torch.float32, device=h.device)
    dh = torch.empty_like(h)
    a_prime = (
        torch.empty(rows, half, dtype=h.dtype, device=h.device)
        if scaled_activation
        else dh
    )
    if rows:
        block = triton.next_power_of_2(half)
        _routed_swiglu_rows_backward[(rows,)](
            h,
            source,
            score,
            dscore,
            dh,
            a_prime,
            half,
            scaled_activation,
            block,
            num_warps=warps(block),
        )
    return dscore, dh, (a_prime if scaled_activation else None)
