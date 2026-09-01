"""Fused SwiGLU forward/backward, matching the production rounding order."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None


_BLOCK = 1024


if triton is not None:

    @triton.jit
    def _forward_kernel(gate_ptr, up_ptr, output_ptr, size, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        silu = gate * tl.sigmoid(gate)
        silu = silu.to(output_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            output_ptr + offsets,
            (silu * up).to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _backward_kernel(
        grad_ptr,
        gate_ptr,
        up_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        size,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # Match PyTorch CUDA: accurate expf plus round-to-nearest division.
        sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        silu = gate * sigmoid
        grad_silu = grad * up
        silu = silu.to(grad_up_ptr.dtype.element_ty).to(tl.float32)
        grad_silu = grad_silu.to(grad_gate_ptr.dtype.element_ty).to(tl.float32)
        # Preserve PyTorch CUDA's left-associated SiLU derivative:
        #   (dy * sigmoid) * (1 + x * (1 - sigmoid))
        grad_gate = (grad_silu * sigmoid) * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad * silu
        tl.store(
            grad_gate_ptr + offsets,
            grad_gate.to(grad_gate_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            grad_up_ptr + offsets,
            grad_up.to(grad_up_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _legacy_backward_kernel(
        grad_ptr,
        gate_ptr,
        up_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        size,
        REFERENCE_CASTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Historical pre-fix kernel, retained only for ablation studies."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        silu = gate * sigmoid
        grad_silu = grad * up
        if REFERENCE_CASTS:
            silu = silu.to(grad_up_ptr.dtype.element_ty).to(tl.float32)
            grad_silu = grad_silu.to(grad_gate_ptr.dtype.element_ty).to(tl.float32)
        grad_gate = grad_silu * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
        grad_up = grad * silu
        tl.store(
            grad_gate_ptr + offsets,
            grad_gate.to(grad_gate_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            grad_up_ptr + offsets,
            grad_up.to(grad_up_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _packed_forward_kernel(
        packed_ptr, output_ptr, rows, width, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1).to(tl.int64)
        columns = block * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = columns < width
        base = packed_ptr + row * (2 * width)
        gate = tl.load(base + columns, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + width + columns, mask=mask, other=0.0).to(tl.float32)
        silu = (gate * tl.sigmoid(gate)).to(output_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            output_ptr + row * width + columns,
            (silu * up).to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _packed_backward_kernel(
        grad_ptr,
        packed_ptr,
        grad_packed_ptr,
        rows,
        width,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1).to(tl.int64)
        columns = block * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = columns < width
        base = packed_ptr + row * (2 * width)
        gate = tl.load(base + columns, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + width + columns, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + row * width + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        silu = gate * sigmoid
        grad_silu = grad * up
        silu = silu.to(grad_packed_ptr.dtype.element_ty).to(tl.float32)
        grad_silu = grad_silu.to(grad_packed_ptr.dtype.element_ty).to(tl.float32)
        grad_gate = (grad_silu * sigmoid) * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad * silu
        grad_base = grad_packed_ptr + row * (2 * width)
        tl.store(
            grad_base + columns,
            grad_gate.to(grad_packed_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            grad_base + width + columns,
            grad_up.to(grad_packed_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _legacy_packed_backward_kernel(
        grad_ptr,
        packed_ptr,
        grad_packed_ptr,
        rows,
        width,
        REFERENCE_CASTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Packed form of the historical pre-fix backward kernel."""
        row = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1).to(tl.int64)
        columns = block * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = columns < width
        base = packed_ptr + row * (2 * width)
        gate = tl.load(base + columns, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + width + columns, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + row * width + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        sigmoid = tl.sigmoid(gate)
        silu = gate * sigmoid
        grad_silu = grad * up
        if REFERENCE_CASTS:
            silu = silu.to(grad_packed_ptr.dtype.element_ty).to(tl.float32)
            grad_silu = grad_silu.to(grad_packed_ptr.dtype.element_ty).to(tl.float32)
        grad_gate = grad_silu * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
        grad_up = grad * silu
        grad_base = grad_packed_ptr + row * (2 * width)
        tl.store(
            grad_base + columns,
            grad_gate.to(grad_packed_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            grad_base + width + columns,
            grad_up.to(grad_packed_ptr.dtype.element_ty),
            mask=mask,
        )


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    return bool(
        triton is not None and all(t.is_cuda and t.is_contiguous() for t in tensors)
    )


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"SwiGLU inputs differ: {gate.shape} vs {up.shape}")
    output = torch.empty_like(gate)
    if _can_use_triton(gate, up):
        _forward_kernel[(triton.cdiv(gate.numel(), _BLOCK),)](
            gate, up, output, gate.numel(), BLOCK=_BLOCK
        )
        return output
    return torch.nn.functional.silu(gate.float()).to(gate.dtype) * up


def swiglu_backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    if _can_use_triton(grad_output, gate, up):
        grid = (triton.cdiv(gate.numel(), _BLOCK),)
        if variant == "triton_legacy":
            _legacy_backward_kernel[grid](
                grad_output,
                gate,
                up,
                grad_gate,
                grad_up,
                gate.numel(),
                REFERENCE_CASTS=True,
                BLOCK=_BLOCK,
            )
        else:
            _backward_kernel[grid](
                grad_output,
                gate,
                up,
                grad_gate,
                grad_up,
                gate.numel(),
                BLOCK=_BLOCK,
            )
        return grad_gate, grad_up

    gate_float = gate.float()
    sigmoid = torch.sigmoid(gate_float)
    grad_float = grad_output.float()
    grad_silu = (grad_output * up).to(gate.dtype).float()
    silu = (gate_float * sigmoid).to(gate.dtype).float()
    if variant == "triton_legacy":
        grad_gate_value = grad_silu * (sigmoid * (1.0 + gate_float * (1.0 - sigmoid)))
    else:
        grad_gate_value = (grad_silu * sigmoid) * (1.0 + gate_float * (1.0 - sigmoid))
    grad_gate.copy_(grad_gate_value.to(gate.dtype))
    grad_up.copy_((grad_float * silu).to(up.dtype))
    return grad_gate, grad_up


def swiglu_packed_forward(packed: torch.Tensor) -> torch.Tensor:
    rows, twice_width = packed.shape
    width = twice_width // 2
    if twice_width != 2 * width:
        raise ValueError("packed SwiGLU width must be even")
    output = torch.empty((rows, width), dtype=packed.dtype, device=packed.device)
    if triton is not None and packed.is_cuda and packed.is_contiguous():
        _packed_forward_kernel[(rows, triton.cdiv(width, _BLOCK))](
            packed, output, rows, width, BLOCK=_BLOCK
        )
        return output
    return swiglu_forward(
        packed[:, :width].contiguous(), packed[:, width:].contiguous()
    )


def swiglu_packed_backward(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    *,
    variant: str,
) -> torch.Tensor:
    grad_packed = torch.empty_like(packed)
    return swiglu_packed_backward_out(
        grad_output,
        packed,
        grad_packed,
        variant=variant,
    )


def swiglu_packed_backward_out(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    grad_packed: torch.Tensor,
    *,
    variant: str,
) -> torch.Tensor:
    """Write the packed VJP into caller-owned storage.

    The functional operation above deliberately remains the model-facing
    default.  This internal form lets an admitted deployment/planning wrapper
    provide storage whose lifetime and aliasing have already been proven.  In
    particular, the Triton kernel is safe when ``grad_packed`` aliases
    ``packed``: each program loads both of its source halves before writing
    either disjoint destination half.  Callers remain responsible for proving
    that no later consumer needs the original packed values.
    """
    rows, twice_width = packed.shape
    width = twice_width // 2
    if (
        grad_packed.shape != packed.shape
        or grad_packed.stride() != packed.stride()
        or grad_packed.dtype != packed.dtype
        or grad_packed.device != packed.device
    ):
        raise ValueError("packed SwiGLU destination must match the source layout")
    grad_output = grad_output.contiguous()
    if triton is not None and packed.is_cuda and packed.is_contiguous():
        grid = (rows, triton.cdiv(width, _BLOCK))
        if variant == "triton_legacy":
            _legacy_packed_backward_kernel[grid](
                grad_output,
                packed,
                grad_packed,
                rows,
                width,
                REFERENCE_CASTS=True,
                BLOCK=_BLOCK,
            )
        else:
            _packed_backward_kernel[grid](
                grad_output,
                packed,
                grad_packed,
                rows,
                width,
                BLOCK=_BLOCK,
            )
        return grad_packed
    # Compute the fallback values before writing either source half so the
    # same alias-safe contract holds without Triton.
    gate = packed[:, :width].contiguous()
    up = packed[:, width:].contiguous()
    gate_grad, up_grad = swiglu_backward(
        grad_output,
        gate,
        up,
        variant=variant,
    )
    grad_packed[:, :width].copy_(gate_grad)
    grad_packed[:, width:].copy_(up_grad)
    return grad_packed


__all__ = [
    "swiglu_backward",
    "swiglu_forward",
    "swiglu_packed_backward",
    "swiglu_packed_backward_out",
    "swiglu_packed_forward",
]
