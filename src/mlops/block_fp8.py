"""Expert weights stored once as block-scaled E4M3.

`BlockFP8Weight` holds an expert weight ``[E, out, in]`` as E4M3 values and one
FP32 scale per 128x128 block (``value ~ data * scale``), the layout of
DeepSeek-style FP8 checkpoints, whose weights it can take as they are. A
block covers both axes, so the same bytes serve the forward GEMM (reducing
over ``in``) and the input gradient (reducing over ``out``): unlike
`QuackFP8Weight`, no transposed copy is kept. Rows stay in ``nn.Linear``
order, a gate/up weight's gate rows then its up rows. With such weights
``mlops.moe`` and ``expert_lora`` compute the experts' GEMMs in E4M3
(``compute_precision="fp8_block"``, the default for them; the grouped GEMM is
``kernels/block_fp8.py``); ``"bf16"`` dequantizes them per call.

The logical dtype describes returned gradients. `quantize` and `dequantize`
differentiate straight through, as `QuackFP8Weight`'s do.
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from .kernels.block_fp8 import dequantize_blocks, quantize_blocks

COMPONENTS = ("_data", "_scale")


class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, dtype):
        return BlockFP8Weight(
            *quantize_blocks(weight.detach()),
            dtype=weight.dtype if dtype is None else dtype,
        )

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _Dequantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, dtype):
        return weight.detach()._dequantize(dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class BlockFP8Weight(torch.Tensor):
    def __repr__(self):
        return (
            f"BlockFP8Weight(shape={tuple(self.shape)}, logical_dtype={self.dtype}, "
            f"device={self.device})"
        )

    @staticmethod
    def __new__(cls, data, scale, *, dtype=torch.bfloat16):
        if data.ndim != 3 or data.dtype != torch.float8_e4m3fn:
            raise ValueError("Expected E4M3 data [experts, out, in]")
        experts, out, inner = data.shape
        expected = (experts, -(-out // 128), -(-inner // 128))
        if tuple(scale.shape) != expected or scale.dtype != torch.float32:
            raise ValueError(f"Expected FP32 block scales {expected}")
        value = torch.Tensor._make_wrapper_subclass(
            cls, tuple(data.shape), device=data.device, dtype=dtype
        )
        value._data, value._scale = data, scale
        return value

    @classmethod
    def quantize(cls, weight, *, dtype=None):
        """A container of ``weight [E, out, in]`` (bf16 or fp32), one scale per
        128x128 block; ``dtype`` is the logical dtype, the weight's own when
        ``None``. Differentiable: the container's gradient is the weight's."""
        if weight.ndim != 3:
            raise ValueError("Expected an [experts, out, in] weight")
        return _Quantize.apply(weight, dtype)

    def components(self):
        return self._data, self._scale

    def __tensor_flatten__(self):
        return list(COMPONENTS), (self.dtype,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        (dtype,) = metadata
        return BlockFP8Weight(*(inner_tensors[n] for n in COMPONENTS), dtype=dtype)

    def _dequantize(self, dtype):
        return dequantize_blocks(
            self._data, self._scale, dtype=self.dtype if dtype is None else dtype
        )

    def dequantize(self, *, dtype=None):
        """The weight the blocks represent, at the logical dtype or ``dtype``.
        Differentiable: its gradient is the container's."""
        if torch.is_grad_enabled() and self.requires_grad:
            return _Dequantize.apply(self, dtype)
        return self._dequantize(dtype)

    def _like(self, components):
        return type(self)(*components, dtype=self.dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten, first = torch.ops.aten, args[0]
        if func is aten._assert_tensor_metadata.default:
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)
        if func is aten.detach.default:
            out = first._like(v.detach() for v in first.components())
            return return_and_correct_aliasing(func, args, kwargs, out)
        if func is aten.clone.default:
            return first._like(v.clone(**kwargs) for v in first.components())
        if func is aten.copy_.default:
            destination, source = args[:2]
            if isinstance(destination, cls):
                values = (
                    source.components()
                    if isinstance(source, cls)
                    # Publication from a master: quantized as at init.
                    else quantize_blocks(source)
                )
                for target, value in zip(destination.components(), values):
                    target.copy_(value)
                return destination
            return destination.copy_(source._dequantize(destination.dtype), **kwargs)
        if func is aten._to_copy.default:
            if kwargs.get("dtype", first.dtype) != first.dtype:
                return first._dequantize(None).to(**kwargs)
            options = {k: v for k, v in kwargs.items() if k != "dtype"}
            return first._like(
                aten._to_copy.default(v, **options) for v in first.components()
            )
        if func in (
            aten.empty_like.default,
            aten.zeros_like.default,
            aten.ones_like.default,
        ):
            return func(
                first._data, **dict(kwargs, dtype=kwargs.get("dtype", first.dtype))
            )
        if func is aten.new_empty_strided.default:
            return func(
                first._data,
                *args[1:],
                **dict(kwargs, dtype=kwargs.get("dtype", first.dtype)),
            )
        raise NotImplementedError(
            f"{func}: use the block FP8 components or dequantize for diagnostics"
        )


__all__ = ["BlockFP8Weight"]
