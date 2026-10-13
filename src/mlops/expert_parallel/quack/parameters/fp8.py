"""Explicit E4M3 payloads and per-channel descales for QuACK SM90 GEMMs.

A weight ``[E, out, in]`` held as the two operands its GEMMs read: E4M3 rows
with a descale per row (``W``, a forward's B operand) and the E4M3 transpose
with a descale per row of the transpose (``Wᵀ``, the backward's), quantized
independently from one source. The logical parameter dtype describes returned
gradients. No master weight payload is retained. Both quantized orientations
and both scale vectors are ordinary tensor inputs after AOT flattening.

A gate/up weight may keep its rows interleaved -- ``gate_j`` and ``up_j``
adjacent, the order QuACK's paired SwiGLU epilogues read -- while its logical
shape, `dequantize` and its gradients stay in ``nn.Linear`` order (gate rows,
then up rows); ``interleaved`` records that. `quantize` and `dequantize`
differentiate straight through, so a bf16 master quantized per call and a
container dequantized per call both receive the gradient computed for them.
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

COMPONENTS = ("_data", "_transposed", "_scale", "_transposed_scale")


def _physical_rows(value):
    """``[E, out, in]`` from gate-then-up rows to gate/up-adjacent rows."""
    experts, out, inner = value.shape
    return value.view(experts, 2, out // 2, inner).transpose(1, 2).reshape(value.shape)


def _logical_rows(value):
    """``[E, out, in]`` from gate/up-adjacent rows back to gate-then-up rows."""
    experts, out, inner = value.shape
    return value.view(experts, out // 2, 2, inner).transpose(1, 2).reshape(value.shape)


def _quantize_components(source, interleaved):
    """``(data, transposed, scale, transposed_scale)`` of a BF16/FP32 weight:
    the kernels on CUDA, the same equations in tensor ops elsewhere."""
    if source.is_cuda:
        from ..kernels.quantize_transpose import quantize_weights_fp8

        return quantize_weights_fp8(source, interleave_rows=interleaved)
    value = source.float()
    if interleaved:
        value = _physical_rows(value)

    def quantize(value):
        scale = (value.abs().amax(dim=-1) / 448.0).clamp_min(1e-12)
        data = (value / scale[..., None]).clamp(-448.0, 448.0)
        return data.to(torch.float8_e4m3fn), scale

    rows, scales = quantize(value)
    columns, column_scales = quantize(value.transpose(-1, -2))
    return rows, columns.contiguous(), scales, column_scales


class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, dtype, interleaved):
        return QuackFP8Weight(
            *_quantize_components(weight.detach(), interleaved),
            dtype=weight.dtype if dtype is None else dtype,
            interleaved=interleaved,
        )

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class _Dequantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, dtype):
        return weight.detach()._dequantize(dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class QuackFP8Weight(torch.Tensor):
    def __repr__(self):
        layout = ", interleaved" if self._interleaved else ""
        return (
            f"QuackFP8Weight(shape={tuple(self.shape)}, logical_dtype={self.dtype}, "
            f"device={self.device}{layout})"
        )

    @staticmethod
    def __new__(
        cls,
        data,
        transposed,
        scale,
        transposed_scale,
        *,
        dtype=torch.float32,
        interleaved=False,
    ):
        value = torch.Tensor._make_wrapper_subclass(
            cls, tuple(data.shape), device=data.device, dtype=dtype
        )
        for name, tensor in zip(
            COMPONENTS, (data, transposed, scale, transposed_scale)
        ):
            setattr(value, name, tensor)
        value._interleaved = bool(interleaved)
        return value

    @classmethod
    def quantize(cls, weight, *, dtype=None, interleaved=False):
        """A container of ``weight [E, out, in]`` (BF16 or FP32): both
        orientations quantized, the rows interleaved when asked; ``dtype`` is
        the logical dtype, the weight's own when ``None``. Differentiable:
        the container's gradient is the weight's."""
        if weight.ndim != 3:
            raise ValueError("Expected an [experts, out, in] weight")
        if interleaved and weight.shape[1] % 2:
            raise ValueError("Interleaving gate and up rows needs an even row count")
        return _Quantize.apply(weight, dtype, interleaved)

    @property
    def interleaved(self):
        return self._interleaved

    def components(self):
        return tuple(getattr(self, name) for name in COMPONENTS)

    def __tensor_flatten__(self):
        return list(COMPONENTS), (self.dtype, self._interleaved)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        dtype, interleaved = metadata
        return QuackFP8Weight(
            *(inner_tensors[n] for n in COMPONENTS),
            dtype=dtype,
            interleaved=interleaved,
        )

    def _dequantize(self, dtype):
        value = self._data.float() * self._scale[..., None]
        if self._interleaved:
            value = _logical_rows(value)
        return value.to(self.dtype if dtype is None else dtype)

    def dequantize(self, *, dtype=None):
        """The weight the rows represent, in ``nn.Linear`` row order, at the
        logical dtype or ``dtype``. Differentiable: its gradient is the
        container's."""
        if torch.is_grad_enabled() and self.requires_grad:
            return _Dequantize.apply(self, dtype)
        return self._dequantize(dtype)

    def _like(self, components):
        return type(self)(*components, dtype=self.dtype, interleaved=self._interleaved)

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
                if isinstance(source, cls):
                    if source._interleaved != destination._interleaved:
                        raise ValueError("FP8 weights of different row orders")
                    for target, value in zip(
                        destination.components(), source.components()
                    ):
                        target.copy_(value)
                else:
                    # Optimizer publication owns conversion into compute
                    # weights: both orientations quantized independently, as
                    # at init, in the destination's row order.
                    for target, value in zip(
                        destination.components(),
                        _quantize_components(source, destination._interleaved),
                    ):
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
            f"{func}: use explicit QuACK FP8 components or dequantize for diagnostics"
        )
