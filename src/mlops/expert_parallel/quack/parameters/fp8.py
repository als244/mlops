"""Explicit E4M3 payloads and per-channel descales for QuACK SM90 GEMMs.

The logical parameter dtype describes returned gradients. No master weight
payload is retained. Both quantized orientations and both scale vectors are
ordinary tensor inputs after AOT flattening.
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

COMPONENTS = ("_data", "_transposed", "_scale", "_transposed_scale")


class QuackFP8Weight(torch.Tensor):
    def __repr__(self):
        return f"QuackFP8Weight(shape={tuple(self.shape)}, logical_dtype={self.dtype}, device={self.device})"

    @staticmethod
    def __new__(cls, data, transposed, scale, transposed_scale, *, dtype=torch.float32):
        value = torch.Tensor._make_wrapper_subclass(
            cls, tuple(data.shape), device=data.device, dtype=dtype
        )
        for name, tensor in zip(
            COMPONENTS, (data, transposed, scale, transposed_scale)
        ):
            setattr(value, name, tensor)
        return value

    def components(self):
        return tuple(getattr(self, name) for name in COMPONENTS)

    def __tensor_flatten__(self):
        return list(COMPONENTS), self.dtype

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return QuackFP8Weight(*(inner_tensors[n] for n in COMPONENTS), dtype=metadata)

    def dequantize(self, *, dtype=None):
        return (self._data.float() * self._scale[..., None]).to(dtype or self.dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten, first = torch.ops.aten, args[0]
        if func is aten._assert_tensor_metadata.default:
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)
        if func is aten.detach.default:
            out = cls(*(v.detach() for v in first.components()), dtype=first.dtype)
            return return_and_correct_aliasing(func, args, kwargs, out)
        if func is aten.clone.default:
            return cls(
                *(v.clone(**kwargs) for v in first.components()), dtype=first.dtype
            )
        if func is aten.copy_.default:
            destination, source = args[:2]
            if isinstance(destination, cls):
                if isinstance(source, cls):
                    for target, value in zip(
                        destination.components(), source.components()
                    ):
                        target.copy_(value)
                else:
                    # Optimizer publication owns conversion into compute weights.
                    # Keep both orientations independently quantized, as at init.
                    def quantize(value):
                        value = value.float()
                        scale = (value.abs().amax(dim=-1) / 448.0).clamp_min(1e-12)
                        data = (value / scale[..., None]).clamp(-448.0, 448.0)
                        return data.to(torch.float8_e4m3fn), scale

                    rows, scales = quantize(source)
                    columns, column_scales = quantize(source.transpose(-1, -2))
                    for target, value in zip(
                        destination.components(), (rows, columns, scales, column_scales)
                    ):
                        target.copy_(value)
                return destination
            return destination.copy_(
                source.dequantize(dtype=destination.dtype), **kwargs
            )
        if func is aten._to_copy.default:
            if kwargs.get("dtype", first.dtype) != first.dtype:
                return first.dequantize().to(**kwargs)
            options = {k: v for k, v in kwargs.items() if k != "dtype"}
            return cls(
                *(aten._to_copy.default(v, **options) for v in first.components()),
                dtype=first.dtype,
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
