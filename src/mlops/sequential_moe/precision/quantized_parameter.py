"""Physical compressed parameters with high-precision logical gradients."""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from .storage import WeightSpec, materialize, pack


class QuantizedWeight(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        components,
        precision,
        gradient_dtype=torch.float32,
        requires_grad=False,
        *,
        spec=None,
    ):
        data = components[0]
        if spec is None:
            spec = WeightSpec(
                precision,
                data.shape[0],
                data.shape[1] * (2 if precision == "nvfp4" else 1),
                128 if precision == "fp8" else 16,
                128 if precision == "fp8" else 16,
            )
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            (spec.rows, spec.cols),
            strides=(spec.cols, 1),
            storage_offset=0,
            device=data.device,
            layout=data.layout,
            dtype=gradient_dtype,
            requires_grad=requires_grad,
        )
        value.compute, value.scale = components[:2]
        value.global_scale = components[2] if len(components) > 2 else None
        value.precision, value.spec = spec.format, spec
        return value

    def components(self):
        return [self.compute, self.scale] + (
            [self.global_scale] if self.global_scale is not None else []
        )

    def materialize(self, dtype=torch.float32):
        return materialize(self.components(), self.spec, dtype)

    def __tensor_flatten__(self):
        names = ["compute", "scale"] + (
            ["global_scale"] if self.global_scale is not None else []
        )
        return names, (self.spec.encode(), self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner, metadata, outer_size, outer_stride):
        text, dtype = metadata
        spec = WeightSpec.parse(text)
        names = ["compute", "scale"] + (
            ["global_scale"] if "global_scale" in inner else []
        )
        return QuantizedWeight([inner[n] for n in names], spec.format, dtype, spec=spec)

    def __repr__(self):
        return f"QuantizedWeight(shape={tuple(self.shape)}, storage={self.spec.encode()}, gradient_dtype={self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        first = args[0]
        aten = torch.ops.aten

        def wrap(values):
            return cls(values, first.precision, first.dtype, spec=first.spec)

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, wrap([v.detach() for v in first.components()])
            )
        if func is aten.clone.default:
            return wrap([v.clone() for v in first.components()])
        if func is aten.copy_.default:
            source = args[1]
            if isinstance(source, cls):
                if source.spec != first.spec:
                    raise ValueError(
                        "Quantized copy requires matching storage metadata"
                    )
                values = source.components()
            else:
                if tuple(source.shape) != tuple(first.shape):
                    raise ValueError("Quantized update requires matching matrix shape")
                block = (
                    128
                    if first.precision == "fp8"
                    else 16
                    if first.precision == "nvfp4"
                    else 32
                )
                block_rows = (
                    first.spec.block_rows
                    if first.precision == "nvfp4"
                    else block
                    if first.precision == "fp8"
                    else 1
                )
                expected = WeightSpec(first.precision, *first.shape, block_rows, block)
                if first.spec != expected:
                    raise ValueError(
                        "This checkpoint layout needs a format-aware optimizer update; "
                        "copy_ supports the constructor's quantization recipe"
                    )
                values, spec = pack(
                    source.to(device=first.device).contiguous(),
                    first.precision,
                    nvfp4_block_rows=first.spec.block_rows,
                )
                if spec != first.spec:
                    raise ValueError(
                        "This checkpoint layout needs a format-aware optimizer update; "
                        "copy_ supports the constructor's quantization recipe"
                    )
            for dest, src in zip(first.components(), values, strict=True):
                dest.copy_(src, **kwargs)
            return first
        if func in (
            aten.empty_like.default,
            aten.zeros_like.default,
            aten.ones_like.default,
        ):
            factory = {
                aten.empty_like.default: torch.empty,
                aten.zeros_like.default: torch.zeros,
                aten.ones_like.default: torch.ones,
            }[func]
            return factory(
                first.shape,
                **dict({"device": first.device, "dtype": first.dtype}, **kwargs),
            )
        if func is aten.new_empty_strided.default:
            return torch.empty_strided(
                *args[1:],
                **dict({"device": first.device, "dtype": first.dtype}, **kwargs),
            )
        if func is aten._to_copy.default:
            options = dict(kwargs)
            dtype = options.pop("dtype", first.dtype)
            if dtype == first.dtype:
                return wrap([func(v, **options) for v in first.components()])
            return func(first.materialize(dtype), **options)
        raise NotImplementedError(
            f"{func}: use QuantizedWeight components or materialize()"
        )
