"""Compute storage with separately specified gradient dtype.

This prototype representation is deliberately isolated. It borrows the component
contract used by the existing EP layers, without importing those backends.
Optimizers consuming this wrapper must support its explicit compute component;
ordinary Parameters are used when storage and gradient dtypes match.
"""

import torch
from torch import nn
from torch.utils._python_dispatch import return_and_correct_aliasing


class ComputeWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, data, gradient_dtype, requires_grad=False):
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            device=data.device,
            layout=data.layout,
            dtype=gradient_dtype,
            requires_grad=requires_grad,
        )
        value.compute = data
        return value

    def __repr__(self):
        return (
            f"ComputeWeight(shape={tuple(self.shape)}, device={self.device}, "
            f"storage_dtype={self.compute.dtype}, gradient_dtype={self.dtype})"
        )

    def __tensor_flatten__(self):
        return ["compute"], self.dtype

    @staticmethod
    def __tensor_unflatten__(inner, metadata, outer_size, outer_stride):
        return ComputeWeight(inner["compute"], metadata)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        first = args[0]
        aten = torch.ops.aten
        if func is aten.detach.default:
            out = cls(first.compute.detach(), first.dtype)
            return return_and_correct_aliasing(func, args, kwargs, out)
        if func is aten.clone.default:
            return cls(first.compute.clone(**kwargs), first.dtype)
        if func is aten._to_copy.default:
            requested = kwargs.get("dtype", first.dtype)
            if requested == first.dtype:
                return cls(
                    func(first.compute, **dict(kwargs, dtype=first.compute.dtype)),
                    first.dtype,
                )
            return func(first.compute, **kwargs)
        if func is aten.copy_.default:
            dst = payload(first)
            dst.copy_(payload(args[1]), **kwargs)
            return first
        if func in (
            aten.zeros_like.default,
            aten.empty_like.default,
            aten.ones_like.default,
        ):
            return func(first.compute, **dict({"dtype": first.dtype}, **kwargs))
        if func is aten.new_empty_strided.default:
            return func(
                first.compute, *args[1:], **dict({"dtype": first.dtype}, **kwargs)
            )
        raise NotImplementedError(
            f"{func}: use the explicit compute component of ComputeWeight"
        )


def payload(value):
    return value.compute if isinstance(value, ComputeWeight) else value


def parameter(value, gradient_dtype):
    return nn.Parameter(
        value if value.dtype == gradient_dtype else ComputeWeight(value, gradient_dtype)
    )
