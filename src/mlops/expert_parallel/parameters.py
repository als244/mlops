"""BF16 compute storage with an explicit logical floating-point gradient dtype.

The wrapper owns no FP32 weight payload. Its ordinary BF16 component is exposed
to capture; registered MoE operations consume that component directly. Arithmetic
outside that boundary must explicitly request the compute data or dequantize.
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._pytree import tree_map


class BF16ComputeWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *, dtype=torch.float32, requires_grad=False):
        if data.dtype != torch.bfloat16:
            raise ValueError("BF16 compute weights require a BF16 payload")
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Logical gradient dtype must be FP32 or BF16")
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            tuple(data.shape),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            device=data.device,
            layout=data.layout,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        value._data = data
        return value

    def __tensor_flatten__(self):
        return ["_data"], self.dtype

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return BF16ComputeWeight(inner_tensors["_data"], dtype=metadata)

    def dequantize(self, *, dtype=None):
        """Explicit materialization for diagnostics, outside the layer's math."""
        return self._data.to(dtype=dtype or self.dtype)

    def __repr__(self):
        return f"BF16ComputeWeight(shape={tuple(self.shape)}, logical_dtype={self.dtype}, device={self.device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        aten = torch.ops.aten
        first = args[0]
        if func is aten._assert_tensor_metadata.default:
            # Check the outer logical metadata, not the BF16 component dtype.
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)
        if func is aten.detach.default:
            out = cls(first._data.detach(), dtype=first.dtype)
            return return_and_correct_aliasing(func, args, kwargs, out)
        if func is aten.clone.default:
            return cls(first._data.clone(**kwargs), dtype=first.dtype)
        if func is aten._to_copy.default:
            requested = kwargs.get("dtype", first.dtype)
            if requested == first.dtype:
                options = dict(kwargs, dtype=torch.bfloat16)
                return cls(
                    aten._to_copy.default(first._data, **options), dtype=first.dtype
                )
            # Requesting the BF16 compute format needs no conversion or copy
            # when device/layout already match. Capture sees the real component.
            return first._data.to(**kwargs)
        if func is aten.copy_.default:
            source = args[1]._data if isinstance(args[1], cls) else args[1]
            destination = first._data if isinstance(first, cls) else first
            destination.copy_(source, **kwargs)
            return first
        view_ops = (
            aten.view.default,
            aten._unsafe_view.default,
            aten.reshape.default,
            aten.slice.Tensor,
            aten.select.int,
            aten.unbind.int,
            aten.transpose.int,
            aten.t.default,
        )
        if func in view_ops:
            physical = tree_map(lambda x: x._data if isinstance(x, cls) else x, args)
            out = tree_map(
                lambda x: (
                    cls(x, dtype=first.dtype) if isinstance(x, torch.Tensor) else x
                ),
                func(*physical, **kwargs),
            )
            return return_and_correct_aliasing(func, args, kwargs, out)
        if func in (
            aten.empty_like.default,
            aten.zeros_like.default,
            aten.ones_like.default,
        ):
            options = dict(kwargs)
            options.setdefault("dtype", first.dtype)
            return func(first._data, **options)
        if func is aten.new_empty_strided.default:
            options = dict(kwargs)
            options.setdefault("dtype", first.dtype)
            return func(first._data, *args[1:], **options)
        raise NotImplementedError(
            f"{func} does not define a BF16 compute-weight operation; "
            "use the explicit component contract or dequantize for diagnostics"
        )
