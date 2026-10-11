"""Shared expert recomputation, with optional ordinary weight gradients."""

import torch

from .kernels import swiglu, swiglu_backward
from .precision.gemm import add_product
from .precision.parameters import ComputeWeight, payload
from .profiling import annotate


def buffers(x, w2, capacity, backward=False):
    c, d, h = capacity, x.shape[1], w2.shape[1]
    shapes = [(c, d), (c, 2 * h), (c, h), (c, d)]
    if backward:
        shapes += [(c, d), (c, h), (c, 2 * h)]
    return [x.new_empty(shape) for shape in shapes]


def chunks(x, capacity, multiple):
    for start in range(0, x.shape[0], capacity):
        valid = min(capacity, x.shape[0] - start)
        yield start, valid, (valid + multiple - 1) // multiple * multiple


@torch.library.custom_op("mlops_seqmoe::shared_forward", mutates_args=())
def shared_forward(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    capacity: int,
    multiple: int,
    clip: float,
    output_dtype: str,
) -> torch.Tensor:
    with annotate("fwd/shared"):
        y = torch.empty_like(x, dtype=getattr(torch, output_dtype))
        work = buffers(x, w2, capacity)
        for start, valid, rows in chunks(x, capacity, multiple):
            with annotate(f"fwd/shared/chunk/{start // capacity}"):
                xc, gu, h, yc = [v[:rows] for v in work]
                xc.zero_()
                xc[:valid].copy_(x[start : start + valid])
                torch.mm(xc, w13.T, out=gu)
                swiglu(gu, h, clip)
                torch.mm(h, w2.T, out=yc)
                y[start : start + valid].copy_(yc[:valid])
        return y


@shared_forward.register_fake
def _forward_fake(x, w13, w2, capacity, multiple, clip, output_dtype):
    return torch.empty_like(x, dtype=getattr(torch, output_dtype))


@torch.library.custom_op("mlops_seqmoe::shared_backward", mutates_args=())
def shared_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    capacity: int,
    multiple: int,
    clip: float,
    train: bool,
    grad_dtype: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with annotate("bwd/shared"):
        dx = torch.empty_like(x)
        grads = (
            [torch.zeros_like(w, dtype=getattr(torch, grad_dtype)) for w in (w13, w2)]
            if train
            else []
        )
        work = buffers(x, w2, capacity, backward=True)
        for start, valid, rows in chunks(x, capacity, multiple):
            with annotate(f"bwd/shared/chunk/{start // capacity}"):
                xc, gu, h, dxc, dyc, dh, dgu = [v[:rows] for v in work]
                xc.zero_()
                xc[:valid].copy_(x[start : start + valid])
                dyc.zero_()
                dyc[:valid].copy_(dy[start : start + valid])
                torch.mm(xc, w13.T, out=gu)
                if train:
                    swiglu(gu, h, clip)
                    add_product(grads[1], dyc.T, h)
                torch.mm(dyc, w2, out=dh)
                swiglu_backward(gu, dh, dgu, clip)
                if train:
                    add_product(grads[0], dgu.T, xc)
                torch.mm(dgu, w13, out=dxc)
                dx[start : start + valid].copy_(dxc[:valid])
        return dx, grads


@shared_backward.register_fake
def _backward_fake(dy, x, w13, w2, capacity, multiple, clip, train, grad_dtype):
    return torch.empty_like(x), (
        [torch.empty_like(w, dtype=getattr(torch, grad_dtype)) for w in (w13, w2)]
        if train
        else []
    )


@torch.library.custom_op("mlops_seqmoe::shared", mutates_args=())
def shared(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    capacity: int,
    multiple: int,
    clip: float,
    output_dtype: str,
) -> torch.Tensor:
    return shared_forward(
        x,
        payload(w13).to(x.dtype),
        payload(w2).to(x.dtype),
        capacity,
        multiple,
        clip,
        output_dtype,
    )


shared.register_fake(_forward_fake)


@torch.library.custom_op("mlops_seqmoe::shared_vjp", mutates_args=())
def shared_vjp(
    dy: torch.Tensor,
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    capacity: int,
    multiple: int,
    clip: float,
    train: bool,
    grad_dtype: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return shared_backward(
        dy,
        x,
        payload(w13).to(x.dtype),
        payload(w2).to(x.dtype),
        capacity,
        multiple,
        clip,
        train,
        grad_dtype,
    )


shared_vjp.register_fake(_backward_fake)


def _setup(ctx, inputs, output):
    x, w13, w2, capacity, multiple, clip, _ = inputs
    ctx.save_for_backward(x, w13, w2)
    ctx.options = (
        capacity,
        multiple,
        clip,
        w13.requires_grad,
        str(w13.dtype).removeprefix("torch."),
    )


def _autograd(ctx, dy):
    dx, gradients = shared_vjp(dy.contiguous(), *ctx.saved_tensors, *ctx.options)
    return dx, *(gradients if gradients else [None, None]), None, None, None, None


def _dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise TypeError("Shared expert operators require positional arguments")
    if func is shared._opoverload:
        x, w13, w2, *options = args
        return shared_forward(
            x, payload(w13).to(x.dtype), payload(w2).to(x.dtype), *options
        )
    dy, x, w13, w2, *options = args
    return shared_backward(
        dy, x, payload(w13).to(x.dtype), payload(w2).to(x.dtype), *options
    )


shared.register_autograd(_autograd, setup_context=_setup)
for operation in (shared, shared_vjp):
    operation.register_torch_dispatch(ComputeWeight, _dispatch)
