"""Router projection and derivatives with bounded row-wise cast/GEMM scratch."""

import torch

from .precision.gemm import add_product
from .precision.parameters import ComputeWeight, payload
from .profiling import annotate


def row_slices(tokens, capacity):
    for start in range(0, tokens, capacity):
        yield slice(start, min(start + capacity, tokens))


@torch.library.custom_op("mlops_seqmoe::router_value", mutates_args=())
def router_value(x: torch.Tensor, weight: torch.Tensor, capacity: int) -> torch.Tensor:
    with annotate("fwd/router"):
        logits = x.new_empty((x.shape[0], weight.shape[0]), dtype=weight.dtype)
        for rows in row_slices(x.shape[0], capacity):
            with annotate("fwd/router/chunk"):
                with annotate("fwd/router/input_cast"):
                    xc = x[rows].to(weight.dtype)
                with annotate("fwd/router/gemm"):
                    torch.mm(xc, weight.T, out=logits[rows])
                del xc
        return logits


@router_value.register_fake
def _value_fake(x, weight, capacity):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=payload(weight).dtype)


@torch.library.custom_op("mlops_seqmoe::router", mutates_args=())
def router(x: torch.Tensor, weight: torch.Tensor, capacity: int) -> torch.Tensor:
    return router_value(x, payload(weight), capacity)


router.register_fake(_value_fake)


@torch.library.custom_op("mlops_seqmoe::router_backward", mutates_args=())
def router_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_dtype: str,
    train: bool,
    capacity: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with annotate("bwd/router"):
        dx = torch.empty_like(x)
        dw = (
            torch.zeros(weight.shape, device=x.device, dtype=getattr(torch, grad_dtype))
            if train
            else None
        )
        for rows in row_slices(x.shape[0], capacity):
            with annotate("bwd/router/upstream_cast"):
                dyc = dy[rows].to(weight.dtype)
            with annotate("bwd/router/input_gradient_gemm"):
                dxc = dyc @ weight
            # copy_ casts directly into the returned activation-gradient storage.
            # Release this chunk before creating scratch for another computation.
            with annotate("bwd/router/input_gradient_cast"):
                dx[rows].copy_(dxc)
            del dxc
            if train:
                # Frozen routers need only dy and weight to compute dx.
                with annotate("bwd/router/weight_gradient"):
                    xc = x[rows].to(weight.dtype)
                    add_product(dw, dyc.T, xc)
                    del xc
            del dyc
        return dx, [dw] if train else []


@router_backward.register_fake
def _backward_fake(dy, x, weight, grad_dtype, train, capacity):
    return torch.empty_like(x), (
        [x.new_empty(weight.shape, dtype=getattr(torch, grad_dtype))] if train else []
    )


@torch.library.custom_op("mlops_seqmoe::router_vjp", mutates_args=())
def router_vjp(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_dtype: str,
    train: bool,
    capacity: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return router_backward(dy, x, payload(weight), grad_dtype, train, capacity)


router_vjp.register_fake(_backward_fake)


def _setup(ctx, inputs, output):
    x, weight, capacity = inputs
    ctx.save_for_backward(x, weight)
    ctx.train = weight.requires_grad
    ctx.grad_dtype = str(weight.dtype).removeprefix("torch.")
    ctx.capacity = capacity


def _autograd(ctx, dy):
    dx, gradients = router_vjp(
        dy.contiguous(), *ctx.saved_tensors, ctx.grad_dtype, ctx.train, ctx.capacity
    )
    return dx, gradients[0] if gradients else None, None


def _dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise TypeError("Router operators require positional arguments")
    if func is router._opoverload:
        x, weight, capacity = args
        return router_value(x, payload(weight), capacity)
    dy, x, weight, grad_dtype, train, capacity = args
    return router_backward(dy, x, payload(weight), grad_dtype, train, capacity)


router.register_autograd(_autograd, setup_context=_setup)
for operation in (router, router_vjp):
    operation.register_torch_dispatch(ComputeWeight, _dispatch)
