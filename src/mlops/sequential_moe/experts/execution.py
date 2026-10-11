"""Chunk loops shared by functional autograd and explicit destination operators.

These functions perform the expert math. Operator schemas and fake kernels live
in operators.py/reuse.py; no planner or allocator policy belongs here.
"""

import torch

from .. import kernels
from ..precision.gemm import BaseGEMMs
from ..profiling import annotate
from .math import Workspace, backward_chunk, forward_chunk


def chunks(bounds, capacity, multiple):
    if (
        bounds.device.type != "cpu"
        or bounds.dtype != torch.int64
        or bounds.shape != (2,)
    ):
        raise ValueError("Expert bounds must be a CPU int64 tensor with shape [2]")
    begin, end = bounds.tolist()
    if not 0 <= begin <= end:
        raise ValueError("Expert bounds must satisfy 0 <= begin <= end")
    for start in range(begin, end, capacity):
        valid = min(capacity, end - start)
        rows = (valid + multiple - 1) // multiple * multiple
        yield start, valid, rows


def forward_expert(
    x: torch.Tensor,
    acc: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
) -> None:
    with annotate("fwd/expert/dispatch_compute"):
        rank = factors[0].shape[0] if factors else 0
        with annotate("fwd/expert/scratch"):
            work = Workspace(x, hidden, rank, capacity)
        with annotate("fwd/expert/weight_decode"):
            base = BaseGEMMs(weights, precision, specs, x, hidden, capacity)
            factors = [v.to(x.dtype) for v in factors]
        for chunk, (start, valid, rows) in enumerate(
            chunks(bounds, capacity, multiple)
        ):
            with annotate(f"fwd/expert/chunk/{chunk}"):
                s = work.rows(rows)
                with annotate("fwd/expert/gather"):
                    kernels.gather(x, order, start, valid, s.x, probs.shape[1])
                with annotate("fwd/expert/compute"):
                    forward_chunk(s, base, factors, scale, clip)
                with annotate("fwd/expert/scatter"):
                    kernels.scatter(
                        s.y, order, probs, start, valid, acc, probs.shape[1], True
                    )
                # Views must not outlive the chunk. Otherwise nonempty experts
                # release workspace after decoded weights, while empty experts
                # release it before them, changing the allocation contract.
                del s


def backward_expert_into(
    dy: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
    grad_dtype: str,
    train_base: bool,
    dx: torch.Tensor,
    dp: torch.Tensor,
    accumulate_probability: bool,
) -> list[torch.Tensor]:
    with annotate("bwd/expert"):
        dtype = getattr(torch, grad_dtype)
        with annotate("bwd/expert/weight_decode_and_scratch"):
            base_grads = (
                [
                    torch.zeros(2 * hidden, x.shape[1], device=x.device, dtype=dtype),
                    torch.zeros(x.shape[1], hidden, device=x.device, dtype=dtype),
                ]
                if train_base
                else []
            )
            factor_grads = [torch.zeros_like(v, dtype=dtype) for v in factors]
            rank = factors[0].shape[0] if factors else 0
            work = Workspace(x, hidden, rank, capacity, backward=True)
            base = BaseGEMMs(
                weights,
                precision,
                specs,
                x,
                hidden,
                capacity,
                backward=True,
                train_base=train_base,
            )
            factors = [v.to(x.dtype) for v in factors]
        for chunk, (start, valid, rows) in enumerate(
            chunks(bounds, capacity, multiple)
        ):
            with annotate(f"bwd/expert/chunk/{chunk}"):
                s = work.rows(rows)
                with annotate("bwd/expert/gather"):
                    kernels.gather(x, order, start, valid, s.x, probs.shape[1])
                with annotate("bwd/expert/recompute"):
                    forward_chunk(s, base, factors, scale, clip)
                with annotate("bwd/expert/upstream_gather"):
                    kernels.gather(dy, order, start, valid, s.dy, probs.shape[1])
                with annotate("bwd/expert/probability_gradient_and_scale"):
                    kernels.weighted_dy(
                        s.dy,
                        s.y,
                        order,
                        probs,
                        dp,
                        start,
                        valid,
                        s.dy,
                        accumulate=accumulate_probability,
                    )
                with annotate("bwd/expert/compute"):
                    backward_chunk(
                        s, base, factors, base_grads, factor_grads, scale, clip
                    )
                with annotate("bwd/expert/input_gradient_scatter"):
                    kernels.scatter(
                        s.y, order, probs, start, valid, dx, probs.shape[1], False
                    )
                del s
        return base_grads + factor_grads
