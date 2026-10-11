"""Fixed-capacity expert scratch and a common normal/LoRA computation."""

from types import SimpleNamespace

import torch

from .. import kernels
from ..precision.gemm import add_product


class Workspace:
    def __init__(self, x, hidden, rank, capacity, backward=False):
        shapes = {
            "x": (capacity, x.shape[1]),
            "gu": (capacity, 2 * hidden),
            "h": (capacity, hidden),
            "y": (capacity, x.shape[1]),
        }
        if rank:
            shapes.update(z13=(capacity, rank), z2=(capacity, rank))
        if backward:
            shapes.update(
                dy=(capacity, x.shape[1]),
                dh=(capacity, hidden),
                dgu=(capacity, 2 * hidden),
            )
            if rank:
                shapes["dz"] = (capacity, rank)
        self.tensors = {name: x.new_empty(shape) for name, shape in shapes.items()}

    def rows(self, count):
        return SimpleNamespace(
            **{name: value[:count] for name, value in self.tensors.items()}
        )


def forward_chunk(s, base, factors, scale, clip):
    base.forward(0, s.x, s.gu)
    if factors:
        a13, b13, a2, b2 = factors
        torch.mm(s.x, a13.T, out=s.z13)
        torch.addmm(s.gu, s.z13, b13.T, alpha=scale, out=s.gu)
    kernels.swiglu(s.gu, s.h, clip)
    base.forward(1, s.h, s.y)
    if factors:
        torch.mm(s.h, a2.T, out=s.z2)
        torch.addmm(s.y, s.z2, b2.T, alpha=scale, out=s.y)


def backward_chunk(s, base, factors, base_grads, factor_grads, scale, clip):
    if base_grads:
        base.weight_gradient(1, s.dy, s.h, base_grads[1])
    if factors:
        a13, b13, a2, b2 = factors
        da13, db13, da2, db2 = factor_grads
        add_product(db2, s.dy.T, s.z2, scale)
        torch.mm(s.dy, b2, out=s.dz)
        s.dz.mul_(scale)
        add_product(da2, s.dz.T, s.h)
    base.backward(1, s.dy, s.dh)
    if factors:
        torch.addmm(s.dh, s.dz, a2, out=s.dh)
    kernels.swiglu_backward(s.gu, s.dh, s.dgu, clip)
    if base_grads:
        base.weight_gradient(0, s.dgu, s.x, base_grads[0])
    if factors:
        add_product(db13, s.dgu.T, s.z13, scale)
        torch.mm(s.dgu, b13, out=s.dz)
        s.dz.mul_(scale)
        add_product(da13, s.dz.T, s.x)
    base.backward(0, s.dgu, s.y)
    if factors:
        torch.addmm(s.y, s.dz, a13, out=s.y)
