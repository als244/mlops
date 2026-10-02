"""Frozen shared expert using TE GEMM/SwiGLU with explicit capture boundaries.

The installed TE operation fuser calls pybind GEMMs that Dynamo cannot trace.
These operators expose its ordinary tensor inputs and saved preactivation;
all numerical kernels are Transformer Engine kernels.
"""

import torch
import transformer_engine_torch as tex
from torch import Tensor
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm


@torch.library.custom_op("mlops_ep_te_lora_shared::forward", mutates_args=())
def forward(x: Tensor, gate_up: Tensor, down: Tensor) -> tuple[Tensor, Tensor]:
    pre, *_ = general_gemm(gate_up, x, out_dtype=torch.bfloat16, layout="TN")
    hidden = tex.swiglu(pre, None)
    output, *_ = general_gemm(down, hidden, out_dtype=torch.bfloat16, layout="TN")
    return output, pre


@forward.register_fake
def _fake(x, gate_up, down):
    return torch.empty_like(x), x.new_empty((x.shape[0], gate_up.shape[0]))


@torch.library.custom_op("mlops_ep_te_lora_shared::backward", mutates_args=())
def backward(dy: Tensor, gate_up: Tensor, down: Tensor, pre: Tensor) -> Tensor:
    dh, *_ = general_gemm(down, dy, out_dtype=torch.bfloat16, layout="NN", grad=True)
    dpre = tex.dswiglu(dh, pre, None)
    dx, *_ = general_gemm(
        gate_up, dpre, out_dtype=torch.bfloat16, layout="NN", grad=True
    )
    return dx


@backward.register_fake
def _b_fake(dy, gate_up, down, pre):
    return torch.empty_like(dy)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[1], inputs[2], output[1])
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)


def _backward(ctx, dy, unused):
    if dy is None:
        return None, None, None
    return backward(dy.contiguous(), *ctx.saved_tensors), None, None


forward.register_autograd(_backward, setup_context=_setup)
