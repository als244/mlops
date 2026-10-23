"""The router GEMM, at the precision the MoE trainers give it.

Every MoE path here -- the ``moe`` operators, the expert-LoRA compositions,
the expert-parallel layers and the references they are checked against --
forms its logits through these functions, so they all route the same way:

* **forward**: the operands at the activation's dtype (bf16 or fp16; the
  router weight, usually an fp32 master, taken to it), the products summed
  in fp32 and the logits written in fp32 -- cuBLAS's bf16 GEMM with an fp32
  output, the form of Megatron-Core's ``moe_router_dtype="fp32"`` and of
  torchtitan's router gate. Scores and top-k then run in fp32 (the routing
  kernels' business). The old form, both operands taken to the weight's
  fp32 and a SIMT GEMM, cost 1.1 ms per 32K-token step on an H100; this one
  about a tenth of that, and keeps no fp32 copy of the activation;
* **the activation's gradient**: the logits' cotangent taken to the
  activation's dtype, one GEMM against the weight at that dtype, summed in
  fp32 and added into the activation's gradient at its own dtype;
* **the weight's gradient**: the cotangent split into two bf16 pieces,
  ``hi = bf16(g)`` and ``mid = bf16(g - hi)``, each multiplied against the
  activation with fp32 sums -- an fp32-quality gradient for the fp32 master
  at the cost of one more small GEMM (torchtitan's ``hi_mid`` backward) --
  then written at the dtype asked for, the weight's own by default.

Off CUDA, or for an fp32 activation, every product is an fp32 product:
exact where the inputs are, and the same answer on every device.
"""

from __future__ import annotations

import torch

from ..expert_parallel.parameters import BF16ComputeWeight
from .matmul import add_product_, product_at

OPERAND_DTYPES = (torch.bfloat16, torch.float16)


def operand_dtype(hidden: torch.Tensor) -> torch.dtype:
    """The dtype the router GEMM's operands take for this activation."""
    if hidden.is_cuda and hidden.dtype in OPERAND_DTYPES:
        return hidden.dtype
    return torch.float32


def weight_operand(weight, dtype):
    # Called inside the router autograd boundary. Read physical storage while
    # the boundary retains the original parameter's logical gradient dtype.
    if isinstance(weight, BF16ComputeWeight):
        weight = weight._data
    return weight.to(dtype)


def router_logits(
    hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """fp32 logits ``[rows, E]`` of ``hidden [rows, D]`` through ``weight
    [E, D]`` (nn.Linear orientation), plus ``bias [E]`` when given."""
    dtype = operand_dtype(hidden)
    logits = product_at(
        hidden.to(dtype), weight_operand(weight, dtype).t(), torch.float32
    )
    if bias is not None:
        logits = logits + bias.to(torch.float32)
    return logits


def add_router_input_gradient_(
    grad_hidden: torch.Tensor, grad_logits: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``grad_hidden += grad_logits @ weight`` in place, at ``grad_hidden``'s
    dtype, with the operands at the activation's dtype."""
    dtype = operand_dtype(grad_hidden)
    add_product_(grad_hidden, grad_logits.to(dtype), weight_operand(weight, dtype))
    return grad_hidden


def router_input_gradient(
    grad_logits: torch.Tensor, weight: torch.Tensor, like: torch.Tensor
) -> torch.Tensor:
    """``grad_logits @ weight`` at ``like``'s dtype (the activation's): the
    operands at that dtype, the sums in fp32 inside the multiply, written at
    ``like``'s dtype directly when that is the operands' dtype."""
    dtype = operand_dtype(like)
    if like.dtype is dtype:
        return grad_logits.to(dtype) @ weight_operand(weight, dtype)
    product = product_at(
        grad_logits.to(dtype), weight_operand(weight, dtype), torch.float32
    )
    return product.to(like.dtype)


def router_weight_gradient(
    grad_logits: torch.Tensor,
    hidden: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``grad_logits.T @ hidden`` summed in fp32 and written at ``dtype``,
    the cotangent carried as two bf16 pieces when the activation is
    bf16/fp16 so the fp32 master's gradient keeps fp32 quality."""
    operands = operand_dtype(hidden)
    if operands is torch.float32:
        return product_at(
            grad_logits.to(torch.float32).t(), hidden.to(torch.float32), torch.float32
        ).to(dtype)
    rows = hidden.to(operands)
    high = grad_logits.to(operands)
    gradient = product_at(high.t(), rows, torch.float32)
    residue = (grad_logits.to(torch.float32) - high.to(torch.float32)).to(operands)
    add_product_(gradient, residue.t(), rows)
    return gradient.to(dtype)


class _RouterLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, weight_grad_dtype):
        ctx.save_for_backward(hidden, weight)
        ctx.has_bias = bias is not None
        ctx.weight_grad_dtype = (
            weight.dtype if weight_grad_dtype is None else weight_grad_dtype
        )
        return router_logits(hidden, weight, bias)

    @staticmethod
    def backward(ctx, grad_logits):
        hidden, weight = ctx.saved_tensors
        grad_logits = grad_logits.contiguous()
        grad_hidden = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_hidden = router_input_gradient(grad_logits, weight, hidden)
        if ctx.needs_input_grad[1]:
            grad_weight = router_weight_gradient(
                grad_logits, hidden, ctx.weight_grad_dtype
            )
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_logits.sum(0)
        return grad_hidden, grad_weight, grad_bias, None


def routed_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    weight_grad_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`router_logits` with its backward, for trace-visible compositions and
    layers: the weight's gradient at ``weight_grad_dtype`` (its own dtype
    by default), the activation's at the activation's dtype."""
    return _RouterLogits.apply(hidden, weight, bias, weight_grad_dtype)


__all__ = [
    "OPERAND_DTYPES",
    "add_router_input_gradient_",
    "operand_dtype",
    "routed_logits",
    "router_input_gradient",
    "router_logits",
    "router_weight_gradient",
]
