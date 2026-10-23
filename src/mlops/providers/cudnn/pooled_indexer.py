"""Preferred SM90 cuDNN implementation of pooled causal index selection."""

from importlib.util import find_spec

import torch

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation


@torch.library.custom_op("mlops::pooled_topk_cudnn_sm90", mutates_args=())
def _select(
    q: torch.Tensor,
    key: torch.Tensor,
    gates: torch.Tensor,
    head_weights: torch.Tensor,
    position_bias: torch.Tensor,
    boundaries: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    from ...kernels.pooled_indexer_cudnn import pooled_topk

    with torch.cuda.device(q.device):
        return pooled_topk(
            q, key, gates, head_weights, position_bias, boundaries, top_k
        )


@_select.register_fake
def _fake(q, key, gates, head_weights, position_bias, boundaries, top_k):
    return torch.empty(
        (q.shape[0], top_k + position_bias.shape[0] - 1),
        device=q.device,
        dtype=torch.int32,
    )


def _supports(
    q,
    key,
    gates,
    head_weights,
    position_bias,
    boundaries,
    *,
    top_k=2048,
    surface,
    **_kwargs,
):
    if surface == "explicit":
        return SupportResult.no("pooled selection is non-differentiable")
    if not all(
        t.is_cuda and t.device == q.device
        for t in (q, key, gates, head_weights, position_bias)
    ):
        return SupportResult.no("pooled selection payloads require one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        return SupportResult.no("cuDNN pooled selection is validated on SM90 only")
    if q.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        return SupportResult.no("cuDNN SM90 scores require BF16 Q and K")
    if head_weights.dtype not in (torch.bfloat16, torch.float32):
        return SupportResult.no("cuDNN SM90 head weights must be BF16 or FP32")
    if q.shape[-1] != 128 or q.shape[1] not in (16, 32, 64):
        return SupportResult.no("cuDNN SM90 requires 16/32/64 heads of width 128")
    if boundaries.device.type != "cpu":
        return SupportResult.no("pooled selection boundaries must be CPU metadata")
    pool = position_bias.shape[0]
    if pool < 1 or top_k % pool or not 1 <= top_k // pool <= 2048:
        return SupportResult.no(
            "cuDNN radix supports 1 to 2048 complete selected pools"
        )
    if any(find_spec(name) is None for name in ("cudnn", "cutlass", "tvm_ffi")):
        return SupportResult.no("cuDNN pooled selection requires mlops[cudnn,quack]")
    return SupportResult.yes()


def apply(q, key, gates, head_weights, position_bias, boundaries, *, top_k=2048):
    return _select(q, key, gates, head_weights, position_bias, boundaries, top_k)


@flop_formula(_select)
def _flops(q, key, gates, head_weights, position_bias, boundaries, top_k, **_kwargs):
    return costs.pooled_topk(
        q, key, gates, head_weights, position_bias, boundaries, top_k=top_k
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "pooled_topk",
        "cudnn.pooled_topk.sm90",
        "cudnn",
        200,
        False,
        _supports,
        apply,
    )
)
